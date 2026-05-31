from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .graph import GraphExpansion, LocalKnowledgeGraph
from .index import BM25Index, SearchHit, load_documents_csv
from .llm import LLMConfig, llm_answer
from .text import normalize_text, best_snippet
from .vector import VectorIndex, rrf_fusion, score_fusion


@dataclass
class RagResult:
    query: str
    expanded_query: str
    matched_terms: List[str]
    expansions: List[str]
    hits: List[SearchHit]
    search_mode: str = "bm25+kg"

    def to_dict(self) -> Dict[str, object]:
        return {
            "query": self.query,
            "expanded_query": self.expanded_query,
            "matched_terms": self.matched_terms,
            "expansions": self.expansions,
            "hits": [asdict(hit) for hit in self.hits],
            "search_mode": self.search_mode,
        }


class RagSearchEngine:
    def __init__(self, index: BM25Index, graph: Optional[LocalKnowledgeGraph] = None):
        self.index = index
        self.graph = graph or LocalKnowledgeGraph()

    @classmethod
    def load(cls, index_path: str = "data/index.json", graph_path: str = "data/sample_kg.csv") -> "RagSearchEngine":
        return cls(BM25Index.load(index_path), LocalKnowledgeGraph.load(graph_path))

    @classmethod
    def build(
        cls,
        data_path: str,
        *,
        graph_path: str = "data/sample_kg.csv",
        index_path: str = "data/index.json",
        limit: Optional[int] = None,
    ) -> "RagSearchEngine":
        graph = LocalKnowledgeGraph.load(graph_path)
        documents = load_documents_csv(data_path, limit=limit)
        index = BM25Index(documents, extra_terms=graph.terms())
        index.save(index_path)
        return cls(index, graph)

    def search(self, query: object, *, top_k: int = 8, use_graph: bool = True) -> RagResult:
        query_text = normalize_text(query)
        if use_graph:
            expansion = self.graph.expand(query_text)
        else:
            expansion = GraphExpansion(query_text, query_text, [], [])
        hits = self.index.search(expansion.expanded_query, top_k=top_k, terms=expansion.matched_terms)
        return RagResult(
            query=query_text,
            expanded_query=expansion.expanded_query,
            matched_terms=expansion.matched_terms,
            expansions=expansion.expansions,
            hits=hits,
        )

    def answer(
        self,
        query: object,
        *,
        top_k: int = 5,
        use_graph: bool = True,
        use_llm: bool = False,
        llm_config: Optional[LLMConfig] = None,
    ) -> Dict[str, object]:
        result = self.search(query, top_k=top_k, use_graph=use_graph)
        if not result.hits:
            return {
                "answer": "没有在当前字幕索引中找到足够相关的证据。可以换一个关键词，或补充新的字幕数据后重建索引。",
                "result": result.to_dict(),
                "mode": "empty",
            }

        if use_llm:
            try:
                llm_text = llm_answer(
                    result.query,
                    [asdict(h) for h in result.hits],
                    result.matched_terms,
                    config=llm_config,
                )
                return {"answer": llm_text, "result": result.to_dict(), "mode": "llm"}
            except Exception as exc:
                return {
                    "answer": f"LLM 生成失败（{exc}），回退到抽取式回答。\n\n" + self._extractive_answer(result),
                    "result": result.to_dict(),
                    "mode": "llm_fallback",
                }

        return {"answer": self._extractive_answer(result), "result": result.to_dict(), "mode": "extractive"}

    def _extractive_answer(self, result: RagResult) -> str:
        evidence_lines = []
        for number, hit in enumerate(result.hits[:3], start=1):
            source = hit.title or f"doc-{hit.doc_id}"
            time = f" {hit.time}" if hit.time else ""
            evidence_lines.append(f"{number}. {source}{time}: {hit.snippet}")

        expansion_text = ""
        if result.matched_terms:
            expansion_text = "图谱补强命中：" + "、".join(result.matched_terms) + "。\n"

        return (
            f"{expansion_text}"
            f"根据当前检索到的字幕证据，问题\u201c{result.query}\u201d可以优先从这些片段理解：\n"
            + "\n".join(evidence_lines)
            + "\n\n这版回答是抽取式的，适合面试演示检索链路；后续可以把这些证据交给 LLM 生成更自然的最终答案。"
        )


class HybridRagSearchEngine(RagSearchEngine):
    def __init__(
        self,
        index: BM25Index,
        graph: Optional[LocalKnowledgeGraph] = None,
        vector_index: Optional[VectorIndex] = None,
    ):
        super().__init__(index, graph)
        self.vector_index = vector_index
        self._use_hybrid = vector_index is not None

    @classmethod
    def load(cls, index_path: str = "data/index.json", graph_path: str = "data/sample_kg.csv") -> "HybridRagSearchEngine":
        return cls(BM25Index.load(index_path), LocalKnowledgeGraph.load(graph_path))

    def configure_vector(
        self,
        api_key: str,
        *,
        model_id: str = "intfloat/multilingual-e5-large",
        proxy: Optional[str] = "http://127.0.0.1:7890",
        cache_path: Optional[str] = None,
        max_workers: int = 1,
    ):
        if self.vector_index is None:
            self.vector_index = VectorIndex(self.index.documents)
        self.vector_index.configure(api_key, model_id=model_id, proxy=proxy)
        self._use_hybrid = True
        if cache_path:
            cached = VectorIndex.load(cache_path, self.index.documents)
            if cached.embeddings:
                self.vector_index.embeddings = cached.embeddings
            elif max_workers > 1:
                self.vector_index.build_incremental(cache_path, max_workers=max_workers)
            else:
                self.vector_index.build()
                self.vector_index.save(cache_path)

    def search(
        self,
        query: object,
        *,
        top_k: int = 8,
        use_graph: bool = True,
        use_hybrid: Optional[bool] = None,
        fusion_mode: str = "score",
        rrf_k: int = 60,
        bm25_weight: float = 0.3,
        vector_weight: float = 0.7,
    ) -> RagResult:
        query_text = normalize_text(query)
        if use_graph:
            expansion = self.graph.expand(query_text)
        else:
            expansion = GraphExpansion(query_text, query_text, [], [])

        bm25_hits_raw = self.index.search_raw(
            expansion.expanded_query, top_k=top_k * 3, terms=expansion.matched_terms
        )

        do_hybrid = use_hybrid if use_hybrid is not None else self._use_hybrid
        if do_hybrid and self.vector_index and self.vector_index.embeddings:
            vector_query = expansion.expanded_query if expansion.expansions else query_text
            vector_hits_raw = self.vector_index.search_raw(vector_query, top_k=top_k * 3)

            if fusion_mode == "rrf":
                fused = rrf_fusion(
                    bm25_hits_raw,
                    vector_hits_raw,
                    k=rrf_k,
                    bm25_weight=bm25_weight / max(vector_weight, 0.01),
                    vector_weight=1.0,
                )
            else:
                fused = score_fusion(
                    bm25_hits_raw,
                    vector_hits_raw,
                    bm25_weight=bm25_weight,
                    vector_weight=vector_weight,
                )

            hits = []
            for doc_id, score in fused[:top_k]:
                if 0 <= doc_id < len(self.index.documents):
                    doc = self.index.documents[doc_id]
                    snippet = best_snippet(doc.text, query_text)
                    hits.append(SearchHit(
                        doc_id=doc_id,
                        score=score,
                        title=doc.title,
                        time=doc.time,
                        text=doc.text,
                        snippet=snippet,
                    ))
            search_mode = "hybrid"
        else:
            hits = self.index.search(expansion.expanded_query, top_k=top_k, terms=expansion.matched_terms)
            search_mode = "bm25+kg"

        return RagResult(
            query=query_text,
            expanded_query=expansion.expanded_query,
            matched_terms=expansion.matched_terms,
            expansions=expansion.expansions,
            hits=hits,
            search_mode=search_mode,
        )

    def answer(
        self,
        query: object,
        *,
        top_k: int = 5,
        use_graph: bool = True,
        use_llm: bool = False,
        llm_config: Optional[LLMConfig] = None,
        use_hybrid: Optional[bool] = None,
    ) -> Dict[str, object]:
        result = self.search(query, top_k=top_k, use_graph=use_graph, use_hybrid=use_hybrid)
        if use_llm and llm_config and llm_config.api_key:
            try:
                llm_result = llm_answer(
                    result.query,
                    [hit.snippet for hit in result.hits[:3]],
                    llm_config,
                )
                return {
                    "answer": llm_result.answer,
                    "result": result.to_dict(),
                    "mode": "llm",
                }
            except Exception:
                return {
                    "answer": self._extractive_answer(result),
                    "result": result.to_dict(),
                    "mode": "llm_fallback",
                }
        return {"answer": self._extractive_answer(result), "result": result.to_dict(), "mode": "extractive"}


def default_paths(root: Optional[str] = None) -> Dict[str, str]:
    base = Path(root or ".")
    return {
        "data": str(base / "data" / "local_subtitles.csv"),
        "graph": str(base / "data" / "sample_kg.csv"),
        "index": str(base / "data" / "index.json"),
    }
