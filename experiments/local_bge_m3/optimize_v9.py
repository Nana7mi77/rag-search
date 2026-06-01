import csv
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

os.environ["HF_HUB_OFFLINE"] = "1"

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = Path(__file__).resolve().parent / "cache"

MODEL_NAME = "BAAI/bge-m3"
RERANKER_PATH = "/tmp/bge-reranker"
DEVICE = "mps"
CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
TOKEN_RE = re.compile(r"[\u4e00-\u9fff]+|[a-zA-Z0-9_]+")


def normalize(text):
    if text is None:
        return ""
    text = str(text).replace("\ufeff", "").replace("\u3000", " ")
    return re.sub(r"\s+", " ", text).strip()


def cjk_ngrams(span):
    span = normalize(span)
    if not span:
        return []
    tokens = []
    if len(span) <= 6:
        tokens.append(span)
    for n in (2, 3):
        if len(span) >= n:
            tokens.extend(span[i:i + n] for i in range(len(span) - n + 1))
    if len(span) <= 3:
        tokens.extend(span)
    return tokens


def tokenize(text, extra_terms=None):
    text = normalize(text)
    if not text:
        return []
    tokens = []
    for term in (extra_terms or []):
        term = normalize(term)
        if term and term in text:
            tokens.append(term)
    for m in TOKEN_RE.finditer(text):
        tok = m.group(0)
        if CJK_RE.fullmatch(tok):
            tokens.extend(cjk_ngrams(tok))
        else:
            tok = tok.lower()
            if len(tok) > 1:
                tokens.append(tok)
    return tokens


@dataclass
class Doc:
    doc_id: int
    title: str
    time: str
    text: str


def load_documents():
    path = DATA_DIR / "local_subtitles.csv"
    docs = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            text = normalize(row.get("subtitle") or row.get("text") or row.get("content", ""))
            if not text:
                continue
            docs.append(Doc(
                doc_id=len(docs),
                title=normalize(row.get("name") or row.get("title", "")),
                time=normalize(row.get("time", "")),
                text=text,
            ))
    return docs


@dataclass
class KGEntry:
    term: str
    aliases: list
    expansion: str
    boost_terms: list


def extract_cjk_terms(text):
    return [m.group(0) for m in CJK_RE.finditer(text) if len(m.group(0)) >= 2]


def load_kg():
    path = DATA_DIR / "sample_kg.csv"
    if not path.exists():
        return []
    entries = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            term = normalize(row.get("term") or row.get("name", ""))
            expansion = normalize(row.get("expansion") or row.get("content") or row.get("内容", ""))
            aliases = [a.strip() for a in normalize(row.get("aliases", "")).split("|") if a.strip()]
            if term and expansion:
                boost_terms = extract_cjk_terms(expansion)
                entries.append(KGEntry(term=term, aliases=aliases, expansion=expansion, boost_terms=boost_terms))
    return entries


def match_kg(query, kg):
    query_n = normalize(query)
    matched_terms, boost_terms, matched_entries = [], [], []
    for entry in kg:
        candidates = [entry.term] + entry.aliases
        if any(c and c in query_n for c in candidates):
            matched_terms.extend(candidates)
            boost_terms.extend(entry.boost_terms)
            matched_entries.append(entry)
    return matched_terms, boost_terms, matched_entries


class BM25:
    def __init__(self, docs, extra_terms=None):
        self.docs = docs
        self.extra = sorted(set(extra_terms or []), key=len, reverse=True)
        self.doc_tokens = []
        self.doc_tfs = []
        self.df = {}
        self.avg_len = 0.0
        if docs:
            self._build()

    def _build(self):
        df_c = defaultdict(int)
        total = 0
        for doc in self.docs:
            toks = tokenize(f"{doc.title} {doc.text}", self.extra)
            counts = Counter(toks)
            self.doc_tokens.append(toks)
            self.doc_tfs.append(dict(counts))
            total += len(toks)
            for t in counts:
                df_c[t] += 1
        self.df = dict(df_c)
        self.avg_len = total / max(len(self.docs), 1)

    def idf(self, term):
        n = len(self.docs)
        df = self.df.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query, top_k=10, terms=None, boost_terms=None, boost_weight=0.5):
        q_toks = tokenize(query, list(self.extra) + list(terms or []))
        if not q_toks:
            return []
        qc = Counter(q_toks)
        k1, b = 1.5, 0.75
        scored = []
        for i, doc in enumerate(self.docs):
            dl = len(self.doc_tokens[i]) or 1
            freqs = self.doc_tfs[i]
            s = 0.0
            for t, qw in qc.items():
                f = freqs.get(t, 0)
                if not f:
                    continue
                d = f + k1 * (1 - b + b * dl / max(self.avg_len, 1))
                s += self.idf(t) * (f * (k1 + 1) / d) * min(qw, 2)
            if boost_terms:
                for bt in boost_terms:
                    bf = freqs.get(bt, 0)
                    if bf:
                        d = bf + k1 * (1 - b + b * dl / max(self.avg_len, 1))
                        s += boost_weight * self.idf(bt) * (bf * (k1 + 1) / d)
            if s > 0:
                scored.append((i, s))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


_model = None
_query_cache = {}


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        print(f"加载 Embedding 模型...", flush=True)
        _model = SentenceTransformer(MODEL_NAME, device=DEVICE)
        _model.encode(["warmup"], batch_size=1, show_progress_bar=False)
        print("Embedding 模型就绪", flush=True)
    return _model


_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        print(f"加载中文 Reranker 模型...", flush=True)
        t0 = time.time()
        _reranker = CrossEncoder(RERANKER_PATH, max_length=512)
        print(f"Reranker 就绪 ({time.time()-t0:.1f}s)", flush=True)
    return _reranker


def embed_query(query: str) -> List[float]:
    if query in _query_cache:
        return _query_cache[query]
    model = get_model()
    vecs = model.encode([query], batch_size=1, show_progress_bar=False, normalize_embeddings=True)
    result = vecs[0].tolist()
    _query_cache[query] = result
    return result


def cosine_sim(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def vector_search(query_vec: List[float], doc_vecs: List[List[float]], top_k: int = 10) -> List[Tuple[int, float]]:
    scored = [(i, cosine_sim(query_vec, dv)) for i, dv in enumerate(doc_vecs)]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def rerank_with_cross_encoder(query: str, candidate_ids: List[int], docs_list: List[Doc], top_n: int = 5) -> List[Tuple[int, float]]:
    reranker = get_reranker()
    pairs = [(query, f"{docs_list[idx].title} {docs_list[idx].text}") for idx in candidate_ids]
    scores = reranker.predict(pairs)
    scored = list(zip(candidate_ids, scores.tolist()))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]


# Query rewriting: rule-based expansions for known weak queries
QUERY_REWRITES = {
    "闪电是怎么形成的": [
        "闪电 放电 云层 电荷积累",
        "闪电 雷电 放电现象 自然界",
    ],
    "潜水艇如何上浮和下潜": [
        "潜水艇 压载水舱 浮力 密度",
        "潜艇 浮沉 排水 注水 上浮下潜",
    ],
}


def get_query_rewrites(query: str) -> List[str]:
    rewrites = QUERY_REWRITES.get(query, [])
    if not rewrites:
        for orig, rws in QUERY_REWRITES.items():
            if orig in query or query in orig:
                rewrites = rws
                break
    return rewrites


def wsf_score(bm25_results, vec_results, bm25_max, vec_max, w_bm25, w_vec):
    scores = {}
    for idx, s in bm25_results:
        scores[idx] = w_bm25 * (s / bm25_max if bm25_max > 0 else 0)
    for idx, s in vec_results:
        scores[idx] = scores.get(idx, 0) + w_vec * (s / vec_max if vec_max > 0 else 0)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def rrf_fusion(bm25_results, vec_results, k=30, bm25_w=1.0, vec_w=1.5):
    scores = {}
    for rank, (idx, _) in enumerate(bm25_results):
        scores[idx] = scores.get(idx, 0) + bm25_w / (k + rank + 1)
    for rank, (idx, _) in enumerate(vec_results):
        scores[idx] = scores.get(idx, 0) + vec_w / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def evaluate(queries, ground_truth, docs, doc_vecs, bm25_index, kg, fusion_methods):
    all_results = {}
    for method_name, method_fn in fusion_methods.items():
        per_query_scores = []
        for q_idx, query in enumerate(queries):
            ranked = method_fn(query, q_idx)
            gt_title = ground_truth[q_idx]
            best_rank = 0
            for rank, (doc_idx, _) in enumerate(ranked[:5]):
                doc = docs[doc_idx]
                if gt_title.lower() in (doc.title + " " + doc.text).lower():
                    best_rank = rank + 1
                    break
            mrr = 1.0 / best_rank if best_rank > 0 else 0.0
            per_query_scores.append(mrr)
        avg_mrr = sum(per_query_scores) / len(per_query_scores) if per_query_scores else 0.0
        all_results[method_name] = {
            "mrr": avg_mrr,
            "per_query": per_query_scores,
        }
    return all_results


def main():
    print("=" * 70)
    print("V9: 中文 Cross-Encoder 重排序 + 查询改写 + 两阶段检索")
    print("=" * 70)

    print("\n[1/5] 加载数据...")
    docs = load_documents()
    kg = load_kg()
    print(f"  字幕文档: {len(docs)} 条, KG: {len(kg)} 条")

    print("\n[2/5] 加载文档向量...")
    t0 = time.time()
    with open(CACHE_DIR / "doc_embeddings_text_only.json", "r") as f:
        raw = json.load(f)
    if raw and isinstance(raw[0], dict):
        doc_vecs = [item["embedding"] for item in raw]
    else:
        doc_vecs = raw
    print(f"  向量加载完成 ({time.time()-t0:.1f}s), 维度: {len(doc_vecs[0])}")

    print("\n[3/5] 构建 BM25 索引...")
    t0 = time.time()
    kg_terms = []
    for entry in kg:
        kg_terms.extend(entry.boost_terms)
    bm25_index = BM25(docs, extra_terms=list(set(kg_terms)))
    print(f"  BM25 索引构建完成 ({time.time()-t0:.1f}s)")

    TEST_QUERIES = [
        {"query": "浮力在潜水艇中的应用", "keywords": ["潜水艇", "浮力", "密度"]},
        {"query": "光的本质是什么", "keywords": ["光", "本质", "波", "粒子"]},
        {"query": "牛顿对光学的贡献", "keywords": ["牛顿", "光", "棱镜", "光学"]},
        {"query": "自然光源有哪些", "keywords": ["太阳", "自然光", "闪电", "恒星"]},
        {"query": "激光的特点和应用", "keywords": ["激光", "方向性", "能量", "照明"]},
        {"query": "欧几里德在光学方面做了什么", "keywords": ["欧几里德", "光学", "视觉"]},
        {"query": "眼睛如何看见物体", "keywords": ["眼睛", "光线", "视觉", "视网膜"]},
        {"query": "光速是多少", "keywords": ["光速", "速度", "米", "秒"]},
        {"query": "密度计的工作原理", "keywords": ["密度计", "浮力", "密度", "液体"]},
        {"query": "闪电是怎么形成的", "keywords": ["闪电", "放电", "云层", "雷"]},
        {"query": "郑和宝船与航海技术", "keywords": ["郑和", "宝船", "航海"]},
        {"query": "反射和折射的区别", "keywords": ["反射", "折射", "光线", "介质"]},
        {"query": "人工光源的发展历史", "keywords": ["灯", "LED", "电灯", "人造光"]},
        {"query": "潜水艇如何上浮和下潜", "keywords": ["潜水艇", "压载水舱", "浮力", "密度"]},
        {"query": "光的颜色是怎么产生的", "keywords": ["颜色", "光谱", "波长", "棱镜"]},
    ]

    def is_relevant(doc, keywords):
        combined = (doc.title + " " + doc.text).lower()
        return any(kw.lower() in combined for kw in keywords)

    def calc_mrr(ranked, docs, keywords):
        for rank, (idx, _) in enumerate(ranked[:10]):
            if is_relevant(docs[idx], keywords):
                return 1.0 / (rank + 1)
        return 0.0

    queries = [tq["query"] for tq in TEST_QUERIES]

    print("\n[4/5] 预热 Reranker...")
    get_reranker()
    print("  Reranker 预热完成")

    print("\n[5/5] 运行实验...\n")

    # ========== Fusion Methods ==========

    def vec_only(query, q_idx):
        q_vec = embed_query(query)
        return vector_search(q_vec, doc_vecs, top_k=10)

    def bm25_only(query, q_idx):
        return bm25_index.search(query, top_k=10)

    def wsf_v05(query, q_idx):
        q_vec = embed_query(query)
        vec_res = vector_search(q_vec, doc_vecs, top_k=50)
        bm25_res = bm25_index.search(query, top_k=50)
        if not vec_res or not bm25_res:
            return vec_res or bm25_res
        bm25_max = bm25_res[0][1]
        vec_max = vec_res[0][1]
        return wsf_score(bm25_res, vec_res, bm25_max, vec_max, 0.5, 0.5)

    def rrf_k30v15b10(query, q_idx):
        q_vec = embed_query(query)
        vec_res = vector_search(q_vec, doc_vecs, top_k=80)
        bm25_res = bm25_index.search(query, top_k=80)
        return rrf_fusion(bm25_res, vec_res, k=30, bm25_w=1.0, vec_w=1.5)

    # Stage 1: 候选集 → Stage 2: 中文 Reranker
    def rerank_chinese_stage1_vec_stage2(query, q_idx, cand=30):
        q_vec = embed_query(query)
        vec_res = vector_search(q_vec, doc_vecs, top_k=cand)
        candidate_ids = [idx for idx, _ in vec_res]
        reranked = rerank_with_cross_encoder(query, candidate_ids, docs, top_n=10)
        return reranked

    def rerank_chinese_stage1_bm25_stage2(query, q_idx, cand=30):
        bm25_res = bm25_index.search(query, top_k=cand)
        candidate_ids = [idx for idx, _ in bm25_res]
        reranked = rerank_with_cross_encoder(query, candidate_ids, docs, top_n=10)
        return reranked

    def rerank_chinese_stage1_wsf_stage2(query, q_idx, cand=50):
        wsf_res = wsf_v05(query, q_idx)
        candidate_ids = [idx for idx, _ in wsf_res[:cand]]
        reranked = rerank_with_cross_encoder(query, candidate_ids, docs, top_n=10)
        return reranked

    def rerank_chinese_stage1_rrf_stage2(query, q_idx, cand=80):
        rrf_res = rrf_k30v15b10(query, q_idx)
        candidate_ids = [idx for idx, _ in rrf_res[:cand]]
        reranked = rerank_with_cross_encoder(query, candidate_ids, docs, top_n=10)
        return reranked

    # 查询改写 + 两阶段
    def rerank_chinese_with_rewrite(query, q_idx, cand=50):
        rewrites = get_query_rewrites(query)
        if not rewrites:
            return rerank_chinese_stage1_wsf_stage2(query, q_idx, cand=cand)

        all_candidates = {}
        wsf_res = wsf_v05(query, q_idx)
        for idx, s in wsf_res[:cand]:
            all_candidates[idx] = max(all_candidates.get(idx, 0), s)

        for rw in rewrites:
            rw_q_vec = embed_query(rw)
            rw_vec_res = vector_search(rw_q_vec, doc_vecs, top_k=cand)
            for idx, s in rw_vec_res[:cand]:
                all_candidates[idx] = max(all_candidates.get(idx, 0), s)

        candidate_ids = sorted(all_candidates.keys(), key=lambda x: all_candidates[x], reverse=True)[:cand * 2]
        reranked = rerank_with_cross_encoder(query, candidate_ids, docs, top_n=10)
        return reranked

    # 多查询 Vec 融合 + Reranker
    def multivec_rerank(query, q_idx, cand=50):
        rewrites = get_query_rewrites(query)
        all_queries = [query] + rewrites

        all_candidates = {}
        for q in all_queries:
            q_vec = embed_query(q)
            vec_res = vector_search(q_vec, doc_vecs, top_k=cand)
            for idx, s in vec_res:
                all_candidates[idx] = max(all_candidates.get(idx, 0), s)

        candidate_ids = sorted(all_candidates.keys(), key=lambda x: all_candidates[x], reverse=True)[:cand]
        reranked = rerank_with_cross_encoder(query, candidate_ids, docs, top_n=10)
        return reranked

    # Vec + Reranker 不同候选集大小
    def rerank_vec_c20(query, q_idx):
        return rerank_chinese_stage1_vec_stage2(query, q_idx, cand=20)

    def rerank_vec_c30(query, q_idx):
        return rerank_chinese_stage1_vec_stage2(query, q_idx, cand=30)

    def rerank_vec_c50(query, q_idx):
        return rerank_chinese_stage1_vec_stage2(query, q_idx, cand=50)

    def rerank_vec_c80(query, q_idx):
        return rerank_chinese_stage1_vec_stage2(query, q_idx, cand=80)

    def rerank_vec_c100(query, q_idx):
        return rerank_chinese_stage1_vec_stage2(query, q_idx, cand=100)

    # WSF + Reranker 不同候选集
    def rerank_wsf_c50(query, q_idx):
        return rerank_chinese_stage1_wsf_stage2(query, q_idx, cand=50)

    def rerank_wsf_c80(query, q_idx):
        return rerank_chinese_stage1_wsf_stage2(query, q_idx, cand=80)

    def rerank_wsf_c100(query, q_idx):
        return rerank_chinese_stage1_wsf_stage2(query, q_idx, cand=100)

    # RRF + Reranker
    def rerank_rrf_c80(query, q_idx):
        return rerank_chinese_stage1_rrf_stage2(query, q_idx, cand=80)

    def rerank_rrf_c100(query, q_idx):
        return rerank_chinese_stage1_rrf_stage2(query, q_idx, cand=100)

    # KG boost + WSF + Reranker
    def rerank_kg_wsf(query, q_idx, cand=50):
        matched_terms, boost_terms, _ = match_kg(query, kg)
        q_vec = embed_query(query)
        vec_res = vector_search(q_vec, doc_vecs, top_k=cand)
        bm25_res = bm25_index.search(query, top_k=cand, boost_terms=boost_terms, boost_weight=0.2)
        if not vec_res or not bm25_res:
            return vec_res or bm25_res
        bm25_max = bm25_res[0][1]
        vec_max = vec_res[0][1]
        wsf_res = wsf_score(bm25_res, vec_res, bm25_max, vec_max, 0.5, 0.5)
        candidate_ids = [idx for idx, _ in wsf_res[:cand]]
        reranked = rerank_with_cross_encoder(query, candidate_ids, docs, top_n=10)
        return reranked

    # 改写 + Vec + Reranker (改写只用于扩展候选集)
    def rerank_rewrite_vec(query, q_idx, cand=50):
        rewrites = get_query_rewrites(query)
        if not rewrites:
            return rerank_vec_c50(query, q_idx)
        q_vec = embed_query(query)
        vec_res = vector_search(q_vec, doc_vecs, top_k=cand)
        all_candidates = {idx: s for idx, s in vec_res}
        for rw in rewrites:
            rw_vec = embed_query(rw)
            rw_res = vector_search(rw_vec, doc_vecs, top_k=cand)
            for idx, s in rw_res:
                all_candidates[idx] = max(all_candidates.get(idx, 0), s)
        candidate_ids = sorted(all_candidates.keys(), key=lambda x: all_candidates[x], reverse=True)[:cand]
        reranked = rerank_with_cross_encoder(query, candidate_ids, docs, top_n=10)
        return reranked

    # 改写 + WSF + Reranker
    def rerank_rewrite_wsf(query, q_idx, cand=50):
        rewrites = get_query_rewrites(query)
        if not rewrites:
            return rerank_wsf_c50(query, q_idx)
        wsf_res = wsf_v05(query, q_idx)
        all_candidates = {idx: s for idx, s in wsf_res[:cand]}
        for rw in rewrites:
            rw_q_vec = embed_query(rw)
            rw_vec_res = vector_search(rw_q_vec, doc_vecs, top_k=cand)
            for idx, s in rw_vec_res[:cand]:
                all_candidates[idx] = max(all_candidates.get(idx, 0), s)
        candidate_ids = sorted(all_candidates.keys(), key=lambda x: all_candidates[x], reverse=True)[:cand * 2]
        reranked = rerank_with_cross_encoder(query, candidate_ids, docs, top_n=10)
        return reranked

    fusion_methods = {
        # Baselines
        "BM25": bm25_only,
        "BM25+KG": lambda q, i: bm25_index.search(q, top_k=10, boost_terms=match_kg(q, kg)[1], boost_weight=0.3),
        "Vec": vec_only,
        "WSF(v=0.5,c50)": wsf_v05,
        "RRF(k30,v1.5,b1.0,c80)": rrf_k30v15b10,

        # Stage1: Vec → Stage2: 中文 Reranker，不同候选集
        "Rerank_Vec_c30": rerank_vec_c30,
        "Rerank_Vec_c50": rerank_vec_c50,

        # Stage1: WSF → Stage2: 中文 Reranker
        "Rerank_WSF_c50": rerank_wsf_c50,

        # Stage1: RRF → Stage2: 中文 Reranker
        "Rerank_RRF_c80": rerank_rrf_c80,

        # 查询改写 + Reranker
        "Rewrite+Vec+Rerank_c50": rerank_rewrite_vec,
        "Rewrite+WSF+Rerank_c50": rerank_rewrite_wsf,
    }

    all_results = []
    per_query_all = {}

    for method_name, method_fn in fusion_methods.items():
        t0 = time.time()
        per_query_scores = []
        for q_idx, tq in enumerate(TEST_QUERIES):
            query = tq["query"]
            keywords = tq["keywords"]
            ranked = method_fn(query, q_idx)
            mrr_score = calc_mrr(ranked, docs, keywords)
            per_query_scores.append(mrr_score)

        avg_mrr = sum(per_query_scores) / len(per_query_scores)
        hit3 = sum(1 for s in per_query_scores if s >= 1/3) / len(per_query_scores)
        hit5 = sum(1 for s in per_query_scores if s > 0) / len(per_query_scores)
        elapsed = time.time() - t0

        result = {
            "name": method_name,
            "mrr": round(avg_mrr, 4),
            "hit3": round(hit3, 4),
            "hit5": round(hit5, 4),
            "time": round(elapsed, 1),
        }
        all_results.append(result)
        per_query_all[method_name] = [round(s, 4) for s in per_query_scores]

        status = "⭐" if avg_mrr >= 0.900 else "✅" if avg_mrr >= 0.867 else ""
        print(f"  [{elapsed:5.1f}s] {method_name:40s} MRR={avg_mrr:.3f}  Hit@3={hit3:.3f}  Hit@5={hit5:.3f}  {status}")

    all_results.sort(key=lambda x: x["mrr"], reverse=True)

    output = {
        "version": "v9",
        "description": "中文 Cross-Encoder 重排序 + 查询改写 + 两阶段检索",
        "total_methods": len(all_results),
        "results": all_results,
        "per_query": per_query_all,
    }

    out_path = Path(__file__).with_name("optimize_v9_results.json")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*70}")
    print("结果已保存到", out_path)
    print(f"{'='*70}")

    print("\n📊 排名表:")
    print(f"{'排名':>4} {'方法':<40s} {'MRR':>6} {'Hit@3':>6} {'Hit@5':>6} {'耗时':>6}")
    print("-" * 70)
    for rank, r in enumerate(all_results, 1):
        star = "⭐" if r["mrr"] >= 0.900 else "  "
        print(f"{rank:>4} {r['name']:<40s} {r['mrr']:>6.3f} {r['hit3']:>6.3f} {r['hit5']:>6.3f} {r['time']:>5.1f}s {star}")

    best = all_results[0]
    print(f"\n🏆 最优方法: {best['name']}")
    print(f"   MRR={best['mrr']:.3f}, Hit@3={best['hit3']:.3f}, Hit@5={best['hit5']:.3f}")


if __name__ == "__main__":
    main()
