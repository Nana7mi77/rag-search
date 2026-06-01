import csv
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ["HF_HUB_OFFLINE"] = "1"

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = Path(__file__).resolve().parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

MODEL_NAME = "BAAI/bge-m3"
DEVICE = "mps"
CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
TOKEN_RE = re.compile(r"[\u4e00-\u9fff]+|[a-zA-Z0-9_]+")


def normalize(text: object) -> str:
    if text is None:
        return ""
    text = str(text).replace("\ufeff", "").replace("\u3000", " ")
    return re.sub(r"\s+", " ", text).strip()


def cjk_ngrams(span: str) -> List[str]:
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


def tokenize(text: object, extra_terms: List[str] = None) -> List[str]:
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


def extract_cjk_terms(text: str) -> List[str]:
    terms = []
    for m in CJK_RE.finditer(text):
        tok = m.group(0)
        if len(tok) >= 2:
            terms.append(tok)
    return terms


@dataclass
class Doc:
    doc_id: int
    title: str
    time: str
    text: str


def load_documents() -> List[Doc]:
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
    aliases: List[str]
    expansion: str
    boost_terms: List[str]


def load_kg() -> List[KGEntry]:
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


def match_kg(query: str, kg: List[KGEntry]) -> Tuple[List[str], List[str], List[KGEntry]]:
    query_n = normalize(query)
    matched_terms = []
    boost_terms = []
    matched_entries = []
    for entry in kg:
        candidates = [entry.term] + entry.aliases
        if any(c and c in query_n for c in candidates):
            matched_terms.extend(candidates)
            boost_terms.extend(entry.boost_terms)
            matched_entries.append(entry)
    return matched_terms, boost_terms, matched_entries


class BM25:
    def __init__(self, docs: List[Doc], extra_terms: List[str] = None):
        self.docs = docs
        self.extra = sorted(set(extra_terms or []), key=len, reverse=True)
        self.doc_tokens = []
        self.doc_tfs = []
        self.df: Dict[str, int] = {}
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

    def idf(self, term: str) -> float:
        n = len(self.docs)
        df = self.df.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 10, terms: List[str] = None,
               boost_terms: List[str] = None, boost_weight: float = 0.5) -> List[Tuple[int, float]]:
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
_query_cache: Dict[str, List[float]] = {}


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        print(f"加载模型 {MODEL_NAME} 到 {DEVICE}...", flush=True)
        _model = SentenceTransformer(MODEL_NAME, device=DEVICE)
        print("MPS warmup...", flush=True)
        _model.encode(["warmup"], batch_size=1, show_progress_bar=False)
        print("模型就绪", flush=True)
    return _model


def embed_query(query: str) -> List[float]:
    if query in _query_cache:
        return _query_cache[query]
    model = get_model()
    vecs = model.encode([query], batch_size=1, show_progress_bar=False, normalize_embeddings=True)
    result = vecs[0].tolist()
    _query_cache[query] = result
    return result


def embed_queries(queries: List[str]) -> List[List[float]]:
    model = get_model()
    results = []
    for q in queries:
        results.append(embed_query(q))
    return results


def cosine_sim(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot


def vector_search(query_vec: List[float], doc_vecs: List[List[float]], top_k: int = 10) -> List[Tuple[int, float]]:
    scored = [(i, cosine_sim(query_vec, dv)) for i, dv in enumerate(doc_vecs)]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def multi_vector_search(query_vecs: List[List[float]], doc_vecs: List[List[float]], top_k: int = 10) -> List[Tuple[int, float]]:
    n_docs = len(doc_vecs)
    scores = [0.0] * n_docs
    for qv in query_vecs:
        for i, dv in enumerate(doc_vecs):
            scores[i] += cosine_sim(qv, dv)
    scored = [(i, s) for i, s in enumerate(scores)]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def load_doc_embeddings() -> List[List[float]]:
    cache_path = CACHE_DIR / "doc_embeddings_text_only.json"
    print(f"加载缓存: {cache_path.name}", flush=True)
    with cache_path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def is_relevant(doc: Doc, keywords: List[str]) -> bool:
    combined = f"{doc.title} {doc.text}".lower()
    return any(kw.lower() in combined for kw in keywords)


def mrr(results, docs, keywords):
    for rank, (idx, _) in enumerate(results, 1):
        if is_relevant(docs[idx], keywords):
            return 1.0 / rank
    return 0.0


def hit_at_k(results, docs, keywords, k=3):
    for idx, _ in results[:k]:
        if is_relevant(docs[idx], keywords):
            return 1.0
    return 0.0


def rrf_fusion(list_a, list_b, k=60, w_a=1.0, w_b=1.0, top_k=5):
    scores: Dict[int, float] = {}
    for rank, (idx, _) in enumerate(list_a, 1):
        scores[idx] = scores.get(idx, 0) + w_a / (k + rank)
    for rank, (idx, _) in enumerate(list_b, 1):
        scores[idx] = scores.get(idx, 0) + w_b / (k + rank)
    merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return merged[:top_k]


def wsf_fusion(list_a, list_b, w_a=0.5, w_b=0.5, top_k=5):
    max_a = max((s for _, s in list_a), default=1.0) or 1.0
    max_b = max((s for _, s in list_b), default=1.0) or 1.0
    scores: Dict[int, float] = {}
    for idx, s in list_a:
        scores[idx] = scores.get(idx, 0) + w_a * (s / max_a)
    for idx, s in list_b:
        scores[idx] = scores.get(idx, 0) + w_b * (s / max_b)
    merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return merged[:top_k]


def max_fusion(list_a, list_b, w_a=0.5, w_b=0.5, top_k=5):
    max_a = max((s for _, s in list_a), default=1.0) or 1.0
    max_b = max((s for _, s in list_b), default=1.0) or 1.0
    scores: Dict[int, float] = {}
    for idx, s in list_a:
        scores[idx] = max(scores.get(idx, 0), w_a * (s / max_a))
    for idx, s in list_b:
        scores[idx] = max(scores.get(idx, 0), w_b * (s / max_b))
    merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return merged[:top_k]


def evaluate(name, fn, docs, kg):
    mrrs, h3s, h5s = [], [], []
    per_q = []
    for tq in TEST_QUERIES:
        r = fn(tq["query"], tq["keywords"], kg)
        m = mrr(r, docs, tq["keywords"])
        h3 = hit_at_k(r, docs, tq["keywords"], 3)
        h5 = hit_at_k(r, docs, tq["keywords"], 5)
        mrrs.append(m)
        h3s.append(h3)
        h5s.append(h5)
        per_q.append(m)
    return {"name": name, "mrr": sum(mrrs) / len(mrrs), "hit3": sum(h3s) / len(h3s), "hit5": sum(h5s) / len(h5s), "per_query_mrr": per_q}


def print_table(headers, rows, title=""):
    if title:
        print(f"\n{'=' * 70}")
        print(f"  {title}")
        print(f"{'=' * 70}")
    col_widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0)) for i, h in enumerate(headers)]
    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    hdr = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, col_widths)) + " |"
    print(sep)
    print(hdr)
    print(sep)
    for row in rows:
        line = "| " + " | ".join(str(v).ljust(w) for v, w in zip(row, col_widths)) + " |"
        print(line)
    print(sep)


def main():
    top_k = 5
    cand_large = 50
    cand_med = 30
    print("=" * 70, flush=True)
    print("  优化实验 V4: 查询扩展 + 大候选集 + 多种融合策略", flush=True)
    print("=" * 70, flush=True)

    docs = load_documents()
    kg = load_kg()
    kg_terms = []
    for e in kg:
        kg_terms.append(e.term)
        kg_terms.extend(e.aliases)
    bm25 = BM25(docs, extra_terms=kg_terms)
    vecs = load_doc_embeddings()
    print(f"文档: {len(docs)}, 向量已加载", flush=True)

    print("预计算查询向量...", flush=True)
    for tq in TEST_QUERIES:
        embed_query(tq["query"])

    kg_expanded_queries = {}
    for tq in TEST_QUERIES:
        q = tq["query"]
        _, boost_terms, entries = match_kg(q, kg)
        expanded_parts = []
        for e in entries:
            expanded_parts.append(e.term)
            expanded_parts.extend(e.aliases[:2])
            expanded_parts.extend(e.boost_terms[:3])
        if expanded_parts:
            expanded_q = f"{q} {' '.join(dict.fromkeys(expanded_parts))}"
        else:
            expanded_q = q
        kg_expanded_queries[q] = expanded_q
        embed_query(expanded_q)

    print(f"  缓存了 {len(_query_cache)} 条查询向量", flush=True)

    experiments = []

    def bm25_only(q, kw, kg_):
        return bm25.search(q, top_k=top_k)

    def bm25_kg(q, kw, kg_):
        _, bt, _ = match_kg(q, kg_)
        return bm25.search(normalize(f"{q} {' '.join(bt[:5])}"), top_k=top_k)

    def vec_only(q, kw, kg_):
        return vector_search(embed_query(q), vecs, top_k=top_k)

    experiments.append(("BM25", bm25_only))
    experiments.append(("BM25+KG", bm25_kg))
    experiments.append(("Vec", vec_only))

    def v2_best(q, kw, kg_):
        return rrf_fusion(
            vector_search(embed_query(q), vecs, top_k=cand_large),
            bm25.search(q, top_k=cand_large),
            k=60, w_a=0.5, w_b=3.0, top_k=top_k)

    experiments.append(("V2-RRF(k60,b3,v0.5)", v2_best))

    for ck in [30, 50, 80]:
        def make_rrf(ck_):
            def fn(q, kw, kg_):
                return rrf_fusion(
                    vector_search(embed_query(q), vecs, top_k=ck_),
                    bm25.search(q, top_k=ck_),
                    k=60, w_a=0.5, w_b=3.0, top_k=top_k)
            return fn
        experiments.append((f"RRF(cand={ck})", make_rrf(ck)))

    for k_val in [30, 60, 90]:
        for wb in [1.5, 2.0, 3.0]:
            for vv in [0.5, 1.0]:
                def make_rrf2(k_, wb_, vv_):
                    def fn(q, kw, kg_):
                        return rrf_fusion(
                            vector_search(embed_query(q), vecs, top_k=cand_large),
                            bm25.search(q, top_k=cand_large),
                            k=k_, w_a=vv_, w_b=wb_, top_k=top_k)
                    return fn
                experiments.append((f"RRF(k{k_val},b{wb},v{vv})", make_rrf2(k_val, wb, vv)))

    for wa in [0.3, 0.5, 0.7]:
        def make_wsf(wa_):
            def fn(q, kw, kg_):
                return wsf_fusion(
                    vector_search(embed_query(q), vecs, top_k=cand_large),
                    bm25.search(q, top_k=cand_large),
                    w_a=wa_, w_b=1.0 - wa_, top_k=top_k)
            return fn
        experiments.append((f"WSF(vec={wa:.1f})", make_wsf(wa)))

    for wa in [0.3, 0.5, 0.7]:
        def make_max(wa_):
            def fn(q, kw, kg_):
                return max_fusion(
                    vector_search(embed_query(q), vecs, top_k=cand_large),
                    bm25.search(q, top_k=cand_large),
                    w_a=wa_, w_b=1.0 - wa_, top_k=top_k)
            return fn
        experiments.append((f"MAX(vec={wa:.1f})", make_max(wa)))

    def vec_kg_expand(q, kw, kg_):
        eq = kg_expanded_queries.get(q, q)
        return vector_search(embed_query(eq), vecs, top_k=top_k)

    experiments.append(("Vec+KG_expand", vec_kg_expand))

    for wb in [2.0, 3.0]:
        for vv in [0.5, 1.0]:
            def make_rrf_kg_exp(wb_, vv_):
                def fn(q, kw, kg_):
                    eq = kg_expanded_queries.get(q, q)
                    return rrf_fusion(
                        vector_search(embed_query(eq), vecs, top_k=cand_large),
                        bm25.search(q, top_k=cand_large),
                        k=60, w_a=vv_, w_b=wb_, top_k=top_k)
                return fn
            experiments.append((f"RRF+KG_exp(b{wb},v{vv})", make_rrf_kg_exp(wb, vv)))

    def rrf_adaptive(q, kw, kg_):
        _, _, entries = match_kg(q, kg_)
        bm25_results = bm25.search(q, top_k=cand_large)
        eq = kg_expanded_queries.get(q, q)
        vec_results = vector_search(embed_query(eq), vecs, top_k=cand_large)
        if entries:
            return rrf_fusion(vec_results, bm25_results, k=60, w_a=1.0, w_b=2.0, top_k=top_k)
        else:
            return rrf_fusion(vec_results, bm25_results, k=60, w_a=0.5, w_b=3.0, top_k=top_k)

    experiments.append(("RRF_adaptive", rrf_adaptive))

    def multi_q_vec(q, kw, kg_):
        eq = kg_expanded_queries.get(q, q)
        _, _, entries = match_kg(q, kg_)
        q_variants = [q]
        if entries:
            for e in entries:
                q_variants.append(f"{q} {e.expansion[:50]}")
        q_vecs = embed_queries(q_variants)
        return multi_vector_search(q_vecs, vecs, top_k=top_k)

    experiments.append(("MultiQ_Vec", multi_q_vec))

    def multi_q_rrf(q, kw, kg_):
        eq = kg_expanded_queries.get(q, q)
        _, _, entries = match_kg(q, kg_)
        q_variants = [q]
        if entries:
            for e in entries:
                q_variants.append(f"{q} {e.expansion[:50]}")
        q_vecs = embed_queries(q_variants)
        vec_results = multi_vector_search(q_vecs, vecs, top_k=cand_large)
        bm25_results = bm25.search(q, top_k=cand_large, boost_terms=match_kg(q, kg_)[1], boost_weight=0.3)
        return rrf_fusion(vec_results, bm25_results, k=60, w_a=1.0, w_b=2.0, top_k=top_k)

    experiments.append(("MultiQ_RRF", multi_q_rrf))

    def vec_expanded_boost(q, kw, kg_):
        eq = kg_expanded_queries.get(q, q)
        vec_results = vector_search(embed_query(eq), vecs, top_k=cand_large)
        bm25_results = bm25.search(q, top_k=cand_large, boost_terms=match_kg(q, kg_)[1], boost_weight=0.5)
        return max_fusion(vec_results, bm25_results, w_a=0.7, w_b=0.3, top_k=top_k)

    experiments.append(("MaxFusion+KG(vec0.7)", vec_expanded_boost))

    print(f"\n运行 {len(experiments)} 组实验 x {len(TEST_QUERIES)} 条查询...", flush=True)
    results = []
    for i, (name, fn) in enumerate(experiments):
        if i % 5 == 0:
            print(f"  [{i + 1}/{len(experiments)}] ...", flush=True)
        r = evaluate(name, fn, docs, kg)
        results.append(r)

    results.sort(key=lambda x: x["mrr"], reverse=True)

    headers = ["Method", "MRR", "Hit@3", "Hit@5"]
    rows = [[r["name"], f"{r['mrr']:.3f}", f"{r['hit3']:.3f}", f"{r['hit5']:.3f}"] for r in results]
    print_table(headers, rows, title="优化实验 V4 结果")

    print(f"\n对比基线: BM25=0.769  BM25+KG=0.867  V2-最优RRF=0.889", flush=True)

    if results:
        top3 = results[:3]
        print(f"\nTop 3:", flush=True)
        for r in top3:
            print(f"  {r['name']}: MRR={r['mrr']:.3f} Hit@3={r['hit3']:.3f} Hit@5={r['hit5']:.3f}", flush=True)

    queries_names = [tq["query"] for tq in TEST_QUERIES]
    best_method = results[0] if results else None
    if best_method:
        pqr = best_method["per_query_mrr"]
        weak = [(i, queries_names[i], pqr[i]) for i in range(len(pqr)) if pqr[i] < 1.0]
        if weak:
            print(f"\n最优方法 [{best_method['name']}] 的弱查询:", flush=True)
            for i, qn, v in weak:
                print(f"  Q{i}: {qn} -> MRR={v:.3f}", flush=True)

    out_path = Path(__file__).resolve().parent / "optimize_v4_results.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({
            "results": [{k: v for k, v in r.items() if k != "per_query_mrr"} for r in results],
            "per_query": {r["name"]: r["per_query_mrr"] for r in results}
        }, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {out_path}", flush=True)


if __name__ == "__main__":
    main()
