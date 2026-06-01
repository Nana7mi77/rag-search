import csv
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

os.environ["HF_HUB_OFFLINE"] = "1"

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = Path(__file__).resolve().parent / "cache"

MODEL_NAME = "BAAI/bge-m3"
RERANKER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
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


def embed_query(query):
    if query in _query_cache:
        return _query_cache[query]
    model = get_model()
    vecs = model.encode([query], batch_size=1, show_progress_bar=False, normalize_embeddings=True)
    result = vecs[0].tolist()
    _query_cache[query] = result
    return result


def cosine_sim(a, b):
    return sum(x * y for x, y in zip(a, b))


def vector_search(query_vec, doc_vecs, top_k=10):
    scored = [(i, cosine_sim(query_vec, dv)) for i, dv in enumerate(doc_vecs)]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def wsf_fusion(list_a, list_b, w_a=0.5, w_b=0.5, top_n=50):
    max_a = max((s for _, s in list_a), default=1.0) or 1.0
    max_b = max((s for _, s in list_b), default=1.0) or 1.0
    scores = {}
    for idx, s in list_a:
        scores[idx] = scores.get(idx, 0) + w_a * (s / max_a)
    for idx, s in list_b:
        scores[idx] = scores.get(idx, 0) + w_b * (s / max_b)
    merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return merged[:top_n]


def rrf_fusion(list_a, list_b, k=60, w_a=1.0, w_b=1.0, top_n=50):
    scores = {}
    for rank, (idx, _) in enumerate(list_a, 1):
        scores[idx] = scores.get(idx, 0) + w_a / (k + rank)
    for rank, (idx, _) in enumerate(list_b, 1):
        scores[idx] = scores.get(idx, 0) + w_b / (k + rank)
    merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return merged[:top_n]


def load_doc_embeddings():
    cache_path = CACHE_DIR / "doc_embeddings_text_only.json"
    with cache_path.open() as f:
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


def is_relevant(doc, keywords):
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
    print("=" * 70, flush=True)
    print("  优化实验 V8: Cross-Encoder 重排序", flush=True)
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

    from sentence_transformers import CrossEncoder
    print(f"加载 Reranker: {RERANKER_NAME}...", flush=True)
    reranker = CrossEncoder(RERANKER_NAME, max_length=512)
    print("Reranker 就绪", flush=True)

    reranker_cache = {}

    def rerank(query, candidate_ids, docs_list, reranker_model, top_n=5):
        cache_key = (query, tuple(candidate_ids))
        if cache_key in reranker_cache:
            return reranker_cache[cache_key]

        pairs = [(query, f"{docs_list[idx].title} {docs_list[idx].text}") for idx in candidate_ids]
        scores = reranker_model.predict(pairs)
        scored = list(zip(candidate_ids, scores.tolist()))
        scored.sort(key=lambda x: x[1], reverse=True)
        result = scored[:top_n]
        reranker_cache[cache_key] = result
        return result

    print("预计算查询向量...", flush=True)
    for tq in TEST_QUERIES:
        embed_query(tq["query"])

    kg_expanded_queries = {}
    for tq in TEST_QUERIES:
        q = tq["query"]
        _, _, entries = match_kg(q, kg)
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

    experiments.append(("BM25", lambda q, kw, kg_: bm25.search(q, top_k=top_k)))
    experiments.append(("BM25+KG", lambda q, kw, kg_: bm25.search(
        normalize(f"{q} {' '.join(match_kg(q, kg_)[1][:5])}"), top_k=top_k)))
    experiments.append(("Vec", lambda q, kw, kg_: vector_search(embed_query(q), vecs, top_k=top_k)))

    def v5_best(q, kw, kg_):
        return wsf_fusion(
            vector_search(embed_query(q), vecs, top_k=50),
            bm25.search(q, top_k=50),
            w_a=0.5, w_b=0.5, top_n=top_k)
    experiments.append(("V5-WSF(v=0.5)", v5_best))

    def v7_best_rrf(q, kw, kg_):
        return rrf_fusion(
            vector_search(embed_query(q), vecs, top_k=80),
            bm25.search(q, top_k=80),
            k=30, w_a=1.5, w_b=1.0, top_n=top_k)
    experiments.append(("V7-RRF(k30,v1.5,b1.0)", v7_best_rrf))

    for cand_size in [15, 20, 30, 50]:
        def make_rerank_wsftop(c):
            def fn(q, kw, kg_):
                wsf_result = wsf_fusion(
                    vector_search(embed_query(q), vecs, top_k=50),
                    bm25.search(q, top_k=50),
                    w_a=0.5, w_b=0.5, top_n=c)
                candidate_ids = [idx for idx, _ in wsf_result]
                reranked = rerank(q, candidate_ids, docs, reranker, top_n=top_k)
                return reranked
            return fn
        experiments.append((f"Rerank_WSF_top{cand_size}", make_rerank_wsftop(cand_size)))

    for cand_size in [15, 20, 30, 50]:
        def make_rerank_rrftop(c):
            def fn(q, kw, kg_):
                rrf_result = rrf_fusion(
                    vector_search(embed_query(q), vecs, top_k=80),
                    bm25.search(q, top_k=80),
                    k=30, w_a=1.5, w_b=1.0, top_n=c)
                candidate_ids = [idx for idx, _ in rrf_result]
                reranked = rerank(q, candidate_ids, docs, reranker, top_n=top_k)
                return reranked
            return fn
        experiments.append((f"Rerank_RRF_top{cand_size}", make_rerank_rrftop(cand_size)))

    for cand_size in [20, 30, 50]:
        def make_rerank_both(c):
            def fn(q, kw, kg_):
                bm25_top = bm25.search(q, top_k=c)
                vec_top = vector_search(embed_query(q), vecs, top_k=c)
                candidate_ids = list(set([idx for idx, _ in bm25_top] + [idx for idx, _ in vec_top]))
                reranked = rerank(q, candidate_ids, docs, reranker, top_n=top_k)
                return reranked
            return fn
        experiments.append((f"Rerank_BM25+Vec_top{cand_size}", make_rerank_both(cand_size)))

    for cand_size in [30, 50]:
        def make_rerank_wsf_kg(c):
            def fn(q, kw, kg_):
                eq = kg_expanded_queries.get(q, q)
                _, bt, _ = match_kg(q, kg_)
                wsf_result = wsf_fusion(
                    vector_search(embed_query(eq), vecs, top_k=50),
                    bm25.search(q, top_k=50, boost_terms=bt, boost_weight=0.3),
                    w_a=0.5, w_b=0.5, top_n=c)
                candidate_ids = [idx for idx, _ in wsf_result]
                reranked = rerank(q, candidate_ids, docs, reranker, top_n=top_k)
                return reranked
            return fn
        experiments.append((f"Rerank_WSF_KG_top{cand_size}", make_rerank_wsf_kg(cand_size)))

    def rerank_vec_fallback(q, kw, kg_):
        wsf_result = wsf_fusion(
            vector_search(embed_query(q), vecs, top_k=50),
            bm25.search(q, top_k=50),
            w_a=0.5, w_b=0.5, top_n=20)
        candidate_ids = [idx for idx, _ in wsf_result]
        reranked = rerank(q, candidate_ids, docs, reranker, top_n=top_k)

        bm25_top = bm25.search(q, top_k=5)
        vec_top = vector_search(embed_query(q), vecs, top_k=5)

        max_bm25 = max((s for _, s in bm25_top), default=1.0) or 1.0
        bm25_conf = (bm25_top[0][1] / max_bm25) if bm25_top else 0
        vec_conf = vec_top[0][1] if vec_top else 0

        reranker_top_score = reranked[0][1] if reranked else -999

        if reranker_top_score > 5.0:
            return reranked

        if bm25_conf < 0.4 and vec_conf > 0.4:
            return vec_top[:top_k]

        return reranked

    experiments.append(("Rerank_vec_fallback", rerank_vec_fallback))

    def rerank_boost_top(q, kw, kg_):
        wsf_result = wsf_fusion(
            vector_search(embed_query(q), vecs, top_k=50),
            bm25.search(q, top_k=50),
            w_a=0.5, w_b=0.5, top_n=30)
        candidate_ids = [idx for idx, _ in wsf_result]

        bm25_top = bm25.search(q, top_k=3)
        vec_top = vector_search(embed_query(q), vecs, top_k=3)
        for idx, _ in bm25_top:
            if idx not in candidate_ids:
                candidate_ids.append(idx)
        for idx, _ in vec_top:
            if idx not in candidate_ids:
                candidate_ids.append(idx)

        reranked = rerank(q, candidate_ids, docs, reranker, top_n=top_k)
        return reranked

    experiments.append(("Rerank_boost_top", rerank_boost_top))

    def rerank_with_query_expansion(q, kw, kg_):
        eq = kg_expanded_queries.get(q, q)
        vec_top = vector_search(embed_query(eq), vecs, top_k=30)
        bm25_top = bm25.search(q, top_k=30)
        candidate_ids = list(set([idx for idx, _ in vec_top] + [idx for idx, _ in bm25_top]))
        reranked = rerank(q, candidate_ids, docs, reranker, top_n=top_k)
        return reranked

    experiments.append(("Rerank_query_expansion", rerank_with_query_expansion))

    def rerank_dual_query(q, kw, kg_):
        eq = kg_expanded_queries.get(q, q)
        vec_top = vector_search(embed_query(q), vecs, top_k=20)
        vec_exp_top = vector_search(embed_query(eq), vecs, top_k=20)
        bm25_top = bm25.search(q, top_k=20)
        candidate_ids = list(set(
            [idx for idx, _ in vec_top] +
            [idx for idx, _ in vec_exp_top] +
            [idx for idx, _ in bm25_top]
        ))
        reranked = rerank(q, candidate_ids, docs, reranker, top_n=top_k)
        return reranked

    experiments.append(("Rerank_dual_query", rerank_dual_query))

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
    print_table(headers, rows, title="优化实验 V8 结果")

    print(f"\n对比基线: BM25=0.769  BM25+KG=0.867  V5最优=0.900", flush=True)

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
        else:
            print(f"\n所有查询 MRR=1.000! 完美!", flush=True)

    out_path = Path(__file__).resolve().parent / "optimize_v8_results.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({
            "results": [{k: v for k, v in r.items() if k != "per_query_mrr"} for r in results],
            "per_query": {r["name"]: r["per_query_mrr"] for r in results}
        }, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {out_path}", flush=True)


if __name__ == "__main__":
    main()
