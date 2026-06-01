import csv
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

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
                entries.append(KGEntry(term=term, aliases=aliases, expansion=expansion))
    return entries


def expand_query_with_kg(query: str, kg: List[KGEntry]) -> Tuple[str, List[str]]:
    query_n = normalize(query)
    matched = []
    expansions = []
    for entry in kg:
        candidates = [entry.term] + entry.aliases
        if any(c and c in query_n for c in candidates):
            matched.append(entry.term)
            expansions.append(entry.expansion)
    if expansions:
        return normalize(f"{query_n} {' '.join(expansions)}"), matched
    return query_n, []


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

    def search(self, query: str, top_k: int = 10, terms: List[str] = None) -> List[Tuple[int, float]]:
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


def cosine_sim(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def vector_search(query_vec: List[float], doc_vecs: List[List[float]], top_k: int = 10) -> List[Tuple[int, float]]:
    scored = [(i, cosine_sim(query_vec, dv)) for i, dv in enumerate(doc_vecs)]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def load_doc_embeddings(mode: str = "text_only") -> List[List[float]]:
    cache_path = CACHE_DIR / f"doc_embeddings_{mode}.json"
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


def score_fusion(list_a, list_b, w_a=1.0, w_b=1.0, top_k=5):
    scores: Dict[int, float] = {}
    for idx, s in list_a:
        scores[idx] = scores.get(idx, 0) + w_a * s
    for idx, s in list_b:
        scores[idx] = scores.get(idx, 0) + w_b * s
    merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return merged[:top_k]


def evaluate(name, fn, docs, kg):
    mrrs, h3s, h5s = [], [], []
    for tq in TEST_QUERIES:
        r = fn(tq["query"], tq["keywords"], kg)
        mrrs.append(mrr(r, docs, tq["keywords"]))
        h3s.append(hit_at_k(r, docs, tq["keywords"], 3))
        h5s.append(hit_at_k(r, docs, tq["keywords"], 5))
    return {"name": name, "mrr": sum(mrrs) / len(mrrs), "hit3": sum(h3s) / len(h3s), "hit5": sum(h5s) / len(h5s), "per_query_mrr": mrrs}


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
    print("  优化实验 V2: 精调权重 + 扩大候选 + 分数融合", flush=True)
    print("=" * 70, flush=True)

    docs = load_documents()
    kg = load_kg()
    kg_terms = []
    for e in kg:
        kg_terms.append(e.term)
        kg_terms.extend(e.aliases)
    bm25 = BM25(docs, extra_terms=kg_terms)
    vecs = load_doc_embeddings("text_only")
    print(f"文档: {len(docs)}, 向量已加载", flush=True)

    print("预计算查询向量...", flush=True)
    for tq in TEST_QUERIES:
        embed_query(tq["query"])
        expanded, _ = expand_query_with_kg(tq["query"], kg)
        if expanded != tq["query"]:
            embed_query(expanded)
    print(f"  缓存了 {len(_query_cache)} 条查询向量", flush=True)

    experiments = []

    experiments.append(("BM25", lambda q, kw, kg_: bm25.search(q, top_k=top_k)))
    experiments.append(("BM25+KG", lambda q, kw, kg_: bm25.search(expand_query_with_kg(q, kg_)[0], top_k=top_k)))
    experiments.append(("Vec", lambda q, kw, kg_: vector_search(embed_query(q), vecs, top_k=top_k)))

    for bm25_w in [1.5, 2.0, 2.5, 3.0]:
        for vec_w in [0.3, 0.5, 0.7, 1.0]:
            name = f"RRF(k60 bm25={bm25_w} vec={vec_w})"
            bw, vw = bm25_w, vec_w
            def make_fn(bw, vw):
                def fn(q, kw, kg_):
                    qv = embed_query(q)
                    vr = vector_search(qv, vecs, top_k=top_k * 3)
                    br = bm25.search(q, top_k=top_k * 3)
                    return rrf_fusion(vr, br, w_a=vw, w_b=bw, top_k=top_k)
                return fn
            experiments.append((name, make_fn(bw, vw)))

    for bm25_w in [1.5, 2.0, 2.5, 3.0]:
        for vec_w in [0.3, 0.5, 0.7, 1.0]:
            name = f"RRF(k30 bm25={bm25_w} vec={vec_w})"
            bw, vw = bm25_w, vec_w
            def make_fn_k30(bw, vw):
                def fn(q, kw, kg_):
                    qv = embed_query(q)
                    vr = vector_search(qv, vecs, top_k=top_k * 3)
                    br = bm25.search(q, top_k=top_k * 3)
                    return rrf_fusion(vr, br, k=30, w_a=vw, w_b=bw, top_k=top_k)
                return fn
            experiments.append((name, make_fn_k30(bw, vw)))

    for bm25_w in [2.0, 3.0]:
        for vec_w in [0.5, 0.7]:
            name = f"RRF+KG(k60 bm25={bm25_w} vec={vec_w})"
            bw, vw = bm25_w, vec_w
            def make_fn_kg(bw, vw):
                def fn(q, kw, kg_):
                    qv = embed_query(q)
                    expanded, _ = expand_query_with_kg(q, kg_)
                    vr = vector_search(qv, vecs, top_k=top_k * 3)
                    br = bm25.search(expanded, top_k=top_k * 3)
                    return rrf_fusion(vr, br, w_a=vw, w_b=bw, top_k=top_k)
                return fn
            experiments.append((name, make_fn_kg(bw, vw)))

    print(f"\n运行 {len(experiments)} 组实验 x {len(TEST_QUERIES)} 条查询...", flush=True)
    results = []
    for i, (name, fn) in enumerate(experiments):
        if i % 10 == 0:
            print(f"  [{i + 1}/{len(experiments)}] ...", flush=True)
        r = evaluate(name, fn, docs, kg)
        results.append(r)

    results.sort(key=lambda x: x["mrr"], reverse=True)

    print("\n--- TOP 15 ---", flush=True)
    headers = ["Method", "MRR", "Hit@3", "Hit@5"]
    rows = [[r["name"], f"{r['mrr']:.3f}", f"{r['hit3']:.3f}", f"{r['hit5']:.3f}"] for r in results[:15]]
    print_table(headers, rows, title="优化实验 V2 结果（Top 15）")

    best_no_kg = [r for r in results if "+KG" not in r["name"] and "baseline" not in r["name"]]
    best_with_kg = [r for r in results if "+KG" in r["name"] or r["name"].startswith("BM25+KG")]

    if best_no_kg:
        print(f"\n无KG最优: {best_no_kg[0]['name']} (MRR={best_no_kg[0]['mrr']:.3f})", flush=True)
    if best_with_kg:
        print(f"含KG最优: {best_with_kg[0]['name']} (MRR={best_with_kg[0]['mrr']:.3f})", flush=True)

    print(f"\nBM25 baseline: 0.769  BM25+KG baseline: 0.867", flush=True)

    out_path = Path(__file__).resolve().parent / "optimize_v2_results.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"results": [{k: v for k, v in r.items() if k != "per_query_mrr"} for r in results],
                    "per_query": {r["name"]: r["per_query_mrr"] for r in results}}, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {out_path}", flush=True)


if __name__ == "__main__":
    main()
