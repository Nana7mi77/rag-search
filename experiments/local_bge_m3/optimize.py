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
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索中文文档："


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


def embed_batch(texts: List[str], batch_size: int = 32, instruction: str = None) -> List[List[float]]:
    model = get_model()
    if instruction:
        texts = [instruction + t for t in texts]
    vecs = model.encode(texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)
    return vecs.tolist()


def embed_query(query: str, instruction: str = None) -> List[float]:
    return embed_batch([query], instruction=instruction)[0]


def cosine_sim(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def add_vectors(a: List[float], b: List[float], weight_a: float = 1.0, weight_b: float = 0.5) -> List[float]:
    result = [weight_a * x + weight_b * y for x, y in zip(a, b)]
    norm = math.sqrt(sum(x * x for x in result))
    if norm > 0:
        result = [x / norm for x in result]
    return result


def vector_search(query_vec: List[float], doc_vecs: List[List[float]], top_k: int = 10) -> List[Tuple[int, float]]:
    scored = [(i, cosine_sim(query_vec, dv)) for i, dv in enumerate(doc_vecs)]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def get_doc_embeddings(docs: List[Doc], mode: str = "text_only", force: bool = False) -> List[List[float]]:
    cache_path = CACHE_DIR / f"doc_embeddings_{mode}.json"
    if cache_path.exists() and not force:
        print(f"  加载缓存: {cache_path.name}", flush=True)
        with cache_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    print(f"  计算文档向量 (mode={mode})...", flush=True)
    if mode == "title_text":
        texts = [f"{doc.title} {doc.text}"[:300] for doc in docs]
    else:
        texts = [doc.text[:200] for doc in docs]
    t0 = time.time()
    vecs = []
    for i in range(0, len(texts), 32):
        batch = texts[i:i + 32]
        if i % 1024 == 0:
            print(f"    {i + 1}-{min(i + 32, len(texts))}/{len(texts)}", flush=True)
        vecs.extend(embed_batch(batch))
    elapsed = time.time() - t0
    print(f"  完成，耗时 {elapsed:.1f}s", flush=True)
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(vecs, f)
    return vecs


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


def evaluate(method_name: str, get_results_fn, docs, kg) -> Dict:
    mrrs, h3s, h5s = [], [], []
    for tq in TEST_QUERIES:
        r = get_results_fn(tq["query"], tq["keywords"], kg)
        mrrs.append(mrr(r, docs, tq["keywords"]))
        h3s.append(hit_at_k(r, docs, tq["keywords"], 3))
        h5s.append(hit_at_k(r, docs, tq["keywords"], 5))
    return {
        "name": method_name,
        "mrr": sum(mrrs) / len(mrrs),
        "hit3": sum(h3s) / len(h3s),
        "hit5": sum(h5s) / len(h5s),
        "per_query_mrr": mrrs,
    }


def rrf_fusion(list_a: List[Tuple[int, float]], list_b: List[Tuple[int, float]],
               k: int = 60, w_a: float = 1.0, w_b: float = 1.0, top_k: int = 5) -> List[Tuple[int, float]]:
    scores: Dict[int, float] = {}
    for rank, (idx, _) in enumerate(list_a, 1):
        scores[idx] = scores.get(idx, 0) + w_a / (k + rank)
    for rank, (idx, _) in enumerate(list_b, 1):
        scores[idx] = scores.get(idx, 0) + w_b / (k + rank)
    merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return merged[:top_k]


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
    print("  Embedding 优化实验", flush=True)
    print("=" * 70, flush=True)

    docs = load_documents()
    kg = load_kg()
    kg_terms = []
    for e in kg:
        kg_terms.append(e.term)
        kg_terms.extend(e.aliases)
    bm25 = BM25(docs, extra_terms=kg_terms)
    print(f"文档: {len(docs)}, 图谱: {len(kg)}, BM25 avg_len: {bm25.avg_len:.0f}", flush=True)

    print("\n[Phase 1] 计算文档向量...", flush=True)
    vecs_text = get_doc_embeddings(docs, mode="text_only")
    print("\n[Phase 2] 计算 title+text 向量...", flush=True)
    vecs_title = get_doc_embeddings(docs, mode="title_text")

    def bm25_fn(q, kw, kg_):
        return bm25.search(q, top_k=top_k)

    def bm25_kg_fn(q, kw, kg_):
        expanded, _ = expand_query_with_kg(q, kg_)
        return bm25.search(expanded, top_k=top_k)

    def vec_text_fn(q, kw, kg_):
        qv = embed_query(q)
        return vector_search(qv, vecs_text, top_k=top_k)

    def vec_title_fn(q, kw, kg_):
        qv = embed_query(q)
        return vector_search(qv, vecs_title, top_k=top_k)

    def vec_instr_fn(q, kw, kg_):
        qv = embed_query(q, instruction=QUERY_INSTRUCTION)
        return vector_search(qv, vecs_title, top_k=top_k)

    def vec_kg_text_fn(q, kw, kg_):
        expanded, _ = expand_query_with_kg(q, kg_)
        qv = embed_query(expanded)
        return vector_search(qv, vecs_text, top_k=top_k)

    def vec_kg_add_fn(q, kw, kg_):
        qv = embed_query(q)
        expanded, _ = expand_query_with_kg(q, kg_)
        if expanded != q:
            ev = embed_query(expanded)
            qv = add_vectors(qv, ev, weight_a=1.0, weight_b=0.5)
        return vector_search(qv, vecs_title, top_k=top_k)

    def hybrid_equal_fn(q, kw, kg_):
        qv = embed_query(q)
        vr = vector_search(qv, vecs_title, top_k=top_k * 2)
        br = bm25.search(q, top_k=top_k * 2)
        return rrf_fusion(vr, br, w_a=1.0, w_b=1.0, top_k=top_k)

    def hybrid_vec_heavy_fn(q, kw, kg_):
        qv = embed_query(q)
        vr = vector_search(qv, vecs_title, top_k=top_k * 2)
        br = bm25.search(q, top_k=top_k * 2)
        return rrf_fusion(vr, br, w_a=1.5, w_b=1.0, top_k=top_k)

    def hybrid_bm25_heavy_fn(q, kw, kg_):
        qv = embed_query(q)
        vr = vector_search(qv, vecs_title, top_k=top_k * 2)
        br = bm25.search(q, top_k=top_k * 2)
        return rrf_fusion(vr, br, w_a=1.0, w_b=1.5, top_k=top_k)

    def hybrid_kg_fn(q, kw, kg_):
        qv = embed_query(q)
        expanded, _ = expand_query_with_kg(q, kg_)
        vr = vector_search(qv, vecs_title, top_k=top_k * 2)
        br = bm25.search(expanded, top_k=top_k * 2)
        return rrf_fusion(vr, br, w_a=1.5, w_b=1.0, top_k=top_k)

    def hybrid_kg_vec_fn(q, kw, kg_):
        expanded, _ = expand_query_with_kg(q, kg_)
        if expanded != q:
            ev = embed_query(expanded)
            qv = embed_query(q)
            qv = add_vectors(qv, ev, weight_a=1.0, weight_b=0.5)
        else:
            qv = embed_query(q)
        vr = vector_search(qv, vecs_title, top_k=top_k * 2)
        br = bm25.search(expanded, top_k=top_k * 2)
        return rrf_fusion(vr, br, w_a=1.5, w_b=1.0, top_k=top_k)

    experiments = [
        ("BM25 (baseline)", bm25_fn),
        ("BM25+KG (baseline)", bm25_kg_fn),
        ("Vec(text)", vec_text_fn),
        ("Vec(title+text)", vec_title_fn),
        ("Vec(title+text)+instr", vec_instr_fn),
        ("Vec+KG(text拼接)", vec_kg_text_fn),
        ("Vec+KG(向量融合)", vec_kg_add_fn),
        ("Hybrid(equal)", hybrid_equal_fn),
        ("Hybrid(vec-heavy)", hybrid_vec_heavy_fn),
        ("Hybrid(bm25-heavy)", hybrid_bm25_heavy_fn),
        ("Hybrid+KG(bm25)", hybrid_kg_fn),
        ("Hybrid+KG(双融合)", hybrid_kg_vec_fn),
    ]

    print(f"\n[Phase 3] 运行 {len(experiments)} 组实验 x {len(TEST_QUERIES)} 条查询...", flush=True)
    results = []
    for name, fn in experiments:
        print(f"\n  >> {name}", flush=True)
        r = evaluate(name, fn, docs, kg)
        results.append(r)
        print(f"     MRR={r['mrr']:.3f}  Hit@3={r['hit3']:.3f}  Hit@5={r['hit5']:.3f}", flush=True)

    results.sort(key=lambda x: x["mrr"], reverse=True)
    headers = ["Method", "MRR", "Hit@3", "Hit@5"]
    rows = [[r["name"], f"{r['mrr']:.3f}", f"{r['hit3']:.3f}", f"{r['hit5']:.3f}"] for r in results]
    print_table(headers, rows, title="优化实验结果（按MRR排序）")

    best = results[0]
    print(f"\n最优方案: {best['name']} (MRR={best['mrr']:.3f})", flush=True)

    out_path = Path(__file__).resolve().parent / "optimize_results.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"results": [{k: v for k, v in r.items() if k != "per_query_mrr"} for r in results],
                    "per_query": {r["name"]: r["per_query_mrr"] for r in results}}, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {out_path}", flush=True)


if __name__ == "__main__":
    main()
