"""
向量检索 vs BM25 对比实验（本地 bge-m3）
使用 sentence-transformers 在本地运行 bge-m3，无需 API
"""
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
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("HF_HUB_OFFLINE", "1")

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = Path(__file__).resolve().parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
TOKEN_RE = re.compile(r"[\u4e00-\u9fff]+|[a-zA-Z0-9_]+")

MODEL_NAME = "BAAI/bge-m3"
DEVICE = "mps"


# ───────────────────────── 文本处理 ─────────────────────────

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


# ───────────────────────── 数据加载 ─────────────────────────

@dataclass
class Doc:
    doc_id: int
    title: str
    time: str
    text: str


def load_documents(limit: int = None) -> List[Doc]:
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
            if limit and len(docs) >= limit:
                break
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


# ───────────────────────── BM25 检索 ─────────────────────────

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


# ───────────────────────── 本地 Embedding ─────────────────────────

_model = None


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        print(f"加载模型 {MODEL_NAME} 到 {DEVICE}...")
        _model = SentenceTransformer(MODEL_NAME, device=DEVICE)
        print(f"模型加载完成，向量维度: {_model.get_sentence_embedding_dimension()}")
        print("MPS warmup...")
        _model.encode(["warmup"], batch_size=1, show_progress_bar=False)
        print("warmup 完成")
    return _model


def embed_batch(texts: List[str], batch_size: int = 32) -> List[List[float]]:
    model = get_model()
    vecs = model.encode(texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)
    return vecs.tolist()


def embed_query(query: str) -> List[float]:
    return embed_batch([query])[0]


def get_doc_embeddings(docs: List[Doc], force: bool = False) -> List[List[float]]:
    cache_path = CACHE_DIR / "doc_embeddings.json"
    if cache_path.exists() and not force:
        print("加载缓存的文档向量...")
        with cache_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    print(f"计算文档向量 ({MODEL_NAME} 本地推理)...")
    texts = [doc.text[:200] for doc in docs]
    t0 = time.time()
    vecs = []
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        if i % 512 == 0:
            print(f"  doc: {i + 1}-{min(i + batch_size, len(texts))}/{len(texts)}", flush=True)
        vecs.extend(embed_batch(batch, batch_size=batch_size))
    elapsed = time.time() - t0
    print(f"文档向量计算完成，耗时 {elapsed:.1f}s，已缓存到 {cache_path}")
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(vecs, f)
    return vecs


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


# ───────────────────────── 评测数据集 ─────────────────────────

TEST_QUERIES = [
    {
        "query": "浮力在潜水艇中的应用",
        "relevant_keywords": ["潜水艇", "浮力", "密度"],
        "description": "直接知识查询",
    },
    {
        "query": "光的本质是什么",
        "relevant_keywords": ["光", "本质", "波", "粒子"],
        "description": "科学概念查询",
    },
    {
        "query": "牛顿对光学的贡献",
        "relevant_keywords": ["牛顿", "光", "棱镜", "光学"],
        "description": "人物+学科查询",
    },
    {
        "query": "自然光源有哪些",
        "relevant_keywords": ["太阳", "自然光", "闪电", "恒星"],
        "description": "列举型查询",
    },
    {
        "query": "激光的特点和应用",
        "relevant_keywords": ["激光", "方向性", "能量", "照明"],
        "description": "概念+应用查询",
    },
    {
        "query": "欧几里德在光学方面做了什么",
        "relevant_keywords": ["欧几里德", "光学", "视觉"],
        "description": "历史人物查询",
    },
    {
        "query": "眼睛如何看见物体",
        "relevant_keywords": ["眼睛", "光线", "视觉", "视网膜"],
        "description": "科学原理查询",
    },
    {
        "query": "光速是多少",
        "relevant_keywords": ["光速", "速度", "米", "秒"],
        "description": "数值型查询",
    },
    {
        "query": "密度计的工作原理",
        "relevant_keywords": ["密度计", "浮力", "密度", "液体"],
        "description": "原理查询（图谱增强场景）",
    },
    {
        "query": "闪电是怎么形成的",
        "relevant_keywords": ["闪电", "放电", "云层", "雷"],
        "description": "自然现象查询",
    },
    {
        "query": "郑和宝船与航海技术",
        "relevant_keywords": ["郑和", "宝船", "航海"],
        "description": "历史事件查询",
    },
    {
        "query": "反射和折射的区别",
        "relevant_keywords": ["反射", "折射", "光线", "介质"],
        "description": "对比型查询",
    },
    {
        "query": "人工光源的发展历史",
        "relevant_keywords": ["灯", "LED", "电灯", "人造光"],
        "description": "历史查询",
    },
    {
        "query": "潜水艇如何上浮和下潜",
        "relevant_keywords": ["潜水艇", "压载水舱", "浮力", "密度"],
        "description": "操作原理查询",
    },
    {
        "query": "光的颜色是怎么产生的",
        "relevant_keywords": ["颜色", "光谱", "波长", "棱镜"],
        "description": "原理解释查询",
    },
]


# ───────────────────────── 评估指标 ─────────────────────────

def is_relevant(doc: Doc, keywords: List[str]) -> bool:
    combined = f"{doc.title} {doc.text}".lower()
    return any(kw.lower() in combined for kw in keywords)


def mrr(results: List[Tuple[int, float]], docs: List[Doc], keywords: List[str]) -> float:
    for rank, (idx, _) in enumerate(results, 1):
        if is_relevant(docs[idx], keywords):
            return 1.0 / rank
    return 0.0


def hit_at_k(results: List[Tuple[int, float]], docs: List[Doc], keywords: List[str], k: int = 3) -> float:
    for idx, _ in results[:k]:
        if is_relevant(docs[idx], keywords):
            return 1.0
    return 0.0


def avg_score_topk(results: List[Tuple[int, float]], k: int = 3) -> float:
    if not results:
        return 0.0
    return sum(s for _, s in results[:k]) / min(k, len(results))


# ───────────────────────── 实验方法 ─────────────────────────

def run_bm25_baseline(query: str, bm25: BM25, top_k: int) -> List[Tuple[int, float]]:
    return bm25.search(query, top_k=top_k)


def run_bm25_kg(query: str, bm25: BM25, kg: List[KGEntry], top_k: int) -> Tuple[List[Tuple[int, float]], List[str]]:
    expanded, matched = expand_query_with_kg(query, kg)
    return bm25.search(expanded, top_k=top_k, terms=matched), matched


def run_vector(query: str, q_vec: List[float], doc_vecs: List[List[float]], top_k: int) -> List[Tuple[int, float]]:
    return vector_search(q_vec, doc_vecs, top_k=top_k)


def run_vector_kg(query: str, kg: List[KGEntry], doc_vecs: List[List[float]], top_k: int) -> Tuple[List[Tuple[int, float]], List[str]]:
    expanded, matched = expand_query_with_kg(query, kg)
    q_vec = embed_query(expanded)
    return vector_search(q_vec, doc_vecs, top_k=top_k), matched


def run_hybrid(query: str, q_vec: List[float], bm25: BM25, doc_vecs: List[List[float]], kg: List[KGEntry], top_k: int) -> Tuple[List[Tuple[int, float]], List[str]]:
    expanded, matched = expand_query_with_kg(query, kg)
    vec_results = vector_search(q_vec, doc_vecs, top_k=top_k * 2)
    bm25_results = bm25.search(expanded, top_k=top_k * 2, terms=matched)
    rrf_scores: Dict[int, float] = {}
    k = 60
    for rank, (idx, _) in enumerate(vec_results, 1):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (k + rank)
    for rank, (idx, _) in enumerate(bm25_results, 1):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (k + rank)
    merged = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return merged[:top_k], matched


# ───────────────────────── 主流程 ─────────────────────────

def print_table(headers: List[str], rows: List[List[str]], title: str = ""):
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
    print("=" * 70)
    print(f"  字幕检索对比实验：BM25 vs {MODEL_NAME} 本地向量检索")
    print("=" * 70)

    print("\n[1/5] 加载数据...")
    docs = load_documents()
    print(f"  文档数: {len(docs)} (全量)")
    kg = load_kg()
    print(f"  图谱词条: {len(kg)}")

    print("\n[2/5] 构建 BM25 索引...")
    kg_terms = []
    for e in kg:
        kg_terms.append(e.term)
        kg_terms.extend(e.aliases)
    bm25 = BM25(docs, extra_terms=kg_terms)
    print(f"  BM25 索引完成，平均文档长度: {bm25.avg_len:.1f} tokens")

    print(f"\n[3/5] 获取文档向量 ({MODEL_NAME} 本地)...")
    doc_vecs = get_doc_embeddings(docs)
    print(f"  向量维度: {len(doc_vecs[0])}")

    print(f"\n[4/5] 运行 {len(TEST_QUERIES)} 条测试查询...")
    methods = ["BM25", "BM25+KG", "Vector", "Vector+KG", "Hybrid"]
    all_results = {m: {"mrr": [], "hit3": [], "hit5": [], "avg_score": []} for m in methods}

    for i, tq in enumerate(TEST_QUERIES):
        query = tq["query"]
        kws = tq["relevant_keywords"]
        desc = tq["description"]
        print(f"\n  [{i + 1}/{len(TEST_QUERIES)}] {query} ({desc})", flush=True)

        try:
            q_vec = embed_query(query)
        except Exception as e:
            print(f"    Query embedding failed: {type(e).__name__}: {e}", flush=True)
            q_vec = None

        r1 = run_bm25_baseline(query, bm25, top_k)
        r2, m2 = run_bm25_kg(query, bm25, kg, top_k)
        r3 = run_vector(query, q_vec, doc_vecs, top_k) if q_vec else []
        r4, m4 = run_vector_kg(query, kg, doc_vecs, top_k) if q_vec else ([], [])
        r5, m5 = run_hybrid(query, q_vec, bm25, doc_vecs, kg, top_k) if q_vec else ([], [])

        results_map = {
            "BM25": r1,
            "BM25+KG": r2,
            "Vector": r3,
            "Vector+KG": r4,
            "Hybrid": r5,
        }

        for method in methods:
            r = results_map[method]
            all_results[method]["mrr"].append(mrr(r, docs, kws))
            all_results[method]["hit3"].append(hit_at_k(r, docs, kws, 3))
            all_results[method]["hit5"].append(hit_at_k(r, docs, kws, 5))
            all_results[method]["avg_score"].append(avg_score_topk(r, 3))

        best_method = max(methods, key=lambda m: all_results[m]["mrr"][-1])
        print(f"    最优: {best_method} (MRR={all_results[best_method]['mrr'][-1]:.2f})", flush=True)

    print(f"\n[5/5] 汇总结果")
    headers = ["Method", "MRR", "Hit@3", "Hit@5", "AvgScore"]
    rows = []
    for m in methods:
        r = all_results[m]
        rows.append([
            m,
            f"{sum(r['mrr']) / len(r['mrr']):.3f}",
            f"{sum(r['hit3']) / len(r['hit3']):.3f}",
            f"{sum(r['hit5']) / len(r['hit5']):.3f}",
            f"{sum(r['avg_score']) / len(r['avg_score']):.4f}",
        ])
    rows.sort(key=lambda x: float(x[1]), reverse=True)
    print_table(headers, rows, title="整体评估结果")

    print("\n--- 逐查询 MRR 对比 ---")
    q_headers = ["Query"] + methods
    q_rows = []
    for i, tq in enumerate(TEST_QUERIES):
        row = [tq["query"][:20]]
        for m in methods:
            v = all_results[m]["mrr"][i]
            row.append(f"{v:.2f}" if v > 0 else "-")
        q_rows.append(row)
    print_table(q_headers, q_rows)

    result_path = Path(__file__).resolve().parent / "results.json"
    output = {
        "model": MODEL_NAME,
        "device": DEVICE,
        "summary": {m: {k: sum(v) / len(v) for k, v in all_results[m].items()} for m in methods},
        "per_query": [],
    }
    for i, tq in enumerate(TEST_QUERIES):
        entry = {"query": tq["query"], "description": tq["description"]}
        for m in methods:
            entry[m] = {
                "mrr": all_results[m]["mrr"][i],
                "hit3": all_results[m]["hit3"][i],
            }
        output["per_query"].append(entry)
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存到: {result_path}")


if __name__ == "__main__":
    main()
