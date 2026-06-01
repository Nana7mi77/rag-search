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
        print(f"加载模型...", flush=True)
        _model = SentenceTransformer(MODEL_NAME, device=DEVICE)
        _model.encode(["warmup"], batch_size=1, show_progress_bar=False)
        print("模型就绪", flush=True)
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


TEST_QUERIES = [
    {"query": "浮力在潜水艇中的应用", "keywords": ["潜水艇", "浮力", "密度"]},
    {"query": "闪电是怎么形成的", "keywords": ["闪电", "放电", "云层", "雷"]},
    {"query": "人工光源的发展历史", "keywords": ["灯", "LED", "电灯", "人造光"]},
]


def is_relevant(doc, keywords):
    combined = f"{doc.title} {doc.text}".lower()
    return any(kw.lower() in combined for kw in keywords)


def main():
    docs = load_documents()
    kg = load_kg()
    kg_terms = []
    for e in kg:
        kg_terms.append(e.term)
        kg_terms.extend(e.aliases)
    bm25 = BM25(docs, extra_terms=kg_terms)
    vecs = None
    cache_path = CACHE_DIR / "doc_embeddings_text_only.json"
    with cache_path.open() as f:
        vecs = json.load(f)

    cand = 50

    for tq in TEST_QUERIES:
        q = tq["query"]
        kw = tq["keywords"]
        print(f"\n{'='*60}")
        print(f"Query: {q}")
        print(f"Keywords: {kw}")
        print(f"{'='*60}")

        bm25_results = bm25.search(q, top_k=cand)
        vec_results = vector_search(embed_query(q), vecs, top_k=cand)

        _, bt, entries = match_kg(q, kg)
        expanded_parts = []
        for e in entries:
            expanded_parts.append(e.term)
            expanded_parts.extend(e.aliases[:2])
            expanded_parts.extend(e.boost_terms[:3])
        eq = f"{q} {' '.join(dict.fromkeys(expanded_parts))}" if expanded_parts else q
        vec_exp_results = vector_search(embed_query(eq), vecs, top_k=cand)

        wsf_result = wsf_fusion(vec_results, bm25_results, w_a=0.5, w_b=0.5, top_n=10)

        print(f"\nBM25 Top 5:")
        for rank, (idx, s) in enumerate(bm25_results[:5], 1):
            doc = docs[idx]
            rel = "REL" if is_relevant(doc, kw) else "   "
            print(f"  #{rank} [{s:8.3f}] {rel} [{idx}] {doc.title[:30]} | {doc.text[:50]}")

        print(f"\nVec Top 5:")
        for rank, (idx, s) in enumerate(vec_results[:5], 1):
            doc = docs[idx]
            rel = "REL" if is_relevant(doc, kw) else "   "
            print(f"  #{rank} [{s:8.4f}] {rel} [{idx}] {doc.title[:30]} | {doc.text[:50]}")

        print(f"\nVec+KG_exp Top 5:")
        for rank, (idx, s) in enumerate(vec_exp_results[:5], 1):
            doc = docs[idx]
            rel = "REL" if is_relevant(doc, kw) else "   "
            print(f"  #{rank} [{s:8.4f}] {rel} [{idx}] {doc.title[:30]} | {doc.text[:50]}")

        print(f"\nWSF(v=0.5) Top 5:")
        for rank, (idx, s) in enumerate(wsf_result[:5], 1):
            doc = docs[idx]
            rel = "REL" if is_relevant(doc, kw) else "   "
            print(f"  #{rank} [{s:8.4f}] {rel} [{idx}] {doc.title[:30]} | {doc.text[:50]}")


if __name__ == "__main__":
    main()
