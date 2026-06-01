import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .bilibili_map import build_bilibili_url, get_bvid, parse_doc_name, parse_start_seconds
from .text import best_snippet, normalize_text, tokenize


TEXT_COLUMNS = ("subtitle", "script", "字幕", "text", "content", "内容")
TITLE_COLUMNS = ("name", "片名", "title", "file")
TIME_COLUMNS = ("time", "时间", "timestamp")


@dataclass
class Document:
    doc_id: int
    title: str
    time: str
    text: str
    meta: Dict[str, str]


@dataclass
class SearchHit:
    doc_id: int
    score: float
    title: str
    time: str
    text: str
    snippet: str
    doc_name: str = ""
    start_seconds: float = 0.0
    bilibili_url: str = ""
    thumbnail_placeholder: bool = True


def _pick_column(row: Dict[str, str], candidates: Iterable[str], explicit: Optional[str] = None) -> Optional[str]:
    if explicit:
        return explicit
    normalized = {key.lower(): key for key in row.keys()}
    for candidate in candidates:
        if candidate in row:
            return candidate
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]
    return None


def load_documents_csv(
    path: str,
    *,
    limit: Optional[int] = None,
    text_col: Optional[str] = None,
    title_col: Optional[str] = None,
    time_col: Optional[str] = None,
) -> List[Document]:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    documents: List[Document] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not row:
                continue
            text_key = _pick_column(row, TEXT_COLUMNS, text_col)
            title_key = _pick_column(row, TITLE_COLUMNS, title_col)
            time_key = _pick_column(row, TIME_COLUMNS, time_col)
            if not text_key:
                raise ValueError(
                    f"Cannot detect text column in {csv_path}. Expected one of: {', '.join(TEXT_COLUMNS)}"
                )
            text = normalize_text(row.get(text_key))
            if not text:
                continue
            doc_id = len(documents)
            documents.append(
                Document(
                    doc_id=doc_id,
                    title=normalize_text(row.get(title_key)) if title_key else "",
                    time=normalize_text(row.get(time_key)) if time_key else "",
                    text=text,
                    meta={str(k): normalize_text(v) for k, v in row.items() if k is not None},
                )
            )
            if limit and len(documents) >= limit:
                break
    return documents


class BM25Index:
    def __init__(self, documents: List[Document], extra_terms: Iterable[str] = ()):
        self.documents = documents
        self.extra_terms = sorted(set(extra_terms), key=len, reverse=True)
        self.doc_tokens: List[List[str]] = []
        self.doc_term_freqs: List[Dict[str, int]] = []
        self.doc_freqs: Dict[str, int] = {}
        self.avg_doc_len = 0.0
        if documents:
            self._build()

    def _build(self) -> None:
        df_counter: Dict[str, int] = defaultdict(int)
        total_len = 0
        for document in self.documents:
            tokens = tokenize(f"{document.title} {document.text}", self.extra_terms)
            counts = Counter(tokens)
            self.doc_tokens.append(tokens)
            self.doc_term_freqs.append(dict(counts))
            total_len += len(tokens)
            for token in counts:
                df_counter[token] += 1
        self.doc_freqs = dict(df_counter)
        self.avg_doc_len = total_len / max(len(self.documents), 1)

    def idf(self, term: str) -> float:
        n_docs = len(self.documents)
        df = self.doc_freqs.get(term, 0)
        return math.log(1 + (n_docs - df + 0.5) / (df + 0.5))

    def search(self, query: object, *, top_k: int = 10, terms: Iterable[str] = ()) -> List[SearchHit]:
        query_text = normalize_text(query)
        query_terms = tokenize(query_text, list(self.extra_terms) + list(terms))
        if not query_terms:
            return []

        query_counts = Counter(query_terms)
        k1 = 1.5
        b = 0.75
        scored: List[SearchHit] = []
        for idx, document in enumerate(self.documents):
            doc_len = len(self.doc_tokens[idx]) or 1
            freqs = self.doc_term_freqs[idx]
            score = 0.0
            for term, query_weight in query_counts.items():
                freq = freqs.get(term, 0)
                if not freq:
                    continue
                denom = freq + k1 * (1 - b + b * doc_len / max(self.avg_doc_len, 1))
                score += self.idf(term) * (freq * (k1 + 1) / denom) * min(query_weight, 2)
            if score > 0:
                doc_name = parse_doc_name(document.title)
                start_seconds = parse_start_seconds(document.time)
                scored.append(
                    SearchHit(
                        doc_id=document.doc_id,
                        score=score,
                        title=document.title,
                        time=document.time,
                        text=document.text,
                        snippet=best_snippet(document.text, query_text, terms),
                        doc_name=doc_name,
                        start_seconds=start_seconds,
                        bilibili_url=build_bilibili_url(get_bvid(doc_name), start_seconds),
                    )
                )

        scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored[:top_k]

    def search_raw(self, query: object, *, top_k: int = 10, terms: Iterable[str] = ()) -> List[Tuple[int, float]]:
        query_text = normalize_text(query)
        query_terms = tokenize(query_text, list(self.extra_terms) + list(terms))
        if not query_terms:
            return []

        query_counts = Counter(query_terms)
        k1 = 1.5
        b = 0.75
        scored: List[Tuple[int, float]] = []
        for idx, document in enumerate(self.documents):
            doc_len = len(self.doc_tokens[idx]) or 1
            freqs = self.doc_term_freqs[idx]
            score = 0.0
            for term, query_weight in query_counts.items():
                freq = freqs.get(term, 0)
                if not freq:
                    continue
                denom = freq + k1 * (1 - b + b * doc_len / max(self.avg_doc_len, 1))
                score += self.idf(term) * (freq * (k1 + 1) / denom) * min(query_weight, 2)
            if score > 0:
                scored.append((document.doc_id, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def save(self, path: str) -> None:
        payload = {
            "documents": [asdict(document) for document in self.documents],
            "extra_terms": self.extra_terms,
            "doc_tokens": self.doc_tokens,
            "doc_term_freqs": self.doc_term_freqs,
            "doc_freqs": self.doc_freqs,
            "avg_doc_len": self.avg_doc_len,
        }
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "BM25Index":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        index = cls(
            [Document(**document) for document in payload["documents"]],
            payload.get("extra_terms", ()),
        )
        index.doc_tokens = payload["doc_tokens"]
        index.doc_term_freqs = payload["doc_term_freqs"]
        index.doc_freqs = payload["doc_freqs"]
        index.avg_doc_len = payload["avg_doc_len"]
        return index
