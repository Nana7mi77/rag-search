import json
import math
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from .index import Document


HF_MODEL = "intfloat/multilingual-e5-large"
HF_API_URL = "https://router.huggingface.co/hf-inference/models/{model_id}"
DEFAULT_PROXY = "http://127.0.0.1:7890"
VECTOR_DIM = 1024
DOC_TEXT_LIMIT = 768


@dataclass
class VectorSearchResult:
    doc_id: int
    score: float


def _format_passage(doc: Document) -> str:
    title = doc.title.strip()
    text = doc.text[:DOC_TEXT_LIMIT].strip()
    if title:
        return f"passage: {title}。{text}"
    return f"passage: {text}"


def _format_query(query: str) -> str:
    return f"query: {query.strip()}"


class VectorIndex:
    def __init__(self, documents: List[Document], embeddings: Optional[List[List[float]]] = None):
        self.documents = documents
        self.embeddings: List[List[float]] = embeddings or []
        self._api_key: str = ""
        self._api_url: str = ""
        self._proxy: Optional[str] = None

    def configure(self, api_key: str, *, model_id: str = HF_MODEL, proxy: Optional[str] = DEFAULT_PROXY):
        self._api_key = api_key
        self._api_url = HF_API_URL.format(model_id=model_id)
        self._proxy = proxy

    @classmethod
    def load(cls, cache_path: str, documents: List[Document]) -> "VectorIndex":
        p = Path(cache_path)
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                embeddings = data
            else:
                embeddings = data.get("embeddings", [])
            if len(embeddings) == len(documents):
                idx = cls(documents, embeddings)
                return idx
        return cls(documents)

    def save(self, cache_path: str):
        p = Path(cache_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump({"dim": VECTOR_DIM, "count": len(self.embeddings), "embeddings": self.embeddings}, f)

    def _api_call(self, text: str, retries: int = 5) -> List[float]:
        payload = json.dumps({"inputs": text}).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        req = urllib.request.Request(self._api_url, data=payload, headers=headers)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler(
            {"http": self._proxy, "https": self._proxy} if self._proxy else {}
        ))
        last_err = None
        for attempt in range(retries):
            try:
                resp = opener.open(req, timeout=30)
                raw = resp.read()
                result = json.loads(raw)
                if isinstance(result, list) and len(result) > 0:
                    if isinstance(result[0], list):
                        return result[0]
                    return result
                raise ValueError(f"Unexpected response format: {type(result)}")
            except urllib.error.HTTPError as e:
                last_err = e
                if e.code == 429:
                    import time
                    time.sleep(min(5.0 * (attempt + 1), 30))
                else:
                    import time
                    time.sleep(2.0 * (attempt + 1))
            except Exception as e:
                last_err = e
                import time
                time.sleep(2.0 * (attempt + 1))
        raise last_err

    def encode_single(self, text: str) -> List[float]:
        return self._api_call(text)

    def _encode_batch_concurrent(
        self,
        texts: List[str],
        *,
        max_workers: int = 4,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> List[List[float]]:
        embeddings: List[Optional[List[float]]] = [None] * len(texts)
        done_count = 0
        total = len(texts)

        def _encode_one(idx: int) -> Tuple[int, List[float]]:
            return idx, self._api_call(texts[idx])

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_encode_one, i): i for i in range(total)}
            for future in as_completed(futures):
                try:
                    idx, emb = future.result()
                    embeddings[idx] = emb
                except Exception:
                    idx = futures[future]
                    embeddings[idx] = [0.0] * VECTOR_DIM
                done_count += 1
                if on_progress and done_count % 10 == 0:
                    on_progress(done_count, total)

        if on_progress:
            on_progress(total, total)
        return [e if e is not None else [0.0] * VECTOR_DIM for e in embeddings]

    def _encode_batch_serial(self, texts: List[str]) -> List[List[float]]:
        all_embeddings = []
        for text in texts:
            try:
                emb = self._api_call(text)
                all_embeddings.append(emb)
            except Exception:
                all_embeddings.append([0.0] * VECTOR_DIM)
        return all_embeddings

    def build(
        self,
        *,
        batch_size: int = 8,
        show_progress: bool = True,
        max_workers: int = 1,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ):
        if len(self.embeddings) == len(self.documents):
            return
        texts = [_format_passage(doc) for doc in self.documents]
        if max_workers > 1:
            self.embeddings = self._encode_batch_concurrent(
                texts, max_workers=max_workers, on_progress=on_progress
            )
        else:
            self.embeddings = self._encode_batch_serial(texts)

    def search(self, query: str, *, top_k: int = 8) -> List[VectorSearchResult]:
        raw = self.search_raw(query, top_k=top_k)
        return [VectorSearchResult(doc_id=did, score=s) for did, s in raw]

    def search_raw(self, query: str, *, top_k: int = 8) -> List[Tuple[int, float]]:
        if not self.embeddings:
            return []
        query_emb = self.encode_single(_format_query(query))
        scores = []
        for doc_id, doc_emb in enumerate(self.embeddings):
            score = _cosine_similarity(query_emb, doc_emb)
            scores.append((doc_id, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def build_incremental(
        self,
        cache_path: str,
        *,
        max_workers: int = 4,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ):
        progress_path = Path(cache_path).with_suffix(".progress.json")
        partial: Dict[int, List[float]] = {}
        if progress_path.exists():
            try:
                with progress_path.open("r", encoding="utf-8") as f:
                    raw = json.load(f)
                if isinstance(raw, dict):
                    partial = {int(k): v for k, v in raw.items()}
            except Exception:
                pass

        to_encode = []
        for i in range(len(self.documents)):
            if i not in partial:
                to_encode.append(i)

        if not to_encode:
            self.embeddings = [partial[i] for i in range(len(self.documents))]
            self.save(cache_path)
            return

        def _encode_one(idx: int) -> Tuple[int, List[float]]:
            return idx, self._api_call(_format_passage(self.documents[idx]))

        done = len(partial)
        total = len(self.documents)
        if on_progress:
            on_progress(done, total)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_encode_one, i): i for i in to_encode}
            for future in as_completed(futures):
                try:
                    idx, emb = future.result()
                    partial[idx] = emb
                except Exception:
                    idx = futures[future]
                    partial[idx] = [0.0] * VECTOR_DIM
                done += 1
                if on_progress and done % 5 == 0:
                    on_progress(done, total)
                if done % 50 == 0:
                    with progress_path.open("w", encoding="utf-8") as f:
                        json.dump({str(k): v for k, v in partial.items()}, f)

        with progress_path.open("w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in partial.items()}, f)

        self.embeddings = [partial[i] for i in range(len(self.documents))]
        self.save(cache_path)


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def normalize_score_list(hits: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
    if not hits:
        return []
    scores = [s for _, s in hits]
    min_s = min(scores)
    max_s = max(scores)
    span = max_s - min_s
    if span < 1e-9:
        return [(did, 1.0) for did, _ in hits]
    return [(did, (s - min_s) / span) for did, s in hits]


def score_fusion(
    bm25_hits: List[Tuple[int, float]],
    vector_hits: List[Tuple[int, float]],
    *,
    bm25_weight: float = 0.3,
    vector_weight: float = 0.7,
) -> List[Tuple[int, float]]:
    bm25_norm = normalize_score_list(bm25_hits)
    vector_norm = normalize_score_list(vector_hits)

    scores: Dict[int, float] = {}
    for doc_id, score in bm25_norm:
        scores[doc_id] = scores.get(doc_id, 0.0) + bm25_weight * score
    for doc_id, score in vector_norm:
        scores[doc_id] = scores.get(doc_id, 0.0) + vector_weight * score
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def rrf_fusion(
    bm25_hits: List[Tuple[int, float]],
    vector_hits: List[Tuple[int, float]],
    *,
    k: int = 60,
    bm25_weight: float = 1.0,
    vector_weight: float = 1.0,
) -> List[Tuple[int, float]]:
    scores: Dict[int, float] = {}
    for rank, (doc_id, _) in enumerate(bm25_hits):
        scores[doc_id] = scores.get(doc_id, 0.0) + bm25_weight / (k + rank + 1)
    for rank, (doc_id, _) in enumerate(vector_hits):
        scores[doc_id] = scores.get(doc_id, 0.0) + vector_weight / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
