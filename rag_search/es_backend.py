import base64
import json
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional
from urllib import error, request

from .graph import GraphExpansion
from .index import SearchHit, load_documents_csv
from .text import best_snippet, normalize_text


@dataclass
class ServiceConfig:
    es_url: str = "http://127.0.0.1:9200"
    es_index: str = "subtitle_segments"
    neo4j_url: str = "http://127.0.0.1:7474"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "ragsearch123"


class HttpJsonClient:
    def __init__(self, base_url: str, username: str = "", password: str = ""):
        self.base_url = base_url.rstrip("/")
        self.username = username
        self.password = password

    def request(self, method: str, path: str, payload: Optional[object] = None) -> object:
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if self.username or self.password:
            token = base64.b64encode(f"{self.username}:{self.password}".encode("utf-8")).decode("ascii")
            headers["Authorization"] = f"Basic {token}"
        req = request.Request(self.base_url + path, data=body, method=method.upper(), headers=headers)
        try:
            with request.urlopen(req, timeout=30) as resp:
                data = resp.read()
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{method} {path} failed: HTTP {exc.code} {detail}") from exc
        if not data:
            return {}
        return json.loads(data.decode("utf-8"))


class ElasticsearchBackend:
    def __init__(self, config: ServiceConfig):
        self.config = config
        self.client = HttpJsonClient(config.es_url)

    def wait_ready(self, timeout_seconds: int = 90) -> None:
        deadline = time.time() + timeout_seconds
        last_error: Optional[Exception] = None
        while time.time() < deadline:
            try:
                self.client.request("GET", "/")
                return
            except Exception as exc:
                last_error = exc
                time.sleep(2)
        raise RuntimeError(f"Elasticsearch is not ready: {last_error}")

    def create_index(self, *, recreate: bool = True) -> None:
        index = self.config.es_index
        if recreate:
            try:
                self.client.request("DELETE", f"/{index}")
            except RuntimeError:
                pass
        settings = {
            "settings": {
                "analysis": {
                    "tokenizer": {
                        "zh_ngram_tokenizer": {
                            "type": "ngram",
                            "min_gram": 2,
                            "max_gram": 3,
                        }
                    },
                    "analyzer": {
                        "zh_ngram": {
                            "type": "custom",
                            "tokenizer": "zh_ngram_tokenizer",
                            "filter": ["lowercase"],
                        }
                    },
                }
            },
            "mappings": {
                "properties": {
                    "doc_id": {"type": "integer"},
                    "title": {"type": "text", "analyzer": "zh_ngram", "fields": {"raw": {"type": "keyword"}}},
                    "time": {"type": "keyword"},
                    "text": {"type": "text", "analyzer": "zh_ngram"},
                }
            },
        }
        self.client.request("PUT", f"/{index}", settings)

    def bulk_import_csv(self, data_path: str, *, limit: Optional[int] = None, batch_size: int = 500) -> int:
        documents = load_documents_csv(data_path, limit=limit)
        count = 0
        batch: List[str] = []
        for doc in documents:
            batch.append(json.dumps({"index": {"_index": self.config.es_index, "_id": doc.doc_id}}, ensure_ascii=False))
            batch.append(
                json.dumps(
                    {
                        "doc_id": doc.doc_id,
                        "title": doc.title,
                        "time": doc.time,
                        "text": doc.text,
                    },
                    ensure_ascii=False,
                )
            )
            count += 1
            if count % batch_size == 0:
                self._send_bulk(batch)
                batch = []
        if batch:
            self._send_bulk(batch)
        self.client.request("POST", f"/{self.config.es_index}/_refresh")
        return count

    def _send_bulk(self, lines: List[str]) -> None:
        data = ("\n".join(lines) + "\n").encode("utf-8")
        req = request.Request(
            self.config.es_url.rstrip("/") + "/_bulk",
            data=data,
            method="POST",
            headers={"Content-Type": "application/x-ndjson", "Accept": "application/json"},
        )
        with request.urlopen(req, timeout=60) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        if payload.get("errors"):
            raise RuntimeError(f"Elasticsearch bulk import had errors: {payload}")

    def search(self, query_text: str, *, top_k: int = 8) -> List[SearchHit]:
        payload = {
            "size": top_k,
            "query": {
                "multi_match": {
                    "query": query_text,
                    "fields": ["title^2", "text"],
                }
            },
            "highlight": {
                "fields": {
                    "text": {"fragment_size": 160, "number_of_fragments": 1},
                }
            },
        }
        raw = self.client.request("POST", f"/{self.config.es_index}/_search", payload)
        hits: List[SearchHit] = []
        for item in raw.get("hits", {}).get("hits", []):
            source = item.get("_source", {})
            text = normalize_text(source.get("text"))
            snippet = best_snippet(text, query_text)
            highlights = item.get("highlight", {}).get("text") or []
            if highlights:
                snippet = normalize_text(highlights[0].replace("<em>", "").replace("</em>", ""))
            hits.append(
                SearchHit(
                    doc_id=int(source.get("doc_id", item.get("_id", 0))),
                    score=float(item.get("_score", 0.0)),
                    title=normalize_text(source.get("title")),
                    time=normalize_text(source.get("time")),
                    text=text,
                    snippet=snippet,
                )
            )
        return hits


class Neo4jExpansionBackend:
    def __init__(self, config: ServiceConfig):
        self.config = config
        self.client = HttpJsonClient(config.neo4j_url, config.neo4j_user, config.neo4j_password)

    def expand(self, query_text: str, *, limit: int = 4) -> GraphExpansion:
        query_text = normalize_text(query_text)
        cypher = """
        MATCH (term:Term)
        WHERE $query CONTAINS term.name
           OR any(alias IN coalesce(term.aliases, []) WHERE alias <> '' AND $query CONTAINS alias)
        RETURN term.name AS name, term.expansion AS expansion
        LIMIT $limit
        """
        payload = {
            "statements": [
                {
                    "statement": cypher,
                    "parameters": {"query": query_text, "limit": limit},
                }
            ]
        }
        try:
            raw = self.client.request("POST", "/db/neo4j/tx/commit", payload)
        except Exception:
            return GraphExpansion(query_text, query_text, [], [])

        rows = raw.get("results", [{}])[0].get("data", [])
        matched = []
        expansions = []
        for row in rows:
            values = row.get("row", [])
            if len(values) >= 2:
                matched.append(normalize_text(values[0]))
                expansions.append(normalize_text(values[1]))
        expanded = normalize_text(f"{query_text} {' '.join(expansions)}")
        return GraphExpansion(query_text, expanded or query_text, matched, expansions)
