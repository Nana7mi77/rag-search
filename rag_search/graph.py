import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .text import normalize_text


@dataclass
class GraphExpansion:
    original_query: str
    expanded_query: str
    matched_terms: List[str]
    expansions: List[str]


class LocalKnowledgeGraph:
    def __init__(self, entries: Iterable[Dict[str, str]] = ()):
        self.entries: List[Dict[str, str]] = []
        for entry in entries:
            term = normalize_text(entry.get("term") or entry.get("name"))
            expansion = normalize_text(entry.get("expansion") or entry.get("content") or entry.get("内容"))
            aliases = normalize_text(entry.get("aliases") or "")
            relation = normalize_text(entry.get("relation") or entry.get("type") or "")
            if term and expansion:
                self.entries.append(
                    {
                        "term": term,
                        "aliases": aliases,
                        "relation": relation,
                        "expansion": expansion,
                    }
                )

    @classmethod
    def load(cls, path: Optional[str]) -> "LocalKnowledgeGraph":
        if not path:
            return cls()
        graph_path = Path(path)
        if not graph_path.exists():
            return cls()
        with graph_path.open("r", encoding="utf-8-sig", newline="") as handle:
            return cls(csv.DictReader(handle))

    def terms(self) -> List[str]:
        values = []
        for entry in self.entries:
            values.append(entry["term"])
            values.extend(part.strip() for part in entry["aliases"].split("|") if part.strip())
        return sorted(set(values), key=len, reverse=True)

    def expand(self, query: object, max_entries: int = 4) -> GraphExpansion:
        query_text = normalize_text(query)
        matched_terms: List[str] = []
        expansions: List[str] = []

        for entry in self.entries:
            candidates = [entry["term"]]
            candidates.extend(part.strip() for part in entry["aliases"].split("|") if part.strip())
            if any(candidate and candidate in query_text for candidate in candidates):
                matched_terms.append(entry["term"])
                expansions.append(entry["expansion"])
            if len(expansions) >= max_entries:
                break

        expansion_text = " ".join(expansions)
        expanded_query = normalize_text(f"{query_text} {expansion_text}")
        return GraphExpansion(
            original_query=query_text,
            expanded_query=expanded_query or query_text,
            matched_terms=matched_terms,
            expansions=expansions,
        )
