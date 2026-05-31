import re
from typing import Iterable, List


_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]+|[a-zA-Z0-9_]+")
_SPACE_RE = re.compile(r"\s+")


def normalize_text(text: object) -> str:
    if text is None:
        return ""
    text = str(text)
    text = text.replace("\ufeff", "").replace("\u3000", " ")
    text = _SPACE_RE.sub(" ", text)
    return text.strip()


def _cjk_ngrams(span: str) -> List[str]:
    tokens: List[str] = []
    span = normalize_text(span)
    if not span:
        return tokens

    if len(span) <= 6:
        tokens.append(span)

    for n in (2, 3):
        if len(span) >= n:
            tokens.extend(span[i : i + n] for i in range(len(span) - n + 1))

    if len(span) <= 3:
        tokens.extend(span)

    return tokens


def tokenize(text: object, extra_terms: Iterable[str] = ()) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []

    tokens: List[str] = []
    for term in extra_terms:
        term = normalize_text(term)
        if term and term in text:
            tokens.append(term)

    for match in _TOKEN_RE.finditer(text):
        token = match.group(0)
        if _CJK_RE.fullmatch(token):
            tokens.extend(_cjk_ngrams(token))
        else:
            token = token.lower()
            if len(token) > 1:
                tokens.append(token)

    return tokens


def best_snippet(text: object, query: object, terms: Iterable[str] = (), max_len: int = 180) -> str:
    text = normalize_text(text)
    query_text = normalize_text(query)
    wanted = [normalize_text(t) for t in terms if normalize_text(t)]
    wanted.extend([t for t in _cjk_ngrams(query_text) if len(t) >= 2])

    if not text:
        return ""

    sentences = re.split(r"(?<=[。！？!?；;])|\s{2,}", text)
    sentences = [normalize_text(s) for s in sentences if normalize_text(s)]
    if not sentences:
        sentences = [text]

    def score(sentence: str) -> int:
        return sum(1 for term in wanted if term and term in sentence)

    selected = max(sentences, key=score)
    if score(selected) == 0:
        selected = text

    if len(selected) <= max_len:
        return selected
    return selected[: max_len - 1].rstrip() + "…"
