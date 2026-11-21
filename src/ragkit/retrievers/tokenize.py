"""Tokenizers for BM25. Default is punctuation-aware whitespace; Thai hook included."""

from __future__ import annotations

import re
from collections.abc import Callable

_TOKEN = re.compile(r"[A-Za-z0-9_]+|[\u0E00-\u0E7F]+", re.UNICODE)


def whitespace_tokenize(text: str) -> list[str]:
    """Lowercase Latin words; keep Thai character runs as a single token (suboptimal)."""
    return [t.lower() for t in _TOKEN.findall(text)]


def thai_char_ngram(text: str, n: int = 2) -> list[str]:
    """Fallback Thai tokenizer with no extra deps: typographic tokens + char n-grams on Thai runs."""
    tokens: list[str] = []
    for tok in _TOKEN.findall(text):
        if re.fullmatch(r"[\u0E00-\u0E7F]+", tok) and len(tok) >= n:
            tokens.extend(tok[i : i + n] for i in range(len(tok) - n + 1))
        else:
            tokens.append(tok.lower())
    return tokens


def pythainlp_tokenize(text: str) -> list[str]:
    """Hook for pythainlp.word_tokenize. Raises if the extra is not installed."""
    try:
        from pythainlp.tokenize import word_tokenize
    except ImportError as exc:
        raise ImportError(
            "tokenizer='pythainlp' requires pythainlp. pip install ragkit[thai]"
        ) from exc
    return [t.strip().lower() for t in word_tokenize(text, engine="newmm") if t.strip()]


TOKENIZERS: dict[str, Callable[[str], list[str]]] = {
    "whitespace": whitespace_tokenize,
    "thai_char_ngram": thai_char_ngram,
    "pythainlp": pythainlp_tokenize,
}


def get_tokenizer(name: str) -> Callable[[str], list[str]]:
    try:
        return TOKENIZERS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown tokenizer {name!r}. Available: {sorted(TOKENIZERS)}") from exc
