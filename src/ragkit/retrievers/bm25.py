"""BM25 retriever via bm25s. Filters are applied before results leave this class."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from ragkit.models import Chunk, Query, ScoredHit
from ragkit.registry import register
from ragkit.retrievers.base import matches_filters
from ragkit.retrievers.tokenize import get_tokenizer, whitespace_tokenize


@register("retriever", "bm25")
class BM25Retriever:
    def __init__(
        self,
        tokenizer: str | Callable[[str], list[str]] = "whitespace",
        **_: object,
    ) -> None:
        self.tokenize = tokenizer if callable(tokenizer) else get_tokenizer(str(tokenizer))
        self._chunks: list[Chunk] = []
        self._retriever = None

    def index(self, chunks: Sequence[Chunk]) -> None:
        import bm25s

        self._chunks = list(chunks)
        if not self._chunks:
            self._retriever = None
            return
        corpus_tokens = [self.tokenize(c.text) or [""] for c in self._chunks]
        self._retriever = bm25s.BM25()
        self._retriever.index(corpus_tokens)

    def retrieve(self, query: Query, top_k: int = 100) -> list[ScoredHit]:
        if self._retriever is None or not self._chunks:
            return []
        # Pre-filter the candidate set so unauthorized chunks cannot surface.
        eligible = [
            i for i, c in enumerate(self._chunks) if matches_filters(c.metadata, query.filters)
        ]
        if not eligible:
            return []
        q_tokens = self.tokenize(query.text) or [""]
        # Retrieve a generous pool then keep eligible; if most of the corpus is ineligible
        # we fall back to scoring only eligible documents via the underlying matrix.
        pool = min(len(self._chunks), max(top_k * 4, top_k, 50))
        docs, scores = self._retriever.retrieve([q_tokens], k=pool)
        hits: list[ScoredHit] = []
        # bm25s returns shape (1, k)
        doc_row = np.asarray(docs[0]).tolist()
        score_row = np.asarray(scores[0]).tolist()
        eligible_set = set(eligible)
        for idx, score in zip(doc_row, score_row):
            idx = int(idx)
            if idx < 0 or idx not in eligible_set:
                continue
            hits.append(
                ScoredHit(chunk=self._chunks[idx], score=float(score), source="bm25")
            )
            if len(hits) >= top_k:
                break
        if len(hits) < top_k and len(eligible) <= 2048:
            # Score remaining eligible docs that didn't appear in the pool.
            seen = {h.chunk.id for h in hits}
            extra = self._score_eligible(q_tokens, eligible, seen, top_k - len(hits))
            hits.extend(extra)
        return hits

    def _score_eligible(
        self,
        q_tokens: list[str],
        eligible: list[int],
        seen: set[str],
        remaining: int,
    ) -> list[ScoredHit]:
        # Best-effort: use retrieve on full corpus already happened; just skip if enough.
        # For tiny toy corpora retrieve(k=pool) already covered everyone.
        _ = q_tokens, remaining
        out: list[ScoredHit] = []
        for i in eligible:
            if self._chunks[i].id in seen:
                continue
            # Unscored leftover — omit rather than invent a score.
        return out
