"""Hybrid retriever: BM25 + dense → RRF, optional doc-level aggregation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ragkit.models import Chunk, Query, ScoredHit
from ragkit.registry import register
from ragkit.retrievers.aggregate import aggregate_max_per_doc
from ragkit.retrievers.bm25 import BM25Retriever
from ragkit.retrievers.dense import DenseRetriever
from ragkit.retrievers.rrf import rrf_fuse_hits


@register("retriever", "hybrid")
class HybridRetriever:
    def __init__(
        self,
        rrf_k: int = 60,
        aggregate: str = "chunk",  # chunk | doc | paper
        bm25: dict[str, Any] | None = None,
        dense: dict[str, Any] | None = None,
        **_: object,
    ) -> None:
        self.rrf_k = rrf_k
        self.aggregate = aggregate
        self.bm25 = BM25Retriever(**(bm25 or {}))
        self.dense = DenseRetriever(**(dense or {}))

    def index(self, chunks: Sequence[Chunk]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def retrieve(self, query: Query, top_k: int = 100) -> list[ScoredHit]:
        # Each branch already applies ACL / metadata filters.
        bm25_hits = self.bm25.retrieve(query, top_k=top_k)
        dense_hits = self.dense.retrieve(query, top_k=top_k)
        fused = rrf_fuse_hits([bm25_hits, dense_hits], k=self.rrf_k)
        if self.aggregate in {"doc", "paper", "document"}:
            fused = aggregate_max_per_doc(fused)
        return fused[:top_k]
