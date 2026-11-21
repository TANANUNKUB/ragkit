from __future__ import annotations

from typing import Protocol, runtime_checkable

from ragkit.models import Query, ScoredHit


@runtime_checkable
class Reranker(Protocol):
    def rerank(self, query: Query, hits: list[ScoredHit], top_k: int = 50) -> list[ScoredHit]: ...
