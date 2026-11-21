from __future__ import annotations

from ragkit.models import Query, ScoredHit
from ragkit.registry import register


@register("reranker", "none")
class NoReranker:
    def __init__(self, **_: object) -> None:
        pass

    def rerank(self, query: Query, hits: list[ScoredHit], top_k: int = 50) -> list[ScoredHit]:
        _ = query
        return hits[:top_k]
