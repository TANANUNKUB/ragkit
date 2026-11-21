from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from ragkit.models import Chunk, Query, ScoredHit


@runtime_checkable
class Retriever(Protocol):
    def index(self, chunks: Sequence[Chunk]) -> None: ...

    def retrieve(self, query: Query, top_k: int = 100) -> list[ScoredHit]: ...


@runtime_checkable
class Embedder(Protocol):
    dim: int

    def embed(self, texts: Sequence[str]) -> Any: ...  # np.ndarray [n, dim]


def matches_filters(metadata: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    """AND-match simple equality / membership filters against chunk metadata.

    Special keys:
      - tenant_id: exact match (missing tenant is treated as public/"*" wildcards)
      - roles: chunk is allowed if its roles contain "*" or intersect the query roles
      - other keys: equality, or membership when the filter value is a list
    """
    if not filters:
        return True
    meta = metadata or {}
    for key, expected in filters.items():
        if key == "roles":
            wanted = set(expected if isinstance(expected, (list, tuple, set)) else [expected])
            have = set(meta.get("roles") or [])
            if "*" in have or "*" in wanted:
                continue
            if not have or have.isdisjoint(wanted):
                return False
            continue
        actual = meta.get(key)
        if expected is None:
            continue
        if isinstance(expected, (list, tuple, set)):
            if actual not in expected:
                return False
        elif actual != expected:
            # tenant_id wildcard: chunks tagged public or * pass a more specific tenant? No.
            # Only allow missing-or-public when filter isn't set. When filter is set, exact match
            # unless the chunk is marked tenant_id="public" or "*".
            if key == "tenant_id" and actual in (None, "public", "*"):
                continue
            return False
    return True
