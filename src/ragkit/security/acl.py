"""ACL / metadata filters. Applied *inside* retrievers (pre-search), never after generation."""

from __future__ import annotations

from typing import Any

from ragkit.models import Query


def merge_filters(*parts: dict[str, Any] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for part in parts:
        if part:
            out.update(part)
    return out


def apply_default_filters(query: Query, defaults: dict[str, Any] | None) -> Query:
    if not defaults:
        return query
    filters = merge_filters(defaults, query.filters)
    return query.model_copy(update={"filters": filters})
