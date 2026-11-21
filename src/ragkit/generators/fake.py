"""Deterministic generator for tests and offline demos. Does not call a model."""

from __future__ import annotations

from ragkit.generators.base import constrain_citations, format_context
from ragkit.models import Answer, Query, ScoredHit
from ragkit.registry import register


@register("generator", "fake")
class FakeGenerator:
    def __init__(self, **_: object) -> None:
        pass

    def generate(self, query: Query, hits: list[ScoredHit]) -> Answer:
        _ = format_context(hits)  # keep the same allowed-id logic as the real generator
        if not hits:
            return Answer(answer="ไม่พบข้อมูลที่เกี่ยวข้อง", citations=[], found=False, hits=hits)
        top = hits[0]
        snippet = top.chunk.text.strip().replace("\n", " ")
        if len(snippet) > 240:
            snippet = snippet[:237] + "..."
        citations = constrain_citations([top.chunk.doc_id], {top.chunk.doc_id, top.chunk.id})
        return Answer(
            answer=f"จากเอกสาร {top.chunk.doc_id}: {snippet}",
            citations=citations,
            found=True,
            hits=hits,
        )
