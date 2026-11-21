"""Aggregate chunk-level hits to document level by taking the max score per doc_id."""

from __future__ import annotations

from ragkit.models import ScoredHit


def aggregate_max_per_doc(hits: list[ScoredHit]) -> list[ScoredHit]:
    best: dict[str, ScoredHit] = {}
    for hit in hits:
        doc_id = hit.chunk.doc_id
        current = best.get(doc_id)
        if current is None or hit.score > current.score:
            best[doc_id] = hit
    return sorted(best.values(), key=lambda h: (-h.score, h.chunk.doc_id))
