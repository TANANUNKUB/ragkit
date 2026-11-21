"""Reciprocal Rank Fusion: score(d) = Σ 1 / (k + rank_i(d))."""

from __future__ import annotations

from collections import defaultdict

from ragkit.models import ScoredHit


def rrf_fuse(ranked_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Fuse lists of ids. Rank is 1-based. Returns ids sorted by RRF score descending."""
    if k <= 0:
        raise ValueError("rrf k must be > 0")
    scores: dict[str, float] = defaultdict(float)
    for lst in ranked_lists:
        for rank, item_id in enumerate(lst, start=1):
            scores[item_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


def rrf_fuse_hits(lists: list[list[ScoredHit]], k: int = 60) -> list[ScoredHit]:
    """Fuse hit lists by chunk.id, keeping the first-seen chunk object for each id."""
    id_lists: list[list[str]] = []
    chunks: dict[str, ScoredHit] = {}
    for lst in lists:
        ids: list[str] = []
        for hit in lst:
            cid = hit.chunk.id
            ids.append(cid)
            if cid not in chunks:
                chunks[cid] = hit
        id_lists.append(ids)
    fused = rrf_fuse(id_lists, k=k)
    return [
        ScoredHit(chunk=chunks[cid].chunk, score=score, source="rrf")
        for cid, score in fused
        if cid in chunks
    ]
