"""Retrieval metrics. Pure functions, no I/O, no model downloads."""

from __future__ import annotations

import math
from collections.abc import Sequence


def _as_ids(ranked: Sequence[str]) -> list[str]:
    return list(ranked)


def recall_at_k(ranked_ids: Sequence[str], gold_ids: Sequence[str], k: int) -> float:
    """|gold ∩ top-k| / |gold|. Empty gold → 0.0 (callers may skip such examples)."""
    if k <= 0:
        raise ValueError("k must be > 0")
    gold = set(gold_ids)
    if not gold:
        return 0.0
    top = set(_as_ids(ranked_ids)[:k])
    return len(gold & top) / len(gold)


def _dcg(relevances: Sequence[float]) -> float:
    return sum(rel / math.log2(i + 1) for i, rel in enumerate(relevances, start=1))


def ndcg_at_k(
    ranked_ids: Sequence[str],
    gold_ids: Sequence[str],
    k: int,
    *,
    graded: dict[str, float] | None = None,
) -> float:
    """Binary nDCG@k unless `graded` maps id → relevance grade (>0)."""
    if k <= 0:
        raise ValueError("k must be > 0")
    gold = set(gold_ids)
    if not gold and not graded:
        return 0.0
    top = _as_ids(ranked_ids)[:k]

    def rel(i: str) -> float:
        if graded is not None:
            return float(graded.get(i, 0.0))
        return 1.0 if i in gold else 0.0

    dcg = _dcg([rel(i) for i in top])
    # Ideal: gold (or graded) items sorted by relevance, padded to k with zeros.
    if graded is not None:
        ideal_vals = sorted(graded.values(), reverse=True)[:k]
    else:
        ideal_vals = [1.0] * min(len(gold), k)
    idcg = _dcg(ideal_vals)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def mrr(ranked_ids: Sequence[str], gold_ids: Sequence[str]) -> float:
    """Mean Reciprocal Rank for a single query: 1/rank of first gold, or 0."""
    gold = set(gold_ids)
    if not gold:
        return 0.0
    for rank, item in enumerate(_as_ids(ranked_ids), start=1):
        if item in gold:
            return 1.0 / rank
    return 0.0


def percentile(values: Sequence[float], p: float) -> float:
    """Linear interpolation percentile. `p` in [0, 100]. Empty → 0.0."""
    if not values:
        return 0.0
    if not 0 <= p <= 100:
        raise ValueError("percentile must be in [0, 100]")
    xs = sorted(float(v) for v in values)
    if len(xs) == 1:
        return xs[0]
    rank = (p / 100.0) * (len(xs) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return xs[lo]
    w = rank - lo
    return xs[lo] * (1.0 - w) + xs[hi] * w
