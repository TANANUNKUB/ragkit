"""Retrieval (and optional answer) evaluation over a list of EvalExample."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from ragkit.eval.metrics import mrr, ndcg_at_k, percentile, recall_at_k
from ragkit.models import Answer, EvalExample, Query, ScoredHit


def _pred_ids(hits: list[ScoredHit], *, level: str) -> list[str]:
    if level in {"doc", "paper", "document"}:
        # Preserve retrieval order; unique doc ids.
        seen: set[str] = set()
        ids: list[str] = []
        for h in hits:
            if h.chunk.doc_id not in seen:
                seen.add(h.chunk.doc_id)
                ids.append(h.chunk.doc_id)
        return ids
    return [h.chunk.id for h in hits]


def evaluate_retrieval(
    examples: Sequence[EvalExample],
    search_fn: Callable[[Query], list[ScoredHit]],
    *,
    k_values: Sequence[int] = (5, 20, 100),
    ndcg_k: int = 10,
    level: str = "doc",
    failure_k: int = 5,
) -> dict[str, Any]:
    """Run retrieval metrics. `level` is 'doc' (compare gold_doc_ids) or 'chunk'."""
    rec_sums = {k: 0.0 for k in k_values}
    ndcg_sum = 0.0
    mrr_sum = 0.0
    n = 0
    latencies_ms: list[float] = []
    failures: list[dict[str, Any]] = []

    for ex in examples:
        gold = ex.gold_doc_ids if level in {"doc", "paper", "document"} else ex.gold_chunk_ids
        if not gold:
            continue
        n += 1
        t0 = time.perf_counter()
        hits = search_fn(Query(text=ex.query, metadata=dict(ex.metadata)))
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        ranked = _pred_ids(hits, level=level)
        for k in k_values:
            rec_sums[k] += recall_at_k(ranked, gold, k)
        ndcg_sum += ndcg_at_k(ranked, gold, ndcg_k)
        mrr_sum += mrr(ranked, gold)
        if recall_at_k(ranked, gold, failure_k) == 0.0:
            failures.append(
                {
                    "id": ex.id,
                    "query": ex.query,
                    "gold": list(gold),
                    "predicted": ranked[: max(k_values)],
                    "metadata": ex.metadata,
                }
            )

    metrics: dict[str, Any] = {
        "n": n,
        "mrr": (mrr_sum / n) if n else 0.0,
        f"ndcg@{ndcg_k}": (ndcg_sum / n) if n else 0.0,
        "latency_p50_ms": percentile(latencies_ms, 50),
        "latency_p95_ms": percentile(latencies_ms, 95),
        "failures": failures,
    }
    for k, total in rec_sums.items():
        metrics[f"recall@{k}"] = (total / n) if n else 0.0
    return metrics


def evaluate_answers(
    examples: Sequence[EvalExample],
    answer_fn: Callable[[Query], Answer],
    *,
    use_ragas: bool = False,
) -> dict[str, Any]:
    """Lightweight answer eval. RAGAS is an optional hook (skipped unless installed + enabled)."""
    n = 0
    found = 0
    citation_hits = 0
    citation_considered = 0
    answers: list[dict[str, Any]] = []
    for ex in examples:
        n += 1
        ans = answer_fn(Query(text=ex.query, metadata=dict(ex.metadata)))
        if ans.found:
            found += 1
        if ex.gold_doc_ids and ans.citations:
            citation_considered += 1
            if set(ans.citations) & set(ex.gold_doc_ids):
                citation_hits += 1
        answers.append({"id": ex.id, "found": ans.found, "citations": ans.citations})

    result: dict[str, Any] = {
        "n": n,
        "found_rate": (found / n) if n else 0.0,
        "citation_precision_loose": (citation_hits / citation_considered) if citation_considered else 0.0,
        "answers": answers,
    }
    if use_ragas:
        result["ragas"] = _maybe_ragas(examples, answers)
    return result


def _maybe_ragas(examples: Sequence[EvalExample], answers: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        import ragas  # noqa: F401
    except ImportError:
        return {"skipped": True, "reason": "ragas is not installed. pip install ragkit[eval]"}
    # TODO: wire Dataset + faithfulness / answer_relevancy when you opt into a judge LLM.
    _ = examples, answers
    return {
        "skipped": True,
        "reason": "RAGAS hook is present but not wired. Plug in a judge LLM in eval/runner.py.",
    }


def write_failures(path: str | Path, failures: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in failures:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
