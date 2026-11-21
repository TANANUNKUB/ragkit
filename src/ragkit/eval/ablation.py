"""Run several configs and write a markdown table plus failures.jsonl."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ragkit.config import load_config
from ragkit.eval.runner import evaluate_retrieval, write_failures
from ragkit.models import EvalExample
from ragkit.pipeline import RAGPipeline


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def markdown_table(rows: list[dict[str, Any]], columns: Sequence[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(_fmt(row.get(c, "")) for c in columns) + " |" for row in rows]
    return "\n".join([header, sep, *body]) + "\n"


def run_ablation(
    config_paths: Sequence[str | Path],
    *,
    examples: Sequence[EvalExample] | None = None,
    out_dir: str | Path = "results",
    k_values: Sequence[int] = (5, 20, 100),
    ndcg_k: int = 10,
    level: str = "doc",
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    all_failures: dict[str, list[dict[str, Any]]] = {}

    for path in config_paths:
        path = Path(path)
        cfg = load_config(path)
        name = path.stem
        pipeline = RAGPipeline.from_config(cfg)
        # If examples aren't supplied, try the adapter (may be a stub).
        eval_set = list(examples) if examples is not None else list(pipeline.loader.load_eval())
        if not pipeline._indexed:
            chunks = pipeline.ingest()
            pipeline.index(chunks)
        metrics = evaluate_retrieval(
            eval_set,
            pipeline.search,
            k_values=k_values,
            ndcg_k=ndcg_k,
            level=cfg.get("eval", {}).get("level", level),
        )
        failures = metrics.pop("failures")
        all_failures[name] = failures
        write_failures(out / f"failures_{name}.jsonl", failures)
        row = {"config": name, **metrics, "n_failures@5": len(failures)}
        rows.append(row)
        (out / f"metrics_{name}.json").write_text(
            json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    columns = ["config", *[f"recall@{k}" for k in k_values], f"ndcg@{ndcg_k}", "mrr", "latency_p50_ms", "latency_p95_ms", "n", "n_failures@5"]
    table = markdown_table(rows, columns)
    (out / "ablation.md").write_text(table, encoding="utf-8")
    return {"rows": rows, "markdown": table, "failures": all_failures}
