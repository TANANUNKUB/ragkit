"""CLI: ingest, index, search, eval, ablate. Stdlib argparse — no extra deps."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ragkit.config import dump_config, load_config
from ragkit.eval.ablation import run_ablation
from ragkit.eval.runner import evaluate_retrieval, write_failures
from ragkit.pipeline import RAGPipeline


def _pipeline(config: str) -> RAGPipeline:
    return RAGPipeline.from_config(load_config(config))


def cmd_ingest(args: argparse.Namespace) -> int:
    pipe = _pipeline(args.config)
    chunks = pipe.ingest()
    out = Path(args.out or "data/chunks.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for ch in chunks:
            fh.write(ch.model_dump_json() + "\n")
    print(f"wrote {len(chunks)} chunks -> {out}")
    return 0


def cmd_index(args: argparse.Namespace) -> int:
    pipe = _pipeline(args.config)
    if args.chunks:
        from ragkit.models import Chunk

        chunks = [Chunk.model_validate_json(line) for line in Path(args.chunks).read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        chunks = pipe.ingest()
    pipe.index(chunks)
    print(f"indexed {len(chunks)} chunks")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    pipe = _pipeline(args.config)
    chunks = pipe.ingest()
    pipe.index(chunks)
    filters = json.loads(args.filters) if args.filters else {}
    if args.answer:
        result = pipe.answer(args.query, filters=filters)
        print(result.model_dump_json(indent=2))
    else:
        hits = pipe.search(args.query, filters=filters)
        payload = [h.model_dump() for h in hits[: args.top]]
        print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    pipe = _pipeline(args.config)
    cfg = pipe.config
    chunks = pipe.ingest()
    pipe.index(chunks)
    examples = list(pipe.loader.load_eval())
    ev = cfg.get("eval") or {}
    metrics = evaluate_retrieval(
        examples,
        pipe.search,
        k_values=tuple(ev.get("k_values") or [5, 20, 100]),
        ndcg_k=int(ev.get("ndcg_k") or 10),
        level=ev.get("level") or "doc",
    )
    failures = metrics.pop("failures")
    out_dir = Path(args.out or "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    write_failures(out_dir / "failures.jsonl", failures)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"failures: {len(failures)} -> {out_dir / 'failures.jsonl'}")
    return 0


def cmd_ablate(args: argparse.Namespace) -> int:
    configs = [p.strip() for p in args.configs.split(",") if p.strip()]
    result = run_ablation(configs, out_dir=args.out or "results")
    print(result["markdown"])
    return 0


def cmd_show_config(args: argparse.Namespace) -> int:
    print(dump_config(load_config(args.config)))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ragkit", description="Dataset-agnostic RAG toolkit")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_config(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--config", "-c", default="configs/base.yaml")

    ing = sub.add_parser("ingest", help="Parse + chunk corpus, write chunks.jsonl")
    add_config(ing)
    ing.add_argument("--out", default="data/chunks.jsonl")
    ing.set_defaults(func=cmd_ingest)

    idx = sub.add_parser("index", help="Ingest (or load chunks) and build retriever indexes")
    add_config(idx)
    idx.add_argument("--chunks", default=None, help="Optional chunks.jsonl from `ingest`")
    idx.set_defaults(func=cmd_index)

    se = sub.add_parser("search", help="Index (in-process) and run one query")
    add_config(se)
    se.add_argument("--query", "-q", required=True)
    se.add_argument("--filters", default=None, help='JSON object, e.g. {"year": 2023}')
    se.add_argument("--answer", action="store_true", help="Run generation as well")
    se.add_argument("--top", type=int, default=10)
    se.set_defaults(func=cmd_search)

    ev = sub.add_parser("eval", help="Run retrieval eval and write results/")
    add_config(ev)
    ev.add_argument("--out", default="results")
    ev.set_defaults(func=cmd_eval)

    ab = sub.add_parser("ablate", help="Run several configs and write a markdown table")
    ab.add_argument("--configs", required=True, help="Comma-separated YAML paths")
    ab.add_argument("--out", default="results")
    ab.set_defaults(func=cmd_ablate)

    sc = sub.add_parser("show-config", help="Print the merged YAML config")
    add_config(sc)
    sc.set_defaults(func=cmd_show_config)

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
