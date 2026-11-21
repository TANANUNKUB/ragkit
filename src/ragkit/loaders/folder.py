"""Generic folder adapter: load .txt / .md / .pdf files, optional eval.jsonl."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from ragkit.models import Document, EvalExample
from ragkit.registry import register


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _read_pdf(path: Path) -> str:
    try:
        import fitz  # pymupdf
    except ImportError as exc:
        raise ImportError(
            f"Reading {path.name} requires pymupdf. Install with: pip install ragkit[pdf]"
        ) from exc
    parts: list[str] = []
    with fitz.open(path) as doc:
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text") or ""
            parts.append(f"## Page {i}\n{text}")
    return "\n\n".join(parts)


@register("loader", "folder")
class FolderAdapter:
    """Load a directory of documents. Optional `eval.jsonl` next to them (or `eval_path`)."""

    def __init__(
        self,
        path: str | Path,
        eval_path: str | Path | None = None,
        tenant_id: str = "public",
        roles: list[str] | None = None,
        **_: object,
    ) -> None:
        self.path = Path(path)
        self.eval_path = Path(eval_path) if eval_path else self.path / "eval.jsonl"
        self.tenant_id = tenant_id
        self.roles = roles or ["*"]

    def load_corpus(self) -> Iterator[Document]:
        if not self.path.exists():
            raise FileNotFoundError(f"Corpus path not found: {self.path}")
        files = sorted(
            p
            for p in self.path.rglob("*")
            if p.is_file() and p.suffix.lower() in {".txt", ".md", ".pdf"} and p.name != "eval.jsonl"
        )
        for path in files:
            text = _read_pdf(path) if path.suffix.lower() == ".pdf" else _read_text_file(path)
            doc_id = path.stem
            yield Document(
                id=doc_id,
                text=text,
                metadata={
                    "source": str(path),
                    "title": path.stem,
                    "tenant_id": self.tenant_id,
                    "roles": list(self.roles),
                },
            )

    def load_eval(self) -> Iterable[EvalExample]:
        if not self.eval_path.exists():
            return []
        examples: list[EvalExample] = []
        with self.eval_path.open(encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                examples.append(
                    EvalExample(
                        id=str(raw.get("id", i)),
                        query=raw["query"],
                        gold_doc_ids=list(raw.get("gold_doc_ids") or raw.get("gold_docs") or []),
                        gold_chunk_ids=list(raw.get("gold_chunk_ids") or []),
                        gold_answer=raw.get("gold_answer") or raw.get("answer"),
                        metadata=dict(raw.get("metadata") or {}),
                    )
                )
        return examples
