"""Fixed-size fallback chunker with overlap. Still avoids space-based sentence splits."""

from __future__ import annotations

from collections.abc import Sequence

from ragkit.chunkers.structure import _as_parsed
from ragkit.chunkers.tokens import ApproxTokenCounter, TokenCounter
from ragkit.models import Chunk, Document
from ragkit.parsers.base import ParsedDocument
from ragkit.registry import register


@register("chunker", "fixed")
class FixedSizeChunker:
    def __init__(
        self,
        max_tokens: int = 600,
        overlap: int = 80,
        token_counter: TokenCounter | None = None,
        **_: object,
    ) -> None:
        if overlap >= max_tokens:
            raise ValueError("overlap must be smaller than max_tokens")
        self.max_tokens = max_tokens
        self.overlap = overlap
        self.counter: TokenCounter = token_counter or ApproxTokenCounter()

    def chunk(self, parsed: ParsedDocument | Document) -> Sequence[Chunk]:
        parsed = _as_parsed(parsed)
        windows = self._windows(parsed.text)
        chunks: list[Chunk] = []
        for i, body in enumerate(windows):
            meta = {**parsed.metadata, "section": parsed.title or "", "chunk_idx": i}
            chunks.append(Chunk(id=f"{parsed.id}#{i}", doc_id=parsed.id, text=body, metadata=meta))
        return chunks

    def _windows(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []
        if self.counter.count(text) <= self.max_tokens:
            return [text]
        out: list[str] = []
        remaining = text
        step = max(1, self.max_tokens - self.overlap)
        while remaining:
            if self.counter.count(remaining) <= self.max_tokens:
                out.append(remaining)
                break
            piece = self.counter.take_prefix(remaining, self.max_tokens)
            if not piece:
                break
            out.append(piece)
            consumed = self.counter.take_prefix(remaining, step)
            remaining = remaining[len(consumed) :]
        return out
