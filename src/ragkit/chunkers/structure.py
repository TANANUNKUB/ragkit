"""Structure-aware chunker: headings → paragraphs → sentences, with token budget + overlap."""

from __future__ import annotations

from collections.abc import Sequence

from ragkit.chunkers.splitting import (
    is_markdown_table,
    iter_sections,
    split_paragraphs,
    split_sentences,
    split_table_keep_header,
)
from ragkit.chunkers.tokens import ApproxTokenCounter, TokenCounter
from ragkit.models import Chunk, Document
from ragkit.parsers.base import ParsedDocument
from ragkit.registry import register


def _as_parsed(doc: ParsedDocument | Document) -> ParsedDocument:
    if isinstance(doc, ParsedDocument):
        return doc
    title = doc.metadata.get("title")
    return ParsedDocument(
        id=doc.id,
        text=doc.text,
        title=title if isinstance(title, str) else None,
        metadata=dict(doc.metadata),
    )


@register("chunker", "structure")
class StructureChunker:
    def __init__(
        self,
        max_tokens: int = 600,
        overlap: int = 80,
        min_tokens: int = 20,
        table_max_rows: int = 30,
        token_counter: TokenCounter | None = None,
        **_: object,
    ) -> None:
        if overlap >= max_tokens:
            raise ValueError("overlap must be smaller than max_tokens")
        self.max_tokens = max_tokens
        self.overlap = overlap
        self.min_tokens = min_tokens
        self.table_max_rows = table_max_rows
        self.counter: TokenCounter = token_counter or ApproxTokenCounter()

    def chunk(self, parsed: ParsedDocument | Document) -> Sequence[Chunk]:
        parsed = _as_parsed(parsed)
        packed: list[tuple[str, str]] = []
        for section_path, body in iter_sections(parsed.text, title=parsed.title):
            reserved = self.counter.count(f"{section_path}\n\n") if section_path else 0
            body_budget = max(1, self.max_tokens - reserved)
            blocks = self._blocks(body, body_budget)
            packed.extend(self._pack_section(section_path, blocks, reserved))

        chunks: list[Chunk] = []
        for i, (section, body) in enumerate(packed):
            prefix = f"{section}\n\n" if section else ""
            text = f"{prefix}{body}".strip()
            # Final invariant: never exceed max_tokens (hard-cut as last resort).
            if self.counter.count(text) > self.max_tokens:
                for j, piece in enumerate(self._hard_cut(text, self.max_tokens)):
                    meta = {**parsed.metadata, "section": section, "chunk_idx": len(chunks)}
                    chunks.append(
                        Chunk(id=f"{parsed.id}#{len(chunks)}", doc_id=parsed.id, text=piece, metadata=meta)
                    )
                continue
            meta = {**parsed.metadata, "section": section, "chunk_idx": i}
            chunks.append(Chunk(id=f"{parsed.id}#{i}", doc_id=parsed.id, text=text, metadata=meta))
        # Re-number chunk_idx / ids to be contiguous after hard-cuts.
        for i, ch in enumerate(chunks):
            ch.metadata["chunk_idx"] = i
            ch.id = f"{parsed.id}#{i}"
        return chunks

    def _blocks(self, body: str, budget: int) -> list[str]:
        out: list[str] = []
        for para in split_paragraphs(body):
            if is_markdown_table(para):
                # Split oversized tables; leave small ones intact.
                pieces = [para]
                if self.counter.count(para) > budget:
                    # Estimate rows that fit: keep header + as many rows as possible.
                    n_lines = max(3, len(para.splitlines()))
                    # Start conservative; split_table_keep_header no-ops if already small.
                    max_rows = max(3, int(n_lines * budget / max(1, self.counter.count(para))))
                    pieces = split_table_keep_header(para, max_rows)
                out.extend(pieces)
                continue
            if self.counter.count(para) <= budget:
                out.append(para)
                continue
            out.extend(self._pack_units(split_sentences(para), budget))
        return out

    def _pack_units(self, units: list[str], budget: int) -> list[str]:
        packed: list[str] = []
        current: list[str] = []
        for unit in units:
            joined = " ".join(current + [unit]).strip() if current else unit
            if current and self.counter.count(joined) > budget:
                packed.append(" ".join(current).strip())
                tail = self.counter.take_suffix(packed[-1], min(self.overlap, max(0, budget - 1)))
                current = [tail, unit] if tail else [unit]
                if self.counter.count(" ".join(s for s in current if s).strip()) > budget:
                    packed.extend(self._hard_cut(unit, budget))
                    current = []
            else:
                if not current and self.counter.count(unit) > budget:
                    packed.extend(self._hard_cut(unit, budget))
                else:
                    current.append(unit)
        if current:
            text = " ".join(s for s in current if s).strip()
            if text:
                packed.append(text)
        return packed

    def _hard_cut(self, text: str, budget: int) -> list[str]:
        out: list[str] = []
        remaining = text
        step = max(1, budget - min(self.overlap, budget // 4) if budget > 1 else budget)
        while remaining:
            if self.counter.count(remaining) <= budget:
                out.append(remaining)
                break
            piece = self.counter.take_prefix(remaining, budget)
            if not piece:
                break
            out.append(piece)
            consumed = self.counter.take_prefix(remaining, step)
            remaining = remaining[len(consumed) :]
        return out

    def _pack_section(
        self, section: str, blocks: list[str], reserved: int
    ) -> list[tuple[str, str]]:
        """Pack same-section blocks under (max_tokens - reserved) without a second overlap pass."""
        budget = max(1, self.max_tokens - reserved)
        packed: list[tuple[str, str]] = []
        cur: list[str] = []

        def flush() -> None:
            nonlocal cur
            body = "\n\n".join(cur).strip()
            if body:
                packed.append((section, body))
            cur = []

        for block in blocks:
            proposed = "\n\n".join(cur + [block]).strip() if cur else block
            if cur and self.counter.count(proposed) > budget:
                flush()
                cur = [block]
                if self.counter.count(block) > budget:
                    for piece in self._hard_cut(block, budget):
                        packed.append((section, piece))
                    cur = []
            else:
                if self.counter.count(block) > budget and not cur:
                    for piece in self._hard_cut(block, budget):
                        packed.append((section, piece))
                else:
                    cur.append(block)
        flush()
        return packed
