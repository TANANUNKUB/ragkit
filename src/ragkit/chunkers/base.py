from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from ragkit.models import Chunk, Document
from ragkit.parsers.base import ParsedDocument


@runtime_checkable
class Chunker(Protocol):
    def chunk(self, parsed: ParsedDocument | Document) -> Sequence[Chunk]: ...
