from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ragkit.models import Document


@dataclass
class ParsedDocument:
    """Parser output: cleaned text plus optional structural metadata."""

    id: str
    text: str
    title: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    # pages: list[{page: int, text: str}] may live in metadata["pages"]


@runtime_checkable
class Parser(Protocol):
    def parse(self, document: Document) -> ParsedDocument: ...
