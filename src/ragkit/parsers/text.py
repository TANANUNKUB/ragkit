"""Identity / light-clean parser for already-textual documents."""

from __future__ import annotations

import re

from ragkit.models import Document
from ragkit.parsers.base import ParsedDocument
from ragkit.registry import register

_REPEATED_BLANK = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _REPEATED_BLANK.sub("\n\n", text)
    return text.strip()


@register("parser", "text")
class TextParser:
    def __init__(self, **_: object) -> None:
        pass

    def parse(self, document: Document) -> ParsedDocument:
        title = document.metadata.get("title")
        return ParsedDocument(
            id=document.id,
            text=clean_text(document.text),
            title=title if isinstance(title, str) else None,
            metadata=dict(document.metadata),
        )
