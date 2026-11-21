"""PDF parser via PyMuPDF (lazy import). Optional Docling hook left as TODO."""

from __future__ import annotations

from pathlib import Path

from ragkit.models import Document
from ragkit.parsers.base import ParsedDocument
from ragkit.parsers.text import clean_text
from ragkit.registry import register


@register("parser", "pdf")
class PdfParser:
    """Parse a Document whose metadata['source'] points at a PDF, or whose text is empty.

    If `document.text` is already populated (e.g. FolderAdapter already extracted it),
    this just cleans the text and records page headings when present.
    """

    def __init__(self, extract_tables_as_markdown: bool = True, **_: object) -> None:
        self.extract_tables_as_markdown = extract_tables_as_markdown

    def parse(self, document: Document) -> ParsedDocument:
        source = document.metadata.get("source")
        if document.text.strip():
            text = clean_text(document.text)
        elif source and Path(str(source)).suffix.lower() == ".pdf":
            text = clean_text(self._extract(Path(str(source))))
        else:
            text = clean_text(document.text)
        title = document.metadata.get("title")
        return ParsedDocument(
            id=document.id,
            text=text,
            title=title if isinstance(title, str) else None,
            metadata=dict(document.metadata),
        )

    def _extract(self, path: Path) -> str:
        try:
            import fitz
        except ImportError as exc:
            raise ImportError("PdfParser requires pymupdf. pip install ragkit[pdf]") from exc
        parts: list[str] = []
        with fitz.open(path) as doc:
            for i, page in enumerate(doc, start=1):
                # TODO: optional table extraction to markdown (page.find_tables()).
                # For financial PDFs, keep tables intact and repeat header rows if split.
                parts.append(f"## Page {i}\n{page.get_text('text') or ''}")
        return "\n\n".join(parts)


@register("parser", "docling")
class DoclingParser:
    """Hook for IBM Docling structured PDF parsing. Not implemented in the skeleton."""

    def __init__(self, **_: object) -> None:
        pass

    def parse(self, document: Document) -> ParsedDocument:
        # TODO: `from docling.document_converter import DocumentConverter`
        # Convert metadata['source'] PDF, export markdown (headings + tables),
        # and fill ParsedDocument. Prefer this for FinanceBench 10-Ks.
        raise NotImplementedError(
            "DoclingParser is a hook. Install docling and convert PDFs to markdown "
            "with headings and tables preserved. See parsers/pdf.py."
        )
