"""FinanceBench adapter stub (PatronusAI/financebench + official PDF bundle).

Field mappings:

Corpus
  - doc_name (PDF stem, e.g. a 10-K) -> Document.id
  - parsed PDF text (keep page breaks / page headings)
                                     -> Document.text
  - source path, company, filing type -> Document.metadata
  - page numbers should survive into Chunk.metadata['page'] after parsing

Eval (open set ~150 questions)
  - question              -> EvalExample.query
  - answer                -> EvalExample.gold_answer
  - doc_name              -> EvalExample.gold_doc_ids  (list of one or more)
  - evidence_page_num     -> EvalExample.metadata['evidence_page_num']
                             (page-level gold — use this when scoring page hits)

PDFs contain financial tables. Prefer a table-aware parser/chunker: do not split
a table mid-way; if you must, repeat the header row in each table chunk.
"""

from __future__ import annotations

from collections.abc import Iterable

from ragkit.models import Document, EvalExample
from ragkit.registry import register


@register("loader", "financebench")
class FinanceBenchAdapter:
    def __init__(
        self,
        path: str | None = None,
        pdf_dir: str | None = None,
        **_: object,
    ) -> None:
        self.path = path
        self.pdf_dir = pdf_dir

    def load_corpus(self) -> Iterable[Document]:
        # TODO: iterate PDFs in pdf_dir (see github.com/patronus-ai/financebench/tree/main/pdfs).
        # Parse with the configured Parser (pymupdf / docling). Document.id = doc_name.
        raise NotImplementedError(
            "FinanceBenchAdapter.load_corpus is a stub. "
            "Parse official 10-K PDFs and set Document.id = doc_name. "
            "Preserve page numbers for page-level evaluation."
        )

    def load_eval(self) -> Iterable[EvalExample]:
        # TODO: load PatronusAI/financebench. Map question, answer, doc_name,
        # evidence_page_num as documented above. Check the dataset license before
        # publishing results in a public portfolio.
        raise NotImplementedError(
            "FinanceBenchAdapter.load_eval is a stub. "
            "Map question -> query, answer -> gold_answer, doc_name -> gold_doc_ids, "
            "evidence_page_num -> metadata."
        )
