"""LitSearch adapter stub (princeton-nlp/LitSearch).

Field mappings (fill these in when wiring the HuggingFace / local JSONL files):

Corpus
  - corpusid            -> Document.id          (and later Chunk.doc_id)
  - title + abstract    -> Document.text        (full text optional, for section chunks)
  - title, year, venue, corpusid
                        -> Document.metadata

Eval
  - query text          -> EvalExample.query
  - gold corpusids      -> EvalExample.gold_doc_ids
  - split / query type (Broad vs Specific)
                        -> EvalExample.metadata

Eval is at *paper* level, not chunk level. After retrieval, aggregate chunk hits
to doc_id with max score before computing Recall@k / nDCG (see retriever.aggregate: doc).
"""

from __future__ import annotations

from collections.abc import Iterable

from ragkit.models import Document, EvalExample
from ragkit.registry import register


@register("loader", "litsearch")
class LitSearchAdapter:
    def __init__(
        self,
        path: str | None = None,
        corpus_split: str = "corpus_lite",
        eval_split: str = "query",
        include_full_text: bool = False,
        **_: object,
    ) -> None:
        self.path = path
        self.corpus_split = corpus_split
        self.eval_split = eval_split
        self.include_full_text = include_full_text

    def load_corpus(self) -> Iterable[Document]:
        # TODO: load princeton-nlp/LitSearch (or a local cache at self.path).
        # Map corpusid -> Document.id, concatenate title + abstract (+ full text
        # if include_full_text), and put year/venue/title into metadata.
        raise NotImplementedError(
            "LitSearchAdapter.load_corpus is a stub. "
            "Load princeton-nlp/LitSearch and map corpusid/title/abstract/full text "
            "as documented in this module. Optimize paper-level aggregation in config "
            "(retriever.aggregate: doc)."
        )

    def load_eval(self) -> Iterable[EvalExample]:
        # TODO: each query has gold corpusids. Put query type (Broad/Specific) in metadata.
        raise NotImplementedError(
            "LitSearchAdapter.load_eval is a stub. "
            "Map query text -> EvalExample.query and gold corpusids -> gold_doc_ids."
        )
