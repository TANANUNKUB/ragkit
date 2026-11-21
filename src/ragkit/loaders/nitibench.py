"""WangchanX-Legal-ThaiCCL-RAG / NitiBench adapter stub (airesearch/...).

Field mappings:

Corpus
  - law section identifier -> Document.id   (one section ≈ one document; often one chunk)
  - section text (Thai)    -> Document.text
  - law name, section no., statute id
                           -> Document.metadata

Eval
  - question (Thai)        -> EvalExample.query
  - positive_contexts      -> EvalExample.gold_doc_ids
    * test set: a single positive law section per question
  - optional written answer (lawyer-authored on the test split)
                           -> EvalExample.gold_answer

Notes for later optimization (do not bake into the core pipeline):
  - BM25 needs a Thai tokenizer (pythainlp hook in retrievers.bm25).
  - Chunk by law section, not by a generic token window, when sections are already clean.
  - Train questions may be synthetic; prefer the official test split for reported numbers.
"""

from __future__ import annotations

from collections.abc import Iterable

from ragkit.models import Document, EvalExample
from ragkit.registry import register


@register("loader", "nitibench")
class NitiBenchAdapter:
    def __init__(
        self,
        path: str | None = None,
        split: str = "test",
        **_: object,
    ) -> None:
        self.path = path
        self.split = split

    def load_corpus(self) -> Iterable[Document]:
        # TODO: load airesearch/WangchanX-Legal-ThaiCCL-RAG corpus / unique sections.
        # Document.id should match whatever identifier appears in positive_contexts.
        raise NotImplementedError(
            "NitiBenchAdapter.load_corpus is a stub. "
            "Load Thai law sections and set Document.id to the section identifier "
            "used in positive_contexts."
        )

    def load_eval(self) -> Iterable[EvalExample]:
        # TODO: map question -> query, positive_contexts -> gold_doc_ids.
        # Test split has a single gold section per question.
        raise NotImplementedError(
            "NitiBenchAdapter.load_eval is a stub. "
            "Map question -> query and positive_contexts -> gold_doc_ids. "
            "Prefer the official test split for reported metrics."
        )
