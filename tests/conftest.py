from __future__ import annotations

import sys
from pathlib import Path

import pytest

from ragkit.chunkers.structure import StructureChunker
from ragkit.generators.fake import FakeGenerator
from ragkit.models import Document
from ragkit.parsers.text import TextParser
from ragkit.pipeline import RAGPipeline
from ragkit.rerankers.none import NoReranker
from ragkit.retrievers.bm25 import BM25Retriever
from ragkit.security.guard import InputGuard

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

TOY_DOCS = [
    Document(
        id="alpha",
        text=(
            "# Contrastive Learning for Sentence Embeddings\n\n"
            "Sentence embeddings can be trained with a contrastive loss that pulls "
            "representations of similar sentences together. SimCSE is a well-known method."
        ),
        metadata={"title": "Contrastive Learning", "tenant_id": "public", "roles": ["*"]},
    ),
    Document(
        id="beta",
        text=(
            "# BM25 Ranking Function\n\n"
            "BM25 is a sparse lexical ranking function used in information retrieval. "
            "It scores a document by term frequency and inverse document frequency."
        ),
        metadata={"title": "BM25", "tenant_id": "public", "roles": ["*"]},
    ),
    Document(
        id="secret",
        text="The vault code is 1234. This document is confidential and internal-only.",
        metadata={"title": "Secret", "tenant_id": "internal", "roles": ["admin"]},
    ),
]


@pytest.fixture
def toy_docs() -> list[Document]:
    return [d.model_copy(deep=True) for d in TOY_DOCS]


@pytest.fixture
def bm25_pipeline(toy_docs: list[Document]) -> RAGPipeline:
    pipe = RAGPipeline(
        parser=TextParser(),
        chunker=StructureChunker(max_tokens=200, overlap=20),
        retriever=BM25Retriever(tokenizer="whitespace"),
        reranker=NoReranker(),
        generator=FakeGenerator(),
        guard=InputGuard(max_query_length=2000, reject_injection=True),
        abstain_threshold=None,
        retrieve_top_k=10,
        rerank_top_k=10,
        generate_top_k=3,
        aggregate="doc",
    )
    chunks = pipe.ingest(toy_docs)
    pipe.index(chunks)
    return pipe
