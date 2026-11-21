"""Core data models. Keep these dataset-agnostic; adapters map their fields into these."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Document(BaseModel):
    """A source document before chunking."""

    id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    # Common metadata keys (optional, not enforced):
    # title, source, date, year, venue, page, pages, tenant_id, roles


class Chunk(BaseModel):
    """A retrievable unit derived from a document."""

    id: str
    doc_id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    # Common metadata keys:
    # source, page, section, date, tenant_id, roles, chunk_idx


class Query(BaseModel):
    text: str
    filters: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ScoredHit(BaseModel):
    chunk: Chunk
    score: float
    source: str = ""  # bm25 | dense | rrf | rerank | ...


class Answer(BaseModel):
    answer: str
    citations: list[str] = Field(default_factory=list)
    found: bool = True
    hits: list[ScoredHit] = Field(default_factory=list)


class EvalExample(BaseModel):
    id: str
    query: str
    gold_doc_ids: list[str] = Field(default_factory=list)
    gold_chunk_ids: list[str] = Field(default_factory=list)
    gold_answer: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
