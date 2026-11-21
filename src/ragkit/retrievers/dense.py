"""Dense retriever: embed query + search a vector store. Hash embedder works offline."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ragkit.models import Chunk, Query, ScoredHit
from ragkit.registry import register
from ragkit.retrievers.embedders import build_embedder
from ragkit.retrievers.stores.memory import MemoryVectorStore


def build_store(name: str, **kwargs: Any):
    if name in {"memory", "numpy", "inmemory"}:
        return MemoryVectorStore(**kwargs)
    if name in {"qdrant"}:
        from ragkit.retrievers.stores.qdrant import QdrantVectorStore

        return QdrantVectorStore(**kwargs)
    raise KeyError(f"Unknown vector store {name!r}. Use 'memory' or 'qdrant'.")


@register("retriever", "dense")
class DenseRetriever:
    def __init__(
        self,
        embedder: str = "hash",
        store: str = "memory",
        model: str = "BAAI/bge-m3",
        dim: int = 64,
        qdrant: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        embedder_kwargs = {"model": model, "dim": dim}
        embedder_kwargs.update({k: v for k, v in kwargs.items() if k in {"device", "ngram"}})
        self.embedder = build_embedder(embedder, **embedder_kwargs)
        store_kwargs = dict(qdrant or {})
        self.store = build_store(store, **store_kwargs)
        self._indexed = False

    def index(self, chunks: Sequence[Chunk]) -> None:
        texts = [c.text for c in chunks]
        vectors = self.embedder.embed(texts)
        self.store.upsert(list(chunks), vectors)
        self._indexed = True

    def retrieve(self, query: Query, top_k: int = 100) -> list[ScoredHit]:
        if not self._indexed:
            return []
        vector = self.embedder.embed([query.text])[0]
        pairs = self.store.search(vector, top_k=top_k, filters=query.filters)
        return [ScoredHit(chunk=chunk, score=score, source="dense") for chunk, score in pairs]
