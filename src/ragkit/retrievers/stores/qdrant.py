"""Qdrant vector store. Imported lazily; not required for tests."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ragkit.models import Chunk


class QdrantVectorStore:
    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection: str = "ragkit",
        api_key: str | None = None,
        vector_size: int | None = None,
        **_: object,
    ) -> None:
        self.url = url
        self.collection = collection
        self.api_key = api_key
        self.vector_size = vector_size
        self._client = None

    def _connect(self):
        if self._client is not None:
            return self._client
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise ImportError("QdrantVectorStore requires qdrant-client. pip install ragkit[qdrant]") from exc
        self._client = QdrantClient(url=self.url, api_key=self.api_key)
        return self._client

    def _ensure_collection(self, dim: int) -> None:
        from qdrant_client.http import models as qm

        client = self._connect()
        existing = {c.name for c in client.get_collections().collections}
        if self.collection not in existing:
            client.create_collection(
                collection_name=self.collection,
                vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
            )

    def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        from qdrant_client.http import models as qm

        vectors = np.asarray(vectors, dtype=np.float32)
        self._ensure_collection(vectors.shape[1])
        points = [
            qm.PointStruct(
                id=_stable_id(chunk.id),
                vector=vectors[i].tolist(),
                payload={"chunk": chunk.model_dump(), "doc_id": chunk.doc_id, **chunk.metadata},
            )
            for i, chunk in enumerate(chunks)
        ]
        self._connect().upsert(collection_name=self.collection, points=points)

    def search(
        self,
        vector: np.ndarray,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[Chunk, float]]:
        from qdrant_client.http import models as qm

        qfilter = _to_qdrant_filter(filters) if filters else None
        hits = self._connect().search(
            collection_name=self.collection,
            query_vector=np.asarray(vector, dtype=np.float32).reshape(-1).tolist(),
            limit=top_k,
            query_filter=qfilter,
        )
        out: list[tuple[Chunk, float]] = []
        for hit in hits:
            payload = hit.payload or {}
            raw = payload.get("chunk") or payload
            chunk = Chunk.model_validate(raw) if not isinstance(raw, Chunk) else raw
            out.append((chunk, float(hit.score)))
        return out

    def clear(self) -> None:
        client = self._connect()
        existing = {c.name for c in client.get_collections().collections}
        if self.collection in existing:
            client.delete_collection(self.collection)


def _stable_id(chunk_id: str) -> str:
    # Qdrant accepts UUID or unsigned int. Hash the string id to a UUID5-like hex.
    import hashlib
    import uuid

    digest = hashlib.sha1(chunk_id.encode("utf-8")).hexdigest()
    return str(uuid.UUID(digest[:32]))


def _to_qdrant_filter(filters: dict[str, Any]):
    from qdrant_client.http import models as qm

    must = []
    for key, value in filters.items():
        if key == "roles":
            wanted = list(value if isinstance(value, (list, tuple, set)) else [value])
            # Match if payload.roles contains * or any requested role.
            must.append(
                qm.Filter(
                    should=[
                        qm.FieldCondition(key="roles", match=qm.MatchValue(value="*")),
                        *[qm.FieldCondition(key="roles", match=qm.MatchValue(value=v)) for v in wanted],
                    ]
                )
            )
            continue
        if isinstance(value, (list, tuple, set)):
            must.append(qm.FieldCondition(key=key, match=qm.MatchAny(any=list(value))))
        else:
            must.append(qm.FieldCondition(key=key, match=qm.MatchValue(value=value)))
    return qm.Filter(must=must) if must else None
