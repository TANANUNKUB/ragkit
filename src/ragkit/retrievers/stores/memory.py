"""In-memory cosine-similarity store. No extra deps beyond numpy. Fine for tests and tiny corpora."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ragkit.models import Chunk
from ragkit.retrievers.base import matches_filters


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return vectors / norms


class MemoryVectorStore:
    def __init__(self, **_: object) -> None:
        self._ids: list[str] = []
        self._chunks: dict[str, Chunk] = {}
        self._vectors: np.ndarray | None = None

    def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        vectors = _l2_normalize(np.asarray(vectors, dtype=np.float32))
        if len(chunks) != vectors.shape[0]:
            raise ValueError("chunks and vectors must have the same length")
        incoming = {c.id: (c, vectors[i]) for i, c in enumerate(chunks)}
        # Rebuild so updates replace existing ids.
        existing_ids = [i for i in self._ids if i not in incoming]
        new_ids = list(incoming.keys())
        if self._vectors is None or not existing_ids:
            kept = np.zeros((0, vectors.shape[1]), dtype=np.float32)
        else:
            index = {cid: n for n, cid in enumerate(self._ids)}
            kept = self._vectors[[index[i] for i in existing_ids]]
        added = np.stack([incoming[i][1] for i in new_ids], axis=0)
        self._ids = existing_ids + new_ids
        self._vectors = np.vstack([kept, added]) if len(existing_ids) else added
        for cid, (chunk, _) in incoming.items():
            self._chunks[cid] = chunk
        # Drop chunks that were replaced — already overwritten. Nothing to evict.

    def search(
        self,
        vector: np.ndarray,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[Chunk, float]]:
        if self._vectors is None or not self._ids:
            return []
        q = _l2_normalize(np.asarray(vector, dtype=np.float32).reshape(1, -1))[0]
        scores = self._vectors @ q
        # Filter first so unauthorized chunks never leave the store.
        eligible = [
            i
            for i, cid in enumerate(self._ids)
            if matches_filters(self._chunks[cid].metadata, filters)
        ]
        if not eligible:
            return []
        elig_scores = scores[eligible]
        k = min(top_k, len(eligible))
        # argpartition then sort the top-k
        part = np.argpartition(-elig_scores, kth=k - 1)[:k]
        ranked = part[np.argsort(-elig_scores[part])]
        out: list[tuple[Chunk, float]] = []
        for local_idx in ranked:
            i = eligible[int(local_idx)]
            out.append((self._chunks[self._ids[i]], float(scores[i])))
        return out

    def clear(self) -> None:
        self._ids = []
        self._chunks = {}
        self._vectors = None
