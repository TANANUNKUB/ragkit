"""Embedders. HashEmbedder is deterministic and offline (tests / smoke). ST is lazy."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class HashEmbedder:
    """Deterministic character-ngram hashed bag vectors. Not a production encoder."""

    def __init__(self, dim: int = 64, ngram: int = 3, **_: object) -> None:
        self.dim = dim
        self.ngram = ngram

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            padded = f" {(text or '').lower()} "
            for j in range(max(1, len(padded) - self.ngram + 1)):
                gram = padded[j : j + self.ngram]
                # Use a stable hash so index and query agree within one process.
                idx = _stable_hash(gram) % self.dim
                sign = 1.0 if _stable_hash(gram + "+") % 2 == 0 else -1.0
                out[i, idx] += sign
            norm = float(np.linalg.norm(out[i]))
            if norm > 0:
                out[i] /= norm
        return out


def _stable_hash(s: str) -> int:
    # Prefer a stable non-salted hash (Python's hash() is randomized).
    import hashlib

    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)


class SentenceTransformerEmbedder:
    def __init__(self, model: str = "BAAI/bge-m3", device: str | None = None, **_: object) -> None:
        self.model_name = model
        self.device = device
        self._model = None
        self.dim = -1

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "SentenceTransformerEmbedder requires sentence-transformers. "
                "pip install ragkit[dense]"
            ) from exc
        self._model = SentenceTransformer(self.model_name, device=self.device)
        self.dim = int(self._model.get_sentence_embedding_dimension())
        return self._model

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        model = self._load()
        vectors = model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float32)


def build_embedder(name: str, **kwargs: object) -> HashEmbedder | SentenceTransformerEmbedder:
    if name in {"hash", "hash_embedder", "debug"}:
        return HashEmbedder(**kwargs)
    if name in {"sentence_transformers", "st", "dense"}:
        return SentenceTransformerEmbedder(**kwargs)
    raise KeyError(f"Unknown embedder {name!r}. Use 'hash' or 'sentence_transformers'.")
