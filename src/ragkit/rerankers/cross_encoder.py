"""Cross-encoder reranker (e.g. BAAI/bge-reranker-v2-m3) via sentence-transformers. Lazy import."""

from __future__ import annotations

from ragkit.models import Query, ScoredHit
from ragkit.registry import register


@register("reranker", "cross_encoder")
class CrossEncoderReranker:
    def __init__(
        self,
        model: str = "BAAI/bge-reranker-v2-m3",
        batch_size: int = 16,
        device: str | None = None,
        **_: object,
    ) -> None:
        self.model_name = model
        self.batch_size = batch_size
        self.device = device
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "CrossEncoderReranker requires sentence-transformers. pip install ragkit[dense]"
            ) from exc
        self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def rerank(self, query: Query, hits: list[ScoredHit], top_k: int = 50) -> list[ScoredHit]:
        if not hits:
            return []
        model = self._load()
        pairs = [(query.text, h.chunk.text) for h in hits]
        scores = model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        reranked = [
            ScoredHit(chunk=h.chunk, score=float(s), source="rerank")
            for h, s in zip(hits, scores)
        ]
        reranked.sort(key=lambda h: -h.score)
        return reranked[:top_k]
