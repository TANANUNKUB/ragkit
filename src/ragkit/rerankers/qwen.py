"""Qwen3-Reranker: pointwise yes/no, score = P(yes) at the last token. Lazy import of transformers."""

from __future__ import annotations

from ragkit.models import Query, ScoredHit
from ragkit.registry import register

DEFAULT_INSTRUCTION = (
    "Given a question, retrieve the document that answers it."
)
_PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query and the "
    "Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
    "<|im_end|>\n<|im_start|>user\n"
)
_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


@register("reranker", "qwen")
class QwenReranker:
    def __init__(
        self,
        model: str = "Qwen/Qwen3-Reranker-0.6B",
        instruction: str = DEFAULT_INSTRUCTION,
        batch_size: int = 8,
        max_length: int = 4096,
        device: str | None = None,
        **_: object,
    ) -> None:
        self.model_name = model
        self.instruction = instruction
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self._tok = None
        self._model = None
        self._yes_id: int | None = None
        self._no_id: int | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "QwenReranker requires transformers and torch. pip install ragkit[llm]"
            ) from exc
        self._torch = torch
        self._tok = AutoTokenizer.from_pretrained(self.model_name, padding_side="left")
        kwargs = {"torch_dtype": torch.float16}
        model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)
        if self.device:
            model = model.to(self.device)
        elif torch.cuda.is_available():
            model = model.cuda()
        self._model = model.eval()
        self._yes_id = self._tok.convert_tokens_to_ids("yes")
        self._no_id = self._tok.convert_tokens_to_ids("no")

    def _prompt(self, query: str, doc: str) -> str:
        return (
            f"{_PREFIX}<Instruct>: {self.instruction}\n"
            f"<Query>: {query}\n<Document>: {doc}{_SUFFIX}"
        )

    def rerank(self, query: Query, hits: list[ScoredHit], top_k: int = 50) -> list[ScoredHit]:
        if not hits:
            return []
        self._load()
        torch = self._torch
        scores: list[float] = []
        with torch.no_grad():
            for start in range(0, len(hits), self.batch_size):
                batch = hits[start : start + self.batch_size]
                texts = [self._prompt(query.text, h.chunk.text) for h in batch]
                encoded = self._tok(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self._model.device) for k, v in encoded.items()}
                logits = self._model(**encoded).logits[:, -1, :]
                pair = torch.stack([logits[:, self._no_id], logits[:, self._yes_id]], dim=1)
                probs = torch.softmax(pair, dim=1)[:, 1]
                scores.extend(float(x) for x in probs.detach().cpu().tolist())
        reranked = [
            ScoredHit(chunk=h.chunk, score=s, source="rerank") for h, s in zip(hits, scores)
        ]
        reranked.sort(key=lambda h: -h.score)
        return reranked[:top_k]
