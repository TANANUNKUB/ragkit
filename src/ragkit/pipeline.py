"""RAGPipeline: wire stages from config (or construct by hand). No LangChain."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from ragkit.chunkers.base import Chunker
from ragkit.config import load_config
from ragkit.generators.base import Generator
from ragkit.loaders.base import DatasetAdapter
from ragkit.models import Answer, Chunk, Document, Query, ScoredHit
from ragkit.observability.tracing import timed_span
from ragkit.parsers.base import Parser
from ragkit.registry import create, ensure_registered
from ragkit.rerankers.base import Reranker
from ragkit.retrievers.aggregate import aggregate_max_per_doc
from ragkit.retrievers.base import Retriever
from ragkit.security.acl import apply_default_filters
from ragkit.security.guard import InputGuard


def _build_tracer(cfg: dict[str, Any]):
    name = (cfg.get("observability") or {}).get("tracer", "noop")
    if name in {"noop", "none", None}:
        from ragkit.observability.tracing import NoOpTracer

        return NoOpTracer()
    if name == "langfuse":
        from ragkit.observability.langfuse import LangfuseTracer

        opts = dict((cfg.get("observability") or {}).get("langfuse") or {})
        return LangfuseTracer(**opts)
    raise KeyError(f"Unknown tracer {name!r}")


class RAGPipeline:
    def __init__(
        self,
        *,
        loader: DatasetAdapter | None = None,
        parser: Parser,
        chunker: Chunker,
        retriever: Retriever,
        reranker: Reranker,
        generator: Generator,
        guard: InputGuard | None = None,
        abstain_threshold: float | None = None,
        retrieve_top_k: int = 100,
        rerank_top_k: int = 50,
        generate_top_k: int = 5,
        aggregate: str = "chunk",
        default_filters: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.loader = loader
        self.parser = parser
        self.chunker = chunker
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator
        self.guard = guard or InputGuard()
        self.abstain_threshold = abstain_threshold
        self.retrieve_top_k = retrieve_top_k
        self.rerank_top_k = rerank_top_k
        self.generate_top_k = generate_top_k
        self.aggregate = aggregate
        self.default_filters = default_filters or {}
        self.config = config or {}
        self._indexed = False
        self.chunks: list[Chunk] = []

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | str) -> RAGPipeline:
        if isinstance(cfg, str):
            cfg = load_config(cfg)
        ensure_registered()
        ds = cfg.get("dataset") or {}
        loader = create("loader", ds.get("adapter", "folder"), **{k: v for k, v in ds.items() if k != "adapter"})
        parser = create("parser", (cfg.get("parser") or {}).get("name", "text"), **{k: v for k, v in (cfg.get("parser") or {}).items() if k != "name"})
        chunker_cfg = dict(cfg.get("chunker") or {})
        chunker = create("chunker", chunker_cfg.pop("name", "structure"), **chunker_cfg)
        retr_cfg = dict(cfg.get("retriever") or {})
        retr_name = retr_cfg.pop("name", "hybrid")
        top_k = int(retr_cfg.pop("top_k", 100))
        aggregate = retr_cfg.pop("aggregate", "chunk")
        retriever = create("retriever", retr_name, **retr_cfg)
        rr_cfg = dict(cfg.get("reranker") or {})
        rerank_top_k = int(rr_cfg.pop("top_k", 50))
        reranker = create("reranker", rr_cfg.pop("name", "none"), **rr_cfg)
        gen_cfg = dict(cfg.get("generator") or {})
        generate_top_k = int(gen_cfg.pop("top_k", 5))
        abstain_threshold = gen_cfg.pop("abstain_threshold", None)
        generator = create("generator", gen_cfg.pop("name", "fake"), **gen_cfg)
        guard = InputGuard(**(cfg.get("security") or {}))
        tracer = _build_tracer(cfg)
        from ragkit.observability.tracing import set_tracer

        set_tracer(tracer)
        return cls(
            loader=loader,
            parser=parser,
            chunker=chunker,
            retriever=retriever,
            reranker=reranker,
            generator=generator,
            guard=guard,
            abstain_threshold=abstain_threshold,
            retrieve_top_k=top_k,
            rerank_top_k=rerank_top_k,
            generate_top_k=generate_top_k,
            aggregate=aggregate,
            default_filters=(cfg.get("security") or {}).get("default_filters") or {},
            config=cfg,
        )

    def ingest(self, documents: Iterable[Document] | None = None) -> list[Chunk]:
        with timed_span("ingest"):
            docs = list(documents) if documents is not None else list(self.loader.load_corpus())
            chunks: list[Chunk] = []
            for doc in docs:
                parsed = self.parser.parse(doc)
                chunks.extend(self.chunker.chunk(parsed))
            self.chunks = chunks
            return chunks

    def index(self, chunks: Sequence[Chunk] | None = None) -> None:
        with timed_span("index"):
            payload = list(chunks if chunks is not None else self.chunks)
            self.chunks = payload
            self.retriever.index(payload)
            self._indexed = True

    def search(self, query: Query | str, filters: dict[str, Any] | None = None) -> list[ScoredHit]:
        q = self._prepare_query(query, filters)
        with timed_span("search", query=q.text):
            with timed_span("retrieve"):
                hits = self.retriever.retrieve(q, top_k=self.retrieve_top_k)
            if self.aggregate in {"doc", "paper", "document"}:
                # Idempotent if the retriever already aggregated.
                hits = aggregate_max_per_doc(hits)
            with timed_span("rerank"):
                hits = self.reranker.rerank(q, hits, top_k=self.rerank_top_k)
            return hits

    def answer(self, query: Query | str, filters: dict[str, Any] | None = None) -> Answer:
        q = self._prepare_query(query, filters)
        with timed_span("answer", query=q.text):
            hits = self.search(q)
            if self._should_abstain(hits):
                return Answer(
                    answer="ไม่พบข้อมูลที่เกี่ยวข้องเพียงพอในคลังเอกสาร",
                    citations=[],
                    found=False,
                    hits=hits,
                )
            context_hits = hits[: self.generate_top_k]
            with timed_span("generate"):
                return self.generator.generate(q, context_hits)

    def _should_abstain(self, hits: list[ScoredHit]) -> bool:
        if not hits:
            return True
        if self.abstain_threshold is None:
            return False
        return hits[0].score < float(self.abstain_threshold)

    def _prepare_query(self, query: Query | str, filters: dict[str, Any] | None) -> Query:
        q = query if isinstance(query, Query) else Query(text=query)
        if filters:
            q = q.model_copy(update={"filters": {**q.filters, **filters}})
        q = apply_default_filters(q, self.default_filters)
        return self.guard.check(q)
