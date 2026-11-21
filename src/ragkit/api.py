"""FastAPI app: POST /search and POST /answer. Optional extra: pip install ragkit[api]."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from ragkit.models import Answer, Query, ScoredHit
from ragkit.pipeline import RAGPipeline

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover
    raise ImportError("API requires FastAPI. pip install ragkit[api]") from exc


class SearchRequest(BaseModel):
    query: str
    filters: dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    hits: list[ScoredHit]


class AnswerRequest(SearchRequest):
    pass


@lru_cache(maxsize=1)
def get_pipeline() -> RAGPipeline:
    path = os.environ.get("RAGKIT_CONFIG", "configs/base.yaml")
    pipe = RAGPipeline.from_config(path)
    # Index is expected to have been built via the CLI (`ragkit index`).
    # If chunks were already indexed in-process, this is a no-op for stores that persist.
    return pipe


def create_app() -> FastAPI:
    app = FastAPI(title="ragkit", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/search", response_model=SearchResponse)
    def search(req: SearchRequest) -> SearchResponse:
        try:
            hits = get_pipeline().search(Query(text=req.query, filters=req.filters))
        except Exception as exc:  # GuardError and missing index surface as 400/500
            from ragkit.security.guard import GuardError

            if isinstance(exc, GuardError):
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            raise
        return SearchResponse(hits=hits)

    @app.post("/answer", response_model=Answer)
    def answer(req: AnswerRequest) -> Answer:
        try:
            return get_pipeline().answer(Query(text=req.query, filters=req.filters))
        except Exception as exc:
            from ragkit.security.guard import GuardError

            if isinstance(exc, GuardError):
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            raise

    return app


app = create_app()
