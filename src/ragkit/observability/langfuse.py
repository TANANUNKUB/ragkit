"""Optional Langfuse tracer. Imported only when observability.tracer = langfuse."""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any


class LangfuseTracer:
    def __init__(
        self,
        public_key: str | None = None,
        secret_key: str | None = None,
        host: str | None = None,
        **_: object,
    ) -> None:
        try:
            from langfuse import Langfuse
        except ImportError as exc:
            raise ImportError("LangfuseTracer requires langfuse. pip install ragkit[obs]") from exc
        self._client = Langfuse(
            public_key=public_key or os.environ.get("LANGFUSE_PUBLIC_KEY"),
            secret_key=secret_key or os.environ.get("LANGFUSE_SECRET_KEY"),
            host=host or os.environ.get("LANGFUSE_HOST"),
        )

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[dict[str, Any]]:
        data: dict[str, Any] = dict(attrs)
        t0 = time.perf_counter()
        lf_span = self._client.start_span(name=name, metadata=attrs)
        try:
            yield data
        except Exception as exc:
            data["error"] = str(exc)
            if lf_span is not None:
                lf_span.update(level="ERROR", status_message=str(exc))
            raise
        finally:
            data["duration_ms"] = (time.perf_counter() - t0) * 1000.0
            if lf_span is not None:
                lf_span.update(metadata=data)
                lf_span.end()

    def flush(self) -> None:
        self._client.flush()
