"""Tracing wrapper: no-op by default. Optional Langfuse via env / config. Decorator times each stage."""

from __future__ import annotations

import functools
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Protocol, runtime_checkable

_tracer: ContextVar["Tracer"] = ContextVar("ragkit_tracer")
_default: "Tracer | None" = None


@runtime_checkable
class Tracer(Protocol):
    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[dict[str, Any]]: ...


class NoOpTracer:
    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[dict[str, Any]]:
        _ = name
        data: dict[str, Any] = dict(attrs)
        t0 = time.perf_counter()
        try:
            yield data
        finally:
            data["duration_ms"] = (time.perf_counter() - t0) * 1000.0


def set_tracer(tracer: Tracer) -> None:
    global _default
    _default = tracer
    _tracer.set(tracer)


def get_tracer() -> Tracer:
    try:
        return _tracer.get()
    except LookupError:
        return _default or NoOpTracer()


@contextmanager
def timed_span(name: str, **attrs: Any) -> Iterator[dict[str, Any]]:
    with get_tracer().span(name, **attrs) as data:
        yield data


def timed(name: str | None = None) -> Callable:
    """Decorator that wraps a function in a tracer span named `name` (or the function name)."""

    def deco(fn: Callable) -> Callable:
        span_name = name or fn.__name__

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with timed_span(span_name):
                return fn(*args, **kwargs)

        return wrapper

    return deco
