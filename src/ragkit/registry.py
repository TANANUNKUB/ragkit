"""Simple name → class registry used to wire stages from YAML config."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")

_REGISTRY: dict[str, dict[str, type]] = {}


def register(kind: str, name: str) -> Callable[[type[T]], type[T]]:
    def deco(cls: type[T]) -> type[T]:
        _REGISTRY.setdefault(kind, {})[name] = cls
        return cls

    return deco


def get(kind: str, name: str) -> type:
    try:
        return _REGISTRY[kind][name]
    except KeyError as exc:
        available = list(_REGISTRY.get(kind, {}))
        raise KeyError(
            f"Unknown {kind} {name!r}. Registered: {available}. "
            "Did you forget to import the implementation module?"
        ) from exc


def create(kind: str, name: str, **kwargs: Any) -> Any:
    return get(kind, name)(**kwargs)


def registered(kind: str) -> list[str]:
    return sorted(_REGISTRY.get(kind, {}))


def ensure_registered() -> None:
    """Import implementation modules so @register decorators run."""
    # Local imports to avoid circulars and keep heavy deps lazy inside modules.
    from ragkit.chunkers import fixed as _fixed  # noqa: F401
    from ragkit.chunkers import structure as _structure  # noqa: F401
    from ragkit.generators import fake as _fake  # noqa: F401
    from ragkit.generators import openai_compat as _oai  # noqa: F401
    from ragkit.loaders import financebench as _fb  # noqa: F401
    from ragkit.loaders import folder as _folder  # noqa: F401
    from ragkit.loaders import litsearch as _ls  # noqa: F401
    from ragkit.loaders import nitibench as _nb  # noqa: F401
    from ragkit.parsers import pdf as _pdf  # noqa: F401
    from ragkit.parsers import text as _text  # noqa: F401
    from ragkit.rerankers import cross_encoder as _ce  # noqa: F401
    from ragkit.rerankers import none as _none  # noqa: F401
    from ragkit.rerankers import qwen as _qwen  # noqa: F401
    from ragkit.retrievers import bm25 as _bm25  # noqa: F401
    from ragkit.retrievers import dense as _dense  # noqa: F401
    from ragkit.retrievers import hybrid as _hybrid  # noqa: F401
