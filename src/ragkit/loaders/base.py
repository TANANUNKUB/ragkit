"""Dataset adapter interface. Each corpus implements this and maps fields into ragkit models."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from ragkit.models import Document, EvalExample


@runtime_checkable
class DatasetAdapter(Protocol):
    def load_corpus(self) -> Iterable[Document]: ...

    def load_eval(self) -> Iterable[EvalExample]: ...
