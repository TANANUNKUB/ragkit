from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ragkit.models import Chunk


@runtime_checkable
class VectorStore(Protocol):
    def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None: ...

    def search(
        self,
        vector: np.ndarray,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[Chunk, float]]: ...

    def clear(self) -> None: ...
