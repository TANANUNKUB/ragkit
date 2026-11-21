"""Token counters that do not download models.

ApproxTokenCounter uses characters/4 — good enough for budgets in a skeleton.
A pythainlp / tiktoken hook can replace this later without changing chunkers.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...

    def take_suffix(self, text: str, n_tokens: int) -> str: ...

    def take_prefix(self, text: str, n_tokens: int) -> str: ...


class ApproxTokenCounter:
    """~4 characters per token. Works offline and is Thai-tolerant (no whitespace needed)."""

    def __init__(self, chars_per_token: float = 4.0) -> None:
        if chars_per_token <= 0:
            raise ValueError("chars_per_token must be > 0")
        self.chars_per_token = chars_per_token

    def count(self, text: str) -> int:
        if not text:
            return 0
        return max(1, int(round(len(text) / self.chars_per_token)))

    def _chars(self, n_tokens: int) -> int:
        return max(0, int(round(n_tokens * self.chars_per_token)))

    def take_suffix(self, text: str, n_tokens: int) -> str:
        n = self._chars(n_tokens)
        return text[-n:] if n else ""

    def take_prefix(self, text: str, n_tokens: int) -> str:
        n = self._chars(n_tokens)
        return text[:n] if n else ""
