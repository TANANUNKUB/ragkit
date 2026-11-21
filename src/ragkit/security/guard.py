"""Input guard: length limit + basic prompt-injection pattern flags."""

from __future__ import annotations

import re

from ragkit.models import Query

# Patterns are conservative flags, not a complete defense.
_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|above|prior)\s+instructions", re.I),
    re.compile(r"ignore\s+the\s+(system|previous)\s+prompt", re.I),
    re.compile(r"you\s+are\s+now\s+", re.I),
    re.compile(r"disregard\s+(all\s+)?(previous|above)", re.I),
    re.compile(r"system\s*prompt\s*:", re.I),
    re.compile(r"<\|im_start\|>", re.I),
    re.compile(r"\[INST\]", re.I),
    re.compile(r"jailbreak", re.I),
    re.compile(r"เพิกเฉย(คำสั่ง|คำขอ)?ก่อนหน้า"),
    re.compile(r"ลืมคำสั่งก่อนหน้า"),
]


class GuardError(ValueError):
    """Raised when a query is rejected by the input guard."""


class InputGuard:
    def __init__(
        self,
        max_query_length: int = 2000,
        enable_injection_guard: bool = True,
        reject_injection: bool = True,
        **_: object,
    ) -> None:
        self.max_query_length = max_query_length
        self.enable_injection_guard = enable_injection_guard
        self.reject_injection = reject_injection

    def check(self, query: Query) -> Query:
        text = query.text or ""
        if len(text) > self.max_query_length:
            raise GuardError(
                f"Query exceeds max_query_length={self.max_query_length} ({len(text)} chars)."
            )
        flags = self.flag_injection(text) if self.enable_injection_guard else []
        if flags and self.reject_injection:
            raise GuardError(f"Query flagged as possible prompt injection: {flags}")
        metadata = dict(query.metadata)
        if flags:
            metadata["injection_flags"] = flags
        return query.model_copy(update={"metadata": metadata})

    @staticmethod
    def flag_injection(text: str) -> list[str]:
        hits: list[str] = []
        for pat in _INJECTION_PATTERNS:
            if pat.search(text):
                hits.append(pat.pattern)
        return hits
