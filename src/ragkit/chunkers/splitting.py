"""Thai-safe splitting helpers. Do not use spaces as sentence boundaries."""

from __future__ import annotations

import re

# Sentence enders: Latin, CJK, Thai formal marks, ellipsis, newlines.
# We deliberately do *not* split on whitespace — Thai often has no spaces.
_SENT_END = re.compile(r"(?<=[.\?!…។！？])\s+|(?<=\n)|(?<=ฯ)\s+")
_HEADING = re.compile(r"^(#{1,6})\s+(.+)$")
_NUMBERED = re.compile(r"^(\d+(?:\.\d+){0,4})\s+(\S.+)$")


def split_sentences(text: str) -> list[str]:
    """Split on punctuation / newlines, not on spaces.

    Hook: replace this with pythainlp.tokenize.sent_tokenize for better Thai.
    """
    text = text.strip()
    if not text:
        return []
    parts = [p.strip() for p in _SENT_END.split(text) if p and p.strip()]
    return parts or [text]


def split_paragraphs(text: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def is_markdown_table(block: str) -> bool:
    lines = [ln for ln in block.splitlines() if ln.strip()]
    if len(lines) < 2:
        return False
    return sum(1 for ln in lines if "|" in ln) >= 2


def split_table_keep_header(block: str, max_rows: int) -> list[str]:
    """Split a markdown table, repeating the header (+ separator) in each piece."""
    lines = block.splitlines()
    if len(lines) <= max_rows or max_rows <= 2:
        return [block]
    header_n = 2 if len(lines) >= 2 else 1
    header, body = lines[:header_n], lines[header_n:]
    step = max(1, max_rows - header_n)
    return ["\n".join(header + body[i : i + step]) for i in range(0, len(body), step)]


def iter_sections(text: str, title: str | None = None) -> list[tuple[str, str]]:
    """Yield (section_path, body) using markdown or numbered headings."""
    prefix = [title] if title else []
    headings: list[str] = []
    buf: list[str] = []
    sections: list[tuple[str, str]] = []

    def path() -> str:
        return " > ".join(p for p in prefix + headings if p)

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            sections.append((path(), body))

    for line in text.splitlines():
        md = _HEADING.match(line)
        numbered = None if md else _NUMBERED.match(line)
        if md:
            flush()
            buf = []
            level = len(md.group(1))
            headings = headings[: level - 1] + [md.group(2).strip()]
        elif numbered and _looks_like_heading_number(numbered.group(1), numbered.group(2)):
            flush()
            buf = []
            level = numbered.group(1).count(".") + 1
            label = f"{numbered.group(1)} {numbered.group(2).strip()}"
            headings = headings[: level - 1] + [label]
        else:
            buf.append(line)
    flush()
    if not sections and text.strip():
        sections.append((title or "", text.strip()))
    return sections


def _looks_like_heading_number(num: str, rest: str) -> bool:
    # Avoid treating "2023 was a good year" or "3.14 is pi" as headings.
    if rest and rest[0].islower():
        return False
    return bool(rest) and len(rest) <= 80
