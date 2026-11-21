"""OpenAI-compatible chat client (Ollama, vLLM, OpenAI, any /v1/chat/completions). Stdlib only."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

from pydantic import ValidationError

from ragkit.generators.base import (
    GROUNDING_INSTRUCTIONS,
    constrain_citations,
    format_context,
)
from ragkit.models import Answer, Query, ScoredHit
from ragkit.registry import register

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def _chat_completions(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    timeout: float,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        # Some servers reject response_format; retry without it.
        if exc.code in {400, 422} and "response_format" in payload:
            payload.pop("response_format")
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        else:
            raise
    return str(body["choices"][0]["message"]["content"])


def _parse_answer_json(raw: str) -> dict:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = _JSON_BLOCK.search(raw)
        if not match:
            raise
        return json.loads(match.group(0))


@register("generator", "openai_compat")
class OpenAICompatGenerator:
    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        model: str = "qwen2.5",
        api_key: str | None = None,
        temperature: float = 0.0,
        timeout: float = 120.0,
        top_k: int = 5,
        **_: object,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self.api_key = api_key or os.environ.get("RAGKIT_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
        self.temperature = temperature
        self.timeout = timeout
        self.top_k = top_k

    def generate(self, query: Query, hits: list[ScoredHit]) -> Answer:
        context_hits = hits[: self.top_k]
        if not context_hits:
            return Answer(answer="ไม่พบข้อมูลที่เกี่ยวข้อง", citations=[], found=False, hits=hits)
        context, allowed = format_context(context_hits)
        user = f"Question:\n{query.text}\n\nDocuments:\n{context}"
        messages = [
            {"role": "system", "content": GROUNDING_INSTRUCTIONS},
            {"role": "user", "content": user},
        ]
        raw = _chat_completions(
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            timeout=self.timeout,
        )
        try:
            data = _parse_answer_json(raw)
            parsed = Answer(
                answer=str(data.get("answer", "")),
                citations=list(data.get("citations") or []),
                found=bool(data.get("found", True)),
                hits=hits,
            )
        except (json.JSONDecodeError, ValidationError, TypeError, ValueError):
            # Retry once with a stricter reminder.
            messages.append({"role": "assistant", "content": raw})
            messages.append(
                {
                    "role": "user",
                    "content": 'Reply with valid JSON only: {"answer": "...", "citations": [], "found": false}',
                }
            )
            raw = _chat_completions(
                base_url=self.base_url,
                api_key=self.api_key,
                model=self.model,
                messages=messages,
                temperature=0.0,
                timeout=self.timeout,
            )
            data = _parse_answer_json(raw)
            parsed = Answer(
                answer=str(data.get("answer", "")),
                citations=list(data.get("citations") or []),
                found=bool(data.get("found", True)),
                hits=hits,
            )
        parsed.citations = constrain_citations(parsed.citations, allowed)
        parsed.hits = hits
        return parsed
