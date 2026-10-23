"""ParsedResponse: one shape for non-streaming JSON and assembled SSE streams."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from .client import Exchange

RAW_LIMIT_BYTES = 200_000
REASONING_FIELDS = ("reasoning_content", "reasoning")


@dataclass
class ParsedResponse:
    status: int | None
    headers: dict[str, str]
    content: str | None = None
    reasoning: str | None = None
    reasoning_field: str | None = None  # "reasoning_content", "reasoning" or None
    reasoning_fields_seen: list[str] = field(default_factory=list)  # every field that carried text
    tool_calls: list[dict[str, Any]] = field(default_factory=list)  # {index, id, name, arguments_raw}
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    refusal: str | None = None
    n_chunks: int = 0
    ttft_s: float | None = None
    total_s: float = 0.0
    stream: bool = False
    sse: bool = False  # the body really was an SSE stream
    done_marker: bool = False
    error: str | None = None  # transport error from the client
    error_message: str | None = None  # API error text for non-2xx or an in-stream error event
    raw: Any = None

    def summary(self) -> dict[str, Any]:
        """Everything except `raw`, for storage next to it."""
        d = asdict(self)
        d.pop("raw")
        return d

    def _usage_int(self, key: str) -> int | None:
        value = (self.usage or {}).get(key)
        return value if isinstance(value, int) else None

    @property
    def prompt_tokens(self) -> int | None:
        return self._usage_int("prompt_tokens")

    @property
    def completion_tokens(self) -> int | None:
        return self._usage_int("completion_tokens")


def parse_exchange(ex: Exchange) -> ParsedResponse:
    p = ParsedResponse(
        status=ex.status, headers=ex.headers, ttft_s=ex.ttft_s, total_s=ex.total_s, stream=ex.stream, error=ex.error
    )
    body = ex.body
    if isinstance(body, list):
        p.sse = True
        _assemble_stream(p, body)
    elif isinstance(body, dict):
        if (ex.status or 0) >= 400 or ("error" in body and "choices" not in body):
            p.error_message = _error_message(body)
        else:
            _parse_completion(p, body)
    elif isinstance(body, str) and body and (ex.status or 0) >= 400:
        p.error_message = body[:2000]
    p.raw = truncate_raw(body)
    return p


def parse_completion_json(body: dict[str, Any], status: int = 200) -> ParsedResponse:
    """Parse a non-streaming completion body (used for responses captured outside ApiClient)."""
    return parse_exchange(Exchange(status, {}, body, False))


def _error_message(body: dict[str, Any]) -> str:
    err = body.get("error", body)
    if isinstance(err, dict):
        msg = err.get("message") or err.get("detail") or json.dumps(err, ensure_ascii=False)
    else:
        msg = str(err)
    if not isinstance(msg, str):
        msg = json.dumps(msg, ensure_ascii=False)
    return msg[:2000]


def _set_reasoning(p: ParsedResponse, message: dict[str, Any]) -> None:
    seen = [f for f in REASONING_FIELDS if message.get(f)]
    p.reasoning_fields_seen = seen
    if seen:
        p.reasoning_field = seen[0]
        p.reasoning = message[seen[0]]


def _parse_completion(p: ParsedResponse, body: dict[str, Any]) -> None:
    choices = body.get("choices") or []
    if choices:
        choice = choices[0]
        message = choice.get("message") or {}
        p.content = message.get("content")
        _set_reasoning(p, message)
        p.refusal = message.get("refusal")
        p.finish_reason = choice.get("finish_reason")
        for i, tc in enumerate(message.get("tool_calls") or []):
            fn = tc.get("function") or {}
            p.tool_calls.append(
                {"index": tc.get("index", i), "id": tc.get("id"), "name": fn.get("name"), "arguments_raw": fn.get("arguments")}
            )
    p.usage = body.get("usage")


def _assemble_stream(p: ParsedResponse, payloads: list[Any]) -> None:
    """Concatenate content and reasoning deltas; merge tool_call deltas by index (CLAUDE.md §7, T4)."""
    content: list[str] = []
    reasoning: list[str] = []
    refusal: list[str] = []
    seen: list[str] = []
    tools: dict[int, dict[str, Any]] = {}
    for item in payloads:
        if item == "[DONE]":
            p.done_marker = True
            continue
        if not isinstance(item, dict):
            continue
        p.n_chunks += 1
        if item.get("error"):
            p.error_message = _error_message(item)
        if item.get("usage"):
            p.usage = item["usage"]
        for choice in item.get("choices") or []:
            if choice.get("index", 0) != 0:
                continue
            delta = choice.get("delta") or {}
            if delta.get("content"):
                content.append(delta["content"])
            for f in REASONING_FIELDS:
                if delta.get(f) and f not in seen:
                    seen.append(f)
            # Some servers mirror the same text into both fields; take one per chunk.
            text = delta.get("reasoning_content") or delta.get("reasoning")
            if text:
                reasoning.append(text)
            if delta.get("refusal"):
                refusal.append(delta["refusal"])
            for tc in delta.get("tool_calls") or []:
                idx = tc.get("index", 0)
                slot = tools.setdefault(idx, {"index": idx, "id": None, "name": None, "arguments_raw": ""})
                if tc.get("id") and not slot["id"]:
                    slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name") and not slot["name"]:
                    slot["name"] = fn["name"]
                args = fn.get("arguments")
                if isinstance(args, str):
                    slot["arguments_raw"] += args
                elif args is not None:
                    slot["arguments_raw"] += json.dumps(args, ensure_ascii=False)
            if choice.get("finish_reason"):
                p.finish_reason = choice["finish_reason"]
    p.content = "".join(content)
    p.reasoning = "".join(reasoning) or None
    p.reasoning_fields_seen = seen
    p.reasoning_field = seen[0] if seen else None
    p.refusal = "".join(refusal) or None
    p.tool_calls = [tools[k] for k in sorted(tools)]


def truncate_raw(body: Any) -> Any:
    """Cap stored raw data at RAW_LIMIT_BYTES. Parsing always happens on the full body first."""
    text = json.dumps(body, ensure_ascii=False, default=str)
    if len(text.encode()) <= RAW_LIMIT_BYTES:
        return body
    if isinstance(body, list):
        kept: list[Any] = []
        size = 2
        for item in body:
            n = len(json.dumps(item, ensure_ascii=False, default=str).encode()) + 1
            if size + n > RAW_LIMIT_BYTES - 200:
                break
            kept.append(item)
            size += n
        kept.append({"_truncated": True, "dropped_items": len(body) - len(kept)})
        return kept
    return {"_truncated": True, "text": text.encode()[:RAW_LIMIT_BYTES].decode("utf-8", "ignore")}
