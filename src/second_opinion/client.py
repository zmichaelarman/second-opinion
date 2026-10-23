"""Async HTTP client for Token Factory and the NVIDIA reference API.

Measured calls use raw httpx so the exact JSON and headers are captured (the OpenAI SDK
hides unknown fields). The auth header is built per request and never stored: callers
keep only the request body, and `keep_headers` only ever sees response headers.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx

from .config import Settings, api_key

_KEEP_HEADER = re.compile(r"^(x-ratelimit-.+|retry-after|.*request-id|x-trace-id)$", re.I)


def keep_headers(headers: httpx.Headers) -> dict[str, str]:
    """The rate-limit subset, Retry-After and any request-id header."""
    return {k.lower(): v for k, v in headers.items() if _KEEP_HEADER.match(k)}


@dataclass
class Exchange:
    """One HTTP attempt before parsing.

    `body` is the full response JSON (or text), or for SSE the list of `data:` payloads
    (each parsed JSON, or the raw string, with "[DONE]" kept as a marker). It is never
    truncated here; `parse.truncate_raw` does that for storage.
    """

    status: int | None
    headers: dict[str, str]
    body: Any
    stream: bool
    error: str | None = None  # transport-level: TIMEOUT, or the httpx exception name
    ttft_s: float | None = None
    total_s: float = 0.0
    content_type: str | None = None


def _json_or_text(text: str) -> Any:
    try:
        return json.loads(text)
    except ValueError:
        return text


def _has_delta(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    for choice in payload.get("choices") or []:
        delta = choice.get("delta") or {}
        if any(delta.get(k) for k in ("content", "reasoning_content", "reasoning", "tool_calls", "refusal")):
            return True
    return False


class ApiClient:
    def __init__(
        self,
        base_url: str,
        api_key_env: str,
        connect_timeout_s: float = 10,
        read_timeout_s: float = 600,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.base_url = base_url
        self.api_key_env = api_key_env
        self._http = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(connect=connect_timeout_s, read=read_timeout_s, write=30.0, pool=None),
            transport=transport,
        )

    @classmethod
    def for_provider(cls, settings: Settings, name: str, transport: httpx.AsyncBaseTransport | None = None) -> "ApiClient":
        p, d = settings.provider(name), settings.defaults
        return cls(p["base_url"], p["api_key_env"], d["connect_timeout_s"], d["read_timeout_s"], transport)

    async def __aenter__(self) -> "ApiClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._http.aclose()

    def _auth(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {api_key(self.api_key_env)}"}

    async def get(self, path: str) -> Exchange:
        t0 = time.monotonic()
        try:
            resp = await self._http.get(path, headers=self._auth())
        except httpx.TimeoutException:
            return Exchange(None, {}, None, False, "TIMEOUT", total_s=time.monotonic() - t0)
        except httpx.TransportError as e:
            return Exchange(None, {}, None, False, type(e).__name__, total_s=time.monotonic() - t0)
        return Exchange(
            resp.status_code,
            keep_headers(resp.headers),
            _json_or_text(resp.text),
            False,
            total_s=time.monotonic() - t0,
            content_type=resp.headers.get("content-type"),
        )

    async def chat(self, body: dict[str, Any]) -> Exchange:
        """POST chat/completions. Streams (SSE) when body["stream"] is true."""
        stream = bool(body.get("stream"))
        t0 = time.monotonic()
        status: int | None = None
        headers: dict[str, str] = {}
        content_type: str | None = None
        payloads: list[Any] = []
        ttft: float | None = None
        try:
            async with self._http.stream("POST", "chat/completions", json=body, headers=self._auth()) as resp:
                status, headers = resp.status_code, keep_headers(resp.headers)
                content_type = resp.headers.get("content-type")
                if stream and status == 200 and "text/event-stream" in (content_type or ""):
                    async for payload in _iter_sse(resp):
                        payloads.append(payload)
                        if ttft is None and _has_delta(payload):
                            ttft = time.monotonic() - t0
                    return Exchange(status, headers, payloads, True, None, ttft, time.monotonic() - t0, content_type)
                raw = await resp.aread()
                body_out = _json_or_text(raw.decode("utf-8", "replace"))
                return Exchange(status, headers, body_out, stream, None, None, time.monotonic() - t0, content_type)
        except httpx.TimeoutException:
            error = "TIMEOUT"
        except httpx.TransportError as e:
            error = type(e).__name__
        # A stream cut short keeps whatever arrived, so the partial exchange is still recorded.
        return Exchange(status, headers, payloads or None, stream, error, ttft, time.monotonic() - t0, content_type)


async def _iter_sse(resp: httpx.Response):
    """Yield each SSE event's data: parsed JSON, the raw string, or "[DONE]"."""
    data_lines: list[str] = []
    async for line in resp.aiter_lines():
        if line == "":
            if data_lines:
                yield _decode_event("\n".join(data_lines))
                data_lines = []
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        if field == "data":
            data_lines.append(value[1:] if value.startswith(" ") else value)
    if data_lines:
        yield _decode_event("\n".join(data_lines))


def _decode_event(data: str) -> Any:
    if data.strip() == "[DONE]":
        return "[DONE]"
    return _json_or_text(data)
