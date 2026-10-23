from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any, Callable

import httpx

from second_opinion.config import AppConfig
from second_opinion.pacer import HeaderAwarePacer
from second_opinion.parse import (
    ParsedResponse,
    filter_headers,
    parse_json_response,
    parse_sse_chunks,
)


def _parse_retry_after(headers: dict[str, str] | Any, default: float = 30.0) -> float:
    if not headers:
        return default
    for k, v in headers.items():
        if str(k).lower() == "retry-after":
            try:
                return max(1.0, float(v))
            except ValueError:
                return default
    return default


class InferenceClient:
    """Async httpx client for Token Factory and NVIDIA Reference endpoints (§8).

    - Captures exact JSON or SSE chunks and rate-limit headers.
    - Never logs or persists Authorization headers.
    - Enforces retry rules:
      * 429: sleep for Retry-After (30s if absent), up to 3 attempts.
      * 5xx or timeout: retry once after 5s.
      * Every attempt is yielded to `on_attempt` so each attempt is recorded as its own CallRecord.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        pacer: HeaderAwarePacer | None = None,
        connect_timeout_s: float = 10.0,
        read_timeout_s: float = 600.0,
        sleep_fn: Any = asyncio.sleep,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self.pacer = pacer
        self._sleep = sleep_fn
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=connect_timeout_s,
                read=read_timeout_s,
                write=30.0,
                pool=30.0,
            ),
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> InferenceClient:
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.aclose()

    def _auth_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    async def _single_call(self, req_body: dict[str, Any]) -> tuple[str, ParsedResponse]:
        url = f"{self.base_url}/chat/completions"
        is_stream = bool(req_body.get("stream", False))
        ts_start = datetime.now(timezone.utc).isoformat()
        t0 = time.perf_counter()

        try:
            if not is_stream:
                resp = await self._client.post(url, json=req_body, headers=self._auth_headers())
                total_s = time.perf_counter() - t0
                if self.pacer is not None:
                    self.pacer.update_from_headers(resp.headers)
                try:
                    body = resp.json()
                except Exception:
                    body = {"_raw_text": resp.text}
                parsed = parse_json_response(
                    status=resp.status_code,
                    headers=resp.headers,
                    body=body,
                    total_s=total_s,
                    ttft_s=total_s,
                )
                return ts_start, parsed

            # Streaming SSE call
            data_payloads: list[str] = []
            ttft_s: float | None = None
            async with self._client.stream("POST", url, json=req_body, headers=self._auth_headers()) as resp:
                if self.pacer is not None:
                    self.pacer.update_from_headers(resp.headers)
                if resp.status_code >= 400:
                    raw_bytes = await resp.aread()
                    total_s = time.perf_counter() - t0
                    text = raw_bytes.decode("utf-8", errors="replace")
                    try:
                        import json

                        err_body = json.loads(text)
                    except Exception:
                        err_body = {"_raw_text": text}
                    parsed = parse_json_response(
                        status=resp.status_code,
                        headers=resp.headers,
                        body=err_body,
                        total_s=total_s,
                    )
                    return ts_start, parsed

                async for line in resp.aiter_lines():
                    if ttft_s is None and line.strip():
                        ttft_s = time.perf_counter() - t0
                    if line.startswith("data:"):
                        payload = line[len("data:") :].strip()
                        data_payloads.append(payload)

                total_s = time.perf_counter() - t0
                parsed = parse_sse_chunks(
                    status=resp.status_code,
                    headers=resp.headers,
                    data_payloads=data_payloads,
                    total_s=total_s,
                    ttft_s=ttft_s,
                )
                return ts_start, parsed

        except httpx.TimeoutException as exc:
            total_s = time.perf_counter() - t0
            return ts_start, ParsedResponse(
                status=0,
                headers={},
                total_s=total_s,
                raw={"error": "timeout", "detail": str(exc)},
                error_type="TIMEOUT",
            )
        except httpx.HTTPError as exc:
            total_s = time.perf_counter() - t0
            return ts_start, ParsedResponse(
                status=0,
                headers={},
                total_s=total_s,
                raw={"error": "http_error", "detail": str(exc)},
                error_type="HTTP_5XX",
            )

    async def chat_completion(
        self,
        req_body: dict[str, Any],
        estimated_tokens: int = 1000,
        on_attempt: Callable[[int, str, ParsedResponse, bool], None] | None = None,
    ) -> ParsedResponse:
        """Execute a chat completion with header-aware pacing and spec retry policy (§8).

        Retry rules:
        - 429: sleep for Retry-After (30 s if absent), up to 3 attempts.
        - 5xx or timeout: retry once after 5 s (up to 2 attempts).
        """
        attempt = 0
        while True:
            attempt += 1
            if self.pacer is not None:
                await self.pacer.acquire(estimated_tokens=estimated_tokens)

            ts_start, parsed = await self._single_call(req_body)

            if parsed.status == 429 and attempt < 3:
                if on_attempt is not None:
                    on_attempt(attempt, ts_start, parsed, False)
                wait_s = _parse_retry_after(parsed.headers, default=30.0)
                await self._sleep(wait_s)
                continue

            is_5xx_or_timeout = (
                parsed.status >= 500
                or parsed.error_type in ("TIMEOUT", "HTTP_5XX")
            )
            if is_5xx_or_timeout and attempt < 2:
                if on_attempt is not None:
                    on_attempt(attempt, ts_start, parsed, False)
                await self._sleep(5.0)
                continue

            if on_attempt is not None:
                on_attempt(attempt, ts_start, parsed, True)
            return parsed
