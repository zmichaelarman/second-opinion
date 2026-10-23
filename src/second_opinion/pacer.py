from __future__ import annotations

import asyncio
import re
import time
from typing import Any


def _parse_number(val: str | None) -> float | None:
    if not val:
        return None
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)", str(val))
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


class HeaderAwarePacer:
    """Token-bucket rate limiter that adapts to Nebius Token Factory x-ratelimit-* headers (§8)."""

    def __init__(self, base_rpm: float = 40.0, clock: Any = time.monotonic, sleep_fn: Any = asyncio.sleep) -> None:
        self.base_rpm = max(1.0, float(base_rpm))
        self._clock = clock
        self._sleep = sleep_fn
        self._lock = asyncio.Lock()
        self.limit_requests: float | None = None
        self.remaining_requests: float | None = None
        self.remaining_tokens: float | None = None
        self.reset_tokens_s: float | None = None
        self.over_limit_until: float = 0.0
        self._next_allowed_ts: float = 0.0

    def effective_rpm(self, now: float | None = None) -> float:
        now_ts = self._clock() if now is None else now
        rate = self.base_rpm
        if self.limit_requests is not None and self.limit_requests > 0:
            rate = min(rate, 0.8 * self.limit_requests)
        if now_ts < self.over_limit_until:
            rate = max(1.0, rate * 0.5)
        return max(1.0, rate)

    def update_from_headers(self, headers: dict[str, str] | Any) -> None:
        if not headers:
            return
        h = {str(k).lower(): str(v) for k, v in headers.items()}
        lim_req = _parse_number(h.get("x-ratelimit-limit-requests"))
        if lim_req is not None and lim_req > 0:
            self.limit_requests = lim_req

        rem_req = _parse_number(h.get("x-ratelimit-remaining-requests"))
        if rem_req is not None:
            self.remaining_requests = rem_req

        rem_tok = _parse_number(h.get("x-ratelimit-remaining-tokens"))
        if rem_tok is not None:
            self.remaining_tokens = rem_tok

        rst_tok = _parse_number(h.get("x-ratelimit-reset-tokens"))
        if rst_tok is not None:
            self.reset_tokens_s = rst_tok

        over = h.get("x-ratelimit-over-limit", "").strip().lower()
        if over in ("yes", "true", "1"):
            self.over_limit_until = self._clock() + 120.0

    async def acquire(self, estimated_tokens: int = 1000) -> float:
        """Wait until the next request is allowed. Returns total seconds waited."""
        waited = 0.0
        async with self._lock:
            now = self._clock()
            # Check remaining tokens
            if (
                self.remaining_tokens is not None
                and self.remaining_tokens < 2 * estimated_tokens
            ):
                token_wait = min(max(self.reset_tokens_s or 5.0, 1.0), 30.0)
                await self._sleep(token_wait)
                waited += token_wait
                # Reset our cached remaining_tokens estimate after waiting
                self.remaining_tokens = None
                now = self._clock()

            rpm = self.effective_rpm(now)
            interval = 60.0 / rpm
            if now < self._next_allowed_ts:
                delay = self._next_allowed_ts - now
                await self._sleep(delay)
                waited += delay
                now = self._clock()
            self._next_allowed_ts = now + interval
        return waited

