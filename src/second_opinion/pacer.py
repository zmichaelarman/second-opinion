"""Header-aware rate limiting (CLAUDE.md §8).

Requests are spaced 60/rate seconds apart, where rate = min(--rpm, 0.8 x the latest
x-ratelimit-limit-requests), halved for 120 s whenever a response carries
x-ratelimit-over-limit: yes. If x-ratelimit-remaining-tokens is below twice the next
request's estimated tokens, the next request waits for x-ratelimit-reset-tokens first.
A 429 pauses everyone for Retry-After.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Awaitable, Callable

OVER_LIMIT_PENALTY_S = 120.0
DEFAULT_TOKEN_WAIT_S = 5.0
_DURATION = re.compile(r"(\d+(?:\.\d+)?)(ms|s|m|h)")


def parse_duration(value: str | None) -> float | None:
    """Seconds from "1.5", "20ms", "6m0s" or "1h2m3s". None if absent or unparsable."""
    if not value:
        return None
    value = value.strip()
    try:
        return float(value)
    except ValueError:
        pass
    parts = _DURATION.findall(value)
    if not parts or "".join(n + u for n, u in parts) != value:
        return None
    scale = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}
    return sum(float(n) * scale[u] for n, u in parts)


def _int(value: str | None) -> int | None:
    try:
        return int(float(value)) if value not in (None, "") else None
    except ValueError:
        return None


class Pacer:
    def __init__(
        self,
        rpm: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ):
        self.base_rpm = rpm
        self.clock = clock
        self.sleep = sleep
        self.limit_rpm: int | None = None
        self.penalty_until = 0.0
        self.not_before = 0.0
        self.next_slot = 0.0
        self.remaining_tokens: int | None = None
        self.tokens_reset_s: float | None = None
        self.over_limit_seen = 0
        self._lock = asyncio.Lock()

    @property
    def rate(self) -> float:
        """Current requests per minute."""
        rate = float(self.base_rpm)
        if self.limit_rpm:
            rate = min(rate, 0.8 * self.limit_rpm)
        if self.clock() < self.penalty_until:
            rate /= 2
        return max(rate, 0.1)

    async def acquire(self, est_tokens: int = 0) -> None:
        async with self._lock:
            now = self.clock()
            start = max(now, self.next_slot, self.not_before)
            if self.remaining_tokens is not None and self.remaining_tokens < 2 * est_tokens:
                wait = self.tokens_reset_s if self.tokens_reset_s is not None else DEFAULT_TOKEN_WAIT_S
                start = max(start, now + wait)
                self.remaining_tokens = None  # waited once; the next response's headers decide again
            self.next_slot = start + 60.0 / self.rate
            delay = start - now
        if delay > 0:
            await self.sleep(delay)

    def observe(self, headers: dict[str, str]) -> None:
        limit = _int(headers.get("x-ratelimit-limit-requests"))
        if limit:
            self.limit_rpm = limit
        remaining = _int(headers.get("x-ratelimit-remaining-tokens"))
        if remaining is not None:
            self.remaining_tokens = remaining
            self.tokens_reset_s = parse_duration(headers.get("x-ratelimit-reset-tokens"))
        if (headers.get("x-ratelimit-over-limit") or "").strip().lower() == "yes":
            self.over_limit_seen += 1
            self.penalty_until = self.clock() + OVER_LIMIT_PENALTY_S

    def pause(self, seconds: float) -> None:
        """Hold every request for `seconds` (after a 429)."""
        self.not_before = max(self.not_before, self.clock() + seconds)
