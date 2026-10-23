"""Spend ledger and hard cap (CLAUDE.md §8).

Spend = usage x config/prices.yaml. Every paid call appends one line to runs/ledger.jsonl,
which is append-only like the rest of runs/, so the total survives across runs. Calls are
reserved at their estimated cost before they are sent; scheduling stops once
spent + in-flight reserve + the next estimate would exceed the cap.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .config import Price


def call_cost(price: Price, usage: dict[str, Any] | None, status: int | None, estimate: float) -> float:
    """Actual cost from usage. Without usage: rejected requests (4xx) cost nothing, while
    errors and timeouts may still have been processed, so they are charged the estimate."""
    usage = usage or {}
    pt, ct = usage.get("prompt_tokens"), usage.get("completion_tokens")
    if isinstance(pt, int) and isinstance(ct, int):
        return price.cost(pt, ct)
    if status is not None and 400 <= status < 500:
        return 0.0
    return estimate


class BudgetGuard:
    def __init__(self, cap_usd: float, ledger_path: Path):
        self.cap = cap_usd
        self.ledger_path = ledger_path
        self.spent = self._load()
        self.in_flight = 0.0

    def _load(self) -> float:
        if not self.ledger_path.exists():
            return 0.0
        total = 0.0
        with self.ledger_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    total += float(json.loads(line).get("cost_usd") or 0.0)
                except ValueError:
                    continue
        return total

    @property
    def remaining(self) -> float:
        return self.cap - self.spent - self.in_flight

    def try_reserve(self, estimate: float) -> bool:
        if self.spent + self.in_flight + estimate > self.cap:
            return False
        self.in_flight += estimate
        return True

    def settle(self, estimate: float, actual: float, entry: dict[str, Any]) -> None:
        """Release the reservation and record the actual cost."""
        self.in_flight = max(0.0, self.in_flight - estimate)
        self.spent += actual
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({**entry, "cost_usd": round(actual, 8)}, ensure_ascii=False, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())
