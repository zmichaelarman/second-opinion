from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from second_opinion.config import AppConfig, REPO_ROOT


class BudgetExceededError(Exception):
    """Raised when spent + in-flight reserve would exceed SECOND_OPINION_BUDGET_USD."""


class BudgetGuard:
    """Spend ledger persisted in runs/ledger.json with hard cap enforcement (§8)."""

    def __init__(
        self,
        config: AppConfig,
        ledger_path: Path | None = None,
        cap_override_usd: float | None = None,
    ) -> None:
        self.config = config
        self.ledger_path = ledger_path or (REPO_ROOT / "runs" / "ledger.json")
        self.cap_usd = float(cap_override_usd if cap_override_usd is not None else config.budget_usd)
        self.in_flight_usd: float = 0.0
        self._lock = asyncio.Lock()
        self._load_or_init()

    def _load_or_init(self) -> dict[str, Any]:
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        if self.ledger_path.exists():
            try:
                data = json.loads(self.ledger_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
        data = {
            "total_spent_usd": 0.0,
            "total_calls": 0,
            "by_model": {},
            "entries": [],
        }
        self._write_ledger(data)
        return data

    def _write_ledger(self, data: dict[str, Any]) -> None:
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.ledger_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @property
    def spent_usd(self) -> float:
        data = self._load_or_init()
        return float(data.get("total_spent_usd", 0.0))

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.cap_usd - (self.spent_usd + self.in_flight_usd))

    def can_schedule(self, estimated_cost_usd: float) -> bool:
        return (self.spent_usd + self.in_flight_usd + estimated_cost_usd) <= self.cap_usd

    async def reserve(self, estimated_cost_usd: float) -> None:
        async with self._lock:
            if (self.spent_usd + self.in_flight_usd + estimated_cost_usd) > self.cap_usd:
                raise BudgetExceededError(
                    f"Budget cap ${self.cap_usd:.2f} reached (spent=${self.spent_usd:.4f}, "
                    f"in_flight=${self.in_flight_usd:.4f}, next=${estimated_cost_usd:.4f})."
                )
            self.in_flight_usd += estimated_cost_usd

    async def commit(
        self,
        reserved_usd: float,
        actual_cost_usd: float,
        model: str,
        stage: str,
        run_id: str,
        usage: dict[str, Any] | None = None,
    ) -> None:
        async with self._lock:
            self.in_flight_usd = max(0.0, self.in_flight_usd - reserved_usd)
            data = self._load_or_init()
            data["total_spent_usd"] = round(float(data.get("total_spent_usd", 0.0)) + actual_cost_usd, 8)
            data["total_calls"] = int(data.get("total_calls", 0)) + 1
            by_model = data.setdefault("by_model", {})
            m_entry = by_model.setdefault(
                model, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "spent_usd": 0.0}
            )
            m_entry["calls"] += 1
            if usage:
                m_entry["prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
                m_entry["completion_tokens"] += int(usage.get("completion_tokens", 0) or 0)
            m_entry["spent_usd"] = round(float(m_entry["spent_usd"]) + actual_cost_usd, 8)
            entries = data.setdefault("entries", [])
            entries.append(
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "run_id": run_id,
                    "stage": stage,
                    "model": model,
                    "cost_usd": round(actual_cost_usd, 8),
                    "usage": usage,
                }
            )
            # Keep last 500 entries in ledger summary so file stays compact
            if len(entries) > 500:
                data["entries"] = entries[-500:]
            self._write_ledger(data)

    def compute_call_cost(self, model: str, usage: dict[str, Any] | None, fallback_in: int = 200, fallback_out: int = 200) -> float:
        if usage and isinstance(usage, dict):
            p_tok = int(usage.get("prompt_tokens", 0) or 0)
            c_tok = int(usage.get("completion_tokens", 0) or 0)
        else:
            p_tok = fallback_in
            c_tok = fallback_out
        return self.config.estimate_cost(model, p_tok, c_tok)

