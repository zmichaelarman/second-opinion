"""Run n trials of one Spec: paced, budgeted, recorded. Shared by shrink, testkit and the reference replay."""

from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..budget import BudgetGuard, call_cost
from ..client import ApiClient
from ..config import Price
from ..contracts.base import Outcome, cell_id
from ..pacer import Pacer, parse_duration
from ..parse import parse_exchange
from ..store import call_record, utc_now
from .spec import Spec

EST_TOKENS = {False: (500, 400), True: (500, 2500), None: (500, 2500)}


class BudgetExhausted(RuntimeError):
    pass


@dataclass
class TrialBatch:
    label: str
    n: int = 0  # PASS/FAIL trials
    k: int = 0  # trials that reproduce the cluster signature
    errors: int = 0
    codes: Counter = field(default_factory=Counter)
    cost_usd: float = 0.0
    stopped: str | None = None
    examples: list[dict[str, Any]] = field(default_factory=list)  # a few parsed responses, failing and passing

    @property
    def rate(self) -> float:
        return self.k / self.n if self.n else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {"label": self.label, "n": self.n, "k": self.k, "errors": self.errors, "rate": self.rate,
                "codes": dict(self.codes), "cost_usd": round(self.cost_usd, 6), "stopped": self.stopped}


class TrialRunner:
    """Owns the client, pacer and budget for a sequence of batches against one provider."""

    def __init__(self, client: ApiClient, pacer: Pacer, budget: BudgetGuard, records: Path, *, stage: str,
                 run_id: str, price: Price | None, concurrency: int = 4, spend_cap: float | None = None,
                 retry_delay_s: float = 5.0):
        self.client, self.pacer, self.budget = client, pacer, budget
        self.records, self.stage, self.run_id = records, stage, run_id
        self.price, self.concurrency = price, concurrency
        self.spend_cap = spend_cap  # extra cap for this runner (e.g. $3 per cluster)
        self.spent = 0.0
        self.retry_delay_s = retry_delay_s

    def estimate(self, spec: Spec) -> float:
        if self.price is None:
            return 0.0
        return self.price.cost(*EST_TOKENS[spec.thinking])

    def _append(self, record: dict[str, Any]) -> None:
        import json
        import os

        self.records.parent.mkdir(parents=True, exist_ok=True)
        with self.records.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())

    async def run(self, spec: Spec, n: int, label: str, extra: dict[str, Any] | None = None,
                  model_override: str | None = None, transform=None) -> TrialBatch:
        """Run n trials. `transform(body)` may adjust the rendered body (the reference replay drops params)."""
        batch = TrialBatch(label)
        body = spec.request()
        if model_override:
            body["model"] = model_override
        if transform:
            body = transform(body)
        est = self.estimate(spec)
        queue: asyncio.Queue[int] = asyncio.Queue()
        for trial in range(n):
            queue.put_nowait(trial)

        async def worker() -> None:
            while batch.stopped is None:
                try:
                    trial = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                await self._one(spec, body, trial, est, batch, label, extra or {})

        await asyncio.gather(*(worker() for _ in range(self.concurrency)))
        return batch

    async def _one(self, spec: Spec, body: dict[str, Any], trial: int, est: float, batch: TrialBatch, label: str,
                   extra: dict[str, Any]) -> None:
        for attempt in (1, 2):
            if self.spend_cap is not None and self.spent + est > self.spend_cap:
                batch.stopped = "cluster budget"
                return
            if not self.budget.try_reserve(est):
                batch.stopped = "global budget"
                return
            await self.pacer.acquire()
            ts = utc_now()
            parsed = parse_exchange(await self.client.chat(body))
            self.pacer.observe(parsed.headers)
            cost = call_cost(self.price, parsed.usage, parsed.status, est) if self.price else 0.0
            self.budget.settle(est, cost, {"ts": ts, "run_id": self.run_id, "stage": self.stage, "model": body["model"],
                                           "contract_id": spec.contract_id, "usage": parsed.usage, **extra})
            self.spent += cost
            batch.cost_usd += cost
            result = spec.check(body, parsed)
            retry = result.outcome == Outcome.ERROR and attempt == 1
            self._append(call_record(
                run_id=self.run_id, stage=self.stage, cell_id=cell_id(f"{spec.cluster_id}|{label}", body["model"], spec.condition),
                contract_id=spec.contract_id, model=body["model"], condition=spec.condition.as_dict(), trial=trial,
                attempt=attempt, ts_start=ts, request=body, parsed=parsed, result=result, cost_usd=cost,
                final=not retry, cluster_id=spec.cluster_id, arm=label, **extra,
            ))
            if retry:
                if "HTTP_429" in result.codes:
                    retry_after = parse_duration(parsed.headers.get("retry-after"))
                    self.pacer.pause(30.0 if retry_after is None else retry_after)
                else:
                    await asyncio.sleep(self.retry_delay_s)
                continue
            if result.outcome == Outcome.ERROR:
                batch.errors += 1
                return
            batch.n += 1
            batch.codes.update(result.codes)
            hit = spec.reproduces(result)
            batch.k += hit
            if len(batch.examples) < 6:
                batch.examples.append({"reproduces": hit, "codes": result.codes, "content_head": (parsed.content or "")[:400],
                                       "reasoning_chars": len(parsed.reasoning or ""), "tool_calls": parsed.tool_calls,
                                       "finish_reason": parsed.finish_reason, "detail": result.detail})
            return
