"""Executes the pilot, screen and escalate stages (CLAUDE.md §8).

Each (cell, trial) is one work item. Workers take items in plan order. Every HTTP attempt
becomes its own CallRecord; only a trial's last attempt is marked final. Retries: a 429
waits for Retry-After (30 s if absent) for up to 3 attempts; a 5xx or timeout is retried
once after 5 s. A final ERROR stays an ERROR.
"""

from __future__ import annotations

import asyncio
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import httpx
from rich.console import Console
from rich.table import Table

from . import analysis
from .budget import BudgetGuard, call_cost
from .client import ApiClient
from .config import Settings
from .contracts.base import BaseContract
from .discovery import DiscoveredModel, discover
from .pacer import Pacer, parse_duration
from .parse import parse_exchange
from .planner import (
    TokenEstimates,
    WorkItem,
    cost_table,
    estimate_cost,
    plan_escalate,
    plan_pilot,
    plan_screen,
    prereg_check,
    registry_hash,
)
from .planner import registry as build_registry
from .store import RunStore, call_record, git_info, utc_now

MAX_ATTEMPTS_429 = 3
MAX_ATTEMPTS_5XX = 2
DEFAULT_RETRY_AFTER_S = 30.0
RETRYABLE_ERRORS = ("HTTP_5XX", "TIMEOUT", "TRANSPORT_ERROR")
STAGES = ("pilot", "screen", "escalate")


class Runner:
    def __init__(
        self,
        *,
        stage: str,
        store: RunStore,
        items: list[WorkItem],
        contracts: dict[str, BaseContract],
        models: dict[str, DiscoveredModel],
        client: ApiClient,
        pacer: Pacer,
        budget: BudgetGuard,
        estimates: TokenEstimates,
        concurrency: int = 6,
        retry_5xx_delay_s: float = 5.0,
        progress_every_s: float = 30.0,
        log: Callable[[str], None] = print,
    ):
        self.stage, self.store, self.items = stage, store, items
        self.contracts, self.models = contracts, models
        self.client, self.pacer, self.budget, self.estimates = client, pacer, budget, estimates
        self.concurrency = concurrency
        self.retry_5xx_delay_s = retry_5xx_delay_s
        self.progress_every_s = progress_every_s
        self.log = log
        self.stop_reason: str | None = None
        self.outcomes: Counter = Counter()
        self.attempts = 0
        self.retries = 0
        self.total = 0
        self.t0 = 0.0

    async def run(self) -> int:
        finished = self.store.final_keys()
        pending = [it for it in self.items if (it.cell.id, it.trial) not in finished]
        if len(pending) < len(self.items):
            self.log(f"resume: {len(self.items) - len(pending)} trials already final; {len(pending)} to run")
        queue: asyncio.Queue[WorkItem] = asyncio.Queue()
        for it in pending:
            queue.put_nowait(it)
        self.total, self.t0 = len(pending), time.monotonic()
        reporter = asyncio.create_task(self._report_loop())
        try:
            await asyncio.gather(*(self._worker(queue) for _ in range(self.concurrency)))
        finally:
            reporter.cancel()
        self.log(self._progress_line())
        if self.stop_reason == "budget":
            self.log(f"stopped: budget cap ${self.budget.cap:.2f} reached (spent ${self.budget.spent:.2f})")
            return 3
        return 0

    async def _worker(self, queue: asyncio.Queue[WorkItem]) -> None:
        while self.stop_reason is None:
            try:
                item = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            await self._trial(item)

    async def _trial(self, item: WorkItem) -> None:
        contract = self.contracts[item.cell.contract_id]
        model = self.models[item.cell.model]
        assert model.price is not None
        cond = item.cell.condition
        body = contract.build_request(model.id, cond, item.trial)
        pt, ct = self.estimates.get(contract.id, cond.thinking)
        estimate = model.price.cost(pt, ct)
        n429 = n5xx = attempt = 0
        while True:
            attempt += 1
            if not self.budget.try_reserve(estimate):
                self.stop_reason = "budget"  # stop scheduling; trials already in flight still finish
                return
            await self.pacer.acquire(int(pt + ct))
            ts = utc_now()
            parsed = parse_exchange(await self.client.chat(body))
            self.pacer.observe(parsed.headers)
            cost = call_cost(model.price, parsed.usage, parsed.status, estimate)
            self.budget.settle(estimate, cost, {
                "ts": ts, "run_id": self.store.run_id, "stage": self.stage, "model": model.id,
                "contract_id": contract.id, "cell_id": item.cell.id, "trial": item.trial, "attempt": attempt,
                "usage": parsed.usage,
            })
            self.attempts += 1
            result = contract.check(body, parsed)
            wait: float | None = None
            if "HTTP_429" in result.codes and n429 < MAX_ATTEMPTS_429 - 1:
                n429 += 1
                retry_after = parse_duration(parsed.headers.get("retry-after"))
                wait = DEFAULT_RETRY_AFTER_S if retry_after is None else retry_after
                self.pacer.pause(wait)  # every worker waits, not just this one
            elif any(c in RETRYABLE_ERRORS for c in result.codes) and n5xx < MAX_ATTEMPTS_5XX - 1:
                n5xx += 1
                wait = self.retry_5xx_delay_s
            self.store.append_call(call_record(
                run_id=self.store.run_id, stage=self.stage, cell_id=item.cell.id, contract_id=contract.id,
                model=model.id, condition=cond.as_dict(), trial=item.trial, attempt=attempt, ts_start=ts,
                request=body, parsed=parsed, result=result, cost_usd=cost, final=wait is None,
            ))
            if wait is None:
                self.outcomes[result.outcome.value] += 1
                return
            self.retries += 1
            if "HTTP_429" not in result.codes:
                await asyncio.sleep(wait)

    async def _report_loop(self) -> None:
        while True:
            await asyncio.sleep(self.progress_every_s)
            self.log(self._progress_line())

    def _progress_line(self) -> str:
        done = sum(self.outcomes.values())
        elapsed = max(time.monotonic() - self.t0, 1e-9)
        per_min = self.attempts / elapsed * 60
        eta = (self.total - done) / (done / elapsed) if done else float("nan")
        eta_s = f"{eta / 60:.0f} min" if eta == eta else "?"
        return (
            f"[{self.stage}] {done}/{self.total} trials | pass {self.outcomes['pass']} fail {self.outcomes['fail']} "
            f"error {self.outcomes['error']} | attempts {self.attempts} (retries {self.retries}) | "
            f"spent ${self.budget.spent:.2f} of ${self.budget.cap:.2f} | {per_min:.1f}/min "
            f"(pacer {self.pacer.rate:.0f} rpm) | ETA {eta_s}"
        )


def parse_max_n(values: list[str] | None) -> dict[str, int]:
    out = {}
    for v in values or []:
        model, sep, n = v.rpartition("=")
        if not sep or not model or not n.isdigit():
            raise ValueError(f"--max-n expects MODEL=N, got {v!r}")
        out[model] = int(n)
    return out


def resolve_models(found: list[DiscoveredModel], spec: str) -> list[DiscoveredModel]:
    if spec in ("", "auto"):
        return [m for m in found if m.runnable]
    wanted = [s.strip() for s in spec.split(",") if s.strip()]
    by_id = {m.id: m for m in found}
    missing = [w for w in wanted if w not in by_id]
    unpriced = [w for w in wanted if w in by_id and by_id[w].price is None]
    if missing or unpriced:
        raise ValueError(f"not listed by /v1/models: {missing}; listed but unpriced or excluded: {unpriced}")
    return [by_id[w] for w in wanted]


def summarize_run(store: RunStore, contracts: dict[str, BaseContract], stop_reason: str | None) -> dict[str, Any]:
    finals = list(store.iter_calls())
    cells = analysis.aggregate(r for r in finals if r.get("final"))
    return {
        "run_id": store.run_id,
        "finished": utc_now(),
        "stop_reason": stop_reason,
        "attempts": len(finals),
        "final_trials": sum(1 for r in finals if r.get("final")),
        "outcomes": dict(Counter(r["outcome"] for r in finals if r.get("final"))),
        "cost_usd": round(sum(r.get("cost_usd") or 0 for r in finals), 6),
        "token_medians": analysis.token_medians(r for r in finals if r.get("final")),
        "cells": [s.as_dict() for s in sorted(cells.values(), key=lambda s: (s.contract_id, s.model, s.cell_id))],
    }


def results_table(summary: dict[str, Any]) -> Table:
    table = Table(title=f"Results: {summary['run_id']}")
    for col in ("Contract", "Model", "Condition", "n", "Fail", "Err", "Top codes"):
        table.add_column(col)
    for c in summary["cells"]:
        cond = c["condition"]
        label = f"think={cond.get('thinking')}" + (" stream" if cond.get("stream") else "") + (
            f" max={cond['max_tokens']}" if cond.get("max_tokens") else "") + (
            f" {cond['shape']}" if cond.get("shape") not in (None, "wrapped") else "")
        top = ", ".join(f"{k}x{v}" for k, v in sorted(c["codes"].items(), key=lambda kv: -kv[1])[:3])
        style = "red" if c["fails"] else ""
        table.add_row(c["contract_id"], c["model"], label, str(c["n"]), f"[{style}]{c['fails']}[/{style}]" if style else "0",
                      str(c["errors"]), top)
    return table


async def run_stage(
    settings: Settings,
    console: Console,
    *,
    stage: str,
    models_spec: str = "auto",
    contracts_spec: str | None = None,
    rpm: float | None = None,
    concurrency: int | None = None,
    max_n: dict[str, int] | None = None,
    yes: bool = False,
    prereg: bool = True,
    resume: Path | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    retry_5xx_delay_s: float = 5.0,
) -> int:
    from .contracts import registry

    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}")
    contracts = registry.select(contracts_spec)
    contract_map = {c.id: c for c in registry.all_contracts()}
    rpm = rpm or settings.defaults["rpm"]
    concurrency = concurrency or settings.defaults["concurrency"]
    async with ApiClient.for_provider(settings, "tokenfactory", transport) as client:
        listing = await client.get("models")
        if listing.status != 200:
            console.print(f"[red]GET /v1/models failed: status={listing.status} error={listing.error}[/red]")
            return 1
        models = resolve_models(discover(listing.body, settings), models_spec)
        if not models:
            console.print("[red]No runnable models (listed, Nemotron, and priced in config/prices.yaml).[/red]")
            return 1
        est = TokenEstimates.from_runs(settings.runs_dir)
        if stage == "pilot":
            items = plan_pilot(contracts, models)
        elif stage == "screen":
            items = plan_screen(contracts, models, settings.defaults["n_screen"], max_n)
        else:
            screened = analysis.aggregate(analysis.iter_records(analysis.stage_runs(settings.runs_dir, "screen")))
            if not screened:
                console.print("[red]No screen runs found under runs/; run the screen stage first.[/red]")
                return 1
            items = plan_escalate(contracts, models, screened, settings.defaults["n_escalate"], max_n)
        store = RunStore.open(resume) if resume else None
        finished = store.final_keys() if store else set()
        pending = [it for it in items if (it.cell.id, it.trial) not in finished]
        model_map = {m.id: m for m in models}
        rows = estimate_cost(pending, model_map, contract_map, est)
        console.print(cost_table(stage, rows, est))
        total_usd = sum(u for _, u in rows.values())
        budget = BudgetGuard(settings.budget_usd, settings.ledger_path)
        console.print(
            f"{len(pending)} calls on {len(models)} models, estimated ${total_usd:.2f}, about "
            f"{len(pending) / rpm:.0f} min at {rpm:g} RPM. Budget: ${budget.spent:.2f} spent of ${budget.cap:.2f}."
        )
        reg_hash = registry_hash(build_registry(settings.defaults))
        if stage == "screen" and prereg:
            ok, msg = prereg_check(settings.root, settings.defaults)
            if not ok:
                console.print(f"[red]Pre-registration check failed: {msg}[/red]")
                return 2
            console.print(f"[green]{msg}[/green]")
        if not yes:
            console.print("[yellow]Paid stage: nothing was sent. Re-run with --yes once this estimate is approved.[/yellow]")
            return 0
        if not pending:
            console.print("Nothing to run.")
            return 0
        store = store or RunStore.create(settings.runs_dir, stage)
        store.write_plan({
            "stage": stage,
            "created": utc_now(),
            "git": git_info(settings.root),
            "registry_hash": reg_hash,
            "prereg_checked": stage == "screen" and prereg,
            "defaults": settings.defaults,
            "models": {m.id: asdict(m.price) for m in models if m.price},
            "contracts": [c.id for c in contracts],
            "max_n": max_n or {},
            "rpm": rpm,
            "concurrency": concurrency,
            "items": len(items),
            "pending": len(pending),
            "estimated_usd": round(total_usd, 4),
            "token_estimates": est.source or "fallbacks",
        })
        log_path = store.dir / "progress.log"

        def log(line: str) -> None:
            stamped = f"{utc_now()} {line}"
            console.print(stamped, markup=False, highlight=False)
            with log_path.open("a") as f:
                f.write(stamped + "\n")

        log(f"run {store.run_id}: {len(pending)} trials, rpm {rpm:g}, concurrency {concurrency}")
        runner = Runner(
            stage=stage, store=store, items=items, contracts=contract_map, models=model_map, client=client,
            pacer=Pacer(rpm), budget=budget, estimates=est, concurrency=concurrency,
            retry_5xx_delay_s=retry_5xx_delay_s, log=log,
        )
        code = await runner.run()
    summary = summarize_run(store, contract_map, runner.stop_reason)
    store.write_summary(summary)
    console.print(results_table(summary))
    console.print(f"Outcomes: {summary['outcomes']}; cost ${summary['cost_usd']:.4f}. Raw data: {store.dir}")
    return code
