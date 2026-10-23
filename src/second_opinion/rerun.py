"""`second-opinion rerun` (CLAUDE.md §13, M6): re-run every finding's minimal repro at n=30 on
Token Factory and mark each one still failing or fixed.

Fixed = no reproduction in 30 trials (the failure rate is then below 10% with 95% confidence)
and significantly below the rate measured during the investigation (Fisher p < 0.05).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
from rich.console import Console

from .budget import BudgetGuard
from .client import ApiClient
from .config import Settings
from .investigator.spec import Spec
from .investigator.trials import EST_TOKENS, TrialRunner
from .pacer import Pacer
from .stats import fisher_two_sided, wilson_ci
from .store import RunStore, git_info, utc_now

N_RERUN = 30


def verdict(k: int, n: int, k0: int, n0: int) -> tuple[str, float | None]:
    if n == 0:
        return "no data", None
    p = fisher_two_sided(k, n, k0, n0) if n0 else None
    if k == 0 and (p is None or p < 0.05):
        return "fixed", p
    if p is not None and p < 0.05 and k / n < k0 / n0:
        return "improved", p
    return "still failing", p


async def rerun(settings: Settings, console: Console, findings_path: Path, *, yes: bool = False, n: int = N_RERUN,
                transport: httpx.AsyncBaseTransport | None = None) -> int:
    findings = [json.loads(x) for x in findings_path.read_text().splitlines() if x.strip()]
    todo = [f for f in findings if f.get("minimal")]
    est = 0.0
    for f in todo:
        price = settings.price_for(f["minimal"]["model"])
        if price:
            est += n * price.cost(*EST_TOKENS[Spec.from_json(f["minimal"]["spec"]).thinking])
    console.print(f"Rerun {len(todo)} finding repro(s) x {n} trials on Token Factory: {len(todo) * n} calls, estimated ${est:.2f}.")
    if not yes:
        console.print("[yellow]Paid stage: nothing was sent. Re-run with --yes once this estimate is approved.[/yellow]")
        return 0
    store = RunStore.create(settings.runs_dir, "rerun")
    store.write_plan({"stage": "rerun", "created": utc_now(), "git": git_info(settings.root), "findings": str(findings_path), "n": n})
    budget = BudgetGuard(settings.budget_usd, settings.ledger_path)
    rows: list[dict[str, Any]] = []
    async with ApiClient.for_provider(settings, "tokenfactory", transport) as tf:
        for f in todo:
            spec = Spec.from_json(f["minimal"]["spec"])
            runner = TrialRunner(tf, Pacer(settings.defaults["rpm"]), budget, store.dir / "calls.jsonl", stage="rerun",
                                 run_id=store.run_id, price=settings.price_for(spec.model))
            batch = await runner.run(spec, n, "rerun", {"finding_id": f["id"]})
            status, p = verdict(batch.k, batch.n, f["minimal"]["k"], f["minimal"]["n"])
            lo, hi = wilson_ci(batch.k, batch.n)
            row = {"finding_id": f["id"], "title": f.get("title"), "status": status, "k": batch.k, "n": batch.n,
                   "rate": batch.rate, "wilson_lo": lo, "wilson_hi": hi, "p_vs_original": p,
                   "original_k": f["minimal"]["k"], "original_n": f["minimal"]["n"], "ts": utc_now()}
            rows.append(row)
            with (store.dir / "rerun.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            console.print(f"  {f['id']} {f.get('title')}: {batch.k}/{batch.n} -> {status}")
    store.write_summary({"run_id": store.run_id, "finished": utc_now(),
                         "status": {s: sum(1 for r in rows if r["status"] == s) for s in {r["status"] for r in rows}}})
    return 0
