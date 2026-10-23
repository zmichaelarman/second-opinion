"""The only API agent-written experiments may import (CLAUDE.md §11.4).

    from second_opinion.testkit import base_request, arm, compare

    BASE = base_request("<cluster_id>")

    def test_hypothesis():
        a = arm(BASE, factor="stream", level=True, n=100)
        b = arm(BASE, factor="stream", level=False, n=100)
        compare(a, b, hypothesis_id="<id>")

The Investigator sets the context in the environment: SECOND_OPINION_INVESTIGATION (the
investigation run directory) and SECOND_OPINION_EXPERIMENT_BUDGET_USD (this experiment's
spend cap). Every trial is paced and budgeted, and recorded under <run>/experiments/.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .budget import BudgetGuard
from .client import ApiClient
from .config import load_settings
from .investigator.spec import Spec
from .investigator.trials import TrialBatch, TrialRunner
from .pacer import Pacer
from .stats import fisher_two_sided, wilson_ci

MAX_N = 300
_transport: httpx.AsyncBaseTransport | None = None  # tests inject a fake server here


@dataclass(frozen=True)
class Base:
    cluster_id: str
    spec: Spec


@dataclass
class Arm:
    factor: str
    level: Any
    batch: TrialBatch

    def as_dict(self) -> dict[str, Any]:
        lo, hi = wilson_ci(self.batch.k, self.batch.n)
        return {"factor": self.factor, "level": self.level, **self.batch.as_dict(), "wilson_lo": lo, "wilson_hi": hi}


def _run_dir() -> Path:
    path = os.environ.get("SECOND_OPINION_INVESTIGATION")
    if not path:
        raise RuntimeError("SECOND_OPINION_INVESTIGATION is not set; experiments run inside `second-opinion investigate`")
    return Path(path)


def base_request(cluster_id: str) -> Base:
    """The minimal failing request the Investigator found for this cluster."""
    path = _run_dir() / "clusters" / cluster_id / "minimal.json"
    return Base(cluster_id, Spec.from_json(json.loads(path.read_text())["spec"]))


def arm(base: Base, factor: str, level: Any, n: int = 100) -> Arm:
    """Run n trials of the base request with one factor set to `level`."""
    n = max(1, min(int(n), MAX_N))
    spec = base.spec.with_factor(factor, level)
    settings = load_settings()
    run_dir = _run_dir()
    cap = os.environ.get("SECOND_OPINION_EXPERIMENT_BUDGET_USD")
    label = f"{factor}={json.dumps(level)}"

    async def go() -> TrialBatch:
        async with ApiClient.for_provider(settings, "tokenfactory", _transport) as client:
            runner = TrialRunner(
                client, Pacer(settings.defaults["rpm"]), BudgetGuard(settings.budget_usd, settings.ledger_path),
                run_dir / "experiments" / f"{base.cluster_id}.calls.jsonl", stage="investigate", run_id=run_dir.name,
                price=settings.price_for(spec.model), spend_cap=float(cap) if cap else None,
            )
            return await runner.run(spec, n, label, {"factor": factor, "level": level})

    return Arm(factor, level, asyncio.run(go()))


def compare(a: Arm, b: Arm, hypothesis_id: str) -> dict[str, Any]:
    """Record both arms and the two-sided Fisher p-value. The Investigator decides (with Holm) later."""
    p = fisher_two_sided(a.batch.k, a.batch.n, b.batch.k, b.batch.n) if a.batch.n and b.batch.n else 1.0
    result = {"hypothesis_id": hypothesis_id, "a": a.as_dict(), "b": b.as_dict(), "p": p}
    out = _run_dir() / "experiments" / f"{hypothesis_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return result
