"""`second-opinion reference` (CLAUDE.md §11.6): replay each finding's minimal request n=30 on
NVIDIA's hosted API for the same model, then attribute an owner.

- Token Factory fails while NVIDIA fails significantly less (Fisher p < 0.05): nebius_platform.
- Both fail at similar rates: nvidia_model for model copy errors (VALUE_MISMATCH-type codes),
  upstream_serving otherwise.
- No NVIDIA model for that Token Factory model: unknown.

NVIDIA's API is a different serving stack: a reference, not ground truth. A Token Factory-only
parameter that NVIDIA rejects is dropped and noted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
from rich.console import Console

from .budget import BudgetGuard
from .client import ApiClient
from .config import ConfigError, Settings
from .investigator.spec import Spec
from .investigator.trials import TrialRunner
from .pacer import Pacer
from .parse import parse_exchange
from .stats import fisher_two_sided, wilson_ci
from .store import RunStore, git_info, utc_now

N_REFERENCE = 30
P_THRESHOLD = 0.05
MODEL_ERROR_CODES = frozenset({"VALUE_MISMATCH", "WRONG_TOOL", "EXTRA_CONTENT"})
DROPPABLE = ("chat_template_kwargs", "stream_options", "reasoning_effort", "top_p")


def attribute(signature: list[str], tf_k: int, tf_n: int, nv_k: int | None, nv_n: int | None) -> tuple[str, float | None]:
    if nv_k is None or not nv_n or not tf_n:
        return "unknown", None
    p = fisher_two_sided(tf_k, tf_n, nv_k, nv_n)
    if p < P_THRESHOLD and nv_k / nv_n < tf_k / tf_n:
        return "nebius_platform", p
    base = {c.split(":", 1)[0] for c in signature}
    return ("nvidia_model" if base <= MODEL_ERROR_CODES else "upstream_serving"), p


def _rejected_param(message: str | None, body: dict[str, Any]) -> str | None:
    text = (message or "").lower()
    for key in DROPPABLE:
        if key in body and key.lower() in text:
            return key
    return next((k for k in DROPPABLE if k in body), None)


async def reference(
    settings: Settings,
    console: Console,
    findings_path: Path,
    *,
    yes: bool = False,
    n: int = N_REFERENCE,
    transport: httpx.AsyncBaseTransport | None = None,
) -> int:
    findings = [json.loads(x) for x in findings_path.read_text().splitlines() if x.strip()]
    targets = settings.defaults.get("reference_models", {})
    rpm = settings.provider("nvidia").get("rpm", 20)
    plan = []
    for f in findings:
        minimal = f.get("minimal") or {}
        price = settings.price_for(minimal.get("model", ""))
        plan.append((f, minimal, targets.get(getattr(price, "role", None) or "")))
    runnable = [p for p in plan if p[1] and p[2]]
    console.print(f"Reference replay: {len(runnable)} of {len(findings)} findings have an NVIDIA counterpart; "
                  f"{len(runnable) * n} calls at {rpm} RPM (about {len(runnable) * n / rpm:.0f} min) on the free tier.")
    for f, minimal, target in plan:
        console.print(f"  {f['id']}: {minimal.get('model')} -> {target or 'no NVIDIA model (owner stays unknown)'}")
    if not yes:
        console.print("[yellow]Nothing was sent. Re-run with --yes once this plan is approved.[/yellow]")
        return 0
    store = RunStore.create(settings.runs_dir, "reference")
    store.write_plan({"stage": "reference", "created": utc_now(), "git": git_info(settings.root),
                      "findings": str(findings_path), "n": n, "targets": targets})
    budget = BudgetGuard(settings.budget_usd, settings.ledger_path)
    try:
        client = ApiClient.for_provider(settings, "nvidia", transport)
    except ConfigError as e:
        console.print(f"[red]{e}[/red]")
        return 2
    async with client:
        for f, minimal, target in plan:
            row: dict[str, Any] = {"finding_id": f["id"], "tf_model": minimal.get("model"), "nvidia_model": target,
                                   "tf_k": minimal.get("k"), "tf_n": minimal.get("n"), "dropped_params": []}
            if not minimal or not target:
                row.update(owner="unknown", p=None, summary="No NVIDIA-hosted counterpart for this model; owner stays unknown.")
            else:
                spec = Spec.from_json(minimal["spec"])
                dropped: list[str] = []

                def transform(body: dict[str, Any]) -> dict[str, Any]:
                    return {k: v for k, v in body.items() if k not in dropped}

                # One probe call: if NVIDIA rejects a Token Factory-only parameter, drop it and note it.
                for _ in range(len(DROPPABLE)):
                    body = transform({**spec.request(), "model": target})
                    probe = parse_exchange(await client.chat({**body, "max_tokens": min(body.get("max_tokens", 64), 64)}))
                    if probe.status is not None and 400 <= probe.status < 500 and probe.status != 429:
                        key = _rejected_param(probe.error_message, body)
                        if key and key not in dropped:
                            dropped.append(key)
                            continue
                    break
                runner = TrialRunner(client, Pacer(rpm), budget, store.dir / "calls.jsonl", stage="reference",
                                     run_id=store.run_id, price=None, concurrency=2)
                batch = await runner.run(spec, n, "nvidia", {"finding_id": f["id"]}, model_override=target, transform=transform)
                owner, p = attribute(spec.signature, minimal.get("k") or 0, minimal.get("n") or 0, batch.k, batch.n)
                lo, hi = wilson_ci(batch.k, batch.n)
                tf_rate = (minimal.get("k") or 0) / (minimal.get("n") or 1)
                row.update(
                    owner=owner, p=p, nv_k=batch.k, nv_n=batch.n, nv_errors=batch.errors, dropped_params=dropped,
                    summary=(f"NVIDIA's API ({target}) failed {batch.k}/{batch.n} ({batch.k / batch.n if batch.n else 0:.0%}, "
                             f"95% CI {lo:.0%}-{hi:.0%}) vs {tf_rate:.0%} on Token Factory"
                             + (f" (Fisher p = {p:.2g})" if p is not None else "")
                             + (f"; dropped {', '.join(dropped)} for NVIDIA" if dropped else "") + "."),
                )
            with (store.dir / "attribution.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            console.print(f"  {f['id']}: owner {row['owner']}. {row['summary']}")
    store.write_summary({"run_id": store.run_id, "finished": utc_now(), "findings": len(findings)})
    return 0


def latest_findings(runs_dir: Path) -> Path | None:
    from .analysis import stage_runs

    for run_dir in reversed(stage_runs(runs_dir, "investigate")):
        if (run_dir / "findings.jsonl").exists():
            return run_dir / "findings.jsonl"
    return None

