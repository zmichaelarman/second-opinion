"""`second-opinion investigate` (CLAUDE.md §11): for each top failure cluster, shrink the request,
propose causes (Ultra), write and run an experiment per cause (Super + testkit), and accept or
reject each cause with Fisher's exact test. One more round runs if every cause is rejected."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Callable

import httpx
from rich.console import Console
from rich.table import Table

from .. import analysis
from ..budget import BudgetGuard, call_cost
from ..client import ApiClient
from ..config import Settings
from ..contracts.base import Condition, OracleResult, Outcome, cell_id
from ..discovery import DiscoveredModel, discover
from ..pacer import Pacer
from ..parse import parse_exchange
from ..report.build import evidence_view, repro
from ..stats import wilson_ci
from ..store import RunStore, call_record, git_info, utc_now
from .cluster import Cluster, build_clusters
from .decide import cause, decide, workaround
from .experiment import run_experiment, write_experiment
from .hypothesize import propose
from .shrink import K, MAX_CANDIDATES, involved_fields, shrink
from .spec import Spec, spec_from_cluster
from .trials import TrialRunner

DEFAULT_MAX_CLUSTERS = 5
DEFAULT_CLUSTER_BUDGET_USD = 3.0
ARM_N = 100
LLM_EST_TOKENS = (3000, 5000)


def _role(models: list[DiscoveredModel], role: str) -> DiscoveredModel | None:
    return next((m for m in models if m.role == role), None)


def clusters_table(clusters: list[Cluster]) -> Table:
    table = Table(title="Failure clusters (most severe first)")
    for col in ("Cluster", "Severity", "Contract", "Codes", "Failed", "Models"):
        table.add_column(col)
    for c in clusters:
        table.add_row(c.id, c.severity.name, c.contract_id, ", ".join(c.codes), f"{c.failures}/{c.trials}", str(len(c.models)))
    return table


def _arm_view(arm: dict[str, Any]) -> dict[str, Any]:
    lo, hi = wilson_ci(arm["k"], arm["n"])
    return {"label": f"{arm['factor']} = {json.dumps(arm['level'])}", "n": arm["n"], "k": arm["k"],
            "rate": arm["k"] / arm["n"] if arm["n"] else 0.0, "lo": lo, "hi": hi}


async def _inline_arms(runner: TrialRunner, spec: Spec, h: dict[str, Any], hid: str, out_dir: Path, n: int) -> dict[str, Any]:
    """Run both arms directly (used when the experiment file fails the safety gate twice)."""
    arms = {}
    for side in ("a", "b"):
        level = h[f"level_{side}"]
        batch = await runner.run(spec.with_factor(h["factor"], level), n, f"{h['factor']}={json.dumps(level)}",
                                 {"factor": h["factor"], "level": level, "hypothesis_id": hid})
        arms[side] = {"factor": h["factor"], "level": level, **batch.as_dict()}
    result = {"hypothesis_id": hid, **arms, "p": None, "fallback": True}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{hid}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return result


async def investigate_cluster(
    cl: Cluster, *, settings: Settings, store: RunStore, tf: ApiClient, budget: BudgetGuard, ultra: DiscoveredModel,
    coder: DiscoveredModel, cluster_budget: float, records: list[dict[str, Any]], console: Console,
    experiment_runner: Callable[..., dict[str, Any]] = run_experiment, arm_n: int = ARM_N,
) -> dict[str, Any]:
    cdir = store.dir / "clusters" / cl.id
    cdir.mkdir(parents=True, exist_ok=True)
    worst = max(cl.cells, key=lambda c: (c["rate"], c["n"]))
    model = worst["model"]
    spec = spec_from_cluster(cl.id, cl.contract_id, model, worst["condition"], list(cl.codes))
    runner = TrialRunner(tf, Pacer(settings.defaults["rpm"]), budget, cdir / "trials.calls.jsonl", stage="investigate",
                         run_id=store.run_id, price=settings.price_for(model), spend_cap=cluster_budget)

    def llm(m: DiscoveredModel, purpose: str):
        async def complete(body: dict[str, Any]) -> str | None:
            est = m.price.cost(*LLM_EST_TOKENS)  # type: ignore[union-attr]
            if runner.spent + est > cluster_budget or not budget.try_reserve(est):
                return None
            ts = utc_now()
            parsed = parse_exchange(await tf.chat(body))
            cost = call_cost(m.price, parsed.usage, parsed.status, est)  # type: ignore[arg-type]
            budget.settle(est, cost, {"ts": ts, "run_id": store.run_id, "stage": "investigate", "model": m.id,
                                      "contract_id": f"investigate.{purpose}", "usage": parsed.usage})
            runner.spent += cost
            store.append_call(call_record(
                run_id=store.run_id, stage="investigate", cell_id=cell_id(f"{cl.id}|{purpose}", m.id, Condition(True)),
                contract_id=f"investigate.{purpose}", model=m.id, condition=None, trial=0, attempt=1, ts_start=ts,
                request=body, parsed=parsed, result=OracleResult(Outcome.PASS if parsed.status == 200 else Outcome.ERROR),
                cost_usd=cost, cluster_id=cl.id,
            ))
            return parsed.content
        return complete

    console.print(f"[bold]{cl.id}[/bold] {cl.title}: shrinking on {model} ({worst['k']}/{worst['n']} failed)")
    shr = await shrink(spec, worst["rate"], worst["n"], worst["k"], involved_fields(cl.examples), runner)
    (cdir / "minimal.json").write_text(json.dumps({"spec": shr.spec.to_json(), "rate": shr.rate, "n": shr.n, "k": shr.k,
                                                   "steps": shr.steps, "model": model}, indent=2, ensure_ascii=False))
    for step in shr.steps:
        console.print(f"  shrink: {step['candidate']}: {step['k']}/{step['n']} -> {'kept' if step['accepted'] else 'rejected'}")
    failing = [e for e in shr.examples if e["reproduces"]] or [
        {"codes": e["codes"], "finish_reason": e["finish_reason"], "content_head": e["content_head"],
         "tool_calls": e["tool_calls"], "detail": e["detail"]} for e in cl.examples]
    passing = [e for e in shr.examples if not e["reproduces"]] or [
        {"codes": r["codes"], "content_head": (r.get("parsed") or {}).get("content", "")[:400],
         "tool_calls": (r.get("parsed") or {}).get("tool_calls"), "finish_reason": (r.get("parsed") or {}).get("finish_reason")}
        for r in records if r["contract_id"] == cl.contract_id and r["outcome"] == "pass"][:3]

    title, accepted, all_results, rejected = None, None, [], []
    for rnd in (1, 2):
        prop = await propose(llm(ultra, "hypothesize"), ultra.id, shr.spec, shr.rate, failing, passing, rejected or None)
        title = title or prop["title"]
        (cdir / f"hypotheses_round{rnd}.json").write_text(json.dumps(prop, indent=2, ensure_ascii=False))
        results = []
        hyps = prop["hypotheses"]
        for i, h in enumerate(hyps, 1):
            remaining = cluster_budget - runner.spent
            if remaining <= 0:
                console.print("  cluster budget spent; stopping")
                break
            hid = f"{cl.id}_h{rnd}{i}"
            per_exp = remaining / (len(hyps) - i + 1)
            path = settings.root / "experiments" / f"{hid}.py"
            wrote = await write_experiment(llm(coder, "experiment_code"), coder.id, cl.id, hid, h, path)
            out = store.dir / "experiments" / f"{hid}.json"
            run_info: dict[str, Any] = {"gate": wrote}
            if wrote["ok"]:
                run_info["pytest"] = await asyncio.to_thread(experiment_runner, path, settings.root, {
                    "SECOND_OPINION_INVESTIGATION": str(store.dir),
                    "SECOND_OPINION_EXPERIMENT_BUDGET_USD": f"{per_exp:.4f}"})
            if out.exists():
                res = json.loads(out.read_text())
                runner.spent += float(res["a"].get("cost_usd", 0)) + float(res["b"].get("cost_usd", 0))
            else:
                console.print(f"  {hid}: experiment file unusable ({'; '.join(wrote['problems']) or 'pytest failed'}); running arms directly")
                res = await _inline_arms(runner, shr.spec, h, hid, store.dir / "experiments", arm_n)
                run_info["fallback"] = True
            results.append({"hypothesis_id": hid, **h, "a": res["a"], "b": res["b"], "run": run_info})
        decided = decide(results, settings.defaults["alpha"])
        for r in decided:
            console.print(f"  {r['hypothesis_id']}: {r['factor']} {json.dumps(r['a']['level'])} {r['a']['k']}/{r['a']['n']} vs "
                          f"{json.dumps(r['b']['level'])} {r['b']['k']}/{r['b']['n']}, p_holm={r['p_holm']:.3g} -> "
                          f"{'ACCEPTED' if r['accepted'] else 'rejected'}")
        all_results += decided
        winners = sorted((r for r in decided if r["accepted"]), key=lambda r: r["p_holm"])
        if winners:
            accepted = winners[0]
            break
        rejected += [{k: r[k] for k in ("factor", "level_a", "level_b", "rationale", "p_holm")} for r in decided]

    cell_arms = [{"label": f"{c['model'].split('/')[-1]} · {Condition.from_dict(c['condition']).label()}", "n": c["n"],
                  "k": c["k"], "rate": c["rate"], "lo": c["wilson_lo"], "hi": c["wilson_hi"]} for c in cl.cells]
    return {
        "id": cl.id,
        "title": title or cl.title,
        "severity": cl.severity.name,
        "owner": "unknown",
        "status": "investigated",
        "contract_id": cl.contract_id,
        "codes": list(cl.codes),
        "models": cl.models,
        "summary": (f"{cl.failures} of {cl.trials} screening trials in the affected cells failed ({cl.rate:.1%}). "
                    f"The minimal request fails {shr.k} of {shr.n} on {model.split('/')[-1]}."
                    + ("" if accepted else " No tested cause was accepted.")),
        "arms": [_arm_view(accepted["a"]), _arm_view(accepted["b"])] if accepted else cell_arms,
        "factor": cause(accepted) if accepted else None,
        "p_value": accepted["p_holm"] if accepted else None,
        "workaround": workaround(accepted) if accepted else None,
        "repro": repro(shr.spec.request()),
        "evidence": [evidence_view(e) for e in cl.examples[:2]],
        "minimal": {"spec": shr.spec.to_json(), "rate": shr.rate, "n": shr.n, "k": shr.k, "model": model},
        "shrink_steps": shr.steps,
        "hypotheses": [{k: r.get(k) for k in ("hypothesis_id", "factor", "level_a", "level_b", "rationale", "p", "p_holm",
                                               "effect", "accepted")} | {"a": _arm_view(r["a"]), "b": _arm_view(r["b"])}
                       for r in all_results],
        "spent_usd": round(runner.spent, 4),
    }


async def investigate(
    settings: Settings,
    console: Console,
    run_dirs: list[Path],
    *,
    hypothesis_model: str | None = None,
    code_model: str | None = None,
    yes: bool = False,
    max_clusters: int = DEFAULT_MAX_CLUSTERS,
    cluster_budget: float = DEFAULT_CLUSTER_BUDGET_USD,
    transport: httpx.AsyncBaseTransport | None = None,
    experiment_runner: Callable[..., dict[str, Any]] = run_experiment,
    arm_n: int = ARM_N,
) -> int:
    records = list(analysis.iter_records(run_dirs))
    clusters = build_clusters(records)
    if not clusters:
        console.print("No failure clusters in these runs; nothing to investigate.")
        return 0
    selected = clusters[:max_clusters]
    console.print(clusters_table(clusters))
    async with ApiClient.for_provider(settings, "tokenfactory", transport) as tf:
        listing = await tf.get("models")
        if listing.status != 200:
            console.print(f"[red]GET /v1/models failed: {listing.status}[/red]")
            return 1
        found = [m for m in discover(listing.body, settings) if m.runnable]
        by_id = {m.id: m for m in found}
        ultra = by_id.get(hypothesis_model) if hypothesis_model else _role(found, "ultra")
        coder = by_id.get(code_model) if code_model else _role(found, "super")
        if not ultra or not coder:
            console.print("[red]Need a priced hypothesis model (Ultra) and code model (Super); see --hypothesis-model/--code-model.[/red]")
            return 2
        per_cluster_calls = MAX_CANDIDATES * K + 6 * 2 * arm_n
        console.print(
            f"Investigate {len(selected)} cluster(s): per cluster up to {MAX_CANDIDATES * K} shrink calls, up to 6 experiments "
            f"of 2 x {arm_n} trials, plus a few {ultra.id} and {coder.id} calls (at most ~{per_cluster_calls} calls), capped at "
            f"${cluster_budget:.2f} per cluster: at most ${len(selected) * cluster_budget:.2f} in total.")
        if not yes:
            console.print("[yellow]Paid stage: nothing was sent. Re-run with --yes once this estimate is approved.[/yellow]")
            return 0
        budget = BudgetGuard(settings.budget_usd, settings.ledger_path)
        store = RunStore.create(settings.runs_dir, "investigate")
        store.write_plan({"stage": "investigate", "created": utc_now(), "git": git_info(settings.root),
                          "runs": [str(d) for d in run_dirs], "clusters": [c.as_dict() for c in selected],
                          "hypothesis_model": ultra.id, "code_model": coder.id, "cluster_budget_usd": cluster_budget})
        findings = []
        for cl in selected:
            finding = await investigate_cluster(cl, settings=settings, store=store, tf=tf, budget=budget, ultra=ultra,
                                                coder=coder, cluster_budget=cluster_budget, records=records, console=console,
                                                experiment_runner=experiment_runner, arm_n=arm_n)
            findings.append(finding)
            with (store.dir / "findings.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(finding, ensure_ascii=False, default=str) + "\n")
    store.write_summary({"run_id": store.run_id, "finished": utc_now(), "clusters": len(findings),
                         "accepted": sum(1 for f in findings if f["factor"]),
                         "spent_usd": round(sum(f["spent_usd"] for f in findings), 4)})
    console.print(f"{len(findings)} findings, {sum(1 for f in findings if f['factor'])} with an accepted cause. "
                  f"Results: {store.dir / 'findings.jsonl'}")
    return 0
