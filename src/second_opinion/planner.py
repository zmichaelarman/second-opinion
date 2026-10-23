from __future__ import annotations

import json
import re
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from second_opinion.config import AppConfig, REPO_ROOT
from second_opinion.contracts.base import Condition, Contract, compute_cell_id
from second_opinion.contracts.registry import (
    compute_registry_hash,
    get_tier1_contracts,
    serialize_registry,
)
from second_opinion.contracts.t3_tool_args import T3_VARIANTS


@dataclass(frozen=True)
class PlannedCell:
    cell_id: str
    contract_id: str
    model: str
    condition: Condition
    n_trials: int
    is_core: bool
    est_prompt_tokens: int
    est_completion_tokens: int
    est_cost_usd: float


def load_pilot_token_medians(runs_root: Path | None = None) -> dict[tuple[str, bool], tuple[int, int]]:
    """Compute median (prompt_tokens, completion_tokens) per (contract_id, thinking_enabled) from pilot runs."""
    root = runs_root or (REPO_ROOT / "runs")
    if not root.exists():
        return {}
    buckets_in: dict[tuple[str, bool], list[int]] = {}
    buckets_out: dict[tuple[str, bool], list[int]] = {}
    for d in sorted(root.glob("*-pilot*")):
        calls_file = d / "calls.jsonl"
        if not calls_file.exists():
            continue
        with calls_file.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    rec = json.loads(s)
                except json.JSONDecodeError:
                    continue
                usage = rec.get("usage")
                if not isinstance(usage, dict):
                    continue
                p_tok = int(usage.get("prompt_tokens", 0) or 0)
                c_tok = int(usage.get("completion_tokens", 0) or 0)
                if p_tok <= 0 and c_tok <= 0:
                    continue
                cid = str(rec.get("contract_id", ""))
                cond = rec.get("condition") or {}
                thinking_on = cond.get("thinking") is not False
                key = (cid, thinking_on)
                buckets_in.setdefault(key, []).append(p_tok)
                buckets_out.setdefault(key, []).append(c_tok)

    medians: dict[tuple[str, bool], tuple[int, int]] = {}
    for key in buckets_in:
        med_in = int(statistics.median(buckets_in[key]))
        med_out = int(statistics.median(buckets_out[key]))
        medians[key] = (med_in, med_out)
    return medians


def estimate_tokens_for_condition(
    contract_id: str,
    cond: Condition,
    pilot_medians: dict[tuple[str, bool], tuple[int, int]] | None = None,
) -> tuple[int, int]:
    thinking_on = cond.thinking is not False
    if pilot_medians and (contract_id, thinking_on) in pilot_medians:
        return pilot_medians[(contract_id, thinking_on)]
    # Fallbacks (§8): thinking off ≈ 400 in / 300 out; thinking on ≈ 400 in / 2000 out
    if cond.thinking is False:
        return (400, 300)
    if cond.max_tokens is not None and cond.max_tokens < 2000:
        return (400, min(2000, cond.max_tokens))
    return (400, 2000)


def parse_max_n_overrides(max_n_args: list[str] | None) -> dict[str, int]:
    overrides: dict[str, int] = {}
    for item in max_n_args or []:
        if "=" not in item:
            continue
        m_id, val = item.split("=", 1)
        overrides[m_id.strip()] = int(val.strip())
    return overrides


def select_cheapest_and_super_models(models: list[str], config: AppConfig) -> tuple[str, str]:
    if not models:
        return ("nvidia/Nemotron-3_5-Lightning", config.code_model)
    cheapest = min(
        models,
        key=lambda m: config.get_price(m).input_per_1m + config.get_price(m).output_per_1m,
    )
    super_model = next((m for m in models if "super" in m.lower()), models[min(1, len(models) - 1)])
    return cheapest, super_model


def load_screening_fatal_cells(runs_root: Path | None = None) -> tuple[set[str], dict[str, int]]:
    """Inspect existing screen/escalate runs to find cells with >=1 fatal failure and completed trial counts."""
    root = runs_root or (REPO_ROOT / "runs")
    fatal_cells: set[str] = set()
    completed_counts: dict[str, set[int]] = {}
    if not root.exists():
        return fatal_cells, {}

    for d in sorted(root.iterdir()):
        if not d.is_dir() or not (d.name.endswith("-screen") or "-screen-" in d.name or d.name.endswith("-escalate") or "-escalate-" in d.name):
            continue
        calls_file = d / "calls.jsonl"
        if not calls_file.exists():
            continue
        with calls_file.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    rec = json.loads(s)
                except json.JSONDecodeError:
                    continue
                if not rec.get("is_final", True):
                    continue
                cell_id = str(rec.get("cell_id", ""))
                trial = int(rec.get("trial", 0))
                if not cell_id:
                    continue
                completed_counts.setdefault(cell_id, set()).add(trial)
                if rec.get("outcome") == "fail":
                    fatal_cells.add(cell_id)

    counts = {cid: len(trials) for cid, trials in completed_counts.items()}
    return fatal_cells, counts


def build_stage_plan(
    stage: str,
    models: list[str],
    config: AppConfig,
    contracts: list[Contract] | None = None,
    max_n_overrides: dict[str, int] | None = None,
    runs_root: Path | None = None,
) -> list[PlannedCell]:
    contracts = contracts if contracts is not None else get_tier1_contracts()
    max_n_overrides = max_n_overrides or {}
    pilot_medians = load_pilot_token_medians(runs_root)
    cells: list[PlannedCell] = []

    if stage == "pilot":
        # 1 trial per condition on cheapest model + 3 trials of each T3 variant on Super (~50 calls)
        cheapest, super_model = select_cheapest_and_super_models(models, config)
        for c in contracts:
            for cond in c.conditions:
                cid = compute_cell_id(c.id, cheapest, cond)
                in_tok, out_tok = estimate_tokens_for_condition(c.id, cond, pilot_medians)
                cost = config.estimate_cost(cheapest, in_tok, out_tok)
                cells.append(
                    PlannedCell(
                        cell_id=cid,
                        contract_id=c.id,
                        model=cheapest,
                        condition=cond,
                        n_trials=1,
                        is_core=(cond in c.core_conditions),
                        est_prompt_tokens=in_tok,
                        est_completion_tokens=out_tok,
                        est_cost_usd=cost,
                    )
                )
        t3_contract = next((c for c in contracts if c.id == "T3"), None)
        if t3_contract is not None:
            for variant in T3_VARIANTS:
                for stream_flag in (False, True) if cheapest == super_model else (False,):
                    pass
                cond = Condition(thinking=True, stream=False, variant=variant)
                cid = compute_cell_id("T3", super_model, cond)
                in_tok, out_tok = estimate_tokens_for_condition("T3", cond, pilot_medians)
                n_t = 3
                cost = n_t * config.estimate_cost(super_model, in_tok, out_tok)
                cells.append(
                    PlannedCell(
                        cell_id=cid,
                        contract_id="T3",
                        model=super_model,
                        condition=cond,
                        n_trials=n_t,
                        is_core=(cond in t3_contract.core_conditions),
                        est_prompt_tokens=in_tok,
                        est_completion_tokens=out_tok,
                        est_cost_usd=cost,
                    )
                )
        return cells

    if stage == "screen":
        for m in models:
            n_target = max_n_overrides.get(m, config.n_screen)
            for c in contracts:
                for cond in c.conditions:
                    cid = compute_cell_id(c.id, m, cond)
                    in_tok, out_tok = estimate_tokens_for_condition(c.id, cond, pilot_medians)
                    cost = n_target * config.estimate_cost(m, in_tok, out_tok)
                    cells.append(
                        PlannedCell(
                            cell_id=cid,
                            contract_id=c.id,
                            model=m,
                            condition=cond,
                            n_trials=n_target,
                            is_core=(cond in c.core_conditions),
                            est_prompt_tokens=in_tok,
                            est_completion_tokens=out_tok,
                            est_cost_usd=cost,
                        )
                    )
        return cells

    if stage == "escalate":
        fatal_cells, existing_counts = load_screening_fatal_cells(runs_root)
        for m in models:
            n_cap = max_n_overrides.get(m, config.n_escalate)
            for c in contracts:
                for cond in c.conditions:
                    cid = compute_cell_id(c.id, m, cond)
                    is_core = cond in c.core_conditions
                    qualifies = is_core or (cid in fatal_cells)
                    if not qualifies:
                        continue
                    already = existing_counts.get(cid, 0)
                    additional = max(0, n_cap - already)
                    if additional <= 0:
                        continue
                    in_tok, out_tok = estimate_tokens_for_condition(c.id, cond, pilot_medians)
                    cost = additional * config.estimate_cost(m, in_tok, out_tok)
                    cells.append(
                        PlannedCell(
                            cell_id=cid,
                            contract_id=c.id,
                            model=m,
                            condition=cond,
                            n_trials= n_cap,
                            is_core=is_core,
                            est_prompt_tokens=in_tok,
                            est_completion_tokens=out_tok,
                            est_cost_usd=cost,
                        )
                    )
        return cells

    raise ValueError(f"Unknown stage: {stage}")


def print_cost_estimate_table(
    stage: str,
    cells: list[PlannedCell],
    console: Console | None = None,
    existing_counts: dict[str, int] | None = None,
) -> float:
    console = console or Console()
    existing_counts = existing_counts or {}
    table = Table(title=f"Second Opinion — Planned Spend Estimate ({stage})")
    table.add_column("Model", style="cyan")
    table.add_column("Contract", style="magenta")
    table.add_column("Cells", justify="right")
    table.add_column("Planned Calls", justify="right")
    table.add_column("Estimated Cost ($)", justify="right", style="green")

    grouped: dict[tuple[str, str], dict[str, float | int]] = {}
    total_calls = 0
    total_cost = 0.0

    for cell in cells:
        already = existing_counts.get(cell.cell_id, 0) if stage == "escalate" else 0
        rem_calls = max(0, cell.n_trials - already)
        key = (cell.model, cell.contract_id)
        entry = grouped.setdefault(key, {"cells": 0, "calls": 0, "cost": 0.0})
        entry["cells"] = int(entry["cells"]) + 1
        entry["calls"] = int(entry["calls"]) + rem_calls
        entry["cost"] = float(entry["cost"]) + cell.est_cost_usd
        total_calls += rem_calls
        total_cost += cell.est_cost_usd

    for (model, contract_id), stats in sorted(grouped.items()):
        table.add_row(
            model,
            contract_id,
            str(stats["cells"]),
            str(stats["calls"]),
            f"${stats['cost']:.4f}",
        )

    table.add_section()
    table.add_row("TOTAL", "ALL", str(len(cells)), str(total_calls), f"${total_cost:.4f}", style="bold")
    console.print(table)
    return total_cost


def generate_test_plan_markdown(
    config: AppConfig,
    contracts: list[Contract] | None = None,
) -> str:
    contracts = contracts if contracts is not None else get_tier1_contracts()
    reg_hash = compute_registry_hash(contracts)
    reg_data = serialize_registry(contracts)

    lines: list[str] = [
        "# Second Opinion — Pre-Registered Test Plan (`prereg-v1`)",
        "",
        f"- **Registry SHA-256:** `{reg_hash}`",
        f"- **Screening sample size (`n_screen`):** `{config.n_screen}` trials per cell",
        f"- **Escalation sample size (`n_escalate`):** `{config.n_escalate}` total trials per cell",
        f"- **Family-wise significance threshold (`alpha`):** `{config.alpha}` (two-sided Fisher exact test with Holm-Bonferroni correction)",
        f"- **Sampling defaults:** `temperature={config.temperature}`, `top_p={config.top_p}`",
        "",
        "---",
        "",
        "## 1. Pre-Registered Comparison Families",
        "",
        "- **F1 (Streaming vs Non-Streaming):** `T3` variants (`leading_zero`, `numeric_types`, `bool_null_nested`, `markup`) at `thinking=True`, `stream=True` vs `stream=False`, per model. Holm correction within family ($\\alpha = 0.01$).",
        "- **F2 (Thinking On vs Off):** `T3` variants (`stream=False`) and `T5` variants (`shape=wrapped`, `stream=False`), `thinking=True` vs `thinking=False`, per model. Holm correction within family ($\\alpha = 0.01$).",
        "- **F3 (Wrapped vs Bare Schema Shape):** `T5.flat_basic` at `thinking=False`, `shape=wrapped` vs `shape=bare`, per model. Holm correction within family ($\\alpha = 0.01$).",
        "",
        "---",
        "",
        "## 2. Trust Matrix Color Rules",
        "",
        "Each model row is evaluated across 6 capability columns (`T5 thinking=False`, `T5 thinking=True`, `T3 non-stream`, `T3 stream`, `T2`, `T1`). Cell color is determined by the worst constituent cell in that column:",
        "",
        "- **Red:** Any constituent cell whose fatal failure-rate 95% Wilson confidence interval lower bound exceeds `1.0%` (`> 0.01`).",
        "- **Amber:** One or more fatal failures observed (`k >= 1`), but all constituent cells have Wilson lower bound $\\le 1.0\\%$.",
        "- **Green:** Zero fatal failures observed (`k = 0`). Displays upper bound `failure rate < 3 / n_pooled` (Rule of Three).",
        "",
        "---",
        "",
        "## 3. Contract Registry (Tier 1)",
        "",
    ]

    total_conds = 0
    for c_dict in reg_data:
        conds = c_dict["conditions"]
        cores = {json.dumps(x, sort_keys=True) for x in c_dict["core_conditions"]}
        total_conds += len(conds)
        lines.append(f"### `{c_dict['id']}` — {c_dict['title']} (Severity: `{c_dict['severity']}`)")
        lines.append("")
        lines.append(f"**Hypothesis:** {c_dict['hypothesis']}")
        lines.append("")
        lines.append("| # | Variant | Thinking | Stream | Max Tokens | Shape | Core Escalation |")
        lines.append("|---|---|---|---|---|---|---|")
        for idx, cond in enumerate(conds, start=1):
            is_core = json.dumps(cond, sort_keys=True) in cores
            lines.append(
                f"| {idx} | `{cond['variant']}` | `{cond['thinking']}` | `{cond['stream']}` | "
                f"`{cond['max_tokens']}` | `{cond['shape']}` | `{'yes' if is_core else 'no'}` |"
            )
        lines.append("")

    lines.append(f"**Total Conditions per Model:** `{total_conds}`")
    lines.append("")
    return "\n".join(lines)


def verify_prereg(repo_root: Path | None = None, contracts: list[Contract] | None = None) -> tuple[bool, list[str]]:
    """Verify that TEST_PLAN.md exists, matches current registry hash, and git tag prereg-v1 exists."""
    root = repo_root or REPO_ROOT
    errors: list[str] = []
    plan_file = root / "TEST_PLAN.md"
    expected_hash = compute_registry_hash(contracts)

    if not plan_file.exists():
        errors.append("TEST_PLAN.md does not exist. Run `second-opinion plan --write` first.")
    else:
        text = plan_file.read_text(encoding="utf-8")
        m = re.search(r"Registry SHA-256:\*\*\s*`([0-9a-fA-F]{64})`", text)
        if not m:
            errors.append("TEST_PLAN.md is missing the Registry SHA-256 header.")
        elif m.group(1).lower() != expected_hash.lower():
            errors.append(
                f"TEST_PLAN.md registry hash ({m.group(1)[:12]}...) does not match code ({expected_hash[:12]}...)."
            )

    try:
        res = subprocess.run(
            ["git", "tag", "-l", "prereg-v1"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        tags = [t.strip() for t in res.stdout.splitlines() if t.strip()]
        if "prereg-v1" not in tags:
            errors.append("Git tag `prereg-v1` not found. Commit TEST_PLAN.md and run `git tag prereg-v1`.")
    except Exception as exc:
        errors.append(f"Unable to verify git tag `prereg-v1`: {exc}")

    return (len(errors) == 0, errors)
