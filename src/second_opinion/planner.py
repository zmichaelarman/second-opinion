"""Planner (CLAUDE.md §8-§9): cells and work items per stage, cost estimates, and the
pre-registration hash written into TEST_PLAN.md."""

from __future__ import annotations

import hashlib
import json
import random
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from rich.table import Table

from . import analysis
from .contracts import base
from .contracts.base import BaseContract, Condition, cell_id
from .discovery import DiscoveredModel
from .oracles import ERROR_CODES, SEVERITY, SOFT_CODES

FALLBACK_TOKENS = {False: (400, 300), True: (400, 2000), None: (400, 2000)}  # §8
SHUFFLE_SEED = 20261023

FAMILIES = {
    "F1": {
        "name": "streaming vs non-streaming",
        "scope": "each T3 variant at thinking on, per model",
        "arms": "(thinking on, stream on) vs (thinking on, stream off)",
    },
    "F2": {
        "name": "thinking on vs off",
        "scope": "each T3 and T5 variant, per model",
        "arms": "T3: (on, no stream) vs (off, no stream); T5: wrapped on vs wrapped off",
    },
    "F3": {
        "name": "wrapped vs bare json_schema shape",
        "scope": "T5.flat_basic, per model",
        "arms": "wrapped vs bare, both at thinking off",
    },
}
TRUST_COLUMNS = {
    "structured_output": "Structured output (T5, thinking off)",
    "structured_output_reasoning": "Structured output with reasoning (T5, thinking on)",
    "tool_args": "Tool arguments (T3, non-stream)",
    "tool_args_stream": "Tool arguments, streaming (T3 stream)",
    "reasoning_control": "Reasoning control (T2)",
    "reasoning_leak": "Reasoning leak (T1)",
}
COLOR_RULES = {
    "red": "any cell whose failure-rate Wilson lower bound exceeds 1%",
    "amber": "failures observed, but not red",
    "green": "zero fatal failures; shown as failure rate < rule_of_three(n_pooled)",
}
OUTCOME_RULE = "Any ERROR-class condition gives ERROR. Otherwise any fatal code gives FAIL. Otherwise PASS."
ESCALATION_RULE = "A cell qualifies if it had at least one fatal failure in screening, or is a core condition."


@dataclass(frozen=True)
class Cell:
    contract_id: str
    model: str
    condition: Condition

    @property
    def id(self) -> str:
        return cell_id(self.contract_id, self.model, self.condition)


@dataclass(frozen=True)
class WorkItem:
    cell: Cell
    trial: int


class TokenEstimates:
    """Median tokens per (contract, thinking) from the latest pilot, with §8 fallbacks."""

    def __init__(self, medians: dict[str, dict[str, float]] | None = None, source: str | None = None):
        self.medians = medians or {}
        self.source = source

    @classmethod
    def from_runs(cls, runs_dir: Path) -> "TokenEstimates":
        for run_dir in reversed(analysis.stage_runs(runs_dir, "pilot")):
            summary = run_dir / "summary.json"
            if summary.exists():
                medians = json.loads(summary.read_text()).get("token_medians")
                if medians:
                    return cls(medians, str(run_dir.name))
        return cls()

    def get(self, contract_id: str, thinking: bool | None) -> tuple[float, float]:
        m = self.medians.get(analysis.token_key(contract_id, thinking))
        return (m["prompt"], m["completion"]) if m else FALLBACK_TOKENS[thinking]


def _rounds(items: list[WorkItem]) -> list[WorkItem]:
    """Order by trial index, shuffled within each round, so a partial run covers cells evenly."""
    rng = random.Random(SHUFFLE_SEED)
    by_trial: dict[int, list[WorkItem]] = defaultdict(list)
    for it in items:
        by_trial[it.trial].append(it)
    out: list[WorkItem] = []
    for trial in sorted(by_trial):
        batch = sorted(by_trial[trial], key=lambda it: (it.cell.id, it.trial))
        rng.shuffle(batch)
        out.extend(batch)
    return out


def cheapest(models: list[DiscoveredModel]) -> DiscoveredModel:
    return min(models, key=lambda m: (m.price.output, m.price.input, m.id))  # type: ignore[union-attr]


def plan_pilot(contracts: list[BaseContract], models: list[DiscoveredModel]) -> list[WorkItem]:
    """1 trial per condition on the cheapest model, plus each T3 condition once on Super (3 per variant)."""
    low = cheapest(models)
    items = [WorkItem(Cell(c.id, low.id, cond), 0) for c in contracts for cond in c.conditions]
    sup = next((m for m in models if m.role == "super"), None)
    if sup and sup.id != low.id:
        items += [WorkItem(Cell(c.id, sup.id, cond), 0) for c in contracts if c.tier == 3 for cond in c.conditions]
    return items


def plan_screen(
    contracts: list[BaseContract], models: list[DiscoveredModel], n: int, max_n: dict[str, int] | None = None
) -> list[WorkItem]:
    max_n = max_n or {}
    items = [
        WorkItem(Cell(c.id, m.id, cond), trial)
        for m in models
        for trial in range(min(n, max_n.get(m.id, n)))
        for c in contracts
        for cond in c.conditions
    ]
    return _rounds(items)


def plan_escalate(
    contracts: list[BaseContract],
    models: list[DiscoveredModel],
    screened: dict[str, analysis.CellStats],
    n_target: int,
    max_n: dict[str, int] | None = None,
) -> list[WorkItem]:
    """Top qualifying cells up to n_target final trials, continuing the trial numbering."""
    max_n = max_n or {}
    items = []
    for m in models:
        target = min(n_target, max_n.get(m.id, n_target))
        for c in contracts:
            for cond in c.conditions:
                cell = Cell(c.id, m.id, cond)
                stats = screened.get(cell.id)
                if cond not in c.core_conditions and not (stats and stats.fails > 0):
                    continue
                done = (stats.n + stats.errors) if stats else 0
                start = (stats.max_trial + 1) if stats else 0
                items += [WorkItem(cell, start + i) for i in range(max(0, target - done))]
    return _rounds(items)


def estimate_cost(items: list[WorkItem], models: dict[str, DiscoveredModel], contracts: dict[str, BaseContract],
                  est: TokenEstimates) -> dict[tuple[str, str], tuple[int, float]]:
    """(model, tier) -> (calls, estimated USD)."""
    rows: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0, 0.0])
    for it in items:
        price = models[it.cell.model].price
        assert price is not None
        pt, ct = est.get(it.cell.contract_id, it.cell.condition.thinking)
        key = (it.cell.model, f"T{contracts[it.cell.contract_id].tier}")
        rows[key][0] += 1
        rows[key][1] += price.cost(pt, ct)
    return {k: (int(v[0]), v[1]) for k, v in sorted(rows.items())}


def cost_table(stage: str, rows: dict[tuple[str, str], tuple[int, float]], est: TokenEstimates) -> Table:
    table = Table(title=f"Plan: {stage} (token estimates: {est.source or 'fallbacks, no pilot yet'})")
    for col in ("Model", "Contracts", "Calls", "Est. $"):
        table.add_column(col, justify="right" if col in ("Calls", "Est. $") else "left")
    for (model, tier), (calls, usd) in rows.items():
        table.add_row(model, tier, str(calls), f"{usd:.2f}")
    table.add_row("[bold]Total[/bold]", "", f"[bold]{sum(c for c, _ in rows.values())}[/bold]",
                  f"[bold]{sum(u for _, u in rows.values()):.2f}[/bold]")
    return table


# Pre-registration -------------------------------------------------------------------------


def registry(defaults: dict[str, Any], contracts: list[BaseContract] | None = None) -> dict[str, Any]:
    from .contracts.registry import all_contracts

    contracts = contracts if contracts is not None else all_contracts()
    return {
        "version": 1,
        "shared_rules": {
            "system_message": base.SYSTEM_MESSAGE,
            "temperature": base.TEMPERATURE,
            "top_p": base.TOP_P,
            "max_tokens_thinking_off": base.MAX_TOKENS_THINKING_OFF,
            "max_tokens_thinking_on_or_omitted": base.MAX_TOKENS_THINKING_ON,
            "thinking_switch": base.THINKING_SWITCH,
            "answer_normalization": "content.strip().lower().strip(\".!\\\"' \")",
        },
        "contracts": [c.describe() for c in contracts],
        "n_screen": defaults["n_screen"],
        "n_escalate": defaults["n_escalate"],
        "alpha": defaults["alpha"],
        "escalation_rule": ESCALATION_RULE,
        "outcome_rule": OUTCOME_RULE,
        "failure_codes": failure_code_table(),
        "families": FAMILIES,
        "trust_matrix": {"columns": TRUST_COLUMNS, "colors": COLOR_RULES},
    }


def failure_code_table() -> dict[str, dict[str, Any]]:
    table = {code: {"fatal": code not in SOFT_CODES, "severity": sev.name} for code, sev in SEVERITY.items()}
    for code in sorted(ERROR_CODES):
        table[code] = {"fatal": None, "severity": None, "outcome": "error"}
    table["TRUNCATED"] = {"fatal": False, "severity": None}
    table["REFUSAL"] = {"fatal": False, "severity": None}
    return table


def registry_hash(reg: dict[str, Any]) -> str:
    blob = json.dumps(reg, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


HASH_LINE = re.compile(r"Registry SHA-256: `([0-9a-f]{64})`")


def render_test_plan(defaults: dict[str, Any], model_ids: list[str] | None = None) -> str:
    from .contracts.registry import all_contracts

    reg = registry(defaults)
    digest = registry_hash(reg)
    contracts = all_contracts()
    n_cond = sum(len(c.conditions) for c in contracts)
    lines = [
        "# Second Opinion: test plan (pre-registration)",
        "",
        f"Generated {date.today().isoformat()} by `second-opinion plan --write` from the contract registry.",
        "",
        f"Registry SHA-256: `{digest}`",
        "",
        "The hash covers every contract, condition and exact request body, the shared request rules, n, alpha, "
        "the comparison families, the trust-matrix rules and the failure codes. The screening stage refuses to run "
        "unless the git tag `prereg-v1` exists and this hash matches the code.",
        "",
        "## Design",
        "",
        "- **Subjects:** every Nemotron chat model that `GET /v1/models` lists and `config/prices.yaml` prices"
        + (f" (at plan time: {', '.join(f'`{m}`' for m in model_ids)})." if model_ids else "."),
        f"- **Sampling:** temperature {base.TEMPERATURE}, top_p {base.TOP_P}. max_tokens "
        f"{base.MAX_TOKENS_THINKING_OFF} with thinking off, {base.MAX_TOKENS_THINKING_ON} with thinking on or omitted, "
        "unless a condition sets it.",
        f"- **Thinking switch:** `{base.THINKING_SWITCH}` (see CLAUDE.md §3 for how this was chosen).",
        f"- **System message:** `{base.SYSTEM_MESSAGE}`",
        "- **Stages:** `pilot` (1 trial per condition; excluded from analysis), "
        f"`screen` (n={defaults['n_screen']} per cell), `escalate` (top up to n={defaults['n_escalate']}). "
        + ESCALATION_RULE,
        f"- **Outcome rule:** {OUTCOME_RULE} ERROR trials are excluded from denominators and reported separately.",
        f"- **Significance:** two-sided Fisher exact tests, Holm-corrected within each family, adjusted alpha = "
        f"{defaults['alpha']}.",
        "- **Pass/fail** is decided by deterministic oracles. No LLM grades any trial.",
        "",
        f"## Contracts ({len(contracts)} contracts, {n_cond} conditions)",
        "",
        "| ID | Severity | Conditions | Core | Hypothesis |",
        "|---|---|---|---|---|",
    ]
    for c in contracts:
        conds = "<br>".join(cond.label() for cond in c.conditions)
        core = "<br>".join(cond.label() for cond in c.conditions if cond in c.core_conditions) or "-"
        lines.append(f"| `{c.id}` | {c.severity.name} | {conds} | {core} | {c.hypothesis} |")
    lines += ["", "## Comparison families", "", "| Family | Comparison | Scope | Arms |", "|---|---|---|---|"]
    lines += [f"| {k} | {v['name']} | {v['scope']} | {v['arms']} |" for k, v in FAMILIES.items()]
    lines += ["", "## Trust matrix", "", "Rows are models. Columns:", ""]
    lines += [f"- {v}" for v in TRUST_COLUMNS.values()]
    lines += ["", "Each cell takes the color of the worst condition in its column:", ""]
    lines += [f"- **{k.capitalize()}:** {v}" for k, v in COLOR_RULES.items()]
    lines += ["", "## Failure codes", "", "| Code | Fatal | Severity |", "|---|---|---|"]
    for code, info in failure_code_table().items():
        fatal = "ERROR outcome" if info.get("outcome") == "error" else ("yes" if info["fatal"] else "soft")
        lines.append(f"| `{code}` | {fatal} | {info['severity'] or '-'} |")
    lines += [
        "",
        "## Not tested",
        "",
        "Model intelligence, load or stress behavior, and headline latency (latency is recorded, not headlined). "
        "Tier 2 and 3 probes (CLAUDE.md §15) are exploratory and outside this plan.",
        "",
    ]
    return "\n".join(lines)


def plan_file_hash(text: str) -> str | None:
    m = HASH_LINE.search(text)
    return m.group(1) if m else None


def prereg_check(root: Path, defaults: dict[str, Any]) -> tuple[bool, str]:
    """True if tag prereg-v1 exists and both the working and the tagged TEST_PLAN.md match the code."""
    current = registry_hash(registry(defaults))

    def git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)

    if git("rev-parse", "-q", "--verify", "refs/tags/prereg-v1").returncode != 0:
        return False, "git tag prereg-v1 not found. Commit TEST_PLAN.md, then run: git tag prereg-v1"
    plan = root / "TEST_PLAN.md"
    if not plan.exists() or plan_file_hash(plan.read_text()) != current:
        return False, "TEST_PLAN.md is missing or its registry hash no longer matches the code."
    tagged = git("show", "prereg-v1:TEST_PLAN.md")
    if tagged.returncode != 0 or plan_file_hash(tagged.stdout) != current:
        return False, "The TEST_PLAN.md in tag prereg-v1 does not match the code's registry hash."
    return True, f"pre-registration ok ({current[:12]})"
