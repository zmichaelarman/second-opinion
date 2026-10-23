"""`second-opinion report` (CLAUDE.md §12): docs/index.html, docs/findings.md, docs/data/ and
FEEDBACK_DRAFT.md, all built from raw run data."""

from __future__ import annotations

import gzip
import json
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .. import analysis
from ..contracts import base as contract_base
from ..contracts import registry
from ..contracts.base import BaseContract, Condition
from ..investigator.cluster import build_clusters
from ..planner import COLOR_RULES, FAMILIES, TRUST_COLUMNS, plan_file_hash
from ..planner import registry as build_registry
from ..planner import registry_hash
from ..stats import fisher_two_sided, holm, rule_of_three
from ..store import git_info

TEMPLATES = Path(__file__).parent / "templates"
CHAT_URL = "https://api.tokenfactory.nebius.com/v1/chat/completions"
RED_LOWER_BOUND = 0.01
SEVERITY_LABEL = {
    "S1_SILENT": ("S1", "Silent corruption", "critical"),
    "S2_LOUD": ("S2", "Loud failure", "serious"),
    "S3_GAP": ("S3", "Gap", "warning"),
    "S4_DOCS": ("S4", "Docs", "neutral"),
}
OWNER_LABEL = {
    "nebius_platform": "Nebius platform",
    "nebius_docs": "Nebius docs",
    "nvidia_model": "NVIDIA model",
    "upstream_serving": "Upstream serving stack",
    "unknown": "Unknown (no reference replay yet)",
}
STATUS_LABEL = {"red": "Unreliable", "amber": "Caution", "green": "Reliable", "none": "Not tested"}
ROLE_NAMES = {"nano": "Nemotron 3 Nano", "lightning": "Nemotron 3.5 Lightning", "super": "Nemotron 3 Super",
              "ultra": "Nemotron 3 Ultra"}


def display_names(models: list[str], settings: Any) -> dict[str, str]:
    names = {}
    for m in models:
        price = settings.price_for(m)
        names[m] = ROLE_NAMES.get(getattr(price, "role", None) or "", model_short(m))
    return names


def cond_key(cond: dict[str, Any] | Condition) -> str:
    d = cond.as_dict() if isinstance(cond, Condition) else cond
    return json.dumps(d, sort_keys=True)


def cond_label(cond: dict[str, Any]) -> str:
    return Condition.from_dict(cond).label() if cond else ""


def model_short(model: str) -> str:
    return model.split("/", 1)[-1]


def trust_column(contract: BaseContract, cond: Condition) -> str | None:
    if contract.tier == 5:
        return "structured_output_reasoning" if cond.thinking else "structured_output"
    if contract.tier == 3:
        return "tool_args_stream" if cond.stream else "tool_args"
    if contract.tier == 2:
        return None if contract.id == "T2.default" else "reasoning_control"  # T2.default is informational
    if contract.tier == 1:
        return "reasoning_leak"
    return None


@dataclass
class TrustCell:
    status: str
    n: int = 0
    fails: int = 0
    worst: dict[str, Any] | None = None
    bound: float | None = None

    @property
    def label(self) -> str:
        return STATUS_LABEL[self.status]

    @property
    def text(self) -> str:
        if self.status == "none":
            return "no data"
        if self.status == "green":
            return f"0/{self.n} failed; rate < {self.bound:.1%}"
        w = self.worst or {}
        return f"{self.fails}/{self.n} failed; worst {w.get('contract_id', '')} {w.get('label', '')} at {w.get('rate', 0):.0%}"

    @property
    def lines(self) -> tuple[str, str]:
        if self.status == "none":
            return ("no data", "")
        if self.status == "green":
            return (f"0 of {self.n} failed", f"rate < {self.bound:.1%} (95% bound)")
        w = self.worst or {}
        return (f"{self.fails} of {self.n} failed",
                f"worst: {w.get('contract_id', '')} {w.get('label', '')}, {w.get('rate', 0):.0%} "
                f"[{w.get('lo', 0):.0%}-{w.get('hi', 0):.0%}]")


def trust_matrix(cells: dict[str, analysis.CellStats], models: list[str]) -> dict[str, dict[str, TrustCell]]:
    contracts = registry.by_id()
    grouped: dict[tuple[str, str], list[analysis.CellStats]] = defaultdict(list)
    for s in cells.values():
        c = contracts.get(s.contract_id)
        if c is None or not s.condition:
            continue
        col = trust_column(c, Condition.from_dict(s.condition))
        if col:
            grouped[(s.model, col)].append(s)
    matrix: dict[str, dict[str, TrustCell]] = {}
    for m in models:
        row = matrix[m] = {}
        for col in TRUST_COLUMNS:
            group = [s for s in grouped.get((m, col), []) if s.n > 0]
            n, fails = sum(s.n for s in group), sum(s.fails for s in group)
            if not group:
                row[col] = TrustCell("none")
                continue
            worst = max(group, key=lambda s: (s.wilson[0], s.rate))
            info = {"contract_id": worst.contract_id, "label": cond_label(worst.condition), "rate": worst.rate,
                    "lo": worst.wilson[0], "hi": worst.wilson[1], "n": worst.n, "fails": worst.fails}
            if any(s.wilson[0] > RED_LOWER_BOUND for s in group):
                row[col] = TrustCell("red", n, fails, info)
            elif fails:
                row[col] = TrustCell("amber", n, fails, info)
            else:
                row[col] = TrustCell("green", n, 0, None, rule_of_three(n))
    return matrix


def _arm(s: analysis.CellStats | None, label: str) -> dict[str, Any] | None:
    if s is None or s.n == 0:
        return None
    lo, hi = s.wilson
    return {"label": label, "n": s.n, "k": s.fails, "rate": s.rate, "lo": lo, "hi": hi}


def family_results(cells: dict[str, analysis.CellStats], models: list[str], alpha: float) -> dict[str, list[dict[str, Any]]]:
    index = {(s.contract_id, s.model, cond_key(s.condition)): s for s in cells.values()}
    contracts = registry.all_contracts()
    specs: dict[str, list[tuple[str, str, Condition, Condition, str, str]]] = defaultdict(list)
    for c in contracts:
        v = getattr(c, "variant", None)
        if c.tier == 3:
            specs["F1"].append((c.id, v, Condition(True, True, variant=v), Condition(True, False, variant=v), "stream", "no stream"))
            specs["F2"].append((c.id, v, Condition(True, False, variant=v), Condition(False, False, variant=v), "thinking on", "thinking off"))
        if c.tier == 5:
            specs["F2"].append((c.id, v, Condition(True, variant=v), Condition(False, variant=v), "thinking on", "thinking off"))
            if v == "flat_basic":
                specs["F3"].append((c.id, v, Condition(False, variant=v, shape="wrapped"),
                                    Condition(False, variant=v, shape="bare"), "wrapped", "bare"))
    out: dict[str, list[dict[str, Any]]] = {}
    for fam in FAMILIES:
        rows = []
        for model in models:
            for contract_id, variant, a, b, la, lb in specs.get(fam, []):
                arm_a = _arm(index.get((contract_id, model, cond_key(a))), la)
                arm_b = _arm(index.get((contract_id, model, cond_key(b))), lb)
                if not (arm_a and arm_b):
                    continue
                p = fisher_two_sided(arm_a["k"], arm_a["n"], arm_b["k"], arm_b["n"])
                rows.append({"family": fam, "model": model, "contract_id": contract_id, "variant": variant,
                             "a": arm_a, "b": arm_b, "p": p})
        for row, adj in zip(rows, holm([r["p"] for r in rows])):
            row["p_holm"] = adj
            row["significant"] = adj < alpha
        out[fam] = rows
    return out


def repro(request: dict[str, Any] | None) -> dict[str, str] | None:
    if not request:
        return None
    body = json.dumps(request, indent=2, ensure_ascii=False)
    stream = bool(request.get("stream"))
    curl = (f"curl{' -N' if stream else ''} {CHAT_URL} \\\n"
            "  -H \"Authorization: Bearer $NEBIUS_API_KEY\" \\\n"
            "  -H \"Content-Type: application/json\" \\\n"
            f"  -d @- <<'JSON'\n{body}\nJSON")
    send = (
        f'with httpx.stream("POST", "{CHAT_URL}", headers=headers, json=body, timeout=600) as resp:\n'
        "    for line in resp.iter_lines():\n        print(line)"
        if stream else
        f'resp = httpx.post("{CHAT_URL}", headers=headers, json=body, timeout=600)\n'
        "print(json.dumps(resp.json(), indent=2, ensure_ascii=False))"
    )
    python = (
        "import json\nimport os\n\nimport httpx\n\n"
        f'body = json.loads(r"""\n{body}\n""")\n'
        'headers = {"Authorization": f"Bearer {os.environ[\'NEBIUS_API_KEY\']}"}\n' + send
    )
    return {"curl": curl, "python": python}


def evidence_view(example: dict[str, Any]) -> dict[str, Any]:
    detail = example.get("detail") or {}
    mismatches = detail.get("mismatches") or []
    tool = (example.get("tool_calls") or [{}])[0] if example.get("tool_calls") else {}
    return {
        "model": example.get("model"),
        "condition": cond_label(example.get("condition") or {}),
        "codes": example.get("codes"),
        "finish_reason": example.get("finish_reason"),
        "content_head": example.get("content_head"),
        "arguments": (tool or {}).get("arguments_raw"),
        "mismatches": mismatches[:4],
        "detail": {k: v for k, v in detail.items() if k != "mismatches"},
    }


def cluster_findings(records: list[dict[str, Any]], names: dict[str, str] | None = None,
                     order: list[str] | None = None) -> list[dict[str, Any]]:
    names = names or {}
    rank = {m: i for i, m in enumerate(order or [])}
    findings = []
    for cl in build_clusters(records):
        d = cl.as_dict()
        cells = sorted(d["cells"], key=lambda c: (rank.get(c["model"], 99), c["model"], cond_label(c["condition"])))
        arms = [{"label": f"{names.get(c['model'], model_short(c['model']))} · {cond_label(c['condition'])}", "n": c["n"],
                 "k": c["k"], "rate": c["rate"], "lo": c["wilson_lo"], "hi": c["wilson_hi"]} for c in cells]
        findings.append({
            "id": d["id"], "title": d["title"], "severity": d["severity"], "owner": "unknown",
            "status": "cluster", "contract_id": d["contract_id"], "codes": d["codes"], "models": d["models"],
            "summary": (f"{d['failures']} of {d['trials']} trials in the affected cells failed "
                        f"({d['rate']:.1%}, 95% CI {d['wilson_lo']:.1%}-{d['wilson_hi']:.1%}) "
                        f"across {len(d['models'])} model(s)."),
            "arms": arms, "evidence": [evidence_view(e) for e in d["examples"][:2]],
            "repro": repro((d["examples"] or [{}])[0].get("request")), "workaround": None,
            "p_value": None, "factor": None,
        })
    return findings


def load_findings(runs_dir: Path) -> list[dict[str, Any]] | None:
    """Investigated findings (latest investigate run, owners updated by any reference replay)."""
    runs = analysis.stage_runs(runs_dir, "investigate")
    for run_dir in reversed(runs):
        path = run_dir / "findings.jsonl"
        if path.exists():
            findings = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
            owners = load_owner_updates(runs_dir)
            for f in findings:
                f.update(owners.get(f["id"], {}))
            return findings
    return None


def load_owner_updates(runs_dir: Path) -> dict[str, dict[str, Any]]:
    updates: dict[str, dict[str, Any]] = {}
    for run_dir in analysis.stage_runs(runs_dir, "reference"):
        path = run_dir / "attribution.jsonl"
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    updates[row["finding_id"]] = {"owner": row["owner"], "reference": row}
    return updates


def latest_docs_run(runs_dir: Path) -> dict[str, Any] | None:
    for run_dir in reversed(analysis.stage_runs(runs_dir, "docs")):
        summary = run_dir / "summary.json"
        if summary.exists():
            findings_path = run_dir / "docs_findings.jsonl"
            findings = [json.loads(x) for x in findings_path.read_text().splitlines() if x.strip()] if findings_path.exists() else []
            return {"run_dir": run_dir, "summary": json.loads(summary.read_text()), "findings": findings}
    return None


def latest_smoke(runs_dir: Path) -> dict[str, Any] | None:
    for run_dir in reversed(analysis.stage_runs(runs_dir, "smoke")):
        summary = run_dir / "summary.json"
        if summary.exists():
            return json.loads(summary.read_text())
    return None


def export_data(run_dirs: list[Path], docs_run: dict[str, Any] | None, cells: dict[str, analysis.CellStats],
                out: Path) -> list[dict[str, Any]]:
    data = out / "data"
    data.mkdir(parents=True, exist_ok=True)
    files = []

    def gz(src: Path, name: str) -> None:
        dest = data / name
        with src.open("rb") as fin, gzip.open(dest, "wb", compresslevel=9) as fout:
            shutil.copyfileobj(fin, fout)
        files.append({"name": name, "bytes": dest.stat().st_size})

    for d in run_dirs:
        if (d / "calls.jsonl").exists():
            gz(d / "calls.jsonl", f"{d.name}.calls.jsonl.gz")
    if docs_run:
        for name in ("docs_findings.jsonl", "blocks.jsonl", "executions.jsonl"):
            if (docs_run["run_dir"] / name).exists():
                gz(docs_run["run_dir"] / name, f"{docs_run['run_dir'].name}.{name}.gz")
    cells_path = data / "cells.json"
    cells_path.write_text(json.dumps([s.as_dict() for s in sorted(cells.values(), key=lambda s: (s.contract_id, s.model, s.cell_id))],
                                     indent=1, ensure_ascii=False))
    files.append({"name": "cells.json", "bytes": cells_path.stat().st_size})
    return files


def axis_max(values: list[float]) -> float:
    """A clean upper bound for a 0-based failure-rate axis."""
    top = max(values, default=0.0)
    for step in (0.05, 0.1, 0.25, 0.5, 1.0):
        if top <= step:
            return step
    return 1.0


def axis_ticks(top: float) -> list[float]:
    """Tick positions as fractions of the axis, landing on round percentages."""
    intervals = 4 if top >= 1.0 else 5  # 0-100 by 25; 0-50 by 10; 0-25 by 5; 0-10 by 2; 0-5 by 1
    return [i / intervals for i in range(intervals + 1)]


def default_runs(runs_dir: Path) -> list[Path]:
    return analysis.stage_runs(runs_dir, "screen") + analysis.stage_runs(runs_dir, "escalate")


@dataclass
class ReportData:
    generated: str
    git: dict[str, Any]
    prereg: dict[str, Any]
    models: list[str]
    columns: dict[str, str]
    color_rules: dict[str, str]
    matrix: dict[str, dict[str, TrustCell]]
    findings: list[dict[str, Any]]
    findings_status: str
    families: dict[str, list[dict[str, Any]]]
    family_specs: dict[str, dict[str, str]]
    docs: dict[str, Any] | None
    smoke: dict[str, Any] | None
    totals: dict[str, Any]
    methods: dict[str, Any]
    data_files: list[dict[str, Any]] = field(default_factory=list)
    runs: list[str] = field(default_factory=list)
    names: dict[str, str] = field(default_factory=dict)


def collect(settings: Any, run_dirs: list[Path], out: Path) -> ReportData:
    records = list(analysis.iter_records(run_dirs))
    cells = analysis.aggregate(records)
    models = sorted({s.model for s in cells.values()}, key=lambda m: (settings.price_for(m).output if settings.price_for(m) else 0, m))
    names = display_names(models, settings)
    investigated = load_findings(settings.runs_dir)
    findings = investigated if investigated is not None else cluster_findings(records, names, models)
    docs = latest_docs_run(settings.runs_dir)
    ts = sorted(r.get("ts_start", "") for r in records if r.get("ts_start"))
    plan_path = settings.root / "TEST_PLAN.md"
    git = git_info(settings.root)
    alpha = settings.defaults["alpha"]
    outcomes = defaultdict(int)
    for r in records:
        outcomes[r["outcome"]] += 1
    totals = {
        "trials": len(records), "pass": outcomes["pass"], "fail": outcomes["fail"], "error": outcomes["error"],
        "cells": len(cells), "models": len(models),
        "cost_usd": round(sum(r.get("cost_usd") or 0 for r in analysis.iter_records(run_dirs, final_only=False)), 2),
        "first": ts[0][:10] if ts else None, "last": ts[-1][:10] if ts else None,
        "findings": len(findings),
        "s1": sum(1 for f in findings if f["severity"] == "S1_SILENT"),
    }
    stages = sorted({d.name.split("-")[2] for d in run_dirs})
    return ReportData(
        generated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        git=git,
        prereg={"tag_commit": git.get("prereg_tag_commit"),
                "plan_hash": plan_file_hash(plan_path.read_text()) if plan_path.exists() else None,
                "code_hash": registry_hash(build_registry(settings.defaults))},
        models=models,
        columns=TRUST_COLUMNS,
        color_rules=COLOR_RULES,
        matrix=trust_matrix(cells, models),
        findings=findings,
        findings_status="investigated" if investigated is not None else "clusters",
        families=family_results(cells, models, alpha),
        family_specs=FAMILIES,
        docs=docs,
        smoke=latest_smoke(settings.runs_dir),
        totals=totals,
        methods={
            "temperature": contract_base.TEMPERATURE, "top_p": contract_base.TOP_P,
            "max_tokens_off": contract_base.MAX_TOKENS_THINKING_OFF, "max_tokens_on": contract_base.MAX_TOKENS_THINKING_ON,
            "thinking_switch": contract_base.THINKING_SWITCH, "alpha": alpha,
            "n_screen": settings.defaults["n_screen"], "n_escalate": settings.defaults["n_escalate"],
            "rpm": settings.defaults["rpm"], "concurrency": settings.defaults["concurrency"],
            "stages": stages, "red_lower_bound": RED_LOWER_BOUND,
            "prices": {m: settings.price_for(m) for m in models},
        },
        runs=[d.name for d in run_dirs],
        names=names,
    )


def render(data: ReportData) -> str:
    env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=select_autoescape(["html", "j2"]),
                      trim_blocks=True, lstrip_blocks=True)
    env.filters["pct"] = lambda x, d=0: f"{x * 100:.{d}f}%"
    env.filters["short"] = model_short
    env.globals.update(SEVERITY_LABEL=SEVERITY_LABEL, OWNER_LABEL=OWNER_LABEL, axis_max=axis_max, axis_ticks=axis_ticks,
                       cond_label=cond_label)
    return env.get_template("report.html.j2").render(d=data)


def build_report(settings: Any, run_dirs: list[Path], out: Path) -> dict[str, Path]:
    from .feedback import render_feedback, render_findings_md, write_feedback

    out.mkdir(parents=True, exist_ok=True)
    data = collect(settings, run_dirs, out)
    records_cells = analysis.aggregate(analysis.iter_records(run_dirs))
    data.data_files = export_data(run_dirs, data.docs, records_cells, out)
    paths = {"html": out / "index.html", "findings": out / "findings.md"}
    paths["html"].write_text(render(data))
    paths["findings"].write_text(render_findings_md(data))
    (out / ".nojekyll").write_text("")
    paths["feedback"] = write_feedback(settings.root, render_feedback(data, settings.root))
    return paths
