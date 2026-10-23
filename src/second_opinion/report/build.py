from __future__ import annotations

import gzip
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from second_opinion.config import AppConfig, REPO_ROOT
from second_opinion.contracts.registry import compute_registry_hash
from second_opinion.discovery import FALLBACK_NEMOTRON_MODELS
from second_opinion.report.feedback import generate_feedback_draft
from second_opinion.stats import trust_matrix_status
from second_opinion.store import get_git_info, load_records_from_runs

TRUST_CAPABILITIES = [
    "Structured output (T5, thinking off)",
    "Structured output with reasoning (T5, thinking on)",
    "Tool arguments (T3, non-stream)",
    "Tool arguments, streaming (T3 stream)",
    "Reasoning control (T2)",
    "Reasoning leak (T1)",
]


def _classify_capability_column(rec: dict[str, Any]) -> str | None:
    cid = str(rec.get("contract_id", ""))
    cond = rec.get("condition") or {}
    if cid == "T5":
        if cond.get("thinking") is True:
            return "Structured output with reasoning (T5, thinking on)"
        return "Structured output (T5, thinking off)"
    if cid == "T3":
        if cond.get("stream"):
            return "Tool arguments, streaming (T3 stream)"
        return "Tool arguments (T3, non-stream)"
    if cid == "T2":
        # T2.default is informational only
        if cond.get("variant") == "default":
            return None
        return "Reasoning control (T2)"
    if cid == "T1":
        return "Reasoning leak (T1)"
    return None


def compute_trust_matrix(records: list[dict[str, Any]], models: list[str] | None = None) -> list[dict[str, Any]]:
    discovered_models = sorted({str(r.get("model", "")) for r in records if r.get("model")})
    model_list = models or discovered_models or list(FALLBACK_NEMOTRON_MODELS)

    # Group by (model, capability_col, cell_id) -> [failures, trials]
    cell_counts: dict[tuple[str, str, str], list[int]] = {}
    for rec in records:
        if not rec.get("is_final", True):
            continue
        if rec.get("stage") in ("pilot", "smoke"):
            continue
        outcome = rec.get("outcome")
        if outcome not in ("pass", "fail"):
            continue
        model = str(rec.get("model", ""))
        col = _classify_capability_column(rec)
        cell_id = str(rec.get("cell_id", ""))
        if not model or not col or not cell_id:
            continue
        key = (model, col, cell_id)
        pair = cell_counts.setdefault(key, [0, 0])
        if outcome == "fail":
            pair[0] += 1
        pair[1] += 1

    rows: list[dict[str, Any]] = []
    for m in model_list:
        row_cells: list[dict[str, Any]] = []
        for col in TRUST_CAPABILITIES:
            constituent = [
                (k_f, n_t)
                for (mod, c_name, _), (k_f, n_t) in cell_counts.items()
                if mod == m and c_name == col
            ]
            status = trust_matrix_status(constituent)
            status["capability"] = col
            row_cells.append(status)
        rows.append({"model": m, "cells": row_cells})
    return rows


def build_report(
    run_dirs: list[Path],
    out_dir: Path,
    config: AppConfig,
    repo_root: Path | None = None,
) -> dict[str, Path]:
    """Generate docs/index.html, docs/findings.md, docs/data/*.jsonl.gz, and FEEDBACK_DRAFT.md (§12)."""
    root = repo_root or REPO_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    records = load_records_from_runs(run_dirs, final_only=True)
    trust_rows = compute_trust_matrix(records)

    # Collect findings.json and docs_findings.jsonl from run_dirs or runs/
    findings: list[dict[str, Any]] = []
    docs_findings: list[dict[str, Any]] = []
    data_archives: list[str] = []

    search_dirs = list(run_dirs)
    runs_base = root / "runs"
    if runs_base.exists():
        for d in sorted(runs_base.iterdir()):
            if d.is_dir() and d not in search_dirs:
                search_dirs.append(d)

    seen_finding_ids: set[str] = set()
    docs_total_from_summary = 0
    for d in search_dirs:
        if not d.exists():
            continue
        f_json = d / "findings.json" if d.is_dir() else (d if d.name == "findings.json" else None)
        if f_json and f_json.exists():
            try:
                loaded_f = json.loads(f_json.read_text(encoding="utf-8"))
                if isinstance(loaded_f, list):
                    for item in loaded_f:
                        fid = str(item.get("finding_id", ""))
                        if fid not in seen_finding_ids:
                            seen_finding_ids.add(fid)
                            findings.append(item)
            except Exception:
                pass

        df_jsonl = d / "docs_findings.jsonl" if d.is_dir() else (d if d.name == "docs_findings.jsonl" else None)
        if df_jsonl and df_jsonl.exists():
            docs_findings = []
            with df_jsonl.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            docs_findings.append(json.loads(line))
                        except Exception:
                            pass
            sum_file = df_jsonl.parent / "summary.json"
            if sum_file.exists():
                try:
                    sdata = json.loads(sum_file.read_text(encoding="utf-8"))
                    docs_total_from_summary = int(sdata.get("total_blocks", 0))
                except Exception:
                    pass

    # Compress calls.jsonl from specified run_dirs into docs/data/*.jsonl.gz
    for d in run_dirs:
        calls_p = d / "calls.jsonl" if d.is_dir() else d
        if calls_p.exists() and calls_p.name.endswith(".jsonl"):
            gz_name = f"{d.name if d.is_dir() else calls_p.stem}.jsonl.gz"
            gz_path = data_dir / gz_name
            with calls_p.open("rb") as f_in, gzip.open(gz_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            data_archives.append(gz_name)

    git_info = get_git_info(root)
    reg_hash = compute_registry_hash()
    docs_repaired = sum(1 for df in docs_findings if df.get("fixed"))
    docs_total = max(docs_total_from_summary, len(docs_findings), 1)
    docs_passed = max(0, docs_total - len(docs_findings))

    templates_dir = Path(__file__).resolve().parent / "templates"
    env = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    tpl = env.get_template("report.html.j2")
    html_text = tpl.render(
        generated_date=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        git_hash=git_info.get("git_hash"),
        git_tag=git_info.get("git_tag") or "prereg-v1",
        registry_hash=reg_hash,
        total_trials=len(records),
        trust_columns=TRUST_CAPABILITIES,
        trust_rows=trust_rows,
        findings=findings,
        docs_findings=docs_findings,
        docs_summary={
            "total": docs_total,
            "passed": docs_passed,
            "repaired": docs_repaired,
        },
        config=config,
        models_list=[r["model"] for r in trust_rows],
        data_archives=data_archives,
    )

    index_html_path = out_dir / "index.html"
    index_html_path.write_text(html_text, encoding="utf-8")

    # Write docs/findings.md
    md_lines = [
        "# Second Opinion — Findings Summary",
        "",
        f"- **Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"- **Git Hash:** `{git_info.get('git_hash') or 'uncommitted'}`",
        f"- **Registry SHA-256:** `{reg_hash}`",
        "",
        "## 1. Trust Matrix",
        "",
        "| Model | " + " | ".join(TRUST_CAPABILITIES) + " |",
        "|---|" + "|".join(["---"] * len(TRUST_CAPABILITIES)) + "|",
    ]
    for r in trust_rows:
        cells_md = [f"**{c['color'].upper()}** ({c['bound_str']})" for c in r["cells"]]
        md_lines.append(f"| `{r['model']}` | " + " | ".join(cells_md) + " |")

    md_lines.extend(["", "## 2. Confirmed Findings", ""])
    if findings:
        for f in findings:
            md_lines.extend(
                [
                    f"### {f.get('title')}",
                    f"- **Severity:** `{f.get('severity')}`",
                    f"- **Owner:** `{f.get('owner')}`",
                    f"- **Models Affected:** `{', '.join(f.get('models_affected', []))}`",
                    f"- **Workaround:** {f.get('workaround')}",
                    "",
                ]
            )
    else:
        md_lines.append("_No failure clusters recorded yet._\n")

    md_lines.extend(["## 3. Documentation Findings", ""])
    if docs_findings:
        for df in docs_findings:
            md_lines.append(
                f"- `{df.get('page_url')}` (block #{df.get('block_idx')}, {df.get('lang')}): "
                f"`{', '.join(df.get('codes', []))}` — Repaired: `{df.get('fixed')}`"
            )
    else:
        md_lines.append("_No documentation findings recorded yet._")

    findings_md_path = out_dir / "findings.md"
    findings_md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    feedback_md = generate_feedback_draft(
        findings=findings,
        docs_findings=docs_findings,
        trust_rows=trust_rows,
        repo_root=root,
    )
    feedback_path = root / "FEEDBACK_DRAFT.md"
    feedback_path.write_text(feedback_md, encoding="utf-8")

    return {
        "index_html": index_html_path,
        "findings_md": findings_md_path,
        "feedback_draft": feedback_path,
    }
