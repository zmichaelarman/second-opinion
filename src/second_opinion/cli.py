from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from second_opinion.budget import BudgetGuard
from second_opinion.byo import run_byo_check
from second_opinion.client import InferenceClient
from second_opinion.config import REPO_ROOT, get_nebius_api_key, load_config
from second_opinion.contracts.base import Owner, Severity
from second_opinion.contracts.registry import get_tier1_contracts
from second_opinion.discovery import (
    FALLBACK_NEMOTRON_MODELS,
    discover_nemotron_models,
)
from second_opinion.docs_check.execute import (
    execute_python_block,
    is_block_eligible_for_execution,
)
from second_opinion.docs_check.extract import extract_code_blocks
from second_opinion.docs_check.fetch import fetch_docs
from second_opinion.docs_check.repair import (
    repair_failing_block,
    write_docs_findings,
)
from second_opinion.docs_check.static import check_code_block_static
from second_opinion.investigator.cluster import cluster_failures
from second_opinion.investigator.loop import run_investigator
from second_opinion.pacer import HeaderAwarePacer
from second_opinion.planner import (
    build_stage_plan,
    generate_test_plan_markdown,
    load_screening_fatal_cells,
    parse_max_n_overrides,
    print_cost_estimate_table,
    verify_prereg,
)
from second_opinion.reference import rerun_findings, run_reference_replay
from second_opinion.report.build import build_report
from second_opinion.runner import run_smoke, run_stage
from second_opinion.store import RunStore, load_records_from_runs

app = typer.Typer(
    name="second-opinion",
    help="Second Opinion: conversion-layer contract suite, docs checker, and root-cause investigator for NVIDIA Nemotron on Nebius Token Factory.",
    add_completion=False,
)
console = Console()


async def _resolve_models_async(models_arg: str, config: object, fallback_on_error: bool = True) -> list[str]:
    if models_arg and models_arg.strip().lower() != "auto":
        return [m.strip() for m in models_arg.split(",") if m.strip()]
    eligible, _ = await discover_nemotron_models(
        config,  # type: ignore[arg-type]
        use_fallback_on_error=fallback_on_error,
    )
    return eligible


@app.command("models")
def cmd_models(
    fallback: bool = typer.Option(
        False,
        "--fallback",
        help="Fall back to configured hint table if NEBIUS_API_KEY is not set or network is unavailable.",
    ),
) -> None:
    """Discover and filter Nemotron models from GET /v1/models, printed as a table."""
    config = load_config()
    try:
        eligible, all_models = asyncio.run(
            discover_nemotron_models(config, use_fallback_on_error=fallback)
        )
    except Exception as exc:
        if fallback:
            eligible = list(FALLBACK_NEMOTRON_MODELS)
            all_models = list(FALLBACK_NEMOTRON_MODELS)
        else:
            console.print(f"[red bold]Error discovering models:[/red bold] {exc}")
            raise typer.Exit(code=1)

    table = Table(title=f"Discovered NVIDIA Nemotron Models ({len(eligible)} of {len(all_models)} total listed)")
    table.add_column("Model ID", style="cyan")
    table.add_column("Role", style="magenta")
    table.add_column("Input ($ / 1M tok)", justify="right", style="green")
    table.add_column("Output ($ / 1M tok)", justify="right", style="green")

    for mid in eligible:
        p = config.get_price(mid)
        table.add_row(mid, p.role, f"${p.input_per_1m:.2f}", f"${p.output_per_1m:.2f}")

    console.print(table)


@app.command("smoke")
def cmd_smoke() -> None:
    """Run <=15-call smoke check: model list, banana thinking off/on, forced leading_zero tool call, and OpenAI SDK view."""
    config = load_config()
    try:
        asyncio.run(run_smoke(config, console=console))
    except Exception as exc:
        console.print(f"[red bold]Smoke test failed:[/red bold] {exc}")
        raise typer.Exit(code=1)


@app.command("plan")
def cmd_plan(
    write: bool = typer.Option(False, "--write", help="Write TEST_PLAN.md to repo root."),
) -> None:
    """Generate pre-registration test plan (TEST_PLAN.md) with registry SHA-256."""
    config = load_config()
    md = generate_test_plan_markdown(config)
    if write:
        out_path = REPO_ROOT / "TEST_PLAN.md"
        out_path.write_text(md, encoding="utf-8")
        console.print(f"[green]Wrote pre-registered test plan to {out_path}[/green]")
        console.print("Next: commit `TEST_PLAN.md` and run `git tag prereg-v1` before running `--stage screen`.")
    else:
        console.print(md)


@app.command("run")
def cmd_run(
    stage: str = typer.Option(..., "--stage", help="Stage to run: pilot | screen | escalate"),
    models: str = typer.Option("auto", "--models", help="auto or comma-separated model IDs"),
    contracts: Optional[str] = typer.Option(None, "--contracts", help="Comma-separated contract IDs (e.g. T1,T3)"),
    rpm: Optional[int] = typer.Option(None, "--rpm", help="Max requests per minute (default 40)"),
    concurrency: Optional[int] = typer.Option(None, "--concurrency", help="Concurrency semaphore (default 6)"),
    max_n: Optional[list[str]] = typer.Option(None, "--max-n", help="Model trial cap e.g. MODEL=150"),
    no_prereg_check: bool = typer.Option(False, "--no-prereg-check", help="Skip prereg-v1 git tag check on screen stage"),
    resume: Optional[Path] = typer.Option(None, "--resume", help="Resume an existing run directory"),
    yes: bool = typer.Option(False, "--yes", help="Confirm paid API spend and execute"),
) -> None:
    """Run contract evaluation stage (pilot, screen, escalate). Prints cost estimate first and exits unless --yes."""
    stage_norm = stage.strip().lower()
    if stage_norm not in ("pilot", "screen", "escalate"):
        console.print(f"[red]Invalid stage {stage!r}. Choose from: pilot, screen, escalate.[/red]")
        raise typer.Exit(code=2)

    config = load_config()
    all_contracts = get_tier1_contracts()
    if contracts:
        wanted = {c.strip().upper() for c in contracts.split(",") if c.strip()}
        selected_contracts = [c for c in all_contracts if c.id.upper() in wanted]
    else:
        selected_contracts = all_contracts

    if stage_norm == "screen" and not no_prereg_check:
        ok, errs = verify_prereg(REPO_ROOT, all_contracts)
        if not ok:
            console.print("[red bold]Pre-registration check failed for `screen` stage:[/red bold]")
            for e in errs:
                console.print(f"  - {e}")
            raise typer.Exit(code=1)

    model_list = asyncio.run(_resolve_models_async(models, config, fallback_on_error=True))
    overrides = parse_max_n_overrides(max_n)
    planned_cells = build_stage_plan(
        stage=stage_norm,
        models=model_list,
        config=config,
        contracts=selected_contracts,
        max_n_overrides=overrides,
    )
    _, existing_counts = load_screening_fatal_cells() if stage_norm == "escalate" else (set(), {})
    print_cost_estimate_table(stage_norm, planned_cells, console=console, existing_counts=existing_counts)

    if not yes:
        console.print("[yellow]Dry-run estimate complete. Pass `--yes` after user approval to execute.[/yellow]")
        raise typer.Exit(code=0)

    store, budget_exceeded = asyncio.run(
        run_stage(
            stage=stage_norm,
            planned_cells=planned_cells,
            config=config,
            rpm=rpm,
            concurrency=concurrency,
            resume_dir=resume,
            console=console,
        )
    )
    console.print(f"[green]Stage `{stage_norm}` saved to {store.run_dir}[/green]")
    if budget_exceeded:
        raise typer.Exit(code=3)


@app.command("docs")
def cmd_docs(
    execute: bool = typer.Option(False, "--execute", help="Execute eligible Python snippets in isolated subprocess"),
    repair: bool = typer.Option(False, "--repair", help="Repair failing blocks with Nemotron 3 Super"),
    yes: bool = typer.Option(False, "--yes", help="Confirm paid API spend for --execute / --repair"),
) -> None:
    """Fetch, extract, and statically check Token Factory docs; optionally --execute and --repair."""
    config = load_config()
    try:
        pages = asyncio.run(fetch_docs(config))
    except Exception as exc:
        console.print(f"[red]Failed to fetch docs: {exc}[/red]")
        raise typer.Exit(code=1)

    try:
        _, live_models = asyncio.run(discover_nemotron_models(config, use_fallback_on_error=True))
        known_models_set: set[str] | None = set(live_models) if len(live_models) > 4 else None
    except Exception:
        live_models = list(FALLBACK_NEMOTRON_MODELS)
        known_models_set = None

    all_blocks = []
    for p in pages:
        all_blocks.extend(extract_code_blocks(p["content"], page_url=p["url"], page_title=p["title"]))

    static_results = []
    failing_items = []
    exec_candidates = []

    for blk in all_blocks:
        sres = check_code_block_static(blk, known_models=known_models_set)
        static_results.append((blk, sres))
        if sres.codes:
            failing_items.append((blk, list(sres.codes), "\n".join(sres.errors)))
        elif execute and is_block_eligible_for_execution(blk, sres.codes):
            exec_candidates.append(blk)

    console.print(
        f"Docs static check: {len(all_blocks)} code blocks across {len(pages)} pages; "
        f"{len(failing_items)} static failures; {len(exec_candidates)} eligible for runtime execution."
    )

    # Spending rule if --execute or --repair is requested
    if execute or repair:
        est_exec_calls = len(exec_candidates) if execute else 0
        est_repair_calls = (len(failing_items) + est_exec_calls) * 2 if repair else 0
        est_cost = est_repair_calls * config.estimate_cost(config.code_model, 600, 1200) + (est_exec_calls * 0.001)

        table = Table(title="Second Opinion — Docs Stage Planned Spend")
        table.add_column("Operation", style="cyan")
        table.add_column("Model", style="magenta")
        table.add_column("Planned Calls", justify="right")
        table.add_column("Estimated Cost ($)", justify="right", style="green")
        if execute:
            table.add_row("Subprocess snippet execution", "snippet-defined", str(est_exec_calls), f"${est_exec_calls * 0.001:.4f}")
        if repair:
            table.add_row("Doc snippet repair agent", config.code_model, str(est_repair_calls), f"${est_cost:.4f}")
        console.print(table)

        if not yes:
            console.print("[yellow]Dry-run estimate complete. Pass `--yes` after user approval to execute.[/yellow]")
            raise typer.Exit(code=0)

    if execute:
        for blk in exec_candidates:
            ok, err_code, err_detail = execute_python_block(blk.code)
            if not ok and err_code:
                failing_items.append((blk, [err_code], err_detail))

    store = RunStore.create_or_resume("docs")
    findings_out: list[dict] = []

    if repair and failing_items:
        api_key = get_nebius_api_key()
        budget = BudgetGuard(config)
        pacer = HeaderAwarePacer(base_rpm=config.rpm)

        async def _repair_all() -> list[dict]:
            out = []
            async with InferenceClient(
                base_url=config.nebius_base_url,
                api_key=api_key,
                pacer=pacer,
            ) as client:
                for blk, codes, err_txt in failing_items:
                    item = await repair_failing_block(
                        block=blk,
                        initial_codes=codes,
                        initial_error=err_txt,
                        known_models=live_models,
                        config=config,
                        client=client,
                        budget=budget,
                        run_id=store.run_id,
                        execute_enabled=execute,
                    )
                    out.append(item)
            return out

        findings_out = asyncio.run(_repair_all())
    else:
        for blk, codes, err_txt in failing_items:
            has_unknown = any(c.startswith("UNKNOWN_PARAM:") for c in codes)
            sev = Severity.S2_LOUD.name if has_unknown else Severity.S4_DOCS.name
            findings_out.append(
                {
                    "page_url": blk.page_url,
                    "page_title": blk.page_title,
                    "block_idx": blk.block_idx,
                    "lang": blk.lang,
                    "codes": codes,
                    "error": err_txt,
                    "fixed": False,
                    "attempts": 0,
                    "patch": "",
                    "severity": sev,
                    "owner": Owner.NEBIUS_DOCS.value,
                }
            )

    out_jsonl = store.run_dir / "docs_findings.jsonl"
    write_docs_findings(findings_out, out_jsonl)
    store.write_summary(
        {
            "run_id": store.run_id,
            "total_blocks": len(all_blocks),
            "failing_blocks": len(findings_out),
            "repaired_blocks": sum(1 for f in findings_out if f.get("fixed")),
        }
    )
    console.print(f"[green]Wrote {len(findings_out)} docs findings to {out_jsonl}[/green]")


@app.command("investigate")
def cmd_investigate(
    runs: list[Path] = typer.Option(..., "--runs", help="Run directories to cluster and investigate"),
    hypothesis_model: Optional[str] = typer.Option(None, "--hypothesis-model", help="Override Ultra model ID"),
    code_model: Optional[str] = typer.Option(None, "--code-model", help="Override Super model ID"),
    yes: bool = typer.Option(False, "--yes", help="Confirm paid API spend and execute"),
) -> None:
    """Cluster fatal failures, shrink requests, propose hypotheses (Ultra), run pytest experiments (Super), and decide."""
    config = load_config()
    records = load_records_from_runs(runs, final_only=True)
    clusters = cluster_failures(records)
    if not clusters:
        console.print("[green]No qualifying failure clusters found in the provided runs.[/green]")
        raise typer.Exit(code=0)

    h_mod = hypothesis_model or config.hypothesis_model
    c_mod = code_model or config.code_model

    table = Table(title="Second Opinion — Investigator Planned Spend Estimate")
    table.add_column("Cluster ID", style="cyan")
    table.add_column("Contract / Codes", style="magenta")
    table.add_column("Models", style="white")
    table.add_column("Est. Calls (Shrink + Hyp + Arms)", justify="right")
    table.add_column("Max Budget ($)", justify="right", style="green")

    total_cap = 0.0
    for cl in clusters:
        cap = config.cluster_budget_usd
        total_cap += cap
        table.add_row(
            cl.cluster_id,
            f"{cl.contract_id}: {', '.join(cl.fatal_codes)}",
            ", ".join(cl.models_affected),
            "~280",
            f"${cap:.2f}",
        )
    table.add_section()
    table.add_row(
        "TOTAL",
        f"Hyp={h_mod}, Code={c_mod}",
        f"{len(clusters)} clusters",
        f"~{len(clusters) * 280}",
        f"${total_cap:.2f}",
        style="bold",
    )
    console.print(table)

    if not yes:
        console.print("[yellow]Dry-run estimate complete. Pass `--yes` after user approval to execute.[/yellow]")
        raise typer.Exit(code=0)

    findings = asyncio.run(
        run_investigator(
            run_dirs=runs,
            config=config,
            hypothesis_model=h_mod,
            code_model=c_mod,
            console=console,
        )
    )
    console.print(f"[green]Investigation complete: {len(findings)} findings produced.[/green]")


@app.command("reference")
def cmd_reference(
    findings: Path = typer.Option(..., "--findings", help="Path to findings.json"),
    yes: bool = typer.Option(False, "--yes", help="Confirm replay execution"),
) -> None:
    """Replay each finding's minimal request n=30 on the NVIDIA Reference API at 20 RPM and attribute Owner."""
    config = load_config()
    items = json.loads(findings.read_text(encoding="utf-8"))
    planned_calls = len(items) * config.n_reference

    table = Table(title="Second Opinion — NVIDIA Reference Replay Estimate")
    table.add_column("Findings", justify="right")
    table.add_column("Trials per Finding", justify="right")
    table.add_column("Total Calls (20 RPM)", justify="right")
    table.add_column("Estimated Cost ($)", justify="right", style="green")
    table.add_row(str(len(items)), str(config.n_reference), str(planned_calls), "$0.0000 (NVIDIA Free Tier)")
    console.print(table)

    if not yes:
        console.print("[yellow]Dry-run estimate complete. Pass `--yes` after user approval to execute.[/yellow]")
        raise typer.Exit(code=0)

    asyncio.run(run_reference_replay(findings_path=findings, config=config, console=console))
    console.print(f"[green]Updated findings with NVIDIA reference attribution in {findings}[/green]")


@app.command("check")
def cmd_check(
    tools: Optional[Path] = typer.Option(None, "--tools", help="Path to user tools JSON file"),
    schemas: Optional[Path] = typer.Option(None, "--schemas", help="Path to user schemas JSON file"),
    models: str = typer.Option("auto", "--models", help="auto or comma-separated model IDs"),
    n: int = typer.Option(50, "--n", help="Trials per tool/schema per model"),
    yes: bool = typer.Option(False, "--yes", help="Confirm paid API spend and execute"),
) -> None:
    """Bring-your-own mode: test custom tool definitions and JSON schemas across Nemotron models."""
    if not tools and not schemas:
        console.print("[red]Provide at least one of `--tools` or `--schemas`.[/red]")
        raise typer.Exit(code=2)

    config = load_config()
    model_list = asyncio.run(_resolve_models_async(models, config, fallback_on_error=True))

    n_items = 0
    if tools and tools.exists():
        t_data = json.loads(tools.read_text(encoding="utf-8"))
        n_items += len(t_data) if isinstance(t_data, list) else 1
    if schemas and schemas.exists():
        s_data = json.loads(schemas.read_text(encoding="utf-8"))
        n_items += len(s_data) if isinstance(s_data, list) else 1

    table = Table(title="Second Opinion — Bring-Your-Own Check Planned Spend")
    table.add_column("Model", style="cyan")
    table.add_column("Tools/Schemas", justify="right")
    table.add_column("Planned Calls", justify="right")
    table.add_column("Estimated Cost ($)", justify="right", style="green")

    total_calls = 0
    total_cost = 0.0
    for m in model_list:
        calls = n_items * n
        cost = calls * config.estimate_cost(m, 400, 1000)
        total_calls += calls
        total_cost += cost
        table.add_row(m, str(n_items), str(calls), f"${cost:.4f}")
    table.add_section()
    table.add_row("TOTAL", str(n_items), str(total_calls), f"${total_cost:.4f}", style="bold")
    console.print(table)

    if not yes:
        console.print("[yellow]Dry-run estimate complete. Pass `--yes` after user approval to execute.[/yellow]")
        raise typer.Exit(code=0)

    asyncio.run(run_byo_check(tools, schemas, model_list, n, config, console=console))


@app.command("report")
def cmd_report(
    runs: list[Path] = typer.Option(..., "--runs", help="Run directories to include in report"),
    out: Path = typer.Option(Path("docs"), "--out", help="Output directory (default: docs/)"),
) -> None:
    """Build self-contained static HTML report (docs/index.html), docs/findings.md, and FEEDBACK_DRAFT.md."""
    config = load_config()
    outputs = build_report(run_dirs=runs, out_dir=out, config=config)
    for label, path in outputs.items():
        console.print(f"[green]Generated {label}:[/green] {path}")


@app.command("rerun")
def cmd_rerun(
    findings: Path = typer.Option(..., "--findings", help="Path to findings.json"),
    yes: bool = typer.Option(False, "--yes", help="Confirm paid API spend and execute"),
) -> None:
    """Re-run every finding's minimal repro at n=30 and mark each still_failing or fixed."""
    config = load_config()
    items = json.loads(findings.read_text(encoding="utf-8"))

    table = Table(title="Second Opinion — Rerun Planned Spend")
    table.add_column("Finding ID", style="cyan")
    table.add_column("Model", style="magenta")
    table.add_column("Planned Calls", justify="right")
    table.add_column("Estimated Cost ($)", justify="right", style="green")

    total_cost = 0.0
    for f in items:
        m = str((f.get("minimal_request") or {}).get("model") or config.code_model)
        cost = config.n_rerun * config.estimate_cost(m, 400, 1000)
        total_cost += cost
        table.add_row(str(f.get("finding_id")), m, str(config.n_rerun), f"${cost:.4f}")
    table.add_section()
    table.add_row("TOTAL", "ALL", str(len(items) * config.n_rerun), f"${total_cost:.4f}", style="bold")
    console.print(table)

    if not yes:
        console.print("[yellow]Dry-run estimate complete. Pass `--yes` after user approval to execute.[/yellow]")
        raise typer.Exit(code=0)

    asyncio.run(rerun_findings(findings_path=findings, config=config, console=console))
    console.print(f"[green]Updated rerun status in {findings}[/green]")


if __name__ == "__main__":
    app()
