"""Command-line entry point (`second-opinion`). See CLAUDE.md §14."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

import typer
from rich.console import Console

from .config import ConfigError, Settings, load_settings

app = typer.Typer(
    help="Check whether Nebius Token Factory keeps its promises on NVIDIA Nemotron models.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


class Stage(str, Enum):
    pilot = "pilot"
    screen = "screen"
    escalate = "escalate"


def _run(make: Callable[[Settings], Coroutine[Any, Any, int]]) -> None:
    try:
        code = asyncio.run(make(load_settings()))
    except ConfigError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(2) from None
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(2) from None
    raise typer.Exit(code)


@app.command()
def models(show_all: bool = typer.Option(False, "--all", help="Show every model, not only Nemotron IDs.")) -> None:
    """List models from GET /v1/models and show which Nemotron models will be tested."""
    from .discovery import list_models

    _run(lambda s: list_models(s, console, show_all))


@app.command()
def smoke() -> None:
    """Run the smoke test (15 calls or fewer) and record it under runs/<ts>-smoke/."""
    from .smoke import run_smoke

    _run(lambda s: run_smoke(s, console))


@app.command()
def plan(write: bool = typer.Option(False, "--write", help="Write TEST_PLAN.md for pre-registration.")) -> None:
    """Show the contract registry and its hash; --write generates TEST_PLAN.md."""
    from .contracts.registry import all_contracts
    from .planner import registry, registry_hash, render_test_plan

    settings = load_settings()
    contracts = all_contracts()
    digest = registry_hash(registry(settings.defaults))
    console.print(f"{len(contracts)} contracts, {sum(len(c.conditions) for c in contracts)} conditions. "
                  f"Registry SHA-256: {digest}")
    for c in contracts:
        console.print(f"  {c.id:<20} {c.severity.name:<10} {len(c.conditions)} condition(s)")
    if write:
        path = settings.root / "TEST_PLAN.md"
        path.write_text(render_test_plan(settings.defaults, sorted(settings.prices)))
        console.print(f"Wrote {path}. Pre-register it before screening:\n"
                      "  git add TEST_PLAN.md && git commit -m \"Pre-register test plan\" && git tag prereg-v1")


@app.command()
def run(
    stage: Stage = typer.Option(..., "--stage", help="pilot, screen or escalate."),
    models_spec: str = typer.Option("auto", "--models", help="auto, or a comma list of model IDs."),
    contracts_spec: Optional[str] = typer.Option(None, "--contracts", help="Comma list of tiers or IDs, e.g. T1,T3."),
    rpm: Optional[float] = typer.Option(None, "--rpm", help="Requests per minute ceiling (default 40)."),
    concurrency: Optional[int] = typer.Option(None, "--concurrency", help="Parallel requests (default 6)."),
    max_n: list[str] = typer.Option([], "--max-n", help="Cap trials per cell for a model: MODEL=N (repeatable)."),
    yes: bool = typer.Option(False, "--yes", help="Actually send the paid calls after the estimate is approved."),
    no_prereg_check: bool = typer.Option(False, "--no-prereg-check", help="Skip the prereg-v1 check (development only)."),
    resume: Optional[Path] = typer.Option(None, "--resume", help="Continue an existing run directory."),
) -> None:
    """Run a stage. Prints the plan and estimated cost, and sends nothing unless --yes is passed."""
    from .runner import parse_max_n, run_stage

    _run(lambda s: run_stage(
        s, console, stage=stage.value, models_spec=models_spec, contracts_spec=contracts_spec, rpm=rpm,
        concurrency=concurrency, max_n=parse_max_n(max_n), yes=yes, prereg=not no_prereg_check, resume=resume,
    ))


@app.command()
def docs(
    execute: bool = typer.Option(False, "--execute", help="Run eligible Python examples and probe unknown parameters (paid)."),
    repair: bool = typer.Option(False, "--repair", help="Let the repair agent fix failing examples (paid)."),
    yes: bool = typer.Option(False, "--yes", help="Send the paid calls after the estimate is approved."),
) -> None:
    """Check every code example in the Token Factory inference docs (static checks are free)."""
    from .docs_check.pipeline import run_docs

    _run(lambda s: run_docs(s, console, execute=execute, repair=repair, yes=yes))


@app.command()
def report(
    runs: list[Path] = typer.Option([], "--runs", help="Run directories (default: every screen and escalate run)."),
    out: Path = typer.Option(Path("docs"), "--out", help="Output directory (GitHub Pages serves /docs)."),
) -> None:
    """Build docs/index.html, docs/findings.md, docs/data/ and FEEDBACK_DRAFT.md from raw run data."""
    from .report.build import build_report, default_runs

    settings = load_settings()
    run_dirs = [p.resolve() for p in runs] or default_runs(settings.runs_dir)
    if not run_dirs:
        console.print("[yellow]No screen or escalate runs found; the report will show docs results only.[/yellow]")
    out_dir = out if out.is_absolute() else settings.root / out
    paths = build_report(settings, run_dirs, out_dir)
    for name, path in paths.items():
        console.print(f"{name}: {path}")


def _latest_findings(settings: Settings, given: Optional[Path]) -> Path:
    from .reference import latest_findings

    path = given or latest_findings(settings.runs_dir)
    if path is None:
        console.print("[red]No findings file. Run `second-opinion investigate` first, or pass --findings.[/red]")
        raise typer.Exit(2)
    return path


@app.command()
def investigate(
    runs: list[Path] = typer.Option([], "--runs", help="Run directories (default: every screen and escalate run)."),
    hypothesis_model: Optional[str] = typer.Option(None, "--hypothesis-model", help="Model that proposes causes (default: Ultra)."),
    code_model: Optional[str] = typer.Option(None, "--code-model", help="Model that writes experiments (default: Super)."),
    max_clusters: int = typer.Option(5, "--max-clusters", help="Investigate the N most severe clusters."),
    cluster_budget: float = typer.Option(3.0, "--cluster-budget", help="Spend cap per cluster in USD."),
    yes: bool = typer.Option(False, "--yes", help="Send the paid calls after the estimate is approved."),
) -> None:
    """Root-cause failure clusters: shrink, hypothesize (Ultra), experiment (Super + testkit), decide (Fisher)."""
    from .investigator.loop import investigate as run_investigation
    from .report.build import default_runs

    def make(s: Settings):
        run_dirs = [p.resolve() for p in runs] or default_runs(s.runs_dir)
        return run_investigation(s, console, run_dirs, hypothesis_model=hypothesis_model, code_model=code_model,
                                 yes=yes, max_clusters=max_clusters, cluster_budget=cluster_budget)

    _run(make)


@app.command()
def reference(
    findings: Optional[Path] = typer.Option(None, "--findings", help="findings.jsonl (default: the latest investigation)."),
    yes: bool = typer.Option(False, "--yes", help="Send the calls after the plan is approved."),
) -> None:
    """Replay each finding on NVIDIA's hosted API (n=30) and attribute an owner."""
    from .reference import reference as run_reference

    _run(lambda s: run_reference(s, console, _latest_findings(s, findings), yes=yes))


@app.command()
def rerun(
    findings: Optional[Path] = typer.Option(None, "--findings", help="findings.jsonl (default: the latest investigation)."),
    yes: bool = typer.Option(False, "--yes", help="Send the paid calls after the estimate is approved."),
) -> None:
    """Re-run every finding's repro at n=30 and mark each still failing or fixed."""
    from .rerun import rerun as run_rerun

    _run(lambda s: run_rerun(s, console, _latest_findings(s, findings), yes=yes))


@app.command()
def check(
    tools: Optional[Path] = typer.Option(None, "--tools", help="Your tool definitions (OpenAI format, JSON list)."),
    schemas: Optional[Path] = typer.Option(None, "--schemas", help="Your JSON schemas ({name: schema})."),
    models_spec: str = typer.Option("auto", "--models", help="auto, or a comma list of model IDs."),
    n: int = typer.Option(50, "--n", help="Trials per argument set or schema, per model."),
    yes: bool = typer.Option(False, "--yes", help="Send the paid calls after the estimate is approved."),
) -> None:
    """Bring your own tools and schemas: per-tool, per-model reliability on Token Factory."""
    from .byo import check as run_check

    _run(lambda s: run_check(s, console, tools, schemas, models_spec=models_spec, n=n, yes=yes))
