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
