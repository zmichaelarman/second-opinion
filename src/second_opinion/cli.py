"""Command-line entry point (`second-opinion`). See CLAUDE.md §14."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any, Callable

import typer
from rich.console import Console

from .config import ConfigError, Settings, load_settings

app = typer.Typer(
    help="Check whether Nebius Token Factory keeps its promises on NVIDIA Nemotron models.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


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
