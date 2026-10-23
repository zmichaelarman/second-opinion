"""Model discovery: GET /v1/models, filtered to Nemotron chat models (CLAUDE.md §3).

Runtime discovery is authoritative; a model is only run if it also has a price row.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from rich.console import Console
from rich.table import Table

from .client import ApiClient
from .config import Price, Settings


@dataclass
class DiscoveredModel:
    id: str
    included: bool
    reason: str
    price: Price | None = None
    raw: dict[str, Any] | None = None

    @property
    def runnable(self) -> bool:
        return self.included and self.price is not None

    @property
    def role(self) -> str | None:
        return self.price.role if self.price else None


def classify(model_id: str, include: list[str], exclude: list[str]) -> tuple[bool, str]:
    low = model_id.lower()
    if not any(term.lower() in low for term in include):
        return False, "not nemotron"
    for term in exclude:
        # Match at the start of a word: "VL" catches "-VL-" and "vl2", not letters inside a word.
        if re.search(rf"(?<![a-z0-9]){re.escape(term.lower())}", low):
            return False, f"excluded ({term})"
    return True, "nemotron"


def discover(models_body: Any, settings: Settings) -> list[DiscoveredModel]:
    rules = settings.defaults["discovery"]
    data = models_body.get("data", []) if isinstance(models_body, dict) else []
    found = []
    for entry in data:
        model_id = entry.get("id") if isinstance(entry, dict) else None
        if not model_id:
            continue
        included, reason = classify(model_id, rules["include"], rules["exclude"])
        price = settings.price_for(model_id) if included else None
        found.append(DiscoveredModel(model_id, included, reason, price, entry))
    return sorted(found, key=lambda d: (not d.included, d.id.lower()))


def runnable(models: list[DiscoveredModel]) -> list[DiscoveredModel]:
    return [m for m in models if m.runnable]


def models_table(models: list[DiscoveredModel], show_all: bool = False) -> Table:
    table = Table(title="Token Factory models (GET /v1/models)")
    for col in ("Model ID", "Status", "Role", "$/1M in", "$/1M out", "Price source"):
        table.add_column(col)
    for m in models:
        if not show_all and "nemotron" not in m.id.lower():
            continue
        if m.runnable:
            status = "[green]will test[/green]"
            if m.price and m.price.model != m.id:
                status += " [yellow](price row casing differs)[/yellow]"
        elif m.included:
            status = "[yellow]no price row[/yellow]"
        else:
            status = f"[dim]{m.reason}[/dim]"
        p = m.price
        table.add_row(
            m.id, status, m.role or "", f"{p.input:.2f}" if p else "", f"{p.output:.2f}" if p else "",
            f"{p.source}, {p.as_of}" if p else "",
        )
    return table


async def list_models(settings: Settings, console: Console, show_all: bool = False) -> int:
    async with ApiClient.for_provider(settings, "tokenfactory") as client:
        ex = await client.get("models")
    if ex.status != 200:
        console.print(f"[red]GET /v1/models failed[/red]: status={ex.status} error={ex.error} body={str(ex.body)[:300]}")
        return 1
    models = discover(ex.body, settings)
    console.print(models_table(models, show_all))
    hidden = sum(1 for m in models if "nemotron" not in m.id.lower())
    console.print(
        f"{len(models)} models listed; {sum(m.included for m in models)} Nemotron after filtering; "
        f"{len(runnable(models))} will be tested." + ("" if show_all else f" {hidden} non-Nemotron hidden (--all).")
    )
    unpriced = [m.id for m in models if m.included and not m.price]
    if unpriced:
        console.print(f"[yellow]No price row in config/prices.yaml (not run): {', '.join(unpriced)}[/yellow]")
    if models and models[0].raw:
        console.print(f"[dim]Fields per model entry: {', '.join(sorted(models[0].raw))}[/dim]")
    return 0
