"""The Tier 1 contract registry (CLAUDE.md §7): 17 contracts, 38 conditions."""

from __future__ import annotations

from . import t1_reasoning_leak, t2_thinking_switch, t3_tool_args, t5_schemas
from .base import BaseContract


def all_contracts() -> list[BaseContract]:
    return [*t1_reasoning_leak.CONTRACTS, *t2_thinking_switch.CONTRACTS, *t3_tool_args.CONTRACTS, *t5_schemas.CONTRACTS]


def by_id() -> dict[str, BaseContract]:
    return {c.id: c for c in all_contracts()}


def select(spec: str | None) -> list[BaseContract]:
    """Contracts matching a comma list of tiers or IDs, e.g. "T1,T3" or "T3.markup,T5". None = all."""
    contracts = all_contracts()
    if not spec:
        return contracts
    wanted = [s.strip() for s in spec.split(",") if s.strip()]
    chosen = [c for c in contracts if any(c.id == w or c.id.startswith(w + ".") for w in wanted)]
    unknown = [w for w in wanted if not any(c.id == w or c.id.startswith(w + ".") for c in contracts)]
    if unknown:
        raise ValueError(f"unknown contract(s): {', '.join(unknown)}")
    return chosen
