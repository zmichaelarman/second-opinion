from __future__ import annotations

import hashlib
import json
from typing import Any

from second_opinion.contracts.base import Contract
from second_opinion.contracts.t1_reasoning_leak import T1ReasoningLeakContract
from second_opinion.contracts.t2_thinking_switch import T2ThinkingSwitchContract
from second_opinion.contracts.t3_tool_args import T3ToolArgsContract
from second_opinion.contracts.t5_schemas import T5SchemasContract


def get_tier1_contracts() -> list[Contract]:
    return [
        T1ReasoningLeakContract(),
        T2ThinkingSwitchContract(),
        T3ToolArgsContract(),
        T5SchemasContract(),
    ]


def get_contract_map() -> dict[str, Contract]:
    return {c.id: c for c in get_tier1_contracts()}


def serialize_registry(contracts: list[Contract] | None = None) -> list[dict[str, Any]]:
    contracts = contracts if contracts is not None else get_tier1_contracts()
    serialized: list[dict[str, Any]] = []
    for c in contracts:
        core_set = c.core_conditions
        serialized.append(
            {
                "id": c.id,
                "tier": c.tier,
                "title": c.title,
                "hypothesis": c.hypothesis,
                "severity": c.severity.name,
                "conditions": [cond.to_dict() for cond in c.conditions],
                "core_conditions": sorted(
                    [cond.to_dict() for cond in core_set],
                    key=lambda d: json.dumps(d, sort_keys=True),
                ),
            }
        )
    return serialized


def compute_registry_hash(contracts: list[Contract] | None = None) -> str:
    data = serialize_registry(contracts)
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

