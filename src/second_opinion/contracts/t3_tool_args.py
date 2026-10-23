from __future__ import annotations

import json
from typing import Any

from second_opinion.contracts.base import (
    Condition,
    OracleResult,
    Severity,
    build_base_chat_request,
)
from second_opinion.fixtures.payloads import (
    T3_EXPECTED_ARGS,
    T3_TOOL_NAMES,
    load_tool_definition,
)
from second_opinion.oracles import check_tool_call
from second_opinion.parse import ParsedResponse

T3_VARIANTS = ("leading_zero", "numeric_types", "bool_null_nested", "markup")


def build_t3_prompt(tool_name: str, expected_args: dict[str, Any]) -> str:
    args_json = json.dumps(expected_args, indent=2, ensure_ascii=False)
    return (
        f"Call `{tool_name}` exactly once. Use exactly these arguments. "
        f"Copy every character and keep every type exactly as given:\n"
        f"```json\n{args_json}\n```"
    )


class T3ToolArgsContract:
    id = "T3"
    tier = 1
    title = "Tool-argument type and markup fidelity (batch and streaming)"
    hypothesis = (
        "Server-side conversion from Nemotron's XML-style tool markup (<tool_call>, "
        "<function=...>, <parameter=...>) into OpenAI tool_calls JSON may coerce "
        "leading-zero strings to integers, lose 64-bit integer precision, or truncate "
        "strings containing XML/HTML delimiter tags."
    )
    severity = Severity.S1_SILENT

    def __init__(self) -> None:
        self.conditions: list[Condition] = []
        for variant in T3_VARIANTS:
            self.conditions.append(Condition(thinking=False, stream=False, variant=variant))
            self.conditions.append(Condition(thinking=True, stream=False, variant=variant))
            self.conditions.append(Condition(thinking=True, stream=True, variant=variant))

        self.core_conditions: set[Condition] = {
            Condition(thinking=True, stream=False, variant="leading_zero"),
            Condition(thinking=True, stream=False, variant="markup"),
        }

    def build_request(self, model: str, cond: Condition, trial: int = 0) -> dict[str, Any]:
        variant = cond.variant
        tool_name = T3_TOOL_NAMES[variant]
        expected_args = T3_EXPECTED_ARGS[variant]
        tool_def = load_tool_definition(variant)
        prompt = build_t3_prompt(tool_name, expected_args)

        req = build_base_chat_request(model, cond, prompt)
        req["tools"] = [tool_def]
        req["tool_choice"] = {"type": "function", "function": {"name": tool_name}}
        return req

    def check(self, req: dict[str, Any], resp: ParsedResponse) -> OracleResult:
        # Infer variant from tool name or request metadata
        tools = req.get("tools") or []
        tool_name = ""
        tool_schema: dict[str, Any] = {}
        if tools and isinstance(tools[0], dict):
            fn = tools[0].get("function") or {}
            tool_name = str(fn.get("name") or "")
            tool_schema = dict(fn.get("parameters") or {})

        variant = "leading_zero"
        for var, tname in T3_TOOL_NAMES.items():
            if tname == tool_name:
                variant = var
                break

        expected_args = req.get("_expected_args") or T3_EXPECTED_ARGS[variant]
        if not tool_schema:
            tool_def = load_tool_definition(variant)
            tool_schema = tool_def["function"]["parameters"]
            tool_name = tool_def["function"]["name"]

        return check_tool_call(
            req=req,
            resp=resp,
            expected_tool=tool_name,
            expected_args=expected_args,
            tool_schema=tool_schema,
        )
