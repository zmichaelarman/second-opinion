"""T3: tool-argument fidelity, including the T4 streaming conditions (CLAUDE.md §7).

Severity S1 for coercion and truncation (silent corruption), S2 otherwise.
"""

from __future__ import annotations

from typing import Any

from ..fixtures.payloads import EXPECTED_ARGS, T3_VARIANTS, forced_tool_choice, load_tool, tool_name, tool_prompt
from ..oracles import Findings, check_no_think_tags, check_stream, check_tool_call, http_gate
from ..parse import ParsedResponse
from .base import THINKING_SWITCH, BaseContract, Condition, OracleResult, Severity, chat_request

CORE_VARIANTS = ("leading_zero", "markup")
DESCRIPTIONS = {
    "leading_zero": "digit strings with leading zeros stay strings",
    "numeric_types": "integers, a big integer (2^53 + 1) and a float keep their exact types and values",
    "bool_null_nested": "false, null, an array and a nested object come back unchanged",
    "markup": "a string holding XML-like markup, quotes, a backslash and non-ASCII text survives intact",
}


def tool_request(model: str, variant: str, cond: Condition, switch: str = THINKING_SWITCH) -> dict[str, Any]:
    tool = load_tool(variant)
    name = tool_name(tool)
    body = chat_request(model, tool_prompt(name, EXPECTED_ARGS[variant]), cond, switch=switch)
    body["tools"] = [tool]
    body["tool_choice"] = forced_tool_choice(name)
    return body


def check_t3(variant: str, req: dict[str, Any], resp: ParsedResponse) -> OracleResult:
    gate = http_gate(resp)
    if gate:
        return gate
    tool = load_tool(variant)
    f = Findings()
    check_tool_call(f, resp, tool_name(tool), EXPECTED_ARGS[variant], tool["function"]["parameters"])
    check_no_think_tags(f, resp)
    if req.get("stream"):
        check_stream(f, resp, include_usage=bool((req.get("stream_options") or {}).get("include_usage")))
    return f.result()


class ToolArgs(BaseContract):
    def __init__(self, variant: str) -> None:
        conditions = [
            Condition(thinking=False, stream=False, variant=variant),
            Condition(thinking=True, stream=False, variant=variant),
            Condition(thinking=True, stream=True, variant=variant),
        ]
        super().__init__(
            id=f"T3.{variant}",
            tier=3,
            title=f"Tool arguments: {variant}",
            hypothesis=f"A forced tool call copies its arguments exactly: {DESCRIPTIONS[variant]}.",
            severity=Severity.S1_SILENT,
            conditions=conditions,
            core_conditions={conditions[1]} if variant in CORE_VARIANTS else set(),
        )
        self.variant = variant

    def build_request(self, model: str, cond: Condition, trial: int) -> dict[str, Any]:
        return tool_request(model, self.variant, cond)

    def check(self, req: dict[str, Any], resp: ParsedResponse) -> OracleResult:
        return check_t3(self.variant, req, resp)


CONTRACTS: list[BaseContract] = [ToolArgs(v) for v in T3_VARIANTS]
