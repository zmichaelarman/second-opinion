from __future__ import annotations

from typing import Any

from second_opinion.contracts.base import (
    Condition,
    OracleResult,
    Severity,
    build_base_chat_request,
)
from second_opinion.fixtures.payloads import T5_PROMPTS, load_schema_definition
from second_opinion.oracles import check_schema_response
from second_opinion.parse import ParsedResponse

T5_VARIANTS = (
    "flat_basic",
    "enum_temptation",
    "no_extra_props",
    "min_max",
    "pattern",
    "array_bounds",
    "defs_ref",
    "anyof_optional",
)


def build_response_format(variant: str, schema: dict[str, Any], shape: str) -> dict[str, Any]:
    if shape == "json_object":
        return {"type": "json_object"}
    if shape == "bare":
        return {"type": "json_schema", "json_schema": schema}
    return {
        "type": "json_schema",
        "json_schema": {
            "name": variant,
            "schema": schema,
            "strict": True,
        },
    }


class T5SchemasContract:
    id = "T5"
    tier = 1
    title = "Structured output JSON schema enforcement and request shapes"
    hypothesis = (
        "Structured output requests across wrapped, bare, and json_object shapes "
        "must return valid JSON conforming to Draft 2020-12 schema constraints "
        "(enums, additionalProperties: false, numeric/array bounds, regex patterns, "
        "$defs/$ref, and anyOf nullable fields) even when prompts tempt violations."
    )
    severity = Severity.S2_LOUD

    def __init__(self) -> None:
        self.conditions: list[Condition] = []
        # 16 conditions: every variant at thinking False and thinking True (wrapped shape)
        for variant in T5_VARIANTS:
            self.conditions.append(
                Condition(thinking=False, stream=False, variant=variant, shape="wrapped")
            )
            self.conditions.append(
                Condition(thinking=True, stream=False, variant=variant, shape="wrapped")
            )
        # +3 conditions on flat_basic: bare, json_object, streaming (thinking False)
        self.conditions.append(
            Condition(thinking=False, stream=False, variant="flat_basic", shape="bare")
        )
        self.conditions.append(
            Condition(thinking=False, stream=False, variant="flat_basic", shape="json_object")
        )
        self.conditions.append(
            Condition(thinking=False, stream=True, variant="flat_basic", shape="wrapped")
        )

        self.core_conditions: set[Condition] = {
            Condition(thinking=False, stream=False, variant="flat_basic", shape="wrapped"),
            Condition(thinking=False, stream=False, variant="defs_ref", shape="wrapped"),
            Condition(thinking=False, stream=False, variant="anyof_optional", shape="wrapped"),
        }

    def build_request(self, model: str, cond: Condition, trial: int = 0) -> dict[str, Any]:
        variant = cond.variant
        prompt = T5_PROMPTS[variant]
        schema = load_schema_definition(variant)
        req = build_base_chat_request(model, cond, prompt)
        req["response_format"] = build_response_format(variant, schema, cond.shape)
        return req

    def check(self, req: dict[str, Any], resp: ParsedResponse) -> OracleResult:
        rf = req.get("response_format") or {}
        rf_type = rf.get("type", "json_schema")
        if rf_type == "json_object":
            return check_schema_response(req, resp, schema=None, shape="json_object")

        js = rf.get("json_schema") or {}
        if isinstance(js, dict) and "schema" in js and isinstance(js["schema"], dict):
            schema = js["schema"]
            shape = "wrapped"
        elif isinstance(js, dict):
            schema = js
            shape = "bare"
        else:
            schema = load_schema_definition("flat_basic")
            shape = "wrapped"

        return check_schema_response(req, resp, schema=schema, shape=shape)
