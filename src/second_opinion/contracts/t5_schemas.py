"""T5: schema enforcement through response_format (CLAUDE.md §7, severity S2)."""

from __future__ import annotations

from typing import Any

from ..fixtures.schemas import load_schema
from ..oracles import Findings, check_json_content, check_stream, http_gate
from ..parse import ParsedResponse
from .base import BaseContract, Condition, OracleResult, Severity, chat_request

PROMPTS = {
    "flat_basic": "Give details of a real film starring Jack Nicholson.",
    "enum_temptation": "Describe the film La La Land, a musical.",
    "no_extra_props": "Describe the film Alien (1979). Also include its box office gross.",
    "min_max": "The critic rated Alien 11 out of 10.",
    "pattern": "My ticket code is abc-12.",
    "array_bounds": "Tag this film: horror, sci-fi, space, classic, thriller.",
    "defs_ref": "Ada (ada@example.com) ordered 2 of SKU A1 and 1 of SKU B7.",
    "anyof_optional": "The person's name is Ada. Nickname and age are unknown.",
}
TEMPTATION = {
    "flat_basic": "a plain film object",
    "enum_temptation": "a genre outside the enum (musical)",
    "no_extra_props": "an extra property (box office gross)",
    "min_max": "a number above the maximum (11 of 10)",
    "pattern": "a code that breaks the pattern (abc-12)",
    "array_bounds": "more array items than maxItems (five tags)",
    "defs_ref": "nested objects through $defs and $ref",
    "anyof_optional": "nullable fields through anyOf with null",
}
CORE_VARIANTS = ("flat_basic", "defs_ref", "anyof_optional")
SHAPES = ("wrapped", "bare", "json_object")


def response_format(variant: str, shape: str) -> dict[str, Any]:
    """The three request shapes the docs show for structured output."""
    if shape == "wrapped":
        return {"type": "json_schema", "json_schema": {"name": variant, "schema": load_schema(variant), "strict": True}}
    if shape == "bare":
        return {"type": "json_schema", "json_schema": load_schema(variant)}
    if shape == "json_object":
        return {"type": "json_object"}
    raise ValueError(f"unknown shape {shape!r}")


class SchemaContract(BaseContract):
    def __init__(self, variant: str) -> None:
        conditions = [Condition(thinking=False, variant=variant), Condition(thinking=True, variant=variant)]
        if variant == "flat_basic":
            conditions += [
                Condition(thinking=False, variant=variant, shape="bare"),
                Condition(thinking=False, variant=variant, shape="json_object"),
                Condition(thinking=False, stream=True, variant=variant),
            ]
        super().__init__(
            id=f"T5.{variant}",
            tier=5,
            title=f"Schema enforcement: {variant}",
            hypothesis=(
                "With response_format json_schema, content is JSON that validates against the schema, "
                f"even when the prompt tempts a violation ({TEMPTATION[variant]})."
            ),
            severity=Severity.S2_LOUD,
            conditions=conditions,
            core_conditions={conditions[0]} if variant in CORE_VARIANTS else set(),
        )
        self.variant = variant

    def build_request(self, model: str, cond: Condition, trial: int) -> dict[str, Any]:
        body = chat_request(model, PROMPTS[self.variant], cond)
        body["response_format"] = response_format(self.variant, cond.shape)
        return body

    def check(self, req: dict[str, Any], resp: ParsedResponse) -> OracleResult:
        # Both json_schema shapes and json_object appear in the docs, so a 4xx is a fatal docs-shape rejection.
        gate = http_gate(resp, docs_shape=True)
        if gate:
            return gate
        f = Findings()
        shape = (req.get("response_format") or {}).get("type")
        if shape == "json_object":
            check_json_content(f, resp, None, require_object=True)
        else:
            check_json_content(f, resp, load_schema(self.variant))
        if resp.refusal:
            f.add("REFUSAL", refusal=resp.refusal[:300])
        if req.get("stream"):
            check_stream(f, resp)
        return f.result()


CONTRACTS: list[BaseContract] = [SchemaContract(v) for v in PROMPTS]
