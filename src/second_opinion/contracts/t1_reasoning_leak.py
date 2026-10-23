"""T1: reasoning leaks under token limits (CLAUDE.md §7, severity S2)."""

from __future__ import annotations

from typing import Any

from ..fixtures.schemas import load_schema
from ..oracles import Findings, check_exact_answer, check_json_content, http_gate
from ..parse import ParsedResponse
from .base import BANANA_PROMPT, BaseContract, Condition, OracleResult, Severity, chat_request
from .t5_schemas import response_format

ALIEN_PROMPT = (
    "Return a JSON object for the film Alien (1979) with title, year and genre. "
    "Genre must be one of the allowed values."
)
TRUNCATION_LIMITS = (128, 512, 2048)


class JsonTruncation(BaseContract):
    def __init__(self) -> None:
        super().__init__(
            id="T1.json_truncation",
            tier=1,
            title="JSON output when max_tokens cuts reasoning short",
            hypothesis=(
                "With thinking on and a json_schema response_format, a response cut off by max_tokens has "
                "empty content or schema-valid JSON in content, never reasoning text."
            ),
            severity=Severity.S2_LOUD,
            conditions=[Condition(thinking=True, max_tokens=m, variant="flat_basic") for m in TRUNCATION_LIMITS],
        )

    def build_request(self, model: str, cond: Condition, trial: int) -> dict[str, Any]:
        body = chat_request(model, ALIEN_PROMPT, cond)
        body["response_format"] = response_format("flat_basic", "wrapped")
        return body

    def check(self, req: dict[str, Any], resp: ParsedResponse) -> OracleResult:
        gate = http_gate(resp, docs_shape=True)
        if gate:
            return gate
        f = Findings()
        check_json_content(f, resp, load_schema("flat_basic"))
        return f.result()


class Separation(BaseContract):
    def __init__(self) -> None:
        super().__init__(
            id="T1.separation",
            tier=1,
            title="Reasoning stays out of content",
            hypothesis="With thinking on, content holds only the answer; reasoning never leaks into it.",
            severity=Severity.S2_LOUD,
            conditions=[Condition(thinking=True)],
        )

    def build_request(self, model: str, cond: Condition, trial: int) -> dict[str, Any]:
        return chat_request(model, BANANA_PROMPT, cond)

    def check(self, req: dict[str, Any], resp: ParsedResponse) -> OracleResult:
        gate = http_gate(resp)
        if gate:
            return gate
        f = Findings()
        check_exact_answer(f, resp, "banana")
        return f.result()


CONTRACTS: list[BaseContract] = [JsonTruncation(), Separation()]
