"""T2: the thinking switch (CLAUDE.md §7, severity S3). All three contracts use the banana prompt."""

from __future__ import annotations

from typing import Any

from ..oracles import THINK_TAG, Findings, check_exact_answer, check_thinking_off, check_thinking_on, http_gate
from ..parse import ParsedResponse
from .base import BANANA_PROMPT, THINKING_SWITCH, BaseContract, Condition, OracleResult, Outcome, Severity, chat_request

EXPECTED_ANSWER = "banana"


def banana_request(model: str, cond: Condition, switch: str = THINKING_SWITCH) -> dict[str, Any]:
    return chat_request(model, BANANA_PROMPT, cond, switch=switch)


def check_banana(thinking: bool | None, req: dict[str, Any], resp: ParsedResponse) -> OracleResult:
    gate = http_gate(resp)
    if gate:
        return gate
    f = Findings()
    check_exact_answer(f, resp, EXPECTED_ANSWER)
    if thinking is False:
        check_thinking_off(f, resp)
    elif thinking is True:
        check_thinking_on(f, resp)
    return f.result()


class ThinkingSwitch(BaseContract):
    def __init__(self, thinking: bool | None) -> None:
        name = {False: "off", True: "on", None: "default"}[thinking]
        hypothesis = {
            False: "enable_thinking=false removes reasoning: no reasoning text and at most 64 completion tokens.",
            True: "enable_thinking=true returns reasoning in a separate field, and content holds only the answer.",
            None: "Informational: whether reasoning is on by default, and which field carries it.",
        }[thinking]
        super().__init__(
            id=f"T2.{name}",
            tier=2,
            title=f"Thinking switch: {name}",
            hypothesis=hypothesis,
            severity=Severity.S3_GAP,
            conditions=[Condition(thinking=thinking)],
        )
        self.thinking = thinking

    def build_request(self, model: str, cond: Condition, trial: int) -> dict[str, Any]:
        return banana_request(model, cond)

    def check(self, req: dict[str, Any], resp: ParsedResponse) -> OracleResult:
        if self.thinking is not None:
            return check_banana(self.thinking, req, resp)
        gate = http_gate(resp)
        if gate:
            return gate
        return OracleResult(
            Outcome.PASS,
            [],
            {
                "reasoning_present": bool((resp.reasoning or "").strip()),
                "reasoning_field": resp.reasoning_field,
                "think_tags_in_content": bool(THINK_TAG.search(resp.content or "")),
                "completion_tokens": resp.completion_tokens,
            },
        )


CONTRACTS: list[BaseContract] = [ThinkingSwitch(False), ThinkingSwitch(True), ThinkingSwitch(None)]
