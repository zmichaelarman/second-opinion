from __future__ import annotations

from typing import Any

from second_opinion.contracts.base import (
    Condition,
    OracleResult,
    Outcome,
    Severity,
    build_base_chat_request,
)
from second_opinion.oracles import (
    apply_common_soft_and_stream_checks,
    build_oracle_result,
    check_http_or_error,
    has_think_tags,
    normalize_exact_answer,
)
from second_opinion.parse import ParsedResponse

BANANA_PROMPT = "Reply with exactly one word, lowercase, no punctuation: banana"


class T2ThinkingSwitchContract:
    id = "T2"
    tier = 1
    title = "Thinking switch (chat_template_kwargs.enable_thinking)"
    hypothesis = (
        "Setting enable_thinking=False disables reasoning and keeps completion_tokens <= 64, "
        "while enable_thinking=True populates a dedicated reasoning field."
    )
    severity = Severity.S3_GAP

    def __init__(self) -> None:
        self.conditions: list[Condition] = [
            Condition(thinking=False, stream=False, variant="off"),
            Condition(thinking=True, stream=False, variant="on"),
            Condition(thinking=None, stream=False, variant="default"),
        ]
        self.core_conditions: set[Condition] = set()

    def build_request(self, model: str, cond: Condition, trial: int = 0) -> dict[str, Any]:
        return build_base_chat_request(model, cond, BANANA_PROMPT)

    def check(self, req: dict[str, Any], resp: ParsedResponse) -> OracleResult:
        early = check_http_or_error(resp, docs_shape=False)
        if early is not None:
            return early

        kw = req.get("chat_template_kwargs")
        thinking_setting = kw.get("enable_thinking") if isinstance(kw, dict) else None

        # T2.default (thinking=None): informational only, always PASS
        if thinking_setting is None:
            comp_tokens = int((resp.usage or {}).get("completion_tokens", 0) or 0)
            detail = {
                "informational": True,
                "default_reasoning_present": bool(resp.reasoning.strip()),
                "reasoning_field": resp.reasoning_field,
                "completion_tokens": comp_tokens,
                "normalized_content": normalize_exact_answer(resp.content),
            }
            soft_codes: list[str] = []
            if resp.finish_reason == "length":
                soft_codes.append("TRUNCATED")
            if resp.refusal:
                soft_codes.append("REFUSAL")
            return OracleResult(
                outcome=Outcome.PASS,
                codes=soft_codes,
                fatal_codes=[],
                soft_codes=soft_codes,
                detail=detail,
                severity=None,
            )

        codes: list[str] = []
        detail = {
            "reasoning_field": resp.reasoning_field,
            "reasoning_len": len(resp.reasoning),
        }
        comp_tokens = int((resp.usage or {}).get("completion_tokens", 0) or 0)
        detail["completion_tokens"] = comp_tokens

        if thinking_setting is False:
            if resp.reasoning.strip() or comp_tokens > 64:
                codes.append("THINKING_NOT_DISABLED")
        elif thinking_setting is True:
            if not resp.reasoning.strip() or resp.reasoning_field is None:
                codes.append("NO_REASONING_FIELD")

        norm = normalize_exact_answer(resp.content)
        if norm != "banana":
            codes.append("EXTRA_CONTENT")
            detail["expected"] = "banana"
            detail["got_normalized"] = norm
            detail["got_raw"] = resp.content[:500]
            if has_think_tags(resp.content):
                codes.append("REASONING_IN_CONTENT")
        elif has_think_tags(resp.content):
            codes.append("REASONING_IN_CONTENT")

        apply_common_soft_and_stream_checks(req, resp, codes, detail)
        return build_oracle_result(codes, detail, docs_shape=False, default_severity=self.severity)

