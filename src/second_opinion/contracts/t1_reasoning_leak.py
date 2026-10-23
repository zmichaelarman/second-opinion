from __future__ import annotations

import json
from typing import Any

from jsonschema import Draft202012Validator

from second_opinion.contracts.base import (
    Condition,
    OracleResult,
    Severity,
    build_base_chat_request,
)
from second_opinion.fixtures.payloads import load_schema_definition
from second_opinion.oracles import (
    apply_common_soft_and_stream_checks,
    build_oracle_result,
    check_http_or_error,
    has_think_tags,
    normalize_exact_answer,
)
from second_opinion.parse import ParsedResponse

JSON_TRUNCATION_PROMPT = (
    "Return a JSON object for the film Alien (1979) with title, year and genre. "
    "Genre must be one of the allowed values."
)
BANANA_PROMPT = "Reply with exactly one word, lowercase, no punctuation: banana"


class T1ReasoningLeakContract:
    id = "T1"
    tier = 1
    title = "Reasoning leaks under token limits and separation"
    hypothesis = (
        "When max_tokens truncates output mid-reasoning or thinking is enabled, "
        "raw <think> blocks or reasoning prose may leak into message.content."
    )
    severity = Severity.S2_LOUD

    def __init__(self) -> None:
        self.conditions: list[Condition] = [
            Condition(thinking=True, stream=False, max_tokens=128, variant="json_truncation"),
            Condition(thinking=True, stream=False, max_tokens=512, variant="json_truncation"),
            Condition(thinking=True, stream=False, max_tokens=2048, variant="json_truncation"),
            Condition(thinking=True, stream=False, max_tokens=None, variant="separation"),
        ]
        self.core_conditions: set[Condition] = set()

    def build_request(self, model: str, cond: Condition, trial: int = 0) -> dict[str, Any]:
        if cond.variant == "json_truncation":
            req = build_base_chat_request(model, cond, JSON_TRUNCATION_PROMPT)
            schema = load_schema_definition("flat_basic")
            req["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "flat_basic",
                    "schema": schema,
                    "strict": True,
                },
            }
            return req
        return build_base_chat_request(model, cond, BANANA_PROMPT)

    def check(self, req: dict[str, Any], resp: ParsedResponse) -> OracleResult:
        early = check_http_or_error(resp, docs_shape=True)
        if early is not None:
            return early

        codes: list[str] = []
        detail: dict[str, Any] = {}
        is_json = "response_format" in req

        if is_json:
            content = resp.content
            if not content.strip():
                if resp.finish_reason == "length":
                    codes.append("TRUNCATED")
                else:
                    codes.append("EMPTY_CONTENT")
                apply_common_soft_and_stream_checks(req, resp, codes, detail)
                return build_oracle_result(codes, detail, docs_shape=True, default_severity=self.severity)

            try:
                parsed = json.loads(content)
            except Exception as exc:
                stripped = content.lstrip()
                if has_think_tags(content) or not stripped.startswith("{"):
                    codes.append("REASONING_IN_CONTENT")
                else:
                    codes.append("JSON_PARSE")
                detail["json_error"] = str(exc)
                detail["content_preview"] = content[:500]
                apply_common_soft_and_stream_checks(req, resp, codes, detail)
                return build_oracle_result(codes, detail, docs_shape=True, default_severity=self.severity)

            schema = load_schema_definition("flat_basic")
            validator = Draft202012Validator(schema)
            errors = sorted(validator.iter_errors(parsed), key=lambda e: list(e.path))
            if errors:
                first_kw = str(errors[0].validator)
                codes.append(f"SCHEMA_INVALID:{first_kw}")
                detail["schema_errors"] = [
                    {"validator": str(e.validator), "path": list(e.path), "message": e.message}
                    for e in errors
                ]
            apply_common_soft_and_stream_checks(req, resp, codes, detail)
            return build_oracle_result(codes, detail, docs_shape=True, default_severity=self.severity)

        # T1.separation: banana prompt
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
        return build_oracle_result(codes, detail, docs_shape=True, default_severity=self.severity)

