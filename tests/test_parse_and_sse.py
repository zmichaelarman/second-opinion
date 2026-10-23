from __future__ import annotations

import json
from pathlib import Path

from second_opinion.contracts.base import Condition, Outcome
from second_opinion.contracts.t3_tool_args import T3ToolArgsContract
from second_opinion.parse import (
    filter_headers,
    parse_json_response,
    parse_sse_chunks,
)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def test_filter_headers_strips_auth() -> None:
    raw_headers = {
        "Authorization": "Bearer <REDACTED>",
        "X-RateLimit-Limit-Requests": "60",
        "X-RateLimit-Remaining-Tokens": "390000",
        "Retry-After": "15",
        "X-Request-Id": "req-99",
        "Set-Cookie": "session=xyz",
    }
    filtered = filter_headers(raw_headers)
    assert "authorization" not in filtered
    assert "set-cookie" not in filtered
    assert filtered["x-ratelimit-limit-requests"] == "60"
    assert filtered["retry-after"] == "15"
    assert filtered["x-request-id"] == "req-99"


def test_parse_sse_chunks_from_recorded_fixture() -> None:
    chunks = json.loads((FIXTURES_DIR / "sse_tool_call_sample.json").read_text(encoding="utf-8"))
    parsed = parse_sse_chunks(
        status=200,
        headers={"x-ratelimit-limit-requests": "60"},
        data_payloads=chunks,
        total_s=1.25,
        ttft_s=0.18,
    )
    assert parsed.status == 200
    assert parsed.reasoning_field == "reasoning_content"
    assert "Preserving leading zeros." in parsed.reasoning
    assert len(parsed.tool_calls) == 1
    tc = parsed.tool_calls[0]
    assert tc["id"] == "call_abc123"
    assert tc["name"] == "lookup_account"
    assert json.loads(tc["arguments_raw"]) == {
        "zip_code": "02134",
        "account_id": "007",
        "numeric_label": "12345",
    }
    assert parsed.finish_reason == "tool_calls"
    assert parsed.usage == {"prompt_tokens": 120, "completion_tokens": 45, "total_tokens": 165}

    # Verify T3 oracle passes on the assembled stream response
    t3 = T3ToolArgsContract()
    cond = Condition(thinking=True, stream=True, variant="leading_zero")
    req = t3.build_request("nvidia/nemotron-3-super-120b-a12b", cond)
    ores = t3.check(req, parsed)
    assert ores.outcome == Outcome.PASS
    assert ores.fatal_codes == []


def test_parse_sse_missing_finish_and_usage() -> None:
    chunks = [
        '{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"lookup_account","arguments":"{\\"zip_code\\":\\"02134\\",\\"account_id\\":\\"007\\",\\"numeric_label\\":\\"12345\\"}"}}]}}]}',
        "[DONE]",
    ]
    parsed = parse_sse_chunks(status=200, headers={}, data_payloads=chunks)
    t3 = T3ToolArgsContract()
    cond = Condition(thinking=True, stream=True, variant="leading_zero")
    req = t3.build_request("nvidia/nemotron-3-super-120b-a12b", cond)
    ores = t3.check(req, parsed)
    assert ores.outcome == Outcome.FAIL
    assert "STREAM_FINISH_MISSING" in ores.fatal_codes
    assert "STREAM_USAGE_MISSING" in ores.soft_codes


def test_parse_json_response_reasoning_fields() -> None:
    body = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "banana",
                    "reasoning": "Alternative reasoning field name",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20},
    }
    parsed = parse_json_response(200, {}, body)
    assert parsed.content == "banana"
    assert parsed.reasoning == "Alternative reasoning field name"
    assert parsed.reasoning_field == "reasoning"
