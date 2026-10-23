from __future__ import annotations

import json

from second_opinion.contracts.base import Condition, Outcome, Severity
from second_opinion.contracts.t1_reasoning_leak import T1ReasoningLeakContract
from second_opinion.contracts.t2_thinking_switch import T2ThinkingSwitchContract
from second_opinion.contracts.t3_tool_args import T3ToolArgsContract
from second_opinion.contracts.t5_schemas import T5SchemasContract
from second_opinion.fixtures.payloads import MARKUP_STRING, T3_EXPECTED_ARGS
from second_opinion.oracles import typed_equal
from second_opinion.parse import ParsedResponse


def test_typed_equal_exact_and_bool_vs_int() -> None:
    assert typed_equal({"a": "02134", "b": 42, "c": False}, {"a": "02134", "b": 42, "c": False}).equal

    # A bool is never equal to an int
    res_bool_int = typed_equal({"flag": False}, {"flag": 0})
    assert not res_bool_int.equal
    assert "TYPE_COERCED" in res_bool_int.codes

    res_int_bool = typed_equal({"count": 1}, {"count": True})
    assert not res_int_bool.equal
    assert "TYPE_COERCED" in res_int_bool.codes


def test_typed_equal_coercions_and_precision_loss() -> None:
    # Leading-zero string coerced to int
    r1 = typed_equal({"zip_code": "02134"}, {"zip_code": 2134})
    assert not r1.equal
    assert r1.codes == ["TYPE_COERCED"]

    # Int coerced to string
    r2 = typed_equal({"street_number": 42}, {"street_number": "42"})
    assert not r2.equal
    assert r2.codes == ["TYPE_COERCED"]

    # Int coerced to float of equal value
    r3 = typed_equal({"street_number": 42}, {"street_number": 42.0})
    assert not r3.equal
    assert r3.codes == ["TYPE_COERCED"]

    # Big-int precision loss (2^53 + 1 -> 2^53)
    r4 = typed_equal({"big_id": 9007199254740993}, {"big_id": 9007199254740992})
    assert not r4.equal
    assert r4.codes == ["TYPE_COERCED"]
    assert r4.detail.get("precision_loss") == "PRECISION_LOSS"


def test_typed_equal_value_truncated_and_mismatch() -> None:
    # Truncated right before </parameter>
    prefix = '<div class="note">5 < 6 && 7 > 3</div>\n'
    r_trunc = typed_equal({"content": MARKUP_STRING}, {"content": prefix})
    assert not r_trunc.equal
    assert r_trunc.codes == ["VALUE_TRUNCATED"]

    # Truncated without trailing newline before </parameter>
    r_trunc2 = typed_equal({"content": MARKUP_STRING}, {"content": prefix.rstrip("\n")})
    assert not r_trunc2.equal
    assert r_trunc2.codes == ["VALUE_TRUNCATED"]

    # Arbitrary mismatch produces VALUE_MISMATCH and unified diff
    r_mis = typed_equal({"zip_code": "02134"}, {"zip_code": "99999"})
    assert not r_mis.equal
    assert r_mis.codes == ["VALUE_MISMATCH"]
    assert "diff" in r_mis.detail


def test_t1_reasoning_leak_oracles() -> None:
    t1 = T1ReasoningLeakContract()
    cond_json = Condition(thinking=True, max_tokens=128, variant="json_truncation")
    req_json = t1.build_request("nvidia/Nemotron-3_5-Lightning", cond_json)

    # Empty content with finish_reason=length -> PASS with soft TRUNCATED
    resp_len = ParsedResponse(status=200, content="", finish_reason="length")
    res_len = t1.check(req_json, resp_len)
    assert res_len.outcome == Outcome.PASS
    assert "TRUNCATED" in res_len.soft_codes

    # Empty content with finish_reason=stop -> FAIL with EMPTY_CONTENT
    resp_stop = ParsedResponse(status=200, content="", finish_reason="stop")
    res_stop = t1.check(req_json, resp_stop)
    assert res_stop.outcome == Outcome.FAIL
    assert "EMPTY_CONTENT" in res_stop.fatal_codes

    # Unclosed <think> tag leaked into content -> FAIL with REASONING_IN_CONTENT
    resp_leak = ParsedResponse(status=200, content="<think>Let me recall Alien 1979...", finish_reason="length")
    res_leak = t1.check(req_json, resp_leak)
    assert res_leak.outcome == Outcome.FAIL
    assert "REASONING_IN_CONTENT" in res_leak.fatal_codes

    # Valid JSON conforming to flat_basic -> PASS
    valid_film = json.dumps({"title": "Alien", "year": 1979, "genre": "sci-fi"})
    resp_ok = ParsedResponse(status=200, content=valid_film, finish_reason="stop")
    assert t1.check(req_json, resp_ok).outcome == Outcome.PASS


def test_t2_thinking_switch_oracles() -> None:
    t2 = T2ThinkingSwitchContract()

    # T2.off: fails if reasoning present or completion_tokens > 64
    req_off = t2.build_request("m", Condition(thinking=False, variant="off"))
    resp_off_bad = ParsedResponse(
        status=200,
        content="banana",
        reasoning="still thinking",
        reasoning_field="reasoning_content",
        usage={"completion_tokens": 120},
        finish_reason="stop",
    )
    res_off_bad = t2.check(req_off, resp_off_bad)
    assert res_off_bad.outcome == Outcome.FAIL
    assert "THINKING_NOT_DISABLED" in res_off_bad.fatal_codes

    resp_off_ok = ParsedResponse(
        status=200,
        content="Banana.",
        reasoning="",
        usage={"completion_tokens": 5},
        finish_reason="stop",
    )
    assert t2.check(req_off, resp_off_ok).outcome == Outcome.PASS

    # T2.on: requires separate reasoning field
    req_on = t2.build_request("m", Condition(thinking=True, variant="on"))
    resp_on_no_reason = ParsedResponse(status=200, content="banana", reasoning="", reasoning_field=None)
    res_on_bad = t2.check(req_on, resp_on_no_reason)
    assert res_on_bad.outcome == Outcome.FAIL
    assert "NO_REASONING_FIELD" in res_on_bad.fatal_codes

    # T2.default: always PASS (informational)
    req_def = t2.build_request("m", Condition(thinking=None, variant="default"))
    resp_def = ParsedResponse(status=200, content="something else", reasoning="thought", reasoning_field="reasoning_content")
    assert t2.check(req_def, resp_def).outcome == Outcome.PASS


def test_t3_tool_call_oracles_and_severity() -> None:
    t3 = T3ToolArgsContract()
    cond = Condition(thinking=True, stream=False, variant="leading_zero")
    req = t3.build_request("m", cond)

    # Coerced zip_code -> FAIL with TYPE_COERCED and Severity.S1_SILENT
    bad_args = {"zip_code": 2134, "account_id": "007", "numeric_label": "12345"}
    resp_coerced = ParsedResponse(
        status=200,
        tool_calls=[{"index": 0, "id": "1", "name": "lookup_account", "arguments_raw": json.dumps(bad_args)}],
        finish_reason="tool_calls",
    )
    res_coerced = t3.check(req, resp_coerced)
    assert res_coerced.outcome == Outcome.FAIL
    assert "TYPE_COERCED" in res_coerced.fatal_codes
    assert res_coerced.severity == Severity.S1_SILENT

    # Raw XML markup leaked in content -> TOOL_MARKUP_IN_CONTENT
    resp_markup_leak = ParsedResponse(
        status=200,
        content="<tool_call><function=lookup_account><parameter=zip_code>02134</parameter>",
        tool_calls=[],
        finish_reason="stop",
    )
    res_leak = t3.check(req, resp_markup_leak)
    assert res_leak.outcome == Outcome.FAIL
    assert "NO_TOOL_CALLS" in res_leak.fatal_codes
    assert "TOOL_MARKUP_IN_CONTENT" in res_leak.fatal_codes

    # Exact match -> PASS
    resp_ok = ParsedResponse(
        status=200,
        tool_calls=[
            {
                "index": 0,
                "id": "1",
                "name": "lookup_account",
                "arguments_raw": json.dumps(T3_EXPECTED_ARGS["leading_zero"]),
            }
        ],
        finish_reason="tool_calls",
    )
    assert t3.check(req, resp_ok).outcome == Outcome.PASS


def test_t5_schema_oracles_and_http_error_handling() -> None:
    t5 = T5SchemasContract()
    cond = Condition(thinking=False, variant="min_max", shape="wrapped")
    req = t5.build_request("m", cond)

    # Violates maximum: 10 -> SCHEMA_INVALID:maximum
    resp_invalid = ParsedResponse(
        status=200,
        content=json.dumps({"title": "Alien", "rating": 11}),
        finish_reason="stop",
    )
    res_inv = t5.check(req, resp_invalid)
    assert res_inv.outcome == Outcome.FAIL
    assert "SCHEMA_INVALID:maximum" in res_inv.fatal_codes

    # HTTP 503 -> ERROR (never FAIL)
    resp_503 = ParsedResponse(status=503, error_type="HTTP_5XX")
    res_503 = t5.check(req, resp_503)
    assert res_503.outcome == Outcome.ERROR

