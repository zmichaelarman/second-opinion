import json

import pytest

from second_opinion.contracts.base import Outcome
from second_opinion.fixtures.payloads import EXPECTED_ARGS, load_tool
from second_opinion.oracles import (
    Findings,
    check_exact_answer,
    check_json_content,
    check_stream,
    check_thinking_off,
    check_thinking_on,
    check_tool_call,
    http_gate,
    outcome_for,
    severity_of,
)
from second_opinion.parse import ParsedResponse

FILM_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "year": {"type": "integer"},
        "genre": {"enum": ["drama", "thriller", "sci-fi", "comedy", "horror", "fantasy"], "type": "string"},
    },
    "required": ["title", "year", "genre"],
}


def resp(**kw) -> ParsedResponse:
    kw.setdefault("status", 200)
    kw.setdefault("headers", {})
    return ParsedResponse(**kw)


def run(check, r, *args, **kwargs):
    f = Findings()
    check(f, r, *args, **kwargs)
    return f.result()


@pytest.mark.parametrize(
    "codes,expected",
    [
        ([], Outcome.PASS),
        (["TRUNCATED", "REFUSAL", "STREAM_USAGE_MISSING", "FINISH_REASON_WRONG"], Outcome.PASS),
        (["SCHEMA_INVALID:enum"], Outcome.FAIL),
        (["TYPE_COERCED", "FINISH_REASON_WRONG"], Outcome.FAIL),
        (["HTTP_5XX", "TYPE_COERCED"], Outcome.ERROR),
        (["TIMEOUT"], Outcome.ERROR),
    ],
)
def test_outcome_rule(codes, expected):
    assert outcome_for(codes) == expected


def test_http_4xx_severity_depends_on_docs_shape():
    assert severity_of("HTTP_4XX").value == 3
    assert severity_of("HTTP_4XX", docs_shape=True).value == 2
    assert severity_of("TYPE_COERCED").value == 1
    assert severity_of("SCHEMA_INVALID:pattern").value == 2


@pytest.mark.parametrize(
    "kw,outcome,codes",
    [
        ({"status": 200}, None, None),
        ({"status": 400, "error_message": "bad"}, Outcome.FAIL, ["HTTP_4XX"]),
        ({"status": 429}, Outcome.ERROR, ["HTTP_429"]),
        ({"status": 503}, Outcome.ERROR, ["HTTP_5XX"]),
        ({"status": None, "error": "TIMEOUT"}, Outcome.ERROR, ["TIMEOUT"]),
        ({"status": None, "error": "ConnectError"}, Outcome.ERROR, ["TRANSPORT_ERROR"]),
        ({"status": 200, "sse": True, "error_message": "boom"}, Outcome.ERROR, ["HTTP_5XX"]),
    ],
)
def test_http_gate(kw, outcome, codes):
    result = http_gate(resp(**kw))
    if outcome is None:
        assert result is None
    else:
        assert (result.outcome, result.codes) == (outcome, codes)


def test_exact_answer_normalizes_case_and_punctuation():
    assert run(check_exact_answer, resp(content="  Banana.\n", finish_reason="stop"), "banana").codes == []


def test_exact_answer_flags_think_tags():
    r = run(check_exact_answer, resp(content="<think>hmm</think>banana", finish_reason="stop"), "banana")
    assert r.codes == ["EXTRA_CONTENT", "REASONING_IN_CONTENT"] and r.outcome == Outcome.FAIL


def test_exact_answer_empty_content():
    assert run(check_exact_answer, resp(content="", finish_reason="stop"), "banana").codes == ["EMPTY_CONTENT"]
    truncated = run(check_exact_answer, resp(content=None, finish_reason="length"), "banana")
    assert truncated.codes == ["TRUNCATED"] and truncated.outcome == Outcome.PASS


def test_thinking_off_and_on():
    assert run(check_thinking_off, resp(reasoning=None, usage={"completion_tokens": 3})).codes == []
    assert run(check_thinking_off, resp(reasoning="x", usage={"completion_tokens": 3})).codes == ["THINKING_NOT_DISABLED"]
    assert run(check_thinking_off, resp(usage={"completion_tokens": 65})).codes == ["THINKING_NOT_DISABLED"]
    assert run(check_thinking_on, resp(reasoning="thought")).codes == []
    on = run(check_thinking_on, resp(reasoning=None, content="<think>x</think>banana"))
    assert on.codes == ["NO_REASONING_FIELD"] and on.detail["NO_REASONING_FIELD"]["think_tags_in_content"]


@pytest.mark.parametrize(
    "content,finish,codes",
    [
        ("", "length", ["TRUNCATED"]),
        ("", "stop", ["EMPTY_CONTENT"]),
        ('{"title": "Alien", "year": 1979, "genre": "sci-fi"}', "stop", []),
        ('{"title": "Alien", "year": 1979, "genre": "space opera"}', "stop", ["SCHEMA_INVALID:enum"]),
        ('{"title": "Alien", "year": "1979", "genre": "sci-fi"}', "stop", ["SCHEMA_INVALID:type"]),
        ("<think>let me see</think>{}", "stop", ["REASONING_IN_CONTENT"]),
        ('Sure! Here it is: {"title": "Alien"}', "stop", ["REASONING_IN_CONTENT"]),
        ('{"title": "Alien", "year": 19', "length", ["TRUNCATED", "JSON_PARSE"]),
        ('{"title": "Alien", "year": NaN, "genre": "sci-fi"}', "stop", ["JSON_PARSE"]),
    ],
)
def test_json_content_oracle(content, finish, codes):
    assert run(check_json_content, resp(content=content, finish_reason=finish), FILM_SCHEMA).codes == codes


def test_json_object_shape_requires_an_object():
    assert run(check_json_content, resp(content="[1, 2]", finish_reason="stop"), None, True).codes == ["SCHEMA_INVALID:type"]
    assert run(check_json_content, resp(content='{"a": 1}', finish_reason="stop"), None, True).codes == []


def _tool_resp(args, name="save_account", finish="tool_calls", content=None):
    raw = args if isinstance(args, str) else json.dumps(args)
    return resp(content=content, finish_reason=finish, tool_calls=[{"index": 0, "id": "c", "name": name, "arguments_raw": raw}])


def _check_lz(r):
    tool = load_tool("leading_zero")
    return run(check_tool_call, r, "save_account", EXPECTED_ARGS["leading_zero"], tool["function"]["parameters"])


def test_tool_call_pass_and_soft_finish_reason():
    assert _check_lz(_tool_resp(EXPECTED_ARGS["leading_zero"])).codes == []
    soft = _check_lz(_tool_resp(EXPECTED_ARGS["leading_zero"], finish="stop"))
    assert soft.codes == ["FINISH_REASON_WRONG"] and soft.outcome == Outcome.PASS


def test_tool_call_missing_with_markup_in_content():
    r = _check_lz(resp(content='<tool_call>\n<function=save_account>\n<parameter=zip_code>02134', finish_reason="stop"))
    assert r.codes == ["NO_TOOL_CALLS", "TOOL_MARKUP_IN_CONTENT"]


def test_tool_call_wrong_tool_and_bad_json():
    assert _check_lz(_tool_resp(EXPECTED_ARGS["leading_zero"], name="other")).codes == ["WRONG_TOOL"]
    assert _check_lz(_tool_resp('{"zip_code": "02134",')).codes == ["ARGS_JSON_INVALID"]


def test_tool_call_coercion_is_silent_failure():
    r = _check_lz(_tool_resp({"zip_code": 2134, "account_id": "007", "numeric_label": "12345"}))
    assert r.codes == ["SCHEMA_INVALID:type", "TYPE_COERCED"] and r.outcome == Outcome.FAIL
    assert r.detail["mismatches"][0]["path"] == "$.zip_code"


def test_stream_oracles():
    assert run(check_stream, resp(sse=True, finish_reason="stop", usage={"completion_tokens": 1})).codes == []
    r = run(check_stream, resp(sse=True, finish_reason=None, usage=None))
    assert r.codes == ["STREAM_FINISH_MISSING", "STREAM_USAGE_MISSING"]
