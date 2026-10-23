from second_opinion.client import Exchange
from second_opinion.parse import RAW_LIMIT_BYTES, parse_exchange, truncate_raw

from fake_tf import completion, stream_chunks


def parse_json(body, status=200):
    return parse_exchange(Exchange(status, {}, body, False))


def parse_stream(payloads):
    return parse_exchange(Exchange(200, {}, payloads, True))


def test_non_stream_reasoning_content():
    p = parse_json(completion("banana", "thinking", usage=(10, 4)))
    assert (p.content, p.reasoning, p.reasoning_field) == ("banana", "thinking", "reasoning_content")
    assert (p.finish_reason, p.prompt_tokens, p.completion_tokens) == ("stop", 10, 4)


def test_non_stream_reasoning_field_named_reasoning():
    p = parse_json(completion("banana", "thinking", reasoning_field="reasoning"))
    assert (p.reasoning, p.reasoning_field, p.reasoning_fields_seen) == ("thinking", "reasoning", ["reasoning"])


def test_non_stream_both_fields_prefers_reasoning_content():
    body = completion("banana", "a")
    body["choices"][0]["message"]["reasoning"] = "a"
    p = parse_json(body)
    assert p.reasoning == "a" and p.reasoning_fields_seen == ["reasoning_content", "reasoning"]


def test_non_stream_tool_calls():
    p = parse_json(completion(None, None, [("save_account", '{"zip_code": "02134"}')], "tool_calls"))
    assert p.tool_calls == [{"index": 0, "id": "call_0", "name": "save_account", "arguments_raw": '{"zip_code": "02134"}'}]


def test_error_body_gives_error_message():
    p = parse_json({"error": {"message": "Unknown parameter: response format"}}, status=400)
    assert p.error_message == "Unknown parameter: response format" and p.content is None


def test_stream_assembly_content_reasoning_usage():
    payloads = stream_chunks(content="banana", reasoning="I think, therefore.", usage=(9, 12)) + ["[DONE]"]
    p = parse_stream(payloads)
    assert (p.content, p.reasoning, p.reasoning_field) == ("banana", "I think, therefore.", "reasoning_content")
    assert (p.finish_reason, p.completion_tokens, p.done_marker, p.sse) == ("stop", 12, True, True)
    assert p.n_chunks == len(payloads) - 1


def test_stream_tool_call_merged_by_index():
    args = '{"zip_code": "02134", "account_id": "007", "numeric_label": "12345"}'
    p = parse_stream(stream_chunks(tool_name="save_account", args=args, finish="tool_calls", split=3))
    assert p.tool_calls == [{"index": 0, "id": "call_0", "name": "save_account", "arguments_raw": args}]
    assert p.finish_reason == "tool_calls"


def test_stream_interleaved_tool_calls():
    def chunk(tc):
        return {"choices": [{"index": 0, "delta": {"tool_calls": [tc]}, "finish_reason": None}]}

    payloads = [
        chunk({"index": 0, "id": "a", "function": {"name": "f", "arguments": '{"x":'}}),
        chunk({"index": 1, "id": "b", "function": {"name": "g", "arguments": '{"y":'}}),
        chunk({"index": 0, "function": {"arguments": " 1}"}}),
        chunk({"index": 1, "function": {"name": "IGNORED", "arguments": " 2}"}}),
    ]
    p = parse_stream(payloads)
    assert [(t["name"], t["arguments_raw"]) for t in p.tool_calls] == [("f", '{"x": 1}'), ("g", '{"y": 2}')]
    assert p.finish_reason is None


def test_stream_without_finish_or_usage():
    p = parse_stream(stream_chunks(content="hi", finish=None, usage=None))
    assert p.finish_reason is None and p.usage is None and not p.done_marker


def test_truncate_raw_keeps_small_bodies_and_caps_big_ones():
    small = {"a": 1}
    assert truncate_raw(small) is small
    big_list = [{"choices": [{"delta": {"content": "x" * 1000}}]} for _ in range(400)]
    capped = truncate_raw(big_list)
    assert capped[-1]["_truncated"] and capped[-1]["dropped_items"] > 0
    assert len(str(capped).encode()) < RAW_LIMIT_BYTES * 1.1
    big_dict = {"text": "y" * (RAW_LIMIT_BYTES + 10)}
    assert truncate_raw(big_dict)["_truncated"]


def test_parse_runs_on_full_body_before_truncation():
    payloads = stream_chunks(content="z" * 300_000, split=1000)
    p = parse_stream(payloads)
    assert len(p.content) == 300_000 and p.raw[-1]["_truncated"]
