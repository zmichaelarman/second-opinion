import json

import httpx
import pytest

from second_opinion.client import ApiClient
from second_opinion.config import ConfigError

from conftest import FAKE_KEY
from fake_tf import FakeTokenFactory, sse_bytes, stream_chunks


def client_for(handler, settings):
    return ApiClient.for_provider(settings, "tokenfactory", httpx.MockTransport(handler))


async def test_json_exchange_keeps_only_rate_limit_and_request_id_headers(settings):
    fake = FakeTokenFactory()
    async with ApiClient.for_provider(settings, "tokenfactory", fake.transport()) as c:
        ex = await c.chat({"model": "m", "messages": [{"role": "user", "content": "x"}]})
    assert ex.status == 200 and ex.body["choices"][0]["message"]["content"] == "Banana."
    assert set(ex.headers) == {"x-ratelimit-limit-requests", "x-ratelimit-remaining-requests",
                               "x-ratelimit-remaining-tokens", "x-request-id"}
    sent = fake.requests[0]
    assert sent.headers["authorization"] == f"Bearer {FAKE_KEY}"
    assert str(sent.url) == "https://api.tokenfactory.nebius.com/v1/chat/completions"
    assert FAKE_KEY not in json.dumps(ex.headers) + json.dumps(ex.body)


async def test_sse_stream_is_split_into_payloads(settings):
    chunks = stream_chunks(content="banana", reasoning="r")

    def handler(request):
        return httpx.Response(200, content=sse_bytes(chunks), headers={"content-type": "text/event-stream"})

    async with client_for(handler, settings) as c:
        ex = await c.chat({"model": "m", "messages": [], "stream": True})
    assert ex.stream and ex.body[-1] == "[DONE]" and ex.body[:-1] == chunks
    assert ex.ttft_s is not None


async def test_sse_edge_cases_crlf_comments_multiline_and_no_final_blank_line(settings):
    raw = (b": keep-alive\r\n\r\n"
           b"event: message\r\ndata: {\"choices\": [{\"index\": 0,\r\ndata:  \"delta\": {\"content\": \"a\"}}]}\r\n\r\n"
           b"data: {\"choices\": [{\"index\": 0, \"delta\": {\"content\": \"b\"}, \"finish_reason\": \"stop\"}]}\r\n\r\n"
           b"data: [DONE]")

    def handler(request):
        return httpx.Response(200, content=raw, headers={"content-type": "text/event-stream; charset=utf-8"})

    async with client_for(handler, settings) as c:
        ex = await c.chat({"model": "m", "messages": [], "stream": True})
    assert ex.body == [{"choices": [{"index": 0, "delta": {"content": "a"}}]},
                       {"choices": [{"index": 0, "delta": {"content": "b"}, "finish_reason": "stop"}]},
                       "[DONE]"]


async def test_stream_requested_but_json_returned(settings):
    def handler(request):
        return httpx.Response(200, json={"choices": []})

    async with client_for(handler, settings) as c:
        ex = await c.chat({"model": "m", "messages": [], "stream": True})
    assert ex.stream and ex.body == {"choices": []}


async def test_error_status_body_and_retry_after(settings):
    def handler(request):
        return httpx.Response(429, json={"error": {"message": "slow down"}}, headers={"retry-after": "7"})

    async with client_for(handler, settings) as c:
        ex = await c.chat({"model": "m", "messages": [], "stream": True})
    assert (ex.status, ex.headers["retry-after"], ex.body["error"]["message"]) == (429, "7", "slow down")


async def test_timeout_and_connect_errors(settings):
    def timeout(request):
        raise httpx.ReadTimeout("slow", request=request)

    def refused(request):
        raise httpx.ConnectError("refused", request=request)

    async with client_for(timeout, settings) as c:
        assert (await c.chat({"model": "m"})).error == "TIMEOUT"
    async with client_for(refused, settings) as c:
        assert (await c.get("models")).error == "ConnectError"


async def test_stream_cut_short_keeps_partial_payloads(settings):
    class Broken(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"choices": [{"index": 0, "delta": {"content": "par"}}]}\n\n'
            raise httpx.ReadError("connection reset")

    def handler(request):
        return httpx.Response(200, stream=Broken(), headers={"content-type": "text/event-stream"})

    async with client_for(handler, settings) as c:
        ex = await c.chat({"model": "m", "stream": True})
    assert ex.error == "ReadError" and ex.body[0]["choices"][0]["delta"]["content"] == "par"


async def test_missing_key_raises_config_error(settings, monkeypatch):
    monkeypatch.delenv("NEBIUS_API_KEY")
    async with ApiClient.for_provider(settings, "tokenfactory", FakeTokenFactory().transport()) as c:
        with pytest.raises(ConfigError):
            await c.get("models")
