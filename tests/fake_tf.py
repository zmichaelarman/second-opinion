"""A fake Token Factory for offline tests: just enough of /v1/models and /v1/chat/completions.

By default it behaves like a correct OpenAI-compatible server. Flags switch on specific bugs.
"""

from __future__ import annotations

import json
import re

import httpx

from conftest import FAKE_KEY

MODEL_IDS = [
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
    "nvidia/Nemotron-3_5-Lightning",
    "nvidia/nemotron-3-super-120b-a12b",
    "nvidia/Nemotron-3-Ultra-550b-a55b",
    "nvidia/Nemotron-Nano-VL-12B-V2",
    "meta-llama/Llama-3.3-70B-Instruct",
]
REASONING = "The user wants a single word. I will answer exactly."
BASE = {"id": "cmpl-fake", "created": 0, "model": "fake"}


def completion(content=None, reasoning=None, tool_calls=None, finish="stop", usage=(50, 5),
               reasoning_field="reasoning_content"):
    message = {"role": "assistant", "content": content}
    if reasoning is not None:
        message[reasoning_field] = reasoning
    if tool_calls:
        message["tool_calls"] = [
            {"id": f"call_{i}", "type": "function", "function": {"name": name, "arguments": args}}
            for i, (name, args) in enumerate(tool_calls)
        ]
    return {**BASE, "object": "chat.completion", "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1], "total_tokens": sum(usage)}}


def stream_chunks(content="", reasoning="", tool_name=None, args="", finish="stop", usage=(50, 20), split=7,
                  reasoning_field="reasoning_content"):
    def chunk(delta, finish_reason=None):
        return {**BASE, "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}

    chunks = [chunk({"role": "assistant", "content": ""})]
    chunks += [chunk({reasoning_field: reasoning[i:i + split]}) for i in range(0, len(reasoning), split)]
    chunks += [chunk({"content": content[i:i + split]}) for i in range(0, len(content), split)]
    if tool_name:
        chunks.append(chunk({"tool_calls": [{"index": 0, "id": "call_0", "type": "function",
                                             "function": {"name": tool_name, "arguments": ""}}]}))
        chunks += [chunk({"tool_calls": [{"index": 0, "function": {"arguments": args[i:i + split]}}]})
                   for i in range(0, len(args), split)]
    if finish:
        chunks.append(chunk({}, finish))
    if usage:
        chunks.append({**BASE, "object": "chat.completion.chunk", "choices": [],
                       "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1], "total_tokens": sum(usage)}})
    return chunks


def sse_bytes(chunks, done=True) -> bytes:
    out = "".join(f"data: {json.dumps(c, ensure_ascii=False)}\n\n" for c in chunks)
    return (out + ("data: [DONE]\n\n" if done else "")).encode()


def json_for_schema(schema, root=None):
    """A minimal instance of a JSON schema (enough for the T5 fixtures)."""
    root = root or schema
    if "$ref" in schema:
        node = root
        for part in schema["$ref"].lstrip("#/").split("/"):
            node = node[part]
        return json_for_schema(node, root)
    if "anyOf" in schema:
        return json_for_schema(next(s for s in schema["anyOf"] if s.get("type") != "null"), root)
    if "enum" in schema:
        return schema["enum"][0]
    t = schema.get("type")
    if t == "object":
        return {k: json_for_schema(v, root) for k, v in schema.get("properties", {}).items()}
    if t == "array":
        return [json_for_schema(schema.get("items", {"type": "string"}), root) for _ in range(max(2, schema.get("minItems", 0)))]
    if t == "integer":
        return 7
    if t == "number":
        return max(schema.get("minimum", 7.5), min(7.5, schema.get("maximum", 7.5)))
    if t == "string":
        return "ABC-1234" if schema.get("pattern") else "Alien"
    return None


class FakeTokenFactory:
    def __init__(self, model_ids=MODEL_IDS, honor_chat_template_kwargs=True, honor_extra_body=False,
                 honor_reasoning_effort=True, coerce_digit_strings=False, rate_limit_headers=None, banana_usage=None,
                 violate_every=0, coerce_only_when_streaming=False, llm_responder=None):
        self.model_ids = model_ids
        self.honor_chat_template_kwargs = honor_chat_template_kwargs
        self.honor_extra_body = honor_extra_body
        self.honor_reasoning_effort = honor_reasoning_effort
        self.coerce_digit_strings = coerce_digit_strings
        self.banana_usage = banana_usage
        self.violate_every = violate_every  # every Nth schema answer breaks an enum (a realistic soft spot)
        self._schema_answers = 0
        self.coerce_only_when_streaming = coerce_only_when_streaming
        self.llm_responder = llm_responder  # body -> content or None: plays Ultra/Super for Investigator tests
        self.rate_limit_headers = rate_limit_headers or {
            "x-ratelimit-limit-requests": "100000", "x-ratelimit-remaining-requests": "99999",
            "x-ratelimit-remaining-tokens": "399000", "x-request-id": "req-fake",
        }
        self.requests: list[httpx.Request] = []

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.headers.get("authorization") != f"Bearer {FAKE_KEY}":
            return httpx.Response(401, json={"error": {"message": "bad key"}})
        headers = {**self.rate_limit_headers, "set-cookie": "session=secret"}
        if request.method == "GET" and request.url.path.endswith("/models"):
            data = [{"id": m, "object": "model", "owned_by": m.split("/")[0]} for m in self.model_ids]
            return httpx.Response(200, json={"object": "list", "data": data}, headers=headers)
        body = json.loads(request.content)
        if self.llm_responder:
            content = self.llm_responder(body)
            if content is not None:
                return httpx.Response(200, json=completion(content, "thinking", usage=(900, 400)), headers=headers)
        thinking = self._thinking(body)
        reasoning = REASONING if thinking else None
        if body.get("tools"):
            name = body["tools"][0]["function"]["name"]
            args = self._args_from_prompt(body, bool(body.get("stream")))
            if body.get("stream"):
                chunks = stream_chunks(reasoning=reasoning or "", tool_name=name, args=args, finish="tool_calls")
                return httpx.Response(200, content=sse_bytes(chunks), headers={**headers, "content-type": "text/event-stream"})
            return httpx.Response(200, json=completion(None, reasoning, [(name, args)], "tool_calls", (120, 40)),
                                  headers=headers)
        content = "Banana."
        usage = self.banana_usage or (50, 30 if thinking else 2)
        fmt = body.get("response_format") or {}
        if fmt.get("type") == "json_schema":
            spec = fmt["json_schema"]
            schema = spec.get("schema", spec) if "name" in spec else spec
            obj = json_for_schema(schema)
            self._schema_answers += 1
            if self.violate_every and self._schema_answers % self.violate_every == 0 and "genre" in obj:
                obj["genre"] = "musical"
            content, usage = json.dumps(obj), (120, 40 if not thinking else 300)
        elif fmt.get("type") == "json_object":
            content, usage = json.dumps({"title": "Chinatown", "year": 1974}), (120, 30)
        if body.get("stream"):
            chunks = stream_chunks(content=content, reasoning=reasoning or "", usage=usage)
            return httpx.Response(200, content=sse_bytes(chunks), headers={**headers, "content-type": "text/event-stream"})
        return httpx.Response(200, json=completion(content, reasoning, usage=usage), headers=headers)

    def _thinking(self, body) -> bool:
        if self.honor_chat_template_kwargs and "chat_template_kwargs" in body:
            return bool(body["chat_template_kwargs"].get("enable_thinking", True))
        if self.honor_extra_body and "extra_body" in body:
            return bool(body["extra_body"].get("chat_template_kwargs", {}).get("enable_thinking", True))
        if self.honor_reasoning_effort and body.get("reasoning_effort") == "none":
            return False
        return True

    def _args_from_prompt(self, body, streaming=False) -> str:
        prompt = body["messages"][-1]["content"]
        args = json.loads(re.search(r"```json\n(.*)\n```", prompt, re.S).group(1))
        if self.coerce_digit_strings and (streaming or not self.coerce_only_when_streaming):
            args = {k: int(v) if isinstance(v, str) and v.isdigit() else v for k, v in args.items()}
        return json.dumps(args, ensure_ascii=False)
