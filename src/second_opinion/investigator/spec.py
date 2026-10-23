"""A re-renderable experiment request (CLAUDE.md §11).

A Spec is the minimal failing request kept in structured form (prompt, tool, expected
arguments, schemas), so it can be shrunk, changed by one factor from the menu, rendered back
into a request body, and graded with the same oracle as its contract.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from typing import Any

from ..contracts import t1_reasoning_leak, t5_schemas
from ..contracts.base import BANANA_PROMPT, Condition, OracleResult, chat_request
from ..fixtures.payloads import EXPECTED_ARGS, forced_tool_choice, load_tool, tool_name, tool_prompt
from ..fixtures.schemas import load_schema
from ..oracles import (
    Findings,
    check_exact_answer,
    check_json_content,
    check_no_think_tags,
    check_stream,
    check_thinking_off,
    check_thinking_on,
    check_tool_call,
    http_gate,
)
from ..parse import ParsedResponse

FACTORS: dict[str, str] = {
    "thinking": "true | false: reasoning on or off",
    "stream": "true | false: streamed (SSE) or not",
    "max_tokens": "an integer token limit",
    "shape": "wrapped | bare: the json_schema request shape (structured-output contracts only)",
    "append_literal": "a string appended to the markup argument (tool-argument contracts only)",
    "remove_schema_keyword": "a JSON Schema keyword deleted from the request schema; grading still uses the original schema",
    "inline_refs": "true | false: inline $ref/$defs in the request schema",
    "arg_type": "'<argument>:string' or '<argument>:integer': change one argument's declared type in the tool schema",
    "temperature": "0 | 1.0",
}


class FactorError(ValueError):
    pass


def parse_bool(level: Any) -> bool:
    if isinstance(level, bool):
        return level
    if str(level).strip().lower() in ("true", "1", "yes", "on"):
        return True
    if str(level).strip().lower() in ("false", "0", "no", "off"):
        return False
    raise FactorError(f"expected true/false, got {level!r}")


def inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    defs = schema.get("$defs", {})

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node and node["$ref"].startswith("#/$defs/"):
                return walk(copy.deepcopy(defs[node["$ref"].split("/")[-1]]))
            return {k: walk(v) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    return walk(schema)


def remove_keyword(schema: Any, keyword: str) -> Any:
    if isinstance(schema, dict):
        return {k: remove_keyword(v, keyword) for k, v in schema.items() if k != keyword}
    if isinstance(schema, list):
        return [remove_keyword(v, keyword) for v in schema]
    return schema


@dataclass
class Spec:
    cluster_id: str
    contract_id: str
    model: str
    signature: list[str]
    thinking: bool | None
    stream: bool = False
    max_tokens: int | None = None
    temperature: float = 1.0
    prompt: str | None = None  # user prompt; T3 prompts are generated from expected_args
    tool: dict[str, Any] | None = None
    expected_args: dict[str, Any] | None = None
    schema: dict[str, Any] | None = None  # sent in the request
    oracle_schema: dict[str, Any] | None = None  # graded against
    shape: str | None = None
    schema_name: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def kind(self) -> str:
        if self.tool is not None:
            return "tool"
        if self.shape is not None:
            return "json"
        return "answer"

    @property
    def condition(self) -> Condition:
        return Condition(thinking=self.thinking, stream=self.stream, max_tokens=self.max_tokens,
                         variant=self.schema_name or "default", shape=self.shape or "wrapped")

    def request(self) -> dict[str, Any]:
        cond = Condition(thinking=self.thinking, stream=self.stream, max_tokens=self.max_tokens)
        if self.kind == "tool":
            assert self.tool is not None and self.expected_args is not None
            name = tool_name(self.tool)
            body = chat_request(self.model, tool_prompt(name, self.expected_args), cond)
            body["tools"] = [self.tool]
            body["tool_choice"] = forced_tool_choice(name)
        elif self.kind == "json":
            body = chat_request(self.model, self.prompt or "", cond)
            if self.shape == "json_object":
                body["response_format"] = {"type": "json_object"}
            elif self.shape == "bare":
                body["response_format"] = {"type": "json_schema", "json_schema": self.schema}
            else:
                body["response_format"] = {"type": "json_schema", "json_schema": {
                    "name": self.schema_name or "schema", "schema": self.schema, "strict": True}}
        else:
            body = chat_request(self.model, self.prompt or BANANA_PROMPT, cond)
        body["temperature"] = self.temperature
        return body

    def check(self, req: dict[str, Any], resp: ParsedResponse) -> OracleResult:
        gate = http_gate(resp, docs_shape=self.kind == "json")
        if gate:
            return gate
        f = Findings()
        if self.kind == "tool":
            assert self.tool is not None and self.expected_args is not None
            check_tool_call(f, resp, tool_name(self.tool), self.expected_args, self.tool["function"]["parameters"])
            check_no_think_tags(f, resp)
        elif self.kind == "json":
            if self.shape == "json_object":
                check_json_content(f, resp, None, require_object=True)
            else:
                check_json_content(f, resp, self.oracle_schema)
        else:
            check_exact_answer(f, resp, "banana")
            if self.contract_id == "T2.off":
                check_thinking_off(f, resp)
            elif self.contract_id == "T2.on":
                check_thinking_on(f, resp)
        if req.get("stream"):
            check_stream(f, resp)
        return f.result()

    def reproduces(self, result: OracleResult) -> bool:
        """A trial counts as this cluster's failure if it shows any of the signature's codes."""
        return bool(set(result.codes) & set(self.signature))

    def with_factor(self, factor: str, level: Any) -> "Spec":
        if factor not in FACTORS:
            raise FactorError(f"{factor!r} is not on the factor menu")
        s = copy.deepcopy(self)
        if factor == "thinking":
            s.thinking = parse_bool(level)
        elif factor == "stream":
            s.stream = parse_bool(level)
        elif factor == "max_tokens":
            s.max_tokens = int(level)
            if s.max_tokens < 1:
                raise FactorError("max_tokens must be positive")
        elif factor == "temperature":
            s.temperature = float(level)
            if not 0 <= s.temperature <= 2:
                raise FactorError("temperature must be between 0 and 2")
        elif factor == "shape":
            if s.kind != "json" or level not in ("wrapped", "bare"):
                raise FactorError("shape applies to json_schema requests, with level wrapped or bare")
            s.shape = level
        elif factor == "append_literal":
            if s.kind != "tool" or not s.expected_args:
                raise FactorError("append_literal applies to tool-argument contracts")
            key = max((k for k, v in s.expected_args.items() if isinstance(v, str)),
                      key=lambda k: len(s.expected_args[k]), default=None)  # type: ignore[index]
            if key is None:
                raise FactorError("no string argument to append to")
            s.expected_args[key] = s.expected_args[key] + str(level)
        elif factor == "remove_schema_keyword":
            if s.kind != "json" or s.schema is None:
                raise FactorError("remove_schema_keyword applies to structured-output contracts")
            s.schema = remove_keyword(s.schema, str(level))
        elif factor == "inline_refs":
            if s.kind != "json" or s.schema is None:
                raise FactorError("inline_refs applies to structured-output contracts")
            if parse_bool(level):
                s.schema = inline_refs(s.schema)
        elif factor == "arg_type":
            arg, _, typ = str(level).partition(":")
            props = ((s.tool or {}).get("function", {}).get("parameters", {}).get("properties", {}))
            if s.kind != "tool" or arg not in props or typ not in ("string", "integer", "number", "boolean"):
                raise FactorError("arg_type expects '<argument>:<type>' for an argument of the tool")
            props[arg]["type"] = typ
        return s

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "Spec":
        return cls(**d)


def spec_from_cluster(cluster_id: str, contract_id: str, model: str, condition: dict[str, Any],
                      signature: list[str]) -> Spec:
    cond = Condition.from_dict(condition)
    base = dict(cluster_id=cluster_id, contract_id=contract_id, model=model, signature=list(signature),
                thinking=cond.thinking, stream=cond.stream, max_tokens=cond.max_tokens)
    tier, _, variant = contract_id.partition(".")
    if tier == "T3":
        return Spec(**base, tool=load_tool(variant), expected_args=copy.deepcopy(EXPECTED_ARGS[variant]))
    if tier == "T5":
        schema = load_schema(variant)
        return Spec(**base, prompt=t5_schemas.PROMPTS[variant], schema=schema, oracle_schema=copy.deepcopy(schema),
                    shape=cond.shape, schema_name=variant)
    if contract_id == "T1.json_truncation":
        schema = load_schema("flat_basic")
        return Spec(**base, prompt=t1_reasoning_leak.ALIEN_PROMPT, schema=schema, oracle_schema=copy.deepcopy(schema),
                    shape="wrapped", schema_name="flat_basic")
    return Spec(**base, prompt=BANANA_PROMPT)


def spec_digest(spec: Spec) -> str:
    return json.dumps(spec.request(), sort_keys=True, ensure_ascii=False)
