"""Free static checks on docs code blocks (CLAUDE.md §10.3).

- Python: ast.parse -> PY_SYNTAX_ERROR
- cURL: the single-quoted --data-binary/--data/-d body, with shell splices like '"$VAR"'
  replaced by {}, must parse as JSON (CURL_JSON_INVALID); its top-level keys must be request
  parameters of the endpoint it calls, per openapi.json (UNKNOWN_PARAM:<key>). Keys are read
  with a tolerant scanner, so a misspelled key is reported even when the JSON is also invalid.
- JS: node --check, when node is on PATH (JS_SYNTAX_ERROR)
- cURL auth: an Authorization header without the Bearer scheme (CURL_AUTH_HEADER)
- Model IDs: IDs missing from the live model list (MODEL_NOT_LISTED), for blocks that target
  Token Factory (a migration page's "before" example legitimately names other providers' models)
"""

from __future__ import annotations

import ast
import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .extract import CodeBlock

# Fallback when openapi.json is unavailable: Token Factory's chat-completions fields (2026-10-23).
CHAT_PARAMS_FALLBACK = frozenset({
    "extra_body", "frequency_penalty", "logit_bias", "logprobs", "max_completion_tokens", "max_tokens", "messages",
    "model", "n", "presence_penalty", "reasoning_effort", "response_format", "service_tier", "stop", "store",
    "stream", "stream_options", "temperature", "tool_choice", "tools", "top_logprobs", "top_p", "user",
})
_SPLICE = re.compile(r"""'"\$\{?\w+\}?"'""")
_BODY = re.compile(r"""(?:--data-binary|--data-raw|--data|-d)\s+'((?:[^']|'"\$\{?\w+\}?"')*)'""", re.S)
_CURL_URL = re.compile(r"""https?://[^\s'"\\]+""")
_MODEL = re.compile(r"""["']?\bmodel["']?\s*[=:]\s*["']([^"']+)["']""")
_AUTH = re.compile(r"""Authorization:\s*([^'"\n]*)""", re.I)
_PLACEHOLDER = re.compile(r"[<>{}]|\.\.\.|YOUR", re.I)


@dataclass
class StaticResult:
    kind: str  # python | curl | js | shell | json | other
    codes: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    models: list[str] = field(default_factory=list)
    endpoint: str | None = None
    body_keys: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def add(self, code: str, error: str) -> None:
        if code not in self.codes:
            self.codes.append(code)
        self.errors.append(f"{code}: {error}")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def request_params(openapi: dict[str, Any] | None) -> dict[str, frozenset[str]]:
    """Top-level JSON request fields for each POST path in the OpenAPI spec."""
    out: dict[str, frozenset[str]] = {"/v1/chat/completions": CHAT_PARAMS_FALLBACK}
    if not openapi:
        return out

    def resolve(node: Any) -> Any:
        while isinstance(node, dict) and "$ref" in node:
            target: Any = openapi
            for part in node["$ref"].lstrip("#/").split("/"):
                target = target[part]
            node = target
        return node

    for path, ops in (openapi.get("paths") or {}).items():
        post = (ops or {}).get("post") or {}
        schema = resolve(((post.get("requestBody") or {}).get("content") or {}).get("application/json", {}).get("schema"))
        props: set[str] = set()
        for part in [schema, *((schema or {}).get("allOf") or [])]:
            props |= set((resolve(part) or {}).get("properties", {}))
        if props:
            out[path] = frozenset(props)
    return out


def check_python(code: str, result: StaticResult) -> None:
    try:
        ast.parse(code)
    except SyntaxError as e:
        result.add("PY_SYNTAX_ERROR", f"{e.msg} (line {e.lineno})")


def curl_body(code: str) -> str | None:
    m = _BODY.search(code)
    return _SPLICE.sub("{}", m.group(1)) if m else None


def top_level_keys(text: str) -> list[str]:
    """String keys directly inside the outermost JSON object, tolerating missing commas."""
    keys: list[str] = []
    depth = 0
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch == '"':
            j, buf = i + 1, []
            while j < n and text[j] != '"':
                if text[j] == "\\" and j + 1 < n:
                    buf.append(text[j:j + 2])
                    j += 2
                    continue
                buf.append(text[j])
                j += 1
            k = j + 1
            while k < n and text[k] in " \t\r\n":
                k += 1
            if depth == 1 and k < n and text[k] == ":":
                keys.append("".join(buf))
            i = j + 1
            continue
        if ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
        i += 1
    return keys


def check_curl(code: str, result: StaticResult, params: dict[str, frozenset[str]]) -> None:
    url = _CURL_URL.search(code)
    result.endpoint = urlparse(url.group(0)).path.rstrip("/") if url else None
    auth = _AUTH.search(code)
    if auth and not auth.group(1).strip().lower().startswith("bearer "):
        result.add("CURL_AUTH_HEADER", f"Authorization header is {auth.group(1).strip()!r}, without the Bearer scheme")
    body = curl_body(code)
    if body is None:
        if re.search(r"(?:--data-binary|--data-raw|--data|-d)\s", code):
            result.notes.append("request body is not a single-quoted string; JSON not checked")
        return
    try:
        json.loads(body)
    except ValueError as e:
        result.add("CURL_JSON_INVALID", str(e))
    result.body_keys = top_level_keys(body)
    allowed = params.get(result.endpoint or "")
    if allowed is None:
        result.notes.append(f"no parameter list for endpoint {result.endpoint}; keys not checked")
        return
    for key in result.body_keys:
        if key not in allowed:
            result.add(f"UNKNOWN_PARAM:{key}", f"{key!r} is not a request field of {result.endpoint}")


def check_js(code: str, result: StaticResult, node: str | None) -> None:
    if not node:
        result.notes.append("node not on PATH; JS syntax not checked")
        return
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "snippet.mjs"
        path.write_text(code)
        proc = subprocess.run([node, "--check", str(path)], capture_output=True, text=True, timeout=30)
    if proc.returncode != 0:
        lines = [ln for ln in proc.stderr.splitlines() if ln.strip()]
        result.add("JS_SYNTAX_ERROR", next((ln for ln in lines if "Error" in ln), lines[-1] if lines else "node --check failed"))


def model_ids(code: str) -> list[str]:
    return [m for m in dict.fromkeys(_MODEL.findall(code)) if not _PLACEHOLDER.search(m)]


def targets_token_factory(code: str) -> bool:
    return "tokenfactory.nebius.com" in code or "NEBIUS_API_KEY" in code


def classify(block: CodeBlock) -> str:
    if block.lang == "python":
        return "python"
    if block.lang == "js":
        return "js"
    if block.lang in ("bash", "") and re.search(r"(^|\s|\()curl\s", block.code):
        return "curl"
    if block.lang == "bash":
        return "shell"
    if block.lang == "json":
        return "json"
    return "other"


def check_block(
    block: CodeBlock,
    params: dict[str, frozenset[str]],
    listed_models: set[str] | None,
    node: str | None = None,
) -> StaticResult:
    result = StaticResult(kind=classify(block))
    if result.kind == "python":
        check_python(block.code, result)
    elif result.kind == "curl":
        check_curl(block.code, result, params)
    elif result.kind == "js":
        check_js(block.code, result, node)
    if result.kind == "json" or (result.kind in ("python", "curl", "js") and targets_token_factory(block.code)):
        result.models = model_ids(block.code)
        if listed_models is None:
            if result.models:
                result.notes.append("live model list unavailable; model IDs not checked")
        else:
            for model in result.models:
                if model not in listed_models:
                    result.add("MODEL_NOT_LISTED", f"{model} is not listed by GET /v1/models")
    return result


def find_node() -> str | None:
    return shutil.which("node")
