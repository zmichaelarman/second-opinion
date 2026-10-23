from __future__ import annotations

import ast
import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from second_opinion.docs_check.extract import CodeBlock

# Fallback list of valid chat/completions request parameters from Token Factory OpenAPI spec (§10.3)
FALLBACK_CHAT_COMPLETION_PARAMS = {
    "messages",
    "model",
    "frequency_penalty",
    "function_call",
    "functions",
    "logit_bias",
    "logprobs",
    "top_logprobs",
    "max_tokens",
    "max_completion_tokens",
    "n",
    "presence_penalty",
    "response_format",
    "seed",
    "stop",
    "stream",
    "stream_options",
    "temperature",
    "top_p",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "user",
    "extra_body",
    "chat_template_kwargs",
    " guided_json",
    "guided_json",
    "guided_regex",
    "guided_choice",
    "guided_grammar",
    "top_k",
    "min_p",
    "repetition_penalty",
    "reasoning_effort",
}

MODEL_ID_REGEX = re.compile(r"""["']?model["']?\s*[=:]\s*["']([^"'$\n{}]+)["']""")


@dataclass
class StaticCheckResult:
    codes: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    referenced_models: list[str] = field(default_factory=list)


def extract_curl_data_bodies(bash_code: str) -> list[str]:
    """Extract single-quoted body after --data-binary, --data, or -d in cURL commands (§10.3)."""
    bodies: list[str] = []
    pattern = re.compile(r"(?:--data-binary|--data|-d)\s+'", re.MULTILINE)
    pos = 0
    while True:
        m = pattern.search(bash_code, pos)
        if not m:
            break
        start = m.end()
        # Scan until closing single quote that isn't part of a shell variable splice like '"$VAR"' or '\''
        i = start
        n = len(bash_code)
        chars: list[str] = []
        while i < n:
            # Check for shell splice '"$VAR"' or '"${VAR}"'
            splice_m = re.match(r"""'"\$(\{[A-Za-z0-9_]+\}|[A-Za-z0-9_]+)"'""", bash_code[i:])
            if splice_m:
                chars.append("{}")
                i += splice_m.end()
                continue
            if bash_code[i] == "'":
                break
            chars.append(bash_code[i])
            i += 1
        bodies.append("".join(chars))
        pos = i + 1
    return bodies


def _tolerantly_extract_top_level_keys(raw_json_body: str) -> list[str]:
    """Extract top-level JSON object keys even when a comma is missing between properties."""
    # First try inserting missing commas after closing brace/bracket/quote/primitive before a new key line
    repaired = re.sub(
        r'([}\]"0-9eE]|true|false|null)\s*\n(\s*"[^"]+"\s*:)',
        r"\1,\n\2",
        raw_json_body,
    )
    try:
        parsed = json.loads(repaired)
        if isinstance(parsed, dict):
            return list(parsed.keys())
    except Exception:
        pass

    # Fallback depth-1 brace scanner for top-level keys
    keys: list[str] = []
    depth = 0
    in_str = False
    escape = False
    str_buf: list[str] = []
    i = 0
    n = len(raw_json_body)
    while i < n:
        ch = raw_json_body[i]
        if in_str:
            if escape:
                str_buf.append(ch)
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
                candidate = "".join(str_buf)
                # Look ahead for ':' at depth == 1
                j = i + 1
                while j < n and raw_json_body[j] in " \t\r\n":
                    j += 1
                if depth == 1 and j < n and raw_json_body[j] == ":":
                    if candidate not in keys:
                        keys.append(candidate)
            else:
                str_buf.append(ch)
        else:
            if ch == '"':
                in_str = True
                str_buf = []
            elif ch in "{[":
                depth += 1
            elif ch in "}]":
                depth = max(0, depth - 1)
        i += 1
    return keys


def check_code_block_static(
    block: CodeBlock,
    known_models: set[str] | None = None,
    allowed_chat_params: set[str] | None = None,
) -> StaticCheckResult:
    res = StaticCheckResult()
    params_allow = allowed_chat_params or FALLBACK_CHAT_COMPLETION_PARAMS

    # 1. Language-specific syntax/schema checks
    if block.lang == "python":
        try:
            ast.parse(block.code)
        except SyntaxError as exc:
            res.codes.append("PY_SYNTAX_ERROR")
            res.errors.append(f"SyntaxError at line {exc.lineno}: {exc.msg}")

    elif block.lang == "bash":
        if "curl" in block.code:
            bodies = extract_curl_data_bodies(block.code)
            is_chat_endpoint = "/chat/completions" in block.code
            for body in bodies:
                cleaned = re.sub(r"""'"\$(\{[A-Za-z0-9_]+\}|[A-Za-z0-9_]+)"'""", "{}", body)
                parsed_obj: Any = None
                try:
                    parsed_obj = json.loads(cleaned)
                except json.JSONDecodeError as exc:
                    if "CURL_JSON_INVALID" not in res.codes:
                        res.codes.append("CURL_JSON_INVALID")
                    res.errors.append(f"Invalid JSON in cURL data body: {exc}")

                if is_chat_endpoint:
                    top_keys = (
                        list(parsed_obj.keys())
                        if isinstance(parsed_obj, dict)
                        else _tolerantly_extract_top_level_keys(cleaned)
                    )
                    for k in top_keys:
                        if k not in params_allow:
                            code = f"UNKNOWN_PARAM:{k}"
                            if code not in res.codes:
                                res.codes.append(code)
                            res.errors.append(f"Unknown chat completion request parameter: {k!r}")

    elif block.lang == "js":
        node_bin = shutil.which("node")
        if node_bin:
            with tempfile.TemporaryDirectory() as tmpdir:
                mjs_path = Path(tmpdir) / "snippet.mjs"
                mjs_path.write_text(block.code, encoding="utf-8")
                try:
                    proc = subprocess.run(
                        [node_bin, "--check", str(mjs_path)],
                        capture_output=True,
                        text=True,
                        timeout=15,
                    )
                    if proc.returncode != 0:
                        res.codes.append("JS_SYNTAX_ERROR")
                        res.errors.append((proc.stderr or proc.stdout).strip())
                except Exception as exc:
                    res.errors.append(f"node --check failed: {exc}")

    # 2. Model ID check across all languages (§10.3)
    for m in MODEL_ID_REGEX.finditer(block.code):
        mid = m.group(1).strip()
        if not mid or mid.startswith("<") or mid in ("...", "string"):
            continue
        res.referenced_models.append(mid)
        if known_models is not None and len(known_models) > 0:
            if mid not in known_models:
                if "MODEL_NOT_LISTED" not in res.codes:
                    res.codes.append("MODEL_NOT_LISTED")
                res.errors.append(f"Model ID {mid!r} is not in live /v1/models list.")

    return res
