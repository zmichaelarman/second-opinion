"""Deterministic oracles and failure codes (CLAUDE.md §6-§7). An LLM never grades a test."""

from __future__ import annotations

import difflib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any

from jsonschema import Draft202012Validator

from .contracts.base import OracleResult, Outcome, Severity
from .parse import ParsedResponse

THINK_TAG = re.compile(r"</?think>", re.I)
TOOL_MARKUP = re.compile(r"<tool_call>|<function=|<parameter=")
SOFT_CODES = frozenset({"STREAM_USAGE_MISSING", "FINISH_REASON_WRONG", "TRUNCATED", "REFUSAL"})
ERROR_CODES = frozenset({"HTTP_5XX", "HTTP_429", "TIMEOUT", "TRANSPORT_ERROR"})
THINKING_OFF_MAX_COMPLETION_TOKENS = 64
BIG_INT = 2**53

SEVERITY: dict[str, Severity] = {
    "HTTP_4XX": Severity.S3_GAP,  # S2 when the request shape is the docs' own
    "EMPTY_CONTENT": Severity.S2_LOUD,
    "JSON_PARSE": Severity.S2_LOUD,
    "SCHEMA_INVALID": Severity.S2_LOUD,
    "REASONING_IN_CONTENT": Severity.S2_LOUD,
    "EXTRA_CONTENT": Severity.S2_LOUD,
    "THINKING_NOT_DISABLED": Severity.S3_GAP,
    "NO_REASONING_FIELD": Severity.S3_GAP,
    "NO_TOOL_CALLS": Severity.S2_LOUD,
    "TOOL_MARKUP_IN_CONTENT": Severity.S2_LOUD,
    "WRONG_TOOL": Severity.S2_LOUD,
    "ARGS_JSON_INVALID": Severity.S2_LOUD,
    "TYPE_COERCED": Severity.S1_SILENT,
    "VALUE_TRUNCATED": Severity.S1_SILENT,
    "VALUE_MISMATCH": Severity.S2_LOUD,
    "STREAM_FINISH_MISSING": Severity.S2_LOUD,
    "STREAM_USAGE_MISSING": Severity.S3_GAP,
    "FINISH_REASON_WRONG": Severity.S3_GAP,
}


def base_code(code: str) -> str:
    return code.split(":", 1)[0]


def is_error_code(code: str) -> bool:
    return base_code(code) in ERROR_CODES


def is_fatal(code: str) -> bool:
    return base_code(code) not in SOFT_CODES and not is_error_code(code)


def outcome_for(codes: list[str]) -> Outcome:
    """Any ERROR-class code gives ERROR; otherwise any fatal code gives FAIL; otherwise PASS."""
    if any(is_error_code(c) for c in codes):
        return Outcome.ERROR
    if any(is_fatal(c) for c in codes):
        return Outcome.FAIL
    return Outcome.PASS


def severity_of(code: str, docs_shape: bool = False) -> Severity | None:
    if base_code(code) == "HTTP_4XX" and docs_shape:
        return Severity.S2_LOUD
    return SEVERITY.get(base_code(code))


def normalize_answer(content: str | None) -> str:
    return (content or "").strip().lower().strip(".!\"' ")


def _reject_constant(name: str) -> Any:
    raise ValueError(f"{name} is not valid JSON")


def strict_json_loads(text: str) -> Any:
    """json.loads without Python's NaN/Infinity extensions."""
    return json.loads(text, parse_constant=_reject_constant)


class Findings:
    """Collects codes in check order, with detail per code."""

    def __init__(self) -> None:
        self.codes: list[str] = []
        self.detail: dict[str, Any] = {}

    def add(self, code: str, **detail: Any) -> None:
        if code not in self.codes:
            self.codes.append(code)
        if detail:
            self.detail.setdefault(code, {}).update(detail)

    def note(self, **detail: Any) -> None:
        self.detail.update(detail)

    def result(self) -> OracleResult:
        return OracleResult(outcome_for(self.codes), list(self.codes), self.detail)


def http_gate(resp: ParsedResponse, docs_shape: bool = False) -> OracleResult | None:
    """ERROR for transport failures, 429 and 5xx; FAIL (HTTP_4XX) for other 4xx; None for a 200."""
    if resp.error:
        code = "TIMEOUT" if resp.error == "TIMEOUT" else "TRANSPORT_ERROR"
        return OracleResult(Outcome.ERROR, [code], {code: {"error": resp.error, "status": resp.status}})
    status = resp.status
    if status == 200:
        if resp.sse and resp.error_message:
            return OracleResult(Outcome.ERROR, ["HTTP_5XX"], {"HTTP_5XX": {"in_stream_error": resp.error_message}})
        return None
    info = {"status": status, "message": resp.error_message}
    if status == 429:
        return OracleResult(Outcome.ERROR, ["HTTP_429"], {"HTTP_429": info})
    if status == 408:
        return OracleResult(Outcome.ERROR, ["TIMEOUT"], {"TIMEOUT": info})
    if status is not None and status >= 500:
        return OracleResult(Outcome.ERROR, ["HTTP_5XX"], {"HTTP_5XX": info})
    if status is not None and 400 <= status < 500:
        return OracleResult(Outcome.FAIL, ["HTTP_4XX"], {"HTTP_4XX": {**info, "docs_shape": docs_shape}})
    return OracleResult(Outcome.ERROR, ["TRANSPORT_ERROR"], {"TRANSPORT_ERROR": info})


def check_exact_answer(f: Findings, resp: ParsedResponse, expected: str) -> None:
    content = resp.content or ""
    if not content.strip():
        if resp.finish_reason == "length":
            f.add("TRUNCATED")
        else:
            f.add("EMPTY_CONTENT", finish_reason=resp.finish_reason)
        return
    if normalize_answer(content) != expected:
        f.add("EXTRA_CONTENT", expected=expected, content_head=content[:500])
        if THINK_TAG.search(content):
            f.add("REASONING_IN_CONTENT")


def check_thinking_off(f: Findings, resp: ParsedResponse) -> None:
    tokens = resp.completion_tokens
    reasoning = (resp.reasoning or "").strip()
    if reasoning or (tokens is not None and tokens > THINKING_OFF_MAX_COMPLETION_TOKENS):
        f.add("THINKING_NOT_DISABLED", reasoning_chars=len(reasoning), completion_tokens=tokens)


def check_thinking_on(f: Findings, resp: ParsedResponse) -> None:
    if not (resp.reasoning or "").strip():
        f.add("NO_REASONING_FIELD", think_tags_in_content=bool(THINK_TAG.search(resp.content or "")))


def check_no_think_tags(f: Findings, resp: ParsedResponse) -> None:
    if THINK_TAG.search(resp.content or ""):
        f.add("REASONING_IN_CONTENT", content_head=(resp.content or "")[:300])


def schema_errors(schema: dict[str, Any], instance: Any) -> list[dict[str, Any]]:
    return [
        {"validator": e.validator, "path": _json_path(e.absolute_path), "message": e.message[:300]}
        for e in Draft202012Validator(schema).iter_errors(instance)
    ]


def _json_path(parts: Any) -> str:
    return "$" + "".join(f"[{p}]" if isinstance(p, int) else f".{p}" for p in parts)


def check_json_content(
    f: Findings, resp: ParsedResponse, schema: dict[str, Any] | None = None, require_object: bool = False
) -> Any:
    """The T1/T5 content oracle. Returns the parsed object, or None."""
    content = resp.content or ""
    if resp.finish_reason == "length":
        f.add("TRUNCATED")
    if not content.strip():
        if resp.finish_reason != "length":
            f.add("EMPTY_CONTENT", finish_reason=resp.finish_reason)
        return None
    try:
        obj = strict_json_loads(content)
    except ValueError as e:
        if THINK_TAG.search(content) or not content.strip().startswith("{"):
            f.add("REASONING_IN_CONTENT", content_head=content[:300])
        else:
            f.add("JSON_PARSE", error=str(e)[:200], content_head=content[:300])
        return None
    if require_object and not isinstance(obj, dict):
        f.add("SCHEMA_INVALID:type", errors=[{"validator": "type", "path": "$", "message": "not a JSON object"}])
        return obj
    if schema is not None:
        errors = schema_errors(schema, obj)
        if errors:
            f.add(f"SCHEMA_INVALID:{errors[0]['validator']}", errors=errors[:20])
    return obj


def check_tool_call(
    f: Findings,
    resp: ParsedResponse,
    expected_name: str,
    expected_args: dict[str, Any],
    params_schema: dict[str, Any],
) -> None:
    """check_tool_call from §7, in order: tool calls present, no markup in content, right tool,
    arguments parse, arguments validate, typed equality, finish_reason (soft)."""
    content = resp.content or ""
    if resp.finish_reason == "length":
        f.add("TRUNCATED")
    if not resp.tool_calls:
        f.add("NO_TOOL_CALLS", finish_reason=resp.finish_reason, content_head=content[:300])
    if TOOL_MARKUP.search(content):
        f.add("TOOL_MARKUP_IN_CONTENT", content_head=content[:300])
    if not resp.tool_calls:
        return
    if len(resp.tool_calls) > 1:
        f.note(n_tool_calls=len(resp.tool_calls))
    call = resp.tool_calls[0]
    if call.get("name") != expected_name:
        f.add("WRONG_TOOL", expected=expected_name, got=call.get("name"))
        return
    raw = call.get("arguments_raw")
    if not isinstance(raw, str):
        f.add("ARGS_JSON_INVALID", error=f"arguments is {type(raw).__name__}, not a JSON string")
        return
    try:
        args = strict_json_loads(raw)
    except ValueError as e:
        f.add("ARGS_JSON_INVALID", error=str(e)[:200], arguments_head=raw[:300])
        return
    errors = schema_errors(params_schema, args)
    if errors:
        f.add(f"SCHEMA_INVALID:{errors[0]['validator']}", errors=errors[:20])
    mismatches = typed_equal(expected_args, args)
    for m in mismatches:
        f.add(m.code)
    if mismatches:
        f.note(mismatches=[m.as_dict() for m in mismatches])
    if resp.finish_reason != "tool_calls":
        f.add("FINISH_REASON_WRONG", finish_reason=resp.finish_reason)


def check_stream(f: Findings, resp: ParsedResponse, include_usage: bool = True) -> None:
    if not resp.sse:
        f.note(stream_requested_but_not_sse=True)
    if not resp.finish_reason:
        f.add("STREAM_FINISH_MISSING", n_chunks=resp.n_chunks, done_marker=resp.done_marker)
    if include_usage and not resp.usage:
        f.add("STREAM_USAGE_MISSING")


# typed_equal (§7) --------------------------------------------------------------------------


@dataclass
class Mismatch:
    code: str  # TYPE_COERCED | VALUE_TRUNCATED | VALUE_MISMATCH
    path: str
    expected: Any
    got: Any
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k in ("expected", "got"):
            if isinstance(d[k], str) and len(d[k]) > 500:
                d[k] = d[k][:500] + "..."
        return d


def json_type(x: Any) -> str:
    if x is None:
        return "null"
    if isinstance(x, bool):
        return "boolean"
    if isinstance(x, int):
        return "integer"
    if isinstance(x, float):
        return "number"
    if isinstance(x, str):
        return "string"
    if isinstance(x, list):
        return "array"
    if isinstance(x, dict):
        return "object"
    return type(x).__name__


def typed_equal(expected: Any, got: Any, path: str = "$") -> list[Mismatch]:
    """Every difference between expected and got, classified. Empty list = equal in value and type."""
    out: list[Mismatch] = []
    _compare(expected, got, path, out)
    return out


def _compare(exp: Any, got: Any, path: str, out: list[Mismatch]) -> None:
    et, gt = json_type(exp), json_type(got)
    if et == "object" and gt == "object":
        for k in exp:
            if k in got:
                _compare(exp[k], got[k], f"{path}.{k}", out)
            else:
                out.append(Mismatch("VALUE_MISMATCH", f"{path}.{k}", exp[k], None, "missing key"))
        for k in got:
            if k not in exp:
                out.append(Mismatch("VALUE_MISMATCH", f"{path}.{k}", None, got[k], "unexpected key"))
        return
    if et == "array" and gt == "array":
        if len(exp) != len(got):
            out.append(Mismatch("VALUE_MISMATCH", path, exp, got, f"length {len(got)}, expected {len(exp)}"))
            return
        for i, (e, g) in enumerate(zip(exp, got)):
            _compare(e, g, f"{path}[{i}]", out)
        return
    if et == gt:
        if exp == got:
            return
        if et == "integer" and abs(exp) >= BIG_INT and abs(got - exp) <= 2:
            out.append(Mismatch("TYPE_COERCED", path, exp, got, "PRECISION_LOSS"))
        elif et == "string":
            out.append(_string_mismatch(exp, got, path))
        else:
            out.append(Mismatch("VALUE_MISMATCH", path, exp, got))
        return
    note = _coercion(exp, got)
    if note:
        out.append(Mismatch("TYPE_COERCED", path, exp, got, note))
    else:
        out.append(Mismatch("VALUE_MISMATCH", path, exp, got, f"got {gt}, expected {et}"))


def _coercion(exp: Any, got: Any) -> str | None:
    """Name the coercion if got is the same value as exp in a different JSON type."""
    et, gt = json_type(exp), json_type(got)
    numeric = ("integer", "number")
    if et == "integer" and gt == "number":
        if got == exp:
            return "INT_TO_FLOAT"
        if abs(exp) >= BIG_INT and abs(got - exp) <= 2:
            return "PRECISION_LOSS"
    if et == "number" and gt == "integer" and got == exp:
        return "FLOAT_TO_INT"
    if et == "string" and gt in numeric:
        try:
            if float(exp) == got:
                return "STRING_TO_NUMBER"
        except ValueError:
            pass
    if et in numeric and gt == "string":
        try:
            value = strict_json_loads(got.strip())
            if json_type(value) in numeric and value == exp:
                return "NUMBER_TO_STRING"
        except ValueError:
            pass
    if et == "boolean" and gt == "string" and got.strip().lower() == str(exp).lower():
        return "BOOL_TO_STRING"
    if et == "boolean" and gt in numeric and got == int(exp):
        return "BOOL_TO_NUMBER"
    if et in numeric and gt == "boolean" and int(got) == exp:
        return "NUMBER_TO_BOOL"
    if et == "null" and gt == "string" and got.strip().lower() in ("null", "none", ""):
        return "NULL_TO_STRING"
    if et in ("array", "object") and gt == "string":
        try:
            if not typed_equal(exp, strict_json_loads(got)):
                return "JSON_ENCODED_STRING"
        except ValueError:
            pass
    return None


def _string_mismatch(exp: str, got: str, path: str) -> Mismatch:
    if len(got) < len(exp) and exp.startswith(got):
        rest = exp[len(got) :].lstrip()
        if rest.startswith("<"):
            return Mismatch("VALUE_TRUNCATED", path, exp, got, f"stops right before {rest[:14]!r}")
    note = "differs only in Unicode normalization; " if unicodedata.normalize("NFC", exp) == unicodedata.normalize("NFC", got) else ""
    return Mismatch("VALUE_MISMATCH", path, exp, got, note + _diff(exp, got))


def _diff(exp: str, got: str) -> str:
    lines = difflib.unified_diff(exp.splitlines(), got.splitlines(), "expected", "got", lineterm="", n=1)
    return "\n".join(lines)[:2000]
