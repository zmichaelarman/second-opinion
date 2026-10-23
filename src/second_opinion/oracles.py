from __future__ import annotations

import difflib
import json
from dataclasses import dataclass, field
from typing import Any

from jsonschema import Draft202012Validator

from second_opinion.contracts.base import OracleResult, Outcome, Severity
from second_opinion.parse import ParsedResponse


FATAL_CODES = {
    "HTTP_4XX",
    "EMPTY_CONTENT",
    "JSON_PARSE",
    "REASONING_IN_CONTENT",
    "EXTRA_CONTENT",
    "THINKING_NOT_DISABLED",
    "NO_REASONING_FIELD",
    "NO_TOOL_CALLS",
    "TOOL_MARKUP_IN_CONTENT",
    "WRONG_TOOL",
    "ARGS_JSON_INVALID",
    "TYPE_COERCED",
    "VALUE_TRUNCATED",
    "VALUE_MISMATCH",
    "STREAM_FINISH_MISSING",
}

SOFT_CODES = {
    "STREAM_USAGE_MISSING",
    "FINISH_REASON_WRONG",
    "TRUNCATED",
    "REFUSAL",
}

ERROR_CODES = {
    "HTTP_5XX",
    "TIMEOUT",
}


def is_fatal_code(code: str) -> bool:
    if code.startswith("SCHEMA_INVALID:"):
        return True
    return code in FATAL_CODES


def code_severity(code: str, docs_shape: bool = False) -> Severity | None:
    if code in ("TYPE_COERCED", "VALUE_TRUNCATED"):
        return Severity.S1_SILENT
    if code == "HTTP_4XX":
        return Severity.S2_LOUD if docs_shape else Severity.S3_GAP
    if code in ("THINKING_NOT_DISABLED", "NO_REASONING_FIELD", "STREAM_USAGE_MISSING", "FINISH_REASON_WRONG"):
        return Severity.S3_GAP
    if code.startswith("SCHEMA_INVALID:") or code in (
        "EMPTY_CONTENT",
        "JSON_PARSE",
        "REASONING_IN_CONTENT",
        "EXTRA_CONTENT",
        "NO_TOOL_CALLS",
        "TOOL_MARKUP_IN_CONTENT",
        "WRONG_TOOL",
        "ARGS_JSON_INVALID",
        "VALUE_MISMATCH",
        "STREAM_FINISH_MISSING",
    ):
        return Severity.S2_LOUD
    return None


def normalize_exact_answer(content: str) -> str:
    return content.strip().lower().strip(".!\"' ")


def has_think_tags(text: str) -> bool:
    lower = text.lower()
    return "<think>" in lower or "</think>" in lower


def has_tool_markup(text: str) -> bool:
    lower = text.lower()
    return "<tool_call>" in lower or "<function=" in lower or "<parameter=" in lower


@dataclass
class TypedEqualResult:
    equal: bool
    codes: list[str] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)


def _is_bool(val: Any) -> bool:
    return isinstance(val, bool)


def _is_strict_int(val: Any) -> bool:
    return isinstance(val, int) and not isinstance(val, bool)


def _is_float(val: Any) -> bool:
    return isinstance(val, float)


def _make_diff(expected: Any, got: Any) -> str:
    exp_s = json.dumps(expected, indent=2, ensure_ascii=False, sort_keys=True).splitlines(keepends=True)
    got_s = json.dumps(got, indent=2, ensure_ascii=False, sort_keys=True).splitlines(keepends=True)
    return "".join(difflib.unified_diff(exp_s, got_s, fromfile="expected", tofile="got"))


def _is_truncated_before_markup(expected: str, got: str) -> bool:
    if not got or got == expected:
        return False
    # Exact prefix or prefix up to trailing whitespace before markup
    for candidate_prefix in (got, got.rstrip("\r\n\t ")):
        if not candidate_prefix:
            continue
        if expected.startswith(candidate_prefix) and len(candidate_prefix) < len(expected):
            remainder = expected[len(candidate_prefix) :]
            stripped_rem = remainder.lstrip("\r\n\t ")
            if (
                remainder.startswith(("</parameter>", "</tool_call>", "<"))
                or stripped_rem.startswith(("</parameter>", "</tool_call>", "<"))
            ):
                return True
    return False


def typed_equal(expected: Any, got: Any, path: str = "$") -> TypedEqualResult:
    """Recursive type-and-value comparison following §7 T3 rules."""
    codes: list[str] = []
    detail: dict[str, Any] = {}

    def _compare(exp: Any, g: Any, cur_path: str) -> None:
        # 1. Both dicts
        if isinstance(exp, dict) and isinstance(g, dict):
            exp_keys = set(exp.keys())
            got_keys = set(g.keys())
            if exp_keys != got_keys:
                if "VALUE_MISMATCH" not in codes:
                    codes.append("VALUE_MISMATCH")
                detail.setdefault("mismatches", []).append(
                    {
                        "path": cur_path,
                        "reason": "dict_keys_differ",
                        "missing_keys": sorted(exp_keys - got_keys),
                        "extra_keys": sorted(got_keys - exp_keys),
                        "diff": _make_diff(exp, g),
                    }
                )
            for k in sorted(exp_keys & got_keys):
                _compare(exp[k], g[k], f"{cur_path}.{k}")
            return

        # 2. Both lists
        if isinstance(exp, list) and isinstance(g, list):
            if len(exp) != len(g):
                if "VALUE_MISMATCH" not in codes:
                    codes.append("VALUE_MISMATCH")
                detail.setdefault("mismatches", []).append(
                    {
                        "path": cur_path,
                        "reason": "list_length_differs",
                        "expected_len": len(exp),
                        "got_len": len(g),
                        "diff": _make_diff(exp, g),
                    }
                )
            for idx in range(min(len(exp), len(g))):
                _compare(exp[idx], g[idx], f"{cur_path}[{idx}]")
            return

        # 3. Bool rules: A bool is never equal to an int
        if _is_bool(exp) or _is_bool(g):
            if _is_bool(exp) and _is_bool(g):
                if exp != g:
                    if "VALUE_MISMATCH" not in codes:
                        codes.append("VALUE_MISMATCH")
                    detail.setdefault("mismatches", []).append(
                        {"path": cur_path, "expected": exp, "got": g, "diff": _make_diff(exp, g)}
                    )
                return
            # One is bool, the other is not
            if _is_bool(exp) and _is_strict_int(g) and int(exp) == g:
                if "TYPE_COERCED" not in codes:
                    codes.append("TYPE_COERCED")
                detail.setdefault("coercions", []).append(
                    {"path": cur_path, "expected": exp, "got": g, "kind": "bool_to_int"}
                )
                return
            if _is_strict_int(exp) and _is_bool(g) and exp == int(g):
                if "TYPE_COERCED" not in codes:
                    codes.append("TYPE_COERCED")
                detail.setdefault("coercions", []).append(
                    {"path": cur_path, "expected": exp, "got": g, "kind": "int_to_bool"}
                )
                return
            if _is_bool(exp) and isinstance(g, str) and g.strip().lower() == str(exp).lower():
                if "TYPE_COERCED" not in codes:
                    codes.append("TYPE_COERCED")
                detail.setdefault("coercions", []).append(
                    {"path": cur_path, "expected": exp, "got": g, "kind": "bool_to_str"}
                )
                return
            if "VALUE_MISMATCH" not in codes:
                codes.append("VALUE_MISMATCH")
            detail.setdefault("mismatches", []).append(
                {"path": cur_path, "expected": exp, "got": g, "diff": _make_diff(exp, g)}
            )
            return

        # 4. Expected string
        if isinstance(exp, str):
            if isinstance(g, str):
                if exp == g:
                    return
                if _is_truncated_before_markup(exp, g):
                    if "VALUE_TRUNCATED" not in codes:
                        codes.append("VALUE_TRUNCATED")
                    detail.setdefault("truncations", []).append(
                        {
                            "path": cur_path,
                            "expected_len": len(exp),
                            "got_len": len(g),
                            "got": g,
                            "missing_suffix": exp[len(g) :],
                        }
                    )
                    return
                if "VALUE_MISMATCH" not in codes:
                    codes.append("VALUE_MISMATCH")
                detail.setdefault("mismatches", []).append(
                    {"path": cur_path, "expected": exp, "got": g, "diff": _make_diff(exp, g)}
                )
                return

            # Expected digit-string vs got int, with int(exp) == got -> TYPE_COERCED
            if _is_strict_int(g):
                s_clean = exp.strip()
                if s_clean.lstrip("-+").isdigit():
                    exp_int = int(s_clean)
                    if exp_int == g:
                        if "TYPE_COERCED" not in codes:
                            codes.append("TYPE_COERCED")
                        detail.setdefault("coercions", []).append(
                            {"path": cur_path, "expected": exp, "got": g, "kind": "str_to_int"}
                        )
                        return
                    if abs(exp_int) >= 2**50 and abs(exp_int - g) <= 2:
                        if "TYPE_COERCED" not in codes:
                            codes.append("TYPE_COERCED")
                        detail["precision_loss"] = "PRECISION_LOSS"
                        detail.setdefault("coercions", []).append(
                            {
                                "path": cur_path,
                                "expected": exp,
                                "got": g,
                                "kind": "PRECISION_LOSS",
                            }
                        )
                        return

            if _is_float(g):
                try:
                    if float(exp.strip()) == g:
                        if "TYPE_COERCED" not in codes:
                            codes.append("TYPE_COERCED")
                        detail.setdefault("coercions", []).append(
                            {"path": cur_path, "expected": exp, "got": g, "kind": "str_to_float"}
                        )
                        return
                except ValueError:
                    pass

            if "VALUE_MISMATCH" not in codes:
                codes.append("VALUE_MISMATCH")
            detail.setdefault("mismatches", []).append(
                {"path": cur_path, "expected": exp, "got": g, "diff": _make_diff(exp, g)}
            )
            return

        # 5. Expected int (strict)
        if _is_strict_int(exp):
            if _is_strict_int(g):
                if exp == g:
                    return
                if abs(exp) >= 2**50 and abs(exp - g) <= 2:
                    if "TYPE_COERCED" not in codes:
                        codes.append("TYPE_COERCED")
                    detail["precision_loss"] = "PRECISION_LOSS"
                    detail.setdefault("coercions", []).append(
                        {"path": cur_path, "expected": exp, "got": g, "kind": "PRECISION_LOSS"}
                    )
                    return
                if "VALUE_MISMATCH" not in codes:
                    codes.append("VALUE_MISMATCH")
                detail.setdefault("mismatches", []).append(
                    {"path": cur_path, "expected": exp, "got": g, "diff": _make_diff(exp, g)}
                )
                return

            if _is_float(g):
                # Expected big int vs got float within +-2 but not equal -> PRECISION_LOSS
                if abs(exp) >= 2**50 and abs(exp - int(g)) <= 2 and exp != int(g):
                    if "TYPE_COERCED" not in codes:
                        codes.append("TYPE_COERCED")
                    detail["precision_loss"] = "PRECISION_LOSS"
                    detail.setdefault("coercions", []).append(
                        {"path": cur_path, "expected": exp, "got": g, "kind": "PRECISION_LOSS"}
                    )
                    return
                # Expected int vs got float of equal value -> TYPE_COERCED
                if float(exp) == g:
                    if "TYPE_COERCED" not in codes:
                        codes.append("TYPE_COERCED")
                    detail.setdefault("coercions", []).append(
                        {"path": cur_path, "expected": exp, "got": g, "kind": "int_to_float"}
                    )
                    return
                if "VALUE_MISMATCH" not in codes:
                    codes.append("VALUE_MISMATCH")
                detail.setdefault("mismatches", []).append(
                    {"path": cur_path, "expected": exp, "got": g, "diff": _make_diff(exp, g)}
                )
                return

            if isinstance(g, str):
                s_clean = g.strip()
                if s_clean.lstrip("-+").isdigit():
                    got_int = int(s_clean)
                    if got_int == exp:
                        if "TYPE_COERCED" not in codes:
                            codes.append("TYPE_COERCED")
                        detail.setdefault("coercions", []).append(
                            {"path": cur_path, "expected": exp, "got": g, "kind": "int_to_str"}
                        )
                        return
                    if abs(exp) >= 2**50 and abs(exp - got_int) <= 2:
                        if "TYPE_COERCED" not in codes:
                            codes.append("TYPE_COERCED")
                        detail["precision_loss"] = "PRECISION_LOSS"
                        detail.setdefault("coercions", []).append(
                            {"path": cur_path, "expected": exp, "got": g, "kind": "PRECISION_LOSS"}
                        )
                        return

            if "VALUE_MISMATCH" not in codes:
                codes.append("VALUE_MISMATCH")
            detail.setdefault("mismatches", []).append(
                {"path": cur_path, "expected": exp, "got": g, "diff": _make_diff(exp, g)}
            )
            return

        # 6. Expected float
        if _is_float(exp):
            if _is_float(g) or _is_strict_int(g):
                if abs(exp - float(g)) <= 1e-9:
                    return
            elif isinstance(g, str):
                try:
                    if abs(exp - float(g.strip())) <= 1e-9:
                        if "TYPE_COERCED" not in codes:
                            codes.append("TYPE_COERCED")
                        detail.setdefault("coercions", []).append(
                            {"path": cur_path, "expected": exp, "got": g, "kind": "float_to_str"}
                        )
                        return
                except ValueError:
                    pass
            if "VALUE_MISMATCH" not in codes:
                codes.append("VALUE_MISMATCH")
            detail.setdefault("mismatches", []).append(
                {"path": cur_path, "expected": exp, "got": g, "diff": _make_diff(exp, g)}
            )
            return

        # 7. None or other types
        if exp is None and g is None:
            return
        if exp != g or type(exp) is not type(g):
            if "VALUE_MISMATCH" not in codes:
                codes.append("VALUE_MISMATCH")
            detail.setdefault("mismatches", []).append(
                {"path": cur_path, "expected": exp, "got": g, "diff": _make_diff(exp, g)}
            )

    _compare(expected, got, path)
    if codes and "diff" not in detail:
        detail["diff"] = _make_diff(expected, got)
    return TypedEqualResult(equal=(len(codes) == 0), codes=codes, detail=detail)


def build_oracle_result(
    codes: list[str],
    detail: dict[str, Any] | None = None,
    docs_shape: bool = False,
    default_severity: Severity = Severity.S2_LOUD,
) -> OracleResult:
    detail = detail or {}
    # Deduplicate preserving order
    unique_codes: list[str] = []
    for c in codes:
        if c not in unique_codes:
            unique_codes.append(c)

    if any(c in ERROR_CODES for c in unique_codes):
        return OracleResult(
            outcome=Outcome.ERROR,
            codes=unique_codes,
            fatal_codes=[],
            soft_codes=[],
            detail=detail,
            severity=None,
        )

    fatal = [c for c in unique_codes if is_fatal_code(c)]
    soft = [c for c in unique_codes if not is_fatal_code(c)]
    if fatal:
        sevs = [code_severity(c, docs_shape=docs_shape) for c in fatal]
        valid_sevs = [s for s in sevs if s is not None]
        worst_sev = min(valid_sevs, key=lambda s: s.value) if valid_sevs else default_severity
        return OracleResult(
            outcome=Outcome.FAIL,
            codes=unique_codes,
            fatal_codes=fatal,
            soft_codes=soft,
            detail=detail,
            severity=worst_sev,
        )

    return OracleResult(
        outcome=Outcome.PASS,
        codes=unique_codes,
        fatal_codes=[],
        soft_codes=soft,
        detail=detail,
        severity=None,
    )


def check_http_or_error(resp: ParsedResponse, docs_shape: bool = False) -> OracleResult | None:
    if resp.error_type in ("TIMEOUT", "HTTP_5XX"):
        return build_oracle_result([resp.error_type], detail={"status": resp.status}, docs_shape=docs_shape)
    if resp.status >= 500 or resp.status == 0:
        code = resp.error_type if resp.error_type in ERROR_CODES else "HTTP_5XX"
        return build_oracle_result([code], detail={"status": resp.status}, docs_shape=docs_shape)
    if 400 <= resp.status < 500:
        return build_oracle_result(
            ["HTTP_4XX"],
            detail={"status": resp.status, "raw": resp.raw},
            docs_shape=docs_shape,
        )
    return None


def apply_common_soft_and_stream_checks(
    req: dict[str, Any],
    resp: ParsedResponse,
    codes: list[str],
    detail: dict[str, Any],
) -> None:
    if resp.refusal:
        codes.append("REFUSAL")
        detail["refusal"] = resp.refusal
    if resp.finish_reason == "length":
        codes.append("TRUNCATED")
    if req.get("stream"):
        if resp.finish_reason is None:
            codes.append("STREAM_FINISH_MISSING")
        if (req.get("stream_options") or {}).get("include_usage") and not resp.usage:
            codes.append("STREAM_USAGE_MISSING")
        for delta_str in resp.content_deltas:
            if has_think_tags(delta_str):
                codes.append("REASONING_IN_CONTENT")
            if has_tool_markup(delta_str):
                codes.append("TOOL_MARKUP_IN_CONTENT")


def check_tool_call(
    req: dict[str, Any],
    resp: ParsedResponse,
    expected_tool: str,
    expected_args: dict[str, Any],
    tool_schema: dict[str, Any],
) -> OracleResult:
    """T3/T4 tool-argument fidelity oracle, checking in the exact order specified in §7."""
    early = check_http_or_error(resp, docs_shape=True)
    if early is not None:
        return early

    codes: list[str] = []
    detail: dict[str, Any] = {}

    # 1. NO_TOOL_CALLS
    if not resp.tool_calls:
        codes.append("NO_TOOL_CALLS")
        if has_tool_markup(resp.content):
            codes.append("TOOL_MARKUP_IN_CONTENT")
        apply_common_soft_and_stream_checks(req, resp, codes, detail)
        return build_oracle_result(codes, detail, docs_shape=True)

    # 2. TOOL_MARKUP_IN_CONTENT
    if has_tool_markup(resp.content):
        codes.append("TOOL_MARKUP_IN_CONTENT")

    tc0 = resp.tool_calls[0]
    got_name = tc0.get("name", "")
    args_raw = tc0.get("arguments_raw", "")

    # 3. WRONG_TOOL
    if got_name != expected_tool:
        codes.append("WRONG_TOOL")
        detail["expected_tool"] = expected_tool
        detail["got_tool"] = got_name

    # 4. ARGS_JSON_INVALID
    try:
        got_args = json.loads(args_raw)
    except Exception as exc:
        codes.append("ARGS_JSON_INVALID")
        detail["args_raw"] = args_raw
        detail["json_error"] = str(exc)
        if resp.finish_reason != "tool_calls":
            codes.append("FINISH_REASON_WRONG")
        apply_common_soft_and_stream_checks(req, resp, codes, detail)
        return build_oracle_result(codes, detail, docs_shape=True)

    # 5. jsonschema validation, giving SCHEMA_INVALID:<kw>
    validator = Draft202012Validator(tool_schema)
    errors = sorted(validator.iter_errors(got_args), key=lambda e: list(e.path))
    if errors:
        first_kw = str(errors[0].validator)
        codes.append(f"SCHEMA_INVALID:{first_kw}")
        detail["schema_errors"] = [
            {"validator": str(e.validator), "path": list(e.path), "message": e.message}
            for e in errors
        ]

    # 6. typed_equal(expected, got)
    teq = typed_equal(expected_args, got_args)
    if not teq.equal:
        codes.extend(teq.codes)
        detail.update(teq.detail)

    # 7. soft FINISH_REASON_WRONG if finish_reason != tool_calls
    if resp.finish_reason != "tool_calls":
        codes.append("FINISH_REASON_WRONG")
        detail["finish_reason"] = resp.finish_reason

    apply_common_soft_and_stream_checks(req, resp, codes, detail)
    return build_oracle_result(codes, detail, docs_shape=True)


def check_schema_response(
    req: dict[str, Any],
    resp: ParsedResponse,
    schema: dict[str, Any] | None,
    shape: str = "wrapped",
) -> OracleResult:
    """T5 schema enforcement oracle (§7)."""
    early = check_http_or_error(resp, docs_shape=True)
    if early is not None:
        return early

    codes: list[str] = []
    detail: dict[str, Any] = {}

    content = resp.content
    if not content.strip():
        if resp.finish_reason == "length":
            codes.append("TRUNCATED")
        else:
            codes.append("EMPTY_CONTENT")
        apply_common_soft_and_stream_checks(req, resp, codes, detail)
        return build_oracle_result(codes, detail, docs_shape=True)

    try:
        parsed = json.loads(content)
    except Exception as exc:
        if has_think_tags(content) or not content.lstrip().startswith("{"):
            codes.append("REASONING_IN_CONTENT")
        else:
            codes.append("JSON_PARSE")
        detail["json_error"] = str(exc)
        detail["content_preview"] = content[:500]
        apply_common_soft_and_stream_checks(req, resp, codes, detail)
        return build_oracle_result(codes, detail, docs_shape=True)

    if shape == "json_object" or schema is None:
        if not isinstance(parsed, dict):
            codes.append("SCHEMA_INVALID:type")
            detail["schema_errors"] = [{"validator": "type", "message": "Expected JSON object (dict)"}]
    else:
        validator = Draft202012Validator(schema)
        errors = sorted(validator.iter_errors(parsed), key=lambda e: list(e.path))
        if errors:
            first_kw = str(errors[0].validator)
            codes.append(f"SCHEMA_INVALID:{first_kw}")
            detail["schema_errors"] = [
                {"validator": str(e.validator), "path": list(e.path), "message": e.message}
                for e in errors
            ]

    apply_common_soft_and_stream_checks(req, resp, codes, detail)
    return build_oracle_result(codes, detail, docs_shape=True)
