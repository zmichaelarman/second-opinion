"""Propose causes (CLAUDE.md §11.3): Nemotron 3 Ultra, thinking on, json_schema output.

Input: the minimal request, up to 3 failing and 3 passing responses, and the factor menu.
Output: up to 3 {factor, level_a, level_b, rationale}. Factors off the menu, or levels that
cannot be applied to this request, are dropped.
"""

from __future__ import annotations

import json
from typing import Any, Awaitable, Callable

from ..contracts.base import THINKING_SWITCH, TEMPERATURE, TOP_P, apply_thinking
from ..oracles import strict_json_loads
from .spec import FACTORS, FactorError, Spec

MAX_HYPOTHESES = 3
MAX_TOKENS = 8192
SYSTEM = (
    "You diagnose failures in an LLM serving platform's OpenAI-compatible conversion layer (reasoning in "
    "<think> tags and Qwen3-Coder-style XML tool calls converted into OpenAI JSON fields). Propose causes that "
    "a controlled experiment can test by changing exactly one factor from the menu."
)


def hypothesis_schema() -> dict[str, Any]:
    level = {"type": ["string", "number", "boolean"]}
    return {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "A short title for the finding, under 12 words."},
            "hypotheses": {
                "type": "array", "minItems": 1, "maxItems": MAX_HYPOTHESES,
                "items": {
                    "type": "object",
                    "properties": {"factor": {"type": "string", "enum": list(FACTORS)}, "level_a": level,
                                   "level_b": level, "rationale": {"type": "string"}},
                    "required": ["factor", "level_a", "level_b", "rationale"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["title", "hypotheses"],
        "additionalProperties": False,
    }


def _brief(example: dict[str, Any]) -> dict[str, Any]:
    keep = ("codes", "finish_reason", "content_head", "tool_calls", "reasoning_chars")
    out = {k: example.get(k) for k in keep if example.get(k) not in (None, [], "")}
    mismatches = (example.get("detail") or {}).get("mismatches")
    if mismatches:
        out["mismatches"] = mismatches[:4]
    return out


def hypothesis_prompt(spec: Spec, rate: float, failing: list[dict[str, Any]], passing: list[dict[str, Any]],
                      rejected: list[dict[str, Any]] | None = None) -> str:
    context = {
        "contract": spec.contract_id,
        "failure_signature": spec.signature,
        "measured_failure_rate": round(rate, 3),
        "minimal_request": spec.request(),
        "failing_responses": [_brief(e) for e in failing[:3]],
        "passing_responses": [_brief(e) for e in passing[:3]],
        "factor_menu": FACTORS,
    }
    text = ("Here is a failure cluster from our contract tests.\n\n" + json.dumps(context, indent=2, ensure_ascii=False)
            + f"\n\nPropose up to {MAX_HYPOTHESES} causes. For each, name one factor from factor_menu and two levels "
            "whose failure rates you expect to differ by at least 5 percentage points; level_a should be the level you "
            "expect to fail more. Also give a short title for the finding.")
    if rejected:
        text += ("\n\nThese hypotheses were already tested and rejected (no significant difference), so propose "
                 "different ones:\n" + json.dumps(rejected, indent=2, ensure_ascii=False))
    return text


def hypothesis_request(model: str, prompt: str) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAX_TOKENS,
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": "hypotheses", "schema": hypothesis_schema(), "strict": True}},
    }
    return apply_thinking(body, True, THINKING_SWITCH)


def validate(raw: list[Any], spec: Spec, already: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Keep menu factors whose two levels both apply to this request and differ; dedupe; at most 3."""
    seen = {(h["factor"], json.dumps(h["level_a"]), json.dumps(h["level_b"])) for h in already or []}
    kept = []
    for h in raw if isinstance(raw, list) else []:
        if not isinstance(h, dict) or h.get("factor") not in FACTORS:
            continue
        a, b = h.get("level_a"), h.get("level_b")
        try:
            sa, sb = spec.with_factor(h["factor"], a), spec.with_factor(h["factor"], b)
        except (FactorError, ValueError, TypeError, KeyError):
            continue
        if sa.request() == sb.request():
            continue
        key = (h["factor"], json.dumps(a), json.dumps(b))
        if key in seen:
            continue
        seen.add(key)
        kept.append({"factor": h["factor"], "level_a": a, "level_b": b, "rationale": str(h.get("rationale", ""))[:600]})
    return kept[:MAX_HYPOTHESES]


def fallback_hypotheses(spec: Spec, already: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Menu defaults, used only when the model's proposal is unusable."""
    options: list[dict[str, Any]] = []
    if spec.kind == "tool":
        options += [{"factor": "stream", "level_a": True, "level_b": False},
                    {"factor": "thinking", "level_a": True, "level_b": False},
                    {"factor": "temperature", "level_a": 1.0, "level_b": 0}]
    elif spec.kind == "json":
        options += [{"factor": "thinking", "level_a": True, "level_b": False},
                    {"factor": "shape", "level_a": "bare", "level_b": "wrapped"},
                    {"factor": "temperature", "level_a": 1.0, "level_b": 0}]
    else:
        options += [{"factor": "temperature", "level_a": 1.0, "level_b": 0}]
    for o in options:
        o["rationale"] = "menu default (the hypothesis model's proposal was unusable)"
    return validate(options, spec, already)


async def propose(
    complete: Callable[[dict[str, Any]], Awaitable[str | None]],
    model: str,
    spec: Spec,
    rate: float,
    failing: list[dict[str, Any]],
    passing: list[dict[str, Any]],
    rejected: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """`complete(body)` sends one chat request and returns the reply content."""
    reply = await complete(hypothesis_request(model, hypothesis_prompt(spec, rate, failing, passing, rejected)))
    title, hyps, problem = None, [], None
    try:
        data = strict_json_loads(reply or "")
        title = str(data.get("title") or "")[:120] or None
        hyps = validate(data.get("hypotheses"), spec, rejected)
        if not hyps:
            problem = "no usable hypotheses in the reply"
    except (ValueError, AttributeError):
        problem = "reply was not valid JSON"
    source = "model"
    if not hyps:
        hyps, source = fallback_hypotheses(spec, rejected), "fallback"
    return {"title": title, "hypotheses": hyps, "source": source, "problem": problem, "raw_reply": (reply or "")[:4000]}
