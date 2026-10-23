from __future__ import annotations

import copy
import json
import math
from typing import Any, Awaitable, Callable

from second_opinion.contracts.t3_tool_args import build_t3_prompt
from second_opinion.testkit import _inline_schema_refs, _remove_keyword_recursive


def retention_threshold(p0: float, k: int = 12) -> int:
    """Keep a reduction if failures >= max(1, ceil(0.5 * k * p0)) (§11.2)."""
    return max(1, int(math.ceil(0.5 * k * max(0.0, min(1.0, p0)))))


def generate_shrink_candidates(req: dict[str, Any], contract_id: str) -> list[tuple[str, dict[str, Any]]]:
    """Generate up to 8 deterministic reduction candidates for a failing request (§11.2)."""
    candidates: list[tuple[str, dict[str, Any]]] = []

    # 1. Stream -> False if currently True
    if req.get("stream"):
        c_stream = copy.deepcopy(req)
        c_stream.pop("stream", None)
        c_stream.pop("stream_options", None)
        candidates.append(("stream=False", c_stream))

    # 2. Thinking -> False if currently True or omitted
    kw = req.get("chat_template_kwargs") or {}
    if kw.get("enable_thinking") is not False:
        c_think = copy.deepcopy(req)
        c_think["chat_template_kwargs"] = {"enable_thinking": False}
        candidates.append(("thinking=False", c_think))

    if contract_id == "T3":
        tools = req.get("tools") or []
        if tools and isinstance(tools[0], dict):
            fn = tools[0].get("function") or {}
            params = fn.get("parameters") or {}
            props = params.get("properties") or {}
            exp_args = dict(req.get("_expected_args") or {})
            if not exp_args:
                # Extract from fenced json in user message if present
                msgs = req.get("messages") or []
                for m in msgs:
                    if m.get("role") == "user" and "```json" in str(m.get("content", "")):
                        try:
                            block = str(m["content"]).split("```json", 1)[1].split("```", 1)[0]
                            exp_args = json.loads(block.strip())
                        except Exception:
                            pass

            # Drop argument keys one at a time (keep at least 1 key)
            keys = list(props.keys())
            if len(keys) > 1:
                for drop_k in keys:
                    if len(candidates) >= 8:
                        break
                    c_drop = copy.deepcopy(req)
                    c_fn = c_drop["tools"][0]["function"]
                    c_props = c_fn["parameters"]["properties"]
                    c_props.pop(drop_k, None)
                    if "required" in c_fn["parameters"]:
                        c_fn["parameters"]["required"] = [
                            k for k in c_fn["parameters"]["required"] if k != drop_k
                        ]
                    new_args = {k: v for k, v in exp_args.items() if k != drop_k}
                    c_drop["_expected_args"] = new_args
                    for m in c_drop.get("messages", []):
                        if m.get("role") == "user":
                            m["content"] = build_t3_prompt(c_fn.get("name", "tool"), new_args)
                    candidates.append((f"drop_arg:{drop_k}", c_drop))

            # Halve markup string (ddmin on lines, then characters)
            if "content" in exp_args and isinstance(exp_args["content"], str):
                content_str = exp_args["content"]
                lines = content_str.splitlines()
                if len(lines) >= 2:
                    mid = max(1, len(lines) // 2)
                    for label, subset in (
                        ("markup_first_half_lines", "\n".join(lines[:mid])),
                        ("markup_second_half_lines", "\n".join(lines[mid:])),
                    ):
                        if len(candidates) >= 8:
                            break
                        c_half = copy.deepcopy(req)
                        new_args = dict(exp_args, content=subset)
                        c_half["_expected_args"] = new_args
                        tname = fn.get("name", "write_file")
                        for m in c_half.get("messages", []):
                            if m.get("role") == "user":
                                m["content"] = build_t3_prompt(tname, new_args)
                        candidates.append((label, c_half))
                elif len(content_str) > 10 and len(candidates) < 8:
                    mid_c = len(content_str) // 2
                    c_chars = copy.deepcopy(req)
                    new_args = dict(exp_args, content=content_str[:mid_c])
                    c_chars["_expected_args"] = new_args
                    tname = fn.get("name", "write_file")
                    for m in c_chars.get("messages", []):
                        if m.get("role") == "user":
                            m["content"] = build_t3_prompt(tname, new_args)
                    candidates.append(("markup_first_half_chars", c_chars))

    elif contract_id == "T5":
        rf = req.get("response_format") or {}
        js = rf.get("json_schema") or {}
        schema = js.get("schema") if isinstance(js, dict) and "schema" in js else js
        if isinstance(schema, dict):
            # Inline $ref if $defs present
            if "$defs" in schema or "definitions" in schema:
                c_inline = copy.deepcopy(req)
                inlined = _inline_schema_refs(schema)
                if "schema" in c_inline["response_format"].get("json_schema", {}):
                    c_inline["response_format"]["json_schema"]["schema"] = inlined
                else:
                    c_inline["response_format"]["json_schema"] = inlined
                candidates.append(("inline_refs", c_inline))

            # Drop schema keywords
            for kw in ("additionalProperties", "minimum", "maximum", "pattern", "minItems", "maxItems"):
                if len(candidates) >= 8:
                    break
                if kw in json.dumps(schema):
                    c_kw = copy.deepcopy(req)
                    c_kw["response_format"] = _remove_keyword_recursive(c_kw["response_format"], kw)
                    candidates.append((f"drop_kw:{kw}", c_kw))

            # Drop properties (keep at least 1)
            props = schema.get("properties") or {}
            prop_keys = list(props.keys())
            if len(prop_keys) > 1:
                for pk in prop_keys:
                    if len(candidates) >= 8:
                        break
                    c_prop = copy.deepcopy(req)
                    target_s = (
                        c_prop["response_format"]["json_schema"]["schema"]
                        if "schema" in c_prop["response_format"].get("json_schema", {})
                        else c_prop["response_format"]["json_schema"]
                    )
                    target_s["properties"].pop(pk, None)
                    if "required" in target_s:
                        target_s["required"] = [r for r in target_s["required"] if r != pk]
                    candidates.append((f"drop_prop:{pk}", c_prop))

    return candidates[:8]


async def shrink_cluster_request(
    initial_req: dict[str, Any],
    contract_id: str,
    p0: float,
    score_fn: Callable[[dict[str, Any], int], Awaitable[tuple[int, int]]],
    k: int = 12,
    max_candidates: int = 8,
) -> tuple[dict[str, Any], float, list[dict[str, Any]]]:
    """Deterministically shrink `initial_req` using at most `max_candidates` evaluations of `k` trials (§11.2).

    `score_fn(candidate_req, k)` returns `(failures, trials)`.
    Returns `(minimal_req, measured_failure_rate, history)`.
    """
    current_req = copy.deepcopy(initial_req)
    current_rate = p0
    min_failures = retention_threshold(p0, k=k)
    history: list[dict[str, Any]] = []
    evaluated = 0

    candidates = generate_shrink_candidates(current_req, contract_id)
    idx = 0
    while idx < len(candidates) and evaluated < max_candidates:
        label, cand_req = candidates[idx]
        idx += 1
        evaluated += 1
        failures, trials = await score_fn(cand_req, k)
        rate = failures / max(1, trials)
        kept = failures >= min_failures
        history.append(
            {
                "step": evaluated,
                "reduction": label,
                "failures": failures,
                "trials": trials,
                "failure_rate": rate,
                "threshold": min_failures,
                "kept": kept,
            }
        )
        if kept:
            current_req = cand_req
            current_rate = rate
            # Regenerate remaining candidates from the newly shrunk request
            new_cands = generate_shrink_candidates(current_req, contract_id)
            seen_labels = {h["reduction"] for h in history}
            candidates = [(l, r) for (l, r) in new_cands if l not in seen_labels]
            idx = 0

    return current_req, current_rate, history

