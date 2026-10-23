from __future__ import annotations

import asyncio
import copy
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from second_opinion.budget import BudgetExceededError, BudgetGuard
from second_opinion.client import InferenceClient
from second_opinion.config import REPO_ROOT, get_nebius_api_key, load_config
from second_opinion.contracts.registry import get_contract_map
from second_opinion.pacer import HeaderAwarePacer
from second_opinion.stats import fisher_two_sided, wilson_ci

FACTOR_MENU = {
    "thinking",
    "stream",
    "max_tokens",
    "shape",
    "append_literal",
    "remove_schema_keyword",
    "inline_refs",
    "arg_type",
    "temperature",
}


@dataclass
class ArmResult:
    factor: str
    level: Any
    failures: int
    trials: int
    failure_rate: float
    wilson_lo: float
    wilson_hi: float
    codes: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _inline_schema_refs(schema: dict[str, Any]) -> dict[str, Any]:
    s = copy.deepcopy(schema)
    defs = s.pop("$defs", None) or s.pop("definitions", None) or {}

    def _resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node and isinstance(node["$ref"], str):
                ref_str = node["$ref"]
                ref_name = ref_str.rsplit("/", 1)[-1]
                if ref_name in defs:
                    return _resolve(copy.deepcopy(defs[ref_name]))
            return {k: _resolve(v) for k, v in node.items()}
        if isinstance(node, list):
            return [_resolve(x) for x in node]
        return node

    return _resolve(s)


def _remove_keyword_recursive(node: Any, kw: str) -> Any:
    if isinstance(node, dict):
        return {k: _remove_keyword_recursive(v, kw) for k, v in node.items() if k != kw}
    if isinstance(node, list):
        return [_remove_keyword_recursive(x, kw) for x in node]
    return node


def apply_factor(base_req: dict[str, Any], factor: str, level: Any) -> dict[str, Any]:
    """Apply a factor from the Investigator factor menu (§11.3) to a copy of base_req."""
    if factor not in FACTOR_MENU:
        raise ValueError(f"Unsupported factor {factor!r}. Allowed: {sorted(FACTOR_MENU)}")

    req = copy.deepcopy(base_req)

    if factor == "thinking":
        req["chat_template_kwargs"] = {"enable_thinking": bool(level)}
        return req

    if factor == "stream":
        if bool(level):
            req["stream"] = True
            req["stream_options"] = {"include_usage": True}
        else:
            req.pop("stream", None)
            req.pop("stream_options", None)
        return req

    if factor == "max_tokens":
        req["max_tokens"] = int(level)
        return req

    if factor == "temperature":
        req["temperature"] = float(level)
        return req

    if factor == "shape":
        rf = req.get("response_format") or {}
        js = rf.get("json_schema") or {}
        if isinstance(js, dict) and "schema" in js:
            raw_schema = js["schema"]
            name = js.get("name", "schema")
        else:
            raw_schema = js
            name = "schema"
        if str(level) == "bare":
            req["response_format"] = {"type": "json_schema", "json_schema": raw_schema}
        else:
            req["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": name, "schema": raw_schema, "strict": True},
            }
        return req

    if factor == "remove_schema_keyword":
        kw = str(level)
        if "response_format" in req:
            req["response_format"] = _remove_keyword_recursive(req["response_format"], kw)
        if "tools" in req:
            req["tools"] = _remove_keyword_recursive(req["tools"], kw)
        return req

    if factor == "inline_refs":
        if bool(level) and "response_format" in req:
            rf = req["response_format"]
            js = rf.get("json_schema")
            if isinstance(js, dict) and "schema" in js and isinstance(js["schema"], dict):
                js["schema"] = _inline_schema_refs(js["schema"])
            elif isinstance(js, dict):
                rf["json_schema"] = _inline_schema_refs(js)
        return req

    if factor == "append_literal":
        suffix = str(level)
        exp_args = copy.deepcopy(req.get("_expected_args") or {})
        if exp_args:
            for k, v in exp_args.items():
                if isinstance(v, str):
                    exp_args[k] = v + suffix
                    break
            req["_expected_args"] = exp_args
            # Update user message JSON block
            msgs = req.get("messages") or []
            for m in msgs:
                if m.get("role") == "user":
                    tname = ((req.get("tool_choice") or {}).get("function") or {}).get("name", "tool")
                    m["content"] = (
                        f"Call `{tname}` exactly once. Use exactly these arguments. "
                        f"Copy every character and keep every type exactly as given:\n"
                        f"```json\n{json.dumps(exp_args, indent=2, ensure_ascii=False)}\n```"
                    )
        return req

    if factor == "arg_type":
        arg_name = ""
        target_type = "string"
        if isinstance(level, dict):
            arg_name = str(level.get("arg", ""))
            target_type = str(level.get("type", "string"))
        elif isinstance(level, (list, tuple)) and len(level) == 2:
            arg_name, target_type = str(level[0]), str(level[1])
        elif isinstance(level, str) and ":" in level:
            arg_name, target_type = [x.strip() for x in level.split(":", 1)]
        else:
            target_type = str(level)

        tools = req.get("tools") or []
        if tools and isinstance(tools[0], dict):
            props = (((tools[0].get("function") or {}).get("parameters") or {}).get("properties")) or {}
            if arg_name and arg_name in props:
                props[arg_name]["type"] = target_type
            elif not arg_name and props:
                first_k = next(iter(props.keys()))
                props[first_k]["type"] = target_type
        return req

    return req


def _load_experiment_context() -> dict[str, Any]:
    ctx_path_str = os.environ.get("SECOND_OPINION_EXPERIMENT_CONTEXT")
    ctx_path = Path(ctx_path_str) if ctx_path_str else (REPO_ROOT / "runs" / "active_experiment_context.json")
    if ctx_path.exists():
        try:
            return json.loads(ctx_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def base_request(cluster_id: str) -> dict[str, Any]:
    """Load the minimal failing request for `cluster_id`."""
    ctx = _load_experiment_context()
    clusters = ctx.get("clusters") or {}
    if cluster_id in clusters:
        return copy.deepcopy(clusters[cluster_id]["minimal_request"])
    if "minimal_request" in ctx:
        return copy.deepcopy(ctx["minimal_request"])
    # Fallback default request for offline tests
    return {
        "model": "nvidia/Nemotron-3_5-Lightning",
        "messages": [
            {"role": "system", "content": "You are a component in an automated test. Follow the user's instructions exactly."},
            {"role": "user", "content": "Reply with one word: banana"},
        ],
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 1024,
        "_contract_id": "T2",
        "_cluster_id": cluster_id,
    }


async def _run_arm_async(req: dict[str, Any], factor: str, level: Any, n: int) -> ArmResult:
    ctx = _load_experiment_context()
    # Support offline mock mode in unit tests
    mock_rates = ctx.get("mock_rates")
    if isinstance(mock_rates, dict):
        key = f"{factor}={level}"
        rate = float(mock_rates.get(key, mock_rates.get("default", 0.0)))
        failures = int(round(rate * n))
        lo, hi = wilson_ci(failures, n)
        return ArmResult(
            factor=factor,
            level=level,
            failures=failures,
            trials=n,
            failure_rate=failures / max(1, n),
            wilson_lo=lo,
            wilson_hi=hi,
            codes={"MOCK_FAIL": failures} if failures else {},
        )

    config = load_config()
    api_key = get_nebius_api_key()
    max_trials = int(ctx.get("max_arm_n", min(n, 50)))
    actual_n = max(1, min(int(n), max_trials))
    cluster_budget = float(ctx.get("cluster_budget_usd", config.cluster_budget_usd))
    budget = BudgetGuard(config, cap_override_usd=min(config.budget_usd, BudgetGuard(config).spent_usd + cluster_budget))
    pacer = HeaderAwarePacer(base_rpm=config.rpm)

    modified_req = apply_factor(req, factor, level)
    contract_id = str(modified_req.get("_contract_id") or ctx.get("contract_id") or "T3")
    send_req = {k: v for k, v in modified_req.items() if not k.startswith("_")}
    model = str(send_req.get("model", config.code_model))
    contract = get_contract_map()[contract_id]

    failures = 0
    completed = 0
    codes_count: dict[str, int] = {}

    async with InferenceClient(
        base_url=config.nebius_base_url,
        api_key=api_key,
        pacer=pacer,
        connect_timeout_s=config.connect_timeout_s,
        read_timeout_s=config.read_timeout_s,
    ) as client:
        for _ in range(actual_n):
            est_cost = config.estimate_cost(model, 400, 1000)
            try:
                await budget.reserve(est_cost)
            except BudgetExceededError:
                break
            resp = await client.chat_completion(send_req, estimated_tokens=1400)
            ores = contract.check(modified_req, resp)
            actual_cost = budget.compute_call_cost(model, resp.usage)
            await budget.commit(est_cost, actual_cost, model, "experiment", str(ctx.get("run_id", "exp")), resp.usage)
            completed += 1
            if ores.outcome.value == "fail":
                failures += 1
                for c in ores.fatal_codes:
                    codes_count[c] = codes_count.get(c, 0) + 1

    trials = max(1, completed)
    lo, hi = wilson_ci(failures, trials)
    return ArmResult(
        factor=factor,
        level=level,
        failures=failures,
        trials=trials,
        failure_rate=failures / trials,
        wilson_lo=lo,
        wilson_hi=hi,
        codes=codes_count,
    )


def arm(base_req: dict[str, Any], factor: str, level: Any, n: int = 100) -> ArmResult:
    """Run an experimental arm modifying `factor` to `level` for `n` trials."""
    return asyncio.run(_run_arm_async(base_req, factor, level, n))


def compare(arm_a: ArmResult, arm_b: ArmResult, hypothesis_id: str) -> dict[str, Any]:
    """Compare two experimental arms with two-sided Fisher exact test and persist result."""
    pval = fisher_two_sided(arm_a.failures, arm_a.trials, arm_b.failures, arm_b.trials)
    diff = abs(arm_a.failure_rate - arm_b.failure_rate)
    result = {
        "hypothesis_id": hypothesis_id,
        "factor": arm_a.factor,
        "arm_a": arm_a.to_dict(),
        "arm_b": arm_b.to_dict(),
        "p_value": pval,
        "rate_diff": diff,
    }
    ctx = _load_experiment_context()
    out_dir_str = ctx.get("experiments_out_dir")
    out_dir = Path(out_dir_str) if out_dir_str else (REPO_ROOT / "runs" / "experiments")
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_id = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in hypothesis_id)
    (out_dir / f"{safe_id}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result

