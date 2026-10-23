from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from second_opinion.budget import BudgetGuard
from second_opinion.client import InferenceClient
from second_opinion.config import AppConfig, get_nebius_api_key
from second_opinion.contracts.base import Condition, build_base_chat_request
from second_opinion.contracts.t3_tool_args import build_t3_prompt
from second_opinion.contracts.t5_schemas import build_response_format
from second_opinion.oracles import check_schema_response, check_tool_call
from second_opinion.pacer import HeaderAwarePacer
from second_opinion.stats import wilson_ci

TRICKY_STRINGS = [
    "02134",
    "007",
    "9876543210",
    '<div class="item">5 < 6 && 7 > 3</div></parameter><tool_call>',
    'He said "hello" \\ backslash; naïve café — 東京',
]

TRICKY_INTS = [
    42,
    0,
    -17,
    9007199254740991,
    9007199254740993,
]

TRICKY_FLOATS = [
    0.5,
    -71.13,
    42.35,
    0.000125,
    99.95,
]


def _generate_value_for_prop(prop_schema: dict[str, Any], seed_idx: int, rng: random.Random) -> Any:
    if "enum" in prop_schema and prop_schema["enum"]:
        enum_vals = prop_schema["enum"]
        return enum_vals[seed_idx % len(enum_vals)]

    if "anyOf" in prop_schema:
        branches = prop_schema["anyOf"]
        branch = branches[seed_idx % len(branches)]
        return _generate_value_for_prop(branch, seed_idx, rng)

    ptype = prop_schema.get("type", "string")
    if isinstance(ptype, list):
        chosen_t = ptype[seed_idx % len(ptype)]
    else:
        chosen_t = ptype

    if chosen_t == "null":
        return None
    if chosen_t == "boolean":
        return (seed_idx % 2) == 0
    if chosen_t == "integer":
        val = TRICKY_INTS[seed_idx % len(TRICKY_INTS)]
        if "minimum" in prop_schema:
            val = max(int(prop_schema["minimum"]), val)
        if "maximum" in prop_schema:
            val = min(int(prop_schema["maximum"]), val)
        return val
    if chosen_t == "number":
        val_f = TRICKY_FLOATS[seed_idx % len(TRICKY_FLOATS)]
        if "minimum" in prop_schema:
            val_f = max(float(prop_schema["minimum"]), val_f)
        if "maximum" in prop_schema:
            val_f = min(float(prop_schema["maximum"]), val_f)
        return val_f
    if chosen_t == "array":
        items_schema = prop_schema.get("items") or {"type": "string"}
        min_items = int(prop_schema.get("minItems", 2))
        max_items = int(prop_schema.get("maxItems", max(min_items, 3)))
        count = max(min_items, min(max_items, 2 + (seed_idx % 2)))
        return [_generate_value_for_prop(items_schema, seed_idx + i, rng) for i in range(count)]
    if chosen_t == "object":
        props = prop_schema.get("properties") or {}
        return {
            k: _generate_value_for_prop(v, seed_idx + idx, rng)
            for idx, (k, v) in enumerate(props.items())
        }
    return TRICKY_STRINGS[seed_idx % len(TRICKY_STRINGS)]


def generate_seeded_tool_arg_sets(tool_def: dict[str, Any], count: int = 5, seed: int = 42) -> list[dict[str, Any]]:
    """Generate `count` tricky but valid argument sets for a user tool definition (§14)."""
    fn = tool_def.get("function") if "function" in tool_def else tool_def
    params = fn.get("parameters") or {"type": "object", "properties": {}}
    props = params.get("properties") or {}
    rng = random.Random(seed)
    arg_sets: list[dict[str, Any]] = []
    for s_idx in range(count):
        args: dict[str, Any] = {}
        for p_idx, (prop_name, prop_schema) in enumerate(props.items()):
            args[prop_name] = _generate_value_for_prop(prop_schema, s_idx + p_idx, rng)
        arg_sets.append(args)
    return arg_sets


async def run_byo_check(
    tools_path: Path | None,
    schemas_path: Path | None,
    models: list[str],
    n_trials: int,
    config: AppConfig,
    console: Console | None = None,
) -> list[dict[str, Any]]:
    """Run bring-your-own tools and schemas reliability checks (§14)."""
    console = console or Console()
    api_key = get_nebius_api_key()
    pacer = HeaderAwarePacer(base_rpm=config.rpm)
    budget = BudgetGuard(config)

    tools_list: list[dict[str, Any]] = []
    if tools_path and tools_path.exists():
        raw_t = json.loads(tools_path.read_text(encoding="utf-8"))
        tools_list = raw_t if isinstance(raw_t, list) else [raw_t]

    schemas_list: list[tuple[str, dict[str, Any]]] = []
    if schemas_path and schemas_path.exists():
        raw_s = json.loads(schemas_path.read_text(encoding="utf-8"))
        if isinstance(raw_s, list):
            for i, s in enumerate(raw_s):
                name = str(s.get("title") or s.get("name") or f"schema_{i}")
                schemas_list.append((name, s))
        elif isinstance(raw_s, dict):
            if "properties" in raw_s or "$defs" in raw_s:
                schemas_list.append((str(raw_s.get("title", "custom_schema")), raw_s))
            else:
                for k, v in raw_s.items():
                    if isinstance(v, dict):
                        schemas_list.append((k, v))

    results: list[dict[str, Any]] = []
    cond = Condition(thinking=True, stream=False)

    async with InferenceClient(
        base_url=config.nebius_base_url,
        api_key=api_key,
        pacer=pacer,
        connect_timeout_s=config.connect_timeout_s,
        read_timeout_s=config.read_timeout_s,
    ) as client:
        for model in models:
            for t_def in tools_list:
                wrapped_tool = (
                    t_def
                    if "function" in t_def
                    else {"type": "function", "function": t_def}
                )
                fn = wrapped_tool["function"]
                t_name = str(fn.get("name", "custom_tool"))
                t_schema = dict(fn.get("parameters") or {"type": "object"})
                arg_sets = generate_seeded_tool_arg_sets(wrapped_tool, count=5)

                fails = 0
                for trial in range(n_trials):
                    expected_args = arg_sets[trial % len(arg_sets)]
                    prompt = build_t3_prompt(t_name, expected_args)
                    req = build_base_chat_request(model, cond, prompt)
                    req["tools"] = [wrapped_tool]
                    req["tool_choice"] = {"type": "function", "function": {"name": t_name}}

                    est = config.estimate_cost(model, 400, 1000)
                    await budget.reserve(est)
                    resp = await client.chat_completion(req, estimated_tokens=1400)
                    ores = check_tool_call(req, resp, t_name, expected_args, t_schema)
                    actual = budget.compute_call_cost(model, resp.usage)
                    await budget.commit(est, actual, model, "byo", "byo", resp.usage)
                    if ores.outcome.value == "fail":
                        fails += 1

                lo, hi = wilson_ci(fails, max(1, n_trials))
                results.append(
                    {
                        "kind": "tool",
                        "name": t_name,
                        "model": model,
                        "failures": fails,
                        "trials": n_trials,
                        "failure_rate": fails / max(1, n_trials),
                        "wilson_ci": [lo, hi],
                    }
                )

            for s_name, s_def in schemas_list:
                fails = 0
                for _ in range(n_trials):
                    req = build_base_chat_request(model, cond, "Produce one valid example object.")
                    req["response_format"] = build_response_format(s_name, s_def, "wrapped")

                    est = config.estimate_cost(model, 400, 1000)
                    await budget.reserve(est)
                    resp = await client.chat_completion(req, estimated_tokens=1400)
                    ores = check_schema_response(req, resp, schema=s_def, shape="wrapped")
                    actual = budget.compute_call_cost(model, resp.usage)
                    await budget.commit(est, actual, model, "byo", "byo", resp.usage)
                    if ores.outcome.value == "fail":
                        fails += 1

                lo, hi = wilson_ci(fails, max(1, n_trials))
                results.append(
                    {
                        "kind": "schema",
                        "name": s_name,
                        "model": model,
                        "failures": fails,
                        "trials": n_trials,
                        "failure_rate": fails / max(1, n_trials),
                        "wilson_ci": [lo, hi],
                    }
                )

    table = Table(title="Bring-Your-Own Reliability Table")
    table.add_column("Type")
    table.add_column("Name", style="magenta")
    table.add_column("Model", style="cyan")
    table.add_column("Failures / Trials", justify="right")
    table.add_column("Failure Rate (95% Wilson CI)", justify="right")
    for r in results:
        lo, hi = r["wilson_ci"]
        table.add_row(
            str(r["kind"]),
            str(r["name"]),
            str(r["model"]),
            f"{r['failures']}/{r['trials']}",
            f"{r['failure_rate'] * 100:.1f}% ([{lo * 100:.1f}%, {hi * 100:.1f}%])",
        )
    console.print(table)
    return results

