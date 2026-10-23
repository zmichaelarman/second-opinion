"""`second-opinion check` (CLAUDE.md §14): bring your own tools and schemas.

Tools: 5 seeded argument sets per tool definition, generated from its JSON schema with tricky
but valid values (leading-zero and digit-only strings, markup, unicode, quotes, integers near
2^53, booleans, null where allowed, nested arrays), each run with the exact-copy prompt and the
T3 oracle. Schemas: "Produce one valid example object." with the T5 oracle. The result is a
per-tool, per-model reliability table.
"""

from __future__ import annotations

import copy
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import httpx
from jsonschema import Draft202012Validator
from rich.console import Console
from rich.table import Table

from . import analysis
from .budget import BudgetGuard
from .client import ApiClient
from .config import Settings
from .contracts.base import BaseContract, Condition, OracleResult, Severity, chat_request
from .discovery import discover
from .fixtures.payloads import forced_tool_choice, tool_prompt
from .oracles import Findings, check_json_content, check_no_think_tags, check_tool_call, http_gate
from .pacer import Pacer
from .parse import ParsedResponse
from .planner import Cell, TokenEstimates, WorkItem, _rounds, cost_table, estimate_cost
from .runner import Runner, resolve_models
from .stats import wilson_ci
from .store import RunStore, git_info, utc_now

SETS_PER_TOOL = 5
SCHEMA_PROMPT = "Produce one valid example object."
TRICKY_STRINGS = ["02134", "007", "12345", '<div class="x">5 < 6 && 7 > 3</div>', 'He said "hi" \\ backslash',
                  "naïve café ✓", "line one\nline two", "</parameter>", "  padded  "]
TRICKY_INTS = [9007199254740993, 42, 0, -7, 2**53 - 1]
TRICKY_NUMBERS = [0.5, -71.13, 1e-07, 3.0, 9007199254740.5]


def _types(schema: dict[str, Any]) -> list[str]:
    t = schema.get("type")
    if isinstance(t, list):
        return t
    if t:
        return [t]
    if "properties" in schema:
        return ["object"]
    if "items" in schema:
        return ["array"]
    return ["string"]


def generate(schema: dict[str, Any], rng: random.Random, root: dict[str, Any] | None = None, depth: int = 0) -> Any:
    """One tricky-but-valid value for a JSON schema."""
    root = root or schema
    if "$ref" in schema:
        node: Any = root
        for part in schema["$ref"].lstrip("#/").split("/"):
            node = node[part]
        return generate(node, rng, root, depth)
    if "const" in schema:
        return schema["const"]
    if "enum" in schema:
        return rng.choice(schema["enum"])
    for key in ("anyOf", "oneOf"):
        if key in schema:
            return generate(rng.choice(schema[key]), rng, root, depth)
    t = rng.choice(_types(schema))
    if t == "null":
        return None
    if t == "boolean":
        return rng.choice([False, True])
    if t == "integer":
        lo, hi = schema.get("minimum", -(2**63)), schema.get("maximum", 2**63)
        options = [v for v in TRICKY_INTS if lo <= v <= hi] or [int(lo)]
        return rng.choice(options)
    if t == "number":
        lo, hi = schema.get("minimum", -1e300), schema.get("maximum", 1e300)
        options = [v for v in TRICKY_NUMBERS if lo <= v <= hi] or [float(lo)]
        return rng.choice(options)
    if t == "array":
        items = schema.get("items", {"type": "string"})
        n = max(schema.get("minItems", 1), min(schema.get("maxItems", 3), 2))
        return [generate(items, rng, root, depth + 1) for _ in range(n)]
    if t == "object":
        props = schema.get("properties", {})
        required = set(schema.get("required", props))
        return {k: generate(v, rng, root, depth + 1) for k, v in props.items() if k in required or rng.random() < 0.5}
    options = TRICKY_STRINGS
    if "pattern" in schema:
        options = [s for s in TRICKY_STRINGS if re.search(schema["pattern"], s)] or [schema.get("default", "x")]
    lo, hi = schema.get("minLength", 0), schema.get("maxLength", 10_000)
    options = [s for s in options if lo <= len(s) <= hi] or ["x" * max(lo, 1)]
    return rng.choice(options)


def argument_sets(tool: dict[str, Any], n: int = SETS_PER_TOOL, seed: int = 0) -> list[dict[str, Any]]:
    params = tool["function"].get("parameters", {"type": "object", "properties": {}})
    validator = Draft202012Validator(params)
    rng = random.Random(f"{seed}:{tool['function']['name']}")
    sets, seen = [], set()
    for _ in range(n * 20):
        args = generate(params, rng)
        key = json.dumps(args, sort_keys=True, ensure_ascii=False)
        if key in seen or not validator.is_valid(args):
            continue
        seen.add(key)
        sets.append(args)
        if len(sets) == n:
            break
    return sets


class ByoTool(BaseContract):
    def __init__(self, tool: dict[str, Any], idx: int, args: dict[str, Any]) -> None:
        name = tool["function"]["name"]
        super().__init__(id=f"BYO.tool.{name}.{idx}", tier=3, title=f"Your tool {name}, argument set {idx}",
                         hypothesis="A forced call copies the arguments exactly.", severity=Severity.S1_SILENT,
                         conditions=[Condition(thinking=None, variant=f"{name}#{idx}")])
        self.tool, self.args, self.tool_name = tool, args, name

    def build_request(self, model: str, cond: Condition, trial: int) -> dict[str, Any]:
        body = chat_request(model, tool_prompt(self.tool_name, self.args), cond)
        body["tools"] = [self.tool]
        body["tool_choice"] = forced_tool_choice(self.tool_name)
        return body

    def check(self, req: dict[str, Any], resp: ParsedResponse) -> OracleResult:
        gate = http_gate(resp)
        if gate:
            return gate
        f = Findings()
        check_tool_call(f, resp, self.tool_name, self.args, self.tool["function"].get("parameters", {}))
        check_no_think_tags(f, resp)
        return f.result()


class ByoSchema(BaseContract):
    def __init__(self, name: str, schema: dict[str, Any]) -> None:
        super().__init__(id=f"BYO.schema.{name}", tier=5, title=f"Your schema {name}",
                         hypothesis="Structured output validates against the schema.", severity=Severity.S2_LOUD,
                         conditions=[Condition(thinking=None, variant=name)])
        self.name, self.schema = name, schema

    def build_request(self, model: str, cond: Condition, trial: int) -> dict[str, Any]:
        body = chat_request(model, SCHEMA_PROMPT, cond)
        body["response_format"] = {"type": "json_schema", "json_schema": {"name": re.sub(r"[^A-Za-z0-9_-]", "_", self.name),
                                                                          "schema": self.schema, "strict": True}}
        return body

    def check(self, req: dict[str, Any], resp: ParsedResponse) -> OracleResult:
        gate = http_gate(resp, docs_shape=True)
        if gate:
            return gate
        f = Findings()
        check_json_content(f, resp, self.schema)
        return f.result()


def load_tools(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    tools = data.get("tools", data) if isinstance(data, dict) else data
    out = []
    for t in tools:
        t = t if "function" in t else {"type": "function", "function": t}
        out.append(copy.deepcopy(t))
    return out


def load_schemas(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return {d["name"]: d["schema"] for d in data}
    return data


def byo_contracts(tools: list[dict[str, Any]], schemas: dict[str, dict[str, Any]]) -> list[BaseContract]:
    contracts: list[BaseContract] = []
    for tool in tools:
        contracts += [ByoTool(tool, i, args) for i, args in enumerate(argument_sets(tool))]
    contracts += [ByoSchema(name, schema) for name, schema in schemas.items()]
    return contracts


def reliability_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """(tool or schema, model) -> pooled failures over its argument sets."""
    pooled: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0, 0])
    for r in records:
        if not r.get("final", True) or r["outcome"] == "error":
            continue
        parts = r["contract_id"].split(".")
        name = f"{parts[1]} {parts[2]}"
        key = (name, r["model"])
        pooled[key][0] += 1
        pooled[key][1] += r["outcome"] == "fail"
    rows = []
    for (name, model), (n, k, _) in sorted(pooled.items()):
        lo, hi = wilson_ci(k, n)
        rows.append({"target": name, "model": model, "n": n, "k": k, "rate": k / n if n else 0, "lo": lo, "hi": hi})
    return rows


async def check(settings: Settings, console: Console, tools_path: Path | None, schemas_path: Path | None, *,
                models_spec: str = "auto", n: int = 50, yes: bool = False,
                transport: httpx.AsyncBaseTransport | None = None) -> int:
    tools = load_tools(tools_path) if tools_path else []
    schemas = load_schemas(schemas_path) if schemas_path else {}
    contracts = byo_contracts(tools, schemas)
    if not contracts:
        console.print("[red]Pass --tools and/or --schemas.[/red]")
        return 2
    for t in tools:
        sets = [c for c in contracts if isinstance(c, ByoTool) and c.tool_name == t["function"]["name"]]
        console.print(f"tool {t['function']['name']}: {len(sets)} argument sets, e.g. {json.dumps(sets[0].args, ensure_ascii=False)[:160] if sets else '(none valid)'}")
    async with ApiClient.for_provider(settings, "tokenfactory", transport) as tf:
        listing = await tf.get("models")
        if listing.status != 200:
            console.print(f"[red]GET /v1/models failed: {listing.status}[/red]")
            return 1
        models = resolve_models(discover(listing.body, settings), models_spec)
        items = _rounds([WorkItem(Cell(c.id, m.id, cond), t) for m in models for c in contracts for cond in c.conditions
                         for t in range(n)])
        contract_map = {c.id: c for c in contracts}
        est = TokenEstimates()
        rows = estimate_cost(items, {m.id: m for m in models}, contract_map, est)
        console.print(cost_table("check", rows, est))
        if not yes:
            console.print("[yellow]Paid stage: nothing was sent. Re-run with --yes once this estimate is approved.[/yellow]")
            return 0
        store = RunStore.create(settings.runs_dir, "check")
        store.write_plan({"stage": "check", "created": utc_now(), "git": git_info(settings.root),
                          "tools": tools, "schemas": schemas, "n": n, "models": [m.id for m in models],
                          "argument_sets": {c.id: c.args for c in contracts if isinstance(c, ByoTool)}})
        runner = Runner(stage="check", store=store, items=items, contracts=contract_map, models={m.id: m for m in models},
                        client=tf, pacer=Pacer(settings.defaults["rpm"]),
                        budget=BudgetGuard(settings.budget_usd, settings.ledger_path), estimates=est,
                        concurrency=settings.defaults["concurrency"], log=lambda line: console.print(line, markup=False))
        code = await runner.run()
    table = Table(title="Your tools and schemas: reliability per model")
    for col in ("Tool / schema", "Model", "Failed", "Rate (95% CI)"):
        table.add_column(col)
    result_rows = reliability_rows(list(analysis.iter_records([store.dir])))
    for r in result_rows:
        table.add_row(r["target"], r["model"], f"{r['k']}/{r['n']}", f"{r['rate']:.1%} ({r['lo']:.1%}-{r['hi']:.1%})")
    console.print(table)
    store.write_summary({"run_id": store.run_id, "finished": utc_now(), "rows": result_rows})
    return code
