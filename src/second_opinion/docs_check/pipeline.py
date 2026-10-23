"""`second-opinion docs` (CLAUDE.md §10).

Without flags it fetches, extracts and statically checks every example (free) and records the
results in runs/<ts>-docs/. --execute also runs eligible Python examples and probes whether
the API rejects or silently ignores unknown parameters; --repair lets the repair agent fix
failing blocks. Both are paid: they print an estimate and send nothing without --yes.
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from typing import Any

import httpx
from rich.console import Console
from rich.table import Table

from ..budget import BudgetGuard, call_cost
from ..client import ApiClient
from ..config import ConfigError, Settings, api_key
from ..contracts.base import THINKING_SWITCH, Condition, OracleResult, Outcome, Severity, apply_thinking, cell_id
from ..discovery import discover
from ..parse import parse_exchange
from ..planner import cheapest
from ..store import RunStore, call_record, git_info, utc_now
from .execute import EST_COST_PER_RUN_USD, Execution, eligible, run_snippet
from .extract import CodeBlock, extract_blocks
from .fetch import fetch_docs
from .repair import MAX_ATTEMPTS, repair_block
from .static import StaticResult, check_block, find_node, request_params

REPAIR_EST_TOKENS = (2500, 4000)
PROBE_MAX_TOKENS = 64
PROBE_VALUES: dict[str, Any] = {  # used when the docs' own value can't be parsed out of invalid JSON
    "response format": {"type": "json_object"},
    "guided_json": {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]},
    "top_k": 50,
}


def finding_severity(codes: list[str], probes: dict[str, dict[str, Any]]) -> Severity:
    """Docs findings are S4, except an unknown parameter the API silently ignores (S2)."""
    for code in codes:
        if code.startswith("UNKNOWN_PARAM:") and probes.get(code.split(":", 1)[1], {}).get("verdict") == "silently ignored":
            return Severity.S2_LOUD
    return Severity.S4_DOCS


def probe_value(key: str, body_text: str | None) -> Any:
    if body_text:
        try:
            parsed = json.loads(body_text)
            if isinstance(parsed, dict) and key in parsed:
                return parsed[key]
        except ValueError:
            pass
    return PROBE_VALUES.get(key)


def probe_request(model: str, key: str, value: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with the single word: hello"}],
        "max_tokens": PROBE_MAX_TOKENS,
        "temperature": 1.0,
        "top_p": 0.95,
    }
    apply_thinking(body, False, THINKING_SWITCH)  # the verified switch, so only `key` is under test
    body[key] = value
    return body


def probe_verdict(key: str, value: Any, status: int | None, content: str | None) -> str:
    if status is None:
        return "error"
    if 400 <= status < 500:
        return "rejected"
    if status != 200:
        return "error"
    asks_json = isinstance(value, dict) and (value.get("type") in ("json_object", "object") or "properties" in value)
    if asks_json:
        try:
            json.loads(content or "")
            return "accepted and honored"
        except ValueError:
            return "silently ignored"
    return "accepted (effect not verifiable)"


def static_table(rows: list[tuple[Any, CodeBlock, StaticResult]]) -> Table:
    table = Table(title="Docs examples: static checks")
    for col in ("Page", "Block", "Line", "Kind", "Codes"):
        table.add_column(col)
    for page, block, static in rows:
        if static.codes:
            table.add_row(page.url.split(".com/", 1)[-1], str(block.block_idx), str(block.line), static.kind,
                          ", ".join(static.codes))
    return table


async def run_docs(
    settings: Settings,
    console: Console,
    *,
    execute: bool = False,
    repair: bool = False,
    yes: bool = False,
    transport: httpx.AsyncBaseTransport | None = None,
    docs_transport: httpx.AsyncBaseTransport | None = None,
    executor=run_snippet,
) -> int:
    cache, pages, openapi = await fetch_docs(settings.runs_dir, transport=docs_transport)
    params = request_params(openapi)
    node = find_node()
    async with ApiClient.for_provider(settings, "tokenfactory", transport) as tf:
        listed: list[str] | None = None
        found = []
        note = None
        try:
            listing = await tf.get("models")
            if listing.status == 200:
                listed = sorted(m.get("id") for m in listing.body.get("data", []) if isinstance(m, dict))
                found = discover(listing.body, settings)
            else:
                note = f"GET /v1/models returned {listing.status}; model IDs not checked"
        except ConfigError:
            note = "NEBIUS_API_KEY not set; model IDs not checked"
        if note:
            console.print(f"[yellow]{note}[/yellow]")
        rows = [(page, block, check_block(block, params, set(listed) if listed is not None else None, node))
                for page in pages for block in extract_blocks(page.url, page.text)]
        kinds = Counter(s.kind for _, _, s in rows)
        failing = [r for r in rows if r[2].codes]
        runnable = [r for r in rows if eligible(r[1], r[2])]
        console.print(static_table(rows))
        console.print(f"{len(pages)} pages, {len(rows)} code blocks ({dict(kinds)}); {len(failing)} fail static checks; "
                      f"{len(runnable)} eligible to execute. Pages cached in {cache}.")

        chat_unknown = sorted({c.split(":", 1)[1] for _, _, s in failing for c in s.codes
                               if c.startswith("UNKNOWN_PARAM:") and s.endpoint == "/v1/chat/completions"})
        nemotron = [m for m in found if m.runnable]
        sup = next((m for m in nemotron if m.role == "super"), None)
        paid = execute or repair
        if paid:
            if listed is None or not nemotron:
                console.print("[red]--execute/--repair need NEBIUS_API_KEY and a priced Nemotron model.[/red]")
                return 2
            low = cheapest(nemotron)
            exec_usd = len(runnable) * EST_COST_PER_RUN_USD if execute else 0.0
            probe_usd = len(chat_unknown) * low.price.cost(60, PROBE_MAX_TOKENS) if execute else 0.0
            repair_model = sup or low
            n_repair = len(failing) + (len(runnable) if execute else 0)  # upper bound: executions may fail too
            repair_usd = n_repair * MAX_ATTEMPTS * repair_model.price.cost(*REPAIR_EST_TOKENS) if repair else 0.0
            console.print(
                "Paid plan: "
                + (f"execute {len(runnable)} examples (~${exec_usd:.3f}); probe {len(chat_unknown)} unknown "
                   f"parameter(s) on {low.id} (~${probe_usd:.4f}); " if execute else "")
                + (f"repair up to {n_repair} blocks x {MAX_ATTEMPTS} attempts with {repair_model.id} "
                   f"(at most ~${repair_usd:.2f}); " if repair else "")
                + f"total at most ~${exec_usd + probe_usd + repair_usd:.2f}."
            )
            if not yes:
                console.print("[yellow]Paid step: nothing was sent. Re-run with --yes once this estimate is approved.[/yellow]")
                return 0

        store = RunStore.create(settings.runs_dir, "docs")
        store.write_plan({
            "stage": "docs", "created": utc_now(), "git": git_info(settings.root), "cache": str(cache),
            "pages": [p.url for p in pages], "execute": execute, "repair": repair, "node": bool(node),
            "models_listed": listed,
        })
        budget = BudgetGuard(settings.budget_usd, settings.ledger_path)
        spacing = 60.0 / settings.defaults["rpm"]
        with (store.dir / "blocks.jsonl").open("a") as f:
            for page, block, static in rows:
                f.write(json.dumps({"page_title": page.title, **block.as_dict(), "static": static.as_dict(),
                                    "eligible": eligible(block, static)}, ensure_ascii=False) + "\n")

        executions: dict[tuple[str, int], Execution] = {}
        probes: dict[str, dict[str, Any]] = {}
        if execute:
            key = api_key(settings.provider("tokenfactory")["api_key_env"])
            for page, block, static in runnable:
                if not budget.try_reserve(EST_COST_PER_RUN_USD):
                    console.print("[red]Budget cap reached; stopping executions.[/red]")
                    break
                run = await asyncio.to_thread(executor, block.code, key)
                executions[(block.page_url, block.block_idx)] = run
                budget.settle(EST_COST_PER_RUN_USD, EST_COST_PER_RUN_USD, {"ts": utc_now(), "run_id": store.run_id, "stage": "docs",
                                                         "note": "execute (flat estimate)", "page": block.page_url,
                                                         "block_idx": block.block_idx})
                with (store.dir / "executions.jsonl").open("a") as f:
                    f.write(json.dumps({"page_url": block.page_url, "block_idx": block.block_idx, **run.as_dict()},
                                       ensure_ascii=False) + "\n")
                console.print(f"  executed {page.url.split('.com/', 1)[-1]}#{block.block_idx}: "
                              + ("ok" if run.ok else f"{run.code} {run.error}"))
                await asyncio.sleep(spacing)
            bodies = {c.split(":", 1)[1]: (b, s) for _, b, s in failing for c in s.codes if c.startswith("UNKNOWN_PARAM:")}
            low = cheapest(nemotron)
            for name in chat_unknown:
                block, static = bodies[name]
                value = probe_value(name, _body_text(block))
                if value is None:
                    probes[name] = {"verdict": "not probed (no usable value)"}
                    continue
                body = probe_request(low.id, name, value)
                est = low.price.cost(60, PROBE_MAX_TOKENS)
                if not budget.try_reserve(est):
                    console.print("[red]Budget cap reached; stopping probes.[/red]")
                    break
                ts = utc_now()
                parsed = parse_exchange(await tf.chat(body))
                cost = call_cost(low.price, parsed.usage, parsed.status, est)
                budget.settle(est, cost, {"ts": ts, "run_id": store.run_id, "stage": "docs", "contract_id": "docs.probe",
                                          "model": low.id, "usage": parsed.usage})
                verdict = probe_verdict(name, value, parsed.status, parsed.content)
                probes[name] = {"verdict": verdict, "status": parsed.status, "model": low.id, "value": value,
                                "error_message": parsed.error_message, "content_head": (parsed.content or "")[:200]}
                result = OracleResult(Outcome.PASS, [], {"probe": probes[name]})
                store.append_call(call_record(
                    run_id=store.run_id, stage="docs", cell_id=cell_id(f"docs.probe.{name}", low.id, Condition(False)),
                    contract_id=f"docs.probe.{name}", model=low.id, condition=None, trial=0, attempt=1, ts_start=ts,
                    request=body, parsed=parsed, result=result, cost_usd=cost,
                ))
                console.print(f"  probe {name!r}: HTTP {parsed.status} -> {verdict}")
                await asyncio.sleep(spacing)
            if any("CURL_AUTH_HEADER" in s.codes for _, _, s in failing):
                probes["auth_without_bearer"] = await _auth_probe(settings, transport)
                console.print(f"  probe Authorization without Bearer: {probes['auth_without_bearer']}")

        to_repair: list[tuple[Any, CodeBlock, StaticResult, str]] = []
        for page, block, static in rows:
            run = executions.get((block.page_url, block.block_idx))
            if static.codes:
                to_repair.append((page, block, static, "; ".join(static.errors)))
            elif run is not None and not run.ok:
                to_repair.append((page, block, static, f"{run.code}: {run.error}\n{run.stderr_tail[-1500:]}"))

        repairs = {}
        if repair:
            model = (sup or cheapest(nemotron))
            for page, block, static, error in to_repair:
                async def complete(body: dict[str, Any], attempt: int, _block=block) -> str | None:
                    est = model.price.cost(*REPAIR_EST_TOKENS)
                    if not budget.try_reserve(est):
                        return None
                    ts = utc_now()
                    parsed = parse_exchange(await tf.chat(body))
                    cost = call_cost(model.price, parsed.usage, parsed.status, est)
                    budget.settle(est, cost, {"ts": ts, "run_id": store.run_id, "stage": "docs",
                                              "contract_id": "docs.repair", "model": model.id, "usage": parsed.usage})
                    store.append_call(call_record(
                        run_id=store.run_id, stage="docs",
                        cell_id=cell_id(f"docs.repair.{_block.page_url}#{_block.block_idx}", model.id, Condition(True)),
                        contract_id="docs.repair", model=model.id, condition=None, trial=0, attempt=attempt,
                        ts_start=ts, request=body, parsed=parsed,
                        result=OracleResult(Outcome.PASS if parsed.status == 200 else Outcome.ERROR, [], {}),
                        cost_usd=cost, page_url=_block.page_url, block_idx=_block.block_idx,
                    ))
                    await asyncio.sleep(spacing)
                    return parsed.content

                key = api_key(settings.provider("tokenfactory")["api_key_env"]) if execute else None
                result = await repair_block(
                    block, page.title, error, listed or [], model=model.id, complete=complete,
                    recheck=lambda b: check_block(b, params, set(listed or []), node),
                    execute=(lambda code: run_snippet(code, key)) if key else None,
                )
                repairs[(block.page_url, block.block_idx)] = result
                console.print(f"  repair {page.url.split('.com/', 1)[-1]}#{block.block_idx}: "
                              + (f"fixed in {result.attempts}" if result.fixed else f"not fixed ({result.last_error})"))

    findings = []
    for page, block, static, error in to_repair:
        run = executions.get((block.page_url, block.block_idx))
        rep = repairs.get((block.page_url, block.block_idx))
        codes = static.codes + ([run.code] if run and not run.ok and run.code else [])
        findings.append({
            "page_url": block.page_url, "page_title": page.title, "block_idx": block.block_idx, "lang": block.lang,
            "line": block.line, "codes": codes, "error": error, "severity": finding_severity(codes, probes).name,
            "owner": "nebius_docs", "probes": {c.split(":", 1)[1]: probes[c.split(":", 1)[1]] for c in codes
                                              if c.startswith("UNKNOWN_PARAM:") and c.split(":", 1)[1] in probes},
            "fixed": rep.fixed if rep else None, "attempts": rep.attempts if rep else 0,
            "patch": rep.patch if rep else None,
        })
    with (store.dir / "docs_findings.jsonl").open("a") as f:
        for row in findings:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    executed_ok = sum(1 for r in executions.values() if r.ok)
    checked_kinds = ("python", "curl", "js") if node else ("python", "curl")  # JS needs node --check
    examples = [(b, st) for _, b, st in rows if st.kind in checked_kinds]

    def passes(block: CodeBlock, static: StaticResult) -> bool:
        run = executions.get((block.page_url, block.block_idx))
        return not static.codes and (run is None or run.ok)
    summary = {
        "run_id": store.run_id, "finished": utc_now(), "pages": len(pages), "blocks": len(rows), "kinds": dict(kinds),
        "static_failures": len(failing), "eligible": len(runnable), "executed": len(executions),
        "executed_ok": executed_ok, "findings": len(findings),
        "repaired": sum(1 for r in repairs.values() if r.fixed), "repair_attempted": len(repairs),
        "probes": probes, "codes": dict(Counter(c for row in findings for c in row["codes"])),
        "examples_checked": len(examples),
        "examples_passing": sum(1 for b, st in examples if passes(b, st)),
        "models_checked": listed is not None,
        "js_checked": bool(node),
        "js_unchecked": 0 if node else kinds.get("js", 0),
    }
    store.write_summary(summary)
    console.print(f"{summary['findings']} docs findings; {summary['examples_passing']} of {summary['examples_checked']} "
                  f"checked code examples pass; {summary['repaired']} repaired"
                  + (f"; {summary['js_unchecked']} JS examples unchecked (node not on PATH)" if summary["js_unchecked"] else "")
                  + f". Results: {store.dir}")
    return 0


def _body_text(block: CodeBlock) -> str | None:
    from .static import curl_body

    return curl_body(block.code)


async def _auth_probe(settings: Settings, transport: httpx.AsyncBaseTransport | None) -> dict[str, Any]:
    """GET /v1/models (free) with the raw key and no Bearer scheme, as one docs example does."""
    base = settings.provider("tokenfactory")["base_url"]
    key = api_key(settings.provider("tokenfactory")["api_key_env"])
    async with httpx.AsyncClient(base_url=base, timeout=30, transport=transport) as http:
        resp = await http.get("models", headers={"Authorization": key})
    return {"status": resp.status_code, "verdict": "accepted" if resp.status_code == 200 else "rejected"}
