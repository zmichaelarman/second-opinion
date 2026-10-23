"""`second-opinion smoke`: at most 15 completion calls that check the plumbing end to end and
settle the UNVERIFIED facts in CLAUDE.md §3 (model IDs, the thinking switch, the reasoning field).

It is exempt from the --yes rule because it is capped at 15 calls (§0.2).

Per runnable model (at most 4): banana with thinking off, banana with thinking on, and one
forced `leading_zero` tool call with thinking on and streaming (which also records SSE
fixtures). On the cheapest model: thinking off through the two alternative switches, and one
call through the openai SDK to see which fields it exposes.
"""

from __future__ import annotations

import asyncio
import functools
import time

import httpx
from dataclasses import asdict, dataclass
from typing import Any, Callable

from rich.console import Console
from rich.table import Table

from .budget import BudgetGuard, call_cost
from .client import ApiClient, Exchange, keep_headers
from .config import Settings, api_key
from .contracts.base import Condition, OracleResult, cell_id
from .contracts.t2_thinking_switch import banana_request, check_banana
from .contracts.t3_tool_args import check_t3, tool_request
from .discovery import DiscoveredModel, discover, models_table, runnable
from .parse import ParsedResponse, parse_exchange
from .store import RunStore, call_record, git_info, utc_now

MAX_CALLS = 15
EST_TOKENS = {False: (400, 300), True: (400, 2000), None: (400, 2000)}  # §8 fallbacks
ROLE_ORDER = ("nano", "lightning", "super", "ultra")


@dataclass
class Step:
    name: str
    model: DiscoveredModel
    cond: Condition
    body: dict[str, Any]
    check: Callable[[dict[str, Any], ParsedResponse], OracleResult]
    via: str = "httpx"

    @property
    def estimate(self) -> float:
        assert self.model.price is not None
        return self.model.price.cost(*EST_TOKENS[self.cond.thinking])


def pick_models(models: list[DiscoveredModel], limit: int = 4) -> list[DiscoveredModel]:
    """Runnable models, one per role first, capped so the smoke stays within 15 calls."""
    pool = runnable(models)
    picked: list[DiscoveredModel] = []
    for role in ROLE_ORDER:
        match = next((m for m in pool if m.role == role and m not in picked), None)
        if match:
            picked.append(match)
    for m in sorted(pool, key=lambda m: (m.price.output, m.id)):  # type: ignore[union-attr]
        if m not in picked:
            picked.append(m)
    return picked[:limit]


def plan_steps(models: list[DiscoveredModel]) -> list[Step]:
    steps: list[Step] = []
    for m in models:
        for thinking in (False, True):
            cond = Condition(thinking=thinking)
            steps.append(
                Step(f"banana_think_{'on' if thinking else 'off'}", m, cond, banana_request(m.id, cond),
                     functools.partial(check_banana, thinking))
            )
        cond = Condition(thinking=True, stream=True, variant="leading_zero")
        steps.append(
            Step("tool_leading_zero_stream", m, cond, tool_request(m.id, "leading_zero", cond),
                 functools.partial(check_t3, "leading_zero"))
        )
    cheap = min(models, key=lambda m: (m.price.output, m.price.input, m.id))  # type: ignore[union-attr]
    for switch in ("extra_body", "reasoning_effort"):
        cond = Condition(thinking=False, variant=f"switch={switch}")
        steps.append(
            Step(f"banana_off_via_{switch}", cheap, cond, banana_request(cheap.id, cond, switch=switch),
                 functools.partial(check_banana, False))
        )
    cond = Condition(thinking=True, variant="leading_zero")
    steps.append(
        Step("sdk_tool_leading_zero", cheap, cond, tool_request(cheap.id, "leading_zero", cond),
             functools.partial(check_t3, "leading_zero"), via="openai-sdk")
    )
    if len(steps) > MAX_CALLS:
        raise RuntimeError(f"smoke plan has {len(steps)} calls; the cap is {MAX_CALLS}")
    return steps


async def _sdk_exchange(
    step: Step, settings: Settings, transport: httpx.AsyncBaseTransport | None = None
) -> tuple[Exchange, dict[str, Any]]:
    """Send the step through the openai SDK and report which fields the SDK exposes."""
    import openai
    from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI

    p = settings.provider("tokenfactory")
    kwargs = dict(step.body)
    extra_body = {"chat_template_kwargs": kwargs.pop("chat_template_kwargs")} if "chat_template_kwargs" in kwargs else None
    view: dict[str, Any] = {"sdk_version": openai.__version__}
    t0 = time.monotonic()
    async with AsyncOpenAI(
        base_url=p["base_url"], api_key=api_key(p["api_key_env"]), max_retries=0,
        timeout=settings.defaults["read_timeout_s"],
        http_client=httpx.AsyncClient(transport=transport) if transport else None,
    ) as sdk:
        try:
            raw = await sdk.chat.completions.with_raw_response.create(**kwargs, extra_body=extra_body)
        except APIStatusError as e:
            return Exchange(e.status_code, keep_headers(e.response.headers), _body(e.response), False,
                            total_s=time.monotonic() - t0), view
        except APITimeoutError:
            return Exchange(None, {}, None, False, "TIMEOUT", total_s=time.monotonic() - t0), view
        except APIConnectionError as e:
            return Exchange(None, {}, None, False, type(e).__name__, total_s=time.monotonic() - t0), view
    http = raw.http_response
    completion = raw.parse()
    choice = completion.choices[0] if completion.choices else None
    message = choice.message if choice else None
    view.update(
        message_declared_fields=sorted(type(message).model_fields) if message else [],
        message_model_extra_keys=sorted((message.model_extra or {}) if message else {}),
        choice_model_extra_keys=sorted((choice.model_extra or {}) if choice else {}),
        completion_model_extra_keys=sorted(completion.model_extra or {}),
        tool_calls_exposed=bool(message and message.tool_calls),
    )
    return Exchange(http.status_code, keep_headers(http.headers), _body(http), False,
                    total_s=time.monotonic() - t0), view


def _body(response: Any) -> Any:
    try:
        return response.json()
    except ValueError:
        return response.text


async def run_smoke(settings: Settings, console: Console, transport: httpx.AsyncBaseTransport | None = None) -> int:
    budget = BudgetGuard(settings.budget_usd, settings.ledger_path)
    async with ApiClient.for_provider(settings, "tokenfactory", transport) as tf:
        listing = await tf.get("models")
        if listing.status != 200:
            console.print(f"[red]GET /v1/models failed[/red]: status={listing.status} error={listing.error} "
                          f"body={str(listing.body)[:300]}")
            return 1
        models = discover(listing.body, settings)
        console.print(models_table(models))
        chosen = pick_models(models)
        if not chosen:
            console.print("[red]No Nemotron model has a price row in config/prices.yaml, so nothing can run.[/red]")
            return 1
        steps = plan_steps(chosen)
        total = sum(s.estimate for s in steps)
        console.print(f"Smoke plan: {len(steps)} completion calls on {len(chosen)} models, "
                      f"estimated ${total:.4f} (cap {MAX_CALLS} calls; exempt from --yes).")
        if budget.spent + total > budget.cap:
            console.print(f"[red]Budget guard: ${budget.spent:.2f} spent of ${budget.cap:.2f}; not running.[/red]")
            return 3

        store = RunStore.create(settings.runs_dir, "smoke")
        store.write_plan({
            "stage": "smoke",
            "created": utc_now(),
            "git": git_info(settings.root),
            "models_listed": [m.id for m in models],
            "models_tested": [m.id for m in chosen],
            "prices": {m.id: asdict(m.price) for m in chosen if m.price},
            "steps": [{"name": s.name, "model": s.model.id, "condition": s.cond.as_dict(), "via": s.via,
                       "request": s.body, "estimate_usd": round(s.estimate, 6)} for s in steps],
            "models_raw": listing.body,
        })
        spacing = 60.0 / settings.defaults["rpm"]
        records: list[dict[str, Any]] = []
        for i, step in enumerate(steps):
            if i:
                await asyncio.sleep(spacing)
            ts = utc_now()
            extra: dict[str, Any] = {"via": step.via}
            if not budget.try_reserve(step.estimate):
                console.print("[red]Budget cap reached; stopping the smoke early.[/red]")
                break
            if step.via == "openai-sdk":
                ex, view = await _sdk_exchange(step, settings, transport)
                extra["sdk_view"] = view
            else:
                ex = await tf.chat(step.body)
            parsed = parse_exchange(ex)
            assert step.model.price is not None
            cost = call_cost(step.model.price, parsed.usage, parsed.status, step.estimate)
            budget.settle(step.estimate, cost, {"ts": ts, "run_id": store.run_id, "stage": "smoke",
                                                "model": step.model.id, "contract_id": step.name, "usage": parsed.usage})
            result = step.check(step.body, parsed)
            record = call_record(
                run_id=store.run_id, stage="smoke", cell_id=cell_id(f"smoke.{step.name}", step.model.id, step.cond),
                contract_id=f"smoke.{step.name}", model=step.model.id, condition=step.cond.as_dict(), trial=0,
                attempt=1, ts_start=ts, request=step.body, parsed=parsed, result=result, cost_usd=cost, **extra,
            )
            store.append_call(record)
            records.append(record)
            console.print(_line(i + 1, len(steps), record))
    summary = summarize(records, chosen)
    store.write_summary(summary)
    _print_summary(console, summary, store)
    return 0


def _line(i: int, n: int, r: dict[str, Any]) -> str:
    p = r["parsed"]
    color = {"pass": "green", "fail": "red", "error": "yellow"}[r["outcome"]]
    return (f"[{i:>2}/{n}] {r['contract_id']:<34} {r['model']:<42} [{color}]{r['outcome']}[/{color}] "
            f"{','.join(r['codes']) or '-'}  status={r['status']} reasoning_field={p['reasoning_field']} "
            f"completion_tokens={(r['usage'] or {}).get('completion_tokens')}")


def _brief(r: dict[str, Any] | None) -> dict[str, Any] | None:
    if r is None:
        return None
    p = r["parsed"]
    return {
        "outcome": r["outcome"],
        "codes": r["codes"],
        "status": r["status"],
        "error_message": p["error_message"],
        "content_head": (p["content"] or "")[:200],
        "reasoning_field": p["reasoning_field"],
        "reasoning_fields_seen": p["reasoning_fields_seen"],
        "reasoning_chars": len(p["reasoning"] or ""),
        "completion_tokens": (r["usage"] or {}).get("completion_tokens"),
        "finish_reason": p["finish_reason"],
        "tool_calls": p["tool_calls"],
    }


def switch_effect(r: dict[str, Any] | None) -> str:
    """What a thinking-off request did: works, ignored, rejected or error."""
    if r is None:
        return "not run"
    if r["status"] is not None and 400 <= r["status"] < 500:
        return f"rejected (HTTP {r['status']}: {(r['parsed']['error_message'] or '')[:120]})"
    if r["outcome"] == "error":
        return f"error ({','.join(r['codes'])})"
    return "ignored" if "THINKING_NOT_DISABLED" in r["codes"] else "works"


def summarize(records: list[dict[str, Any]], chosen: list[DiscoveredModel]) -> dict[str, Any]:
    by = {(r["model"], r["contract_id"]): r for r in records}
    models = []
    for m in chosen:
        off, on = by.get((m.id, "smoke.banana_think_off")), by.get((m.id, "smoke.banana_think_on"))
        tool = by.get((m.id, "smoke.tool_leading_zero_stream"))
        models.append({
            "model": m.id,
            "role": m.role,
            "thinking_off_via_chat_template_kwargs": switch_effect(off),
            "reasoning_field_non_stream": on["parsed"]["reasoning_field"] if on else None,
            "reasoning_field_stream": tool["parsed"]["reasoning_field"] if tool else None,
            "banana_think_off": _brief(off),
            "banana_think_on": _brief(on),
            "tool_leading_zero_stream": _brief(tool),
        })
    probes = {r["contract_id"]: r for r in records if r["contract_id"].startswith("smoke.banana_off_via_")}
    sdk = next((r for r in records if r.get("via") == "openai-sdk"), None)
    return {
        "models": models,
        "thinking_off_probes": {
            "model": next(iter(probes.values()))["model"] if probes else None,
            "extra_body": switch_effect(probes.get("smoke.banana_off_via_extra_body")),
            "reasoning_effort_none": switch_effect(probes.get("smoke.banana_off_via_reasoning_effort")),
            "detail": {k: _brief(v) for k, v in probes.items()},
        },
        "sdk_view": {**(sdk.get("sdk_view") or {}), "call": _brief(sdk)} if sdk else None,
        "calls": len(records),
        "cost_usd": round(sum(r["cost_usd"] for r in records), 6),
    }


def _print_summary(console: Console, s: dict[str, Any], store: RunStore) -> None:
    table = Table(title="Smoke summary")
    for col in ("Model", "Thinking off (top-level chat_template_kwargs)", "Reasoning field", "Stream field",
                "Forced tool call (stream)"):
        table.add_column(col)
    for m in s["models"]:
        tool = m["tool_leading_zero_stream"] or {}
        table.add_row(m["model"], m["thinking_off_via_chat_template_kwargs"], str(m["reasoning_field_non_stream"]),
                      str(m["reasoning_field_stream"]), f"{tool.get('outcome')} {','.join(tool.get('codes') or [])}")
    console.print(table)
    pr = s["thinking_off_probes"]
    console.print(f"Thinking off on {pr['model']}: extra_body -> {pr['extra_body']}; "
                  f"reasoning_effort=none -> {pr['reasoning_effort_none']}")
    if s["sdk_view"]:
        v = s["sdk_view"]
        console.print(f"openai SDK {v.get('sdk_version')}: message.model_extra keys = {v.get('message_model_extra_keys')}; "
                      f"tool call -> {(v.get('call') or {}).get('outcome')} {(v.get('call') or {}).get('codes')}")
    console.print(f"{s['calls']} calls, ${s['cost_usd']:.4f}. Raw data: {store.dir}")
