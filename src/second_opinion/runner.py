from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from rich.console import Console
from rich.table import Table

from second_opinion.budget import BudgetExceededError, BudgetGuard
from second_opinion.client import InferenceClient
from second_opinion.config import AppConfig, get_nebius_api_key
from second_opinion.contracts.base import Condition, compute_cell_id
from second_opinion.contracts.registry import get_contract_map
from second_opinion.discovery import discover_nemotron_models
from second_opinion.parse import ParsedResponse
from second_opinion.planner import PlannedCell, load_screening_fatal_cells
from second_opinion.pacer import HeaderAwarePacer
from second_opinion.store import CallRecord, RunStore


async def run_smoke(
    config: AppConfig,
    console: Console | None = None,
    runs_root: Path | None = None,
) -> dict[str, Any]:
    """Execute M0 smoke test (<=15 calls):
    1. List models.
    2. For each Nemotron model: 1 banana call with thinking=False and 1 with thinking=True.
    3. For each Nemotron model: 1 forced leading_zero tool call.
    4. 1 call through the openai SDK, printing model_extra keys.
    """
    console = console or Console()
    api_key = get_nebius_api_key()
    store = RunStore.create_or_resume("smoke", runs_root=runs_root)
    budget = BudgetGuard(config)
    pacer = HeaderAwarePacer(base_rpm=config.rpm)

    eligible_models, all_models = await discover_nemotron_models(config, api_key=api_key)
    models = eligible_models[:4]  # Keep total calls <= 15 (1 + 4*2 + 4*1 + 1 = 14 calls)

    store.write_plan(
        {
            "models": models,
            "all_models_count": len(all_models),
            "rpm": config.rpm,
        }
    )

    contracts = get_contract_map()
    t2 = contracts["T2"]
    t3 = contracts["T3"]

    smoke_rows: list[dict[str, Any]] = []

    async with InferenceClient(
        base_url=config.nebius_base_url,
        api_key=api_key,
        pacer=pacer,
        connect_timeout_s=config.connect_timeout_s,
        read_timeout_s=config.read_timeout_s,
    ) as client:
        for model in models:
            # 2a. Banana thinking=False and thinking=True
            for think_flag, variant in ((False, "off"), (True, "on")):
                cond = Condition(thinking=think_flag, stream=False, variant=variant)
                req = t2.build_request(model, cond, trial=0)
                cell_id = compute_cell_id("T2", model, cond)
                est_cost = config.estimate_cost(model, 200, 200 if not think_flag else 1000)
                await budget.reserve(est_cost)

                attempts_log: list[tuple[int, str, ParsedResponse, bool]] = []

                def _on_attempt(
                    att: int, ts_s: str, p_resp: ParsedResponse, is_fin: bool
                ) -> None:
                    attempts_log.append((att, ts_s, p_resp, is_fin))

                resp = await client.chat_completion(
                    req,
                    estimated_tokens=500,
                    on_attempt=_on_attempt,
                )
                ores = t2.check(req, resp)
                actual_cost = budget.compute_call_cost(model, resp.usage)
                await budget.commit(est_cost, actual_cost, model, "smoke", store.run_id, resp.usage)

                for att, ts_s, p_resp, is_fin in attempts_log:
                    att_ores = ores if is_fin else t2.check(req, p_resp)
                    rec = CallRecord(
                        run_id=store.run_id,
                        stage="smoke",
                        cell_id=cell_id,
                        contract_id="T2",
                        model=model,
                        condition=cond.to_dict(),
                        trial=0,
                        attempt=att,
                        ts_start=ts_s,
                        request=req,
                        status=p_resp.status,
                        headers=p_resp.headers,
                        raw=p_resp.raw,
                        parsed=p_resp.to_dict(),
                        outcome=att_ores.outcome.value,
                        codes=att_ores.codes,
                        detail=att_ores.detail,
                        usage=p_resp.usage,
                        cost_usd=actual_cost if is_fin else 0.0,
                        is_final=is_fin,
                    )
                    store.append_call(rec)

                smoke_rows.append(
                    {
                        "model": model,
                        "check": f"banana (thinking={think_flag})",
                        "status": resp.status,
                        "outcome": ores.outcome.value,
                        "codes": ores.codes,
                        "reasoning_field": resp.reasoning_field,
                        "reasoning_len": len(resp.reasoning),
                        "content": resp.content[:80],
                    }
                )

            # 3. Forced leading_zero tool call
            cond_t3 = Condition(thinking=True, stream=False, variant="leading_zero")
            req_t3 = t3.build_request(model, cond_t3, trial=0)
            cell_id_t3 = compute_cell_id("T3", model, cond_t3)
            est_cost_t3 = config.estimate_cost(model, 400, 1000)
            await budget.reserve(est_cost_t3)

            attempts_t3: list[tuple[int, str, ParsedResponse, bool]] = []

            def _on_attempt_t3(
                att: int, ts_s: str, p_resp: ParsedResponse, is_fin: bool
            ) -> None:
                attempts_t3.append((att, ts_s, p_resp, is_fin))

            resp_t3 = await client.chat_completion(
                req_t3,
                estimated_tokens=800,
                on_attempt=_on_attempt_t3,
            )
            ores_t3 = t3.check(req_t3, resp_t3)
            actual_cost_t3 = budget.compute_call_cost(model, resp_t3.usage)
            await budget.commit(est_cost_t3, actual_cost_t3, model, "smoke", store.run_id, resp_t3.usage)

            for att, ts_s, p_resp, is_fin in attempts_t3:
                att_ores = ores_t3 if is_fin else t3.check(req_t3, p_resp)
                rec = CallRecord(
                    run_id=store.run_id,
                    stage="smoke",
                    cell_id=cell_id_t3,
                    contract_id="T3",
                    model=model,
                    condition=cond_t3.to_dict(),
                    trial=0,
                    attempt=att,
                    ts_start=ts_s,
                    request=req_t3,
                    status=p_resp.status,
                    headers=p_resp.headers,
                    raw=p_resp.raw,
                    parsed=p_resp.to_dict(),
                    outcome=att_ores.outcome.value,
                    codes=att_ores.codes,
                    detail=att_ores.detail,
                    usage=p_resp.usage,
                    cost_usd=actual_cost_t3 if is_fin else 0.0,
                    is_final=is_fin,
                )
                store.append_call(rec)

            smoke_rows.append(
                {
                    "model": model,
                    "check": "T3 leading_zero (forced tool)",
                    "status": resp_t3.status,
                    "outcome": ores_t3.outcome.value,
                    "codes": ores_t3.codes,
                    "reasoning_field": resp_t3.reasoning_field,
                    "reasoning_len": len(resp_t3.reasoning),
                    "content": (resp_t3.tool_calls[0]["arguments_raw"] if resp_t3.tool_calls else resp_t3.content)[:80],
                }
            )

    # 4. One call through the openai SDK, printing which fields the SDK exposes (model_extra keys)
    sdk_info: dict[str, Any] = {}
    if models:
        sdk_model = models[0]
        oai = AsyncOpenAI(api_key=api_key, base_url=config.nebius_base_url)
        try:
            await pacer.acquire(estimated_tokens=500)
            sdk_resp = await oai.chat.completions.create(
                model=sdk_model,
                messages=[
                    {"role": "system", "content": "You are a component in an automated test."},
                    {"role": "user", "content": "Reply with one word: banana"},
                ],
                temperature=1.0,
                top_p=0.95,
                max_tokens=512,
                extra_body={"chat_template_kwargs": {"enable_thinking": True}},
            )
            msg = sdk_resp.choices[0].message
            resp_extra = list((sdk_resp.model_extra or {}).keys())
            msg_extra = list((msg.model_extra or {}).keys())
            choice_extra = list((sdk_resp.choices[0].model_extra or {}).keys())
            usage_dict = sdk_resp.usage.model_dump() if sdk_resp.usage else None
            sdk_cost = budget.compute_call_cost(sdk_model, usage_dict)
            await budget.commit(0.0, sdk_cost, sdk_model, "smoke_sdk", store.run_id, usage_dict)
            sdk_info = {
                "model": sdk_model,
                "response_model_extra_keys": resp_extra,
                "choice_model_extra_keys": choice_extra,
                "message_model_extra_keys": msg_extra,
                "has_reasoning_content_attr": hasattr(msg, "reasoning_content"),
                "has_reasoning_attr": hasattr(msg, "reasoning"),
            }
        finally:
            await oai.close()

    table = Table(title=f"Smoke Results ({store.run_id})")
    table.add_column("Model", style="cyan")
    table.add_column("Check")
    table.add_column("HTTP")
    table.add_column("Outcome")
    table.add_column("Codes")
    table.add_column("Reasoning Field")
    table.add_column("Preview")

    for r in smoke_rows:
        table.add_row(
            str(r["model"]),
            str(r["check"]),
            str(r["status"]),
            str(r["outcome"]),
            ", ".join(r["codes"]) or "-",
            str(r["reasoning_field"] or "-"),
            str(r["content"]),
        )
    console.print(table)
    if sdk_info:
        console.print(f"[bold]OpenAI SDK model_extra view:[/bold] {sdk_info}")

    summary = {
        "run_id": store.run_id,
        "stage": "smoke",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "models": models,
        "rows": smoke_rows,
        "sdk_view": sdk_info,
    }
    store.write_summary(summary)
    return summary


async def run_stage(
    stage: str,
    planned_cells: list[PlannedCell],
    config: AppConfig,
    rpm: int | None = None,
    concurrency: int | None = None,
    runs_root: Path | None = None,
    resume_dir: Path | None = None,
    console: Console | None = None,
) -> tuple[RunStore, bool]:
    """Execute a planned stage (pilot, screen, escalate).
    Returns (store, budget_exceeded). If budget_exceeded is True, caller should exit with code 3.
    """
    console = console or Console()
    api_key = get_nebius_api_key()
    store = RunStore.create_or_resume(stage, runs_root=runs_root, resume_dir=resume_dir)
    budget = BudgetGuard(config)
    effective_rpm = rpm or config.rpm
    effective_conc = concurrency or config.concurrency
    pacer = HeaderAwarePacer(base_rpm=effective_rpm)
    sem = asyncio.Semaphore(effective_conc)
    contracts_map = get_contract_map()

    store.write_plan(
        {
            "rpm": effective_rpm,
            "concurrency": effective_conc,
            "cells_count": len(planned_cells),
            "models": sorted({c.model for c in planned_cells}),
            "contracts": sorted({c.contract_id for c in planned_cells}),
        }
    )

    done_in_store = store.completed_trials()
    _, prior_counts = load_screening_fatal_cells(runs_root) if stage == "escalate" else (set(), {})

    work_items: list[tuple[PlannedCell, int]] = []
    for cell in planned_cells:
        start_trial = prior_counts.get(cell.cell_id, 0) if stage == "escalate" else 0
        for trial in range(start_trial, cell.n_trials):
            if (cell.cell_id, trial) in done_in_store:
                continue
            work_items.append((cell, trial))

    budget_exceeded = False
    stop_event = asyncio.Event()
    counts = {"pass": 0, "fail": 0, "error": 0, "total_cost_usd": 0.0}

    async with InferenceClient(
        base_url=config.nebius_base_url,
        api_key=api_key,
        pacer=pacer,
        connect_timeout_s=config.connect_timeout_s,
        read_timeout_s=config.read_timeout_s,
    ) as client:

        async def _worker(cell: PlannedCell, trial: int) -> None:
            nonlocal budget_exceeded
            if stop_event.is_set():
                return
            async with sem:
                if stop_event.is_set():
                    return
                single_est_cost = config.estimate_cost(
                    cell.model, cell.est_prompt_tokens, cell.est_completion_tokens
                )
                try:
                    await budget.reserve(single_est_cost)
                except BudgetExceededError as exc:
                    if not budget_exceeded:
                        console.print(f"[red bold]{exc}[/red bold]")
                    budget_exceeded = True
                    stop_event.set()
                    return

                contract = contracts_map[cell.contract_id]
                req = contract.build_request(cell.model, cell.condition, trial=trial)
                attempts_log: list[tuple[int, str, ParsedResponse, bool]] = []

                def _on_attempt(
                    att: int, ts_s: str, p_resp: ParsedResponse, is_fin: bool
                ) -> None:
                    attempts_log.append((att, ts_s, p_resp, is_fin))

                resp = await client.chat_completion(
                    req,
                    estimated_tokens=cell.est_prompt_tokens + cell.est_completion_tokens,
                    on_attempt=_on_attempt,
                )
                ores = contract.check(req, resp)
                actual_cost = budget.compute_call_cost(
                    cell.model,
                    resp.usage,
                    fallback_in=cell.est_prompt_tokens,
                    fallback_out=cell.est_completion_tokens,
                )
                await budget.commit(
                    single_est_cost,
                    actual_cost,
                    cell.model,
                    stage,
                    store.run_id,
                    resp.usage,
                )

                for att, ts_s, p_resp, is_fin in attempts_log:
                    att_ores = ores if is_fin else contract.check(req, p_resp)
                    rec = CallRecord(
                        run_id=store.run_id,
                        stage=stage,
                        cell_id=cell.cell_id,
                        contract_id=cell.contract_id,
                        model=cell.model,
                        condition=cell.condition.to_dict(),
                        trial=trial,
                        attempt=att,
                        ts_start=ts_s,
                        request=req,
                        status=p_resp.status,
                        headers=p_resp.headers,
                        raw=p_resp.raw,
                        parsed=p_resp.to_dict(),
                        outcome=att_ores.outcome.value,
                        codes=att_ores.codes,
                        detail=att_ores.detail,
                        usage=p_resp.usage,
                        cost_usd=actual_cost if is_fin else 0.0,
                        is_final=is_fin,
                    )
                    store.append_call(rec)

                counts[ores.outcome.value] = int(counts.get(ores.outcome.value, 0)) + 1
                counts["total_cost_usd"] = float(counts["total_cost_usd"]) + actual_cost

        tasks = [asyncio.create_task(_worker(cell, trial)) for cell, trial in work_items]
        if tasks:
            await asyncio.gather(*tasks)

    summary = {
        "run_id": store.run_id,
        "stage": stage,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "budget_exceeded": budget_exceeded,
        "counts": counts,
    }
    store.write_summary(summary)
    return store, budget_exceeded
