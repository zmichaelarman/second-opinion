from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from second_opinion.budget import BudgetExceededError, BudgetGuard
from second_opinion.config import load_config
from second_opinion.contracts.registry import (
    compute_registry_hash,
    get_tier1_contracts,
)
from second_opinion.discovery import (
    FALLBACK_NEMOTRON_MODELS,
    filter_nemotron_models,
)
from second_opinion.fixtures.payloads import load_schema_definition
from second_opinion.fixtures.schemas.models import generate_pydantic_schemas
from second_opinion.pacer import HeaderAwarePacer
from second_opinion.planner import (
    build_stage_plan,
    generate_test_plan_markdown,
)
from second_opinion.store import CallRecord, RunStore


def test_contract_registry_counts_and_hash() -> None:
    contracts = get_tier1_contracts()
    by_id = {c.id: len(c.conditions) for c in contracts}
    assert by_id == {"T1": 4, "T2": 3, "T3": 12, "T5": 19}
    assert sum(by_id.values()) == 38

    h1 = compute_registry_hash(contracts)
    h2 = compute_registry_hash(contracts)
    assert len(h1) == 64
    assert h1 == h2


def test_pydantic_generated_schemas_match_committed_json() -> None:
    generated = generate_pydantic_schemas()
    for name, schema in generated.items():
        committed = load_schema_definition(name)
        assert schema == committed


def test_discovery_filtering_rules() -> None:
    sample_ids = [
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
        "nvidia/Nemotron-3_5-Lightning",
        "nvidia/nemotron-3-super-120b-a12b",
        "nvidia/Nemotron-3-Ultra-550b-a55b",
        "nvidia/Nemotron-VL-12B",
        "nvidia/Nemotron-Omni-8B",
        "nvidia/Llama-3.1-Nemotron-Safety-Guard-8B",
        "nvidia/nv-embed-nemotron",
        "nvidia/nv-rerank-nemotron",
        "Qwen/Qwen3-235B-A22B",
    ]
    filtered = filter_nemotron_models(sample_ids)
    assert len(filtered) == 4
    assert set(filtered) == set(FALLBACK_NEMOTRON_MODELS)


def test_planner_pilot_and_screen_cells() -> None:
    cfg = load_config()
    models = list(FALLBACK_NEMOTRON_MODELS)

    pilot_cells = build_stage_plan("pilot", models, cfg)
    # 38 conditions on cheapest + 4 T3 variants on Super = 42 cells, 38*1 + 4*3 = 50 calls
    assert len(pilot_cells) == 42
    assert sum(c.n_trials for c in pilot_cells) == 50

    screen_cells = build_stage_plan("screen", models, cfg)
    # 38 conditions * 4 models = 152 cells
    assert len(screen_cells) == 152
    assert all(c.n_trials == 50 for c in screen_cells)

    plan_md = generate_test_plan_markdown(cfg)
    assert compute_registry_hash() in plan_md


def test_header_aware_pacer_and_budget_guard(tmp_path: Path) -> None:
    fake_time = [100.0]

    def clock() -> float:
        return fake_time[0]

    async def fake_sleep(dt: float) -> None:
        fake_time[0] += dt

    pacer = HeaderAwarePacer(base_rpm=40.0, clock=clock, sleep_fn=fake_sleep)
    assert pacer.effective_rpm() == 40.0

    pacer.update_from_headers({"x-ratelimit-limit-requests": "30"})
    assert pacer.effective_rpm() == 24.0  # 0.8 * 30

    pacer.update_from_headers({"x-ratelimit-over-limit": "yes"})
    assert pacer.effective_rpm() == 12.0  # halved for 120s

    fake_time[0] += 125.0
    assert pacer.effective_rpm() == 24.0

    # Budget guard test
    cfg = load_config()
    ledger_file = tmp_path / "ledger.json"
    guard = BudgetGuard(cfg, ledger_path=ledger_file, cap_override_usd=0.05)

    async def _exercise_budget() -> None:
        await guard.reserve(0.03)
        with pytest.raises(BudgetExceededError):
            await guard.reserve(0.03)
        await guard.commit(0.03, 0.02, "nvidia/Nemotron-3_5-Lightning", "test", "run1")
        assert abs(guard.spent_usd - 0.02) < 1e-6

    asyncio.run(_exercise_budget())


def test_run_store_append_and_resume(tmp_path: Path) -> None:
    store = RunStore.create_or_resume("screen", runs_root=tmp_path)
    rec = CallRecord(
        run_id=store.run_id,
        stage="screen",
        cell_id="abc123def456",
        contract_id="T2",
        model="nvidia/Nemotron-3_5-Lightning",
        condition={"thinking": False, "stream": False, "variant": "off"},
        trial=0,
        attempt=1,
        ts_start="2026-10-23T12:00:00Z",
        request={"model": "m", "headers": {"Authorization": "Bearer <REDACTED>"}},
        status=200,
        headers={},
        raw={},
        parsed={},
        outcome="pass",
        is_final=True,
    )
    store.append_call(rec)
    loaded = store.load_calls(final_only=True)
    assert len(loaded) == 1
    assert "headers" not in loaded[0]["request"]
    assert ("abc123def456", 0) in store.completed_trials()
