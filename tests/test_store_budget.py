import json
from datetime import datetime

from second_opinion.budget import BudgetGuard, call_cost
from second_opinion.config import Price
from second_opinion.store import RunStore

PRICE = Price("m", input=1.0, output=3.0, source="test", as_of="2026-10-23")


def test_run_dirs_never_collide(tmp_path):
    now = datetime(2026, 10, 23, 11, 30)
    a = RunStore.create(tmp_path, "pilot", now)
    b = RunStore.create(tmp_path, "pilot", now)
    assert (a.run_id, b.run_id) == ("20261023-1130-pilot", "20261023-1130-pilot-2")


def test_calls_append_and_final_keys_ignore_torn_lines(tmp_path):
    store = RunStore.create(tmp_path, "screen")
    store.append_call({"cell_id": "c1", "trial": 0, "final": False})
    store.append_call({"cell_id": "c1", "trial": 0, "final": True})
    store.append_call({"cell_id": "c2", "trial": 3, "final": True})
    with store.calls_path.open("a") as f:
        f.write('{"cell_id": "c3", "trial"')  # torn by a crash
    assert store.final_keys() == {("c1", 0), ("c2", 3)}
    assert len(list(store.iter_calls())) == 3


def test_plan_is_written_once(tmp_path):
    store = RunStore.create(tmp_path, "screen")
    first = store.write_plan({"v": 1})
    second = store.write_plan({"v": 2})
    assert first.name == "plan.json" and second.name.startswith("plan-resume-")
    assert json.loads(first.read_text()) == {"v": 1}


def test_call_cost_rules():
    assert call_cost(PRICE, {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}, 200, 9.0) == 4.0
    assert call_cost(PRICE, None, 400, 9.0) == 0.0
    assert call_cost(PRICE, None, 500, 9.0) == 9.0
    assert call_cost(PRICE, None, None, 9.0) == 9.0
    assert call_cost(PRICE, None, 200, 9.0) == 9.0


def test_budget_guard_reserves_settles_and_persists(tmp_path):
    ledger = tmp_path / "runs" / "ledger.jsonl"
    guard = BudgetGuard(1.0, ledger)
    assert guard.try_reserve(0.6)
    assert not guard.try_reserve(0.5)  # 0.6 in flight + 0.5 > 1.0
    guard.settle(0.6, 0.25, {"run_id": "r"})
    assert guard.try_reserve(0.5)
    guard.settle(0.5, 0.5, {"run_id": "r"})
    assert round(guard.spent, 6) == 0.75 and guard.in_flight == 0
    again = BudgetGuard(1.0, ledger)
    assert round(again.spent, 6) == 0.75
    assert not again.try_reserve(0.3)
