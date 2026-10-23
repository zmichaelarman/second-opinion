import json
import subprocess

from second_opinion import analysis
from second_opinion.contracts import registry
from second_opinion.discovery import discover
from second_opinion.planner import (
    TokenEstimates,
    estimate_cost,
    plan_escalate,
    plan_pilot,
    plan_screen,
    prereg_check,
    render_test_plan,
)

from fake_tf import MODEL_IDS

DEFAULTS = {"n_screen": 50, "n_escalate": 300, "alpha": 0.01}


def models(settings):
    return [m for m in discover({"data": [{"id": m} for m in MODEL_IDS]}, settings) if m.runnable]


def test_pilot_is_about_50_calls(settings):
    items = plan_pilot(registry.all_contracts(), models(settings))
    assert len(items) == 38 + 12
    assert {it.cell.model for it in items if it.cell.contract_id.startswith("T5")} == {"nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"}
    assert {it.cell.model for it in items if it.cell.contract_id.startswith("T3")} == {
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B", "nvidia/nemotron-3-super-120b-a12b"}


def test_screen_counts_rounds_and_max_n(settings):
    ms = models(settings)
    items = plan_screen(registry.all_contracts(), ms, 50)
    assert len(items) == 152 * 50
    assert [it.trial for it in items[:152]] == [0] * 152  # round 0 covers every cell first
    capped = plan_screen(registry.all_contracts(), ms, 50, {"nvidia/Nemotron-3-Ultra-550b-a55b": 10})
    assert len(capped) == 152 * 50 - 38 * 40


def test_escalate_tops_up_core_and_failing_cells(settings):
    ms = models(settings)[:1]
    contracts = registry.select("T3.leading_zero,T2.off")
    screened = {}
    for it in plan_screen(contracts, ms, 50):
        s = screened.setdefault(it.cell.id, analysis.CellStats(it.cell.id, it.cell.contract_id, it.cell.model, it.cell.condition.as_dict()))
        s.n += 1
        s.max_trial = max(s.max_trial, it.trial)
    t2_off = next(it.cell for it in plan_screen(contracts, ms, 1) if it.cell.contract_id == "T2.off")
    screened[t2_off.id].fails = 1
    items = plan_escalate(contracts, ms, screened, 300)
    cells = {it.cell.id for it in items}
    lz_core = next(c for c in registry.by_id()["T3.leading_zero"].core_conditions)
    assert len(cells) == 2  # the T3 core condition and the failing T2 cell
    assert any(it.cell.condition == lz_core for it in items)
    assert min(it.trial for it in items) == 50 and max(it.trial for it in items) == 299
    assert len(items) == 2 * 250


def test_cost_estimate_uses_pilot_medians(settings, tmp_path):
    ms = {m.id: m for m in models(settings)}
    contracts = registry.by_id()
    items = plan_pilot(registry.select("T2.off"), list(ms.values()))
    fallback = estimate_cost(items, ms, contracts, TokenEstimates())
    assert fallback == {("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B", "T2"): (1, (400 * 0.06 + 300 * 0.24) / 1e6)}
    run = settings.runs_dir / "20261023-1200-pilot"
    run.mkdir(parents=True)
    (run / "summary.json").write_text(json.dumps({"token_medians": {"T2.off|off": {"prompt": 50, "completion": 2, "n": 1}}}))
    est = TokenEstimates.from_runs(settings.runs_dir)
    assert est.source == run.name and est.get("T2.off", False) == (50, 2) and est.get("T2.on", True) == (400, 2000)


def git(root, *args):
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True,
                   env={"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
                        "GIT_COMMITTER_EMAIL": "t@t", "PATH": "/usr/bin:/bin:/opt/homebrew/bin"})


def test_prereg_check_needs_tag_and_matching_hash(tmp_path):
    ok, msg = prereg_check(tmp_path, DEFAULTS)
    assert not ok and "prereg-v1" in msg
    git(tmp_path, "init", "-q")
    (tmp_path / "TEST_PLAN.md").write_text(render_test_plan(DEFAULTS))
    git(tmp_path, "add", "TEST_PLAN.md")
    git(tmp_path, "commit", "-qm", "plan")
    git(tmp_path, "tag", "prereg-v1")
    ok, msg = prereg_check(tmp_path, DEFAULTS)
    assert ok, msg
    ok, msg = prereg_check(tmp_path, {**DEFAULTS, "n_screen": 20})  # the code changed after tagging
    assert not ok


def test_test_plan_lists_every_contract():
    text = render_test_plan(DEFAULTS, ["nvidia/a", "nvidia/b"])
    assert "Registry SHA-256: `" in text and "`nvidia/a`" in text
    for c in registry.all_contracts():
        assert f"`{c.id}`" in text
