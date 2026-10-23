"""The runner against the fake Token Factory: retries, budget stop, resume and the --yes gate."""

import asyncio
import json

import httpx
from rich.console import Console

from second_opinion.runner import run_stage

from fake_tf import FakeTokenFactory


def run(settings, transport, **kw):
    kw.setdefault("stage", "screen")
    kw.setdefault("prereg", False)
    kw.setdefault("yes", True)
    console = Console(record=True, width=220)
    code = asyncio.run(run_stage(settings, console, transport=transport, retry_5xx_delay_s=0, **kw))
    return code, console.export_text()


def records(settings):
    out = []
    for d in sorted(settings.runs_dir.glob("*-screen*")):
        out += [json.loads(x) for x in (d / "calls.jsonl").read_text().splitlines()]
    return out


def test_without_yes_nothing_is_sent(settings):
    fake = FakeTokenFactory()
    code, out = run(settings, fake.transport(), yes=False, contracts_spec="T2")
    assert code == 0 and "nothing was sent" in out
    assert [r.method for r in fake.requests] == ["GET"]
    assert not settings.runs_dir.exists() or not any(settings.runs_dir.iterdir())


def test_screen_refuses_without_prereg_tag(settings):
    code, out = run(settings, FakeTokenFactory().transport(), prereg=True, contracts_spec="T2")
    assert code == 2 and "prereg-v1" in out


def test_small_screen_runs_every_trial(settings):
    settings.defaults["n_screen"] = 3
    fake = FakeTokenFactory()
    code, out = run(settings, fake.transport(), contracts_spec="T2,T3.leading_zero")
    rs = records(settings)
    assert code == 0 and len(rs) == (3 + 3) * 4 * 3
    assert all(r["final"] and r["outcome"] == "pass" for r in rs), {(r["contract_id"], tuple(r["codes"])) for r in rs if r["outcome"] != "pass"}
    [run_dir] = list(settings.runs_dir.glob("*-screen"))
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["outcomes"] == {"pass": 72} and len(summary["cells"]) == 24
    assert "T2.off|off" in summary["token_medians"]
    plan = json.loads((run_dir / "plan.json").read_text())
    assert plan["pending"] == 72 and len(plan["registry_hash"]) == 64


def test_429_then_success_and_5xx_twice_is_error(settings):
    settings.defaults["n_screen"] = 1
    sent_429 = {"done": False}
    fake = FakeTokenFactory()

    def handler(request):
        if request.method == "POST":
            body = json.loads(request.content)
            thinking = (body.get("chat_template_kwargs") or {}).get("enable_thinking")
            if thinking is False and not sent_429["done"]:
                sent_429["done"] = True
                return httpx.Response(429, json={"error": {"message": "slow"}}, headers={"retry-after": "0"})
            if thinking is True:
                return httpx.Response(503, json={"error": {"message": "down"}})
        return fake(request)

    code, _ = run(settings, httpx.MockTransport(handler), contracts_spec="T2", models_spec="nvidia/nemotron-3-super-120b-a12b", concurrency=1)
    rs = records(settings)
    by_contract = {}
    for r in rs:
        by_contract.setdefault(r["contract_id"], []).append(r)
    on = by_contract["T2.on"]
    assert [(r["attempt"], r["final"], r["codes"]) for r in on] == [(1, False, ["HTTP_5XX"]), (2, True, ["HTTP_5XX"])]
    assert on[-1]["outcome"] == "error"
    off = by_contract["T2.off"]
    assert [(r["attempt"], r["final"], r["outcome"]) for r in off] == [(1, False, "error"), (2, True, "pass")]
    assert off[0]["codes"] == ["HTTP_429"]
    assert sum(r["final"] for r in rs) == 3


def test_budget_stop_exits_3_and_resume_finishes(settings):
    settings.defaults["n_screen"] = 5
    fake = FakeTokenFactory(banana_usage=(400, 300))  # actual cost equals the fallback estimate
    settings.budget_usd = 0.00025  # room for two Nano calls at (400 in, 300 out)
    code, out = run(settings, fake.transport(), contracts_spec="T2.off", models_spec="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B", concurrency=1)
    assert code == 3 and "budget cap" in out
    done = [r for r in records(settings) if r["final"]]
    assert 0 < len(done) < 5
    settings.budget_usd = 40
    [run_dir] = list(settings.runs_dir.glob("*-screen"))
    code, _ = run(settings, fake.transport(), contracts_spec="T2.off", models_spec="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B", resume=run_dir)
    finals = [(r["cell_id"], r["trial"]) for r in records(settings) if r["final"]]
    assert code == 0 and len(finals) == 5 and len(set(finals)) == 5
