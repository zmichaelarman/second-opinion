"""The whole smoke flow against the fake Token Factory: no network, no key, no spend."""

import json

from rich.console import Console

from second_opinion.smoke import run_smoke

from conftest import FAKE_KEY
from fake_tf import FakeTokenFactory


def _run(settings, fake):
    import asyncio

    console = Console(record=True, width=200)
    code = asyncio.run(run_smoke(settings, console, fake.transport()))
    [run_dir] = [d for d in (settings.runs_dir).iterdir() if d.is_dir()]
    records = [json.loads(line) for line in (run_dir / "calls.jsonl").read_text().splitlines()]
    summary = json.loads((run_dir / "summary.json").read_text())
    return code, run_dir, records, summary, console.export_text()


def test_smoke_records_15_calls_and_summarizes(settings):
    fake = FakeTokenFactory()
    code, run_dir, records, summary, _ = _run(settings, fake)
    assert code == 0 and len(records) == 15
    assert (run_dir / "plan.json").exists()
    outcomes = {(r["contract_id"], r["outcome"]) for r in records}
    assert outcomes == {
        ("smoke.banana_think_off", "pass"), ("smoke.banana_think_on", "pass"),
        ("smoke.tool_leading_zero_stream", "pass"), ("smoke.banana_off_via_reasoning_effort", "pass"),
        ("smoke.sdk_tool_leading_zero", "pass"),
        ("smoke.banana_off_via_extra_body", "fail"),  # the fake ignores extra_body, as a real server may
    }
    assert {m["thinking_off_via_chat_template_kwargs"] for m in summary["models"]} == {"works"}
    assert {m["reasoning_field_non_stream"] for m in summary["models"]} == {"reasoning_content"}
    assert summary["thinking_off_probes"]["reasoning_effort_none"] == "works"
    assert summary["thinking_off_probes"]["extra_body"] == "ignored"
    assert "reasoning_content" in summary["sdk_view"]["message_model_extra_keys"]
    ledger = (settings.runs_dir / "ledger.jsonl").read_text().splitlines()
    assert len(ledger) == 15
    assert abs(sum(json.loads(x)["cost_usd"] for x in ledger) - summary["cost_usd"]) < 1e-6


def test_smoke_never_stores_the_api_key(settings):
    code, run_dir, *_ = _run(settings, FakeTokenFactory())
    for path in run_dir.iterdir():
        text = path.read_text()
        assert FAKE_KEY not in text and "Bearer" not in text and "set-cookie" not in text.lower()


def test_smoke_reports_ignored_switch_and_coercion(settings):
    fake = FakeTokenFactory(honor_chat_template_kwargs=False, coerce_digit_strings=True)
    code, _, records, summary, out = _run(settings, fake)
    assert {m["thinking_off_via_chat_template_kwargs"] for m in summary["models"]} == {"ignored"}
    tool = [r for r in records if r["contract_id"] == "smoke.tool_leading_zero_stream"]
    assert all("TYPE_COERCED" in r["codes"] for r in tool)
    assert "ignored" in out
