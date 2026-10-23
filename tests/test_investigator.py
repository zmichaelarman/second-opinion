import asyncio
import importlib.util
import json
import os
import re

import pytest
from rich.console import Console

from second_opinion import analysis, testkit
from second_opinion.budget import BudgetGuard
from second_opinion.client import ApiClient
from second_opinion.contracts import registry
from second_opinion.contracts.base import Condition
from second_opinion.investigator.cluster import build_clusters
from second_opinion.investigator.decide import decide, workaround
from second_opinion.investigator.experiment import TEMPLATE, gate, run_experiment, write_experiment
from second_opinion.investigator.hypothesize import fallback_hypotheses, propose, validate
from second_opinion.investigator.loop import investigate
from second_opinion.investigator.shrink import involved_fields, shrink, threshold
from second_opinion.investigator.spec import FactorError, inline_refs, remove_keyword, spec_from_cluster
from second_opinion.investigator.trials import TrialRunner
from second_opinion.pacer import Pacer
from second_opinion.parse import parse_completion_json
from second_opinion.runner import run_stage

from fake_tf import FakeTokenFactory, completion

SUPER = "nvidia/nemotron-3-super-120b-a12b"


@pytest.mark.parametrize("contract", registry.all_contracts(), ids=lambda c: c.id)
def test_spec_renders_exactly_the_contract_request(contract):
    for cond in contract.conditions:
        spec = spec_from_cluster("c", contract.id, "m", cond.as_dict(), ["X"])
        assert spec.request() == contract.build_request("m", cond, 0), (contract.id, cond)


def test_spec_grades_like_the_contract():
    contract = registry.by_id()["T3.leading_zero"]
    cond = contract.conditions[1]
    spec = spec_from_cluster("c", contract.id, "m", cond.as_dict(), ["TYPE_COERCED"])
    args = '{"zip_code": 2134, "account_id": "007", "numeric_label": "12345"}'
    resp = parse_completion_json(completion(None, "r", [("save_account", args)], "tool_calls"))
    req = spec.request()
    assert spec.check(req, resp).codes == contract.check(req, resp).codes
    assert spec.reproduces(spec.check(req, resp))


def test_factors_apply_and_refuse_where_they_do_not_fit():
    t3 = spec_from_cluster("c", "T3.markup", "m", Condition(True, False, variant="markup").as_dict(), ["VALUE_TRUNCATED"])
    assert t3.with_factor("stream", "true").request()["stream"] is True
    assert t3.with_factor("thinking", False).request()["chat_template_kwargs"] == {"enable_thinking": False}
    assert t3.with_factor("append_literal", "</x>").expected_args["content"].endswith("</x>")
    assert t3.with_factor("arg_type", "path:integer").tool["function"]["parameters"]["properties"]["path"]["type"] == "integer"
    assert t3.with_factor("temperature", 0).request()["temperature"] == 0.0
    for bad in (("shape", "bare"), ("remove_schema_keyword", "enum"), ("arg_type", "nope:string"), ("colour", "red")):
        with pytest.raises(FactorError):
            t3.with_factor(*bad)
    t5 = spec_from_cluster("c", "T5.defs_ref", "m", Condition(False, variant="defs_ref").as_dict(), ["SCHEMA_INVALID:type"])
    assert "$defs" not in json.dumps(t5.with_factor("inline_refs", True).schema)
    assert t5.with_factor("shape", "bare").request()["response_format"]["json_schema"] == t5.schema
    stripped = t5.with_factor("remove_schema_keyword", "title")
    assert '"title"' not in json.dumps(stripped.schema) and stripped.oracle_schema == t5.oracle_schema


def test_schema_helpers():
    schema = {"$defs": {"A": {"type": "string", "title": "A"}}, "properties": {"a": {"$ref": "#/$defs/A"}}, "title": "T"}
    assert inline_refs(schema) == {"properties": {"a": {"type": "string", "title": "A"}}, "title": "T"}
    assert remove_keyword(schema, "title") == {"$defs": {"A": {"type": "string"}}, "properties": {"a": {"$ref": "#/$defs/A"}}}


def test_threshold_and_involved_fields():
    assert threshold(1.0) == 6 and threshold(0.1) == 1 and threshold(0.5) == 3
    examples = [{"detail": {"mismatches": [{"path": "$.zip_code"}]}},
                {"detail": {"SCHEMA_INVALID:enum": {"errors": [{"path": "$.genre"}]}}}]
    assert involved_fields(examples) == {"zip_code", "genre"}


def test_decide_holm_and_effect():
    results = [
        {"factor": "stream", "a": {"k": 40, "n": 100, "rate": 0.4, "level": True}, "b": {"k": 2, "n": 100, "rate": 0.02, "level": False}},
        {"factor": "thinking", "a": {"k": 5, "n": 100, "rate": 0.05, "level": True}, "b": {"k": 4, "n": 100, "rate": 0.04, "level": False}},
    ]
    decided = decide(results)
    assert decided[0]["accepted"] and not decided[1]["accepted"]
    assert decided[0]["p_holm"] >= decided[0]["p"]
    assert workaround(decided[0]).startswith("set stream to false: 2% failures")


def test_hypothesis_validation_and_fallback():
    spec = spec_from_cluster("c", "T3.leading_zero", "m", Condition(True, True, variant="leading_zero").as_dict(), ["TYPE_COERCED"])
    raw = [{"factor": "stream", "level_a": True, "level_b": False, "rationale": "r"},
           {"factor": "shape", "level_a": "bare", "level_b": "wrapped", "rationale": "not a json contract"},
           {"factor": "magic", "level_a": 1, "level_b": 2, "rationale": "off menu"},
           {"factor": "stream", "level_a": True, "level_b": False, "rationale": "duplicate"},
           {"factor": "thinking", "level_a": True, "level_b": True, "rationale": "same level"}]
    assert [h["factor"] for h in validate(raw, spec)] == ["stream"]
    assert {h["factor"] for h in fallback_hypotheses(spec)} == {"stream", "thinking", "temperature"}

    async def bad(body):
        return "not json at all"

    out = asyncio.run(propose(bad, "ultra", spec, 0.5, [], []))
    assert out["source"] == "fallback" and out["problem"] == "reply was not valid JSON"


def test_safety_gate():
    ok = TEMPLATE.format(cluster_id="c1", hypothesis_id="c1_h11", factor="stream", level_a=True, level_b=False)
    assert gate(ok) == []
    assert any("import os" in p for p in gate("import os\n" + ok))
    assert any("open" in p for p in gate(ok + "\nx = open('/etc/passwd')\n"))
    assert any("attribute" in p for p in gate(ok + "\nBASE.__class__\n"))
    assert any("_transport" in p for p in gate(ok.replace("import base_request", "import _transport, base_request")))
    assert any("getattr" in p for p in gate(ok + "\ngetattr(BASE, 'spec')\n"))
    assert any("more than 60 lines" in p for p in gate(ok + "\n" * 70))
    assert any("must call arm" in p for p in gate("from second_opinion.testkit import arm\n\ndef test_x():\n    pass\n"))


def test_write_experiment_asks_for_rewrites(tmp_path):
    good = TEMPLATE.format(cluster_id="c", hypothesis_id="h", factor="stream", level_a=True, level_b=False)
    replies = iter(["```python\nimport os\n" + good + "```", "```python\n" + good + "```"])

    async def complete(body):
        return next(replies)

    h = {"factor": "stream", "level_a": True, "level_b": False, "rationale": "r"}
    out = asyncio.run(write_experiment(complete, "super", "c", "h", h, tmp_path / "h.py"))
    assert out["ok"] and out["attempts"] == 2 and (tmp_path / "h.py").read_text() == good
    assert not (tmp_path / "h.py.tmp").exists()

    async def always_bad(body):
        return "```python\nimport subprocess\n```"

    out = asyncio.run(write_experiment(always_bad, "super", "c", "h2", h, tmp_path / "h2.py"))
    assert not out["ok"] and out["attempts"] == 3 and (tmp_path / "h2.py.tmp").exists() and not (tmp_path / "h2.py").exists()


def test_run_experiment_subprocess(tmp_path):
    path = tmp_path / "test_trivial.py"
    path.write_text("def test_ok():\n    assert True\n")
    out = run_experiment(path, tmp_path, {})
    assert out["returncode"] == 0, out


def _runner(settings, fake, records):
    client = ApiClient.for_provider(settings, "tokenfactory", fake.transport())
    return client, TrialRunner(client, Pacer(60000), BudgetGuard(40, settings.ledger_path), records, stage="investigate",
                               run_id="r", price=settings.price_for(SUPER), retry_delay_s=0)


def test_shrink_drops_uninvolved_arguments_but_keeps_the_failure(settings, tmp_path):
    fake = FakeTokenFactory(coerce_digit_strings=True)
    spec = spec_from_cluster("c", "T3.leading_zero", SUPER, Condition(True, True, variant="leading_zero").as_dict(), ["TYPE_COERCED"])

    async def go():
        client, runner = _runner(settings, fake, tmp_path / "t.jsonl")
        async with client:
            return await shrink(spec, 1.0, 50, 50, {"zip_code"}, runner)

    result = asyncio.run(go())
    assert list(result.spec.expected_args) == ["zip_code"]
    assert result.spec.stream is False and result.spec.thinking is False  # the bug does not need either
    assert result.k == result.n == 12
    assert len(result.steps) <= 8


def _llm_responder(cluster_holder):
    def respond(body):
        fmt = body.get("response_format") or {}
        if fmt.get("type") == "json_schema" and fmt["json_schema"].get("name") == "hypotheses":
            return json.dumps({"title": "Digit strings become integers when streaming",
                               "hypotheses": [{"factor": "stream", "level_a": True, "level_b": False, "rationale": "stream parser"},
                                              {"factor": "temperature", "level_a": 1.0, "level_b": 0, "rationale": "sampling"}]})
        text = body["messages"][-1]["content"]
        if "Write a pytest file" in text:
            text = text.split("```python", 1)[1]  # the template block, not the API description above it
            cid = re.search(r'base_request\("([^"]+)"\)', text).group(1)
            hid = re.search(r'hypothesis_id="([^"]+)"', text).group(1)
            factor = re.search(r'factor="([^"]+)"', text).group(1)
            levels = re.findall(r"level=([^,]+), n=100", text)
            return "```python\n" + TEMPLATE.format(cluster_id=cid, hypothesis_id=hid, factor=factor,
                                                   level_a=eval(levels[0]), level_b=eval(levels[1])).replace("n=100", "n=40") + "```"
        return None
    return respond


def test_investigate_end_to_end_accepts_the_stream_cause(settings, monkeypatch):
    settings.defaults["n_screen"] = 6
    buggy = FakeTokenFactory(coerce_digit_strings=True, coerce_only_when_streaming=True, llm_responder=_llm_responder(None))
    quiet = Console(file=open(os.devnull, "w"))
    asyncio.run(run_stage(settings, quiet, stage="screen", contracts_spec="T3.leading_zero", models_spec=SUPER,
                          prereg=False, yes=True, transport=buggy.transport(), retry_5xx_delay_s=0))
    runs = analysis.stage_runs(settings.runs_dir, "screen")
    [cl] = build_clusters(analysis.iter_records(runs))
    assert cl.codes == ("SCHEMA_INVALID:type", "TYPE_COERCED") and cl.failures == 6

    monkeypatch.setenv("SECOND_OPINION_ROOT", str(settings.root))
    monkeypatch.setattr(testkit, "_transport", buggy.transport())

    def in_process(path, root, env):
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.test_hypothesis()
        return {"returncode": 0}

    console = Console(record=True, width=200)
    code = asyncio.run(investigate(settings, console, runs, yes=True, transport=buggy.transport(),
                                   experiment_runner=in_process, arm_n=40))
    assert code == 0
    [inv] = analysis.stage_runs(settings.runs_dir, "investigate")
    [finding] = [json.loads(x) for x in (inv / "findings.jsonl").read_text().splitlines()]
    assert finding["title"] == "Digit strings become integers when streaming"
    assert finding["factor"].startswith("stream (true vs false)")
    assert finding["workaround"].startswith("set stream to false")
    assert [a["rate"] for a in finding["arms"]] == [1.0, 0.0]
    assert finding["minimal"]["spec"]["stream"] is True  # shrinking kept stream: the bug needs it
    assert (settings.root / "experiments" / f"{cl.id}_h11.py").exists()
    out = console.export_text()
    assert "ACCEPTED" in out


def test_reference_attribution_rules():
    from second_opinion.reference import attribute
    assert attribute(["TYPE_COERCED"], 30, 30, 0, 30)[0] == "nebius_platform"
    assert attribute(["VALUE_MISMATCH"], 10, 30, 9, 30)[0] == "nvidia_model"
    assert attribute(["TYPE_COERCED"], 10, 30, 9, 30)[0] == "upstream_serving"
    assert attribute(["TYPE_COERCED"], 10, 30, None, None)[0] == "unknown"


def test_reference_end_to_end(settings, tmp_path):
    from second_opinion.reference import reference
    spec = spec_from_cluster("c1", "T3.leading_zero", SUPER, Condition(True, False, variant="leading_zero").as_dict(), ["TYPE_COERCED"])
    findings = tmp_path / "findings.jsonl"
    findings.write_text(json.dumps({"id": "c1", "minimal": {"spec": spec.to_json(), "model": SUPER, "k": 30, "n": 30}}) + "\n"
                        + json.dumps({"id": "c2", "minimal": {"spec": spec.to_json(), "model": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
                                                              "k": 5, "n": 30}}) + "\n")
    settings.defaults["providers"]["nvidia"]["rpm"] = 60000
    nvidia = FakeTokenFactory()  # NVIDIA's stack copies the digits correctly
    console = Console(record=True, width=200)
    assert asyncio.run(reference(settings, console, findings, yes=True, n=10, transport=nvidia.transport())) == 0
    [run] = analysis.stage_runs(settings.runs_dir, "reference")
    rows = {r["finding_id"]: r for r in map(json.loads, (run / "attribution.jsonl").read_text().splitlines())}
    assert rows["c1"]["owner"] == "nebius_platform" and rows["c1"]["nv_k"] == 0
    assert rows["c2"]["owner"] == "unknown"


def test_testkit_arm_and_compare_in_process(settings, monkeypatch, tmp_path):
    inv = tmp_path / "inv"
    spec = spec_from_cluster("cx", "T3.leading_zero", SUPER, Condition(True, False, variant="leading_zero").as_dict(), ["TYPE_COERCED"])
    (inv / "clusters" / "cx").mkdir(parents=True)
    (inv / "clusters" / "cx" / "minimal.json").write_text(json.dumps({"spec": spec.to_json()}))
    monkeypatch.setenv("SECOND_OPINION_INVESTIGATION", str(inv))
    monkeypatch.setenv("SECOND_OPINION_ROOT", str(settings.root))
    monkeypatch.setenv("SECOND_OPINION_EXPERIMENT_BUDGET_USD", "5")
    monkeypatch.setattr(testkit, "_transport", FakeTokenFactory(coerce_digit_strings=True, coerce_only_when_streaming=True).transport())
    base = testkit.base_request("cx")
    a = testkit.arm(base, factor="stream", level=True, n=8)
    b = testkit.arm(base, factor="stream", level=False, n=8)
    result = testkit.compare(a, b, hypothesis_id="cx_h11")
    assert (a.batch.k, b.batch.k) == (8, 0) and result["p"] < 0.01
    assert json.loads((inv / "experiments" / "cx_h11.json").read_text())["a"]["k"] == 8
    assert len((inv / "experiments" / "cx.calls.jsonl").read_text().splitlines()) == 16
