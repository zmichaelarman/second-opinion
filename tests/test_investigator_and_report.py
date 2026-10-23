from __future__ import annotations

import asyncio
from pathlib import Path

from second_opinion.byo import generate_seeded_tool_arg_sets
from second_opinion.config import load_config
from second_opinion.contracts.base import Owner, Severity
from second_opinion.fixtures.payloads import load_tool_definition
from second_opinion.investigator.cluster import cluster_failures
from second_opinion.investigator.decide import evaluate_experiment_decisions
from second_opinion.investigator.experiment import (
    check_experiment_safety_gate,
    render_canonical_experiment_code,
)
from second_opinion.investigator.shrink import (
    retention_threshold,
    shrink_cluster_request,
)
from second_opinion.reference import attribute_owner
from second_opinion.report.build import build_report
from second_opinion.testkit import apply_factor


def test_cluster_failures_and_shrink() -> None:
    records = [
        {
            "is_final": True,
            "contract_id": "T3",
            "model": "nvidia/nemotron-3-super-120b-a12b",
            "outcome": "fail",
            "codes": ["TYPE_COERCED"],
            "request": {
                "model": "nvidia/nemotron-3-super-120b-a12b",
                "stream": True,
                "chat_template_kwargs": {"enable_thinking": True},
                "tools": [load_tool_definition("leading_zero")],
                "_expected_args": {"zip_code": "02134", "account_id": "007", "numeric_label": "12345"},
            },
        },
        {
            "is_final": True,
            "contract_id": "T3",
            "model": "nvidia/nemotron-3-super-120b-a12b",
            "outcome": "pass",
            "codes": [],
            "request": {},
        },
    ]
    # Even 1 failure qualifies if severity is S1 (TYPE_COERCED)
    clusters = cluster_failures(records)
    assert len(clusters) == 1
    assert clusters[0].severity == Severity.S1_SILENT

    assert retention_threshold(0.5, k=12) == 3

    async def fake_score(cand: dict, k: int) -> tuple[int, int]:
        # Keep reduction if stream was removed
        return (6, k) if not cand.get("stream") else (0, k)

    min_req, rate, hist = asyncio.run(
        shrink_cluster_request(
            clusters[0].representative_request,
            "T3",
            p0=0.5,
            score_fn=fake_score,
            k=12,
            max_candidates=4,
        )
    )
    assert "stream" not in min_req
    assert rate == 0.5
    assert len(hist) >= 1


def test_experiment_ast_safety_gate() -> None:
    valid_code = render_canonical_experiment_code("t3_1234", "t3_1234_h1", "stream", True, False, n=30)
    ok, errs = check_experiment_safety_gate(valid_code)
    assert ok, f"Expected valid code to pass safety gate, got: {errs}"

    # Disallowed import
    bad_import = "import os\n" + valid_code
    ok_imp, errs_imp = check_experiment_safety_gate(bad_import)
    assert not ok_imp
    assert any("import" in e.lower() for e in errs_imp)

    # Banned builtin open/eval
    bad_builtin = valid_code + "\nopen('secret.txt')\n"
    ok_b, errs_b = check_experiment_safety_gate(bad_builtin)
    assert not ok_b
    assert any("open" in e for e in errs_b)

    # Private attribute starting with _
    bad_attr = valid_code + "\nBASE._secret\n"
    ok_a, errs_a = check_experiment_safety_gate(bad_attr)
    assert not ok_a
    assert any("starting with `_`" in e for e in errs_a)

    # > 60 lines
    too_long = valid_code + ("\n# comment" * 60)
    ok_l, errs_l = check_experiment_safety_gate(too_long)
    assert not ok_l
    assert any("60 lines" in e for e in errs_l)


def test_apply_factor_menu_and_decide_and_owner() -> None:
    base = {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "s",
                "schema": {
                    "$defs": {"Item": {"type": "object", "minimum": 0}},
                    "properties": {"x": {"$ref": "#/$defs/Item"}},
                },
                "strict": True,
            },
        },
    }
    r_bare = apply_factor(base, "shape", "bare")
    assert "schema" not in r_bare["response_format"]["json_schema"]

    r_inline = apply_factor(base, "inline_refs", True)
    assert "$defs" not in r_inline["response_format"]["json_schema"]["schema"]

    r_rm = apply_factor(base, "remove_schema_keyword", "minimum")
    assert "minimum" not in str(r_rm)

    # Decide: Fisher adjusted p < 0.01 and |rate_a - rate_b| >= 0.05
    exp_results = [
        {
            "hypothesis_id": "h1",
            "factor": "stream",
            "arm_a": {"level": True, "failures": 25, "trials": 30, "failure_rate": 25 / 30, "wilson_lo": 0.65, "wilson_hi": 0.93},
            "arm_b": {"level": False, "failures": 0, "trials": 30, "failure_rate": 0.0, "wilson_lo": 0.0, "wilson_hi": 0.11},
            "p_value": 1e-10,
            "rate_diff": 25 / 30,
        }
    ]
    eval_res = evaluate_experiment_decisions(exp_results)
    assert eval_res[0]["accepted"] is True

    # Owner attribution rules
    owner_tf, _ = attribute_owner(20, 30, 0, 30, ["TYPE_COERCED"])
    assert owner_tf == Owner.NEBIUS_PLATFORM

    owner_model, _ = attribute_owner(15, 30, 14, 30, ["VALUE_MISMATCH"])
    assert owner_model == Owner.NVIDIA_MODEL

    owner_up, _ = attribute_owner(15, 30, 14, 30, ["TYPE_COERCED"])
    assert owner_up == Owner.UPSTREAM_SERVING

    owner_unk, _ = attribute_owner(15, 30, None, None, ["TYPE_COERCED"])
    assert owner_unk == Owner.UNKNOWN


def test_byo_generator_and_report_builder(tmp_path: Path) -> None:
    tool_def = load_tool_definition("leading_zero")
    arg_sets = generate_seeded_tool_arg_sets(tool_def, count=5)
    assert len(arg_sets) == 5
    assert all(set(a.keys()) == {"zip_code", "account_id", "numeric_label"} for a in arg_sets)

    cfg = load_config()
    out_dir = tmp_path / "docs"
    outputs = build_report(run_dirs=[], out_dir=out_dir, config=cfg, repo_root=tmp_path)
    assert outputs["index_html"].exists()
    assert outputs["findings_md"].exists()
    assert outputs["feedback_draft"].exists()
    html = outputs["index_html"].read_text(encoding="utf-8")
    assert "What to Trust" in html
    assert "prefers-color-scheme: dark" in html
