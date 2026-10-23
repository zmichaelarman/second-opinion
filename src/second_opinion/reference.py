from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from rich.console import Console

from second_opinion.budget import BudgetGuard
from second_opinion.client import InferenceClient
from second_opinion.config import AppConfig, get_nebius_api_key, get_nvidia_api_key
from second_opinion.contracts.base import Owner
from second_opinion.contracts.registry import get_contract_map
from second_opinion.pacer import HeaderAwarePacer
from second_opinion.stats import fisher_two_sided, wilson_ci

NVIDIA_MODEL_MAP: dict[str, str | None] = {
    "nvidia/nemotron-3-super-120b-a12b": "nvidia/nemotron-3-super-120b-a12b",
    "nvidia/nemotron-3-ultra-550b-a55b": "nvidia/nemotron-3-ultra-550b-a55b",
    "nvidia/Nemotron-3-Ultra-550b-a55b": "nvidia/nemotron-3-ultra-550b-a55b",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B": "nvidia/nemotron-3-nano-30b-a3b",
    "nvidia/nemotron-3-nano-30b-a3b": "nvidia/nemotron-3-nano-30b-a3b",
    "nvidia/Nemotron-3_5-Lightning": None,
}


def map_to_nvidia_model(tf_model_id: str) -> str | None:
    if tf_model_id in NVIDIA_MODEL_MAP:
        return NVIDIA_MODEL_MAP[tf_model_id]
    lower = tf_model_id.lower()
    for k, v in NVIDIA_MODEL_MAP.items():
        if k.lower() == lower:
            return v
    return None


def attribute_owner(
    tf_failures: int,
    tf_trials: int,
    nv_failures: int | None,
    nv_trials: int | None,
    fatal_codes: list[str],
) -> tuple[Owner, float | None]:
    """Determine Owner attribution from Token Factory vs NVIDIA replay counts (§11.6).

    Rules:
    - No mapping available (nv_failures is None or nv_trials == 0): UNKNOWN.
    - Token Factory fails while NVIDIA fails significantly less (Fisher p < 0.05 and tf_rate > nv_rate): NEBIUS_PLATFORM.
    - Both fail at similar rates: NVIDIA_MODEL for VALUE_MISMATCH-type codes, UPSTREAM_SERVING otherwise.
    """
    if nv_failures is None or not nv_trials or nv_trials <= 0:
        return (Owner.UNKNOWN, None)

    pval = fisher_two_sided(tf_failures, tf_trials, nv_failures, nv_trials)
    tf_rate = tf_failures / max(1, tf_trials)
    nv_rate = nv_failures / max(1, nv_trials)

    if tf_rate > nv_rate and pval < 0.05:
        return (Owner.NEBIUS_PLATFORM, pval)

    if any(c in ("VALUE_MISMATCH", "EXTRA_CONTENT") for c in fatal_codes) and set(fatal_codes) <= {
        "VALUE_MISMATCH",
        "EXTRA_CONTENT",
    }:
        return (Owner.NVIDIA_MODEL, pval)

    return (Owner.UPSTREAM_SERVING, pval)


async def run_reference_replay(
    findings_path: Path,
    config: AppConfig,
    n_trials: int | None = None,
    console: Console | None = None,
) -> list[dict[str, Any]]:
    """Replay each finding's minimal request n=30 on the NVIDIA API at 20 RPM (§11.6)."""
    console = console or Console()
    n = n_trials or config.n_reference
    findings = json.loads(findings_path.read_text(encoding="utf-8"))
    nv_key = get_nvidia_api_key()
    pacer = HeaderAwarePacer(base_rpm=config.nvidia_rpm)
    contracts_map = get_contract_map()

    async with InferenceClient(
        base_url=config.nvidia_base_url,
        api_key=nv_key,
        pacer=pacer,
        connect_timeout_s=config.connect_timeout_s,
        read_timeout_s=config.read_timeout_s,
    ) as client:
        for finding in findings:
            min_req = copy.deepcopy(finding.get("minimal_request") or {})
            tf_model = str(min_req.get("model") or (finding.get("models_affected") or [""])[0])
            nv_model = map_to_nvidia_model(tf_model)
            fatal_codes = list(finding.get("fatal_codes") or [])

            if nv_model is None:
                finding["owner"] = Owner.UNKNOWN.value
                finding["reference_replay"] = {
                    "mapped_model": None,
                    "note": f"No NVIDIA reference endpoint mapping for {tf_model}",
                }
                continue

            contract_id = str(finding.get("contract_id", "T3"))
            contract = contracts_map[contract_id]
            send_req = {k: v for k, v in min_req.items() if not k.startswith("_")}
            send_req["model"] = nv_model

            dropped_params: list[str] = []
            nv_failures = 0
            nv_completed = 0

            for trial_idx in range(n):
                resp = await client.chat_completion(send_req, estimated_tokens=1000)
                # If NVIDIA API rejects a Token Factory-only parameter on first trial, drop and retry (§11.6)
                if trial_idx == 0 and 400 <= resp.status < 500:
                    for tf_param in ("chat_template_kwargs", "stream_options"):
                        if tf_param in send_req:
                            send_req.pop(tf_param, None)
                            dropped_params.append(tf_param)
                    if dropped_params:
                        resp = await client.chat_completion(send_req, estimated_tokens=1000)

                ores = contract.check(min_req, resp)
                nv_completed += 1
                if ores.outcome.value == "fail":
                    nv_failures += 1

            tf_rate = float(finding.get("shrunk_failure_rate") or finding.get("cluster_failure_rate") or 0.5)
            tf_trials = 30
            tf_fails = int(round(tf_rate * tf_trials))

            owner, pval = attribute_owner(
                tf_failures=tf_fails,
                tf_trials=tf_trials,
                nv_failures=nv_failures,
                nv_trials=nv_completed,
                fatal_codes=fatal_codes,
            )
            lo, hi = wilson_ci(nv_failures, max(1, nv_completed))
            finding["owner"] = owner.value
            finding["reference_replay"] = {
                "mapped_model": nv_model,
                "nv_failures": nv_failures,
                "nv_trials": nv_completed,
                "nv_failure_rate": nv_failures / max(1, nv_completed),
                "nv_wilson_ci": [lo, hi],
                "fisher_p_value": pval,
                "dropped_params": dropped_params,
            }

    findings_path.write_text(json.dumps(findings, indent=2), encoding="utf-8")
    return findings


async def rerun_findings(
    findings_path: Path,
    config: AppConfig,
    n_trials: int | None = None,
    console: Console | None = None,
) -> list[dict[str, Any]]:
    """Re-run every finding's minimal repro at n=30 on Token Factory and mark still_failing or fixed (§13 M6)."""
    console = console or Console()
    n = n_trials or config.n_rerun
    findings = json.loads(findings_path.read_text(encoding="utf-8"))
    api_key = get_nebius_api_key()
    pacer = HeaderAwarePacer(base_rpm=config.rpm)
    budget = BudgetGuard(config)
    contracts_map = get_contract_map()

    async with InferenceClient(
        base_url=config.nebius_base_url,
        api_key=api_key,
        pacer=pacer,
        connect_timeout_s=config.connect_timeout_s,
        read_timeout_s=config.read_timeout_s,
    ) as client:
        for finding in findings:
            min_req = copy.deepcopy(finding.get("minimal_request") or {})
            send_req = {k: v for k, v in min_req.items() if not k.startswith("_")}
            model = str(send_req.get("model") or config.code_model)
            contract_id = str(finding.get("contract_id", "T3"))
            contract = contracts_map[contract_id]

            fails = 0
            done = 0
            for _ in range(n):
                est = config.estimate_cost(model, 400, 1000)
                await budget.reserve(est)
                resp = await client.chat_completion(send_req, estimated_tokens=1400)
                ores = contract.check(min_req, resp)
                actual = budget.compute_call_cost(model, resp.usage)
                await budget.commit(est, actual, model, "rerun", "rerun", resp.usage)
                done += 1
                if ores.outcome.value == "fail":
                    fails += 1

            lo, hi = wilson_ci(fails, max(1, done))
            status_label = "still_failing" if fails > 0 else "fixed"
            finding["rerun"] = {
                "status": status_label,
                "failures": fails,
                "trials": done,
                "failure_rate": fails / max(1, done),
                "wilson_ci": [lo, hi],
            }
            console.print(
                f"Finding [bold]{finding.get('finding_id')}[/bold]: {status_label} ({fails}/{done})"
            )

    findings_path.write_text(json.dumps(findings, indent=2), encoding="utf-8")
    return findings
