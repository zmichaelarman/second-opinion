from __future__ import annotations

import json
from typing import Any

from second_opinion.contracts.base import Owner
from second_opinion.investigator.cluster import FailureCluster
from second_opinion.stats import holm


def format_curl_repro(req: dict[str, Any], base_url: str = "https://api.tokenfactory.nebius.com/v1") -> str:
    clean_req = {k: v for k, v in req.items() if not k.startswith("_")}
    body_json = json.dumps(clean_req, indent=2, ensure_ascii=False)
    return (
        f"curl -X POST '{base_url.rstrip('/')}/chat/completions' \\\n"
        "  -H 'Content-Type: application/json' \\\n"
        "  -H \"Authorization: Bearer $NEBIUS_API_KEY\" \\\n"
        f"  -d '{body_json}'"
    )


def format_python_repro(req: dict[str, Any], base_url: str = "https://api.tokenfactory.nebius.com/v1") -> str:
    clean_req = {k: v for k, v in req.items() if not k.startswith("_")}
    body_repr = json.dumps(clean_req, indent=2, ensure_ascii=False)
    return (
        "import os\n"
        "import httpx\n\n"
        f"payload = {body_repr}\n"
        "resp = httpx.post(\n"
        f"    '{base_url.rstrip('/')}/chat/completions',\n"
        "    headers={'Authorization': f\"Bearer {os.environ['NEBIUS_API_KEY']}\"},\n"
        "    json=payload,\n"
        "    timeout=120.0,\n"
        ")\n"
        "print(resp.status_code, resp.json())\n"
    )


def evaluate_experiment_decisions(
    experiment_results: list[dict[str, Any]],
    alpha: float = 0.01,
    min_effect_size: float = 0.05,
) -> list[dict[str, Any]]:
    """Apply two-sided Fisher test + Holm correction across cluster hypotheses (§11.5).

    Accept if adjusted p < 0.01 and |rate_a - rate_b| >= 0.05.
    """
    if not experiment_results:
        return []
    raw_pvals = [float(r.get("p_value", 1.0)) for r in experiment_results]
    adj_pvals = holm(raw_pvals)

    evaluated: list[dict[str, Any]] = []
    for r, adj_p in zip(experiment_results, adj_pvals):
        item = dict(r)
        rate_diff = float(item.get("rate_diff", 0.0))
        accepted = (adj_p < alpha) and (rate_diff >= min_effect_size)
        item["adjusted_p_value"] = adj_p
        item["accepted"] = accepted
        evaluated.append(item)
    return evaluated


def build_finding_record(
    cluster: FailureCluster,
    minimal_req: dict[str, Any],
    shrunk_rate: float,
    evaluated_experiments: list[dict[str, Any]],
    base_url: str = "https://api.tokenfactory.nebius.com/v1",
) -> dict[str, Any]:
    accepted = [e for e in evaluated_experiments if e.get("accepted")]
    best = (
        min(accepted, key=lambda e: float(e["adjusted_p_value"]))
        if accepted
        else (min(evaluated_experiments, key=lambda e: float(e.get("p_value", 1.0))) if evaluated_experiments else None)
    )

    workaround = "Validate output client-side with strict schema/type checks and retry on conversion failure."
    accepted_factor = None
    if best is not None:
        arm_a = best["arm_a"]
        arm_b = best["arm_b"]
        safer_arm = arm_a if float(arm_a["failure_rate"]) < float(arm_b["failure_rate"]) else arm_b
        if best.get("accepted"):
            accepted_factor = best["factor"]
            workaround = (
                f"Set `{best['factor']}={safer_arm['level']!r}` "
                f"(reduces failure rate from {max(arm_a['failure_rate'], arm_b['failure_rate']) * 100:.1f}% "
                f"to {safer_arm['failure_rate'] * 100:.1f}%)."
            )

    codes_str = ", ".join(cluster.fatal_codes)
    title = f"[{cluster.contract_id}] {codes_str} on {', '.join(cluster.models_affected)}"
    if accepted_factor:
        title += f" (driven by `{accepted_factor}`)"

    return {
        "finding_id": f"finding_{cluster.cluster_id}",
        "cluster_id": cluster.cluster_id,
        "contract_id": cluster.contract_id,
        "title": title,
        "severity": cluster.severity.name,
        "owner": Owner.UNKNOWN.value,
        "models_affected": cluster.models_affected,
        "fatal_codes": cluster.fatal_codes,
        "cluster_failure_rate": cluster.failure_rate,
        "shrunk_failure_rate": shrunk_rate,
        "accepted_factor": accepted_factor,
        "workaround": workaround,
        "minimal_request": minimal_req,
        "curl_repro": format_curl_repro(minimal_req, base_url=base_url),
        "python_repro": format_python_repro(minimal_req, base_url=base_url),
        "experiments": evaluated_experiments,
    }

