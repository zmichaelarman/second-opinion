from __future__ import annotations

import json
from typing import Any

from second_opinion.budget import BudgetGuard
from second_opinion.client import InferenceClient
from second_opinion.config import AppConfig
from second_opinion.investigator.cluster import FailureCluster
from second_opinion.testkit import FACTOR_MENU

HYPOTHESIS_SCHEMA = {
    "type": "object",
    "properties": {
        "hypotheses": {
            "type": "array",
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "factor": {
                        "type": "string",
                        "enum": sorted(FACTOR_MENU),
                    },
                    "level_a": {},
                    "level_b": {},
                    "rationale": {"type": "string"},
                },
                "required": ["factor", "level_a", "level_b", "rationale"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["hypotheses"],
    "additionalProperties": False,
}


def filter_valid_hypotheses(raw_hypotheses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep up to 3 hypotheses whose factor is on FACTOR_MENU (§11.3)."""
    valid: list[dict[str, Any]] = []
    for h in raw_hypotheses:
        if not isinstance(h, dict):
            continue
        factor = str(h.get("factor", "")).strip()
        if factor not in FACTOR_MENU:
            continue
        if "level_a" not in h or "level_b" not in h:
            continue
        valid.append(
            {
                "factor": factor,
                "level_a": h["level_a"],
                "level_b": h["level_b"],
                "rationale": str(h.get("rationale", "")),
            }
        )
        if len(valid) >= 3:
            break
    return valid


def fallback_hypotheses_for_cluster(cluster: FailureCluster) -> list[dict[str, Any]]:
    """Deterministic fallback hypotheses from FACTOR_MENU if LLM output fails to parse."""
    if cluster.contract_id == "T3":
        return [
            {
                "factor": "thinking",
                "level_a": True,
                "level_b": False,
                "rationale": "Test whether enabling reasoning alters tool-call XML-to-JSON conversion fidelity.",
            },
            {
                "factor": "stream",
                "level_a": True,
                "level_b": False,
                "rationale": "Test whether incremental SSE chunk parsing introduces argument corruption vs batch.",
            },
        ]
    if cluster.contract_id == "T5":
        return [
            {
                "factor": "shape",
                "level_a": "wrapped",
                "level_b": "bare",
                "rationale": "Compare wrapped OpenAI json_schema shape against bare schema shape.",
            },
            {
                "factor": "thinking",
                "level_a": True,
                "level_b": False,
                "rationale": "Test whether reasoning mode interferes with guided JSON schema enforcement.",
            },
        ]
    return [
        {
            "factor": "max_tokens",
            "level_a": 128,
            "level_b": 2048,
            "rationale": "Test whether low max_tokens triggers unclosed <think> leak into content.",
        }
    ]


async def propose_hypotheses(
    cluster: FailureCluster,
    minimal_req: dict[str, Any],
    config: AppConfig,
    client: InferenceClient,
    budget: BudgetGuard,
    run_id: str,
    rejected_prior: list[dict[str, Any]] | None = None,
    hypothesis_model: str | None = None,
) -> list[dict[str, Any]]:
    """Use Nemotron 3 Ultra (thinking on, response_format json_schema) to propose up to 3 factor hypotheses (§11.3)."""
    model = hypothesis_model or config.hypothesis_model
    failing_samples = [
        {
            "parsed": r.get("parsed", {}),
            "codes": r.get("codes", []),
            "detail": r.get("detail", {}),
        }
        for r in cluster.sample_failing_records[:3]
    ]
    passing_samples = [
        {
            "parsed": r.get("parsed", {}),
            "codes": r.get("codes", []),
        }
        for r in cluster.sample_passing_records[:3]
    ]

    prior_note = ""
    if rejected_prior:
        prior_note = (
            f"\nPreviously rejected hypotheses (do NOT repeat these): "
            f"{json.dumps(rejected_prior, ensure_ascii=False)}\n"
        )

    prompt = (
        f"Cluster ID: {cluster.cluster_id} (Contract {cluster.contract_id})\n"
        f"Fatal failure codes: {cluster.fatal_codes}\n"
        f"Models affected: {cluster.models_affected}\n"
        f"Minimal reproducing request:\n{json.dumps(minimal_req, indent=2, ensure_ascii=False)}\n\n"
        f"Sample failing responses (up to 3):\n{json.dumps(failing_samples, indent=2, ensure_ascii=False)[:3000]}\n\n"
        f"Sample passing responses (up to 3):\n{json.dumps(passing_samples, indent=2, ensure_ascii=False)[:2000]}\n\n"
        f"Allowed factor menu: {sorted(FACTOR_MENU)}\n"
        f"{prior_note}\n"
        "Propose up to 3 controlled 2-arm factor experiments ({factor, level_a, level_b, rationale}) "
        "using ONLY factors from the factor menu."
    )

    req = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a root-cause analysis investigator diagnosing conversion-layer defects "
                    "on LLM serving infrastructure. Respond strictly with valid JSON matching the schema."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 4096,
        "chat_template_kwargs": {"enable_thinking": True},
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "hypotheses_response",
                "schema": HYPOTHESIS_SCHEMA,
                "strict": True,
            },
        },
    }

    est_cost = config.estimate_cost(model, 1200, 2000)
    await budget.reserve(est_cost)
    resp = await client.chat_completion(req, estimated_tokens=3200)
    actual_cost = budget.compute_call_cost(model, resp.usage, 1200, 2000)
    await budget.commit(est_cost, actual_cost, model, "hypothesize", run_id, resp.usage)

    try:
        parsed = json.loads(resp.content)
        raw_list = parsed.get("hypotheses") if isinstance(parsed, dict) else []
        valid = filter_valid_hypotheses(raw_list or [])
        if valid:
            return valid
    except Exception:
        pass

    return fallback_hypotheses_for_cluster(cluster)
