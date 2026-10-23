from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rich.console import Console

from second_opinion.budget import BudgetExceededError, BudgetGuard
from second_opinion.client import InferenceClient
from second_opinion.config import AppConfig, REPO_ROOT, get_nebius_api_key
from second_opinion.contracts.registry import get_contract_map
from second_opinion.investigator.cluster import FailureCluster, cluster_failures
from second_opinion.investigator.decide import (
    build_finding_record,
    evaluate_experiment_decisions,
)
from second_opinion.investigator.experiment import generate_and_run_experiment
from second_opinion.investigator.hypothesize import propose_hypotheses
from second_opinion.investigator.shrink import shrink_cluster_request
from second_opinion.pacer import HeaderAwarePacer
from second_opinion.store import RunStore, load_records_from_runs


async def run_investigator(
    run_dirs: list[Path],
    config: AppConfig,
    hypothesis_model: str | None = None,
    code_model: str | None = None,
    console: Console | None = None,
    runs_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Execute the 4-step Investigator loop (Cluster -> Shrink -> Hypothesize -> Experiment -> Decide) (§11)."""
    console = console or Console()
    records = load_records_from_runs(run_dirs, final_only=True)
    clusters = cluster_failures(records)
    if not clusters:
        console.print("[green]No failure clusters met threshold for investigation.[/green]")
        return []

    api_key = get_nebius_api_key()
    store = RunStore.create_or_resume("investigate", runs_root=runs_root)
    pacer = HeaderAwarePacer(base_rpm=config.rpm)
    contracts_map = get_contract_map()
    findings: list[dict[str, Any]] = []

    exp_out_dir = store.run_dir / "experiments"
    exp_out_dir.mkdir(parents=True, exist_ok=True)
    ctx_path = store.run_dir / "active_experiment_context.json"

    async with InferenceClient(
        base_url=config.nebius_base_url,
        api_key=api_key,
        pacer=pacer,
        connect_timeout_s=config.connect_timeout_s,
        read_timeout_s=config.read_timeout_s,
    ) as client:
        for cluster in clusters:
            console.print(f"[bold cyan]Investigating cluster {cluster.cluster_id}[/bold cyan] ({cluster.fatal_codes})")
            base_budget = BudgetGuard(config)
            cluster_cap = min(config.budget_usd, base_budget.spent_usd + config.cluster_budget_usd)
            cluster_budget = BudgetGuard(config, cap_override_usd=cluster_cap)
            contract = contracts_map[cluster.contract_id]

            async def _score_candidate(cand_req: dict[str, Any], k: int) -> tuple[int, int]:
                send_req = {k_name: v for k_name, v in cand_req.items() if not k_name.startswith("_")}
                model_id = str(send_req.get("model", cluster.models_affected[0]))
                fails = 0
                trials_done = 0
                for _ in range(k):
                    est = config.estimate_cost(model_id, 400, 1000)
                    try:
                        await cluster_budget.reserve(est)
                    except BudgetExceededError:
                        break
                    resp = await client.chat_completion(send_req, estimated_tokens=1400)
                    ores = contract.check(cand_req, resp)
                    actual = cluster_budget.compute_call_cost(model_id, resp.usage)
                    await cluster_budget.commit(est, actual, model_id, "shrink", store.run_id, resp.usage)
                    trials_done += 1
                    if ores.outcome.value == "fail":
                        fails += 1
                return fails, max(1, trials_done)

            # Step 2: Shrink
            minimal_req, shrunk_rate, shrink_history = await shrink_cluster_request(
                initial_req=cluster.representative_request,
                contract_id=cluster.contract_id,
                p0=cluster.failure_rate,
                score_fn=_score_candidate,
                k=12,
                max_candidates=8,
            )

            # Save context for testkit
            ctx_payload = {
                "run_id": store.run_id,
                "contract_id": cluster.contract_id,
                "cluster_budget_usd": config.cluster_budget_usd,
                "experiments_out_dir": str(exp_out_dir),
                "max_arm_n": 30,
                "clusters": {
                    cluster.cluster_id: {
                        "minimal_request": minimal_req,
                        "contract_id": cluster.contract_id,
                    }
                },
            }
            ctx_path.write_text(json.dumps(ctx_payload, indent=2), encoding="utf-8")

            # Step 3 & 4 & 5: Up to 2 rounds if all hypotheses rejected in Round 1 (§11.5)
            all_evaluated: list[dict[str, Any]] = []
            rejected_context: list[dict[str, Any]] = []
            h_counter = 0

            for round_idx in range(2):
                try:
                    hypotheses = await propose_hypotheses(
                        cluster=cluster,
                        minimal_req=minimal_req,
                        config=config,
                        client=client,
                        budget=cluster_budget,
                        run_id=store.run_id,
                        rejected_prior=rejected_context if round_idx > 0 else None,
                        hypothesis_model=hypothesis_model,
                    )
                except BudgetExceededError:
                    break

                round_results: list[dict[str, Any]] = []
                for h in hypotheses:
                    h_counter += 1
                    try:
                        exp_res = await generate_and_run_experiment(
                            cluster_id=cluster.cluster_id,
                            h_idx=h_counter,
                            hypothesis=h,
                            config=config,
                            client=client,
                            budget=cluster_budget,
                            run_id=store.run_id,
                            context_path=ctx_path,
                            experiments_out_dir=exp_out_dir,
                            code_model=code_model,
                            n_per_arm=30,
                        )
                        round_results.append(exp_res)
                    except BudgetExceededError:
                        break

                evaluated = evaluate_experiment_decisions(
                    round_results, alpha=config.alpha, min_effect_size=0.05
                )
                all_evaluated.extend(evaluated)
                if any(e.get("accepted") for e in evaluated):
                    break
                rejected_context.extend(evaluated)

            finding = build_finding_record(
                cluster=cluster,
                minimal_req=minimal_req,
                shrunk_rate=shrunk_rate,
                evaluated_experiments=all_evaluated,
                base_url=config.nebius_base_url,
            )
            finding["shrink_history"] = shrink_history
            findings.append(finding)

    findings_path = store.run_dir / "findings.json"
    findings_path.write_text(json.dumps(findings, indent=2), encoding="utf-8")
    store.write_summary({"run_id": store.run_id, "findings_count": len(findings)})
    return findings

