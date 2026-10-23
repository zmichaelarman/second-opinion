from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any

from second_opinion.contracts.base import Severity
from second_opinion.oracles import code_severity, is_fatal_code


@dataclass
class FailureCluster:
    cluster_id: str
    contract_id: str
    fatal_codes: list[str]
    severity: Severity
    models_affected: list[str]
    failure_count: int
    total_trials: int
    failure_rate: float
    sample_failing_records: list[dict[str, Any]] = field(default_factory=list)
    sample_passing_records: list[dict[str, Any]] = field(default_factory=list)
    representative_request: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["severity"] = self.severity.name
        return d


def cluster_failures(records: list[dict[str, Any]]) -> list[FailureCluster]:
    """Group fatal failures by signature: (contract_id, sorted fatal codes) (§11.1).

    - Keep clusters with >= 3 failures, or >= 1 if any code is S1.
    - Rank by severity, then by (number of models affected * failure rate).
    """
    # Track total trials per (contract_id, model) for denominator
    trials_by_contract_model: dict[tuple[str, str], int] = {}
    passing_by_contract: dict[str, list[dict[str, Any]]] = {}
    grouped_fails: dict[tuple[str, tuple[str, ...]], list[dict[str, Any]]] = {}

    for rec in records:
        if not rec.get("is_final", True):
            continue
        cid = str(rec.get("contract_id", ""))
        model = str(rec.get("model", ""))
        if not cid or not model:
            continue
        trials_by_contract_model[(cid, model)] = trials_by_contract_model.get((cid, model), 0) + 1

        outcome = rec.get("outcome")
        if outcome == "pass":
            passing_by_contract.setdefault(cid, []).append(rec)
        elif outcome == "fail":
            codes = [str(c) for c in (rec.get("codes") or []) if is_fatal_code(str(c))]
            if not codes:
                continue
            sig = (cid, tuple(sorted(set(codes))))
            grouped_fails.setdefault(sig, []).append(rec)

    clusters: list[FailureCluster] = []
    for (cid, fatal_tuple), fail_recs in grouped_fails.items():
        fatal_list = list(fatal_tuple)
        sevs = [code_severity(c, docs_shape=True) or Severity.S2_LOUD for c in fatal_list]
        worst_sev = min(sevs, key=lambda s: s.value)
        has_s1 = worst_sev == Severity.S1_SILENT

        if len(fail_recs) < 3 and not (has_s1 and len(fail_recs) >= 1):
            continue

        models_affected = sorted({str(r.get("model", "")) for r in fail_recs})
        total_trials = sum(trials_by_contract_model.get((cid, m), 0) for m in models_affected)
        total_trials = max(total_trials, len(fail_recs))
        failure_rate = len(fail_recs) / total_trials

        sig_str = f"{cid}:{','.join(fatal_list)}"
        short_hash = hashlib.sha256(sig_str.encode("utf-8")).hexdigest()[:8]
        cluster_id = f"{cid.lower()}_{short_hash}"

        rep_req = dict(fail_recs[0].get("request") or {})
        rep_req["_contract_id"] = cid

        clusters.append(
            FailureCluster(
                cluster_id=cluster_id,
                contract_id=cid,
                fatal_codes=fatal_list,
                severity=worst_sev,
                models_affected=models_affected,
                failure_count=len(fail_recs),
                total_trials=total_trials,
                failure_rate=failure_rate,
                sample_failing_records=fail_recs[:3],
                sample_passing_records=(passing_by_contract.get(cid) or [])[:3],
                representative_request=rep_req,
            )
        )

    # Sort by severity ascending (S1=1 first), then by (len(models_affected) * failure_rate) descending
    clusters.sort(
        key=lambda c: (c.severity.value, -(len(c.models_affected) * c.failure_rate), -c.failure_count)
    )
    return clusters
