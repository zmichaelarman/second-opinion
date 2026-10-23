"""Accept or reject causes (CLAUDE.md §11.5).

Two-sided Fisher exact test per hypothesis, Holm-corrected across the cluster's hypotheses.
A cause is accepted if its adjusted p < alpha (0.01) and the failure rates of its two levels
differ by at least 5 percentage points.
"""

from __future__ import annotations

import json
from typing import Any

from ..stats import fisher_two_sided, holm

MIN_EFFECT = 0.05


def decide(results: list[dict[str, Any]], alpha: float = 0.01, min_effect: float = MIN_EFFECT) -> list[dict[str, Any]]:
    """Each result has a, b ({n, k, rate, ...}) and the hypothesis. Adds p, p_holm, effect, accepted."""
    for r in results:
        a, b = r["a"], r["b"]
        r["p"] = fisher_two_sided(a["k"], a["n"], b["k"], b["n"]) if a["n"] and b["n"] else 1.0
        r["effect"] = abs((a["k"] / a["n"] if a["n"] else 0) - (b["k"] / b["n"] if b["n"] else 0))
    for r, adj in zip(results, holm([r["p"] for r in results])):
        r["p_holm"] = adj
        r["accepted"] = adj < alpha and r["effect"] >= min_effect
    return results


def workaround(result: dict[str, Any]) -> str:
    a, b = result["a"], result["b"]
    lo, hi = (a, b) if a["rate"] <= b["rate"] else (b, a)
    return (f"set {result['factor']} to {json.dumps(lo['level'])}: {lo['rate']:.0%} failures "
            f"vs {hi['rate']:.0%} at {json.dumps(hi['level'])}")


def cause(result: dict[str, Any]) -> str:
    return (f"{result['factor']} ({json.dumps(result['a']['level'])} vs {json.dumps(result['b']['level'])}): "
            f"{result.get('rationale', '')}").strip()
