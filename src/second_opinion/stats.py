from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy import stats


def wilson_ci(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for binomial proportion k/n."""
    if n <= 0:
        return (0.0, 1.0)
    k = max(0, min(k, n))
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt((p * (1.0 - p) / n) + (z2 / (4.0 * n * n)))
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    if k == 0:
        lo = 0.0
    if k == n:
        hi = 1.0
    return (lo, hi)


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact Clopper-Pearson binomial confidence interval."""
    if n <= 0:
        return (0.0, 1.0)
    k = max(0, min(k, n))
    lo = 0.0 if k == 0 else float(stats.beta.ppf(alpha / 2.0, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return (lo, hi)


def rule_of_three(n: int) -> float:
    """Upper 95% bound when 0 failures are observed in n trials: 3/n."""
    if n <= 0:
        return 1.0
    return min(1.0, 3.0 / float(n))


def fisher_two_sided(f1: int, n1: int, f2: int, n2: int) -> float:
    """Two-sided Fisher's exact test for two binomial proportions f1/n1 vs f2/n2."""
    if n1 <= 0 or n2 <= 0:
        return 1.0
    f1 = max(0, min(f1, n1))
    f2 = max(0, min(f2, n2))
    p1 = n1 - f1
    p2 = n2 - f2
    table = [[f1, p1], [f2, p2]]
    _, pval = stats.fisher_exact(table, alternative="two-sided")
    return float(pval)


def holm(pvals: Sequence[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values preserving original order."""
    m = len(pvals)
    if m == 0:
        return []
    indexed = sorted(enumerate(pvals), key=lambda x: x[1])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj = p * (m - rank)
        running_max = max(running_max, adj)
        adjusted[orig_idx] = min(1.0, running_max)
    return adjusted


def bootstrap_quantile_ci(
    x: Sequence[float],
    q: float,
    B: int = 10000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Bootstrap confidence interval for quantile q (in [0, 1])."""
    arr = np.asarray(x, dtype=float)
    if arr.size == 0:
        return (0.0, 0.0)
    if arr.size == 1:
        val = float(arr[0])
        return (val, val)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(B, arr.size))
    samples = arr[idx]
    quantiles = np.quantile(samples, q, axis=1)
    lo = float(np.quantile(quantiles, alpha / 2.0))
    hi = float(np.quantile(quantiles, 1.0 - alpha / 2.0))
    return (lo, hi)


def trust_matrix_status(cells: Sequence[tuple[int, int]]) -> dict[str, object]:
    """Evaluate a capability column across constituent cells (k_failures, n_trials).

    Rules (§9):
    - Red: any cell whose failure-rate Wilson lower bound exceeds 1% (> 0.01).
    - Amber: failures observed, but not red.
    - Green: zero fatal failures. Show the bound 'failure rate < rule_of_three(n_pooled)'.
    """
    n_pooled = sum(n for _, n in cells if n > 0)
    k_pooled = sum(k for k, n in cells if n > 0)
    if n_pooled == 0:
        return {
            "color": "gray",
            "k_pooled": 0,
            "n_pooled": 0,
            "worst_wilson_lo": 0.0,
            "bound_str": "no data",
        }

    worst_lo = 0.0
    worst_rate = 0.0
    for k, n in cells:
        if n <= 0:
            continue
        lo, _ = wilson_ci(k, n)
        worst_lo = max(worst_lo, lo)
        worst_rate = max(worst_rate, k / n)

    if worst_lo > 0.01:
        color = "red"
        bound_str = f"worst cell {worst_rate * 100:.1f}% (CI lower {worst_lo * 100:.1f}%)"
    elif k_pooled > 0:
        color = "amber"
        bound_str = f"{k_pooled}/{n_pooled} ({k_pooled / n_pooled * 100:.1f}%)"
    else:
        color = "green"
        r3 = rule_of_three(n_pooled)
        bound_str = f"failure rate < {r3 * 100:.1f}% (0/{n_pooled})"

    return {
        "color": color,
        "k_pooled": k_pooled,
        "n_pooled": n_pooled,
        "worst_wilson_lo": worst_lo,
        "worst_rate": worst_rate,
        "bound_str": bound_str,
    }
