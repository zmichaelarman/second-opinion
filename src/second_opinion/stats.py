"""Statistics (CLAUDE.md §9). Unit-tested against known values."""

from __future__ import annotations

import math

import numpy as np
from scipy import stats as st

Z95 = 1.959964


def wilson_ci(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    lo = 0.0 if k == 0 else float(st.beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(st.beta.ppf(1 - alpha / 2, k + 1, n - k))
    return (lo, hi)


def rule_of_three(n: int) -> float:
    """95% upper bound on the failure rate after n trials with zero failures."""
    return 3.0 / n if n > 0 else 1.0


def fisher_two_sided(f1: int, n1: int, f2: int, n2: int) -> float:
    """Two-sided Fisher exact p-value for f1/n1 vs f2/n2 failures."""
    return float(st.fisher_exact([[f1, n1 - f1], [f2, n2 - f2]], alternative="two-sided").pvalue)


def holm(pvals: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, in the input order."""
    m = len(pvals)
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(sorted(range(m), key=lambda j: pvals[j])):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adjusted[i] = running
    return adjusted


def bootstrap_quantile_ci(
    x: list[float] | np.ndarray, q: float, B: int = 10000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile bootstrap CI for the q-quantile (used for latency, which is reported, not headlined)."""
    arr = np.asarray(x, dtype=float)
    if arr.size == 0:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    estimates = np.empty(B)
    chunk = max(1, 2_000_000 // arr.size)  # bound memory for large samples
    for start in range(0, B, chunk):
        stop = min(B, start + chunk)
        idx = rng.integers(0, arr.size, size=(stop - start, arr.size))
        estimates[start:stop] = np.quantile(arr[idx], q, axis=1)
    return (float(np.quantile(estimates, alpha / 2)), float(np.quantile(estimates, 1 - alpha / 2)))


def mann_whitney_u(a: list[float], b: list[float]) -> float:
    """Two-sided Mann-Whitney U p-value (Tier 3: does low_effort reduce reasoning tokens?)."""
    return float(st.mannwhitneyu(a, b, alternative="two-sided").pvalue)
