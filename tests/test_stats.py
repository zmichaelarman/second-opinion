from __future__ import annotations

import math

from second_opinion.stats import (
    bootstrap_quantile_ci,
    clopper_pearson,
    fisher_two_sided,
    holm,
    rule_of_three,
    trust_matrix_status,
    wilson_ci,
)


def test_wilson_ci_known_values() -> None:
    lo0, hi0 = wilson_ci(0, 50)
    assert lo0 == 0.0
    assert math.isclose(hi0, 0.07135, rel_tol=1e-2)

    lo1, hi1 = wilson_ci(25, 100)
    assert math.isclose(lo1, 0.1754, rel_tol=1e-2)
    assert math.isclose(hi1, 0.3431, rel_tol=1e-2)

    lo_full, hi_full = wilson_ci(50, 50)
    assert hi_full == 1.0
    assert lo_full > 0.92


def test_clopper_pearson_known_values() -> None:
    lo0, hi0 = clopper_pearson(0, 50)
    assert lo0 == 0.0
    assert math.isclose(hi0, 0.0711, rel_tol=1e-2)

    lo_half, hi_half = clopper_pearson(5, 10)
    assert math.isclose(lo_half, 0.18709, rel_tol=1e-2)
    assert math.isclose(hi_half, 0.81291, rel_tol=1e-2)


def test_rule_of_three() -> None:
    assert math.isclose(rule_of_three(300), 0.01)
    assert math.isclose(rule_of_three(50), 0.06)
    assert rule_of_three(0) == 1.0


def test_fisher_two_sided_and_holm() -> None:
    # Identical proportions -> p = 1.0
    assert math.isclose(fisher_two_sided(5, 50, 5, 50), 1.0)

    # Strong difference -> very small p-value
    p_sig = fisher_two_sided(25, 30, 1, 30)
    assert p_sig < 1e-6

    # Holm-Bonferroni correction preserves original order and enforces monotonicity
    raw = [0.04, 0.001, 0.015]
    adj = holm(raw)
    # Sorted: 0.001*3=0.003, 0.015*2=0.030, 0.04*1=0.040
    assert math.isclose(adj[1], 0.003)
    assert math.isclose(adj[2], 0.030)
    assert math.isclose(adj[0], 0.040)


def test_bootstrap_quantile_ci() -> None:
    data = [float(i) for i in range(1, 101)]
    lo, hi = bootstrap_quantile_ci(data, 0.5, B=2000, seed=0)
    assert 35.0 <= lo <= 50.5 <= hi <= 65.0
    # Deterministic with same seed
    assert bootstrap_quantile_ci(data, 0.5, B=2000, seed=0) == (lo, hi)


def test_trust_matrix_color_rules() -> None:
    # Green: 0 failures across pooled cells
    st_green = trust_matrix_status([(0, 100), (0, 200)])
    assert st_green["color"] == "green"
    assert "1.0%" in str(st_green["bound_str"])

    # Amber: 1 failure in 300 trials (Wilson lower bound ~0.06% <= 1.0%)
    st_amber = trust_matrix_status([(1, 300), (0, 300)])
    assert st_amber["color"] == "amber"

    # Red: 15 failures in 100 trials (Wilson lower bound ~9.3% > 1.0%)
    st_red = trust_matrix_status([(15, 100), (0, 300)])
    assert st_red["color"] == "red"
