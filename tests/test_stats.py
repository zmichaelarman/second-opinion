import pytest

from second_opinion.stats import (
    bootstrap_quantile_ci,
    clopper_pearson,
    fisher_two_sided,
    holm,
    mann_whitney_u,
    rule_of_three,
    wilson_ci,
)


def approx(pair, expected, tol=1e-4):
    return all(abs(a - b) < tol for a, b in zip(pair, expected))


def test_wilson_known_values():
    assert approx(wilson_ci(0, 10), (0.0, 0.2775))
    assert approx(wilson_ci(5, 10), (0.2366, 0.7634))
    assert approx(wilson_ci(10, 10), (0.7225, 1.0))
    assert wilson_ci(0, 0) == (0.0, 1.0)


def test_clopper_pearson_known_values():
    assert approx(clopper_pearson(0, 10), (0.0, 0.3085))
    assert approx(clopper_pearson(5, 10), (0.1871, 0.8129))
    assert approx(clopper_pearson(10, 10), (0.6915, 1.0))


def test_rule_of_three():
    assert rule_of_three(300) == pytest.approx(0.01)
    assert rule_of_three(0) == 1.0


def test_fisher_known_values():
    # Lady tasting tea: two-sided p = 34/70.
    assert fisher_two_sided(3, 4, 1, 4) == pytest.approx(34 / 70)
    # Perfect separation 10/10 vs 0/10: p = 2 / C(20, 10).
    assert fisher_two_sided(10, 10, 0, 10) == pytest.approx(2 / 184756)
    assert fisher_two_sided(5, 50, 5, 50) == pytest.approx(1.0)


def test_holm_known_values():
    assert holm([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])
    assert holm([]) == []


def test_bootstrap_quantile_ci():
    assert bootstrap_quantile_ci([2.0] * 50, 0.5) == (2.0, 2.0)
    data = list(range(1, 101))
    lo, hi = bootstrap_quantile_ci(data, 0.5, B=2000, seed=0)
    assert lo < 50.5 < hi and bootstrap_quantile_ci(data, 0.5, B=2000, seed=0) == (lo, hi)


def test_mann_whitney():
    assert mann_whitney_u([1, 2, 3, 4, 5] * 4, [10, 11, 12, 13, 14] * 4) < 0.001
