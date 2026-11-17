"""Unit tests for src/count_tail.py: full count-tail cross-validation.

Reference: "Portfolio-Size Constraints on Default Correlation: Exact Coherence
Tests and Sharp Asymptotic Bounds", Section 8.1 (full-tail integration) and
Section 8.2 (cross-validation of the two-event criterion, Theorem 4.1).
"""
from __future__ import annotations

import numpy as np
import pytest

from src.count_tail import (
    count_tail_probabilities,
    first_incompatible_n,
    incompatibility_measure,
    two_event_criterion_holds,
)
from src.gaussian_mixture import orthant_probability, survival_probability


def test_count_tail_probabilities_are_nonincreasing_in_k():
    tail = count_tail_probabilities(n=20, p=0.1, rho=0.2)
    assert all(tail[i] >= tail[i + 1] - 1e-10 for i in range(len(tail) - 1))


def test_count_tail_endpoints_match_orthant_and_one_minus_survival():
    n, p, rho = 15, 0.05, 0.25
    tail = count_tail_probabilities(n, p, rho)
    # P(K >= n) = A_n(p, rho); P(K >= 1) = 1 - S_n(p, rho).
    assert tail[-1] == pytest.approx(orthant_probability(n, p, rho), abs=1e-7)
    assert tail[0] == pytest.approx(1.0 - survival_probability(n, p, rho), abs=1e-7)


def test_incompatibility_measure_zero_when_tails_equal():
    tail = count_tail_probabilities(n=10, p=0.05, rho=0.1)
    assert incompatibility_measure(tail, tail) == pytest.approx(0.0, abs=1e-12)


def test_incompatibility_measure_nonnegative():
    tail0 = count_tail_probabilities(n=10, p=0.02, rho=0.1)
    tail1 = count_tail_probabilities(n=10, p=0.025, rho=0.4)
    v = incompatibility_measure(tail0, tail1)
    assert v >= 0.0


def test_incompatibility_measure_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        incompatibility_measure(np.zeros(3), np.zeros(4))


def test_two_event_criterion_holds_for_dominant_pair():
    """A pair with higher p and higher rho that still raises both extreme
    events should satisfy the stochastic-ordering criterion."""
    assert two_event_criterion_holds(
        n=20, p0=0.01, rho0=0.05, p1=0.02, rho1=0.05
    )


def test_first_incompatible_n_matches_paper_section_8_3():
    """Paper Sec. 8.3: the pair (p0,rho0)=(0.02,0.10) -> (p1,rho1)=(0.025,0.40)
    first becomes incompatible at n = 7."""
    n_star = first_incompatible_n(0.02, 0.10, 0.025, 0.40, n_max=50)
    assert n_star == 7


def test_cross_validation_agrees_with_direct_criterion():
    """The two-event criterion from log-orthant probabilities must agree with
    thresholding the full count-tail probabilities directly, for several
    random parameter pairs (paper Sec. 8.2, Step 5)."""
    rng = np.random.default_rng(12345)
    n = 8
    for _ in range(5):
        p0, p1 = sorted(rng.uniform(0.05, 0.6, size=2))
        rho0, rho1 = rng.uniform(0.0, 0.9, size=2)
        criterion = two_event_criterion_holds(n, p0, rho0, p1, rho1)
        tail0 = count_tail_probabilities(n, p0, rho0)
        tail1 = count_tail_probabilities(n, p1, rho1)
        full_tail_holds = bool(np.all(tail1 >= tail0 - 1e-6))
        assert criterion == full_tail_holds
