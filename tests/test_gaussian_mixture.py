"""Unit tests for src/gaussian_mixture.py against closed-form identities.

Reference: "Portfolio-Size Constraints on Default Correlation: Exact Coherence
Tests and Sharp Asymptotic Bounds", Section 8.2 (Numerical verification).
"""
from __future__ import annotations

import numpy as np
import pytest

from src.gaussian_mixture import (
    c_of_p,
    log_orthant_probability,
    log_survival_probability,
    mu_sigma,
    orthant_probability,
    survival_probability,
)


def test_c_of_p_matches_norm_ppf():
    assert c_of_p(0.5) == pytest.approx(0.0, abs=1e-12)
    assert c_of_p(0.975) == pytest.approx(1.9599639845, abs=1e-8)


def test_c_of_p_rejects_out_of_range():
    with pytest.raises(ValueError):
        c_of_p(0.0)
    with pytest.raises(ValueError):
        c_of_p(1.0)


def test_mu_sigma_rejects_rho_out_of_range():
    with pytest.raises(ValueError):
        mu_sigma(0.1, 1.0)
    with pytest.raises(ValueError):
        mu_sigma(0.1, -0.1)


def test_uniform_identity_A_n_half_half():
    """Paper Sec. 8.2: at p = rho = 1/2 the mixing variable is uniform on
    (0, 1), so A_n(1/2, 1/2) = 1/(n+1) exactly."""
    for n in [1, 2, 3, 5, 8, 13, 21, 50, 100, 250]:
        expected = 1.0 / (n + 1)
        computed = orthant_probability(n, 0.5, 0.5)
        assert computed == pytest.approx(expected, abs=1e-10)


def test_arcsin_identity_A_2():
    """Paper Sec. 8.2: A_2(1/2, rho) = 1/4 + arcsin(rho)/(2*pi)."""
    for rho in [0.001, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.97]:
        expected = 0.25 + np.arcsin(rho) / (2.0 * np.pi)
        computed = orthant_probability(2, 0.5, rho)
        assert computed == pytest.approx(expected, abs=1e-10)


def test_rho_zero_closed_form():
    """At rho = 0, defaults are independent Bernoulli(p): A_n(p,0) = p^n."""
    n, p = 37, 0.07
    assert orthant_probability(n, p, 1e-12) == pytest.approx(p**n, rel=1e-6)


def test_survival_is_complement_orthant_of_one_minus_p():
    n, p, rho = 20, 0.03, 0.2
    s = survival_probability(n, p, rho)
    a_complement = orthant_probability(n, 1.0 - p, rho)
    assert s == pytest.approx(a_complement, rel=1e-10)


def test_log_probabilities_are_negative_and_consistent_with_exp():
    n, p, rho = 50, 0.02, 0.15
    a = log_orthant_probability(n, p, rho)
    s = log_survival_probability(n, p, rho)
    assert a < 0.0
    assert s < 0.0
    assert np.exp(a) == pytest.approx(orthant_probability(n, p, rho), rel=1e-10)
    assert np.exp(s) == pytest.approx(survival_probability(n, p, rho), rel=1e-10)


def test_orthant_probability_increasing_in_p():
    """A_n(p, rho) must increase in p for fixed rho (Lemma 5.1 of the paper)."""
    n, rho = 30, 0.2
    values = [orthant_probability(n, p, rho) for p in [0.01, 0.05, 0.1, 0.2]]
    assert all(values[i] < values[i + 1] for i in range(len(values) - 1))


def test_orthant_probability_increasing_in_rho():
    """A_n(p, rho) must increase in rho for fixed p (Lemma 5.1 of the paper)."""
    n, p = 30, 0.1
    values = [orthant_probability(n, p, rho) for rho in [0.0, 0.1, 0.3, 0.5, 0.8]]
    assert all(values[i] < values[i + 1] for i in range(len(values) - 1))


def test_extreme_portfolio_size_does_not_overflow():
    """The log-domain quadrature must remain finite for n up to 10**160
    (paper Sec. 8.4, exp_3)."""
    a = log_orthant_probability(10**160, 0.02, 0.15)
    assert np.isfinite(a)
    assert a < 0.0
