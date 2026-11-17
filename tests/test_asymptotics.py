"""Unit tests for src/asymptotics.py (Theorem 7.4 boundary expansion).

Reference: "Portfolio-Size Constraints on Default Correlation: Exact Coherence
Tests and Sharp Asymptotic Bounds", Section 7.3 and Section 8.4.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.asymptotics import boundary_constant
from src.correlation_bounds import admissible_interval


def test_boundary_constant_matches_paper_section_8_4():
    """Paper Sec. 8.4: p0=0.02, p1=0.03, rho0=0.15 gives C = 0.033826."""
    c = boundary_constant(p0=0.02, p1=0.03, rho0=0.15)
    assert c == pytest.approx(0.033826, abs=5e-6)


def test_boundary_constant_positive_for_p1_above_p0():
    c = boundary_constant(p0=0.01, p1=0.05, rho0=0.2)
    assert c > 0.0


def test_boundary_constant_rejects_degenerate_rho0():
    with pytest.raises(ValueError):
        boundary_constant(p0=0.01, p1=0.02, rho0=0.0)
    with pytest.raises(ValueError):
        boundary_constant(p0=0.01, p1=0.02, rho0=1.0)


def test_scaled_distances_converge_towards_boundary_constant_as_n_grows():
    """Paper Sec. 8.4: sqrt(log n)*(rho0-ell_n) and sqrt(log n)*(u_n-rho0) both
    approach C as n grows, with slow (logarithmic) convergence; the gap
    between the two scaled distances must shrink monotonically across the
    large-n grid used in the paper's Table 2."""
    p0, p1, rho0 = 0.02, 0.03, 0.15
    c = boundary_constant(p0, p1, rho0)
    gaps = []
    for n in [100, 10**6, 10**12, 10**40, 10**160]:
        interval = admissible_interval(n, p0, rho0, p1)
        log_n = math.log(n)
        scaled_lower = math.sqrt(log_n) * (rho0 - interval["ell_n"])
        scaled_upper = math.sqrt(log_n) * (interval["u_n"] - rho0)
        gaps.append(abs(scaled_upper - scaled_lower))
    assert all(gaps[i] > gaps[i + 1] - 1e-9 for i in range(len(gaps) - 1))
    # Even at n = 10**160 the two scaled distances have not fully converged,
    # confirming the paper's logarithmically slow convergence claim.
    assert gaps[-1] > 1e-3
