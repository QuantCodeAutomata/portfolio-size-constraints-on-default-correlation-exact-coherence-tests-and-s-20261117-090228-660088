"""Unit tests for src/correlation_bounds.py (Theorem 5.2 and Algorithm 1).

Reference: "Portfolio-Size Constraints on Default Correlation: Exact Coherence
Tests and Sharp Asymptotic Bounds", Section 5.1 and Section 8.
"""
from __future__ import annotations

import pytest

from src.correlation_bounds import (
    admissible_interval,
    project_correlation,
    repair_correlation,
    sequential_repair,
)


def test_admissible_interval_contains_rho0():
    """I_n(p0, rho0; p1) must contain rho0 itself (Theorem 5.2, since rho1=rho0
    with p1>=p0 keeps a_n(p1,rho0)>=a0 and s_n(p1,rho0)<=s0 only at p1=p0;
    more generally ell_n <= rho0 <= u_n must hold by construction)."""
    result = admissible_interval(n=100, p0=0.02, rho0=0.10, p1=0.025)
    assert result["ell_n"] <= 0.10 <= result["u_n"]


def test_admissible_interval_rejects_decreasing_p():
    with pytest.raises(ValueError):
        admissible_interval(n=50, p0=0.05, rho0=0.1, p1=0.02)


def test_exp2_values_match_paper_section_8_3():
    """Paper Sec. 8.3: n=100, (p0,rho0)=(0.02,0.10), p1=0.025 gives a repaired
    correlation of approximately 0.153059."""
    result = repair_correlation(n=100, p0=0.02, rho0=0.10, p1=0.025, rho1_raw=0.40)
    assert result["rho1_star"] == pytest.approx(0.153059, abs=5e-5)
    assert result["ell_n"] < result["rho1_star"] <= result["u_n"]


def test_project_correlation_clips_to_bounds():
    assert project_correlation(0.5, 0.1, 0.3) == pytest.approx(0.3)
    assert project_correlation(0.05, 0.1, 0.3) == pytest.approx(0.1)
    assert project_correlation(0.2, 0.1, 0.3) == pytest.approx(0.2)


def test_sequential_repair_first_horizon_unrepaired():
    df = sequential_repair(
        n=100, p_grid=[0.01, 0.018], rho_raw_grid=[0.12, 0.30]
    )
    assert df.iloc[0]["repaired_rho"] == pytest.approx(0.12)


def test_sequential_repair_matches_paper_table4():
    """Paper Sec. 8.6, Table 4: five-horizon repaired correlations."""
    p_grid = [0.010, 0.018, 0.026, 0.033, 0.040]
    rho_raw_grid = [0.120, 0.300, 0.080, 0.250, 0.150]
    df = sequential_repair(n=100, p_grid=p_grid, rho_raw_grid=rho_raw_grid)
    expected_repaired = [0.120000, 0.291198, 0.269229, 0.254583, 0.242514]
    for computed, expected in zip(df["repaired_rho"], expected_repaired):
        assert computed == pytest.approx(expected, abs=5e-5)


def test_sequential_repair_requires_nondecreasing_p():
    with pytest.raises(ValueError):
        sequential_repair(n=50, p_grid=[0.02, 0.01], rho_raw_grid=[0.1, 0.1])


def test_sequential_repair_requires_matching_lengths():
    with pytest.raises(ValueError):
        sequential_repair(n=50, p_grid=[0.02, 0.03], rho_raw_grid=[0.1])
