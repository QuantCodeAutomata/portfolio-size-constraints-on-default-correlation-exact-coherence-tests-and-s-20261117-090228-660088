"""Asymptotic correlation-boundary expansion.

Implements Theorem 7.4 of the paper "Portfolio-Size Constraints on Default
Correlation: Exact Coherence Tests and Sharp Asymptotic Bounds": for fixed
0 < p0 < p1 < 1 and 0 < rho0 < rho_bar < 1, as n -> infinity the admissible
interval boundaries satisfy
    ell_n = rho0 - C/sqrt(log n) + O(log log n / log n),
    u_n   = rho0 + C/sqrt(log n) + O(log log n / log n),
with
    C = rho0 * sqrt(2*(1 - rho0)) * [c(p1) - c(p0)]   (Eq. 27).
"""

from __future__ import annotations

from src.gaussian_mixture import c_of_p


def boundary_constant(p0: float, p1: float, rho0: float) -> float:
    """C = rho0*sqrt(2*(1-rho0))*(c(p1) - c(p0)) (paper Eq. 27)."""
    if not (0.0 < rho0 < 1.0):
        raise ValueError(f"rho0 must lie in (0, 1), got {rho0}")
    c1, c0 = c_of_p(p1), c_of_p(p0)
    return float(rho0 * (2.0 * (1.0 - rho0)) ** 0.5 * (c1 - c0))
