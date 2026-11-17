"""Admissible correlation interval and sequential repair.

Implements Theorem 5.2 (the admissible later-correlation interval
I_n(p0, rho0; p1) = [ell_n, u_n]) and Algorithm 1 (feasible correlation band
and one-step projection), together with the multi-horizon sequential repair
of Eq. (18), Section 5.1, of the paper "Portfolio-Size Constraints on Default
Correlation: Exact Coherence Tests and Sharp Asymptotic Bounds".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize

from src import config
from src.gaussian_mixture import log_orthant_probability, log_survival_probability

_TOL_KEYS = ("quad_abstol", "quad_reltol", "mode_root_abstol", "half_width")


def _tol_kwargs(**overrides) -> dict:
    defaults = dict(
        quad_abstol=config.QUAD_ABSTOL,
        quad_reltol=config.QUAD_RELTOL,
        mode_root_abstol=config.MODE_ROOT_ABSTOL,
        half_width=config.MODE_INTEGRATION_HALF_WIDTH,
    )
    defaults.update({k: v for k, v in overrides.items() if k in _TOL_KEYS})
    return defaults


def admissible_interval(
    n: int,
    p0: float,
    rho0: float,
    p1: float,
    rho_bar: float = config.RHO_BAR,
    root_abstol: float = config.ROOT_ABSTOL,
    **tol_overrides,
) -> dict:
    """Theorem 5.2: compute I_n(p0, rho0; p1) = [ell_n, u_n].

    Algorithm 1, steps 1-3:
      1. a0 = a_n(p0, rho0), s0 = s_n(p0, rho0).
      2. ell_n = 0 if n*log(p1) >= a0, else the root of a_n(p1, r) = a0 on (0, rho0].
      3. u_n = rho_bar if s_n(p1, rho_bar) <= s0, else the root of s_n(p1, r) = s0
         on [rho0, rho_bar).

    Returns a dict with keys 'ell_n', 'u_n', 'a0', 's0'.
    """
    if p1 < p0:
        raise ValueError("Theorem 5.2 requires a nondecreasing cumulative probability p1 >= p0")
    tol = _tol_kwargs(**tol_overrides)
    a0 = log_orthant_probability(n, p0, rho0, **tol)
    s0 = log_survival_probability(n, p0, rho0, **tol)

    # Lower boundary ell_n (Eq. 16). a_n(p1, .) is strictly increasing in r
    # (Lemma 5.1), so when a_n(p1, 0) = n*log(p1) already exceeds a0 the root
    # lies at or below 0 and ell_n = 0; otherwise, since a_n(p1, rho0) >= a0
    # (p1 >= p0 and a_n increasing in p), a unique root exists on (0, rho0].
    if rho0 == 0.0 or n * np.log(p1) >= a0:
        ell_n = 0.0
    else:
        def f_lower(r: float) -> float:
            return log_orthant_probability(n, p1, r, **tol) - a0

        ell_n = optimize.brentq(f_lower, 1e-14, rho0, xtol=root_abstol, rtol=8.9e-16)

    # Upper boundary u_n (Eq. 17).
    s_at_bar = log_survival_probability(n, p1, rho_bar, **tol)
    if s_at_bar <= s0:
        u_n = rho_bar
    else:
        def f_upper(r: float) -> float:
            return log_survival_probability(n, p1, r, **tol) - s0

        u_n = optimize.brentq(f_upper, rho0, rho_bar, xtol=root_abstol, rtol=8.9e-16)

    return {"ell_n": float(ell_n), "u_n": float(u_n), "a0": float(a0), "s0": float(s0)}


def project_correlation(rho1_raw: float, ell_n: float, u_n: float) -> float:
    """rho1* = min(u_n, max(ell_n, rho1_raw)) (Eq. 18 / Algorithm 1, step 4)."""
    return float(min(u_n, max(ell_n, rho1_raw)))


def repair_correlation(
    n: int,
    p0: float,
    rho0: float,
    p1: float,
    rho1_raw: float,
    rho_bar: float = config.RHO_BAR,
    root_abstol: float = config.ROOT_ABSTOL,
    **tol_overrides,
) -> dict:
    """Algorithm 1 (full one-step repair): interval plus the projected rho1*."""
    interval = admissible_interval(
        n, p0, rho0, p1, rho_bar, root_abstol=root_abstol, **tol_overrides
    )
    rho1_star = project_correlation(rho1_raw, interval["ell_n"], interval["u_n"])
    return {**interval, "rho1_star": rho1_star}


def sequential_repair(
    n: int,
    p_grid: list[float],
    rho_raw_grid: list[float],
    rho_bar: float = config.RHO_BAR,
    root_abstol: float = config.ROOT_ABSTOL,
    **tol_overrides,
) -> pd.DataFrame:
    """Multi-horizon sequential repair (Eq. 18, Section 5.1).

    rho_0* = rho_raw_grid[0] (no repair at the first horizon). For each
    subsequent horizon j, the interval [ell_j, u_j] is computed from the
    *preceding repaired* correlation rho_{j-1}* and the projection
    rho_j* = min(u_j, max(ell_j, rho_raw_grid[j])) is applied.

    Returns a DataFrame with one row per horizon and columns:
    'horizon', 'p', 'raw_rho', 'repaired_rho', 'ell', 'u'.
    """
    if len(p_grid) != len(rho_raw_grid):
        raise ValueError("p_grid and rho_raw_grid must have the same length")
    if any(p_grid[j + 1] < p_grid[j] for j in range(len(p_grid) - 1)):
        raise ValueError("p_grid must be nondecreasing across horizons")

    rows = []
    rho_prev_star = float(rho_raw_grid[0])
    rows.append(
        {
            "horizon": 0,
            "p": p_grid[0],
            "raw_rho": rho_raw_grid[0],
            "repaired_rho": rho_prev_star,
            "ell": np.nan,
            "u": np.nan,
        }
    )
    for j in range(1, len(p_grid)):
        result = repair_correlation(
            n,
            p_grid[j - 1],
            rho_prev_star,
            p_grid[j],
            rho_raw_grid[j],
            rho_bar,
            root_abstol=root_abstol,
            **tol_overrides,
        )
        rho_prev_star = result["rho1_star"]
        rows.append(
            {
                "horizon": j,
                "p": p_grid[j],
                "raw_rho": rho_raw_grid[j],
                "repaired_rho": rho_prev_star,
                "ell": result["ell_n"],
                "u": result["u_n"],
            }
        )
    return pd.DataFrame(rows)
