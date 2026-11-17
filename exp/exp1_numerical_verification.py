"""Numerical verification of the log-domain quadrature and root finding.

Reproduces Section 8.2 ("Numerical verification") of the paper
"Portfolio-Size Constraints on Default Correlation: Exact Coherence Tests and
Sharp Asymptotic Bounds":

1. The closed-form identity A_n(1/2, 1/2) = 1/(n+1) (the mixing variable
   Q(1/2, 1/2, Z) is uniform on (0,1) when p = rho = 1/2).
2. The closed-form identity A_2(1/2, rho) = 1/4 + arcsin(rho)/(2*pi).
3. Cross-validation of the two-event criterion (Theorem 4.1) against
   independent full count-tail probability integration, for n = 8, 25, 60.
4. A secondary accuracy run with tightened quadrature/root tolerances and an
   expanded mode-centered integration interval.

Run as: python -m exp.exp1_numerical_verification
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from src import config
from src.correlation_bounds import admissible_interval
from src.count_tail import count_tail_probabilities, two_event_criterion_holds
from src.gaussian_mixture import (
    log_orthant_probability,
    orthant_probability,
)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def identity_uniform_mixture() -> pd.DataFrame:
    """A_n(1/2, 1/2) = 1/(n+1) for the n values in config.EXP1_N_VALUES_UNIFORM_IDENTITY."""
    rows = []
    for n in config.EXP1_N_VALUES_UNIFORM_IDENTITY:
        a = orthant_probability(n, 0.5, 0.5)
        expected = 1.0 / (n + 1)
        rows.append(
            {
                "n": n,
                "A_n_computed": a,
                "A_n_expected": expected,
                "abs_error_prob": abs(a - expected),
                "abs_error_log": abs(np.log(a) - np.log(expected)),
            }
        )
    return pd.DataFrame(rows)


def identity_arcsin() -> pd.DataFrame:
    """A_2(1/2, rho) = 1/4 + arcsin(rho)/(2*pi) for config.EXP1_RHO_VALUES_ARCSIN_IDENTITY."""
    rows = []
    for rho in config.EXP1_RHO_VALUES_ARCSIN_IDENTITY:
        a = orthant_probability(2, 0.5, rho)
        expected = 0.25 + np.arcsin(rho) / (2.0 * np.pi)
        rows.append(
            {
                "rho": rho,
                "A_2_computed": a,
                "A_2_expected": expected,
                "abs_error_prob": abs(a - expected),
                "abs_error_log": abs(np.log(a) - np.log(expected)),
            }
        )
    return pd.DataFrame(rows)


def cross_validate_two_event_criterion() -> pd.DataFrame:
    """Compare Theorem 4.1 (two-event criterion) against full count-tail integration.

    For each portfolio size n in config.EXP1_CROSS_CHECK_N_VALUES, draws
    config.EXP1_CROSS_CHECK_PAIRS_PER_N random parameter pairs
    (p0, rho0) -> (p1, rho1) and checks that the two-event criterion's
    stochastic-ordering verdict agrees with the verdict obtained by directly
    comparing all n count-tail probabilities P(K0 >= k) vs P(K1 >= k).
    """
    rng = np.random.default_rng(config.EXP1_CROSS_CHECK_SEED)
    rows = []
    for n in config.EXP1_CROSS_CHECK_N_VALUES:
        for _ in range(config.EXP1_CROSS_CHECK_PAIRS_PER_N):
            p0, p1 = sorted(rng.uniform(0.01, 0.5, size=2))
            rho0, rho1 = rng.uniform(0.0, 0.9, size=2)
            criterion_holds = two_event_criterion_holds(n, p0, rho0, p1, rho1)
            tail0 = count_tail_probabilities(n, p0, rho0)
            tail1 = count_tail_probabilities(n, p1, rho1)
            # Direct stochastic-order check: K0 <=_st K1 iff P(K0>=k)<=P(K1>=k) for all k.
            full_tail_holds = bool(np.all(tail1 - tail0 >= -1e-8))
            rows.append(
                {
                    "n": n,
                    "p0": p0,
                    "rho0": rho0,
                    "p1": p1,
                    "rho1": rho1,
                    "criterion_holds": criterion_holds,
                    "full_tail_holds": full_tail_holds,
                    "agree": criterion_holds == full_tail_holds,
                }
            )
    return pd.DataFrame(rows)


def secondary_accuracy_run() -> dict:
    """Tightened-tolerance re-evaluation (paper Sec. 8.2, Step 6).

    Recomputes a representative set of log-orthant probabilities and the
    exp_3 admissible-correlation boundaries with the tightened tolerances of
    config.*_TIGHT and reports the largest absolute changes.
    """
    tight = dict(
        quad_abstol=config.QUAD_ABSTOL_TIGHT,
        quad_reltol=config.QUAD_RELTOL_TIGHT,
        mode_root_abstol=config.MODE_ROOT_ABSTOL_TIGHT,
        half_width=config.MODE_INTEGRATION_HALF_WIDTH_TIGHT,
    )
    probe_points = [
        (10, 0.02, 0.10),
        (100, 0.02, 0.10),
        (100, 0.025, 0.40),
        (1000, 0.03, 0.15),
        (50, 0.065, 0.30),
    ]
    log_a_changes = []
    for n, p, rho in probe_points:
        baseline = log_orthant_probability(n, p, rho)
        tightened = log_orthant_probability(n, p, rho, **tight)
        log_a_changes.append(abs(baseline - tightened))

    boundary_changes = []
    for n in config.EXP3_N_VALUES:
        baseline = admissible_interval(n, config.EXP3_P0, config.EXP3_RHO0, config.EXP3_P1)
        tightened = admissible_interval(
            n, config.EXP3_P0, config.EXP3_RHO0, config.EXP3_P1,
            root_abstol=config.ROOT_ABSTOL_TIGHT, **tight,
        )
        boundary_changes.append(abs(baseline["ell_n"] - tightened["ell_n"]))
        boundary_changes.append(abs(baseline["u_n"] - tightened["u_n"]))

    return {
        "max_abs_change_log_orthant": float(max(log_a_changes)),
        "max_abs_change_correlation_boundary": float(max(boundary_changes)),
        "probe_points": probe_points,
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    uniform_df = identity_uniform_mixture()
    arcsin_df = identity_arcsin()
    cross_df = cross_validate_two_event_criterion()
    secondary = secondary_accuracy_run()

    uniform_df.to_csv(RESULTS_DIR / "exp1_identity_uniform.csv", index=False)
    arcsin_df.to_csv(RESULTS_DIR / "exp1_identity_arcsin.csv", index=False)
    cross_df.to_csv(RESULTS_DIR / "exp1_cross_validation.csv", index=False)

    max_err_prob = max(uniform_df["abs_error_prob"].max(), arcsin_df["abs_error_prob"].max())
    max_err_log = max(uniform_df["abs_error_log"].max(), arcsin_df["abs_error_log"].max())
    all_agree = bool(cross_df["agree"].all())
    n_disagree = int((~cross_df["agree"]).sum())

    summary = {
        "n_uniform_identities": len(uniform_df),
        "n_arcsin_identities": len(arcsin_df),
        "max_abs_error_probability_scale": float(max_err_prob),
        "max_abs_error_log_scale": float(max_err_log),
        "n_cross_validation_pairs": len(cross_df),
        "cross_validation_all_agree": all_agree,
        "cross_validation_n_disagreements": n_disagree,
        "secondary_accuracy_run": secondary,
    }
    with open(RESULTS_DIR / "exp1_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("exp_1 summary:")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
