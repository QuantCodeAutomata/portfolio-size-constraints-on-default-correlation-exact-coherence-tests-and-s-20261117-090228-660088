"""Sensitivity of admissible correlation intervals to default probabilities
and to the earlier correlation.

Reproduces Section 8.5 ("Sensitivity to default probabilities and initial
correlation"), Table 3, of the paper "Portfolio-Size Constraints on Default
Correlation: Exact Coherence Tests and Sharp Asymptotic Bounds": the exact
admissible interval [ell_n, u_n] for six (p0, p1, rho0) configurations, each
evaluated at n = 50 and n = 500.

Run as: python -m exp.exp4_sensitivity_analysis
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src import config
from src.correlation_bounds import admissible_interval

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for (p0, p1, rho0) in config.EXP4_CONFIGS:
        for n in config.EXP4_N_VALUES:
            interval = admissible_interval(n, p0, rho0, p1, rho_bar=config.RHO_BAR)
            rows.append(
                {
                    "p0": p0,
                    "p1": p1,
                    "rho0": rho0,
                    "n": n,
                    "ell_n": interval["ell_n"],
                    "u_n": interval["u_n"],
                    "width": interval["u_n"] - interval["ell_n"],
                    "downward_move": rho0 - interval["ell_n"],
                    "upward_move": interval["u_n"] - rho0,
                }
            )
    table3 = pd.DataFrame(rows)
    table3.to_csv(RESULTS_DIR / "exp4_table3_sensitivity.csv", index=False)

    # The upper endpoint is expected to move more than the lower one in every
    # configuration (paper Sec. 8.5).
    upper_moves_more = bool((table3["upward_move"] > table3["downward_move"]).all())

    summary = {
        "n_configurations": len(config.EXP4_CONFIGS),
        "n_portfolio_sizes": len(config.EXP4_N_VALUES),
        "upper_endpoint_moves_more_than_lower_in_all_rows": upper_moves_more,
    }
    with open(RESULTS_DIR / "exp4_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("exp_4 summary:")
    print(json.dumps(summary, indent=2, default=str))
    print(table3.to_string(index=False))


if __name__ == "__main__":
    main()
