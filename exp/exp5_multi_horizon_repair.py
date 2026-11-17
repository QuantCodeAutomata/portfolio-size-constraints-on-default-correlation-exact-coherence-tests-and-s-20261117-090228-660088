"""Multi-horizon sequential repair of oscillating raw correlations.

Reproduces Section 8.6 ("Several horizons"), Table 4, of the paper
"Portfolio-Size Constraints on Default Correlation: Exact Coherence Tests and
Sharp Asymptotic Bounds": the sequential projection (Algorithm 1 / Eq. 18) is
applied to a five-horizon grid of nondecreasing cumulative default
probabilities and oscillating raw correlations, repairing them to a
compatible sequence while preserving each horizon's marginal default
probability.

Run as: python -m exp.exp5_multi_horizon_repair
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import config
from src.correlation_bounds import sequential_repair
from src.gaussian_mixture import log_orthant_probability, log_survival_probability

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    n = config.EXP5_N
    p_grid = config.EXP5_P_GRID
    rho_raw_grid = config.EXP5_RHO_RAW_GRID

    table = sequential_repair(n, p_grid, rho_raw_grid, rho_bar=config.RHO_BAR)

    raw_survival = [
        float(np.exp(log_survival_probability(n, p_grid[j], rho_raw_grid[j])))
        for j in range(len(p_grid))
    ]
    repaired_survival = [
        float(np.exp(log_survival_probability(n, p_grid[j], table["repaired_rho"].iloc[j])))
        for j in range(len(p_grid))
    ]
    table["raw_survival"] = raw_survival
    table["repaired_survival"] = repaired_survival
    table["repaired_all_default"] = [
        float(np.exp(log_orthant_probability(n, p_grid[j], table["repaired_rho"].iloc[j])))
        for j in range(len(p_grid))
    ]

    table.to_csv(RESULTS_DIR / "exp5_table4_multi_horizon.csv", index=False)

    repaired_survival_arr = np.array(repaired_survival)
    repaired_all_default_arr = table["repaired_all_default"].to_numpy()
    survival_nonincreasing = bool(np.all(np.diff(repaired_survival_arr) <= 1e-9))
    all_default_nondecreasing = bool(np.all(np.diff(repaired_all_default_arr) >= -1e-9))

    summary = {
        "n": n,
        "p_grid": p_grid,
        "raw_rho_grid": rho_raw_grid,
        "repaired_rho_grid": table["repaired_rho"].tolist(),
        "raw_survival": raw_survival,
        "repaired_survival": repaired_survival,
        "repaired_all_default": table["repaired_all_default"].tolist(),
        "repaired_survival_nonincreasing": survival_nonincreasing,
        "repaired_all_default_nondecreasing": all_default_nondecreasing,
    }
    with open(RESULTS_DIR / "exp5_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    fig, ax = plt.subplots(figsize=(8, 5))
    horizons = table["horizon"] + 1
    ax.plot(horizons, rho_raw_grid, "o--", label="raw rho")
    ax.plot(horizons, table["repaired_rho"], "s-", label="repaired rho*")
    ax.set_xlabel("horizon")
    ax.set_ylabel("correlation")
    ax.set_title("Sequential repair of oscillating raw correlations (five horizons)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp5_figure_multi_horizon.png", dpi=150)
    plt.close(fig)

    print("exp_5 summary:")
    print(json.dumps(summary, indent=2, default=str))
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
