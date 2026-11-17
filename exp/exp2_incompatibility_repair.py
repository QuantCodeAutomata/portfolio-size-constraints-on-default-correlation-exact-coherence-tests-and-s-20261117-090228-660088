"""Constructed incompatibility example and one-step repair for n = 100 borrowers.

Reproduces Section 8.3 ("Increasing mean defaults with an incompatible
distribution") of the paper "Portfolio-Size Constraints on Default
Correlation: Exact Coherence Tests and Sharp Asymptotic Bounds":

- An earlier forecast (p0, rho0) = (0.02, 0.10) and a raw later forecast
  (p1, rho1_raw) = (0.025, 0.40) for n = 100 borrowers.
- The raw later forecast violates absorption compatibility (the no-default
  probability S_n increases despite p1 > p0).
- The admissible interval and the one-step projection (Algorithm 1, Eq. 18)
  repair rho1 to the closest feasible value, restoring compatibility.
- The incompatibility diagnostic V_n (Eq. 3) and the first portfolio size at
  which the fixed pair (p0, rho0) -> (p1, rho1_raw) becomes incompatible.

Run as: python -m exp.exp2_incompatibility_repair
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import config
from src.correlation_bounds import admissible_interval, project_correlation
from src.count_tail import (
    count_tail_probabilities,
    first_incompatible_n,
    incompatibility_measure,
)
from src.gaussian_mixture import log_orthant_probability, log_survival_probability

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    n = config.EXP2_N
    p0, rho0 = config.EXP2_P0, config.EXP2_RHO0
    p1, rho1_raw = config.EXP2_P1, config.EXP2_RHO1_RAW

    a0 = log_orthant_probability(n, p0, rho0)
    s0 = log_survival_probability(n, p0, rho0)
    a1_raw = log_orthant_probability(n, p1, rho1_raw)
    s1_raw = log_survival_probability(n, p1, rho1_raw)

    survival_earlier = float(np.exp(s0))
    survival_raw = float(np.exp(s1_raw))
    monotonicity_violated = survival_raw > survival_earlier

    # Step 3-4: admissible interval and one-step projection (Algorithm 1).
    interval = admissible_interval(n, p0, rho0, p1, rho_bar=config.RHO_BAR)
    rho1_star = project_correlation(rho1_raw, interval["ell_n"], interval["u_n"])
    s1_star = log_survival_probability(n, p1, rho1_star)
    survival_repaired = float(np.exp(s1_star))

    # Step 5: full count-tail probabilities and the incompatibility measure V_100.
    tail0 = count_tail_probabilities(n, p0, rho0)
    tail1_raw = count_tail_probabilities(n, p1, rho1_raw)
    tail1_repaired = count_tail_probabilities(n, p1, rho1_star)
    v_raw = incompatibility_measure(tail0, tail1_raw)
    v_repaired = incompatibility_measure(tail0, tail1_repaired)

    # Step 6: changes in count-tail probabilities for k = 1,...,K_MAX_REPORT.
    k_max = config.EXP2_K_MAX_REPORT
    k_values = np.arange(1, k_max + 1)
    change_raw = tail1_raw[:k_max] - tail0[:k_max]
    change_repaired = tail1_repaired[:k_max] - tail0[:k_max]

    import pandas as pd

    table1 = pd.DataFrame(
        {
            "k": k_values,
            "P_K0_ge_k": tail0[:k_max],
            "P_K1_raw_ge_k": tail1_raw[:k_max],
            "change_raw": change_raw,
            "P_K1_repaired_ge_k": tail1_repaired[:k_max],
            "change_repaired": change_repaired,
        }
    )
    table1.to_csv(RESULTS_DIR / "exp2_table1_tail_changes.csv", index=False)

    # Step 7: first portfolio size at which the fixed pair becomes incompatible.
    first_n = first_incompatible_n(p0, rho0, p1, rho1_raw, config.EXP2_N_SEARCH_MAX)

    summary = {
        "n": n,
        "p0": p0,
        "rho0": rho0,
        "p1": p1,
        "rho1_raw": rho1_raw,
        "log10_A_n_earlier": float(a0 / np.log(10)),
        "log10_A_n_raw": float(a1_raw / np.log(10)),
        "survival_earlier": survival_earlier,
        "survival_raw": survival_raw,
        "monotonicity_violated": bool(monotonicity_violated),
        "admissible_interval": interval,
        "rho1_star": rho1_star,
        "log10_A_n_repaired": float(log_orthant_probability(n, p1, rho1_star) / np.log(10)),
        "survival_repaired": survival_repaired,
        "V_100_raw": v_raw,
        "V_100_repaired": v_repaired,
        "first_incompatible_n": first_n,
    }
    with open(RESULTS_DIR / "exp2_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.plot(k_values, change_raw, marker="o", markersize=3, label="raw rho1 = 0.40")
    ax.plot(k_values, change_repaired, marker="s", markersize=3, label=f"repaired rho1* = {rho1_star:.6f}")
    ax.set_xlabel("threshold k")
    ax.set_ylabel(r"$P(K_1 \geq k) - P(K_0 \geq k)$")
    ax.set_title("Change in count-tail probabilities: raw vs. repaired later correlation (n=100)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp2_figure1_tail_changes.png", dpi=150)
    plt.close(fig)

    print("exp_2 summary:")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
