"""Portfolio-size dependence of admissible correlation intervals.

Reproduces Section 8.4 ("Portfolio size and slow convergence") of the paper
"Portfolio-Size Constraints on Default Correlation: Exact Coherence Tests and
Sharp Asymptotic Bounds": exact admissible correlation intervals
[ell_n, u_n] for portfolio sizes n = 10 up to n = 10**160, the asymptotic
boundary constant C (Eq. 27), and the scaled distances
sqrt(log n)*(rho0 - ell_n) and sqrt(log n)*(u_n - rho0) converging slowly
(logarithmically) to C.

Run as: python -m exp.exp3_asymptotic_boundary
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.asymptotics import boundary_constant
from src.correlation_bounds import admissible_interval

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def _log_n(n: int) -> float:
    """log(n), computed without overflow for n up to 10**160 (n held as a Python int)."""
    if n < 10**300:
        return math.log(n)
    # log(10**k) = k*log(10); n is an exact power of ten in config.EXP3_N_VALUES.
    exponent = len(str(n)) - 1
    return exponent * math.log(10.0)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    p0, p1, rho0 = config.EXP3_P0, config.EXP3_P1, config.EXP3_RHO0

    c_constant = boundary_constant(p0, p1, rho0)

    rows = []
    for n in config.EXP3_N_VALUES:
        interval = admissible_interval(n, p0, rho0, p1, rho_bar=config.RHO_BAR)
        log_n = _log_n(n)
        sqrt_log_n = math.sqrt(log_n)
        scaled_lower = sqrt_log_n * (rho0 - interval["ell_n"])
        scaled_upper = sqrt_log_n * (interval["u_n"] - rho0)
        rows.append(
            {
                "n": str(n),
                "log10_n": math.log10(n),
                "ell_n": interval["ell_n"],
                "u_n": interval["u_n"],
                "width": interval["u_n"] - interval["ell_n"],
                "scaled_distance_lower": scaled_lower,
                "scaled_distance_upper": scaled_upper,
            }
        )
    table2 = pd.DataFrame(rows)
    table2.to_csv(RESULTS_DIR / "exp3_table2_portfolio_size.csv", index=False)

    # Nested-interval check (Proposition 5.4): I_{n+1} subset I_n for the
    # smallest consecutive gap available in the mathematical diagnostic grid.
    n_consec = [10, 11, 12, 20, 50]
    nested_rows = []
    for n in n_consec:
        i_n = admissible_interval(n, p0, rho0, p1, rho_bar=config.RHO_BAR)
        i_np1 = admissible_interval(n + 1, p0, rho0, p1, rho_bar=config.RHO_BAR)
        nested = (i_np1["ell_n"] >= i_n["ell_n"] - 1e-9) and (i_np1["u_n"] <= i_n["u_n"] + 1e-9)
        nested_rows.append({"n": n, "nested_I_n+1_subset_I_n": bool(nested)})
    nested_df = pd.DataFrame(nested_rows)
    nested_df.to_csv(RESULTS_DIR / "exp3_nested_interval_check.csv", index=False)

    summary = {
        "p0": p0,
        "p1": p1,
        "rho0": rho0,
        "rho_bar": config.RHO_BAR,
        "boundary_constant_C": c_constant,
        "nested_interval_holds_for_all_checked": bool(nested_df["nested_I_n+1_subset_I_n"].all()),
    }
    with open(RESULTS_DIR / "exp3_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Figure: shrinking interval and scaled endpoints across log10(n).
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (lo_exp, hi_exp) in zip(axes, [(1, 20), (0, 160)]):
        mask = (table2["log10_n"] >= lo_exp - 1e-9) & (table2["log10_n"] <= hi_exp + 1e-9)
        sub = table2[mask]
        ax.plot(sub["log10_n"], sub["ell_n"], "o-", label=r"$\ell_n$")
        ax.plot(sub["log10_n"], sub["u_n"], "s-", label=r"$u_n$")
        ax.axhline(rho0, color="gray", linestyle="--", linewidth=0.8, label=r"$\rho_0$")
        ax.set_xlabel(r"$\log_{10} n$")
        ax.set_ylabel("admissible correlation")
        ax.set_title(f"log10(n) in [{lo_exp}, {hi_exp}]")
        ax.legend()
    fig.suptitle("Admissible correlation interval vs. portfolio size (Section 8.4)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp3_figure_interval_vs_n.png", dpi=150)
    plt.close(fig)

    print("exp_3 summary:")
    print(json.dumps(summary, indent=2, default=str))
    print(table2.to_string(index=False))


if __name__ == "__main__":
    main()
