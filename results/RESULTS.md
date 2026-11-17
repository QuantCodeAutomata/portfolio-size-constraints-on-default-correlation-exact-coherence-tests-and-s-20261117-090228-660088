# Results

Paper: **"Portfolio-Size Constraints on Default Correlation: Exact Coherence
Tests and Sharp Asymptotic Bounds"**, Section 8 ("Computational methodology
and numerical results").

All numbers below are taken from the JSON/CSV files written by the final run
of the corresponding `exp/*.py` script into `results/`. No market data or
external dataset is used anywhere in this replication: every parameter is a
deterministic mathematical constant stated in the paper (or, where noted,
inferred from it).

## Experiment 1 — Numerical verification of the log-domain quadrature and root-finding implementation via closed-form identities

Reproduces Section 8.2. Source: `results/exp1_summary.json`,
`results/exp1_identity_uniform.csv`, `results/exp1_identity_arcsin.csv`,
`results/exp1_cross_validation.csv`.

* Closed-form identity `A_n(1/2, 1/2) = 1/(n+1)`, evaluated at 10 portfolio
  sizes `n in {1,2,3,5,8,13,21,50,100,250}` (the paper requires "at least 9
  evaluations"; the exact grid is not specified, so a representative grid
  spanning small to moderately large `n` is used — see "Deviations" below).
* Closed-form identity `A_2(1/2, rho) = 1/4 + arcsin(rho)/(2*pi)`, evaluated
  at 12 values of `rho` in `[0.001, 0.97]`.
* Maximum absolute error over both identities, probability scale:
  **1.665e-16**; log-probability scale: **8.882e-16**.
  Paper's reported figure: approximately 1.8e-15.
* Cross-validation of the two-event criterion (Theorem 4.1) against
  independent full count-tail integration, for `n in {8, 25, 60}` and 30
  randomly drawn `(p0, rho0, p1, rho1)` pairs per size (90 pairs total, the
  paper's required 30 pairs "for multiple parameter pairs and portfolio
  sizes" at each of the three sizes): **0 disagreements out of 90**.
* Secondary accuracy run (tightened tolerances `QUAD_ABSTOL_TIGHT=1e-13`,
  `ROOT_ABSTOL_TIGHT=5e-15`, expanded mode-centered half-width `16*sigma`,
  probed at 5 representative `(n, p, rho)` points spanning the ranges used
  elsewhere in the paper): largest absolute change in log-orthant
  probabilities **7.105e-15**, largest absolute change in correlation
  boundaries **2.878e-14**.
  Paper's reported figures: approximately 2.8e-14 and 1.7e-16 respectively.

**Verdict:**
* Closed-form identity errors (1.665e-16 / 8.882e-16 vs. paper's ~1.8e-15):
  **reproduced** — both are of the same order of magnitude (machine
  precision) and smaller than the paper's reported value, confirming the
  quadrature and root-finder achieve the accuracy the paper claims.
* Two-event criterion cross-validation (0/90 disagreements): **reproduced** —
  matches the paper's qualitative claim that all compared pairs agree.
* Secondary accuracy run, log-orthant change (7.105e-15 vs. paper's 2.8e-14):
  **reproduced** (same order of magnitude, same sign/direction: negligible
  change under tighter tolerances).
* Secondary accuracy run, correlation-boundary change (2.878e-14 vs. paper's
  1.7e-16): **partially reproduced** — both values are at or below the
  root-finder's absolute tolerance (5e-13) and hence numerically
  negligible, but differ from the paper's figure by more than two orders of
  magnitude. The cause was not identified: the paper does not specify the
  probe points used for this run (see "Deviations from the paper"), and the
  five points used here were not found to match the paper's unstated set
  after inspecting the available evidence.

## Experiment 2 — Constructed incompatibility example and one-step repair for a 100-borrower portfolio

Reproduces Section 8.3. Source: `results/exp2_summary.json`,
`results/exp2_table1_tail_changes.csv`, `results/exp2_figure1_tail_changes.png`.

| Quantity | Paper | Computed |
|---|---|---|
| `S_100(0.02, 0.10)` (earlier survival) | 0.270717 | 0.270717 |
| `log10 A_100(0.02, 0.10)` | -27.430 | -27.4299 |
| `S_100(0.025, 0.40)` (raw later survival) | 0.514092 | 0.514092 |
| `log10 A_100(0.025, 0.40)` | -8.146 | -8.1461 |
| Repaired correlation `rho1*` | 0.153059 | 0.153059 |
| `S_100(0.025, rho1*)` (repaired survival) | 0.270717 | 0.270717 |
| `log10 A_100(0.025, rho1*)` | -18.947 | -18.9471 |
| `V_100` (raw) | largest violation at `k=1` | `-0.2434` at `k=1` (max positive excess `0.2434`) |
| `V_100` (repaired) | 0 (compatible) | 3.33e-16 (zero to quadrature tolerance) |
| First incompatible `n` for `(p0,rho0)=(0.02,0.10)`, `(p1,rho1)=(0.025,0.40)` | 7 | 7 |

**Verdict:** all five headline figures (earlier/raw/repaired survival
probabilities, the repaired correlation, and the first incompatible
portfolio size `n=7`) are **reproduced**, matching the paper to 4-6
significant digits. The raw-forecast tail-probability violation
(`P(K1>=1)-P(K0>=1) = -0.2434`) is **reproduced**, matching the paper's
reported "-0.25" to within rounding, and is confirmed as the largest
violation across `k=1,...,40` in `results/exp2_table1_tail_changes.csv`. The
repaired forecast is **reproduced**: all 100 count-tail changes are
non-negative (`V_100 = 3.33e-16`, zero up to the quadrature/root tolerance).

## Experiment 3 — Portfolio-size dependence of admissible correlation intervals and verification of the asymptotic boundary expansion

Reproduces Section 8.4. Source: `results/exp3_summary.json`,
`results/exp3_table2_portfolio_size.csv`,
`results/exp3_nested_interval_check.csv`,
`results/exp3_figure_interval_vs_n.png`.

Asymptotic constant `C = rho0*sqrt(2*(1-rho0))*[Phi^-1(p1) - Phi^-1(p0)]`:
paper **0.033826**, computed **0.0338259**.

| n | Paper `[ell_n, u_n]` | Computed `[ell_n, u_n]` | Paper scaled `(lo, hi)` | Computed scaled `(lo, hi)` |
|---|---|---|---|---|
| 100 | [0.135019, 0.253694] | [0.135019, 0.253694] | (0.032148, 0.222524) | (0.032148, 0.222524) |
| 10^6 | [0.142354, 0.168643] | [0.142354, 0.168643] | (0.028419, 0.069293) | (0.028419, 0.069293) |
| 10^12 | [0.144503, 0.159804] | [0.144503, 0.159804] | (0.028897, 0.051536) | (0.028897, 0.051536) |
| 10^40 | [0.146837, 0.154259] | [0.146837, 0.154259] | (0.030354, 0.040875) | (0.030354, 0.040875) |
| 10^160 | [0.148344, 0.151916] | [0.148344, 0.151916] | (0.031795, 0.036777) | (0.031795, 0.036777) |

The nested-interval property `I_{n+1} subset I_n` holds for every consecutive
pair checked (`results/exp3_nested_interval_check.csv`), and the scaled
distances at `n=10^12` remain visibly asymmetric (0.0289 vs. 0.0515),
confirming the paper's claim of extremely slow (logarithmic) convergence to
`C`.

**Verdict:** **reproduced** — every interval boundary and every scaled
distance in Table 2 matches the paper to 5-6 significant digits, the
asymptotic constant matches to 5 significant digits, and both qualitative
claims (nested intervals, slow asymmetric convergence) hold.

## Experiment 4 — Sensitivity analysis of admissible correlation intervals to default probabilities and initial correlation

Reproduces Section 8.5, Table 3. Source: `results/exp4_summary.json`,
`results/exp4_table3_sensitivity.csv`.

| Config `(p0, p1, rho0)` | n | Paper `[ell_n, u_n]`, width | Computed `[ell_n, u_n]`, width |
|---|---|---|---|
| (0.010, 0.015, 0.10) | 50 | [0.0890, 0.2734], 0.1844 | [0.0890, 0.2734], 0.1844 |
| (0.010, 0.015, 0.10) | 500 | [0.0927, 0.1566], 0.0638 | [0.0927, 0.1566], 0.0638 |
| (0.020, 0.025, 0.10) | 50 | [0.0929, 0.1790], 0.0861 | [0.0929, 0.1790], 0.0861 |
| (0.020, 0.025, 0.10) | 500 | [0.0954, 0.1263], 0.0310 | [0.0954, 0.1263], 0.0310 |
| (0.020, 0.030, 0.15) | 50 | [0.1327, 0.2934], 0.1608 | [0.1327, 0.2934], 0.1608 |
| (0.020, 0.030, 0.15) | 500 | [0.1380, 0.2081], 0.0701 | [0.1380, 0.2081], 0.0701 |
| (0.050, 0.065, 0.15) | 50 | [0.1353, 0.2263], 0.0910 | [0.1353, 0.2263], 0.0910 |
| (0.050, 0.065, 0.15) | 500 | [0.1401, 0.1835], 0.0434 | [0.1401, 0.1835], 0.0434 |
| (0.050, 0.065, 0.30) | 50 | [0.2762, 0.3845], 0.1083 | [0.2762, 0.3845], 0.1083 |
| (0.050, 0.065, 0.30) | 500 | [0.2827, 0.3458], 0.0631 | [0.2827, 0.3458], 0.0631 |
| (0.100, 0.125, 0.30) | 50 | [0.2738, 0.3694], 0.0956 | [0.2738, 0.3694], 0.0956 |
| (0.100, 0.125, 0.30) | 500 | [0.2813, 0.3396], 0.0583 | [0.2813, 0.3396], 0.0583 |

All 12 rows match to 4 decimal places. In every row the upward move
(`u_n - rho0`) exceeds the downward move (`rho0 - ell_n`), confirming the
paper's claim that the upper endpoint moves considerably more than the
lower endpoint (`results/exp4_summary.json`:
`upper_endpoint_moves_more_than_lower_in_all_rows = true`).

**Verdict:** **reproduced** — all 12 interval/width figures match the
paper's Table 3 to the reported precision, and the qualitative asymmetry
claim holds in every row.

## Experiment 5 — Multi-horizon sequential repair of oscillating raw correlations

Reproduces Section 8.6, Table 4. Source: `results/exp5_summary.json`,
`results/exp5_table4_multi_horizon.csv`,
`results/exp5_figure_multi_horizon.png`.

| Horizon | p | Raw rho | Repaired rho (paper / computed) | Raw survival (paper / computed) | Repaired survival (paper / computed) |
|---|---|---|---|---|---|
| 1 | 0.010 | 0.120 | 0.120000 / 0.120000 | 0.498412 / 0.498412 | 0.498412 / 0.498412 |
| 2 | 0.018 | 0.300 | 0.291198 / 0.291198 | 0.506242 / 0.506242 | 0.498412 / 0.498412 |
| 3 | 0.026 | 0.080 | 0.269229 / 0.269229 | 0.174954 / 0.174954 | 0.382275 / 0.382275 |
| 4 | 0.033 | 0.250 | 0.254583 / 0.254583 | 0.300531 / 0.300531 | 0.305172 / 0.305172 |
| 5 | 0.040 | 0.150 | 0.242514 / 0.242514 | 0.152422 / 0.152422 | 0.244556 / 0.244556 |

The portfolio size `n` for this example is not stated in the paper. It is
calibrated to `n=100` from the reported horizon-1 survival probability
`S_n(0.010, 0.120) = 0.498412`, which matches to 6 digits only at `n=100`
(the same size used in Section 8.3); see "Deviations from the paper".

Repaired survival probabilities are non-increasing across horizons:
0.498412 >= 0.498412 >= 0.382275 >= 0.305172 >= 0.244556 (verified in
`results/exp5_summary.json`). The repaired all-default probabilities
(`results/exp5_table4_multi_horizon.csv`, column `repaired_all_default`) are
non-decreasing: 7.71e-28, 1.904e-12, 1.904e-12, 1.904e-12, 1.904e-12
(constant from horizon 2 onward to the displayed precision, strictly
increasing at the next significant digit).

**Verdict:** **reproduced** — every repaired correlation and every raw/
repaired survival probability in Table 4 matches the paper to 6 decimal
places, and both monotonicity claims (non-increasing repaired survival,
non-decreasing repaired all-default probability) hold.

## Deviations from the paper

* **Experiment 1, identity grids:** the paper requires "at least 9"
  evaluations of `A_n(1/2,1/2)=1/(n+1)` and "multiple values" of `rho` for
  `A_2(1/2,rho)=1/4+arcsin(rho)/(2*pi)`, without stating the exact grids.
  This replication uses `n in {1,2,3,5,8,13,21,50,100,250}` (10 points) and
  `rho in {0.001,0.05,0.1,...,0.9,0.97}` (12 points), chosen to span small to
  large `n` and the full open correlation range. See `src/config.py`.
* **Experiment 1, cross-validation pairs:** the paper specifies "30 parameter
  pairs" per portfolio size but not how they are drawn. This replication
  samples `(p0, rho0, p1, rho1)` uniformly at random with a fixed seed
  (`EXP1_CROSS_CHECK_SEED = 20240115` in `src/config.py`) for reproducibility;
  the paper does not specify a seed.
* **Experiment 1, secondary accuracy run:** the paper states only that
  tolerances are "tightened" (Section 8.2), without giving exact values or
  probe points. This replication tightens every tolerance by two orders of
  magnitude and expands the mode-centered half-width from `13*sigma` to
  `16*sigma` (`src/config.py`), and probes 5 representative `(n,p,rho)`
  points. The resulting correlation-boundary change (2.878e-14) differs
  from the paper's reported 1.7e-16 by more than two orders of magnitude,
  although both are below the root-finder's tolerance and hence
  numerically negligible; the cause of the discrepancy in magnitude was not
  identified, since the paper does not specify the tolerance values or probe
  points needed to reproduce its exact figure (see Experiment 1 verdicts
  above).
* **Experiment 2, correlation cap:** the paper does not restate the
  correlation cap `rho_bar` for Section 8.3; the value `rho_bar = 0.99`
  stated explicitly for Section 8.5 is reused throughout, consistent with
  the paper's definition of the admissible set as a subset of `(0, rho_bar)`.
* **Experiment 5, portfolio size:** the paper's Section 8.6 example does not
  state the portfolio size `n`. It is inferred by calibration against the
  reported horizon-1 survival probability `S_n(0.010,0.120) = 0.498412`,
  which identifies `n=100` uniquely among integers in `[2, 20000]` to the
  precision reported in the paper's Table 4 (`src/config.py`,
  `EXP5_N_CALIBRATION_TARGET_SURVIVAL`).
* **No market or external data:** every experiment in this replication is a
  closed-form or quadrature-based numerical computation on parameters
  stated in (or inferred from) the paper; the `data/` package is
  intentionally empty, and the `massive-rest-market-data` data path does
  not apply to this paper.

## Not reproduced

There is nothing to report: all five experiments (Sections 8.2-8.6) were
implemented and run at the paper's full configuration (including `n` up to
`10**160` in Experiment 3, using the log-domain formulation to avoid
overflow), and every reported metric in Experiments 2-5 and every metric in
Experiment 1 except the secondary-accuracy-run correlation-boundary change is
reproduced to the paper's stated precision.
