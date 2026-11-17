# Portfolio-Size Constraints on Default Correlation: Exact Coherence Tests and Sharp Asymptotic Bounds

Replication of selected numerical experiments from the paper **"Portfolio-Size
Constraints on Default Correlation: Exact Coherence Tests and Sharp Asymptotic
Bounds"** (Section 8, "Computational methodology and numerical results").

The paper studies a single-factor Gaussian (CreditMetrics/Vasicek-type) model
of homogeneous portfolio credit risk. For a portfolio of `n` exchangeable
borrowers with cumulative default probability `p` and asset correlation
`rho`, it defines the orthant probability `A_n(p, rho) = P(K = n)` (all
default) and its complement `S_n(p, rho) = A_n(1-p, rho) = P(K = 0)` (no
default), evaluated through a mode-centered, log-domain Gaussian quadrature
(Eq. 34). Two forecasts `(p0, rho0)` and `(p1, rho1)` at consecutive horizons
are *absorption-compatible* when every count-tail probability `P(K >= k)` is
non-decreasing in time (Theorem 4.1 reduces this to a two-event criterion
comparing `A_n` and `S_n`). For a fixed `(p0, rho0)` and a later probability
`p1 > p0`, Theorem 5.2 characterizes the admissible correlation interval
`[ell_n, u_n]` by two monotone scalar root equations, and Section 7.3 gives
its asymptotic expansion `rho_n^pm = rho0 +/- C/sqrt(log n) + o(1/sqrt(log n))`.

This repository implements the shared numerical machinery (the log-domain
orthant integral, the two-event compatibility criterion, the admissible
correlation interval and the sequential repair projection of Section 5.1) in
`src/`, and reproduces five experiments from Section 8 in `exp/`. All
experiments are purely computational: every parameter is a deterministic
mathematical constant from the paper, and no market or external data is
used.

## Experiments

| Script | Paper section | What it reproduces |
|---|---|---|
| `exp.exp1_numerical_verification` | 8.2 | Closed-form identities for `A_n`, cross-validation of the two-event criterion against full count-tail integration, and a tightened-tolerance accuracy run |
| `exp.exp2_incompatibility_repair` | 8.3 | A 100-borrower incompatibility example and the one-step correlation repair, Table 1 / Figure 1, plus the first incompatible `n` |
| `exp.exp3_asymptotic_boundary` | 8.4 | Admissible intervals from `n=10` to `n=10^160`, the asymptotic constant `C`, Table 2, and the nested-interval property |
| `exp.exp4_sensitivity_analysis` | 8.5 | Table 3: admissible intervals for six `(p0, p1, rho0)` configurations at `n=50` and `n=500` |
| `exp.exp5_multi_horizon_repair` | 8.6 | Table 4: sequential repair of oscillating raw correlations across five horizons |

## Repository layout

```
src/            Shared numerical library (flat modules)
  gaussian_mixture.py   Log-domain mode-centered orthant integral a_n, s_n (Eq. 33-35)
  count_tail.py         Full count-tail probabilities P(K>=k) and the two-event criterion
  correlation_bounds.py Admissible interval [ell_n, u_n] (Theorem 5.2 / Algorithm 1) and sequential repair
  asymptotics.py        Asymptotic boundary constant C (Section 7.3)
  config.py             All experiment parameters (module-level constants)
exp/            One script per experiment, calling only src/
data/           Empty package: no market or external data is used by this replication
tests/          pytest unit tests of src/ (synthetic / closed-form inputs only)
results/        Tables, figures and JSON summaries written by the experiment scripts
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

No API keys or external data are required: every experiment is a
self-contained numerical computation on parameters taken from the paper.

## Running the tests

```bash
python -m pytest -m "not market_data"
```

(no test in this repository is marked `market_data`, since no experiment
loads market data; the command above is equivalent to plain `python -m
pytest`).

## Running the experiments

```bash
python -m exp.exp1_numerical_verification
python -m exp.exp2_incompatibility_repair
python -m exp.exp3_asymptotic_boundary
python -m exp.exp4_sensitivity_analysis
python -m exp.exp5_multi_horizon_repair
```

or, from the repository root:

```bash
bash reproduce.sh
```

Each script overwrites its output files directly in `results/` (CSV tables,
PNG figures and a JSON summary); no per-experiment subdirectories are used.
See `results/RESULTS.md` for the comparison of these outputs against the
paper's reported values.

## Numerical methods

* Standard normal CDF/quantile/density: `scipy.stats.norm`.
* Mode-centered orthant integral `a_n(p, rho) = log A_n(p, rho)` (Eq. 33-35):
  adaptive quadrature (`scipy.integrate.quad`, `abstol=1e-11`,
  `reltol=3e-11`) over `[x* - 13*sigma, x* + 13*sigma]`, with the mode `x*`
  found by bracketed root finding (`scipy.optimize.brentq`, `xtol=2e-12`) on
  the strictly concave log-integrand `h(x) = n*log(Phi(x)) - (x-mu)^2/(2*sigma^2)`.
* Admissible correlation boundaries (Theorem 5.2): bracketed root finding
  (`scipy.optimize.brentq`, `xtol=5e-13`) of the monotone scalar equations
  `a_n(p1, r) = a_n(p0, rho0)` and `s_n(p1, r) = s_n(p0, rho0)`.
  All computations remain in the log domain so that portfolio sizes up to
  `n = 10^160` (Section 8.4) do not overflow floating-point arithmetic.
* Full count-tail probabilities, used only to independently cross-check the
  two-event criterion: `scipy.integrate.quad` (`abstol=1e-12`,
  `reltol=3e-12`) of the Gaussian-mixed binomial tail, with binomial
  coefficients from `scipy.special.comb`.

## Deviations and unresolved details

See `results/RESULTS.md`, sections "Deviations from the paper" and "Not
reproduced", for every place where a detail left open by the paper required
an explicit, documented choice (for example the exact grid of identity
checks in Section 8.2, the random seed for the 30 cross-validation pairs per
portfolio size, or the portfolio size `n` used in the Section 8.6 example,
which is not stated in the paper and is calibrated from the reported
horizon-1 survival probability).
