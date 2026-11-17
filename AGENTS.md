# Repository guide

This repository reproduces five numerical experiments from Section 8 of
"Portfolio-Size Constraints on Default Correlation: Exact Coherence Tests
and Sharp Asymptotic Bounds". The paper's single-factor Gaussian default
model, its mode-centered log-domain orthant integral (Eq. 33-35), its
two-event absorption-compatibility criterion (Theorem 4.1), its admissible
correlation interval (Theorem 5.2 / Algorithm 1) and its sequential repair
projection (Section 5.1) are implemented once in `src/` and reused by every
experiment script in `exp/`. Every parameter is a deterministic mathematical
constant from the paper; no market data, external dataset or API key is
used anywhere in this repository.

## Layout

```
src/            Shared numerical library (flat modules, no sub-packages)
  gaussian_mixture.py    a_n(p, rho) = log A_n(p, rho), s_n(p, rho) = log S_n(p, rho)
  count_tail.py          Full count-tail probabilities and the two-event criterion
  correlation_bounds.py  Admissible interval [ell_n, u_n] and sequential repair
  asymptotics.py         Asymptotic boundary constant C (Section 7.3)
  config.py              All experiment parameters, as module-level constants
exp/            One script per experiment (exp1 ... exp5), each calling only src/
data/           Empty package; no experiment in this repository loads data
tests/          pytest unit tests of src/ on synthetic / closed-form inputs
results/        Tables (CSV), figures (PNG) and JSON summaries written by exp/
```

## Commands

Setup:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tests:

```bash
python -m pytest -m "not market_data"
```

(there is no `market_data` marker here; this is equivalent to
`python -m pytest` and is kept for consistency with other replications in
this project family).

Experiments (from the repository root, as modules):

```bash
python -m exp.exp1_numerical_verification
python -m exp.exp2_incompatibility_repair
python -m exp.exp3_asymptotic_boundary
python -m exp.exp4_sensitivity_analysis
python -m exp.exp5_multi_horizon_repair
```

or `bash reproduce.sh` to run the tests and every experiment in sequence.
All five experiments already run at the paper's full configuration
(including `n` up to `10**160` in `exp3`); there is no reduced mode.

## Conventions

* All experiment parameters live in `src/config.py` as module-level
  `UPPER_CASE` constants, each with a comment stating its meaning and the
  paper section it comes from (or "not specified in the paper" when it does
  not).
* `data/` holds only an empty `__init__.py`: this replication needs no
  market data or synthetic data generator, since every input is an
  analytic parameter from the paper.
* Each `exp/*.py` script writes its CSV tables, PNG figures and a JSON
  summary directly into `results/`, overwriting previous output on rerun.
  After a rerun, check the numbers and verdicts in `results/RESULTS.md`
  against the new files and update them by hand if they changed.
* Tests in `tests/` exercise `src/` on closed-form identities and synthetic
  inputs with known answers (for example `A_n(1/2, 1/2) = 1/(n+1)`); they do
  not depend on any live network call.
