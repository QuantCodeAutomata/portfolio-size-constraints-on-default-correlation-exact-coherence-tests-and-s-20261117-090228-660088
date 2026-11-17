#!/usr/bin/env bash
# Reproduces every experiment of this repository end to end.
#
# "Portfolio-Size Constraints on Default Correlation: Exact Coherence Tests
# and Sharp Asymptotic Bounds", Section 8.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

echo "=== Running tests ==="
python -m pytest -m "not market_data" || exit 1

echo "=== exp1: numerical verification (Section 8.2) ==="
python -m exp.exp1_numerical_verification

echo "=== exp2: 100-borrower incompatibility and repair (Section 8.3) ==="
python -m exp.exp2_incompatibility_repair

echo "=== exp3: asymptotic boundary expansion (Section 8.4) ==="
python -m exp.exp3_asymptotic_boundary

echo "=== exp4: sensitivity analysis (Section 8.5) ==="
python -m exp.exp4_sensitivity_analysis

echo "=== exp5: multi-horizon sequential repair (Section 8.6) ==="
python -m exp.exp5_multi_horizon_repair

echo "=== Done. See results/RESULTS.md for the comparison against the paper. ==="
