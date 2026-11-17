"""Module-level configuration for every experiment.

All parameters are deterministic mathematical constants taken from the paper
"Portfolio-Size Constraints on Default Correlation: Exact Coherence Tests and
Sharp Asymptotic Bounds", Section 8 (Computational methodology and numerical
results), unless a comment states otherwise.
"""

# ---------------------------------------------------------------------------
# Numerical tolerances (Section 8.1, "Evaluation of the admissible correlation
# interval")
# ---------------------------------------------------------------------------

# Absolute and relative tolerances for the mode-centered orthant integral
# quadrature (Eq. 34), paper Sec. 8.1.
RHO_ZERO_THRESHOLD = 1e-9  # below this, use the rho=0 closed form a_n(p,0)=n*log(p) (paper Sec. 8.1); avoids the degenerate sigma->0 quadrature, not specified numerically in the paper
QUAD_ABSTOL = 1e-11
QUAD_RELTOL = 3e-11

# Absolute tolerance (xtol) for scalar root finding of the admissible
# correlation boundaries ell_n, u_n (Theorem 5.2 / Algorithm 1), paper Sec. 8.1.
ROOT_ABSTOL = 5e-13

# Absolute tolerance (xtol) for the mode root x* solving Eq. 35, paper Sec. 8.1.
MODE_ROOT_ABSTOL = 2e-12

# Half-width, in units of sigma, of the mode-centered integration interval
# [x* - 13*sigma, x* + 13*sigma] used in Eq. 34, paper Sec. 8.1.
MODE_INTEGRATION_HALF_WIDTH = 13.0

# Tolerances for the independent full count-tail probability integration used
# to cross-validate the two-event criterion (paper Sec. 8.1, last paragraph).
FULLTAIL_QUAD_ABSTOL = 1e-12
FULLTAIL_QUAD_RELTOL = 3e-12

# Correlation cap used throughout Section 8 (explicitly stated for Sec. 8.5;
# the paper does not restate it for Sec. 8.3, so the same cap is used there).
RHO_BAR = 0.99

# ---------------------------------------------------------------------------
# Secondary accuracy run (exp_1, Step 6): tightened tolerances and expanded
# mode-centered interval. The paper states only that tolerances are
# "tightened" (Sec. 8.2) without giving exact values; two orders of magnitude
# tighter than the baseline tolerances above are used here, not specified in
# the paper beyond the qualitative description.
QUAD_ABSTOL_TIGHT = 1e-13
QUAD_RELTOL_TIGHT = 3e-13
ROOT_ABSTOL_TIGHT = 5e-15
MODE_ROOT_ABSTOL_TIGHT = 2e-14
MODE_INTEGRATION_HALF_WIDTH_TIGHT = 16.0

# ---------------------------------------------------------------------------
# exp_1: Numerical verification of the log-domain quadrature and root finding
# ---------------------------------------------------------------------------

# Portfolio sizes n for the closed-form identity A_n(1/2, 1/2) = 1/(n+1)
# (paper Sec. 8.2). The paper requires "at least 9 evaluations"; the exact
# set of n values is not specified, so a representative grid of 10 sizes
# spanning small to moderately large n is used.
EXP1_N_VALUES_UNIFORM_IDENTITY = [1, 2, 3, 5, 8, 13, 21, 50, 100, 250]

# Correlation grid for the identity A_2(1/2, rho) = 1/4 + arcsin(rho)/(2*pi)
# (paper Sec. 8.2). The exact grid is not specified in the paper; values are
# restricted to rho in [0, 1), the domain of the one-factor model of Eq. (4)
# that the implementation's mode-centered integral (Eq. 33) represents.
EXP1_RHO_VALUES_ARCSIN_IDENTITY = [
    0.001, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.97,
]

# Portfolio sizes used for cross-validating the two-event criterion against
# full count-tail integration (paper Sec. 8.2, Step 5).
EXP1_CROSS_CHECK_N_VALUES = [8, 25, 60]

# Number of random (p, rho) pairs per portfolio size for the cross-check
# (paper requires "30 parameter pairs"); seed fixes reproducibility, which the
# paper does not specify.
EXP1_CROSS_CHECK_PAIRS_PER_N = 30
EXP1_CROSS_CHECK_SEED = 20240115

# ---------------------------------------------------------------------------
# exp_2: Constructed incompatibility example and one-step repair (Sec. 8.3)
# ---------------------------------------------------------------------------

EXP2_N = 100           # portfolio size (borrowers), paper Sec. 8.3
EXP2_P0 = 0.02         # earlier cumulative default probability, paper Sec. 8.3
EXP2_RHO0 = 0.10       # earlier correlation, paper Sec. 8.3
EXP2_P1 = 0.025        # later cumulative default probability, paper Sec. 8.3
EXP2_RHO1_RAW = 0.40   # raw (unrepaired) later correlation, paper Sec. 8.3
EXP2_K_MAX_REPORT = 40     # threshold range reported in Table 1 / Figure 1, paper Sec. 8.3
EXP2_N_SEARCH_MAX = 50     # upper bound for the search of the first incompatible n, paper Sec 8.3 (expected n=7)

# ---------------------------------------------------------------------------
# exp_3: Portfolio-size dependence and asymptotic boundary expansion (Sec. 8.4)
# ---------------------------------------------------------------------------

EXP3_P0 = 0.02     # earlier cumulative default probability, paper Sec. 8.4
EXP3_P1 = 0.03     # later cumulative default probability, paper Sec. 8.4
EXP3_RHO0 = 0.15   # earlier correlation, paper Sec. 8.4
EXP3_N_VALUES = [10, 100, 1000, 10**6, 10**12, 10**40, 10**160]  # paper Table 2

# ---------------------------------------------------------------------------
# exp_4: Sensitivity analysis (Sec. 8.5, Table 3)
# ---------------------------------------------------------------------------

# Six (p0, p1, rho0) configurations from Table 3, paper Sec. 8.5.
EXP4_CONFIGS = [
    (0.010, 0.015, 0.10),
    (0.020, 0.025, 0.10),
    (0.020, 0.030, 0.15),
    (0.050, 0.065, 0.15),
    (0.050, 0.065, 0.30),
    (0.100, 0.125, 0.30),
]
EXP4_N_VALUES = [50, 500]  # portfolio sizes, paper Sec. 8.5

# ---------------------------------------------------------------------------
# exp_5: Multi-horizon sequential repair (Sec. 8.6, Table 4)
# ---------------------------------------------------------------------------

EXP5_P_GRID = [0.010, 0.018, 0.026, 0.033, 0.040]        # cumulative default probabilities per horizon, paper Table 4
EXP5_RHO_RAW_GRID = [0.120, 0.300, 0.080, 0.250, 0.150]   # raw correlations per horizon, paper Table 4

# The portfolio size for this example is not stated in the paper (Sec. 8.6).
# It is identified from the first horizon's reported survival probability:
# S_n(p_0, rho_0) = S_n(0.010, 0.120) = 0.498412 (Table 4) holds to within the
# quadrature tolerance above exactly at n = 100, the same portfolio size used
# in the Section 8.3 example (Sec. 8.3). No other n in [2, 20000] matches.
EXP5_N = 100
EXP5_N_CALIBRATION_TARGET_SURVIVAL = 0.498412  # Table 4, horizon 1, paper Sec. 8.6
