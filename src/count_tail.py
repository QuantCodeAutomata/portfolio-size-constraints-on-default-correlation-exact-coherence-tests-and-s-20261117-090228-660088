"""Full count-tail probabilities and the two-event criterion cross-check.

Implements the independent full count-tail integration used in the paper
"Portfolio-Size Constraints on Default Correlation: Exact Coherence Tests and
Sharp Asymptotic Bounds" to verify the two-event criterion of Theorem 4.1 and
to compute the incompatibility diagnostic V_n of Eq. (3), Section 2.1.

For the Gaussian one-factor mixture, conditional on the latent factor Z the
default probability is Q(p, rho, Z) = Phi((c(p) + sqrt(rho) Z) / sqrt(1 - rho))
and K | Z ~ Binomial(n, Q). Hence
    P(K >= k) = E_Z[ sum_{j=k}^n C(n,j) Q^j (1-Q)^(n-j) ],
integrated over Z ~ N(0,1) by adaptive quadrature (paper Sec. 8.1, last
paragraph: "Full-tail integration is reserved for verification...").
"""

from __future__ import annotations

import numpy as np
from scipy import integrate
from scipy.special import comb
from scipy.stats import norm

from src import config
from src.gaussian_mixture import c_of_p


def _binomial_tail_matrix(n: int) -> np.ndarray:
    """Binomial coefficients C(n, j) for j = 0, ..., n, used for every k."""
    return comb(n, np.arange(n + 1), exact=False)


def count_tail_probabilities(
    n: int,
    p: float,
    rho: float,
    quad_abstol: float = config.FULLTAIL_QUAD_ABSTOL,
    quad_reltol: float = config.FULLTAIL_QUAD_RELTOL,
    z_half_width: float = 12.0,
) -> np.ndarray:
    """P(K >= k) for k = 1, ..., n via full latent-factor integration.

    Returns an array of length n indexed by k - 1 (index 0 is P(K >= 1), ...,
    index n-1 is P(K >= n) = A_n(p, rho) in the paper's all-default notation).
    """
    if not (0.0 < p < 1.0):
        raise ValueError(f"p must lie in (0, 1), got {p}")
    if not (0.0 <= rho < 1.0):
        raise ValueError(f"rho must lie in [0, 1), got {rho}")
    c = c_of_p(p)
    coeffs = _binomial_tail_matrix(n)  # C(n, j), j=0..n
    j_index = np.arange(n + 1)

    def integrand(z: float) -> np.ndarray:
        if rho > 0.0:
            q = norm.cdf((c + np.sqrt(rho) * z) / np.sqrt(1.0 - rho))
        else:
            q = p
        q = min(max(q, 1e-300), 1.0 - 1e-16)
        log_pmf = (
            np.log(coeffs)
            + j_index * np.log(q)
            + (n - j_index) * np.log1p(-q)
        )
        pmf = np.exp(log_pmf)
        return pmf * norm.pdf(z)

    tail = np.zeros(n)
    grid = integrate.quad_vec(
        integrand, -z_half_width, z_half_width, epsabs=quad_abstol, epsrel=quad_reltol
    )[0]
    # grid[j] = P(K = j); tail probabilities P(K >= k) = sum_{j>=k} grid[j]
    cum_from_top = np.cumsum(grid[::-1])[::-1]
    tail = cum_from_top[1:]  # drop j=0 term to align with k=1,...,n
    return tail


def incompatibility_measure(tail0: np.ndarray, tail1: np.ndarray) -> float:
    """V_n = max_{1<=k<=n} [P(K0>=k) - P(K1>=k)]_+ (paper Eq. 3)."""
    if tail0.shape != tail1.shape:
        raise ValueError("tail0 and tail1 must have the same length")
    diff = tail0 - tail1
    return float(np.max(np.clip(diff, 0.0, None)))


def two_event_criterion_holds(
    n: int, p0: float, rho0: float, p1: float, rho1: float, **tol_overrides
) -> bool:
    """Theorem 4.1 via Corollary 4.2: K0 <=_st K1 iff both extreme events hold.

    Evaluated directly from the orthant/survival probabilities rather than
    the admissible-interval machinery, for use as an independent criterion
    check (paper Sec. 8.1/8.2).
    """
    from src.gaussian_mixture import log_orthant_probability, log_survival_probability

    a0 = log_orthant_probability(n, p0, rho0, **tol_overrides)
    a1 = log_orthant_probability(n, p1, rho1, **tol_overrides)
    s0 = log_survival_probability(n, p0, rho0, **tol_overrides)
    s1 = log_survival_probability(n, p1, rho1, **tol_overrides)
    return (a1 >= a0 - 1e-9) and (s1 <= s0 + 1e-9)


def first_incompatible_n(
    p0: float, rho0: float, p1: float, rho1: float, n_max: int, **tol_overrides
) -> int | None:
    """Smallest integer portfolio size n in [2, n_max] at which the fixed pair
    (p0, rho0) -> (p1, rho1) violates the two-event criterion (Theorem 4.1).

    Proposition 5.4 shows admissible intervals are nested in n, so feasibility
    is non-increasing in n; the first failure is found by a linear scan from
    n = 2 (matching the small search range used in the paper's example).
    """
    for n in range(2, n_max + 1):
        if not two_event_criterion_holds(n, p0, rho0, p1, rho1, **tol_overrides):
            return n
    return None
