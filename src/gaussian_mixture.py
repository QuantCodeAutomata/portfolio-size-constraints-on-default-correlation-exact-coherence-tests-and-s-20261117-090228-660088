"""Gaussian one-factor binomial-mixture probabilities.

Implements the orthant-probability integral and its log-domain, mode-centered
evaluation from the paper "Portfolio-Size Constraints on Default Correlation:
Exact Coherence Tests and Sharp Asymptotic Bounds", Section 8.1
("Evaluation of the admissible correlation interval"), Eqs. (14), (33)-(35).

For a homogeneous one-factor Gaussian copula portfolio of size n with
cumulative default probability p and asset correlation rho, let
    c(p) = Phi^{-1}(p),  mu = c(p) / sqrt(1 - rho),  sigma = sqrt(rho / (1 - rho)).
Then
    A_n(p, rho) = P(no borrower has defaulted) = E[ Phi(mu + sigma Z)^n ]
is the orthant probability (all n coordinates of an equicorrelated standard
normal vector below c(p)), and S_n(p, rho) = A_n(1 - p, rho) is the
survival/no-default probability P(K = 0). All computations are carried out in
log space (a_n = log A_n, s_n = log S_n) to remain numerically stable for
portfolio sizes up to n = 10**160 (exp_3).
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import integrate, optimize
from scipy.stats import norm

from src import config


def c_of_p(p: float) -> float:
    """c(p) = Phi^{-1}(p), the probit default threshold (paper, before Eq. 12)."""
    if not (0.0 < p < 1.0):
        raise ValueError(f"p must lie in (0, 1), got {p}")
    return float(norm.ppf(p))


def mu_sigma(p: float, rho: float) -> tuple[float, float]:
    """mu = c(p) / sqrt(1 - rho), sigma = sqrt(rho / (1 - rho)) (paper Eq. 14)."""
    if not (0.0 <= rho < 1.0):
        raise ValueError(f"rho must lie in [0, 1), got {rho}")
    c = c_of_p(p)
    mu = c / np.sqrt(1.0 - rho)
    sigma = np.sqrt(rho / (1.0 - rho))
    return float(mu), float(sigma)


def _h(x: np.ndarray, n: float, mu: float, sigma: float) -> np.ndarray:
    """h(x) = n*log(Phi(x)) - (x - mu)^2 / (2*sigma^2) (paper Eq. 34).

    Uses norm.logcdf (log_ndtr) rather than log(norm.cdf(x)) so that the far
    right tail (Phi(x) extremely close to 1, needed for very large n) retains
    its precision instead of rounding to log(1) = 0.
    """
    return n * norm.logcdf(x) + (-((x - mu) ** 2) / (2.0 * sigma**2))


def _mode_equation(x: float, n: float, mu: float, sigma: float) -> float:
    """n*phi(x)/Phi(x) - (x - mu)/sigma^2 = 0 at the mode x* (paper Eq. 35).

    The ratio phi(x)/Phi(x) is evaluated as exp(logpdf(x) - logcdf(x)) for
    numerical stability in the far tails.
    """
    log_ratio = norm.logpdf(x) - norm.logcdf(x)
    return n * np.exp(log_ratio) - (x - mu) / sigma**2


def _find_mode(
    n: float, mu: float, sigma: float, root_abstol: float
) -> float:
    """Bracket and solve for the mode x* of the strictly concave h (Eq. 35).

    h is strictly concave (paper Sec. 8.1), so the mode equation has a unique
    root. The search starts from mu and expands geometrically until the
    derivative changes sign, then brackets with brentq.
    """
    f = lambda x: _mode_equation(x, n, mu, sigma)
    lo, hi = mu - max(1.0, sigma), mu + max(1.0, sigma)
    f_lo, f_hi = f(lo), f(hi)
    # f is decreasing (derivative of concave h): f(lo) > 0 > f(hi) at the root.
    step = max(1.0, sigma)
    iterations = 0
    while f_lo <= 0.0 and iterations < 200:
        lo -= step
        f_lo = f(lo)
        step *= 1.5
        iterations += 1
    step = max(1.0, sigma)
    iterations = 0
    while f_hi >= 0.0 and iterations < 200:
        hi += step
        f_hi = f(hi)
        step *= 1.5
        iterations += 1
    if not (f_lo > 0.0 and f_hi < 0.0):
        raise RuntimeError(
            f"Failed to bracket the mode root: f(lo)={f_lo}, f(hi)={f_hi}"
        )
    root = optimize.brentq(f, lo, hi, xtol=root_abstol, rtol=8.9e-16)
    return float(root)


def log_orthant_probability(
    n: int,
    p: float,
    rho: float,
    quad_abstol: float = config.QUAD_ABSTOL,
    quad_reltol: float = config.QUAD_RELTOL,
    mode_root_abstol: float = config.MODE_ROOT_ABSTOL,
    half_width: float = config.MODE_INTEGRATION_HALF_WIDTH,
) -> float:
    """a_n(p, rho) = log A_n(p, rho) via the mode-centered quadrature.

    Implements Eqs. (33)-(35): finds the mode x* of the strictly concave
    h(x) = n*log(Phi(x)) - (x-mu)^2/(2*sigma^2), then evaluates
        a_n = h(x*) - log(sigma*sqrt(2*pi)) + log(integral exp{h(x)-h(x*)} dx)
    over [x* - half_width*sigma, x* + half_width*sigma].
    At rho = 0, the closed form a_n(p, 0) = n*log(p) is used (paper, end of
    Sec. 8.1).
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if rho < config.RHO_ZERO_THRESHOLD:
        # A_n(p, 0) = p^n exactly (independent Bernoulli defaults); by
        # continuity of A_n in rho this closed form is also used for the
        # numerically degenerate rho -> 0+ range where sigma = sqrt(rho/(1-rho))
        # becomes too small for the mode-centered quadrature below to resolve.
        return n * np.log(p)
    mu, sigma = mu_sigma(p, rho)
    n_f = float(n)
    x_star = _find_mode(n_f, mu, sigma, mode_root_abstol)
    h_star = float(_h(np.array([x_star]), n_f, mu, sigma)[0])

    def scaled_integrand(x: float) -> float:
        return float(np.exp(_h(np.array([x]), n_f, mu, sigma)[0] - h_star))

    lo = x_star - half_width * sigma
    hi = x_star + half_width * sigma
    # For extreme portfolio sizes (exp_3 uses n up to 10**160) the integrand
    # exp{h(x)-h(x*)} can be sharply peaked; points=[x_star] tells the adaptive
    # quadrature where the peak sits. scipy's IntegrationWarning ("maximum
    # number of subdivisions achieved") can still fire at these extreme scales
    # even though the estimated absolute error already satisfies quad_reltol
    # (checked below); it is suppressed only after that check passes.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", integrate.IntegrationWarning)
        integral, abserr = integrate.quad(
            scaled_integrand,
            lo,
            hi,
            epsabs=quad_abstol,
            epsrel=quad_reltol,
            limit=4000,
            points=[x_star],
        )
    if caught and abserr > quad_abstol + quad_reltol * abs(integral):
        raise RuntimeError(
            f"orthant quadrature failed to converge: n={n}, p={p}, rho={rho}, "
            f"abserr={abserr}, integral={integral}"
        )
    return h_star - np.log(sigma * np.sqrt(2.0 * np.pi)) + np.log(integral)


def log_survival_probability(
    n: int,
    p: float,
    rho: float,
    quad_abstol: float = config.QUAD_ABSTOL,
    quad_reltol: float = config.QUAD_RELTOL,
    mode_root_abstol: float = config.MODE_ROOT_ABSTOL,
    half_width: float = config.MODE_INTEGRATION_HALF_WIDTH,
) -> float:
    """s_n(p, rho) = a_n(1 - p, rho) (paper Sec. 8.1)."""
    if rho == 0.0:
        return n * np.log1p(-p)
    return log_orthant_probability(
        n, 1.0 - p, rho, quad_abstol, quad_reltol, mode_root_abstol, half_width
    )


def orthant_probability(n: int, p: float, rho: float, **kwargs) -> float:
    """A_n(p, rho) = exp(a_n(p, rho)) = P(K = n), the all-default probability."""
    return float(np.exp(log_orthant_probability(n, p, rho, **kwargs)))


def survival_probability(n: int, p: float, rho: float, **kwargs) -> float:
    """S_n(p, rho) = exp(s_n(p, rho)) = P(K = 0), the no-default probability."""
    return float(np.exp(log_survival_probability(n, p, rho, **kwargs)))
