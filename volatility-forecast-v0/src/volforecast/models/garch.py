"""GARCH-family volatility models with maximum-likelihood estimation (Stage 1).

Volatility is *latent* — we never observe sigma_t, only returns. The defining
empirical facts the models must capture:
  - returns are ~serially uncorrelated, but
  - squared returns are strongly autocorrelated (volatility CLUSTERS).

GARCH(1,1):
    sigma^2_t = omega + alpha * eps^2_{t-1} + beta * sigma^2_{t-1}
  alpha = reaction to news, beta = persistence, alpha+beta<1 for stationarity,
  long-run variance  sigma_bar^2 = omega / (1 - alpha - beta).

EGARCH(1,1) (Nelson):
    ln sigma^2_t = omega + beta ln sigma^2_{t-1}
                 + alpha (|z_{t-1}| - E|z|) + gamma z_{t-1}
  models log-variance (positivity is automatic) and gamma<0 encodes the LEVERAGE
  effect — negative shocks raise vol more than positive ones, the same asymmetry
  that produces equity skew in Project 1.

We implement the likelihood and recursion by hand (scipy.optimize for the MLE)
rather than calling a library, matching the course's derive-before-coding stance.
Variance targeting is used to stabilise estimation: omega is pinned to the sample
variance so the optimiser only searches (alpha, beta).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

Innovation = Literal["gaussian", "student-t"]


# --------------------------------------------------------------------------- #
#  GARCH(1,1)
# --------------------------------------------------------------------------- #

def _garch_recursion(
    eps: NDArray, omega: float, alpha: float, beta: float, sigma2_0: float
) -> NDArray:
    """Filter the conditional variance path sigma^2_t given parameters and the
    return innovations eps_t = r_t - mu. Pure recursion, no estimation.
    """
    n = len(eps)
    sigma2 = np.empty(n)
    sigma2[0] = sigma2_0
    for t in range(1, n):
        sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
    return sigma2


def _garch_negloglik(
    params: NDArray, eps: NDArray, uncond_var: float, innovation: Innovation
) -> float:
    """Negative Gaussian / Student-t log-likelihood with variance targeting.

    params = (alpha, beta[, nu]). omega is implied by variance targeting:
        omega = (1 - alpha - beta) * uncond_var
    so the long-run variance equals the sample variance by construction — this
    removes a near-flat direction from the optimisation and is standard practice.
    """
    alpha, beta = params[0], params[1]
    if alpha < 0 or beta < 0 or alpha + beta >= 0.9999:
        return 1e10  # enforce non-negativity + stationarity
    omega = (1.0 - alpha - beta) * uncond_var
    if omega <= 0:
        return 1e10

    sigma2 = _garch_recursion(eps, omega, alpha, beta, uncond_var)
    if np.any(sigma2 <= 0) or not np.all(np.isfinite(sigma2)):
        return 1e10

    if innovation == "gaussian":
        ll = -0.5 * np.sum(np.log(2 * np.pi) + np.log(sigma2) + eps**2 / sigma2)
    else:  # student-t
        nu = params[2]
        if nu <= 2.0:
            return 1e10
        from scipy.special import gammaln
        z2 = eps**2 / sigma2
        c = (gammaln((nu + 1) / 2) - gammaln(nu / 2)
             - 0.5 * np.log(np.pi * (nu - 2)))
        ll = np.sum(c - 0.5 * np.log(sigma2)
                    - (nu + 1) / 2 * np.log1p(z2 / (nu - 2)))
    return -ll


@dataclass
class GARCHFit:
    """Fitted GARCH(1,1). Carries everything needed to forecast and to explain
    the result in an interview (persistence, long-run vol, half-life of a vol
    shock).
    """
    omega: float
    alpha: float
    beta: float
    mu: float
    innovation: str
    nu: float = np.inf
    loglik: float = np.nan
    conditional_var: NDArray = field(default_factory=lambda: np.array([]))
    converged: bool = False

    @property
    def persistence(self) -> float:
        return self.alpha + self.beta

    @property
    def long_run_var(self) -> float:
        p = self.persistence
        return self.omega / (1 - p) if p < 1 else np.inf

    @property
    def long_run_vol(self) -> float:
        return float(np.sqrt(self.long_run_var))

    @property
    def vol_shock_half_life(self) -> float:
        """Days for a variance shock to decay halfway back to the long-run level:
        ln(0.5)/ln(persistence). Near-integrated (persistence ~1) => very slow
        decay, the HSI-policy-shock signature.
        """
        p = self.persistence
        return float(np.log(0.5) / np.log(p)) if 0 < p < 1 else np.inf

    def forecast_var(self, horizon: int) -> NDArray:
        """Multi-step-ahead variance forecast. Mean-reverts geometrically to the
        long-run variance:
            E[sigma^2_{t+h}] = sigma_bar^2 + (alpha+beta)^{h-1} (sigma^2_{t+1} - sigma_bar^2)
        """
        last_eps2 = float(self.conditional_var[-1])  # proxy current level
        sig2_next = (self.omega + self.alpha * last_eps2
                     + self.beta * self.conditional_var[-1])
        lr = self.long_run_var
        p = self.persistence
        out = np.empty(horizon)
        for h in range(1, horizon + 1):
            out[h - 1] = lr + (p ** (h - 1)) * (sig2_next - lr)
        return out


def fit_garch(
    returns: NDArray, innovation: Innovation = "gaussian", demean: bool = True
) -> GARCHFit:
    """Fit GARCH(1,1) by maximum likelihood with variance targeting.

    returns: simple or log returns (in level units, e.g. 0.01 = 1%). We estimate
    only (alpha, beta[, nu]); omega follows from variance targeting; mu is the
    sample mean (small for daily equity data).
    """
    r = np.asarray(returns, dtype=float)
    mu = float(r.mean()) if demean else 0.0
    eps = r - mu
    uncond_var = float(np.var(eps))

    x0 = [0.08, 0.90] + ([8.0] if innovation == "student-t" else [])
    bounds = [(1e-6, 0.5), (1e-6, 0.999)] + ([(2.1, 200.0)] if innovation == "student-t" else [])

    res = minimize(
        _garch_negloglik, x0, args=(eps, uncond_var, innovation),
        method="L-BFGS-B", bounds=bounds,
    )
    alpha, beta = float(res.x[0]), float(res.x[1])
    nu = float(res.x[2]) if innovation == "student-t" else np.inf
    omega = (1.0 - alpha - beta) * uncond_var
    sigma2 = _garch_recursion(eps, omega, alpha, beta, uncond_var)

    return GARCHFit(
        omega=omega, alpha=alpha, beta=beta, mu=mu, innovation=innovation,
        nu=nu, loglik=float(-res.fun), conditional_var=sigma2,
        converged=bool(res.success),
    )


# --------------------------------------------------------------------------- #
#  EGARCH(1,1) — captures the leverage effect
# --------------------------------------------------------------------------- #

_E_ABS_Z = np.sqrt(2.0 / np.pi)  # E|z| for a standard normal


def _egarch_recursion(
    eps: NDArray, omega: float, alpha: float, beta: float, gamma: float, lnv0: float
) -> NDArray:
    n = len(eps)
    lnv = np.empty(n)
    lnv[0] = lnv0
    for t in range(1, n):
        sigma_prev = np.sqrt(np.exp(lnv[t - 1]))
        z = eps[t - 1] / sigma_prev if sigma_prev > 0 else 0.0
        lnv[t] = (omega + beta * lnv[t - 1]
                  + alpha * (abs(z) - _E_ABS_Z) + gamma * z)
    return np.exp(lnv)


def _egarch_negloglik(params: NDArray, eps: NDArray, lnv0: float) -> float:
    omega, alpha, beta, gamma = params
    if abs(beta) >= 0.9999:  # |beta|<1 for stationarity in log-variance
        return 1e10
    sigma2 = _egarch_recursion(eps, omega, alpha, beta, gamma, lnv0)
    if not np.all(np.isfinite(sigma2)) or np.any(sigma2 <= 0):
        return 1e10
    ll = -0.5 * np.sum(np.log(2 * np.pi) + np.log(sigma2) + eps**2 / sigma2)
    return -ll


@dataclass
class EGARCHFit:
    omega: float
    alpha: float
    beta: float
    gamma: float
    mu: float
    loglik: float = np.nan
    conditional_var: NDArray = field(default_factory=lambda: np.array([]))
    converged: bool = False

    @property
    def has_leverage(self) -> bool:
        """gamma < 0 means negative shocks raise vol more — the leverage effect."""
        return self.gamma < 0


def fit_egarch(returns: NDArray, demean: bool = True) -> EGARCHFit:
    """Fit EGARCH(1,1) by maximum likelihood. The sign of gamma is the headline:
    gamma<0 confirms the leverage/asymmetry that drives equity skew.
    """
    r = np.asarray(returns, dtype=float)
    mu = float(r.mean()) if demean else 0.0
    eps = r - mu
    lnv0 = float(np.log(np.var(eps)))

    x0 = [lnv0 * 0.05, 0.15, 0.95, -0.05]
    res = minimize(_egarch_negloglik, x0, args=(eps, lnv0), method="Nelder-Mead",
                   options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 5000})
    omega, alpha, beta, gamma = map(float, res.x)
    sigma2 = _egarch_recursion(eps, omega, alpha, beta, gamma, lnv0)
    return EGARCHFit(omega, alpha, beta, gamma, mu, float(-res.fun), sigma2,
                     bool(res.success))
