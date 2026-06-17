"""Monte Carlo pricing of European options under risk-neutral GBM (Stage 2).

Implements:
  - plain MC from first principles (exact GBM terminal sampling, no time-stepping
    needed for path-independent European payoffs);
  - antithetic variates (pair Z, -Z);
  - control variates using the underlying's known risk-neutral expectation
    E^Q[S_T] = S_0 e^{r tau} as the control (always available, no extra model).

Every estimator returns a point estimate AND a standard error, because an MC
price without an error bar is not a validated number -- the SE is what lets the
Stage 2 validation assert "MC matches Black-Scholes within tolerance".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
from numpy.typing import NDArray

OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class MCResult:
    """Monte Carlo estimate with its standard error and 95% CI half-width."""

    price: float
    std_error: float
    n_paths: int

    @property
    def ci95(self) -> tuple[float, float]:
        h = 1.959963984540054 * self.std_error
        return (self.price - h, self.price + h)


def _payoff(S_T: NDArray, K: float, option_type: OptionType) -> NDArray:
    if option_type == "call":
        return np.maximum(S_T - K, 0.0)
    if option_type == "put":
        return np.maximum(K - S_T, 0.0)
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def _terminal_prices(
    S0: float, r: float, q: float, sigma: float, tau: float, Z: NDArray
) -> NDArray:
    """Exact risk-neutral GBM terminal sampling:
    S_T = S0 exp[(r - q - 0.5 sigma^2) tau + sigma sqrt(tau) Z].
    """
    drift = (r - q - 0.5 * sigma**2) * tau
    diff = sigma * np.sqrt(tau) * Z
    return S0 * np.exp(drift + diff)


def mc_price_plain(
    S0: float,
    K: float,
    sigma: float,
    tau: float,
    r: float,
    q: float = 0.0,
    option_type: OptionType = "call",
    n_paths: int = 100_000,
    seed: int | None = None,
) -> MCResult:
    """Plain Monte Carlo. Error ~ O(n^{-1/2}), reported via the sample SE."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_paths)
    S_T = _terminal_prices(S0, r, q, sigma, tau, Z)
    disc = np.exp(-r * tau)
    payoffs = disc * _payoff(S_T, K, option_type)
    price = float(payoffs.mean())
    se = float(payoffs.std(ddof=1) / np.sqrt(n_paths))
    return MCResult(price=price, std_error=se, n_paths=n_paths)


def mc_price_antithetic(
    S0: float,
    K: float,
    sigma: float,
    tau: float,
    r: float,
    q: float = 0.0,
    option_type: OptionType = "call",
    n_paths: int = 100_000,
    seed: int | None = None,
) -> MCResult:
    """Antithetic variates: draw n/2 normals Z, use both Z and -Z.

    The estimator averages the *pair means* P_i = 0.5*(g(Z_i)+g(-Z_i)); the SE is
    computed across these n/2 pair-means, which correctly reflects the reduced
    variance (and avoids the classic bug of computing SE as if the 2*(n/2) draws
    were independent).
    """
    rng = np.random.default_rng(seed)
    half = n_paths // 2
    Z = rng.standard_normal(half)
    disc = np.exp(-r * tau)

    S_plus = _terminal_prices(S0, r, q, sigma, tau, Z)
    S_minus = _terminal_prices(S0, r, q, sigma, tau, -Z)
    g_plus = disc * _payoff(S_plus, K, option_type)
    g_minus = disc * _payoff(S_minus, K, option_type)
    pair_means = 0.5 * (g_plus + g_minus)

    price = float(pair_means.mean())
    se = float(pair_means.std(ddof=1) / np.sqrt(half))
    return MCResult(price=price, std_error=se, n_paths=2 * half)


def mc_price_control_variate(
    S0: float,
    K: float,
    sigma: float,
    tau: float,
    r: float,
    q: float = 0.0,
    option_type: OptionType = "call",
    n_paths: int = 100_000,
    seed: int | None = None,
) -> MCResult:
    """Control variate using the discounted terminal price as control.

    Control X = e^{-r tau} S_T, with known mean E[X] = S0 e^{-q tau}.
    Optimal coefficient c* = -Cov(Y, X)/Var(X) estimated from the sample
    (a small in-sample bias, standard and negligible at large n).
    """
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_paths)
    disc = np.exp(-r * tau)

    S_T = _terminal_prices(S0, r, q, sigma, tau, Z)
    Y = disc * _payoff(S_T, K, option_type)
    X = disc * S_T
    EX = S0 * np.exp(-q * tau)

    cov = np.cov(Y, X, ddof=1)
    c_star = -cov[0, 1] / cov[1, 1]
    Y_cv = Y + c_star * (X - EX)

    price = float(Y_cv.mean())
    se = float(Y_cv.std(ddof=1) / np.sqrt(n_paths))
    return MCResult(price=price, std_error=se, n_paths=n_paths)


def convergence_study(
    pricer: Callable[..., MCResult],
    n_grid: list[int],
    **kwargs,
) -> list[tuple[int, float, float]]:
    """Run `pricer` over a grid of path counts; return (n, price, se) tuples.

    Used by the Stage 2 convergence experiment to verify the O(n^{-1/2}) SE decay
    empirically (SE should roughly halve when n quadruples).
    """
    out = []
    for n in n_grid:
        res = pricer(n_paths=n, **kwargs)
        out.append((n, res.price, res.std_error))
    return out
