"""Closed-form European option pricing: Black-Scholes (spot) and Black-76 (futures).

Implements the Stage 1 closed-form solution derived via risk-neutral valuation,
plus the analytic Greeks that fall out of the same derivation. Black-76 is the
futures-referenced variant required for APAC index options (KOSPI200, Nikkei,
SGX-listed contracts) which are quoted off futures, not spot + carry.

All functions are vectorized over arrays via numpy and degrade gracefully to
scalars. No look-ahead, no global state: pure functions of their arguments.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm

OptionType = Literal["call", "put"]

_SQRT_2PI = np.sqrt(2.0 * np.pi)


def _d1_d2(
    F: NDArray, K: NDArray, sigma: NDArray, tau: NDArray
) -> tuple[NDArray, NDArray]:
    """Return (d1, d2) in *forward* terms. Works for both BS and Black-76 once
    the inputs are expressed via the forward price F.

    d1 = [ln(F/K) + 0.5 * sigma^2 * tau] / (sigma * sqrt(tau))
    d2 = d1 - sigma * sqrt(tau)

    Guards against sigma*sqrt(tau) == 0 (expiry / zero-vol) by returning +/-inf
    of the correct sign so that Phi(d) collapses to the intrinsic indicator.
    """
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    tau = np.asarray(tau, dtype=float)

    vol_sqrt_t = sigma * np.sqrt(tau)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(F / K) + 0.5 * sigma**2 * tau) / vol_sqrt_t
        d2 = d1 - vol_sqrt_t
    # Where vol_sqrt_t == 0, d1/d2 are +-inf depending on moneyness sign.
    degenerate = vol_sqrt_t == 0
    if np.any(degenerate):
        sign = np.sign(np.log(np.where(K > 0, F / K, 1.0)))
        d1 = np.where(degenerate, np.where(sign >= 0, np.inf, -np.inf), d1)
        d2 = np.where(degenerate, np.where(sign >= 0, np.inf, -np.inf), d2)
    return d1, d2


def black76_price(
    F: ArrayLike,
    K: ArrayLike,
    sigma: ArrayLike,
    tau: ArrayLike,
    r: ArrayLike,
    option_type: OptionType = "call",
) -> NDArray:
    """Black-76 price of a European option on a *futures* price F.

    C = e^{-r tau} [F Phi(d1) - K Phi(d2)]
    P = e^{-r tau} [K Phi(-d2) - F Phi(-d1)]

    Parameters
    ----------
    F : forward / futures price.
    K : strike.
    sigma : volatility (annualised).
    tau : time to expiry in years.
    r : risk-free rate (continuous) used only for discounting in Black-76.
    option_type : 'call' or 'put'.
    """
    F, K, sigma, tau, r = (np.asarray(x, dtype=float) for x in (F, K, sigma, tau, r))
    d1, d2 = _d1_d2(F, K, sigma, tau)
    disc = np.exp(-r * tau)
    if option_type == "call":
        price = disc * (F * norm.cdf(d1) - K * norm.cdf(d2))
    elif option_type == "put":
        price = disc * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    return price


def black_scholes_price(
    S: ArrayLike,
    K: ArrayLike,
    sigma: ArrayLike,
    tau: ArrayLike,
    r: ArrayLike,
    q: ArrayLike = 0.0,
    option_type: OptionType = "call",
) -> NDArray:
    """Black-Scholes price on *spot* S with continuous dividend yield q.

    Internally converts to the forward F = S e^{(r - q) tau} and calls Black-76,
    which makes the spot/forward relationship explicit rather than hidden in two
    near-duplicate formulas. This is the cleanest way to keep BS and Black-76
    consistent by construction.
    """
    S, q = np.asarray(S, dtype=float), np.asarray(q, dtype=float)
    tau = np.asarray(tau, dtype=float)
    r = np.asarray(r, dtype=float)
    F = S * np.exp((r - q) * tau)
    return black76_price(F, K, sigma, tau, r, option_type)


@dataclass(frozen=True)
class Greeks:
    """Container for the standard first/second-order Greeks (per option)."""

    delta: NDArray
    gamma: NDArray
    vega: NDArray
    theta: NDArray
    rho: NDArray


def black_scholes_greeks(
    S: ArrayLike,
    K: ArrayLike,
    sigma: ArrayLike,
    tau: ArrayLike,
    r: ArrayLike,
    q: ArrayLike = 0.0,
    option_type: OptionType = "call",
) -> Greeks:
    """Analytic Greeks for a spot Black-Scholes option.

    vega = S e^{-q tau} phi(d1) sqrt(tau)   (per unit vol, NOT per 1%)
    This is the same vega that governs IV-solver stability in Stage 3: it -> 0
    for deep OTM and very short tau, which is exactly where Newton breaks.
    """
    S, K, sigma, tau, r, q = (
        np.asarray(x, dtype=float) for x in (S, K, sigma, tau, r, q)
    )
    F = S * np.exp((r - q) * tau)
    d1, d2 = _d1_d2(F, K, sigma, tau)
    sqrt_t = np.sqrt(tau)
    pdf_d1 = norm.pdf(d1)
    disc_q = np.exp(-q * tau)
    disc_r = np.exp(-r * tau)

    vega = S * disc_q * pdf_d1 * sqrt_t
    gamma = disc_q * pdf_d1 / (S * sigma * sqrt_t)

    if option_type == "call":
        delta = disc_q * norm.cdf(d1)
        theta = (
            -S * disc_q * pdf_d1 * sigma / (2 * sqrt_t)
            - r * K * disc_r * norm.cdf(d2)
            + q * S * disc_q * norm.cdf(d1)
        )
        rho = K * tau * disc_r * norm.cdf(d2)
    elif option_type == "put":
        delta = -disc_q * norm.cdf(-d1)
        theta = (
            -S * disc_q * pdf_d1 * sigma / (2 * sqrt_t)
            + r * K * disc_r * norm.cdf(-d2)
            - q * S * disc_q * norm.cdf(-d1)
        )
        rho = -K * tau * disc_r * norm.cdf(-d2)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

    return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)


def put_call_parity_residual(
    call: ArrayLike,
    put: ArrayLike,
    S: ArrayLike,
    K: ArrayLike,
    tau: ArrayLike,
    r: ArrayLike,
    q: ArrayLike = 0.0,
) -> NDArray:
    """C - P - (S e^{-q tau} - K e^{-r tau}). Should be ~0 for consistent prices.

    This is the Stage 1 sanity identity and the Stage 4 data-cleaning filter in
    one function: feed it model prices to test the implementation, or market
    bid/ask mids to detect stale/crossed quotes.
    """
    call, put, S, K, tau, r, q = (
        np.asarray(x, dtype=float) for x in (call, put, S, K, tau, r, q)
    )
    return call - put - (S * np.exp(-q * tau) - K * np.exp(-r * tau))
