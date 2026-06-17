"""Implied volatility inversion (Stage 3).

The production lesson from Stage 3: never call a bare Newton solver. Newton's
step is (price_error / vega), and vega -> 0 for deep-OTM / very short-dated
options -- exactly the KOSPI200 / HSI wing quotes you most need to invert -- so
the step explodes and diverges. The robust design is:

  1. Pre-check no-arbitrage bounds. If the price is below intrinsic or above the
     forward bound, NO sigma solves the equation -- return NaN with a reason,
     never hand an impossible root to a solver that will hang or return garbage.
  2. Brent (bracketed, derivative-free, guaranteed convergence) as the primary
     solver.
  3. Optional Newton polish from the Brent root for a few quadratic-convergence
     iterations, falling back to the Brent value if Newton misbehaves.

Works in Black-76 (forward) terms so it is correct for futures-referenced APAC
index options out of the box.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from volsurface.pricing.black_scholes import black76_price

OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class IVResult:
    """Implied vol plus diagnostics needed for downstream weighting.

    vega_at_solution is the key one: an IV solved at near-zero vega is
    statistically noisy and should be DOWN-WEIGHTED in the Stage 5 surface fit,
    not trusted equally with an ATM point.
    """

    iv: float
    converged: bool
    vega_at_solution: float
    n_iter: int
    reason: str = ""


def _forward_intrinsic(F: float, K: float, option_type: OptionType) -> float:
    return max(F - K, 0.0) if option_type == "call" else max(K - F, 0.0)


def _arbitrage_bounds_ok(
    price: float, F: float, K: float, tau: float, r: float, option_type: OptionType
) -> tuple[bool, str]:
    """Undiscounted price must lie within [intrinsic_fwd, upper_fwd].

    Lower: discounted forward intrinsic. Upper: e^{-r tau} F for a call,
    e^{-r tau} K for a put. Outside this band there is no real implied vol.
    """
    disc = np.exp(-r * tau)
    lower = disc * _forward_intrinsic(F, K, option_type)
    upper = disc * (F if option_type == "call" else K)
    if price < lower - 1e-12:
        return False, f"price {price:.6g} below lower no-arb bound {lower:.6g}"
    if price > upper + 1e-12:
        return False, f"price {price:.6g} above upper no-arb bound {upper:.6g}"
    return True, ""


def _black76_vega(F: float, K: float, sigma: float, tau: float, r: float) -> float:
    """Black-76 vega = e^{-r tau} F phi(d1) sqrt(tau)."""
    if sigma <= 0 or tau <= 0:
        return 0.0
    vol_sqrt_t = sigma * np.sqrt(tau)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * tau) / vol_sqrt_t
    return float(np.exp(-r * tau) * F * norm.pdf(d1) * np.sqrt(tau))


def implied_vol(
    price: float,
    F: float,
    K: float,
    tau: float,
    r: float,
    option_type: OptionType = "call",
    sigma_lo: float = 1e-4,
    sigma_hi: float = 5.0,
    newton_polish: bool = True,
    tol: float = 1e-8,
) -> IVResult:
    """Invert a single Black-76 price to implied volatility.

    Returns an IVResult; on failure iv is NaN and `reason` explains why (arb
    violation vs bracketing failure vs non-convergence) so the data pipeline can
    quarantine the quote intelligently rather than dropping it blindly.
    """
    ok, reason = _arbitrage_bounds_ok(price, F, K, tau, r, option_type)
    if not ok:
        return IVResult(np.nan, False, 0.0, 0, reason)

    def objective(sigma: float) -> float:
        return float(black76_price(F, K, sigma, tau, r, option_type)) - price

    f_lo, f_hi = objective(sigma_lo), objective(sigma_hi)
    if f_lo * f_hi > 0:
        # Root not bracketed in the default range. Try widening once.
        sigma_hi_wide = 10.0
        if objective(sigma_lo) * objective(sigma_hi_wide) > 0:
            return IVResult(
                np.nan, False, 0.0, 0,
                f"root not bracketed in [{sigma_lo}, {sigma_hi_wide}]",
            )
        sigma_hi = sigma_hi_wide

    # Brent: robust, bracketed, derivative-free primary solver.
    sigma_star, rr = brentq(
        objective, sigma_lo, sigma_hi, xtol=tol, full_output=True
    )
    n_iter = rr.iterations
    converged = rr.converged

    # Optional Newton polish for a couple of quadratic steps.
    if newton_polish and converged:
        s = sigma_star
        for _ in range(3):
            v = _black76_vega(F, K, s, tau, r)
            if v < 1e-10:
                break  # vega too small: Newton unsafe, keep Brent value
            step = objective(s) / v
            s_new = s - step
            if s_new <= 0 or not np.isfinite(s_new):
                break
            if abs(s_new - s) < tol:
                s = s_new
                break
            s = s_new
        # Accept polish only if it didn't worsen the residual.
        if abs(objective(s)) <= abs(objective(sigma_star)):
            sigma_star = s

    vega = _black76_vega(F, K, sigma_star, tau, r)
    return IVResult(
        iv=float(sigma_star),
        converged=bool(converged),
        vega_at_solution=vega,
        n_iter=int(n_iter),
        reason="",
    )


def implied_vol_chain(
    prices,
    F,
    strikes,
    tau,
    r,
    option_types,
):
    """Vectorised convenience wrapper over a chain. Returns a list of IVResult.

    Kept as a simple loop (each solve is independent and cheap) rather than a
    fake-vectorised version that would obscure the per-quote diagnostics, which
    are the whole point of the IVResult container.
    """
    import numpy as _np

    prices = _np.atleast_1d(prices)
    strikes = _np.atleast_1d(strikes)
    n = len(prices)

    def _b(x):
        x = _np.atleast_1d(x)
        return x if len(x) == n else _np.full(n, x[0])

    F_, tau_, r_ = _b(F), _b(tau), _b(r)
    if isinstance(option_types, str):
        option_types = [option_types] * n

    return [
        implied_vol(
            float(prices[i]), float(F_[i]), float(strikes[i]),
            float(tau_[i]), float(r_[i]), option_types[i],
        )
        for i in range(n)
    ]
