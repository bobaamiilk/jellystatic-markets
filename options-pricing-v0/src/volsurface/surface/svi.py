"""SVI volatility surface construction (Stage 5) -- the flagship deliverable.

Raw SVI (Gatheral) total-variance slice in log-moneyness k = ln(K/F):

    w(k) = a + b [ rho (k - m) + sqrt((k - m)^2 + s^2) ]

with parameters (a, b, rho, m, s). Total implied variance w = sigma_IV^2 * tau;
implied vol is sigma_IV(k) = sqrt(w(k) / tau).

Static (butterfly) no-arbitrage sufficient condition: b (1 + |rho|) <= 4.
We impose this as a hard inequality constraint in the calibration (SLSQP),
because an unconstrained least-squares fit has no reason to respect it and can
"fit well" while embedding a butterfly arbitrage.

Calendar no-arbitrage across maturities (total variance non-decreasing in tau at
fixed k) is checked post-fit by `check_calendar_arbitrage`; the full joint-fit
fix (SSVI) is noted as the production upgrade.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize


@dataclass(frozen=True)
class SVIParams:
    """One calibrated SVI slice (single maturity)."""

    a: float
    b: float
    rho: float
    m: float
    s: float
    tau: float

    def total_variance(self, k: NDArray) -> NDArray:
        k = np.asarray(k, dtype=float)
        return self.a + self.b * (
            self.rho * (k - self.m) + np.sqrt((k - self.m) ** 2 + self.s**2)
        )

    def implied_vol(self, k: NDArray) -> NDArray:
        w = self.total_variance(k)
        return np.sqrt(np.maximum(w, 1e-12) / self.tau)

    def butterfly_margin(self) -> float:
        """4 - b(1+|rho|). Non-negative => static no-arb sufficient condition met."""
        return 4.0 - self.b * (1.0 + abs(self.rho))


def _svi_w(params: NDArray, k: NDArray) -> NDArray:
    a, b, rho, m, s = params
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + s**2))


def calibrate_svi_slice(
    k: NDArray,
    total_variance: NDArray,
    tau: float,
    weights: NDArray | None = None,
    enforce_no_arb: bool = True,
    n_restarts: int = 5,
    seed: int | None = 0,
) -> tuple[SVIParams, float]:
    """Calibrate one SVI slice by weighted least squares with the b(1+|rho|)<=4
    constraint.

    Parameters
    ----------
    k : log-moneyness array.
    total_variance : market total variance w_i = (sigma_IV_i)^2 * tau.
    weights : per-point weights (vega or inverse-spread). Defaults to equal.
    enforce_no_arb : impose the butterfly constraint via SLSQP.
    n_restarts : multi-start to dodge local optima (the loss surface is shallow
        near the constraint boundary on noisy slices).

    Returns (SVIParams, rmse).
    """
    k = np.asarray(k, dtype=float)
    w_mkt = np.asarray(total_variance, dtype=float)
    if weights is None:
        weights = np.ones_like(k)
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()

    def objective(p: NDArray) -> float:
        resid = _svi_w(p, k) - w_mkt
        return float(np.sum(weights * resid**2))

    # Constraints: b >= 0, s > 0, |rho| <= 1, and butterfly b(1+|rho|) <= 4.
    bounds = [
        (1e-8, np.max(w_mkt) * 2 + 1e-6),  # a
        (1e-8, 10.0),                       # b
        (-0.999, 0.999),                    # rho
        (k.min() - 1.0, k.max() + 1.0),     # m
        (1e-4, 5.0),                        # s
    ]
    constraints = []
    if enforce_no_arb:
        constraints.append(
            {"type": "ineq", "fun": lambda p: 4.0 - p[1] * (1.0 + abs(p[2]))}
        )

    rng = np.random.default_rng(seed)
    best_p, best_obj = None, np.inf
    atm_var = float(np.interp(0.0, k, w_mkt)) if len(k) else 0.04
    for i in range(n_restarts):
        x0 = np.array([
            atm_var * (0.5 + rng.random()),         # a near ATM variance
            0.1 + rng.random(),                      # b
            -0.5 + rng.random() - 0.5,               # rho in (-1,0) bias (equity skew)
            rng.normal(0, 0.1),                      # m
            0.1 + rng.random() * 0.4,                # s
        ])
        x0[2] = float(np.clip(x0[2], -0.95, 0.95))
        try:
            res = minimize(
                objective, x0, method="SLSQP",
                bounds=bounds, constraints=constraints,
                options={"maxiter": 500, "ftol": 1e-12},
            )
        except Exception:
            continue
        if res.success and res.fun < best_obj:
            best_obj, best_p = res.fun, res.x

    if best_p is None:
        raise RuntimeError("SVI calibration failed on all restarts.")

    a, b, rho, m, s = best_p
    params = SVIParams(a=a, b=b, rho=rho, m=m, s=s, tau=tau)
    rmse = float(np.sqrt(np.mean((_svi_w(best_p, k) - w_mkt) ** 2)))
    return params, rmse


def check_butterfly_arbitrage(params: SVIParams, k_grid: NDArray) -> bool:
    """Verify the fitted slice has non-negative risk-neutral density on a dense
    grid (the g(k) function from Gatheral). Returns True if arb-free.

    g(k) = (1 - k w'/(2w))^2 - (w'/2)^2 (1/w + 1/4) + w''/2 >= 0.
    A direct numerical check that catches violations the sufficient condition
    b(1+|rho|)<=4 might miss for extreme slices.
    """
    k = np.asarray(k_grid, dtype=float)
    w = params.total_variance(k)
    dk = k[1] - k[0]
    wp = np.gradient(w, dk)
    wpp = np.gradient(wp, dk)
    g = (
        (1 - k * wp / (2 * w)) ** 2
        - (wp / 2) ** 2 * (1.0 / w + 0.25)
        + wpp / 2
    )
    return bool(np.all(g >= -1e-6))


def check_calendar_arbitrage(slices: list[SVIParams], k_grid: NDArray) -> bool:
    """Total variance must be non-decreasing in tau at every k. Returns True if
    no calendar arbitrage across the supplied (tau-ordered) slices."""
    ordered = sorted(slices, key=lambda s: s.tau)
    k = np.asarray(k_grid, dtype=float)
    prev_w = None
    for sl in ordered:
        w = sl.total_variance(k)
        if prev_w is not None and np.any(w < prev_w - 1e-6):
            return False
        prev_w = w
    return True


class VolSurface:
    """A fitted surface: a collection of SVI slices keyed by maturity, with
    interpolation in the maturity dimension via total variance (the correct
    quantity to interpolate -- linear in w, not in sigma)."""

    def __init__(self, slices: list[SVIParams]):
        self.slices = sorted(slices, key=lambda s: s.tau)
        self.taus = np.array([s.tau for s in self.slices])

    def implied_vol(self, k: float, tau: float) -> float:
        """IV at arbitrary (log-moneyness, maturity) via linear interpolation in
        total variance across the two bracketing maturity slices."""
        ws = np.array([s.total_variance(np.array([k]))[0] for s in self.slices])
        if tau <= self.taus[0]:
            w = ws[0]
        elif tau >= self.taus[-1]:
            w = ws[-1]
        else:
            i = np.searchsorted(self.taus, tau)
            t0, t1 = self.taus[i - 1], self.taus[i]
            frac = (tau - t0) / (t1 - t0)
            w = (1 - frac) * ws[i - 1] + frac * ws[i]
        return float(np.sqrt(max(w, 1e-12) / tau))

    def is_arbitrage_free(self, k_grid: NDArray) -> dict:
        """Run both butterfly (per slice) and calendar (across slices) checks."""
        butterfly = {s.tau: check_butterfly_arbitrage(s, k_grid) for s in self.slices}
        calendar = check_calendar_arbitrage(self.slices, k_grid)
        return {"butterfly_per_slice": butterfly, "calendar": calendar}
