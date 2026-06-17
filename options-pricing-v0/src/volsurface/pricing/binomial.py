"""Cox-Ross-Rubinstein binomial tree (Stage 2).

Handles European and American exercise. American support is the whole point:
HKEX single-stock options (Tencent, HSBC, AIA) are American on dividend-paying
underlyings, where early exercise near ex-div is genuinely optimal and plain
Black-Scholes is structurally incapable of pricing them.

Includes:
  - CRR European / American pricing with vectorised backward induction;
  - an explicit probability-validity guard (p must lie in [0,1]);
  - escrowed discrete-dividend adjustment (subtract PV of known cash dividends
    from S0 before building the tree);
  - Richardson extrapolation (2*V_{2n} - V_n) to damp the odd/even oscillation.
"""
from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
from numpy.typing import NDArray

OptionType = Literal["call", "put"]
Exercise = Literal["european", "american"]


def _crr_params(sigma: float, r: float, q: float, dt: float):
    """CRR up/down factors and risk-neutral probability.

    u = e^{sigma sqrt(dt)}, d = 1/u, p = (e^{(r-q) dt} - d)/(u - d).
    Raises if p falls outside [0,1] -- a real failure mode at high vol / coarse
    dt that must never silently produce a garbage price.
    """
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    if not (0.0 <= p <= 1.0):
        raise ValueError(
            f"CRR risk-neutral probability p={p:.4f} outside [0,1]; "
            f"increase n_steps or check sigma/r/q/dt (sigma={sigma}, dt={dt})."
        )
    return u, d, p


def binomial_price(
    S0: float,
    K: float,
    sigma: float,
    tau: float,
    r: float,
    q: float = 0.0,
    option_type: OptionType = "call",
    exercise: Exercise = "european",
    n_steps: int = 500,
    discrete_dividends: Sequence[tuple[float, float]] | None = None,
) -> float:
    """Price a vanilla option on a CRR tree.

    Parameters
    ----------
    discrete_dividends : optional list of (time_in_years, cash_amount). Handled
        by the escrowed-dividend method: PV of dividends with ex-date <= tau is
        subtracted from S0 before the tree is built. Approximate but standard,
        and far better than pretending a lumpy cash dividend is a continuous q.
    """
    S0_adj = S0
    if discrete_dividends:
        pv_divs = sum(
            amt * np.exp(-r * t) for (t, amt) in discrete_dividends if 0 < t <= tau
        )
        S0_adj = S0 - pv_divs
        if S0_adj <= 0:
            raise ValueError("PV of dividends exceeds spot; escrowed model invalid.")

    dt = tau / n_steps
    u, d, p = _crr_params(sigma, r, q, dt)
    disc = np.exp(-r * dt)

    # Terminal asset prices: S0_adj * u^j * d^(n-j) for j = 0..n.
    j = np.arange(n_steps + 1)
    S_T = S0_adj * u**j * d ** (n_steps - j)

    if option_type == "call":
        values = np.maximum(S_T - K, 0.0)
    else:
        values = np.maximum(K - S_T, 0.0)

    # Backward induction, vectorised over each level.
    for step in range(n_steps - 1, -1, -1):
        values = disc * (p * values[1:] + (1.0 - p) * values[:-1])
        if exercise == "american":
            j = np.arange(step + 1)
            S_node = S0_adj * u**j * d ** (step - j)
            if option_type == "call":
                intrinsic = np.maximum(S_node - K, 0.0)
            else:
                intrinsic = np.maximum(K - S_node, 0.0)
            values = np.maximum(values, intrinsic)

    return float(values[0])


def binomial_price_richardson(
    S0: float,
    K: float,
    sigma: float,
    tau: float,
    r: float,
    q: float = 0.0,
    option_type: OptionType = "call",
    exercise: Exercise = "european",
    n_steps: int = 500,
    **kwargs,
) -> float:
    """Richardson-extrapolated tree price.

    Numerical subtlety (validated empirically, see tests): the naive 2*V(2n)-V(n)
    extrapolation only cancels CRR's leading error cleanly when n is *even*,
    because CRR's error oscillates with period 2 in n -- at odd n, n and 2n sit
    on opposite phases of the oscillation and the extrapolation is actually
    WORSE than the raw tree. The robust fix used here is two-pronged:

      1. force the base step to be even (n -> n + (n % 2));
      2. average the two-point Richardson estimate over the even base n and the
         next even step, which damps any residual parity dependence.

    This is a real production lesson: a textbook extrapolation formula applied
    without regard to the oscillation parity gives you a worse number, silently.
    """
    n = n_steps + (n_steps % 2)  # nearest even >= n_steps

    def rich(base: int) -> float:
        v_n = binomial_price(
            S0, K, sigma, tau, r, q, option_type, exercise, base, **kwargs
        )
        v_2n = binomial_price(
            S0, K, sigma, tau, r, q, option_type, exercise, 2 * base, **kwargs
        )
        return 2.0 * v_2n - v_n

    return 0.5 * (rich(n) + rich(n + 2))


def convergence_path(
    S0: float,
    K: float,
    sigma: float,
    tau: float,
    r: float,
    q: float,
    option_type: OptionType,
    exercise: Exercise,
    n_grid: Sequence[int],
) -> list[tuple[int, float]]:
    """Return (n_steps, price) over a grid -- used to *show* the oscillation that
    motivates Richardson extrapolation in the Stage 2 validation."""
    return [
        (n, binomial_price(S0, K, sigma, tau, r, q, option_type, exercise, n))
        for n in n_grid
    ]
