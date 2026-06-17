"""Synthetic return generators with KNOWN volatility ground truth.

Same philosophy as Projects 1-2: simulate from a model whose parameters we
control, so estimation can be validated by recovery and the latent sigma_t is
actually known for scoring.

  - simulate_garch: GARCH(1,1) with given (omega, alpha, beta); we get back both
    returns and the TRUE conditional variance path.
  - simulate_egarch: includes a leverage term (gamma<0) so EGARCH should recover
    a negative gamma.
  - simulate_regime_returns: a two-state (calm/crisis) vol process to exercise
    the regime detector.

Defaults evoke daily APAC index returns: ~1% daily vol, strong persistence.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def simulate_garch(
    n: int = 2000, omega: float = 2e-6, alpha: float = 0.08, beta: float = 0.90,
    seed: int = 0,
) -> tuple[NDArray, NDArray]:
    """Simulate GARCH(1,1) returns. Returns (returns, true_conditional_variance).

    Default persistence alpha+beta = 0.98 (near-integrated, like real equity
    indices). Long-run daily vol = sqrt(omega/(1-alpha-beta)) ~ 1%.
    """
    rng = np.random.default_rng(seed)
    r = np.empty(n)
    sig2 = np.empty(n)
    sig2[0] = omega / (1 - alpha - beta)
    for t in range(n):
        if t > 0:
            sig2[t] = omega + alpha * r[t - 1] ** 2 + beta * sig2[t - 1]
        r[t] = np.sqrt(sig2[t]) * rng.standard_normal()
    return r, sig2


def simulate_egarch(
    n: int = 2000, omega: float = -0.10, alpha: float = 0.15, beta: float = 0.97,
    gamma: float = -0.08, seed: int = 0,
) -> tuple[NDArray, NDArray]:
    """Simulate EGARCH(1,1) with a leverage term. gamma<0 => negative shocks
    raise vol more. Returns (returns, true_conditional_variance).
    """
    rng = np.random.default_rng(seed)
    E_ABS_Z = np.sqrt(2.0 / np.pi)
    r = np.empty(n)
    lnv = np.empty(n)
    lnv[0] = omega / (1 - beta)
    for t in range(n):
        if t > 0:
            sig_prev = np.sqrt(np.exp(lnv[t - 1]))
            z = r[t - 1] / sig_prev if sig_prev > 0 else 0.0
            lnv[t] = (omega + beta * lnv[t - 1]
                      + alpha * (abs(z) - E_ABS_Z) + gamma * z)
        r[t] = np.sqrt(np.exp(lnv[t])) * rng.standard_normal()
    return r, np.exp(lnv)


def simulate_regime_returns(
    n: int = 1500, calm_vol: float = 0.008, crisis_vol: float = 0.030,
    p_stay_calm: float = 0.98, p_stay_crisis: float = 0.92, seed: int = 0,
) -> tuple[NDArray, NDArray]:
    """Two-state Markov regime vol process. Returns (returns, true_state) where
    state 0=calm, 1=crisis. Exercises the regime detector and exposure scaling.
    """
    rng = np.random.default_rng(seed)
    state = np.zeros(n, dtype=int)
    for t in range(1, n):
        if state[t - 1] == 0:
            state[t] = 0 if rng.random() < p_stay_calm else 1
        else:
            state[t] = 1 if rng.random() < p_stay_crisis else 0
    vol = np.where(state == 0, calm_vol, crisis_vol)
    r = vol * rng.standard_normal(n)
    return r, state
