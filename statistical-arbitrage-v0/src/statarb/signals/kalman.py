"""Time-varying hedge ratio via Kalman filter (Stage 2).

A static OLS beta goes stale: the Samsung / SK Hynix relationship, for example,
shifts as their product mix diverges through a memory-chip cycle. Two upgrades
over static beta:

  - rolling-window OLS (simple, but a hard bias-variance tradeoff in the window
    length), and
  - a Kalman filter treating beta as a latent random walk (smoothly adaptive,
    no arbitrary window).

State-space model (random-walk beta, optional intercept):
    state:        beta_t = beta_{t-1} + w_t,        w_t ~ N(0, Q)
    observation:  y_t    = beta_t * x_t + v_t,      v_t ~ N(0, R)

The filter is causal by construction: beta_t uses only information up to t, so
spreads built from it carry no look-ahead — exactly what the Stage-4 backtest
requires.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class KalmanResult:
    beta: NDArray          # filtered hedge ratio per time step
    intercept: NDArray     # filtered intercept per time step (zeros if disabled)
    spread: NDArray        # y_t - (intercept_t + beta_t * x_t), i.e. innovation
    state_cov_trace: NDArray  # trace of state covariance (uncertainty over time)


def kalman_hedge_ratio(
    y: NDArray,
    x: NDArray,
    delta: float = 1e-4,
    obs_var: float = 1e-3,
    with_intercept: bool = True,
) -> KalmanResult:
    """Estimate a time-varying hedge ratio with a Kalman filter.

    Parameterisation follows the common quant convention (Chan): the state
    transition covariance is Q = delta/(1-delta) * I, controlling how fast beta
    is allowed to move, and obs_var = R is the measurement noise.

      - larger delta  => beta adapts faster (more responsive, noisier)
      - larger obs_var => beta adapts slower (smoother, more lag)

    These two knobs are the Kalman analogue of the rolling-window length: the
    bias-variance tradeoff doesn't disappear, it just becomes (Q, R). In
    production you'd estimate them by MLE; here they're explicit inputs.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    n = len(y)
    dim = 2 if with_intercept else 1

    # State transition noise covariance.
    wn = delta / (1.0 - delta)
    Q = wn * np.eye(dim)

    beta_out = np.zeros(n)
    alpha_out = np.zeros(n)
    spread_out = np.zeros(n)
    cov_trace = np.zeros(n)

    # Priors.
    state = np.zeros(dim)             # [intercept, beta] or [beta]
    P = np.eye(dim)                   # state covariance

    for t in range(n):
        # Observation matrix H_t maps state -> predicted y_t.
        H = np.array([1.0, x[t]]) if with_intercept else np.array([x[t]])

        # --- Predict ---
        # State is a random walk, so predicted state = previous state.
        P = P + Q

        # --- Innovation (this is the tradeable spread) ---
        y_pred = H @ state
        e = y[t] - y_pred             # prediction error / spread
        S = H @ P @ H + obs_var       # innovation variance

        # --- Update (Kalman gain) ---
        Kgain = (P @ H) / S
        state = state + Kgain * e
        P = P - np.outer(Kgain, H) @ P

        if with_intercept:
            alpha_out[t], beta_out[t] = state[0], state[1]
        else:
            beta_out[t] = state[0]
        spread_out[t] = e
        cov_trace[t] = np.trace(P)

    return KalmanResult(beta_out, alpha_out, spread_out, cov_trace)


def rolling_ols_hedge_ratio(y: NDArray, x: NDArray, window: int = 60) -> NDArray:
    """Rolling-window OLS hedge ratio — the simpler time-varying alternative.

    Causal: beta_t is fit on the window ending at t-1..t (no future data). The
    first `window-1` entries are NaN. Exposes the bias-variance tradeoff
    directly: short window = responsive but noisy, long window = stable but laggy.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    n = len(y)
    beta = np.full(n, np.nan)
    for t in range(window, n + 1):
        xs = x[t - window:t]
        ys = y[t - window:t]
        xm, ym = xs.mean(), ys.mean()
        var = np.sum((xs - xm) ** 2)
        if var > 0:
            beta[t - 1] = np.sum((xs - xm) * (ys - ym)) / var
    return beta
