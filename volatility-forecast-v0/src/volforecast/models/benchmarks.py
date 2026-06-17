"""Benchmark volatility estimators (Stage 2): rolling window and EWMA.

These are the yardsticks GARCH must beat. The key teaching points are encoded in
the code and tests:

  - ROLLING (equal-weight) variance suffers the GHOSTING artifact: a single large
    return holds the estimate elevated for exactly `window` days, then the
    estimate drops off a cliff when that return exits the window — a phantom
    regime change with no market event.

  - EWMA is GARCH(1,1) with omega=0, alpha=(1-lambda), beta=lambda, i.e.
    alpha+beta=1. That unit persistence means NO mean reversion: the multi-step
    EWMA forecast is a flat line. EWMA is a fine *level* estimator but a poor
    *forecaster* at horizon. We expose this equivalence directly.

All estimators are causal (use only past returns) so they can be dropped into
the walk-forward evaluator without look-ahead.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def rolling_variance(returns: NDArray, window: int = 21) -> NDArray:
    """Equal-weighted rolling variance of (demeaned) returns. Causal: entry t
    uses returns (t-window+1 .. t). First window-1 entries are NaN.

    Watch for ghosting: when a large |return| leaves the trailing window the
    estimate drops abruptly even though nothing happened in the market that day.
    """
    r = np.asarray(returns, dtype=float)
    n = len(r)
    out = np.full(n, np.nan)
    for t in range(window - 1, n):
        w = r[t - window + 1:t + 1]
        out[t] = np.var(w)
    return out


def ewma_variance(returns: NDArray, lam: float = 0.94, var0: float | None = None) -> NDArray:
    """Exponentially-weighted moving-average variance (RiskMetrics).

        sigma^2_t = (1-lambda) r^2_{t-1} + lambda sigma^2_{t-1}

    lambda=0.94 is the RiskMetrics daily default. Causal by construction. This is
    exactly GARCH(1,1) with omega=0, alpha=1-lambda, beta=lambda (see
    `ewma_as_garch_params`), hence no mean reversion.
    """
    r = np.asarray(returns, dtype=float)
    n = len(r)
    out = np.empty(n)
    out[0] = var0 if var0 is not None else float(np.var(r[: min(n, 21)]))
    for t in range(1, n):
        out[t] = (1.0 - lam) * r[t - 1] ** 2 + lam * out[t - 1]
    return out


def ewma_as_garch_params(lam: float = 0.94) -> dict[str, float]:
    """Return the GARCH(1,1) parameters EWMA is equivalent to. Makes the identity
    explicit: omega=0, alpha=1-lambda, beta=lambda, persistence=1.
    """
    return {"omega": 0.0, "alpha": 1.0 - lam, "beta": lam,
            "persistence": 1.0}


def ewma_forecast(last_var: float, horizon: int) -> NDArray:
    """EWMA multi-step variance forecast — a FLAT line at the current level,
    because persistence=1 means no pull toward any long-run variance. Contrast
    with GARCH's geometric decay to sigma_bar^2.
    """
    return np.full(horizon, float(last_var))
