"""Walk-forward, out-of-sample forecast evaluation (Stage 3 harness).

In-sample fit quality is not evidence a forecaster works. This harness re-fits
each model on a rolling in-sample window, FREEZES it, produces one-step-ahead
forecasts on the next out-of-sample block, rolls forward, and scores only the
concatenated OOS forecasts. It then runs Diebold-Mariano on the OOS QLIKE series
so the comparison comes with a significance verdict — including the honest
"no significant difference" outcome, which means don't pay for GARCH complexity.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from volforecast.models.garch import fit_garch
from volforecast.models.benchmarks import ewma_variance, rolling_variance
from volforecast.evaluation.losses import (
    qlike_loss, mse_loss, per_period_qlike, diebold_mariano,
)


@dataclass
class WalkForwardResult:
    oos_var_garch: NDArray
    oos_var_ewma: NDArray
    oos_var_roll: NDArray
    r2_oos: NDArray
    qlike: dict[str, float] = field(default_factory=dict)
    mse: dict[str, float] = field(default_factory=dict)
    dm_garch_vs_ewma: object = None


def walk_forward_eval(
    returns: NDArray,
    train_size: int = 500,
    test_size: int = 21,
    ewma_lambda: float = 0.94,
    roll_window: int = 21,
) -> WalkForwardResult:
    """Compare GARCH(1,1), EWMA, and rolling variance out-of-sample.

    For each block: fit GARCH on the trailing train window (freeze), then forecast
    one-step variance across the test block by running each model's recursion with
    frozen params over the realised returns. Score against r_t^2 (the proxy).
    """
    r = np.asarray(returns, dtype=float)
    n = len(r)

    g = np.full(n, np.nan)
    e = np.full(n, np.nan)
    rl = np.full(n, np.nan)

    start = 0
    while start + train_size + test_size <= n:
        tr0, tr1 = start, start + train_size
        te0, te1 = tr1, tr1 + test_size

        fit = fit_garch(r[tr0:tr1])
        # one-step GARCH variance across the test block, recursion seeded from
        # the last in-sample conditional variance (frozen params).
        sig2 = fit.conditional_var[-1]
        mu = fit.mu
        for t in range(te0, te1):
            eps_prev = r[t - 1] - mu
            sig2 = fit.omega + fit.alpha * eps_prev**2 + fit.beta * sig2
            g[t] = sig2

        # EWMA + rolling over the same realised returns (causal, frozen lambda/window)
        e_full = ewma_variance(r[:te1], lam=ewma_lambda)
        rl_full = rolling_variance(r[:te1], window=roll_window)
        e[te0:te1] = e_full[te0:te1]
        rl[te0:te1] = rl_full[te0:te1]

        start += test_size

    r2 = r**2
    mask = np.isfinite(g) & np.isfinite(e) & np.isfinite(rl)
    r2m = np.where(mask, r2, np.nan)

    ql = {
        "GARCH": qlike_loss(g, r2m),
        "EWMA": qlike_loss(e, r2m),
        "Rolling": qlike_loss(rl, r2m),
    }
    ms = {
        "GARCH": mse_loss(g, r2m),
        "EWMA": mse_loss(e, r2m),
        "Rolling": mse_loss(rl, r2m),
    }
    dm = diebold_mariano(per_period_qlike(g, r2m), per_period_qlike(e, r2m))

    return WalkForwardResult(g, e, rl, r2m, ql, ms, dm)
