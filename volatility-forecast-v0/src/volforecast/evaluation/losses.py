"""Forecast evaluation (Stage 3) — you cannot use just any loss function.

The target sigma^2_t is latent; we score forecasts against the noisy proxy r_t^2
(which is chi-square-1 dispersed around the truth). Patton (2011) showed only
certain losses are ROBUST — meaning the noisy proxy doesn't change the expected
ranking of models. MSE-on-variance and QLIKE are robust; MAE-on-vol, R^2 of vol,
and RMSE-on-sigma are NOT — proxy noise can crown the wrong model.

  MSE   = mean( (sigma2_hat - r2)^2 )                 symmetric
  QLIKE = mean( r2/sigma2_hat + ln sigma2_hat )       asymmetric

QLIKE penalises UNDER-prediction of variance more heavily — which matches the
economics: under-sizing risk into a crisis (margin calls) is far costlier than
over-caution. It's also less dominated by a few extreme days than MSE.

Diebold-Mariano tests whether two models differ significantly in predictive
accuracy, using a HAC (Newey-West) variance because forecast-error differentials
are autocorrelated (especially for overlapping multi-step forecasts).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm


def mse_loss(sigma2_hat: NDArray, r2: NDArray) -> float:
    """Mean squared error on VARIANCE. Robust (Patton) but dominated by the few
    largest-|return| days, so it has high sampling variance in short samples.
    """
    a = np.asarray(sigma2_hat, dtype=float)
    b = np.asarray(r2, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.mean((a[m] - b[m]) ** 2))


def qlike_loss(sigma2_hat: NDArray, r2: NDArray) -> float:
    """QLIKE loss. Robust AND asymmetric (penalises under-prediction more).
    Generally the primary metric for vol forecasts.
    """
    a = np.asarray(sigma2_hat, dtype=float)
    b = np.asarray(r2, dtype=float)
    m = np.isfinite(a) & np.isfinite(b) & (a > 0)
    return float(np.mean(b[m] / a[m] + np.log(a[m])))


def mae_vol_loss(sigma2_hat: NDArray, r2: NDArray) -> float:
    """MAE on VOLATILITY. Included deliberately as a NON-robust loss: under a
    noisy proxy its expected ranking can disagree with the true ranking. Use it
    only to demonstrate the pitfall, never as the deciding metric.
    """
    a = np.sqrt(np.asarray(sigma2_hat, dtype=float))
    b = np.sqrt(np.asarray(r2, dtype=float))
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.mean(np.abs(a[m] - b[m])))


@dataclass
class DMResult:
    statistic: float
    pvalue: float
    better: str   # "A", "B", or "neither"
    mean_loss_diff: float


def _hac_variance(d: NDArray, lag: int | None = None) -> float:
    """Newey-West HAC long-run variance of the loss differential series d.

    Forecast-error differentials are serially correlated, so the naive variance
    understates uncertainty and DM over-rejects. The Bartlett-kernel HAC fixes
    this. Default lag ~ n^(1/3).
    """
    d = d[np.isfinite(d)]
    n = len(d)
    if lag is None:
        lag = max(1, int(np.floor(n ** (1 / 3))))
    d0 = d - d.mean()
    gamma0 = np.dot(d0, d0) / n
    var = gamma0
    for k in range(1, lag + 1):
        w = 1.0 - k / (lag + 1)               # Bartlett weight
        cov = np.dot(d0[k:], d0[:-k]) / n
        var += 2.0 * w * cov
    return float(var)


def diebold_mariano(
    loss_a: NDArray, loss_b: NDArray, lag: int | None = None
) -> DMResult:
    """Diebold-Mariano test of equal predictive accuracy between models A and B.

    loss_a, loss_b are per-period loss series (same loss function, e.g. QLIKE).
    H0: equal expected loss. A significant negative statistic => A has lower loss
    (A is better); positive => B better. Uses a HAC variance for the differential.
    """
    la = np.asarray(loss_a, dtype=float)
    lb = np.asarray(loss_b, dtype=float)
    m = np.isfinite(la) & np.isfinite(lb)
    d = la[m] - lb[m]
    n = len(d)
    dbar = float(d.mean())
    lrv = _hac_variance(d, lag=lag)
    if lrv <= 0 or n < 2:
        return DMResult(0.0, 1.0, "neither", dbar)
    stat = dbar / np.sqrt(lrv / n)
    pval = float(2 * (1 - norm.cdf(abs(stat))))
    if pval < 0.05:
        better = "A" if dbar < 0 else "B"
    else:
        better = "neither"
    return DMResult(float(stat), pval, better, dbar)


def per_period_qlike(sigma2_hat: NDArray, r2: NDArray) -> NDArray:
    """Per-period QLIKE series (for feeding into the DM test)."""
    a = np.asarray(sigma2_hat, dtype=float)
    b = np.asarray(r2, dtype=float)
    out = np.full(len(a), np.nan)
    m = np.isfinite(a) & np.isfinite(b) & (a > 0)
    out[m] = b[m] / a[m] + np.log(a[m])
    return out


def per_period_mse(sigma2_hat: NDArray, r2: NDArray) -> NDArray:
    a = np.asarray(sigma2_hat, dtype=float)
    b = np.asarray(r2, dtype=float)
    out = np.full(len(a), np.nan)
    m = np.isfinite(a) & np.isfinite(b)
    out[m] = (a[m] - b[m]) ** 2
    return out
