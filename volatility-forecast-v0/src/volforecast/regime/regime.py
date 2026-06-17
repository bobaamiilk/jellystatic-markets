"""Regime detection (Stage 5) — robust proxies over fragile models.

Two distinct phenomena, often confused:
  - REGIME SWITCH: recurring (calm <-> crisis <-> calm). The process revisits
    states; a crisis state has a CORRELATION signature (cross-asset corr -> 1),
    not just high vol.
  - STRUCTURAL BREAK: permanent, one-off (Stock Connect repricing the A-H
    relationship). A reversion-expecting model waits forever.

Markov-switching GARCH can model recurring regimes, but it's numerically fragile
(path explosion, label switching, local optima). Many desks therefore prefer a
SIMPLE, ROBUST proxy that's roughly right over a sophisticated model that's
fragile: a vol-percentile / realized-correlation signal. We implement that proxy
and a lightweight 2-state classifier; the philosophy is to SIZE to the regime,
not to bet on timing the turn (detection is always lagged).

This module also produces the regime scalar consumed by the stat-arb book
(Project 2) for de-grossing — the cross-system integration point.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

Regime = Literal["calm", "stressed"]


@dataclass
class RegimeResult:
    state: NDArray          # 0 = calm, 1 = stressed (per time step)
    vol_percentile: NDArray
    exposure_scalar: NDArray  # in [min_scalar, 1]: multiply book gross by this
    frac_stressed: float


def vol_percentile_regime(
    sigma_hat_daily: NDArray,
    lookback: int = 252,
    stress_pct: float = 0.80,
    min_scalar: float = 0.25,
) -> RegimeResult:
    """Classify regime by the trailing percentile of forecast vol.

    A day is 'stressed' if current forecast vol sits above its `stress_pct`
    trailing percentile. The exposure scalar shrinks smoothly from 1 (calm) to
    `min_scalar` (extreme) so the response is graded, not a hard on/off switch —
    you size to the regime rather than trying to time the exact turn.

    Causal: percentile uses only the trailing `lookback` window.
    """
    s = np.asarray(sigma_hat_daily, dtype=float)
    n = len(s)
    pct = np.full(n, np.nan)
    state = np.zeros(n, dtype=int)
    scalar = np.ones(n)

    for t in range(n):
        lo = max(0, t - lookback + 1)
        window = s[lo:t + 1]
        if len(window) < 20:
            continue
        rank = float(np.mean(window <= s[t]))   # percentile of today's vol
        pct[t] = rank
        if rank >= stress_pct:
            state[t] = 1
            # Linear de-gross from 1 at the stress threshold to min_scalar at 100th pct.
            frac = (rank - stress_pct) / max(1e-9, (1.0 - stress_pct))
            scalar[t] = 1.0 - (1.0 - min_scalar) * frac
    return RegimeResult(state, pct, scalar, float(np.mean(state)))


def realized_correlation(returns: pd.DataFrame, window: int = 21) -> NDArray:
    """Average pairwise realized correlation over a trailing window — the crisis
    SIGNATURE. In APAC risk-off, cross-asset correlation spikes toward 1 and
    diversification evaporates exactly when it's needed. Returns a series aligned
    to window end-points (NaN before the first full window).
    """
    rets = returns.pct_change().dropna() if (returns.values > 0).all() else returns
    arr = rets.values
    n = len(arr)
    out = np.full(n, np.nan)
    for t in range(window - 1, n):
        seg = arr[t - window + 1:t + 1]
        c = np.corrcoef(seg, rowvar=False)
        iu = np.triu_indices_from(c, k=1)
        out[t] = float(np.nanmean(c[iu]))
    return out


def detect_structural_break_cusum(spread: NDArray, threshold: float = 1.0) -> int:
    """CUSUM break detector for an unknown break date. Returns the index of the
    first detected break, or -1 if none.

    Cumulative sum of standardised deviations from the in-sample mean; if it
    exceeds `threshold * sqrt(n) * sigma` the relationship has likely shifted
    permanently (vs a recurring regime, which CUSUM would not flag persistently).
    Detection is lagged by construction — hence the pre-committed hard stops in
    Project 2 rather than reliance on the detector alone.
    """
    s = np.asarray(spread, dtype=float)
    n = len(s)
    mu, sigma = s.mean(), s.std(ddof=1)
    if sigma == 0:
        return -1
    cusum = np.cumsum(s - mu)
    bound = threshold * np.sqrt(n) * sigma
    hits = np.where(np.abs(cusum) > bound)[0]
    return int(hits[0]) if len(hits) else -1
