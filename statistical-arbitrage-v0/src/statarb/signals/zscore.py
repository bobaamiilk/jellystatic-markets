"""Spread z-scores, trading signals, and break monitoring (Stages 2 & 5).

The spread is normalised to a z-score; we trade extremes and bet on reversion:

    z_t = (e_t - mu_e) / sigma_e
    enter short spread when z_t > +entry, long when z_t < -entry
    exit near z_t ~ 0, hard stop at |z_t| > stop

Two failure modes the curriculum stresses, handled here:

  1. The z-score quietly assumes stable, near-normal residuals. Real spreads have
     volatility clustering and fat tails, so a fixed +/-2 threshold means
     different things in different regimes. -> volatility-adjusted z-score using
     an EWMA spread variance.

  2. You cannot tell slow mean-reversion from a structural break in real time;
     detection is always lagged. -> a hard stop-loss and a max-holding-period are
     pre-committed, plus a rolling-ADF health check to flag a decaying spread.

All statistics are computed causally (trailing windows / EWMA) so signals carry
no look-ahead.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray

from statarb.cointegration.engle_granger import adf_test


class Position(IntEnum):
    FLAT = 0
    LONG = 1     # long the spread (long y, short beta*x): bet z rises back to 0
    SHORT = -1   # short the spread: bet z falls back to 0


def rolling_zscore(spread: NDArray, window: int = 60) -> NDArray:
    """Causal rolling z-score. Uses the trailing `window` observations ending at
    t (inclusive) so z_t never sees the future. First window-1 entries are NaN.
    """
    spread = np.asarray(spread, dtype=float)
    n = len(spread)
    z = np.full(n, np.nan)
    for t in range(window - 1, n):
        w = spread[t - window + 1:t + 1]
        sd = w.std(ddof=1)
        if sd > 0:
            z[t] = (spread[t] - w.mean()) / sd
    return z


def vol_adjusted_zscore(spread: NDArray, window: int = 60, lam: float = 0.94) -> NDArray:
    """z-score using an EWMA variance for the denominator.

    The mean is a trailing window mean; the dispersion is an EWMA (RiskMetrics
    lambda) so the threshold adapts to the spread's *current* volatility regime
    rather than assuming homoskedastic, normal residuals. This is the practical
    fix for "z=2 is not a fixed probability under fat tails".
    """
    spread = np.asarray(spread, dtype=float)
    n = len(spread)
    z = np.full(n, np.nan)
    ewma_var = np.nan
    for t in range(n):
        if t == 0:
            ewma_var = spread[0] ** 2
            continue
        ewma_var = lam * ewma_var + (1.0 - lam) * spread[t - 1] ** 2
        if t >= window - 1:
            w = spread[t - window + 1:t + 1]
            sd = np.sqrt(ewma_var)
            if sd > 0:
                z[t] = (spread[t] - w.mean()) / sd
    return z


@dataclass
class SignalConfig:
    entry: float = 2.0
    exit: float = 0.5
    stop: float = 4.0
    max_holding: int = 30      # bars; pre-committed hard exit (Stage 5)


def generate_signals(zscore: NDArray, cfg: SignalConfig | None = None) -> NDArray:
    """Turn a z-score path into a position path with entry/exit/stop/time rules.

    Returns an int array of Position values. The logic is a deterministic state
    machine; crucially the signal at t depends only on z up to t, so when the
    backtest executes it at t+1 there is no look-ahead.

    Risk rules are pre-committed (not discretionary) because in real time you
    can't distinguish a slow reverter from a broken pair:
      - hard stop at |z| > stop  (the pair may have structurally broken)
      - max_holding bars         (capital shouldn't sit in a non-reverting trade)
    """
    cfg = cfg or SignalConfig()
    z = np.asarray(zscore, dtype=float)
    n = len(z)
    pos = np.zeros(n, dtype=int)
    state = Position.FLAT
    bars_held = 0
    cooldown = False   # set after a stop-loss: don't re-enter a still-extreme spread

    for t in range(n):
        zt = z[t]
        if np.isnan(zt):
            pos[t] = int(state)
            continue

        if state == Position.FLAT:
            # After a stop-loss, wait until the spread normalises (|z| < exit)
            # before allowing a new entry — re-entering a still-blown-out spread
            # is the classic way to keep paying into a structurally broken pair.
            if cooldown:
                if abs(zt) < cfg.exit:
                    cooldown = False
            if not cooldown:
                if zt > cfg.entry:
                    state, bars_held = Position.SHORT, 0
                elif zt < -cfg.entry:
                    state, bars_held = Position.LONG, 0
        else:
            bars_held += 1
            hit_stop = abs(zt) > cfg.stop
            timed_out = bars_held >= cfg.max_holding
            reverted = abs(zt) < cfg.exit
            if hit_stop:
                state, bars_held, cooldown = Position.FLAT, 0, True
            elif timed_out or reverted:
                state, bars_held = Position.FLAT, 0
        pos[t] = int(state)

    return pos


def rolling_adf_health(spread: NDArray, window: int = 120, step: int = 5) -> NDArray:
    """Rolling ADF p-value on the spread — a live break-detection health metric.

    A spread that was cointegrated can decohere (Stock Connect repricing the A-H
    relationship; an index reconstitution). Rising ADF p-values over time warn
    that stationarity is weakening *before* the z-score stop fires. Returns an
    array aligned to the window end-points (NaN elsewhere); detection is
    inherently lagged by ~half the window — which is exactly why we ALSO keep a
    hard z-score stop.
    """
    spread = np.asarray(spread, dtype=float)
    n = len(spread)
    out = np.full(n, np.nan)
    for end in range(window, n + 1, step):
        seg = spread[end - window:end]
        try:
            _, pval = adf_test(seg, regression="c")
            out[end - 1] = pval
        except Exception:
            out[end - 1] = np.nan
    return out
