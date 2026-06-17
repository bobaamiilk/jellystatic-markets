"""Volatility targeting — turning a forecast into position size (Stage 4).

Scale exposure inversely to forecast vol so the book holds roughly CONSTANT risk:

    w_t = sigma_target / sigma_hat_t

If the forecast were perfect, realised vol would equal the target exactly:

    sigma_realised = w_t * sigma_t = sigma_target * (sigma_t / sigma_hat_t)

so forecast error feeds straight into tracking error — which is precisely WHY the
Stage-3 evaluation matters here.

The core limitation (and why caps/floors are non-negotiable):
  - vol targeting de-levers AFTER vol rises, so it protects against PERSISTENCE
    (forecastable) but not the ONSET JUMP (a BOJ surprise hits at full size);
  - it sizes UP most in calm regimes — maximally levered exactly when a jump is
    most damaging.
Hence: cap leverage, floor the vol estimate, and band the signal to limit
turnover. A separate jump/tail overlay (long OTM puts — the skew from Project 1)
handles the onset risk vol targeting cannot.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class VolTargetConfig:
    target_vol: float = 0.10        # annualised target (e.g. 10%)
    max_leverage: float = 3.0       # cap — never size up without bound
    vol_floor: float = 0.03         # floor on sigma_hat — avoids blow-up in calm
    ann_factor: float = 252.0
    rebalance_band: float = 0.10    # only re-trade if weight moves >10% (turnover)


@dataclass
class VolTargetResult:
    weights: NDArray
    realised_returns: NDArray
    realised_vol: float
    gross_turnover: float
    leverage_capped_frac: float     # fraction of days the cap bound


def vol_target_weights(sigma_hat_daily: NDArray, cfg: VolTargetConfig | None = None) -> NDArray:
    """Compute capped, floored, band-rebalanced target weights from a daily vol
    forecast. sigma_hat_daily is DAILY vol (sqrt of variance forecast).
    """
    cfg = cfg or VolTargetConfig()
    sig_ann = np.asarray(sigma_hat_daily, dtype=float) * np.sqrt(cfg.ann_factor)
    sig_ann = np.maximum(sig_ann, cfg.vol_floor)         # floor
    raw = np.clip(cfg.target_vol / sig_ann, 0.0, cfg.max_leverage)  # cap

    # Band rebalancing: hold the weight unless it has drifted more than the band.
    w = np.empty_like(raw)
    w[0] = raw[0]
    for t in range(1, len(raw)):
        if abs(raw[t] - w[t - 1]) > cfg.rebalance_band * max(w[t - 1], 1e-6):
            w[t] = raw[t]
        else:
            w[t] = w[t - 1]
    return w


def apply_vol_target(
    returns: NDArray, sigma_hat_daily: NDArray, cfg: VolTargetConfig | None = None
) -> VolTargetResult:
    """Apply vol targeting to an asset's return stream with t+1 sizing.

    Weight decided from the forecast available at t is applied to the return from
    t to t+1 (causal). Reports realised annualised vol (should sit near target if
    the forecast is decent) and turnover.
    """
    cfg = cfg or VolTargetConfig()
    r = np.asarray(returns, dtype=float)
    w = vol_target_weights(sigma_hat_daily, cfg)

    # t+1 application: weight_{t-1} scales return_t.
    realised = w[:-1] * r[1:]
    realised_vol = float(np.std(realised, ddof=1) * np.sqrt(cfg.ann_factor))
    turnover = float(np.sum(np.abs(np.diff(w))))
    capped_frac = float(np.mean(np.isclose(w, cfg.max_leverage)))
    return VolTargetResult(w, realised, realised_vol, turnover, capped_frac)
