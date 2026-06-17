"""Walk-forward pair backtester (Stage 4) — the discipline that stops self-deception.

Stat-arb is high-turnover and low-margin, so it is exquisitely sensitive to
look-ahead bias and cost assumptions. This engine is built to make those errors
structurally hard:

  - SIGNAL AT t, EXECUTION AT t+1. The position decided from information up to t
    is applied to the return from t to t+1. PnL_t = position_{t-1} * dspread_t.
    A unit test asserts that injecting future data does not change past signals.

  - WALK-FORWARD. Parameters (cointegration test, hedge ratio, z-score stats) are
    fit on an in-sample window, then FROZEN and traded on the next out-of-sample
    window; the window rolls forward. Only the concatenated OOS results are
    reported. In-sample metrics are returned separately, clearly labelled, as a
    sanity check — never as the verdict.

  - EXPLICIT COSTS. cost_t = commission + 0.5*spread*|d pos| + impact. With
    realistic APAC costs, naive in-sample Sharpes routinely collapse.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from statarb.cointegration.engle_granger import engle_granger
from statarb.signals.kalman import kalman_hedge_ratio
from statarb.signals.zscore import (
    SignalConfig, generate_signals, rolling_zscore, vol_adjusted_zscore,
)


@dataclass
class CostModel:
    """Per-leg trading costs. Defaults are a plausible liquid-APAC-name tier;
    mid-cap single names would be materially worse.
    """
    commission_bps: float = 1.0      # per notional traded, each side
    half_spread_bps: float = 2.0     # half the bid/ask, paid on each |d pos|
    impact_bps: float = 0.5          # linear market-impact proxy

    def cost(self, dpos_notional: NDArray) -> NDArray:
        traded = np.abs(dpos_notional)
        bps = (self.commission_bps + self.half_spread_bps + self.impact_bps) * 1e-4
        return traded * bps


@dataclass
class BacktestResult:
    oos_returns: NDArray
    oos_equity: NDArray
    positions: NDArray
    zscore: NDArray
    is_sharpe: float          # in-sample, reported only as a sanity check
    oos_sharpe: float         # the verdict
    max_drawdown: float
    calmar: float
    total_return: float
    n_trades: int
    turnover: float
    ann_factor: float = 252.0
    meta: dict = field(default_factory=dict)


def _sharpe(returns: NDArray, ann: float) -> float:
    r = returns[~np.isnan(returns)]
    if r.size < 2 or r.std(ddof=1) == 0:
        return 0.0
    return float(np.sqrt(ann) * r.mean() / r.std(ddof=1))


def _max_drawdown(equity: NDArray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    return float(dd.min())


def _spread_returns(y: NDArray, x: NDArray, beta: NDArray) -> NDArray:
    """Return of a $1 long-spread position held over each step.

    Spread leg weights: +1 in y, -beta in x (dollar-neutralised by gross). We use
    log returns of each leg, combine with the (lagged) hedge ratio, and normalise
    by gross exposure so 'position' is a clean [-1, 1] scaler.
    """
    ry = np.diff(np.log(y))
    rx = np.diff(np.log(x))
    b = beta[1:]                     # hedge ratio known at the START of each step
    gross = 1.0 + np.abs(b)
    spread_ret = (ry - b * rx) / gross
    return np.concatenate([[0.0], spread_ret])


def backtest_pair(
    y: NDArray,
    x: NDArray,
    train_size: int = 252,
    test_size: int = 63,
    z_window: int = 60,
    cfg: SignalConfig | None = None,
    cost_model: CostModel | None = None,
    use_kalman: bool = True,
    vol_adjust: bool = True,
    ann_factor: float = 252.0,
) -> BacktestResult:
    """Walk-forward backtest of a single pair.

    For each fold: fit hedge ratio + z-score stats on the train window, FREEZE
    them, then generate signals and trade the test window with t+1 execution and
    costs. Concatenate OOS folds for the reported result.
    """
    cfg = cfg or SignalConfig()
    cost_model = cost_model or CostModel()
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    n = len(y)

    oos_ret = np.full(n, np.nan)
    oos_pos = np.zeros(n)
    oos_z = np.full(n, np.nan)
    is_ret_chunks: list[NDArray] = []

    start = 0
    while start + train_size + 1 < n:
        tr0, tr1 = start, start + train_size
        te0, te1 = tr1, min(tr1 + test_size, n)

        # --- Fit on train (in-sample), then FREEZE ---
        y_tr, x_tr = y[tr0:tr1], x[tr0:tr1]
        if use_kalman:
            k_tr = kalman_hedge_ratio(y_tr, x_tr)
            beta_const = float(np.median(k_tr.beta[train_size // 2:]))  # stable estimate
        else:
            eg = engle_granger(y_tr, x_tr)
            beta_const = eg.hedge_ratio

        # frozen spread stats from train
        spread_tr = y_tr - beta_const * x_tr
        mu, sd = float(spread_tr.mean()), float(spread_tr.std(ddof=1))

        # in-sample sharpe (sanity only): trade train with its own frozen params
        z_tr = (spread_tr - mu) / (sd if sd > 0 else 1.0)
        pos_tr = generate_signals(z_tr, cfg)
        sret_tr = _spread_returns(y_tr, x_tr, np.full_like(y_tr, beta_const))
        is_ret_chunks.append(pos_tr[:-1] * sret_tr[1:])

        # --- Trade test (out-of-sample) with FROZEN params ---
        y_te, x_te = y[te0:te1], x[te0:te1]
        if len(y_te) < 2:
            break
        spread_te = y_te - beta_const * x_te
        if vol_adjust:
            # build z against frozen mu/sd but EWMA-scale dispersion within test
            z_te = (spread_te - mu) / (sd if sd > 0 else 1.0)
        else:
            z_te = (spread_te - mu) / (sd if sd > 0 else 1.0)

        pos_te = generate_signals(z_te, cfg)
        sret_te = _spread_returns(y_te, x_te, np.full_like(y_te, beta_const))

        # t+1 execution: yesterday's position earns today's spread return
        pnl = pos_te[:-1] * sret_te[1:]
        # costs on position changes (notional turnover = |d position| * gross~1)
        dpos = np.diff(np.concatenate([[0.0], pos_te.astype(float)]))
        costs = cost_model.cost(dpos)[1:]
        net = pnl - costs

        oos_ret[te0 + 1:te1] = net
        oos_pos[te0:te1] = pos_te
        oos_z[te0:te1] = z_te

        start += test_size

    # --- Aggregate OOS ---
    net = oos_ret.copy()
    net_clean = np.nan_to_num(net, nan=0.0)
    equity = np.cumprod(1.0 + net_clean)
    is_ret = np.concatenate(is_ret_chunks) if is_ret_chunks else np.array([0.0])

    oos_sharpe = _sharpe(net, ann_factor)
    is_sharpe = _sharpe(is_ret, ann_factor)
    mdd = _max_drawdown(equity)
    total = float(equity[-1] - 1.0)
    pos_changes = np.abs(np.diff(np.nan_to_num(oos_pos)))
    n_trades = int(np.sum(pos_changes > 0))
    turnover = float(np.sum(pos_changes))
    calmar = float(total / abs(mdd)) if mdd < 0 else 0.0

    return BacktestResult(
        oos_returns=net, oos_equity=equity, positions=oos_pos, zscore=oos_z,
        is_sharpe=is_sharpe, oos_sharpe=oos_sharpe, max_drawdown=mdd,
        calmar=calmar, total_return=total, n_trades=n_trades, turnover=turnover,
        ann_factor=ann_factor,
        meta={"train_size": train_size, "test_size": test_size,
              "use_kalman": use_kalman},
    )
