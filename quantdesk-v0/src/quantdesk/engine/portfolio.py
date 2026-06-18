"""Regime-scaled stat-arb portfolio engine — the integration point.

Wires three frozen packages into a single causal pipeline:

    prices
      → statarb.screen_universe          (cluster + FDR cointegration screen)
      → statarb.is_tradeable             (filter: half-life, capital controls)
      → statarb.backtest_pair            (walk-forward OOS, t+1 execution)
      → volforecast.fit_garch            (market vol forecast on equal-wt return)
      → volforecast.vol_percentile_regime (regime exposure_scalar)
      → book_return[t] = scalar[t-1] * mean(pair_returns[t])   ← LAGGED

The one-step lag on the regime scalar is the no-look-ahead invariant: the
de-grossing decision at time t is made from information available at t-1.
A unit test asserts this directly (see quantdesk/tests/test_engine.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.typing import NDArray

# --- frozen upstream imports (do not copy or reimplement) ---
from statarb.cointegration import engle_granger, is_tradeable, screen_universe
from statarb.backtest import CostModel, BacktestResult, backtest_pair
from statarb.signals import SignalConfig

from volforecast.models import fit_garch
from volforecast.models.benchmarks import ewma_variance
from volforecast.regime import vol_percentile_regime, detect_structural_break_cusum

from quantdesk.config import QuantDeskConfig


# --------------------------------------------------------------------------- #
#  Result types
# --------------------------------------------------------------------------- #

@dataclass
class PairContribution:
    a: str
    b: str
    oos_sharpe: float
    max_drawdown: float
    calmar: float
    cusum_break_idx: int | None   # index into the full prices DataFrame, or None


@dataclass
class PortfolioResult:
    oos_equity: pd.Series          # cumulative equity (starts at 1.0)
    oos_returns: pd.Series         # daily book returns, regime-scaled
    unscaled_returns: pd.Series    # same returns WITHOUT regime scaling (for lag test)
    combined_sharpe: float         # OOS Sharpe of the regime-scaled book
    unscaled_sharpe: float         # OOS Sharpe without regime overlay (honest comparison)
    max_drawdown: float
    calmar: float
    regime_scalar: pd.Series       # exposure_scalar path (1.0 = full, <1 = de-grossed)
    pair_contributions: list[PairContribution]
    cusum_break_flags: dict[str, int]  # "A/B" -> break index in prices
    pct_days_degrossed: float          # fraction of OOS days where scalar < 1
    n_pairs: int                       # tradeable pairs surviving the screen


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

def _sharpe(returns: NDArray, ann: float) -> float:
    r = returns[np.isfinite(returns)]
    if r.size < 2 or r.std(ddof=1) == 0:
        return 0.0
    return float(np.sqrt(ann) * r.mean() / r.std(ddof=1))


def _max_drawdown(equity: NDArray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / np.where(peak > 0, peak, 1.0)
    return float(dd.min())


def _build_signal_config(cfg: QuantDeskConfig) -> SignalConfig:
    return SignalConfig(
        entry=cfg.signal.entry,
        exit=cfg.signal.exit,
        stop=cfg.signal.stop,
        max_holding=cfg.signal.max_holding,
    )


def _build_cost_model(cfg: QuantDeskConfig) -> CostModel:
    return CostModel(
        commission_bps=cfg.cost.commission_bps,
        half_spread_bps=cfg.cost.half_spread_bps,
        impact_bps=cfg.cost.impact_bps,
    )


def _market_sigma_daily(prices: pd.DataFrame) -> NDArray:
    """GARCH(1,1) daily vol forecast on the equal-weight market return.

    Returns an array aligned to prices (length n). The forecast at index t is
    computed from returns through t-1, so it is fully causal when used as a
    regime signal at time t.
    """
    rets_all = prices.pct_change().mean(axis=1).values  # length n, NaN at 0
    rets_clean = rets_all[~np.isnan(rets_all)]           # length n-1

    try:
        fit = fit_garch(rets_clean)
        sigma_fit = np.sqrt(np.maximum(fit.conditional_var, 1e-12))
    except Exception:
        # Fall back to EWMA if GARCH optimiser fails (e.g. degenerate data)
        var_fit = ewma_variance(rets_clean, lam=0.94)
        sigma_fit = np.sqrt(np.maximum(var_fit, 1e-12))

    # Prepend one entry so the array aligns with the full prices index.
    # full_sigma[t] = sigma_fit[t-1]: the vol estimate available at time t,
    # based on returns through t-1.
    full_sigma = np.empty(len(prices))
    full_sigma[0] = sigma_fit[0]
    full_sigma[1:] = sigma_fit
    return full_sigma


# --------------------------------------------------------------------------- #
#  Main engine
# --------------------------------------------------------------------------- #

def run_regime_scaled_statarb(
    prices: pd.DataFrame,
    cfg: QuantDeskConfig,
) -> PortfolioResult:
    """Run the integrated regime-scaled stat-arb pipeline.

    Parameters
    ----------
    prices:
        Aligned daily price DataFrame (rows = dates, cols = tickers). Values
        must be positive (raw levels, not log). Should already be point-in-time
        via PriceData.as_of() before being passed in.
    cfg:
        Validated QuantDeskConfig.

    Returns
    -------
    PortfolioResult with OOS equity, Sharpe, drawdown, regime scalar path,
    per-pair contributions, and CUSUM break flags.
    """
    n = len(prices)
    log_prices = np.log(prices)

    # ------------------------------------------------------------------ #
    # Step 1 — Cointegration screen: cluster → FDR-corrected pair list
    # ------------------------------------------------------------------ #
    screen = screen_universe(
        log_prices,
        n_clusters=cfg.n_clusters,
        fdr=cfg.fdr,
    )

    signal_cfg = _build_signal_config(cfg)
    cost_model = _build_cost_model(cfg)

    # ------------------------------------------------------------------ #
    # Step 2 — Filter to tradeable pairs and run walk-forward backtests
    # ------------------------------------------------------------------ #
    pair_results: list[tuple[object, BacktestResult]] = []

    for cand in screen.candidates:
        y = log_prices[cand.a].values
        x = log_prices[cand.b].values
        eg = engle_granger(y, x, y_name=cand.a, x_name=cand.b)
        tradeable, _ = is_tradeable(eg)
        if not tradeable:
            continue

        bt = backtest_pair(
            y, x,
            train_size=cfg.train_size,
            test_size=cfg.test_size,
            cfg=signal_cfg,
            cost_model=cost_model,
            ann_factor=cfg.ann_factor,
        )
        pair_results.append((cand, bt))

    # ------------------------------------------------------------------ #
    # Step 3 — Market vol forecast → regime exposure_scalar
    # ------------------------------------------------------------------ #
    full_sigma = _market_sigma_daily(prices)

    regime = vol_percentile_regime(
        full_sigma,
        lookback=cfg.regime_lookback,
        stress_pct=cfg.regime_stress_pct,
        min_scalar=cfg.regime_min_scalar,
    )
    scalar = regime.exposure_scalar   # length n; scalar[t] uses data through t

    # ------------------------------------------------------------------ #
    # Handle the no-surviving-pairs edge case
    # ------------------------------------------------------------------ #
    if not pair_results:
        zeros = pd.Series(np.zeros(n), index=prices.index)
        ones = pd.Series(np.ones(n), index=prices.index)
        return PortfolioResult(
            oos_equity=ones.copy(),
            oos_returns=zeros.copy(),
            unscaled_returns=zeros.copy(),
            combined_sharpe=0.0,
            unscaled_sharpe=0.0,
            max_drawdown=0.0,
            calmar=0.0,
            regime_scalar=pd.Series(scalar, index=prices.index),
            pair_contributions=[],
            cusum_break_flags={},
            pct_days_degrossed=float(np.mean(scalar < 1.0)),
            n_pairs=0,
        )

    # ------------------------------------------------------------------ #
    # Step 4 — Combine pair OOS return streams (equal weight)
    #
    # oos_returns from each BacktestResult is NaN during the in-sample
    # train window and any fold gaps; nanmean ignores those days so the
    # book return only reflects days where at least one pair is live.
    # ------------------------------------------------------------------ #
    returns_matrix = np.array([bt.oos_returns for _, bt in pair_results])
    # shape: (n_pairs, n_days)

    has_data = np.isfinite(returns_matrix)           # (n_pairs, n_days)
    any_live = has_data.any(axis=0)                  # (n_days,)

    raw_sum = np.where(has_data, returns_matrix, 0.0).sum(axis=0)
    count = has_data.sum(axis=0)
    mean_pair_ret = np.where(any_live, raw_sum / np.maximum(count, 1), 0.0)

    # ------------------------------------------------------------------ #
    # Step 5 — Apply LAGGED regime scalar
    #
    # scalar[t-1] is the de-grossing weight decided from data available up
    # to t-1; it is applied to the return realised from t-1 to t.
    # This preserves the no-look-ahead property already proven in statarb.
    # ------------------------------------------------------------------ #
    lagged_scalar = np.empty(n)
    lagged_scalar[0] = scalar[0]          # no prior bar — use same-day scalar
    lagged_scalar[1:] = scalar[:-1]       # lagged_scalar[t] = scalar[t-1]

    book_returns = lagged_scalar * mean_pair_ret

    # ------------------------------------------------------------------ #
    # Step 6 — Portfolio metrics
    # ------------------------------------------------------------------ #
    equity = np.cumprod(1.0 + book_returns)

    sharpe = _sharpe(book_returns, cfg.ann_factor)
    unscaled_sharpe = _sharpe(mean_pair_ret, cfg.ann_factor)
    mdd = _max_drawdown(equity)
    total_ret = float(equity[-1] - 1.0)
    calmar = total_ret / abs(mdd) if mdd < 0 else 0.0
    pct_degrossed = float(np.mean(scalar < 1.0))

    # ------------------------------------------------------------------ #
    # Step 7 — CUSUM break detection on each pair's live spread
    # ------------------------------------------------------------------ #
    cusum_breaks: dict[str, int] = {}
    contributions: list[PairContribution] = []

    for cand, bt in pair_results:
        y = log_prices[cand.a].values
        x = log_prices[cand.b].values
        eg = engle_granger(y, x, y_name=cand.a, x_name=cand.b)
        spread = y - eg.hedge_ratio * x

        # Run CUSUM only on the OOS portion (after the first train window).
        oos_start = cfg.train_size
        oos_spread = spread[oos_start:]
        break_in_oos = detect_structural_break_cusum(oos_spread)

        label = f"{cand.a}/{cand.b}"
        cusum_idx = (oos_start + break_in_oos) if break_in_oos >= 0 else None
        if cusum_idx is not None:
            cusum_breaks[label] = cusum_idx

        contributions.append(PairContribution(
            a=cand.a,
            b=cand.b,
            oos_sharpe=bt.oos_sharpe,
            max_drawdown=bt.max_drawdown,
            calmar=bt.calmar,
            cusum_break_idx=cusum_idx,
        ))

    return PortfolioResult(
        oos_equity=pd.Series(equity, index=prices.index),
        oos_returns=pd.Series(book_returns, index=prices.index),
        unscaled_returns=pd.Series(mean_pair_ret, index=prices.index),
        combined_sharpe=sharpe,
        unscaled_sharpe=unscaled_sharpe,
        max_drawdown=mdd,
        calmar=calmar,
        regime_scalar=pd.Series(scalar, index=prices.index),
        pair_contributions=contributions,
        cusum_break_flags=cusum_breaks,
        pct_days_degrossed=pct_degrossed,
        n_pairs=len(pair_results),
    )
