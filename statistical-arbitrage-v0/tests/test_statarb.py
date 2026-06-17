"""Tests for the statistical-arbitrage engine.

Encodes the curriculum's validation requirements:
  - EG detects a known cointegrated pair and rejects an independent one;
  - the recovered hedge ratio and half-life match ground truth;
  - the Kalman filter tracks a time-varying beta;
  - z-score signals respect entry/exit/stop and carry NO look-ahead;
  - the multi-asset screen recovers within-cluster pairs and FDR cuts noise;
  - the walk-forward backtester executes at t+1 (a future-data injection test).
"""
import numpy as np
import pytest

from statarb.cointegration import (
    engle_granger, is_tradeable, half_life_of_mean_reversion,
    screen_universe, johansen_rank, benjamini_hochberg,
)
from statarb.signals import (
    kalman_hedge_ratio, rolling_ols_hedge_ratio, rolling_zscore,
    generate_signals, SignalConfig, Position,
)
from statarb.backtest import backtest_pair, CostModel
from statarb.utils import (
    make_cointegrated_pair, make_independent_pair, make_universe,
)


# ---------------- Stage 1: cointegration ----------------

def test_detects_cointegrated_pair():
    y, x = make_cointegrated_pair(n=750, beta=1.5, half_life=15.0, seed=0)
    res = engle_granger(y.values, x.values)
    assert res.is_cointegrated
    assert res.pvalue < 0.05


def test_rejects_independent_pair():
    a, b = make_independent_pair(n=750, seed=1)
    res = engle_granger(a.values, b.values)
    # Two independent random walks should not test as cointegrated.
    assert not res.is_cointegrated


def test_recovers_hedge_ratio():
    # ln(y) - beta ln(x) is stationary, so EG on log-prices recovers beta.
    y, x = make_cointegrated_pair(n=1500, beta=1.5, half_life=15.0,
                                  spread_vol=0.25, common_vol=0.02, seed=3)
    res = engle_granger(np.log(y.values), np.log(x.values))
    assert abs(res.hedge_ratio - 1.5) < 0.15


def test_half_life_recovered_in_ballpark():
    y, x = make_cointegrated_pair(n=1500, beta=1.0, half_life=20.0,
                                  spread_vol=0.6, seed=5)
    res = engle_granger(np.log(y.values), np.log(x.values))
    # Half-life estimation is noisy; require the right order of magnitude.
    assert 8.0 < res.half_life < 45.0


def test_half_life_of_random_walk_is_not_tradeable():
    # A random walk has no real mean reversion; finite samples can show weak
    # spurious reversion, so the economically correct check is "far too slow to
    # trade" (or infinite), not literally infinite.
    rng = np.random.default_rng(0)
    rw = np.cumsum(rng.normal(0, 1, 500))
    hl = half_life_of_mean_reversion(rw)
    assert (not np.isfinite(hl)) or hl > 60.0


def test_tradeability_filters_capital_controls():
    y, x = make_cointegrated_pair(n=750, seed=0)
    res = engle_granger(np.log(y.values), np.log(x.values))
    ok, reason = is_tradeable(res, capital_controls=True)
    assert not ok and "capital controls" in reason


# ---------------- Stage 2: signals ----------------

def test_kalman_tracks_time_varying_beta():
    # Build x as a random walk and y = beta_t * x with beta drifting 1.0 -> 2.0.
    rng = np.random.default_rng(7)
    n = 600
    x = np.cumsum(rng.normal(0, 0.5, n)) + 50
    beta_true = np.linspace(1.0, 2.0, n)
    y = beta_true * x + rng.normal(0, 0.5, n)
    res = kalman_hedge_ratio(y, x, delta=1e-3, with_intercept=False)
    # Late-sample beta should be closer to 2.0 than to 1.0.
    late = res.beta[-50:].mean()
    assert abs(late - 2.0) < abs(late - 1.0)


def test_rolling_ols_is_causal():
    # First window-1 entries must be NaN (no estimate without enough history).
    y, x = make_cointegrated_pair(n=300, seed=0)
    beta = rolling_ols_hedge_ratio(y.values, x.values, window=60)
    assert np.all(np.isnan(beta[:59]))
    assert np.isfinite(beta[60])


def test_signals_respect_entry_and_exit():
    # Hand-built z path: cross +entry then revert toward 0.
    z = np.array([0, 0, 2.5, 2.0, 1.0, 0.3, 0.0])
    cfg = SignalConfig(entry=2.0, exit=0.5, stop=4.0, max_holding=30)
    pos = generate_signals(z, cfg)
    assert pos[2] == int(Position.SHORT)   # entered short when z > 2
    assert pos[-1] == int(Position.FLAT)   # exited when |z| < 0.5


def test_signals_hit_stop_loss():
    z = np.array([0, 2.5, 3.0, 4.5, 5.0])   # blows through the stop
    cfg = SignalConfig(entry=2.0, exit=0.5, stop=4.0, max_holding=30)
    pos = generate_signals(z, cfg)
    assert pos[3] == int(Position.FLAT)    # stop fires when |z| > 4
    # Cooldown: do NOT re-enter at index 4 even though z=5 > entry, because the
    # spread is still blown out (a likely structural break).
    assert pos[4] == int(Position.FLAT)


def test_signals_no_lookahead():
    # Signal at time t must not change when FUTURE z values are altered.
    rng = np.random.default_rng(11)
    z = rng.normal(0, 1.5, 200)
    cfg = SignalConfig()
    pos_full = generate_signals(z, cfg)
    cut = 100
    pos_prefix = generate_signals(z[:cut], cfg)
    assert np.array_equal(pos_full[:cut], pos_prefix)


# ---------------- Stage 3: multi-asset screen ----------------

def test_screen_recovers_within_cluster_pairs():
    prices, true_pairs = make_universe(n=750, n_clusters=3, per_cluster=4, seed=2)
    res = screen_universe(prices, n_clusters=3, fdr=0.10)
    found = {frozenset((c.a, c.b)) for c in res.candidates}
    truth = {frozenset(p) for p in true_pairs}
    # Most true within-cluster pairs should be recovered.
    recovered = len(found & truth)
    assert recovered >= 0.5 * len(truth)


def test_fdr_monotone_in_level():
    # A stricter FDR can never discover MORE than a looser one.
    pvals = np.array([0.001, 0.008, 0.02, 0.04, 0.2, 0.5, 0.9])
    loose = benjamini_hochberg(pvals, fdr=0.20).sum()
    strict = benjamini_hochberg(pvals, fdr=0.05).sum()
    assert strict <= loose


def test_johansen_finds_rank_on_cointegrated_system():
    y, x = make_cointegrated_pair(n=800, beta=1.5, half_life=15.0, seed=9)
    import pandas as pd
    df = pd.DataFrame({"y": np.log(y.values), "x": np.log(x.values)})
    rank = johansen_rank(df)
    assert rank >= 1


# ---------------- Stage 4: backtest ----------------

def test_backtest_runs_and_reports_oos():
    y, x = make_cointegrated_pair(n=1000, beta=1.5, half_life=12.0, seed=4)
    res = backtest_pair(np.log(y.values), np.log(x.values),
                        train_size=252, test_size=63, use_kalman=True)
    assert np.isfinite(res.oos_sharpe)
    assert res.oos_equity[-1] > 0          # equity stays positive
    assert res.max_drawdown <= 0


def test_costs_reduce_returns():
    y, x = make_cointegrated_pair(n=1000, beta=1.5, half_life=12.0, seed=4)
    cheap = backtest_pair(np.log(y.values), np.log(x.values),
                          cost_model=CostModel(0.0, 0.0, 0.0))
    pricey = backtest_pair(np.log(y.values), np.log(x.values),
                           cost_model=CostModel(5.0, 5.0, 2.0))
    # Higher costs cannot produce a higher total return.
    assert pricey.total_return <= cheap.total_return + 1e-9


def test_backtest_no_lookahead_in_execution():
    # Appending future data must not change OOS returns already realised before
    # the appended region (t+1 execution => causal).
    y, x = make_cointegrated_pair(n=900, beta=1.5, half_life=12.0, seed=8)
    ly, lx = np.log(y.values), np.log(x.values)
    full = backtest_pair(ly, lx, train_size=252, test_size=63)
    trunc = backtest_pair(ly[:700], lx[:700], train_size=252, test_size=63)
    # Compare the overlapping OOS region up to the truncation point.
    a = np.nan_to_num(full.oos_returns[:680])
    b = np.nan_to_num(trunc.oos_returns[:680])
    assert np.allclose(a, b, atol=1e-9)
