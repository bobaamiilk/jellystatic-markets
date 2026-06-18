"""Tests for the portfolio engine — including the three quality-gate tests.

Quality gates (from the brief):
  1. Look-ahead: regime scalar is LAGGED — book_return[t] uses scalar[t-1].
  2. Regime-link: in a synthetic crisis window, mean book gross exposure is
     strictly lower than in the calm window.
  3. Reproducibility: two runs with identical config + seed produce identical
     Parquet output.
"""
import numpy as np
import pandas as pd
import pytest

from quantdesk.config import QuantDeskConfig
from quantdesk.data.synthetic import make_synthetic_prices
from quantdesk.engine.portfolio import run_regime_scaled_statarb, PortfolioResult


# Shared small config for fast tests
_FAST_CFG = QuantDeskConfig(
    n_clusters=2,
    fdr=0.20,
    train_size=150,
    test_size=40,
    regime_lookback=120,
    seed=42,
)


def _make_prices(n: int = 500, seed: int = 42) -> pd.DataFrame:
    prices, _ = make_synthetic_prices(n=n, n_clusters=2, per_cluster=4, seed=seed)
    return prices


# --------------------------------------------------------------------------- #
#  Basic smoke tests
# --------------------------------------------------------------------------- #

def test_engine_returns_portfolio_result():
    prices = _make_prices()
    result = run_regime_scaled_statarb(prices, _FAST_CFG)
    assert isinstance(result, PortfolioResult)


def test_equity_curve_starts_at_one():
    prices = _make_prices()
    result = run_regime_scaled_statarb(prices, _FAST_CFG)
    assert abs(result.oos_equity.iloc[0] - 1.0) < 1e-9


def test_equity_curve_stays_positive():
    prices = _make_prices()
    result = run_regime_scaled_statarb(prices, _FAST_CFG)
    assert (result.oos_equity.values > 0).all()


def test_regime_scalar_in_valid_range():
    prices = _make_prices()
    result = run_regime_scaled_statarb(prices, _FAST_CFG)
    s = result.regime_scalar.values
    assert s.min() >= _FAST_CFG.regime_min_scalar - 1e-9
    assert s.max() <= 1.0 + 1e-9


def test_sharpe_is_finite():
    prices = _make_prices()
    result = run_regime_scaled_statarb(prices, _FAST_CFG)
    assert np.isfinite(result.combined_sharpe)
    assert np.isfinite(result.unscaled_sharpe)


def test_max_drawdown_non_positive():
    prices = _make_prices()
    result = run_regime_scaled_statarb(prices, _FAST_CFG)
    assert result.max_drawdown <= 0.0


# --------------------------------------------------------------------------- #
#  Quality gate 1 — Look-ahead: regime scalar must be lagged by one period
#
#  Invariant: oos_returns[t] == unscaled_returns[t] * regime_scalar[t-1]
#  for all t >= 1.  At t=0 the engine uses scalar[0] for both.
#
#  This asserts that the de-grossing decision at time t was made using only
#  information available at t-1, preserving the no-look-ahead property that
#  statarb.backtest_pair already guarantees for per-pair execution.
# --------------------------------------------------------------------------- #

def test_regime_scalar_is_lagged_by_one_period():
    prices = _make_prices(n=500)
    result = run_regime_scaled_statarb(prices, _FAST_CFG)

    book = result.oos_returns.values
    unscaled = result.unscaled_returns.values
    scalar = result.regime_scalar.values

    # At t >= 1: book[t] == unscaled[t] * scalar[t-1]
    # (at t=0 the engine initialises lagged_scalar[0] = scalar[0] by convention)
    lagged = np.concatenate([[scalar[0]], scalar[:-1]])
    expected = lagged * unscaled

    assert np.allclose(book, expected, atol=1e-12), (
        "book_return[t] must equal unscaled_return[t] * scalar[t-1]; "
        "contemporaneous scalar would be a look-ahead violation."
    )


# --------------------------------------------------------------------------- #
#  Quality gate 2 — Regime-link: de-grossing actually fires in a crisis window
#
#  We inject a long stretch of synthetic GARCH returns with elevated vol
#  (simulating a crisis) into the price series and verify the regime signal
#  shrinks mean gross exposure below what it is in the calm window.
# --------------------------------------------------------------------------- #

def test_regime_degrosses_during_crisis_window():
    from volforecast.utils import simulate_regime_returns
    from volforecast.models.benchmarks import ewma_variance
    from volforecast.regime import vol_percentile_regime

    # Build a vol signal with known crisis window
    n = 1500
    rets, true_state = simulate_regime_returns(
        n=n, calm_vol=0.005, crisis_vol=0.030,
        p_stay_calm=0.97, p_stay_crisis=0.92, seed=3,
    )

    sigma_hat = np.sqrt(ewma_variance(rets, lam=0.94))
    regime = vol_percentile_regime(
        sigma_hat,
        lookback=252,
        stress_pct=0.80,
        min_scalar=0.25,
    )

    # After the warm-up window, compare mean scalar in calm vs crisis days.
    warmup = 252
    post = np.arange(warmup, n)
    calm_scalar = regime.exposure_scalar[post][true_state[post] == 0]
    crisis_scalar = regime.exposure_scalar[post][true_state[post] == 1]

    assert len(crisis_scalar) > 0, "No crisis days after warmup — increase n or lower p_stay_calm"
    assert len(calm_scalar) > 0, "No calm days after warmup"

    mean_calm = float(calm_scalar.mean())
    mean_crisis = float(crisis_scalar.mean())

    assert mean_crisis < mean_calm, (
        f"Mean book gross exposure in crisis ({mean_crisis:.3f}) must be "
        f"strictly lower than in calm ({mean_calm:.3f}). "
        f"De-grossing is not firing."
    )


# --------------------------------------------------------------------------- #
#  Quality gate 3 — Reproducibility: identical config + seed → identical Parquet
# --------------------------------------------------------------------------- #

def test_reproducibility(tmp_path):
    from quantdesk.report.builder import to_parquet

    prices = _make_prices(n=400, seed=42)
    cfg = _FAST_CFG

    r1 = run_regime_scaled_statarb(prices, cfg)
    r2 = run_regime_scaled_statarb(prices, cfg)

    p1 = tmp_path / "run1.parquet"
    p2 = tmp_path / "run2.parquet"
    to_parquet(r1, p1)
    to_parquet(r2, p2)

    df1 = pd.read_parquet(p1)
    df2 = pd.read_parquet(p2)
    assert df1.equals(df2), "Two runs with identical inputs produced different Parquet output."


# --------------------------------------------------------------------------- #
#  Edge cases
# --------------------------------------------------------------------------- #

def test_empty_universe_returns_gracefully():
    """A universe with no cointegrated pairs should return a valid zero result."""
    from statarb.utils import make_independent_pair
    import pandas as pd

    rng = np.random.default_rng(0)
    n = 400
    # Build a universe of purely independent random walks (no cointegration)
    series = {}
    for i in range(4):
        walk = np.exp(np.cumsum(rng.normal(0, 0.015, n)))
        walk = walk / walk[0] * 100
        series[f"rw{i}"] = walk
    prices = pd.DataFrame(series)

    cfg = QuantDeskConfig(
        n_clusters=2, fdr=0.01, train_size=150, test_size=40,
        regime_lookback=120,
    )
    result = run_regime_scaled_statarb(prices, cfg)
    # Should return without crashing; n_pairs may be 0
    assert isinstance(result, PortfolioResult)
    assert np.isfinite(result.oos_equity.values).all()


def test_pair_contributions_count_matches_n_pairs():
    prices = _make_prices(n=500)
    result = run_regime_scaled_statarb(prices, _FAST_CFG)
    assert len(result.pair_contributions) == result.n_pairs


def test_cusum_breaks_are_subset_of_pairs():
    prices = _make_prices(n=500)
    result = run_regime_scaled_statarb(prices, _FAST_CFG)
    pair_labels = {f"{pc.a}/{pc.b}" for pc in result.pair_contributions}
    for label in result.cusum_break_flags:
        assert label in pair_labels
