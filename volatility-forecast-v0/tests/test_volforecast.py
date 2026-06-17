"""Tests for the volatility-forecasting engine.

Encodes the curriculum's validation requirements:
  - GARCH MLE recovers known parameters and respects stationarity;
  - EGARCH recovers a negative leverage term;
  - EWMA equals constrained GARCH (omega=0, alpha+beta=1) and forecasts flat;
  - QLIKE & MSE are robust; the GARCH forecast beats a deliberately-bad one;
  - Diebold-Mariano returns 'neither' for identical models;
  - vol targeting brings realised vol toward target and respects the cap;
  - the regime detector flags the crisis state and de-grosses.
"""
import numpy as np
import pytest

from volforecast.models import (
    fit_garch, fit_egarch, rolling_variance, ewma_variance,
    ewma_as_garch_params, ewma_forecast,
)
from volforecast.evaluation import (
    qlike_loss, mse_loss, mae_vol_loss, diebold_mariano, per_period_qlike,
    walk_forward_eval,
)
from volforecast.trading import apply_vol_target, VolTargetConfig
from volforecast.regime import vol_percentile_regime, detect_structural_break_cusum
from volforecast.utils import simulate_garch, simulate_egarch, simulate_regime_returns


# ---------------- Stage 1: GARCH / EGARCH ----------------

def test_garch_recovers_parameters():
    r, _ = simulate_garch(n=5000, omega=2e-6, alpha=0.08, beta=0.90, seed=0)
    fit = fit_garch(r)
    assert fit.converged
    # Persistence is the best-identified quantity in GARCH; recover it tightly.
    assert abs(fit.persistence - 0.98) < 0.03
    assert abs(fit.alpha - 0.08) < 0.04


def test_garch_stationarity_enforced():
    r, _ = simulate_garch(n=2000, seed=1)
    fit = fit_garch(r)
    assert fit.persistence < 1.0          # alpha + beta < 1
    assert fit.omega > 0
    assert np.isfinite(fit.long_run_var)


def test_garch_forecast_mean_reverts():
    r, _ = simulate_garch(n=3000, omega=2e-6, alpha=0.08, beta=0.90, seed=2)
    fit = fit_garch(r)
    fc = fit.forecast_var(horizon=2000)
    # Long-horizon forecast should approach the long-run variance (geometric
    # decay at rate persistence; near-integrated fits need a long horizon).
    assert abs(fc[-1] - fit.long_run_var) / fit.long_run_var < 0.05


def test_egarch_recovers_leverage_sign():
    r, _ = simulate_egarch(n=5000, gamma=-0.08, seed=0)
    fit = fit_egarch(r)
    # The headline result: leverage term is negative.
    assert fit.gamma < 0
    assert fit.has_leverage


# ---------------- Stage 2: benchmarks ----------------

def test_ewma_equals_constrained_garch():
    p = ewma_as_garch_params(lam=0.94)
    assert p["omega"] == 0.0
    assert abs(p["alpha"] - 0.06) < 1e-12
    assert abs(p["beta"] - 0.94) < 1e-12
    assert abs(p["persistence"] - 1.0) < 1e-12


def test_ewma_forecast_is_flat():
    fc = ewma_forecast(last_var=4e-4, horizon=50)
    # No mean reversion => every horizon equals the current level.
    assert np.allclose(fc, 4e-4)
    assert fc.std() < 1e-15        # flat up to floating-point error


def test_rolling_variance_is_causal():
    r, _ = simulate_garch(n=500, seed=3)
    rv = rolling_variance(r, window=21)
    assert np.all(np.isnan(rv[:20]))
    assert np.isfinite(rv[21])


def test_rolling_ghosting_artifact():
    # A single huge return should keep rolling variance elevated for exactly
    # `window` days, then drop sharply when it exits — the ghosting artifact.
    r = np.concatenate([np.full(40, 0.001), [0.10], np.full(40, 0.001)])
    w = 21
    rv = rolling_variance(r, window=w)
    spike_idx = 40
    # variance stays elevated while the shock is in the window...
    assert rv[spike_idx + w - 1] > 5 * rv[spike_idx - 1]
    # ...then drops off a cliff the day the shock leaves the window.
    assert rv[spike_idx + w] < 0.5 * rv[spike_idx + w - 1]


# ---------------- Stage 3: evaluation ----------------

def test_qlike_and_mse_prefer_true_variance():
    r, true_var = simulate_garch(n=3000, seed=4)
    r2 = r**2
    good = true_var
    bad = np.full_like(true_var, true_var.mean())   # constant-vol forecast
    # Both robust losses should rank the true-variance forecast better (lower).
    assert qlike_loss(good, r2) < qlike_loss(bad, r2)
    assert mse_loss(good, r2) < mse_loss(bad, r2)


def test_qlike_penalises_underprediction_more():
    # Same absolute variance error, but under-prediction should cost more in QLIKE.
    r2 = np.full(1000, 4e-4)
    over = np.full(1000, 5e-4)    # over-predict by 1e-4
    under = np.full(1000, 3e-4)   # under-predict by 1e-4
    assert qlike_loss(under, r2) > qlike_loss(over, r2)


def test_dm_identical_models_is_neither():
    r, _ = simulate_garch(n=1000, seed=5)
    r2 = r**2
    fc = ewma_variance(r)
    dm = diebold_mariano(per_period_qlike(fc, r2), per_period_qlike(fc, r2))
    assert dm.better == "neither"


def test_walk_forward_runs_oos():
    r, _ = simulate_garch(n=2000, omega=2e-6, alpha=0.08, beta=0.90, seed=6)
    res = walk_forward_eval(r, train_size=500, test_size=21)
    assert np.isfinite(res.qlike["GARCH"])
    assert np.isfinite(res.qlike["EWMA"])
    # On true GARCH data, GARCH should not be materially worse than EWMA on QLIKE.
    assert res.qlike["GARCH"] <= res.qlike["EWMA"] + 0.05


# ---------------- Stage 4: vol targeting ----------------

def test_vol_target_brings_realised_vol_near_target():
    r, true_var = simulate_garch(n=3000, seed=7)
    sigma_hat = np.sqrt(true_var)        # near-perfect forecast (truth)
    cfg = VolTargetConfig(target_vol=0.10, max_leverage=10.0, vol_floor=1e-4)
    out = apply_vol_target(r, sigma_hat, cfg)
    # With a near-perfect forecast realised vol should land close to target.
    assert abs(out.realised_vol - 0.10) < 0.03


def test_vol_target_respects_leverage_cap():
    r, true_var = simulate_garch(n=2000, seed=8)
    sigma_hat = np.sqrt(true_var)
    cfg = VolTargetConfig(target_vol=0.50, max_leverage=2.0, vol_floor=1e-4)
    out = apply_vol_target(r, sigma_hat, cfg)
    assert out.weights.max() <= 2.0 + 1e-9


# ---------------- Stage 5: regime ----------------

def test_regime_flags_crisis_state():
    r, state = simulate_regime_returns(n=1500, seed=0)
    # Use a trailing rolling vol as the forecast input to the detector.
    rv = np.sqrt(ewma_variance(r, lam=0.94))
    reg = vol_percentile_regime(rv, lookback=252, stress_pct=0.80)
    # Among true crisis days (after the percentile warmup), a good fraction
    # should be flagged stressed.
    n = len(state)
    crisis_post_warmup = (state == 1) & (np.arange(n) >= 252)
    detected = reg.state == 1
    overlap = np.mean(detected[crisis_post_warmup])
    assert overlap > 0.3
    assert reg.frac_stressed > 0


def test_regime_degrosses_in_stress():
    r, _ = simulate_regime_returns(n=1500, seed=1)
    rv = np.sqrt(ewma_variance(r, lam=0.94))
    reg = vol_percentile_regime(rv)
    # Exposure scalar must be <= 1 everywhere and < 1 somewhere (de-grossing).
    assert reg.exposure_scalar.max() <= 1.0 + 1e-9
    assert reg.exposure_scalar.min() < 1.0


def test_cusum_detects_structural_break():
    rng = np.random.default_rng(0)
    # Stationary then a permanent level shift (a structural break).
    pre = rng.normal(0, 1, 300)
    post = rng.normal(5, 1, 300)
    spread = np.concatenate([pre, post])
    idx = detect_structural_break_cusum(spread, threshold=1.0)
    assert idx != -1
