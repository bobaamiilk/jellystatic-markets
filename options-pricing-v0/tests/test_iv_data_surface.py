"""Tests for IV inversion (Stage 3), cleaning (Stage 4), surface (Stage 5)."""
import numpy as np
import pytest

from volsurface.pricing import black76_price
from volsurface.iv import implied_vol
from volsurface.data import clean_chain
from volsurface.surface import (
    calibrate_svi_slice, VolSurface, check_butterfly_arbitrage,
    check_calendar_arbitrage,
)
from volsurface.utils.synthetic_data import generate_chain, make_true_surface

F, R = 100.0, 0.04


# ---------- Stage 3: implied volatility ----------

def test_iv_round_trip_recovers_known_sigma():
    # Generate a price from a known sigma, invert, recover sigma.
    true_sigma = 0.23
    price = float(black76_price(F, 105.0, true_sigma, 0.5, R, "call"))
    res = implied_vol(price, F, 105.0, 0.5, R, "call")
    assert res.converged
    assert abs(res.iv - true_sigma) < 1e-6


def test_iv_round_trip_across_strikes_and_maturities():
    for K in (80.0, 95.0, 100.0, 110.0, 125.0):
        for tau in (0.08, 0.5, 1.0):
            true_sigma = 0.25
            price = float(black76_price(F, K, true_sigma, tau, R, "call"))
            res = implied_vol(price, F, K, tau, R, "call")
            assert res.converged, f"failed K={K} tau={tau}"
            assert abs(res.iv - true_sigma) < 1e-5


def test_iv_rejects_price_below_intrinsic():
    # Price below discounted intrinsic -> no solution, flagged with a reason.
    intrinsic = np.exp(-R * 0.5) * max(F - 80.0, 0.0)
    res = implied_vol(intrinsic * 0.5, F, 80.0, 0.5, R, "call")
    assert not res.converged
    assert np.isnan(res.iv)
    assert "below lower" in res.reason


def test_iv_rejects_price_above_upper_bound():
    upper = np.exp(-R * 0.5) * F
    res = implied_vol(upper * 1.5, F, 100.0, 0.5, R, "call")
    assert not res.converged
    assert "above upper" in res.reason


def test_iv_reports_low_vega_in_deep_wing():
    # Deep OTM short-dated: should still solve via Brent but flag tiny vega.
    true_sigma = 0.20
    price = float(black76_price(F, 145.0, true_sigma, 0.05, R, "call"))
    res = implied_vol(price, F, 145.0, 0.05, R, "call")
    if res.converged:  # if priceable at all
        assert res.vega_at_solution < 1.0  # vega genuinely small in the wing


# ---------- Stage 4: chain cleaning ----------

def test_cleaner_removes_crossed_and_zero_bid():
    chain = generate_chain(F=F, r=R, inject_noise=True)
    clean, report = clean_chain(chain, F=F, r=R)
    assert report.n_crossed >= 1
    assert report.n_zero_bid >= 1
    # No surviving quote should be crossed.
    assert (clean["bid"] < clean["ask"]).all()


def test_cleaner_flags_butterfly_violation():
    chain = generate_chain(F=F, r=R, inject_noise=True)
    clean, report = clean_chain(chain, F=F, r=R)
    assert report.n_arb_violations >= 1


def test_clean_chain_preserves_good_quotes():
    chain = generate_chain(F=F, r=R, inject_noise=False)
    clean, report = clean_chain(chain, F=F, r=R)
    # A clean synthetic chain should lose very little.
    assert report.n_output > 0.9 * report.n_input


# ---------- Stage 5: SVI surface ----------

def test_svi_recovers_known_slice():
    # Take one true slice, generate its total variance, refit, compare IV.
    true = make_true_surface()[2]  # tau = 0.5
    strikes = np.arange(80.0, 121.0, 2.5)
    k = np.log(strikes / F)
    w_mkt = true.total_variance(k)
    fitted, rmse = calibrate_svi_slice(k, w_mkt, tau=true.tau)
    assert rmse < 1e-4
    # IV recovered to high accuracy across the grid.
    assert np.max(np.abs(fitted.implied_vol(k) - true.implied_vol(k))) < 1e-3


def test_svi_calibration_respects_butterfly_constraint():
    true = make_true_surface()[3]  # tau = 1.0
    strikes = np.arange(70.0, 131.0, 2.5)
    k = np.log(strikes / F)
    w_mkt = true.total_variance(k)
    fitted, _ = calibrate_svi_slice(k, w_mkt, tau=true.tau, enforce_no_arb=True)
    assert fitted.butterfly_margin() >= -1e-6  # b(1+|rho|) <= 4


def test_fitted_slice_is_butterfly_arbitrage_free():
    true = make_true_surface()[1]
    strikes = np.arange(75.0, 126.0, 2.5)
    k = np.log(strikes / F)
    fitted, _ = calibrate_svi_slice(k, true.total_variance(k), tau=true.tau)
    k_dense = np.linspace(-0.5, 0.5, 200)
    assert check_butterfly_arbitrage(fitted, k_dense)


def test_full_surface_calendar_arbitrage_free():
    surface = make_true_surface()
    k_dense = np.linspace(-0.4, 0.4, 100)
    assert check_calendar_arbitrage(surface, k_dense)


def test_volsurface_interpolation_monotone_in_total_variance():
    surface = VolSurface(make_true_surface())
    # Total variance at fixed k must be non-decreasing across interpolated taus.
    k = 0.0
    taus = np.linspace(0.1, 1.0, 20)
    w = [surface.implied_vol(k, t) ** 2 * t for t in taus]
    assert all(w[i + 1] >= w[i] - 1e-6 for i in range(len(w) - 1))


def test_negative_skew_present_in_true_surface():
    # Sanity: OTM put IV > OTM call IV (the equity skew the data encodes).
    sl = make_true_surface()[2]
    iv_put_wing = sl.implied_vol(np.array([np.log(85.0 / F)]))[0]
    iv_call_wing = sl.implied_vol(np.array([np.log(115.0 / F)]))[0]
    assert iv_put_wing > iv_call_wing
