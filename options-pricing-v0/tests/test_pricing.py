"""Tests for the pricing layer (Stages 1-2).

These encode the validation requirements stated in the curriculum:
  - closed-form sanity identities (limits, put-call parity, delta bounds);
  - MC matches Black-Scholes within standard-error bands;
  - variance reduction actually reduces variance;
  - tree converges to closed form and Richardson is more stable than raw.
"""
import numpy as np
import pytest

from volsurface.pricing import (
    black_scholes_price, black76_price, black_scholes_greeks,
    put_call_parity_residual, mc_price_plain, mc_price_antithetic,
    mc_price_control_variate, binomial_price, binomial_price_richardson,
)

# A standard liquid-option test case.
S0, K, SIGMA, TAU, R, Q = 100.0, 100.0, 0.20, 1.0, 0.04, 0.0


# ---------- Stage 1: closed-form sanity identities ----------

def test_put_call_parity_holds_exactly():
    c = black_scholes_price(S0, K, SIGMA, TAU, R, Q, "call")
    p = black_scholes_price(S0, K, SIGMA, TAU, R, Q, "put")
    resid = put_call_parity_residual(c, p, S0, K, TAU, R, Q)
    assert abs(float(resid)) < 1e-10


def test_zero_vol_limit_is_discounted_intrinsic():
    # As sigma -> 0, call -> max(S - K e^{-r tau}, 0).
    c = black_scholes_price(S0, K, 1e-9, TAU, R, Q, "call")
    expected = max(S0 - K * np.exp(-R * TAU), 0.0)
    assert abs(float(c) - expected) < 1e-6


def test_expiry_limit_is_intrinsic():
    c = black_scholes_price(110.0, K, SIGMA, 1e-9, R, Q, "call")
    assert abs(float(c) - 10.0) < 1e-4


def test_call_delta_in_unit_interval():
    g = black_scholes_greeks(S0, K, SIGMA, TAU, R, Q, "call")
    assert 0.0 < float(g.delta) < 1.0


def test_vega_positive_and_peaks_near_atm():
    g_atm = black_scholes_greeks(S0, 100.0, SIGMA, TAU, R, Q, "call")
    g_otm = black_scholes_greeks(S0, 150.0, SIGMA, TAU, R, Q, "call")
    assert float(g_atm.vega) > 0
    assert float(g_atm.vega) > float(g_otm.vega)  # vega collapses in the wings


def test_black76_matches_bs_via_forward():
    # BS(spot) must equal Black-76 evaluated at the forward F = S e^{(r-q)tau}.
    F = S0 * np.exp((R - Q) * TAU)
    c_bs = black_scholes_price(S0, K, SIGMA, TAU, R, Q, "call")
    c_76 = black76_price(F, K, SIGMA, TAU, R, "call")
    assert abs(float(c_bs) - float(c_76)) < 1e-10


# ---------- Stage 2: Monte Carlo ----------

def test_mc_plain_matches_bs_within_se():
    bs = float(black_scholes_price(S0, K, SIGMA, TAU, R, Q, "call"))
    res = mc_price_plain(S0, K, SIGMA, TAU, R, Q, "call", n_paths=200_000, seed=1)
    # Within 4 standard errors is an extremely safe band.
    assert abs(res.price - bs) < 4 * res.std_error


def test_antithetic_reduces_variance():
    plain = mc_price_plain(S0, K, SIGMA, TAU, R, Q, "call", n_paths=100_000, seed=2)
    anti = mc_price_antithetic(S0, K, SIGMA, TAU, R, Q, "call", n_paths=100_000, seed=2)
    assert anti.std_error < plain.std_error


def test_control_variate_reduces_variance():
    plain = mc_price_plain(S0, K, SIGMA, TAU, R, Q, "call", n_paths=100_000, seed=3)
    cv = mc_price_control_variate(S0, K, SIGMA, TAU, R, Q, "call", n_paths=100_000, seed=3)
    assert cv.std_error < plain.std_error


def test_mc_standard_error_decays_as_sqrt_n():
    se_small = mc_price_plain(S0, K, SIGMA, TAU, R, Q, "call", n_paths=10_000, seed=4).std_error
    se_large = mc_price_plain(S0, K, SIGMA, TAU, R, Q, "call", n_paths=160_000, seed=4).std_error
    # 16x paths -> SE should drop by ~4x. Allow generous tolerance.
    ratio = se_small / se_large
    assert 3.0 < ratio < 5.0


# ---------- Stage 2: binomial tree ----------

def test_tree_converges_to_bs_european():
    bs = float(black_scholes_price(S0, K, SIGMA, TAU, R, Q, "call"))
    tree = binomial_price(S0, K, SIGMA, TAU, R, Q, "call", "european", n_steps=2000)
    assert abs(tree - bs) < 1e-2


def test_american_call_no_dividend_equals_european():
    # Classic identity: American call on non-dividend stock = European call.
    eu = binomial_price(S0, K, SIGMA, TAU, R, 0.0, "call", "european", n_steps=1000)
    am = binomial_price(S0, K, SIGMA, TAU, R, 0.0, "call", "american", n_steps=1000)
    assert abs(eu - am) < 1e-6


def test_american_put_premium_positive():
    # American put >= European put (early exercise has value).
    eu = binomial_price(S0, K, SIGMA, TAU, R, 0.0, "put", "european", n_steps=1000)
    am = binomial_price(S0, K, SIGMA, TAU, R, 0.0, "put", "american", n_steps=1000)
    assert am > eu - 1e-9


def test_richardson_more_stable_than_raw_tree():
    bs = float(black_scholes_price(S0, K, SIGMA, TAU, R, Q, "call"))
    raw_errors, rich_errors = [], []
    for n in (50, 60, 70, 80, 90, 100):
        raw = binomial_price(S0, K, SIGMA, TAU, R, Q, "call", "european", n)
        rich = binomial_price_richardson(S0, K, SIGMA, TAU, R, Q, "call", "european", n)
        raw_errors.append(abs(raw - bs))
        rich_errors.append(abs(rich - bs))
    # The parity-robust Richardson estimate should be at least an order of
    # magnitude more accurate than the raw tree across the board.
    assert max(rich_errors) < 0.1 * max(raw_errors)


def test_crr_probability_guard_raises():
    # p exits [0,1] when the rate drift dominates the vol step:
    # (r-q)*dt > sigma*sqrt(dt), i.e. LOW vol, HIGH rate, coarse dt.
    # (A common student error is to expect HIGH vol to break it -- it doesn't.)
    with pytest.raises(ValueError):
        binomial_price(S0, K, sigma=0.05, tau=1.0, r=0.80, q=0.0,
                       option_type="call", exercise="european", n_steps=1)


def test_discrete_dividend_lowers_call_value():
    no_div = binomial_price(S0, K, SIGMA, TAU, R, 0.0, "call", "american", n_steps=500)
    with_div = binomial_price(
        S0, K, SIGMA, TAU, R, 0.0, "call", "american", n_steps=500,
        discrete_dividends=[(0.5, 3.0)],
    )
    assert with_div < no_div  # a dividend reduces the call's value
