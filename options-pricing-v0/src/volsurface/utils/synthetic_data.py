"""Synthetic APAC-style option chain generator.

Generates an arbitrage-free chain from a known SVI surface, then optionally
injects realistic market noise (bid/ask spreads, stale/crossed quotes, a
butterfly violation). This lets every downstream component be validated against
ground truth: we KNOW the true IV that produced the prices, so we can assert the
IV solver and surface fitter recover it.

Defaults are tuned to resemble an equity-index surface with negative skew
(KOSPI200 / HSI / Nikkei style): ATM vol ~20%, left-wing rich.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from volsurface.pricing.black_scholes import black76_price
from volsurface.surface.svi import SVIParams


def make_true_surface() -> list[SVIParams]:
    """A hand-built, arbitrage-free multi-maturity SVI surface with equity skew.

    rho < 0 produces the negative skew (puts richer than calls) seen on every
    APAC equity index. Total variance increases with tau (calendar-arb-free).
    """
    return [
        SVIParams(a=0.010, b=0.10, rho=-0.40, m=0.00, s=0.12, tau=0.08),
        SVIParams(a=0.018, b=0.13, rho=-0.42, m=0.00, s=0.15, tau=0.25),
        SVIParams(a=0.030, b=0.16, rho=-0.45, m=0.00, s=0.18, tau=0.50),
        SVIParams(a=0.050, b=0.20, rho=-0.48, m=0.00, s=0.20, tau=1.00),
    ]


def generate_chain(
    F: float = 100.0,
    r: float = 0.04,
    strikes: np.ndarray | None = None,
    surface: list[SVIParams] | None = None,
    spread_bps: float = 50.0,
    inject_noise: bool = False,
    seed: int | None = 42,
) -> pd.DataFrame:
    """Generate a synthetic chain priced off a known SVI surface.

    Parameters
    ----------
    F : forward price.
    r : risk-free rate.
    strikes : strike grid; defaults to 70..130 in steps of 2.5.
    spread_bps : half-spread in basis points of price for the bid/ask.
    inject_noise : if True, add a crossed quote, a zero-bid quote, and a
        butterfly violation to exercise the Stage 4 cleaner.

    Returns a tidy DataFrame: strike, maturity, option_type, bid, ask, plus the
    ground-truth columns true_iv and true_price for validation.
    """
    rng = np.random.default_rng(seed)
    if strikes is None:
        strikes = np.arange(70.0, 130.0 + 1e-9, 2.5)
    if surface is None:
        surface = make_true_surface()

    rows = []
    for sl in surface:
        tau = sl.tau
        k = np.log(strikes / F)
        true_iv = sl.implied_vol(k)
        for otype in ("call", "put"):
            price = black76_price(F, strikes, true_iv, tau, r, otype)
            half = spread_bps * 1e-4 * np.maximum(price, 1e-4)
            bid = np.maximum(price - half, 0.0)
            ask = price + half
            for Ki, p, iv, b, a in zip(strikes, price, true_iv, bid, ask):
                rows.append({
                    "strike": float(Ki), "maturity": float(tau),
                    "option_type": otype, "bid": float(b), "ask": float(a),
                    "true_iv": float(iv), "true_price": float(p),
                })

    df = pd.DataFrame(rows)

    if inject_noise:
        # 1) crossed quote (bid > ask)
        idx = df.index[(df.option_type == "call") & (df.maturity == 0.25)][5]
        df.loc[idx, ["bid", "ask"]] = [df.loc[idx, "ask"] * 1.2, df.loc[idx, "ask"]]
        # 2) zero-bid quote
        idx2 = df.index[(df.option_type == "put") & (df.maturity == 0.5)][-1]
        df.loc[idx2, "bid"] = 0.0
        # 3) butterfly violation: dent a single mid downward
        idx3 = df.index[(df.option_type == "call") & (df.maturity == 1.0)][8]
        df.loc[idx3, ["bid", "ask"]] = [
            df.loc[idx3, "bid"] * 0.5, df.loc[idx3, "ask"] * 0.5,
        ]

    return df.reset_index(drop=True)
