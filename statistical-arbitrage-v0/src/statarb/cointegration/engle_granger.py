"""Cointegration testing — the statistical foundation of stat-arb (Stage 1).

Two I(1) price series are *cointegrated* if some linear combination is I(0)
(stationary, mean-reverting). We trade that stationary spread, not the direction
of either leg.

Engle-Granger two-step:
  1. OLS  y_t = alpha + beta * x_t + e_t   (beta is the hedge ratio)
  2. ADF-test the residual e_t for stationarity, using cointegration-specific
     critical values (the residual is *estimated*, so standard ADF CVs are wrong).

Why OLS is valid on non-stationary data: under true cointegration the OLS beta is
*super-consistent* — it converges at rate T, not sqrt(T). This is the one place
regressing an I(1) on an I(1) is not spurious.

These wrap statsmodels for the well-trodden tests (ADF, Johansen) and add the
domain-specific pieces: hedge-ratio direction handling, half-life of mean
reversion, and an honest tradeable-vs-merely-cointegrated distinction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from statsmodels.tsa.stattools import adfuller, coint


@dataclass
class CointResult:
    """Engle-Granger cointegration test result plus the trading-relevant pieces.

    hedge_ratio (beta) is what you actually short against one unit of y.
    half_life is the OU mean-reversion time — a live health metric: if it drifts
    up over time the relationship is decaying and the edge is dying.
    """

    is_cointegrated: bool
    pvalue: float
    hedge_ratio: float
    intercept: float
    adf_stat: float
    adf_pvalue: float
    half_life: float
    spread: NDArray
    dependent: str = "y"
    independent: str = "x"


def ols_hedge_ratio(y: NDArray, x: NDArray) -> tuple[float, float, NDArray]:
    """Static OLS hedge ratio: y = alpha + beta x + e. Returns (beta, alpha, resid).

    Super-consistent under cointegration. This is the Stage-1 static estimate;
    Stage 2 upgrades it to a rolling / Kalman time-varying beta.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha, beta = float(coef[0]), float(coef[1])
    resid = y - (alpha + beta * x)
    return beta, alpha, resid


def half_life_of_mean_reversion(spread: NDArray) -> float:
    """Half-life from an Ornstein-Uhlenbeck fit on the spread.

    Discretized OU:  d e_t = lambda * e_{t-1} + eps  (regress delta on lag).
    lambda < 0 => mean-reverting; half-life = -ln(2) / lambda.

    Returns np.inf if the spread is not mean-reverting (lambda >= 0). A rising
    half-life across re-estimations is the early-warning signal that a pair is
    breaking down (Stage 5).
    """
    spread = np.asarray(spread, dtype=float)
    lagged = spread[:-1]
    delta = np.diff(spread)
    # delta = lambda * lagged + const + eps
    X = np.column_stack([np.ones_like(lagged), lagged])
    coef, *_ = np.linalg.lstsq(X, delta, rcond=None)
    lam = float(coef[1])
    if lam >= 0:
        return np.inf
    return float(-np.log(2.0) / lam)


def adf_test(series: NDArray, regression: str = "c") -> tuple[float, float]:
    """Augmented Dickey-Fuller. Returns (statistic, pvalue). H0: unit root.

    A low p-value rejects the unit root => the series is stationary. Used on the
    EG residual; note the p-value there is only approximate because the residual
    is estimated (engle_granger() uses the proper MacKinnon CVs instead).
    """
    series = np.asarray(series, dtype=float)
    stat, pval, *_ = adfuller(series, regression=regression, autolag="AIC")
    return float(stat), float(pval)


def engle_granger(
    y: NDArray,
    x: NDArray,
    significance: float = 0.05,
    y_name: str = "y",
    x_name: str = "x",
) -> CointResult:
    """Full Engle-Granger test with the correct cointegration critical values.

    Uses statsmodels.coint (which applies MacKinnon CVs appropriate for an
    *estimated* residual) for the decision p-value, and reports the OLS hedge
    ratio and the residual half-life for trading.

    NOTE on asymmetry: regressing y~x and x~y can give different answers in
    finite samples. Convention here: caller passes the more-liquid / higher-vol
    leg as `x` (the regressor). For a definitive symmetric answer use Johansen.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)

    beta, alpha, resid = ols_hedge_ratio(y, x)
    # coint() returns the EG test statistic and a properly-sized p-value.
    eg_stat, eg_pvalue, _ = coint(y, x, trend="c", autolag="AIC")
    adf_stat, adf_pval = adf_test(resid, regression="c")
    hl = half_life_of_mean_reversion(resid)

    return CointResult(
        is_cointegrated=bool(eg_pvalue < significance),
        pvalue=float(eg_pvalue),
        hedge_ratio=beta,
        intercept=alpha,
        adf_stat=adf_stat,
        adf_pvalue=adf_pval,
        half_life=hl,
        spread=resid,
        dependent=y_name,
        independent=x_name,
    )


def is_tradeable(
    result: CointResult,
    max_half_life: float = 60.0,
    min_half_life: float = 1.0,
    capital_controls: bool = False,
) -> tuple[bool, str]:
    """Cointegrated is NOT the same as tradeable. Apply the desk filters.

    The A-H share premium is the canonical APAC trap: statistically cointegrated
    (same company, dual-listed) yet untradeable because capital controls / Stock
    Connect quotas block the cross-boundary execution. Convergence happens via
    policy, not arbitrage capital.

    Returns (tradeable, reason).
    """
    if not result.is_cointegrated:
        return False, "not cointegrated at chosen significance"
    if capital_controls:
        return False, "capital controls prevent execution (e.g. A-H premium)"
    if not np.isfinite(result.half_life):
        return False, "spread not mean-reverting (infinite half-life)"
    if result.half_life > max_half_life:
        return False, f"half-life {result.half_life:.1f}d too slow (edge decays / capital tied up)"
    if result.half_life < min_half_life:
        return False, f"half-life {result.half_life:.2f}d too fast (noise / eaten by costs)"
    return True, "tradeable"
