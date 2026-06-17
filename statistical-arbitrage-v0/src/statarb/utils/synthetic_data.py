"""Synthetic APAC-style price generator with KNOWN cointegration ground truth.

Mirrors the Project 1 philosophy: generate data where we control the truth, so
every component can be validated by recovery. We can build:

  - a cointegrated pair (a shared stochastic trend + a stationary OU spread), so
    the EG test SHOULD detect it and recover the known hedge ratio and half-life;
  - an independent (non-cointegrated) pair, so the test should NOT fire (guards
    against false positives);
  - a clustered universe (sectors sharing trends) to exercise the multi-asset
    screen, FDR control, and Johansen rank.

Defaults evoke APAC index/single-name behaviour: ~20% annualised vol, daily bars.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_cointegrated_pair(
    n: int = 750,
    beta: float = 1.5,
    half_life: float = 15.0,
    spread_vol: float = 0.6,
    common_vol: float = 0.012,
    seed: int = 0,
    s0: tuple[float, float] = (100.0, 60.0),
) -> tuple[pd.Series, pd.Series]:
    """y and x share a common I(1) trend; their spread is a mean-reverting OU.

    Construction:
      common random walk  c_t  (the shared stochastic trend)
      x_t = exp(level_x + c_t)
      spread OU:  e_t = phi e_{t-1} + noise,  phi = exp(-ln2/half_life)
      y_t = exp(level_y + beta * c_t + e_t)

    So ln(y) - beta*ln(x) = const + e_t is stationary => cointegrated with the
    known hedge ratio `beta` and known `half_life`.
    """
    rng = np.random.default_rng(seed)
    phi = np.exp(-np.log(2.0) / half_life)
    sigma_e = spread_vol * np.sqrt(1 - phi**2)

    common = np.cumsum(rng.normal(0, common_vol, n))
    e = np.zeros(n)
    for t in range(1, n):
        e[t] = phi * e[t - 1] + rng.normal(0, sigma_e)

    ln_x = np.log(s0[1]) + common
    ln_y = np.log(s0[0]) + beta * common + e
    idx = pd.RangeIndex(n, name="t")
    return pd.Series(np.exp(ln_y), index=idx, name="Y"), pd.Series(np.exp(ln_x), index=idx, name="X")


def make_independent_pair(
    n: int = 750, vol: float = 0.012, seed: int = 1,
    s0: tuple[float, float] = (100.0, 100.0),
) -> tuple[pd.Series, pd.Series]:
    """Two independent random walks — NOT cointegrated. The EG test should fail
    to reject the unit root in the residual (no spurious detection).
    """
    rng = np.random.default_rng(seed)
    a = np.exp(np.log(s0[0]) + np.cumsum(rng.normal(0, vol, n)))
    b = np.exp(np.log(s0[1]) + np.cumsum(rng.normal(0, vol, n)))
    idx = pd.RangeIndex(n, name="t")
    return pd.Series(a, index=idx, name="A"), pd.Series(b, index=idx, name="B")


def make_universe(
    n: int = 750,
    n_clusters: int = 3,
    per_cluster: int = 4,
    seed: int = 2,
) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    """A clustered universe: each cluster shares a common trend (so within-cluster
    pairs are cointegrated), clusters are independent of each other (cross-cluster
    pairs are not). Returns (prices, list of truly-cointegrated pair names).

    This is the ground truth for the Stage-3 screen: a correct screen should
    recover within-cluster pairs and reject cross-cluster ones, and FDR should
    discard the chance "discoveries".
    """
    rng = np.random.default_rng(seed)
    cols: dict[str, np.ndarray] = {}
    true_pairs: list[tuple[str, str]] = []

    for c in range(n_clusters):
        # Strong shared sector factor: it must dominate DAILY returns so that
        # same-sector names co-move (high return correlation => clustering finds
        # them). The cointegrating spread is a slow, small-amplitude OU on top.
        common = np.cumsum(rng.normal(0, 0.018, n))
        base = 50 + 50 * rng.random()
        members = []
        for k in range(per_cluster):
            name = f"C{c}A{k}"
            members.append(name)
            beta = 0.8 + 0.6 * rng.random()
            phi = np.exp(-np.log(2.0) / rng.uniform(8, 20))
            # Small daily innovation => spread mean-reverts slowly with modest
            # amplitude, leaving daily co-movement dominated by `common`.
            se = 0.04 * np.sqrt(1 - phi**2)
            e = np.zeros(n)
            for t in range(1, n):
                e[t] = phi * e[t - 1] + rng.normal(0, se)
            ln_p = np.log(base + 5 * k) + beta * common + e
            cols[name] = np.exp(ln_p)
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                true_pairs.append((members[i], members[j]))

    prices = pd.DataFrame(cols, index=pd.RangeIndex(n, name="t"))
    return prices, true_pairs
