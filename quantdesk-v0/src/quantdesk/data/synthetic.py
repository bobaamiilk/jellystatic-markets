"""Synthetic data fallback — delegates entirely to the frozen upstream packages.

No duplication of GARCH or cointegration logic. All generation is done by:
  - statarb.utils.make_universe  (clustered, cointegrated APAC-style prices)
  - volforecast.utils.simulate_garch  (returns with known conditional variance)

This lets the full pipeline run with zero external data for testing and demos.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from statarb.utils import make_universe
from volforecast.utils import simulate_garch


def make_synthetic_prices(
    n: int = 750,
    n_clusters: int = 3,
    per_cluster: int = 4,
    seed: int = 42,
) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    """Return (prices, true_cointegrated_pairs) for an APAC-style clustered universe.

    Delegates to statarb.utils.make_universe: within-cluster pairs share a
    common stochastic trend (cointegrated with known ground-truth hedge ratios);
    cross-cluster pairs are independent.  The returned prices are raw price
    levels (>0), suitable for log-transformation before cointegration testing.
    """
    return make_universe(n=n, n_clusters=n_clusters, per_cluster=per_cluster, seed=seed)


def make_synthetic_market_returns(
    n: int = 750,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (returns, true_conditional_variance) from a GARCH(1,1) process.

    Delegates to volforecast.utils.simulate_garch. Used to build the market-
    index return series that feeds the vol forecast + regime de-grossing signal.
    """
    return simulate_garch(
        n=n,
        omega=2e-6,
        alpha=0.08,
        beta=0.90,
        seed=seed,
    )
