"""Multi-asset cointegration screening (Stage 3).

Going from one pair to a universe creates two problems:

1. COMBINATORIAL EXPLOSION + MULTIPLE TESTING. n assets => n(n-1)/2 pairs. At
   n=200 that's ~20,000 tests; at 5% significance you get ~1,000 "cointegrated"
   pairs BY CHANCE. Naive screening is mostly false discoveries.

   Defences here:
     - economic pre-filtering via correlation clustering (only test within
       economically coherent groups), and
     - Benjamini-Hochberg FDR control on the surviving p-values.

2. >2-ASSET RELATIONSHIPS. Johansen finds the *number* of independent
   cointegrating vectors in a system (its rank) symmetrically — no need to pick
   a dependent variable as Engle-Granger does.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from statarb.cointegration.engle_granger import engle_granger


def correlation_distance(prices: pd.DataFrame) -> NDArray:
    """Distance matrix d_ij = sqrt(2 (1 - rho_ij)) on RETURNS.

    Correlation is computed on returns (stationary), never on prices (the
    classic spurious-correlation trap). d in [0, 2]: perfectly correlated => 0.
    """
    rets = prices.pct_change().dropna()
    corr = rets.corr().values.copy()
    np.fill_diagonal(corr, 1.0)
    dist = np.sqrt(np.clip(2.0 * (1.0 - corr), 0.0, None))
    np.fill_diagonal(dist, 0.0)
    return dist


def cluster_assets(prices: pd.DataFrame, n_clusters: int = 4) -> dict[int, list[str]]:
    """Hierarchical clustering on the correlation distance — the economic
    pre-filter. We only test cointegration WITHIN clusters, cutting the test
    count by roughly a factor of n_clusters and removing economically
    nonsensical pairs (e.g. a miner vs a bank) before they can be false
    discoveries.
    """
    names = list(prices.columns)
    dist = correlation_distance(prices)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    clusters: dict[int, list[str]] = {}
    for name, lab in zip(names, labels):
        clusters.setdefault(int(lab), []).append(name)
    return clusters


def benjamini_hochberg(pvalues: NDArray, fdr: float = 0.10) -> NDArray:
    """Benjamini-Hochberg step-up. Returns a boolean mask of discoveries that
    control the false-discovery rate at level `fdr`.

    Controlling FDR (expected proportion of false positives among discoveries)
    is the right tool here, not family-wise error (Bonferroni) — we can tolerate
    a few bad pairs as long as the discovered set is mostly real, and we keep
    far more power than Bonferroni would on 20k tests.
    """
    pvalues = np.asarray(pvalues, dtype=float)
    m = len(pvalues)
    if m == 0:
        return np.array([], dtype=bool)
    order = np.argsort(pvalues)
    ranked = pvalues[order]
    thresholds = fdr * (np.arange(1, m + 1) / m)
    passed = ranked <= thresholds
    mask = np.zeros(m, dtype=bool)
    if passed.any():
        kmax = np.max(np.where(passed)[0])  # largest rank that passes
        mask[order[: kmax + 1]] = True
    return mask


@dataclass
class PairCandidate:
    a: str
    b: str
    pvalue: float
    hedge_ratio: float
    half_life: float
    cluster: int


@dataclass
class ScreenResult:
    candidates: list[PairCandidate]
    n_tested: int
    n_raw_significant: int
    n_after_fdr: int
    clusters: dict[int, list[str]] = field(default_factory=dict)


def screen_universe(
    prices: pd.DataFrame,
    n_clusters: int = 4,
    fdr: float = 0.10,
    significance: float = 0.05,
) -> ScreenResult:
    """Full Stage-3 screen: cluster -> test within clusters -> FDR-correct.

    Returns only the FDR-surviving pairs as ranked candidates. The counts
    (tested / raw-significant / after-FDR) make the multiple-testing problem
    explicit so you can see how many "significant" pairs FDR discarded.
    """
    clusters = cluster_assets(prices, n_clusters=n_clusters)

    raw: list[PairCandidate] = []
    pvals: list[float] = []
    for cid, members in clusters.items():
        for a, b in combinations(members, 2):
            res = engle_granger(prices[a].values, prices[b].values,
                                significance=significance, y_name=a, x_name=b)
            raw.append(PairCandidate(a, b, res.pvalue, res.hedge_ratio,
                                     res.half_life, cid))
            pvals.append(res.pvalue)

    n_tested = len(raw)
    n_raw_sig = int(np.sum(np.array(pvals) < significance)) if pvals else 0

    mask = benjamini_hochberg(np.array(pvals), fdr=fdr) if pvals else np.array([], bool)
    survivors = [c for c, keep in zip(raw, mask) if keep]
    survivors.sort(key=lambda c: c.pvalue)

    return ScreenResult(
        candidates=survivors,
        n_tested=n_tested,
        n_raw_significant=n_raw_sig,
        n_after_fdr=len(survivors),
        clusters=clusters,
    )


def johansen_rank(prices: pd.DataFrame, det_order: int = 0, k_ar_diff: int = 1) -> int:
    """Johansen trace-test cointegration rank of a multi-asset system.

    rank r = number of independent cointegrating relationships. r=0 => no
    cointegration; r>=1 => tradeable basket(s) exist. Symmetric in the assets
    (unlike Engle-Granger). Compares trace statistics to the 95% critical
    values returned by statsmodels.
    """
    arr = prices.values
    res = coint_johansen(arr, det_order, k_ar_diff)
    trace_stats = res.lr1            # trace statistics
    crit_95 = res.cvt[:, 1]          # 95% critical values
    rank = int(np.sum(trace_stats > crit_95))
    return rank
