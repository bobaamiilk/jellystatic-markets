"""End-to-end statistical-arbitrage demo (run directly).

Mirrors Project 1's run_pipeline: build a synthetic APAC-style universe with
KNOWN cointegration structure, run the full screen (cluster -> test -> FDR),
then walk-forward backtest the strongest surviving pair and plot the result.

    python -m statarb.examples.run_pipeline
"""
from __future__ import annotations

import numpy as np

from statarb.cointegration import engle_granger, is_tradeable, screen_universe
from statarb.backtest import backtest_pair, CostModel
from statarb.signals import SignalConfig
from statarb.utils import make_universe


def main() -> None:
    print("=" * 64)
    print("STATISTICAL ARBITRAGE — END-TO-END PIPELINE (synthetic APAC universe)")
    print("=" * 64)

    # 1) Universe with known sector structure (3 sectors x 4 names).
    prices, true_pairs = make_universe(n=1000, n_clusters=3, per_cluster=4, seed=2)
    print(f"\n[1] Universe: {prices.shape[1]} assets, {prices.shape[0]} days, "
          f"{len(true_pairs)} truly-cointegrated within-sector pairs.")

    # 2) Screen: cluster -> within-cluster EG tests -> FDR control.
    screen = screen_universe(prices, n_clusters=3, fdr=0.10)
    print(f"\n[2] Screen: tested {screen.n_tested} pairs, "
          f"{screen.n_raw_significant} raw-significant, "
          f"{screen.n_after_fdr} survive FDR(10%).")
    print("    Clusters found:")
    for cid, members in screen.clusters.items():
        print(f"      cluster {cid}: {sorted(members)}")
    found = {frozenset((c.a, c.b)) for c in screen.candidates}
    truth = {frozenset(p) for p in true_pairs}
    print(f"    Recovered {len(found & truth)}/{len(truth)} true pairs; "
          f"{len(found - truth)} false discoveries.")

    if not screen.candidates:
        print("No tradeable candidates; exiting.")
        return

    # 3) Tradeability filter on the best candidate.
    best = screen.candidates[0]
    res = engle_granger(np.log(prices[best.a].values), np.log(prices[best.b].values),
                        y_name=best.a, x_name=best.b)
    ok, reason = is_tradeable(res)
    print(f"\n[3] Best pair {best.a}~{best.b}: p={best.pvalue:.4f}, "
          f"hedge={res.hedge_ratio:.3f}, half-life={res.half_life:.1f}d -> {reason}")

    # 4) Walk-forward backtest with realistic costs.
    bt = backtest_pair(
        np.log(prices[best.a].values), np.log(prices[best.b].values),
        train_size=252, test_size=63, use_kalman=True,
        cfg=SignalConfig(entry=2.0, exit=0.5, stop=4.0, max_holding=30),
        cost_model=CostModel(commission_bps=1.0, half_spread_bps=2.0, impact_bps=0.5),
    )
    print(f"\n[4] Walk-forward backtest (frozen params, t+1 execution, costs on):")
    print(f"      OOS Sharpe     : {bt.oos_sharpe:6.2f}   (in-sample {bt.is_sharpe:.2f} — sanity only)")
    print(f"      Total return   : {bt.total_return*100:6.1f}%")
    print(f"      Max drawdown   : {bt.max_drawdown*100:6.1f}%")
    print(f"      Calmar         : {bt.calmar:6.2f}")
    print(f"      Trades         : {bt.n_trades}")

    # 5) Plot (optional — needs matplotlib).
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        ax[0].plot(np.log(prices[best.a].values), label=best.a, lw=1)
        ax[0].plot(np.log(prices[best.b].values), label=best.b, lw=1)
        ax[0].set_title(f"Log prices: {best.a} vs {best.b}")
        ax[0].legend(loc="upper left")

        ax[1].plot(bt.zscore, color="#F2B137", lw=1)
        for lvl, ls in [(2, "--"), (-2, "--"), (4, ":"), (-4, ":")]:
            ax[1].axhline(lvl, color="grey", ls=ls, lw=0.7)
        ax[1].set_title("OOS spread z-score (entry +/-2, stop +/-4)")

        ax[2].plot(bt.oos_equity, color="#35D0C0", lw=1.5)
        ax[2].axhline(1.0, color="grey", lw=0.7)
        ax[2].set_title(f"OOS equity curve (Sharpe {bt.oos_sharpe:.2f}, "
                        f"MaxDD {bt.max_drawdown*100:.1f}%)")
        ax[2].set_xlabel("trading day")
        fig.tight_layout()
        out = "reports/statarb_backtest.png"
        fig.savefig(out, dpi=130)
        print(f"\n[5] Saved plot -> {out}")
    except ImportError:
        print("\n[5] matplotlib not available; skipping plot.")

    print("\nDone.")


if __name__ == "__main__":
    main()
