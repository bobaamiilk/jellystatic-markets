"""End-to-end volatility-forecasting demo (run directly).

Simulates APAC-index-like returns with KNOWN volatility, then walks the full
pipeline: fit GARCH(1,1) and EGARCH (leverage), compare GARCH vs EWMA vs rolling
out-of-sample with QLIKE + a Diebold-Mariano significance test, run vol targeting,
and derive the regime exposure signal that would govern the stat-arb book.

    python -m volforecast.examples.run_pipeline
"""
from __future__ import annotations

import numpy as np

from volforecast.models import fit_garch, fit_egarch, ewma_variance
from volforecast.evaluation import walk_forward_eval
from volforecast.trading import apply_vol_target, VolTargetConfig
from volforecast.regime import vol_percentile_regime
from volforecast.utils import simulate_garch, simulate_regime_returns


def main() -> None:
    print("=" * 64)
    print("VOLATILITY FORECASTING — END-TO-END PIPELINE (synthetic APAC index)")
    print("=" * 64)

    # 1) Simulate GARCH returns with known parameters (truth for validation).
    r, true_var = simulate_garch(n=3000, omega=2e-6, alpha=0.08, beta=0.90, seed=0)
    print(f"\n[1] Simulated {len(r)} daily returns. True params: "
          f"alpha=0.08, beta=0.90, persistence=0.98, "
          f"long-run vol={np.sqrt(2e-6/(1-0.98))*100:.2f}%/day.")

    # 2) Fit GARCH(1,1) and EGARCH.
    g = fit_garch(r)
    eg = fit_egarch(r)
    print(f"\n[2] GARCH(1,1) MLE: alpha={g.alpha:.3f}, beta={g.beta:.3f}, "
          f"persistence={g.persistence:.3f}")
    print(f"      long-run vol = {g.long_run_vol*100:.2f}%/day, "
          f"vol-shock half-life = {g.vol_shock_half_life:.0f} days")
    print(f"    EGARCH leverage gamma = {eg.gamma:+.3f} "
          f"({'leverage present' if eg.has_leverage else 'no leverage'})")

    # 3) Walk-forward OOS comparison with Diebold-Mariano.
    wf = walk_forward_eval(r, train_size=500, test_size=21)
    print(f"\n[3] Walk-forward OOS QLIKE (lower=better):")
    for k, v in sorted(wf.qlike.items(), key=lambda kv: kv[1]):
        print(f"      {k:8s}: {v:+.4f}")
    dm = wf.dm_garch_vs_ewma
    verdict = {"A": "GARCH significantly better",
               "B": "EWMA significantly better",
               "neither": "no significant difference (don't pay for GARCH)"}[dm.better]
    print(f"    Diebold-Mariano (GARCH vs EWMA): stat={dm.statistic:+.2f}, "
          f"p={dm.pvalue:.3f} -> {verdict}")

    # 4) Vol targeting using the GARCH conditional vol.
    sigma_hat = np.sqrt(g.conditional_var)
    vt = apply_vol_target(r, sigma_hat,
                          VolTargetConfig(target_vol=0.10, max_leverage=3.0))
    print(f"\n[4] Vol targeting (target 10%/yr, cap 3x):")
    print(f"      realised vol  = {vt.realised_vol*100:.1f}%/yr "
          f"(target 10.0%)")
    print(f"      gross turnover= {vt.gross_turnover:.1f}, "
          f"cap bound {vt.leverage_capped_frac*100:.1f}% of days")

    # 5) Regime signal (would de-gross the stat-arb book — Project 2 handoff).
    rr, state = simulate_regime_returns(n=1500, seed=3)
    rv = np.sqrt(ewma_variance(rr, lam=0.94))
    reg = vol_percentile_regime(rv, lookback=252, stress_pct=0.80, min_scalar=0.25)
    print(f"\n[5] Regime signal on a 2-state series: "
          f"{reg.frac_stressed*100:.0f}% of days flagged stressed; "
          f"min exposure scalar applied = {reg.exposure_scalar.min():.2f}")
    print("    -> this scalar multiplies stat-arb gross exposure (cross-system link).")

    # 6) Plot.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(3, 1, figsize=(10, 9))
        ann = np.sqrt(252)
        ax[0].plot(np.sqrt(true_var) * ann, label="true vol", color="grey", lw=1)
        ax[0].plot(np.sqrt(g.conditional_var) * ann, label="GARCH", color="#35D0C0", lw=1)
        ax[0].plot(np.sqrt(ewma_variance(r)) * ann, label="EWMA", color="#F2B137", lw=0.8, alpha=0.8)
        ax[0].set_title("Annualised volatility: true vs GARCH vs EWMA")
        ax[0].legend(loc="upper right")

        models = list(wf.qlike.keys())
        vals = [wf.qlike[m] for m in models]
        ax[1].bar(models, vals, color=["#35D0C0", "#F2B137", "#A78BFA"])
        ax[1].set_title(f"OOS QLIKE by model (lower=better) — "
                        f"DM GARCH vs EWMA p={dm.pvalue:.2f}")

        ax[2].plot(reg.exposure_scalar, color="#E5484D", lw=1.2)
        ax[2].fill_between(range(len(reg.state)), 0, reg.state, color="#E5484D",
                           alpha=0.12, step="pre", label="stressed regime")
        ax[2].set_title("Regime exposure scalar (de-grosses stat-arb book in stress)")
        ax[2].set_ylim(0, 1.1)
        ax[2].set_xlabel("trading day")
        ax[2].legend(loc="lower left")
        fig.tight_layout()
        out = "reports/volforecast_pipeline.png"
        fig.savefig(out, dpi=130)
        print(f"\n[6] Saved plot -> {out}")
    except ImportError:
        print("\n[6] matplotlib not available; skipping plot.")

    print("\nDone.")


if __name__ == "__main__":
    main()
