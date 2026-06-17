"""End-to-end pipeline demo: raw chain -> clean -> IV -> SVI surface -> plots.

Runs the entire Project 1 system on a synthetic APAC-style chain and produces:
  - smile curves (IV vs strike, per maturity)
  - term structure (ATM IV vs maturity)
  - 3D implied-vol surface
  - a console report of cleaning stats, fit RMSE, and arbitrage checks

This is both the integration test and the source of the visual deliverables.
Run:  python -m volsurface.examples.run_pipeline
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from volsurface.data import clean_chain
from volsurface.iv import implied_vol
from volsurface.surface import calibrate_svi_slice, VolSurface
from volsurface.utils.synthetic_data import generate_chain

F, R = 100.0, 0.04


def invert_chain_to_iv(clean: pd.DataFrame) -> pd.DataFrame:
    """Invert every cleaned mid to implied vol, carrying vega diagnostics."""
    records = []
    for _, row in clean.iterrows():
        res = implied_vol(
            price=row["mid"], F=F, K=row["strike"], tau=row["maturity"],
            r=R, option_type=row["option_type"],
        )
        if res.converged and np.isfinite(res.iv):
            records.append({
                "strike": row["strike"], "maturity": row["maturity"],
                "option_type": row["option_type"], "iv": res.iv,
                "vega": res.vega_at_solution, "log_moneyness": np.log(row["strike"] / F),
                "spread_weight": row.get("spread_weight", 1.0),
            })
    return pd.DataFrame(records)


def fit_surface(iv_df: pd.DataFrame) -> tuple[VolSurface, dict]:
    """Fit one SVI slice per maturity (OTM quotes only: puts left, calls right),
    vega-weighted. Returns the surface and per-slice fit diagnostics."""
    slices, diagnostics = [], {}
    for tau, g in iv_df.groupby("maturity"):
        # Use OTM wing of each type: puts for k<0, calls for k>=0 (standard).
        otm = g[
            ((g["option_type"] == "put") & (g["log_moneyness"] < 0))
            | ((g["option_type"] == "call") & (g["log_moneyness"] >= 0))
        ].sort_values("log_moneyness")
        k = otm["log_moneyness"].to_numpy()
        w = (otm["iv"].to_numpy() ** 2) * tau
        weights = otm["vega"].to_numpy() + 1e-6  # vega-weighting
        params, rmse = calibrate_svi_slice(k, w, tau=tau, weights=weights)
        slices.append(params)
        diagnostics[tau] = {
            "rmse_total_var": rmse,
            "butterfly_margin": params.butterfly_margin(),
            "n_points": len(k),
        }
    return VolSurface(slices), diagnostics


def run(make_plots: bool = True) -> dict:
    print("=" * 64)
    print("PROJECT 1 PIPELINE: chain -> clean -> IV -> SVI surface")
    print("=" * 64)

    # 1. Raw chain with injected noise
    raw = generate_chain(F=F, r=R, inject_noise=True)
    print(f"\n[1] Raw chain: {len(raw)} quotes")

    # 2. Clean + arbitrage filter
    clean, report = clean_chain(raw, F=F, r=R)
    print(f"[2] Cleaning: {report.summary()}")

    # 3. Invert to IV
    iv_df = invert_chain_to_iv(clean)
    print(f"[3] IV inversion: {len(iv_df)} quotes inverted successfully")
    print(f"    median vega = {iv_df['vega'].median():.3f}, "
          f"min vega = {iv_df['vega'].min():.4f} (wing quotes)")

    # 4. Fit SVI surface
    surface, diag = fit_surface(iv_df)
    print("[4] SVI calibration per maturity:")
    for tau, d in sorted(diag.items()):
        print(f"    tau={tau:.2f}: RMSE(w)={d['rmse_total_var']:.2e}  "
              f"butterfly_margin={d['butterfly_margin']:.3f}  "
              f"n={d['n_points']}")

    # 5. Arbitrage re-check on the FITTED surface (dense grid)
    k_dense = np.linspace(-0.4, 0.4, 200)
    arb = surface.is_arbitrage_free(k_dense)
    print("[5] Arbitrage checks on FITTED surface:")
    print(f"    calendar arbitrage-free: {arb['calendar']}")
    bf = all(arb["butterfly_per_slice"].values())
    print(f"    all slices butterfly arbitrage-free: {bf}")

    # 6. Recover known truth: compare fitted ATM IV to the true-surface ATM IV
    print("[6] Ground-truth recovery (ATM IV, fitted vs true):")
    true_atm = {0.08: None, 0.25: None, 0.50: None, 1.00: None}
    truth = raw.groupby("maturity")["true_iv"].apply(
        lambda s: s.iloc[(s.index % len(s)).argmin()]
    )
    for sl in surface.slices:
        fitted_atm = sl.implied_vol(np.array([0.0]))[0]
        # nearest true ATM iv from raw (strike==100 -> k=0)
        true_row = raw[(raw["maturity"] == sl.tau) & (raw["strike"] == 100.0)]
        true_atm_iv = true_row["true_iv"].iloc[0] if len(true_row) else np.nan
        print(f"    tau={sl.tau:.2f}: fitted={fitted_atm:.4f}  "
              f"true={true_atm_iv:.4f}  err={abs(fitted_atm-true_atm_iv):.2e}")

    if make_plots:
        _plot(iv_df, surface)
        print("\n[7] Plots written: smile.png, term_structure.png, surface_3d.png")

    print("\nPipeline complete.")
    return {"report": report, "diagnostics": diag, "arbitrage": arb,
            "surface": surface, "iv_df": iv_df}


def _plot(iv_df: pd.DataFrame, surface: VolSurface) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    maturities = sorted(iv_df["maturity"].unique())
    k_grid = np.linspace(-0.35, 0.35, 100)

    # Smile curves
    fig, ax = plt.subplots(figsize=(8, 5))
    for tau in maturities:
        g = iv_df[iv_df["maturity"] == tau].sort_values("log_moneyness")
        ax.scatter(g["log_moneyness"], g["iv"], s=12, alpha=0.5)
        sl = next(s for s in surface.slices if abs(s.tau - tau) < 1e-9)
        ax.plot(k_grid, sl.implied_vol(k_grid), label=f"tau={tau:.2f}")
    ax.set_xlabel("log-moneyness  k = ln(K/F)")
    ax.set_ylabel("implied volatility")
    ax.set_title("Volatility smile: SVI fit vs inverted market IV")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("smile.png", dpi=120)
    plt.close(fig)

    # Term structure (ATM)
    fig, ax = plt.subplots(figsize=(8, 5))
    atm_iv = [next(s for s in surface.slices if abs(s.tau - t) < 1e-9)
              .implied_vol(np.array([0.0]))[0] for t in maturities]
    ax.plot(maturities, atm_iv, "o-")
    ax.set_xlabel("maturity tau (years)")
    ax.set_ylabel("ATM implied volatility")
    ax.set_title("ATM term structure")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("term_structure.png", dpi=120)
    plt.close(fig)

    # 3D surface
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    T = np.linspace(min(maturities), max(maturities), 40)
    KK, TT = np.meshgrid(k_grid, T)
    IV = np.array([[surface.implied_vol(k, t) for k in k_grid] for t in T])
    ax.plot_surface(KK, TT, IV, cmap="viridis", alpha=0.9)
    ax.set_xlabel("log-moneyness")
    ax.set_ylabel("maturity")
    ax.set_zlabel("implied vol")
    ax.set_title("Implied volatility surface")
    fig.tight_layout()
    fig.savefig("surface_3d.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    run()
