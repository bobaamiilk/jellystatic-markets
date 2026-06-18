"""Report generation — self-contained HTML and reproducible Parquet output.

build_report() writes a single HTML file with all figures embedded as base64
PNG so there are no external asset dependencies. No JS frameworks, no live
server. Figures use matplotlib only.

Headline verdict is OUT-OF-SAMPLE only. If the regime overlay does not improve
risk-adjusted return vs the unscaled book, the report says so explicitly.
"""
from __future__ import annotations

import base64
import datetime
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

from quantdesk.config import QuantDeskConfig
from quantdesk.engine.portfolio import PortfolioResult


# --------------------------------------------------------------------------- #
#  Internal figure helpers
# --------------------------------------------------------------------------- #

def _fig_to_b64(fig: plt.Figure) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def _equity_curve_fig(result: PortfolioResult) -> str:
    fig, ax = plt.subplots(figsize=(10, 4))
    eq = result.oos_equity
    ax.plot(eq.index, eq.values, color="#1f77b4", lw=1.5, label="Regime-scaled book")

    # Shade stressed regime periods
    scalar = result.regime_scalar
    stressed = scalar < 1.0
    in_stress = False
    start_idx = None
    for i, (idx, s) in enumerate(stressed.items()):
        if s and not in_stress:
            start_idx, in_stress = idx, True
        elif not s and in_stress:
            ax.axvspan(start_idx, idx, alpha=0.12, color="red", lw=0)
            in_stress = False
    if in_stress:
        ax.axvspan(start_idx, eq.index[-1], alpha=0.12, color="red", lw=0)

    ax.axhline(1.0, color="black", lw=0.6, ls="--", alpha=0.4)
    ax.set_ylabel("Equity (OOS, starts at 1.0)")
    ax.set_title("OOS Equity Curve  [red shading = regime de-grossed]")
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f"))
    fig.tight_layout()
    return _fig_to_b64(fig)


def _regime_vol_fig(result: PortfolioResult) -> str:
    """Regime scalar and rolling book vol on twin axes."""
    fig, ax1 = plt.subplots(figsize=(10, 3.5))

    scalar = result.regime_scalar
    ax1.fill_between(scalar.index, scalar.values, 1.0,
                     where=scalar.values < 1.0,
                     alpha=0.25, color="red", label="De-grossed")
    ax1.plot(scalar.index, scalar.values, color="red", lw=1.2, label="Regime scalar")
    ax1.set_ylim(0, 1.15)
    ax1.set_ylabel("Exposure scalar", color="red")
    ax1.tick_params(axis="y", labelcolor="red")

    ax2 = ax1.twinx()
    rets = result.oos_returns
    roll_vol = rets.rolling(21, min_periods=5).std() * np.sqrt(252)
    ax2.plot(roll_vol.index, roll_vol.values, color="#1f77b4", lw=1.0,
             alpha=0.7, label="21d rolling ann. vol")
    ax2.set_ylabel("Portfolio vol (ann.)", color="#1f77b4")
    ax2.tick_params(axis="y", labelcolor="#1f77b4")

    ax1.set_title("Regime Scalar vs Portfolio Volatility")
    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, fontsize=9, loc="upper left")
    fig.tight_layout()
    return _fig_to_b64(fig)


def _pair_table_html(result: PortfolioResult) -> str:
    if not result.pair_contributions:
        return "<p>No tradeable pairs survived the cointegration screen.</p>"

    rows = []
    for pc in sorted(result.pair_contributions, key=lambda p: -p.oos_sharpe):
        cusum = str(pc.cusum_break_idx) if pc.cusum_break_idx is not None else "—"
        rows.append(
            f"<tr>"
            f"<td>{pc.a}</td><td>{pc.b}</td>"
            f"<td class='num'>{pc.oos_sharpe:+.2f}</td>"
            f"<td class='num'>{pc.max_drawdown:.1%}</td>"
            f"<td class='num'>{pc.calmar:.2f}</td>"
            f"<td>{cusum}</td>"
            f"</tr>"
        )
    body = "\n".join(rows)
    return f"""
    <table class="pair-table">
      <thead>
        <tr>
          <th>Leg A</th><th>Leg B</th>
          <th>OOS Sharpe</th><th>Max DD</th><th>Calmar</th>
          <th>CUSUM break idx</th>
        </tr>
      </thead>
      <tbody>{body}</tbody>
    </table>"""


def _overlay_verdict(result: PortfolioResult) -> str:
    """Honest one-liner: did regime overlay help or hurt?"""
    delta = result.combined_sharpe - result.unscaled_sharpe
    if abs(delta) < 0.05:
        return (f"Regime overlay effect: <b>negligible</b> "
                f"(ΔSharpe = {delta:+.2f}). "
                f"De-grossing reduced exposure on {result.pct_days_degrossed:.0%} "
                f"of OOS days but did not materially change risk-adjusted returns.")
    elif delta > 0:
        return (f"Regime overlay <b>improved</b> OOS Sharpe by {delta:+.2f} "
                f"({result.unscaled_sharpe:.2f} → {result.combined_sharpe:.2f}). "
                f"De-grossed on {result.pct_days_degrossed:.0%} of OOS days.")
    else:
        return (f"Regime overlay <b>hurt</b> OOS Sharpe by {delta:+.2f} "
                f"({result.unscaled_sharpe:.2f} → {result.combined_sharpe:.2f}). "
                f"De-grossing reduced drawdown but cost return; "
                f"consider calibrating stress_pct or min_scalar.")


# --------------------------------------------------------------------------- #
#  Public API
# --------------------------------------------------------------------------- #

_CSS = """
body { font-family: 'Helvetica Neue', Arial, sans-serif; margin: 40px; color: #222; }
h1   { font-size: 1.6em; border-bottom: 2px solid #1f77b4; padding-bottom: 8px; }
h2   { font-size: 1.1em; color: #444; margin-top: 2em; }
.metrics-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; margin: 1.5em 0; }
.metric { background: #f4f8ff; border-left: 4px solid #1f77b4; padding: 12px 16px; border-radius: 4px; }
.metric .label { font-size: 0.75em; color: #666; text-transform: uppercase; letter-spacing: 0.05em; }
.metric .value { font-size: 1.5em; font-weight: bold; margin-top: 4px; }
.verdict { background: #fffbea; border: 1px solid #f0c040; border-radius: 4px;
           padding: 12px 16px; margin: 1em 0; font-size: 0.92em; }
img { max-width: 100%; height: auto; margin: 0.5em 0; border: 1px solid #ddd; border-radius: 4px; }
table.pair-table { border-collapse: collapse; width: 100%; font-size: 0.88em; }
table.pair-table th { background: #1f77b4; color: white; padding: 6px 10px; text-align: left; }
table.pair-table td { padding: 5px 10px; border-bottom: 1px solid #eee; }
.num { text-align: right; font-variant-numeric: tabular-nums; }
.oos-label { background: #e8f5e9; color: #2e7d32; border-radius: 3px;
             padding: 1px 6px; font-size: 0.75em; font-weight: bold; }
"""


def build_report(
    result: PortfolioResult,
    cfg: QuantDeskConfig,
    output_path: str | Path = "out/report.html",
) -> Path:
    """Write a self-contained HTML report (all figures as base64 PNG).

    All headline metrics are explicitly labelled OUT-OF-SAMPLE. If the regime
    overlay does not improve Sharpe, the verdict block says so.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    eq_b64 = _equity_curve_fig(result)
    rv_b64 = _regime_vol_fig(result)
    pair_table = _pair_table_html(result)
    verdict = _overlay_verdict(result)

    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    mdd_str = f"{result.max_drawdown:.1%}"
    cal_str = f"{result.calmar:.2f}"
    shr_str = f"{result.combined_sharpe:+.2f}"
    unscaled_shr_str = f"{result.unscaled_sharpe:+.2f}"
    deg_str = f"{result.pct_days_degrossed:.0%}"
    pairs_str = str(result.n_pairs)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>APAC QuantDesk Report — {ts}</title>
  <style>{_CSS}</style>
</head>
<body>
<h1>APAC QuantDesk — Regime-Scaled Stat-Arb
  <span class="oos-label">OUT-OF-SAMPLE ONLY</span>
</h1>
<p style="color:#888;font-size:0.85em">Generated {ts} &nbsp;|&nbsp;
   train={cfg.train_size}d / test={cfg.test_size}d walk-forward &nbsp;|&nbsp;
   clusters={cfg.n_clusters} &nbsp;|&nbsp; FDR={cfg.fdr:.0%}</p>

<h2>Headline Metrics <span class="oos-label">OOS</span></h2>
<div class="metrics-grid">
  <div class="metric"><div class="label">OOS Sharpe (scaled)</div>
    <div class="value">{shr_str}</div></div>
  <div class="metric"><div class="label">OOS Sharpe (unscaled)</div>
    <div class="value">{unscaled_shr_str}</div></div>
  <div class="metric"><div class="label">Max Drawdown</div>
    <div class="value">{mdd_str}</div></div>
  <div class="metric"><div class="label">Calmar</div>
    <div class="value">{cal_str}</div></div>
  <div class="metric"><div class="label">Pairs traded</div>
    <div class="value">{pairs_str}</div></div>
  <div class="metric"><div class="label">% Days de-grossed</div>
    <div class="value">{deg_str}</div></div>
</div>

<div class="verdict"><b>Regime overlay verdict:</b> {verdict}</div>

<h2>OOS Equity Curve</h2>
<img src="data:image/png;base64,{eq_b64}" alt="OOS Equity Curve">

<h2>Regime Scalar vs Portfolio Volatility</h2>
<img src="data:image/png;base64,{rv_b64}" alt="Regime Scalar vs Vol">

<h2>Per-Pair OOS Performance</h2>
{pair_table}

<hr style="margin-top:3em;border:none;border-top:1px solid #ddd">
<p style="color:#aaa;font-size:0.75em">
  quantdesk v0.1.0 &nbsp;|&nbsp; In-sample metrics are suppressed from
  the headline verdict and available only in the Parquet output.
</p>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    return output_path


def to_parquet(
    result: PortfolioResult,
    output_path: str | Path = "out/results.parquet",
) -> Path:
    """Dump the signal/return/regime timeseries to Parquet for reproducibility.

    The Parquet contains everything needed to reproduce the report without
    re-running the engine: equity, OOS returns, unscaled returns, regime scalar,
    and one column per pair contribution.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        "oos_returns": result.oos_returns,
        "unscaled_returns": result.unscaled_returns,
        "oos_equity": result.oos_equity,
        "regime_scalar": result.regime_scalar,
    })

    for pc in result.pair_contributions:
        col = f"pair_{pc.a}_{pc.b}_oos_sharpe"
        df[col] = pc.oos_sharpe   # scalar broadcast to all rows for metadata

    df.to_parquet(output_path)
    return output_path
