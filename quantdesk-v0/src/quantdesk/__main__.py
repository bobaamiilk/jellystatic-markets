"""CLI entry point.

    python -m quantdesk run --config config.yaml
    python -m quantdesk run --synthetic

Writes report.html and results.parquet to out/ and prints headline metrics.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def _run(args: argparse.Namespace) -> None:
    from quantdesk.config import QuantDeskConfig
    from quantdesk.engine.portfolio import run_regime_scaled_statarb
    from quantdesk.report.builder import build_report, to_parquet

    if args.synthetic:
        from quantdesk.data.synthetic import make_synthetic_prices
        cfg = QuantDeskConfig()
        print("quantdesk: generating synthetic APAC universe …")
        prices, true_pairs = make_synthetic_prices(
            n=750, n_clusters=cfg.n_clusters, per_cluster=4, seed=cfg.seed
        )
        print(f"  {len(prices.columns)} tickers, {len(prices)} days, "
              f"{len(true_pairs)} known cointegrated pairs")
    elif args.config:
        cfg = QuantDeskConfig.from_yaml(args.config)
        if not cfg.tickers:
            print("quantdesk: config.tickers is empty — falling back to synthetic data.")
            from quantdesk.data.synthetic import make_synthetic_prices
            prices, _ = make_synthetic_prices(
                n=750, n_clusters=cfg.n_clusters, per_cluster=4, seed=cfg.seed
            )
        else:
            from quantdesk.data.loader import PriceData
            csv_dir = args.data_dir or Path(args.config).parent / "data"
            loader = PriceData(csv_dir, cfg.tickers)
            prices = loader.prices
            print(f"quantdesk: loaded {len(cfg.tickers)} tickers from {csv_dir}")
    else:
        print("error: specify --config <path> or --synthetic", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out)
    print(f"quantdesk: running pipeline ({len(prices)} bars, "
          f"train={cfg.train_size}d / test={cfg.test_size}d) …")

    t0 = time.perf_counter()
    result = run_regime_scaled_statarb(prices, cfg)
    elapsed = time.perf_counter() - t0

    html_path = build_report(result, cfg, out_dir / "report.html")
    pq_path = to_parquet(result, out_dir / "results.parquet")

    # --- Headline metrics to stdout (OOS only) ---
    print("\n" + "=" * 56)
    print("  APAC QuantDesk — OOS Results")
    print("=" * 56)
    print(f"  Pairs traded          : {result.n_pairs}")
    print(f"  OOS Sharpe (scaled)   : {result.combined_sharpe:+.3f}")
    print(f"  OOS Sharpe (unscaled) : {result.unscaled_sharpe:+.3f}")
    print(f"  Max drawdown          : {result.max_drawdown:.1%}")
    print(f"  Calmar ratio          : {result.calmar:.3f}")
    print(f"  % days de-grossed     : {result.pct_days_degrossed:.0%}")
    print(f"  CUSUM breaks flagged  : {len(result.cusum_break_flags)}")
    print(f"  Runtime               : {elapsed:.1f}s")
    print("=" * 56)
    print(f"\n  Report : {html_path}")
    print(f"  Parquet: {pq_path}\n")

    if result.n_pairs == 0:
        print("WARNING: no tradeable pairs survived the screen. "
              "Try loosening fdr or reducing n_clusters.", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="quantdesk",
        description="APAC regime-scaled stat-arb pipeline",
    )
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="Run the full pipeline")
    run_p.add_argument(
        "--config", type=Path, metavar="PATH",
        help="Path to config.yaml",
    )
    run_p.add_argument(
        "--synthetic", action="store_true",
        help="Run on generated synthetic data (no input files needed)",
    )
    run_p.add_argument(
        "--data-dir", type=Path, metavar="DIR",
        help="Directory of per-ticker CSVs (default: <config_dir>/data/)",
    )
    run_p.add_argument(
        "--out", default="out", metavar="DIR",
        help="Output directory for report.html and results.parquet (default: out/)",
    )

    args = parser.parse_args()

    if args.command == "run":
        _run(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
