"""Tests for the report builder."""
import pandas as pd
import numpy as np
import pytest

from quantdesk.config import QuantDeskConfig
from quantdesk.data.synthetic import make_synthetic_prices
from quantdesk.engine.portfolio import run_regime_scaled_statarb, PortfolioResult
from quantdesk.report.builder import build_report, to_parquet


_FAST_CFG = QuantDeskConfig(
    n_clusters=2,
    fdr=0.20,
    train_size=150,
    test_size=40,
    regime_lookback=120,
)


def _get_result() -> PortfolioResult:
    prices, _ = make_synthetic_prices(n=400, n_clusters=2, per_cluster=4, seed=99)
    return run_regime_scaled_statarb(prices, _FAST_CFG)


def test_build_report_creates_html(tmp_path):
    result = _get_result()
    out = build_report(result, _FAST_CFG, tmp_path / "report.html")
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert "<html" in content
    assert "data:image/png;base64," in content
    assert "OUT-OF-SAMPLE" in content


def test_report_embeds_all_sections(tmp_path):
    result = _get_result()
    out = build_report(result, _FAST_CFG, tmp_path / "report.html")
    content = out.read_text(encoding="utf-8")
    # Headline metrics block
    assert "OOS Sharpe" in content
    assert "Max Drawdown" in content
    assert "Calmar" in content
    # Regime overlay verdict
    assert "Regime overlay" in content
    # Per-pair table header
    assert "Leg A" in content


def test_report_contains_no_external_assets(tmp_path):
    result = _get_result()
    out = build_report(result, _FAST_CFG, tmp_path / "report.html")
    content = out.read_text(encoding="utf-8")
    # No <script src=...> or <link href=...> pointing to external URLs
    assert "http://" not in content
    assert "https://" not in content
    assert "<script" not in content


def test_to_parquet_creates_file(tmp_path):
    result = _get_result()
    p = to_parquet(result, tmp_path / "results.parquet")
    assert p.exists()
    df = pd.read_parquet(p)
    assert "oos_returns" in df.columns
    assert "oos_equity" in df.columns
    assert "regime_scalar" in df.columns
    assert "unscaled_returns" in df.columns


def test_parquet_index_matches_prices_index(tmp_path):
    prices, _ = make_synthetic_prices(n=400, n_clusters=2, per_cluster=4, seed=77)
    result = run_regime_scaled_statarb(prices, _FAST_CFG)
    p = to_parquet(result, tmp_path / "r.parquet")
    df = pd.read_parquet(p)
    assert len(df) == len(prices)


def test_to_parquet_is_deterministic(tmp_path):
    prices, _ = make_synthetic_prices(n=350, n_clusters=2, per_cluster=4, seed=5)
    r1 = run_regime_scaled_statarb(prices, _FAST_CFG)
    r2 = run_regime_scaled_statarb(prices, _FAST_CFG)
    p1 = to_parquet(r1, tmp_path / "a.parquet")
    p2 = to_parquet(r2, tmp_path / "b.parquet")
    assert pd.read_parquet(p1).equals(pd.read_parquet(p2))
