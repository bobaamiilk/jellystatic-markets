"""quantdesk — regime-scaled APAC stat-arb orchestration layer.

Wires volsurface (IV/skew context), statarb (cointegration screen +
walk-forward backtest), and volforecast (GARCH vol forecast + regime
de-grossing) into a single, reproducible pipeline.

Entry points:
    quantdesk.config      — QuantDeskConfig (pydantic, loads from YAML)
    quantdesk.data        — PriceData (CSV loader with as_of guard), synthetic fallback
    quantdesk.engine      — run_regime_scaled_statarb, PortfolioResult
    quantdesk.report      — build_report (self-contained HTML), to_parquet
"""
__version__ = "0.1.0"
