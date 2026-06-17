"""statarb — statistical arbitrage engine (APAC focus).

Pipeline: screen a universe for cointegration (with FDR control) -> estimate a
time-varying hedge ratio (Kalman) -> build vol-adjusted z-score signals ->
walk-forward backtest with t+1 execution and realistic costs -> monitor for
structural breaks. Validated against synthetic data with known ground truth.
"""
__version__ = "0.1.0"
