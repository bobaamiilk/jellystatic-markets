"""PriceData — point-in-time CSV loader for APAC daily close prices.

Aligns dates across tickers, forward-fills gaps, and exposes an as_of() guard
that returns only data available up to a given date. Everything downstream uses
as_of() so the look-ahead guarantee propagates automatically.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


class PriceData:
    """Load and align daily close prices from a directory of per-ticker CSVs.

    CSV format: index column is a date (YYYY-MM-DD), one column per price field.
    The `price_col` argument picks which column to use; if absent the first column
    is taken. Missing dates are forward-filled (flagged via `is_filled`).
    """

    def __init__(
        self,
        csv_dir: str | Path,
        tickers: list[str],
        price_col: str = "close",
    ) -> None:
        csv_dir = Path(csv_dir)
        frames: dict[str, pd.Series] = {}
        missing: list[str] = []

        for ticker in tickers:
            path = csv_dir / f"{ticker}.csv"
            if not path.exists():
                missing.append(ticker)
                continue
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            col = price_col if price_col in df.columns else df.columns[0]
            frames[ticker] = df[col].rename(ticker)

        if missing:
            raise FileNotFoundError(
                f"CSV files not found for tickers: {missing} in {csv_dir}"
            )
        if not frames:
            raise ValueError("No tickers provided.")

        combined = pd.DataFrame(frames).sort_index()
        self._is_filled: pd.DataFrame = combined.isna()
        self._prices: pd.DataFrame = combined.ffill().dropna(how="all")

    # ---------------------------------------------------------------------- #

    def as_of(self, date: str | pd.Timestamp) -> pd.DataFrame:
        """Return all price data available strictly up to `date` (inclusive).

        This is the look-ahead guard: downstream callers pass `as_of(today)` so
        they can never accidentally consume future prices.
        """
        return self._prices.loc[: pd.Timestamp(date)].copy()

    @property
    def prices(self) -> pd.DataFrame:
        """Full aligned price frame (use as_of() in live/backtest contexts)."""
        return self._prices.copy()

    @property
    def is_filled(self) -> pd.DataFrame:
        """Boolean mask — True where a value was forward-filled from a prior day."""
        return self._is_filled.copy()

    @property
    def tickers(self) -> list[str]:
        return list(self._prices.columns)

    @property
    def start(self) -> pd.Timestamp:
        return self._prices.index[0]

    @property
    def end(self) -> pd.Timestamp:
        return self._prices.index[-1]

    def __len__(self) -> int:
        return len(self._prices)
