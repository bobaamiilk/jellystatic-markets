"""Tests for the quantdesk data layer."""
import numpy as np
import pandas as pd
import pytest

from quantdesk.data.synthetic import make_synthetic_prices, make_synthetic_market_returns
from quantdesk.data.events import load_events, events_in_range, MacroEvent
from quantdesk.data.loader import PriceData


# --------------------------------------------------------------------------- #
#  Synthetic data
# --------------------------------------------------------------------------- #

def test_synthetic_prices_shape():
    prices, true_pairs = make_synthetic_prices(n=300, n_clusters=2, per_cluster=3, seed=0)
    assert prices.shape == (300, 6)
    assert len(true_pairs) > 0
    assert (prices.values > 0).all()


def test_synthetic_prices_reproducible():
    p1, _ = make_synthetic_prices(n=200, seed=7)
    p2, _ = make_synthetic_prices(n=200, seed=7)
    assert p1.equals(p2)


def test_synthetic_market_returns():
    rets, var = make_synthetic_market_returns(n=500, seed=0)
    assert len(rets) == 500
    assert (var > 0).all()
    assert np.isfinite(rets).all()


# --------------------------------------------------------------------------- #
#  Events
# --------------------------------------------------------------------------- #

def test_load_default_events():
    events = load_events()
    assert len(events) > 0
    assert all(isinstance(e, MacroEvent) for e in events)


def test_events_in_range():
    events = load_events()
    subset = events_in_range(events, "2024-07-01", "2024-09-30")
    assert all("2024-07" <= e.date <= "2024-09-30" for e in subset)


def test_load_events_from_yaml(tmp_path):
    import yaml
    data = {"events": [
        {"date": "2025-01-01", "label": "Test event"},
        {"date": "2025-06-01", "label": "Another event"},
    ]}
    p = tmp_path / "events.yaml"
    p.write_text(yaml.dump(data))
    events = load_events(p)
    assert len(events) == 2
    assert events[0].label == "Test event"


# --------------------------------------------------------------------------- #
#  PriceData loader
# --------------------------------------------------------------------------- #

def _write_csv_prices(tmp_path, ticker: str, prices: pd.Series) -> None:
    df = pd.DataFrame({"close": prices})
    df.index.name = "date"
    df.to_csv(tmp_path / f"{ticker}.csv")


def test_price_data_loads_and_aligns(tmp_path):
    idx = pd.date_range("2020-01-02", periods=100, freq="B")
    _write_csv_prices(tmp_path, "A", pd.Series(100 + np.arange(100, dtype=float), index=idx))
    _write_csv_prices(tmp_path, "B", pd.Series(50 + np.arange(100, dtype=float), index=idx))

    pd_ = PriceData(tmp_path, ["A", "B"])
    assert pd_.prices.shape == (100, 2)
    assert list(pd_.prices.columns) == ["A", "B"]


def test_price_data_as_of_guard(tmp_path):
    idx = pd.date_range("2020-01-02", periods=50, freq="B")
    _write_csv_prices(tmp_path, "X", pd.Series(np.arange(50, dtype=float) + 10, index=idx))
    pd_ = PriceData(tmp_path, ["X"])

    cutoff = idx[24]
    subset = pd_.as_of(cutoff)
    assert len(subset) == 25
    assert subset.index[-1] == cutoff


def test_price_data_missing_ticker_raises(tmp_path):
    idx = pd.date_range("2020-01-02", periods=10, freq="B")
    _write_csv_prices(tmp_path, "A", pd.Series(np.ones(10), index=idx))
    with pytest.raises(FileNotFoundError):
        PriceData(tmp_path, ["A", "MISSING"])
