"""Unified configuration for the quantdesk pipeline.

A single pydantic model captures every knob: universe/tickers, walk-forward
windows, signal thresholds, cost model, vol targeting, and regime de-grossing
parameters. Load from config.yaml; validated on construction so the engine
never sees invalid state.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


class SignalCfg(BaseModel):
    """Maps to statarb.signals.SignalConfig — keep field names identical."""
    entry: float = 2.0
    exit: float = 0.5
    stop: float = 4.0
    max_holding: int = 30


class CostCfg(BaseModel):
    """Maps to statarb.backtest.CostModel."""
    commission_bps: float = 1.0
    half_spread_bps: float = 2.0
    impact_bps: float = 0.5


class VolTargetCfg(BaseModel):
    """Maps to volforecast.trading.VolTargetConfig."""
    target_vol: float = 0.10
    max_leverage: float = 3.0
    vol_floor: float = 0.03
    rebalance_band: float = 0.10


class QuantDeskConfig(BaseModel):
    # Universe
    tickers: list[str] = Field(default_factory=list)
    n_clusters: int = Field(3, gt=0)
    fdr: float = Field(0.10, gt=0, lt=1)

    # Walk-forward windows (trading days)
    train_size: int = Field(252, gt=0)
    test_size: int = Field(63, gt=0)

    # Sub-model configs
    signal: SignalCfg = Field(default_factory=SignalCfg)
    cost: CostCfg = Field(default_factory=CostCfg)
    vol_target: VolTargetCfg = Field(default_factory=VolTargetCfg)

    # Regime de-grossing
    regime_lookback: int = Field(252, gt=0)
    regime_stress_pct: float = 0.80
    regime_min_scalar: float = Field(0.25, ge=0, lt=1)

    # Misc
    ann_factor: float = Field(252.0, gt=0)
    seed: int = 42

    @field_validator("regime_stress_pct")
    @classmethod
    def _stress_in_open_unit_interval(cls, v: float) -> float:
        if not (0.0 < v < 1.0):
            raise ValueError(f"regime_stress_pct must be in (0, 1), got {v}")
        return v

    @classmethod
    def from_yaml(cls, path: str | Path) -> "QuantDeskConfig":
        with open(path) as fh:
            data: dict[str, Any] = yaml.safe_load(fh) or {}
        return cls.model_validate(data)
