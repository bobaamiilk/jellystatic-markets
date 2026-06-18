"""Tests for QuantDeskConfig validation and YAML loading."""
import pytest
from pydantic import ValidationError

from quantdesk.config import QuantDeskConfig, SignalCfg, CostCfg, VolTargetCfg


def test_default_config_is_valid():
    cfg = QuantDeskConfig()
    assert cfg.train_size > 0
    assert cfg.test_size > 0
    assert 0 < cfg.regime_stress_pct < 1
    assert cfg.regime_min_scalar >= 0


def test_train_size_must_be_positive():
    with pytest.raises(ValidationError):
        QuantDeskConfig(train_size=0)


def test_test_size_must_be_positive():
    with pytest.raises(ValidationError):
        QuantDeskConfig(test_size=-1)


def test_stress_pct_boundary():
    with pytest.raises(ValidationError):
        QuantDeskConfig(regime_stress_pct=0.0)
    with pytest.raises(ValidationError):
        QuantDeskConfig(regime_stress_pct=1.0)
    cfg = QuantDeskConfig(regime_stress_pct=0.75)
    assert cfg.regime_stress_pct == 0.75


def test_nested_signal_config():
    cfg = QuantDeskConfig(signal={"entry": 1.5, "exit": 0.3, "stop": 3.5, "max_holding": 20})
    assert cfg.signal.entry == 1.5
    assert cfg.signal.max_holding == 20


def test_yaml_roundtrip(tmp_path):
    import yaml
    data = {
        "tickers": ["NKY", "HSI"],
        "train_size": 200,
        "test_size": 50,
        "regime_stress_pct": 0.85,
        "signal": {"entry": 1.8, "exit": 0.4, "stop": 3.5, "max_holding": 25},
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.dump(data))
    cfg = QuantDeskConfig.from_yaml(p)
    assert cfg.tickers == ["NKY", "HSI"]
    assert cfg.train_size == 200
    assert cfg.regime_stress_pct == 0.85
    assert cfg.signal.entry == 1.8
