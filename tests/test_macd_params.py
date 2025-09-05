import sys
import types
from pathlib import Path

import pandas as pd
import pytest

# ensure repo root on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# stub heavy optional dependencies
sys.modules.setdefault("pandas_ta", types.ModuleType("pandas_ta"))
sys.modules.setdefault("vectorbt", types.ModuleType("vectorbt"))

import fitness  # noqa: E402
import indicator_library  # noqa: E402
import strategy_engine  # noqa: E402


def _base_data():
    return pd.DataFrame(
        {
            "Open": [1, 2, 3, 4],
            "High": [1, 2, 3, 4],
            "Low": [1, 2, 3, 4],
            "Close": [1, 2, 3, 4],
            "Volume": [1, 1, 1, 1],
        },
        index=pd.date_range("2020-01-01", periods=4, freq="D"),
    )


def test_invalid_macd_params_raise():
    df = _base_data()
    with pytest.raises(ValueError):
        indicator_library.calculate_macd(df, fast=5, slow=5, signal=1)
    with pytest.raises(ValueError):
        indicator_library.calculate_macd(df, fast=5, slow=10, signal=10)
    with pytest.raises(ValueError):
        indicator_library.calculate_macd(df, fast=5, slow=10, signal=0)
    with pytest.raises(TypeError):
        indicator_library.calculate_macd(df, fast=5.5, slow=10, signal=3)


def test_fast_equal_slow_flat_histogram(monkeypatch):
    df = _base_data()

    def macd_flat(data, fast, slow, signal):
        return pd.DataFrame({"MACDh_0": [0, 0, 0, 0]}, index=data.index)

    monkeypatch.setattr(indicator_library, "calculate_macd", macd_flat)
    monkeypatch.setitem(strategy_engine.INDICATOR_MAPPING, "macd", macd_flat)

    rules = {
        "entry_rules": {
            "conditions": [
                {
                    "indicator": "macd",
                    "params": {"fast": 5, "slow": 5, "signal": 3},
                    "condition": {"type": "indicator_is_above_value", "value": 0},
                }
            ]
        }
    }

    entries = strategy_engine.process_strategy_rules(df, rules)
    assert entries.sum() == 0


def test_valid_macd_triggers_rule(monkeypatch):
    df = _base_data()

    def macd_good(data, fast, slow, signal):
        return pd.DataFrame({"MACDh_0": [-1, -0.5, 0.5, 1.0]}, index=data.index)

    monkeypatch.setattr(indicator_library, "calculate_macd", macd_good)
    monkeypatch.setitem(strategy_engine.INDICATOR_MAPPING, "macd", macd_good)

    rules = {
        "entry_rules": {
            "conditions": [
                {
                    "indicator": "macd",
                    "params": {"fast": 5, "slow": 10, "signal": 3},
                    "condition": {"type": "indicator_is_above_value", "value": 0},
                }
            ]
        }
    }

    entries = strategy_engine.process_strategy_rules(df, rules)
    assert entries.sum() > 0
    assert entries.iloc[-1]


def test_sample_macd_params_always_valid():
    import numpy as np

    from tuner import sample_macd_params

    rng = np.random.default_rng(0)
    for _ in range(500):
        params = sample_macd_params(rng)
        assert params["fast"] < params["slow"]
        assert 1 <= params["signal"] < params["slow"]


def test_macd_fallback_to_line(monkeypatch):
    df = _base_data()

    def macd_line_only(data, fast, slow, signal):
        return pd.DataFrame({"MACD_foo": [-1, -0.5, 0.5, 1.0]}, index=data.index)

    monkeypatch.setattr(indicator_library, "calculate_macd", macd_line_only)
    monkeypatch.setitem(strategy_engine.INDICATOR_MAPPING, "macd", macd_line_only)

    rules = {
        "entry_rules": {
            "conditions": [
                {
                    "indicator": "macd",
                    "params": {"fast": 5, "slow": 10, "signal": 3},
                    "condition": {"type": "indicator_is_above_value", "value": 0},
                }
            ]
        }
    }

    entries = strategy_engine.process_strategy_rules(df, rules)
    assert entries.iloc[-1]


def test_macd_column_override(monkeypatch):
    df = _base_data()

    def macd_hist(data, fast, slow, signal):
        return pd.DataFrame(
            {"MACDh_0": [-1, -0.5, 0.5, 1.0], "MACD_line": [0, 0, 0, 0]},
            index=data.index,
        )

    monkeypatch.setattr(indicator_library, "calculate_macd", macd_hist)
    monkeypatch.setitem(strategy_engine.INDICATOR_MAPPING, "macd", macd_hist)

    rules = {
        "entry_rules": {
            "conditions": [
                {
                    "indicator": "macd",
                    "params": {"fast": 5, "slow": 10, "signal": 3},
                    "condition": {
                        "type": "indicator_is_above_value",
                        "value": 0,
                        "column": "MACDh_0",
                    },
                }
            ]
        }
    }

    entries = strategy_engine.process_strategy_rules(df, rules)
    assert entries.iloc[-1]


def test_macd_column_override_missing(monkeypatch):
    df = _base_data()

    def macd_hist(data, fast, slow, signal):
        return pd.DataFrame({"MACDh_0": [-1, -0.5, 0.5, 1.0]}, index=data.index)

    monkeypatch.setattr(indicator_library, "calculate_macd", macd_hist)
    monkeypatch.setitem(strategy_engine.INDICATOR_MAPPING, "macd", macd_hist)

    rules = {
        "entry_rules": {
            "conditions": [
                {
                    "indicator": "macd",
                    "params": {"fast": 5, "slow": 10, "signal": 3},
                    "condition": {
                        "type": "indicator_is_above_value",
                        "value": 0,
                        "column": "DoesNotExist",
                    },
                }
            ]
        }
    }

    with pytest.raises(KeyError, match="Requested column 'DoesNotExist'"):
        strategy_engine.process_strategy_rules(df, rules)


def test_inject_genes_repairs_nested_macd():
    base_rules = {
        "entry_rules": {
            "conditions": [
                {
                    "indicator": "macd",
                    "params": {"fast": 5, "slow": 4, "signal": 0},
                    "condition": {"type": "indicator_is_above_value", "value": 0},
                }
            ],
            "nested": [
                {
                    "conditions": [
                        {
                            "indicator": "macd",
                            "params": {"fast": 8, "slow": 7, "signal": 10},
                            "condition": {
                                "type": "indicator_is_above_value",
                                "value": 0,
                            },
                        }
                    ]
                }
            ],
        }
    }

    repaired = fitness._inject_genes_into_rules(base_rules, {}, [])
    top_params = repaired["entry_rules"]["conditions"][0]["params"]
    nested_params = repaired["entry_rules"]["nested"][0]["conditions"][0]["params"]
    assert top_params["fast"] < top_params["slow"]
    assert 1 <= top_params["signal"] < top_params["slow"]
    assert nested_params["fast"] < nested_params["slow"]
    assert 1 <= nested_params["signal"] < nested_params["slow"]
