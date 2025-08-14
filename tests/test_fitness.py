import sys
import types
from pathlib import Path

# Ensure repository root is on the import path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Stub heavy optional dependencies before importing modules that use them
sys.modules.setdefault('pandas_ta', types.ModuleType('pandas_ta'))
sys.modules.setdefault('vectorbt', types.ModuleType('vectorbt'))

import pandas as pd  # noqa: E402
import fitness  # noqa: E402


def test_exception_logging(capsys, monkeypatch):
    """FitnessEvaluator prints exception messages"""
    ohlc = pd.DataFrame({'Close': [1, 2, 3]})
    evaluator = fitness.FitnessEvaluator(ohlc, {}, {})

    def raise_error(*args, **kwargs):
        raise ValueError("boom")

    monkeypatch.setattr(fitness.engine, 'process_strategy_rules', raise_error)

    score = evaluator(None, [], 0)

    captured = capsys.readouterr()
    assert "boom" in captured.out
    assert score == -999.0


def test_exit_rule_param_dict(monkeypatch):
    """Non-numeric exit rule parameters are ignored."""

    # Build minimal OHLC data with enough rows to satisfy min_trades
    n = fitness.config.FITNESS_WEIGHTS["min_trades"]
    ohlc = pd.DataFrame({"Close": [1.0] * n})

    # Configure a stop-loss whose value is still a gene dictionary
    base_rules = {
        "entry_rules": {"conditions": []},
        "exit_rules": {
            "stop_loss": {
                "is_active": True,
                "params": {"value": {"gene": "x", "low": 0.01, "high": 0.1}},
            }
        },
    }

    evaluator = fitness.FitnessEvaluator(ohlc, base_rules, {})

    # Always generate an entry signal
    monkeypatch.setattr(
        fitness.engine,
        "process_strategy_rules",
        lambda data, rules: pd.Series(True, index=data.index),
    )

    captured = {}

    class DummyPF:
        def stats(self):
            return {
                "Sortino Ratio": 0.0,
                "Profit Factor": 1.0,
                "Max Drawdown [%]": 0.0,
            }

    def fake_from_signals(**kwargs):
        captured.update(kwargs)
        return DummyPF()

    monkeypatch.setattr(
        fitness.vbt,
        "Portfolio",
        types.SimpleNamespace(from_signals=fake_from_signals),
        raising=False,
    )

    score = evaluator(None, [], 0)
    assert "sl_stop" not in captured
    assert score == 0.5


def test_indicator_value_dict(monkeypatch, capsys):
    """Non-numeric indicator comparison values are treated as inactive."""

    ohlc = pd.DataFrame({"Close": [1.0, 2.0, 3.0]})

    base_rules = {
        "entry_rules": {
            "conditions": [
                {
                    "is_active": True,
                    "indicator": "rsi",
                    "params": {"period": 14},
                    # Value remains a gene dict, simulating failed injection
                    "condition": {
                        "type": "indicator_is_above_value",
                        "value": {"gene": "x"},
                    },
                }
            ]
        }
    }

    evaluator = fitness.FitnessEvaluator(ohlc, base_rules, {})

    # Stub RSI calculation and mapping to avoid external dependencies
    series = pd.Series([50.0, 50.0, 50.0], index=ohlc.index)
    monkeypatch.setitem(fitness.engine.INDICATOR_MAPPING, "rsi", lambda data, period: series)

    score = evaluator(None, [], 0)
    captured = capsys.readouterr()
    assert "dict" not in captured.out.lower()
    assert score == -1.0


def test_indicator_param_dict(monkeypatch, capsys):
    """Indicators with non-numeric params are treated as inactive."""

    ohlc = pd.DataFrame({"Close": [1.0, 2.0, 3.0]})

    base_rules = {
        "entry_rules": {
            "conditions": [
                {
                    "is_active": True,
                    "indicator": "ema",
                    "params": {"period": {"gene": "x"}},
                    "condition": {"type": "price_is_above_indicator"},
                }
            ]
        }
    }

    evaluator = fitness.FitnessEvaluator(ohlc, base_rules, {})

    # The indicator function should not be called because the period is
    # non-numeric.  If it were, this dummy implementation would raise.
    def boom(*args, **kwargs):
        raise AssertionError("indicator should not be executed")

    monkeypatch.setitem(fitness.engine.INDICATOR_MAPPING, "ema", boom)

    score = evaluator(None, [], 0)
    captured = capsys.readouterr()
    assert "dict" not in captured.out.lower()
    assert score == -1.0
