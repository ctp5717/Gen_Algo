import sys
import types
from pathlib import Path

import pandas as pd

# Ensure repository root on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Stub heavy deps
sys.modules.setdefault("pandas_ta", types.ModuleType("pandas_ta"))
try:  # prefer real vectorbt if available
    import vectorbt  # noqa: F401
except Exception:  # pragma: no cover
    sys.modules.setdefault("vectorbt", types.ModuleType("vectorbt"))

import fitness  # noqa: E402
import indicator_library  # noqa: E402
import strategy_engine  # noqa: E402


def test_backtester_integration_or_vote(monkeypatch):
    data = pd.DataFrame(
        {
            "Open": [1, 1, 1, 1],
            "High": [1, 1, 1, 1],
            "Low": [1, 1, 1, 1],
            "Close": [1, 1, 1, 1],
            "Volume": [1, 1, 1, 1],
        },
        index=pd.date_range("2020-01-01", periods=4, freq="D"),
    )

    a = pd.Series([1, 0, 1, 0], index=data.index)
    b = pd.Series([0, 1, 1, 0], index=data.index)
    c = pd.Series([1, 1, 0, 0], index=data.index)

    def ind_a(ohlc, period=None):
        return a

    def ind_b(ohlc, period=None):
        return b

    def ind_c(ohlc, period=None):
        return c

    monkeypatch.setattr(indicator_library, "calculate_ema", ind_a)
    monkeypatch.setattr(indicator_library, "calculate_rsi", ind_b)
    monkeypatch.setattr(indicator_library, "calculate_atr", ind_c)
    monkeypatch.setitem(strategy_engine.INDICATOR_MAPPING, "ema", ind_a)
    monkeypatch.setitem(strategy_engine.INDICATOR_MAPPING, "rsi", ind_b)
    monkeypatch.setitem(strategy_engine.INDICATOR_MAPPING, "atr", ind_c)

    trade_counts = []

    class DummyPF:
        def __init__(self, trades):
            self._trades = types.SimpleNamespace(count=lambda: trades)

        def stats(self):
            return {
                "Sortino Ratio": 1.0,
                "Profit Factor": 1.0,
                "Max Drawdown [%]": 0.0,
            }

    def fake_from_signals(**kwargs):
        trades = int(kwargs["entries"].sum())
        trade_counts.append(trades)
        return DummyPF(trades)

    monkeypatch.setattr(
        fitness.vbt.Portfolio, "from_signals", fake_from_signals, raising=False
    )

    cond_a = {
        "indicator": "ema",
        "params": {},
        "condition": {"type": "indicator_is_above_value", "value": 0.5},
    }
    cond_b = {
        "indicator": "rsi",
        "params": {},
        "condition": {"type": "indicator_is_above_value", "value": 0.5},
    }
    cond_c = {
        "indicator": "atr",
        "params": {},
        "condition": {"type": "indicator_is_above_value", "value": 0.5},
    }

    rules_or = {
        "entry_rules": {"combination_logic": "OR", "conditions": [cond_a, cond_b]}
    }
    rules_vote = {
        "entry_rules": {
            "combination_logic": "VOTE",
            "conditions": [cond_a, cond_b, cond_c],
            "vote_threshold": None,
        }
    }
    rules_vote_strict = {
        "entry_rules": {
            "combination_logic": "VOTE",
            "conditions": [cond_a, cond_b, cond_c],
            "vote_threshold": 3,
        }
    }

    evaluator_or = fitness.FitnessEvaluator(data, rules_or, {})
    evaluator_vote = fitness.FitnessEvaluator(data, rules_vote, {})
    evaluator_vote_strict = fitness.FitnessEvaluator(data, rules_vote_strict, {})

    evaluator_or(None, [], 0)
    evaluator_vote(None, [], 0)
    evaluator_vote_strict(None, [], 0)

    assert trade_counts == [1, 1, 0]


def test_exit_rule_params(monkeypatch):
    data = pd.DataFrame({"Close": [1, 2, 3]}, index=pd.date_range("2020", periods=3))

    monkeypatch.setattr(
        strategy_engine,
        "process_strategy_rules",
        lambda o, r: pd.Series([True, True, True], index=data.index),
    )

    captured = {}

    class DummyPF:
        def __init__(self):
            self._trades = types.SimpleNamespace(count=lambda: 1)

        def stats(self):
            return {"Sortino Ratio": 1.0, "Profit Factor": 1.0, "Max Drawdown [%]": 0.0}

    def fake_from_signals(**kwargs):
        captured["sl_stop"] = kwargs.get("sl_stop")
        captured["sl_trail"] = kwargs.get("sl_trail")
        captured["tp_stop"] = kwargs.get("tp_stop")
        captured["exits"] = kwargs.get("exits")
        return DummyPF()

    monkeypatch.setattr(
        fitness.vbt.Portfolio, "from_signals", fake_from_signals, raising=False
    )
    monkeypatch.setattr(
        fitness.config, "USE_DYNAMIC_EXIT_SIMULATOR", False, raising=False
    )

    rules = {
        "exit_rules": {
            "stop_loss": {
                "is_active": True,
                "type": "percentage",
                "params": {"value": 0.05},
            },
            "trailing_stop": {
                "is_active": True,
                "type": "percentage",
                "params": {"value": 0.03},
            },
            "take_profit": {
                "is_active": True,
                "type": "percentage",
                "params": {"value": 0.07},
            },
        }
    }

    evaluator = fitness.FitnessEvaluator(data, rules, {})
    evaluator(None, [], 0)

    assert captured["sl_stop"] == 0.05
    assert captured["sl_trail"] == 0.03
    assert captured["tp_stop"] == 0.07
    assert captured["exits"].sum() == 0
