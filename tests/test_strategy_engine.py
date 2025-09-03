import sys
import types
from pathlib import Path

import pandas as pd
import pytest

# Ensure repository root is on the import path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Stub heavy optional dependencies before importing modules
sys.modules.setdefault("pandas_ta", types.ModuleType("pandas_ta"))
sys.modules.setdefault("vectorbt", types.ModuleType("vectorbt"))

import indicator_library  # noqa: E402
import strategy_engine  # noqa: E402


def test_process_strategy_rules_simple(monkeypatch):
    data = pd.DataFrame(
        {
            "Open": [1, 2, 3, 4, 5],
            "High": [1, 2, 3, 4, 5],
            "Low": [1, 2, 3, 4, 5],
            "Close": [10, 11, 12, 13, 14],
            "Volume": [100, 100, 100, 100, 100],
        },
        index=pd.date_range("2020-01-01", periods=5, freq="D"),
    )

    # Patch indicator calculations with simple deterministic series
    def ema_func(ohlc, period):
        return ohlc["Close"] - 1

    def rsi_func(ohlc, period):
        return pd.Series(60, index=ohlc.index)

    monkeypatch.setattr(indicator_library, "calculate_ema", ema_func)
    monkeypatch.setattr(indicator_library, "calculate_rsi", rsi_func)
    monkeypatch.setitem(strategy_engine.INDICATOR_MAPPING, "ema", ema_func)
    monkeypatch.setitem(strategy_engine.INDICATOR_MAPPING, "rsi", rsi_func)

    rules = {
        "entry_rules": {
            "combination_logic": "AND",
            "conditions": [
                {
                    "indicator": "ema",
                    "params": {"period": 3},
                    "condition": {"type": "price_is_above_indicator"},
                },
                {
                    "indicator": "rsi",
                    "params": {"period": 2},
                    "condition": {"type": "indicator_is_above_value", "value": 50},
                },
            ],
        }
    }

    signal = strategy_engine.process_strategy_rules(data, rules)

    assert signal.all()


def test_combination_logic_variants(monkeypatch):
    data = pd.DataFrame(
        {
            "Open": [1, 1, 1, 1, 1],
            "High": [1, 1, 1, 1, 1],
            "Low": [1, 1, 1, 1, 1],
            "Close": [1, 1, 1, 1, 1],
            "Volume": [1, 1, 1, 1, 1],
        },
        index=pd.date_range("2020-01-01", periods=5, freq="D"),
    )

    # Boolean patterns for the three indicators
    a = pd.Series([1, 1, 0, 0, 1], index=data.index)
    b = pd.Series([0, 1, 0, 1, 0], index=data.index)
    c = pd.Series([1, 0, 1, 0, 1], index=data.index)

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

    # AND logic
    rules_and = {
        "entry_rules": {"combination_logic": "AND", "conditions": [cond_a, cond_b]}
    }
    expected_and = (a > 0.5) & (b > 0.5)
    result_and = strategy_engine.process_strategy_rules(data, rules_and)
    pd.testing.assert_series_equal(result_and, expected_and)

    # OR logic
    rules_or = {
        "entry_rules": {"combination_logic": "OR", "conditions": [cond_a, cond_b]}
    }
    expected_or = (a > 0.5) | (b > 0.5)
    result_or = strategy_engine.process_strategy_rules(data, rules_or)
    pd.testing.assert_series_equal(result_or, expected_or)

    # VOTE logic default threshold (ceil(3/2)=2)
    rules_vote_default = {
        "entry_rules": {
            "combination_logic": "VOTE",
            "conditions": [cond_a, cond_b, cond_c],
        }
    }
    expected_vote_default = (
        (a > 0.5).astype(int) + (b > 0.5).astype(int) + (c > 0.5).astype(int)
    ) >= 2
    result_vote_default = strategy_engine.process_strategy_rules(
        data, rules_vote_default
    )
    pd.testing.assert_series_equal(result_vote_default, expected_vote_default)

    # VOTE logic explicit threshold
    rules_vote_explicit = {
        "entry_rules": {
            "combination_logic": "VOTE",
            "vote_threshold": 3,
            "conditions": [cond_a, cond_b, cond_c],
        }
    }
    expected_vote_explicit = (a > 0.5) & (b > 0.5) & (c > 0.5)
    result_vote_explicit = strategy_engine.process_strategy_rules(
        data, rules_vote_explicit
    )
    pd.testing.assert_series_equal(result_vote_explicit, expected_vote_explicit)


def test_combination_logic_invalid(monkeypatch):
    data = pd.DataFrame({"Close": [1, 2]}, index=pd.date_range("2020-01-01", periods=2))

    def ind(ohlc, period=None):
        return pd.Series([1, 1], index=ohlc.index)

    monkeypatch.setattr(indicator_library, "calculate_ema", ind)
    monkeypatch.setitem(strategy_engine.INDICATOR_MAPPING, "ema", ind)

    cond = {
        "indicator": "ema",
        "params": {},
        "condition": {"type": "indicator_is_above_value", "value": 0.5},
    }

    # Invalid combination_logic
    rules_invalid_logic = {
        "entry_rules": {"combination_logic": "XOR", "conditions": [cond]}
    }
    with pytest.raises(ValueError):
        strategy_engine.process_strategy_rules(data, rules_invalid_logic)

    # Invalid vote_threshold
    rules_invalid_vote = {
        "entry_rules": {
            "combination_logic": "VOTE",
            "vote_threshold": 0,
            "conditions": [cond],
        }
    }
    with pytest.raises(ValueError):
        strategy_engine.process_strategy_rules(data, rules_invalid_vote)


def test_nan_handling_toggle(monkeypatch):
    data = pd.DataFrame(
        {"Close": [1, 1, 1]}, index=pd.date_range("2020-01-01", periods=3)
    )

    a = pd.Series([True, pd.NA, True], index=data.index, dtype="boolean")
    b = pd.Series([False, True, pd.NA], index=data.index, dtype="boolean")
    c = pd.Series([False, True, True], index=data.index, dtype="boolean")

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

    rules_and_false = {
        "entry_rules": {
            "combination_logic": "AND",
            "treat_nan_as_false": False,
            "conditions": [cond_a, cond_b],
        }
    }
    expected_and_false = pd.Series(
        [False, pd.NA, pd.NA], index=data.index, dtype="boolean"
    )
    result_and_false = strategy_engine.process_strategy_rules(data, rules_and_false)
    pd.testing.assert_series_equal(
        result_and_false.astype("boolean"), expected_and_false
    )

    rules_and_true = {
        "entry_rules": {
            "combination_logic": "AND",
            "treat_nan_as_false": True,
            "conditions": [cond_a, cond_b],
        }
    }
    expected_and_true = pd.Series(
        [False, False, False], index=data.index, dtype="boolean"
    )
    result_and_true = strategy_engine.process_strategy_rules(data, rules_and_true)
    pd.testing.assert_series_equal(result_and_true.astype("boolean"), expected_and_true)

    rules_vote_false = {
        "entry_rules": {
            "combination_logic": "VOTE",
            "vote_threshold": 2,
            "treat_nan_as_false": False,
            "conditions": [cond_a, cond_b, cond_c],
        }
    }
    expected_vote_false = pd.Series(
        [False, pd.NA, pd.NA], index=data.index, dtype="boolean"
    )
    result_vote_false = strategy_engine.process_strategy_rules(data, rules_vote_false)
    pd.testing.assert_series_equal(
        result_vote_false.astype("boolean"), expected_vote_false
    )

    rules_vote_true = {
        "entry_rules": {
            "combination_logic": "VOTE",
            "vote_threshold": 2,
            "treat_nan_as_false": True,
            "conditions": [cond_a, cond_b, cond_c],
        }
    }
    expected_vote_true = pd.Series(
        [False, True, True], index=data.index, dtype="boolean"
    )
    result_vote_true = strategy_engine.process_strategy_rules(data, rules_vote_true)
    pd.testing.assert_series_equal(
        result_vote_true.astype("boolean"), expected_vote_true
    )


def test_combination_logic_default(monkeypatch):
    data = pd.DataFrame({"Close": [1, 1]}, index=pd.date_range("2020", periods=2))

    a = pd.Series([True, pd.NA], index=data.index, dtype="boolean")
    b = pd.Series([True, True], index=data.index, dtype="boolean")

    def ind_a(ohlc, period=None):
        return a

    def ind_b(ohlc, period=None):
        return b

    monkeypatch.setitem(strategy_engine.INDICATOR_MAPPING, "a", ind_a)
    monkeypatch.setitem(strategy_engine.INDICATOR_MAPPING, "b", ind_b)

    cond_a = {
        "indicator": "a",
        "params": {},
        "condition": {"type": "indicator_is_above_value", "value": 0},
    }
    cond_b = {
        "indicator": "b",
        "params": {},
        "condition": {"type": "indicator_is_above_value", "value": 0},
    }

    rules = {"entry_rules": {"conditions": [cond_a, cond_b]}}

    result = strategy_engine.process_strategy_rules(data, rules)
    expected = pd.Series([True, False], index=data.index, dtype="boolean")
    pd.testing.assert_series_equal(result.astype("boolean"), expected)


def test_dataframe_indicator_or_vote(monkeypatch):
    data = pd.DataFrame({"Close": [1, 1, 1]}, index=pd.date_range("2020", periods=3))

    def multi1(ohlc, period=None):
        return pd.DataFrame({"col1": [1, 0, 1], "col2": [0, 0, 0]}, index=ohlc.index)

    def multi2(ohlc, period=None):
        return pd.DataFrame({"col1": [0, 1, 1], "col2": [0, 0, 0]}, index=ohlc.index)

    monkeypatch.setitem(strategy_engine.INDICATOR_MAPPING, "m1", multi1)
    monkeypatch.setitem(strategy_engine.INDICATOR_MAPPING, "m2", multi2)

    cond1 = {
        "indicator": "m1",
        "params": {},
        "condition": {"type": "indicator_is_above_value", "value": 0.5},
    }
    cond2 = {
        "indicator": "m2",
        "params": {},
        "condition": {"type": "indicator_is_above_value", "value": 0.5},
    }

    rules_or = {
        "entry_rules": {"combination_logic": "OR", "conditions": [cond1, cond2]}
    }
    expected_or = pd.Series(
        [True, True, True], index=data.index, dtype="boolean", name="col1"
    )
    result_or = strategy_engine.process_strategy_rules(data, rules_or)
    pd.testing.assert_series_equal(result_or.astype("boolean"), expected_or)

    rules_vote = {
        "entry_rules": {
            "combination_logic": "VOTE",
            "vote_threshold": 2,
            "conditions": [cond1, cond2],
        }
    }
    expected_vote = pd.Series([False, False, True], index=data.index, dtype="boolean")
    result_vote = strategy_engine.process_strategy_rules(data, rules_vote)
    pd.testing.assert_series_equal(result_vote.astype("boolean"), expected_vote)


def test_parameter_type_validation():
    data = pd.DataFrame({"Close": [1]}, index=pd.date_range("2020", periods=1))

    with pytest.raises(TypeError):
        strategy_engine.process_strategy_rules(
            data, {"entry_rules": {"vote_threshold": 1.5, "conditions": []}}
        )

    with pytest.raises(TypeError):
        strategy_engine.process_strategy_rules(
            data, {"entry_rules": {"treat_nan_as_false": "no", "conditions": []}}
        )
