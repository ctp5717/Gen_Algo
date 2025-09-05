import sys
import types
from pathlib import Path

import numpy as np
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


def test_generate_signal_from_value_validates_value_param():
    series = pd.Series([1, 2, 3])

    with pytest.raises(ValueError, match="'value' must be provided"):
        strategy_engine._generate_signal_from_value(
            series, {"type": "indicator_is_above_value"}
        )

    with pytest.raises(ValueError, match="'value' must be provided"):
        strategy_engine._generate_signal_from_value(
            series, {"type": "indicator_is_above_value", "value": None}
        )

    with pytest.raises(TypeError, match="int or float"):
        strategy_engine._generate_signal_from_value(
            series, {"type": "indicator_is_above_value", "value": "bad"}
        )


def test_generate_signal_dispatch_and_unknown(monkeypatch):
    data = pd.DataFrame({"Close": [1, 2]}, index=pd.date_range("2020", periods=2))
    indicator = pd.Series([0, 0], index=data.index)

    called: dict[str, bool] = {}

    def fake(price, band):  # noqa: ANN001
        called["run"] = True
        return pd.Series([True, False], index=price.index)

    monkeypatch.setitem(
        strategy_engine.CONDITION_FUNCTIONS, "price_is_above_upper_band", fake
    )

    res = strategy_engine._generate_signal(
        data, indicator, {"type": "price_is_above_upper_band"}
    )

    assert called.get("run") is True
    pd.testing.assert_series_equal(
        res.astype(bool), pd.Series([True, False], index=data.index)
    )

    with pytest.warns(UserWarning, match="Unknown condition type 'unknown'"):
        unknown = strategy_engine._generate_signal(data, indicator, {"type": "unknown"})
    assert not unknown.any()


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


def test_vote_default_threshold_even(monkeypatch):
    data = pd.DataFrame({"Close": [1, 2]}, index=pd.date_range("2020", periods=2))

    def make_ind(vals):
        return lambda ohlc, period=None: pd.Series(vals, index=ohlc.index)

    a = make_ind([1, 0])
    b = make_ind([0, 1])
    c = make_ind([1, 1])
    d = make_ind([0, 0])
    monkeypatch.setitem(strategy_engine.INDICATOR_MAPPING, "a", a)
    monkeypatch.setitem(strategy_engine.INDICATOR_MAPPING, "b", b)
    monkeypatch.setitem(strategy_engine.INDICATOR_MAPPING, "c", c)
    monkeypatch.setitem(strategy_engine.INDICATOR_MAPPING, "d", d)

    cond_template = {
        "params": {},
        "condition": {"type": "indicator_is_above_value", "value": 0.5},
    }
    conds = []
    for name in ["a", "b", "c", "d"]:
        cond = {"indicator": name, **cond_template}
        conds.append(cond)
    rules = {"entry_rules": {"combination_logic": "VOTE", "conditions": conds}}
    result = strategy_engine.process_strategy_rules(data, rules)
    expected = (
        (a(data) > 0.5).astype(int)
        + (b(data) > 0.5).astype(int)
        + (c(data) > 0.5).astype(int)
        + (d(data) > 0.5).astype(int)
    ) >= 2
    pd.testing.assert_series_equal(result, expected)


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

    # Invalid vote_threshold now normalized to 1
    rules_invalid_vote = {
        "entry_rules": {
            "combination_logic": "VOTE",
            "vote_threshold": 0,
            "conditions": [cond],
        }
    }
    strategy_engine.process_strategy_rules(data, rules_invalid_vote)


def test_combination_logic_case_insensitive(monkeypatch):
    data = pd.DataFrame({"Close": [1, 1]}, index=pd.date_range("2020", periods=2))

    def ind(ohlc, period=None):
        return pd.Series([1, 0], index=ohlc.index)

    monkeypatch.setitem(strategy_engine.INDICATOR_MAPPING, "ema", ind)
    cond = {
        "indicator": "ema",
        "params": {},
        "condition": {"type": "indicator_is_above_value", "value": 0.5},
    }

    rules_or = {"entry_rules": {"combination_logic": "or", "conditions": [cond, cond]}}
    res_or = strategy_engine.process_strategy_rules(data, rules_or)
    pd.testing.assert_series_equal(
        res_or.astype(bool), pd.Series([True, False], index=data.index)
    )

    rules_vote = {
        "entry_rules": {"combination_logic": "vote", "conditions": [cond, cond]}
    }
    res_vote = strategy_engine.process_strategy_rules(data, rules_vote)
    pd.testing.assert_series_equal(
        res_vote.astype(bool), pd.Series([True, False], index=data.index)
    )


def test_combination_logic_gene_dict(monkeypatch):
    data = pd.DataFrame({"Close": [1, 1]}, index=pd.date_range("2020", periods=2))

    a = pd.Series([1, 0], index=data.index)
    b = pd.Series([0, 1], index=data.index)

    def ind_a(ohlc, period=None):
        return a

    def ind_b(ohlc, period=None):
        return b

    monkeypatch.setattr(indicator_library, "calculate_ema", ind_a)
    monkeypatch.setattr(indicator_library, "calculate_rsi", ind_b)
    monkeypatch.setitem(strategy_engine.INDICATOR_MAPPING, "ema", ind_a)
    monkeypatch.setitem(strategy_engine.INDICATOR_MAPPING, "rsi", ind_b)

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

    rules = {
        "entry_rules": {
            "combination_logic": {"name": "cl", "options": ["AND", "OR"]},
            "conditions": [cond_a, cond_b],
        }
    }

    result = strategy_engine.process_strategy_rules(data, rules)
    expected = (a > 0.5) & (b > 0.5)
    pd.testing.assert_series_equal(result, expected)


def test_single_condition_vote(monkeypatch):
    data = pd.DataFrame({"Close": [1, 2]}, index=pd.date_range("2020", periods=2))

    def ind(ohlc, period=None):
        return pd.Series([1, 2], index=ohlc.index)

    monkeypatch.setitem(strategy_engine.INDICATOR_MAPPING, "ema", ind)
    cond = {
        "indicator": "ema",
        "params": {},
        "condition": {"type": "indicator_is_above_value", "value": 0.5},
    }

    rules_default = {"entry_rules": {"combination_logic": "VOTE", "conditions": [cond]}}
    with pytest.warns(
        RuntimeWarning, match="Single active condition; normalized combination_logic"
    ):
        res_default = strategy_engine.process_strategy_rules(data, rules_default)
    pd.testing.assert_series_equal(
        res_default.astype(bool), pd.Series([True, True], index=data.index)
    )

    rules_explicit = {
        "entry_rules": {
            "combination_logic": "VOTE",
            "vote_threshold": 1,
            "conditions": [cond],
        }
    }
    res_explicit = strategy_engine.process_strategy_rules(data, rules_explicit)
    pd.testing.assert_series_equal(
        res_explicit.astype(bool), pd.Series([True, True], index=data.index)
    )

    rules_bad = {
        "entry_rules": {
            "combination_logic": "VOTE",
            "vote_threshold": 2,
            "conditions": [cond],
        }
    }
    with pytest.warns(
        RuntimeWarning, match="Single active condition; normalized combination_logic"
    ):
        res_bad = strategy_engine.process_strategy_rules(data, rules_bad)
    pd.testing.assert_series_equal(
        res_bad.astype(bool), pd.Series([True, True], index=data.index)
    )


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


def test_bband_band_hint_and_fallback(monkeypatch):
    data = pd.DataFrame({"Close": [1, 2, 3]}, index=pd.date_range("2020", periods=3))

    class Accessor:
        def __init__(self, series):
            self._s = series

        def crossed_above(self, other):
            return self._s > other

        def crossed_below(self, other):
            return self._s < other

    monkeypatch.setattr(
        pd.Series, "vbt", property(lambda s: Accessor(s)), raising=False
    )

    def bb_func(ohlc, period=None, std_dev=None):
        return pd.DataFrame(
            {
                "BBU": [0, 5, 0],
                "BBM": [0, 0, 5],
                "BBL": [5, 0, 0],
            },
            index=ohlc.index,
        )

    monkeypatch.setattr(indicator_library, "calculate_bbands", bb_func)
    monkeypatch.setitem(strategy_engine.INDICATOR_MAPPING, "bbands", bb_func)

    # Explicit band hints
    rules_upper = {
        "entry_rules": {
            "conditions": [
                {
                    "indicator": "bbands",
                    "params": {},
                    "condition": {
                        "type": "price_is_above_indicator",
                        "band": "upper",
                    },
                }
            ]
        }
    }
    res_upper = strategy_engine.process_strategy_rules(data, rules_upper)
    pd.testing.assert_series_equal(
        res_upper.astype(bool), pd.Series([True, False, True], index=data.index)
    )

    rules_middle = {
        "entry_rules": {
            "conditions": [
                {
                    "indicator": "bbands",
                    "params": {},
                    "condition": {
                        "type": "price_is_above_indicator",
                        "band": "middle",
                    },
                }
            ]
        }
    }
    res_middle = strategy_engine.process_strategy_rules(data, rules_middle)
    pd.testing.assert_series_equal(
        res_middle.astype(bool), pd.Series([True, True, False], index=data.index)
    )

    rules_mid_syn = {
        "entry_rules": {
            "conditions": [
                {
                    "indicator": "bbands",
                    "params": {},
                    "condition": {
                        "type": "price_is_above_indicator",
                        "band": "mid",
                    },
                }
            ]
        }
    }
    res_mid_syn = strategy_engine.process_strategy_rules(data, rules_mid_syn)
    pd.testing.assert_series_equal(
        res_mid_syn.astype(bool), pd.Series([True, True, False], index=data.index)
    )

    rules_basis_syn = {
        "entry_rules": {
            "conditions": [
                {
                    "indicator": "bbands",
                    "params": {},
                    "condition": {
                        "type": "price_is_above_indicator",
                        "band": "basis",
                    },
                }
            ]
        }
    }
    res_basis_syn = strategy_engine.process_strategy_rules(data, rules_basis_syn)
    pd.testing.assert_series_equal(
        res_basis_syn.astype(bool),
        pd.Series([True, True, False], index=data.index),
    )

    rules_lower = {
        "entry_rules": {
            "conditions": [
                {
                    "indicator": "bbands",
                    "params": {},
                    "condition": {
                        "type": "price_is_above_indicator",
                        "band": "lower",
                    },
                }
            ]
        }
    }
    res_lower = strategy_engine.process_strategy_rules(data, rules_lower)
    pd.testing.assert_series_equal(
        res_lower.astype(bool), pd.Series([False, True, True], index=data.index)
    )

    # Fallback to condition type keywords
    rules_fallback_upper = {
        "entry_rules": {
            "conditions": [
                {
                    "indicator": "bbands",
                    "params": {},
                    "condition": {"type": "price_crosses_above_upper_band"},
                }
            ]
        }
    }
    res_fu = strategy_engine.process_strategy_rules(data, rules_fallback_upper)
    pd.testing.assert_series_equal(
        res_fu.astype(bool), pd.Series([True, False, True], index=data.index)
    )

    rules_fallback_default = {
        "entry_rules": {
            "conditions": [
                {
                    "indicator": "bbands",
                    "params": {},
                    "condition": {"type": "price_is_above_indicator"},
                }
            ]
        }
    }
    res_fd = strategy_engine.process_strategy_rules(data, rules_fallback_default)
    pd.testing.assert_series_equal(
        res_fd.astype(bool), pd.Series([True, True, False], index=data.index)
    )

    # Variant condition types
    rules_is_above_upper = {
        "entry_rules": {
            "conditions": [
                {
                    "indicator": "bbands",
                    "params": {},
                    "condition": {"type": "price_is_above_upper_band"},
                }
            ]
        }
    }
    res_is_above_upper = strategy_engine.process_strategy_rules(
        data, rules_is_above_upper
    )
    pd.testing.assert_series_equal(
        res_is_above_upper.astype(bool),
        pd.Series([True, False, True], index=data.index),
    )

    rules_is_below_lower = {
        "entry_rules": {
            "conditions": [
                {
                    "indicator": "bbands",
                    "params": {},
                    "condition": {"type": "price_is_below_lower_band"},
                }
            ]
        }
    }
    res_is_below_lower = strategy_engine.process_strategy_rules(
        data, rules_is_below_lower
    )
    pd.testing.assert_series_equal(
        res_is_below_lower.astype(bool),
        pd.Series([True, False, False], index=data.index),
    )

    rules_is_below_upper = {
        "entry_rules": {
            "conditions": [
                {
                    "indicator": "bbands",
                    "params": {},
                    "condition": {"type": "price_is_below_upper_band"},
                }
            ]
        }
    }
    res_is_below_upper = strategy_engine.process_strategy_rules(
        data, rules_is_below_upper
    )
    pd.testing.assert_series_equal(
        res_is_below_upper.astype(bool),
        pd.Series([False, True, False], index=data.index),
    )

    rules_is_above_lower = {
        "entry_rules": {
            "conditions": [
                {
                    "indicator": "bbands",
                    "params": {},
                    "condition": {"type": "price_is_above_lower_band"},
                }
            ]
        }
    }
    res_is_above_lower = strategy_engine.process_strategy_rules(
        data, rules_is_above_lower
    )
    pd.testing.assert_series_equal(
        res_is_above_lower.astype(bool),
        pd.Series([False, True, True], index=data.index),
    )

    rules_is_above_middle = {
        "entry_rules": {
            "conditions": [
                {
                    "indicator": "bbands",
                    "params": {},
                    "condition": {"type": "price_is_above_middle_band"},
                }
            ]
        }
    }
    res_is_above_middle = strategy_engine.process_strategy_rules(
        data, rules_is_above_middle
    )
    pd.testing.assert_series_equal(
        res_is_above_middle.astype(bool),
        pd.Series([True, True, False], index=data.index),
    )

    rules_is_below_middle = {
        "entry_rules": {
            "conditions": [
                {
                    "indicator": "bbands",
                    "params": {},
                    "condition": {"type": "price_is_below_middle_band"},
                }
            ]
        }
    }
    res_is_below_middle = strategy_engine.process_strategy_rules(
        data, rules_is_below_middle
    )
    pd.testing.assert_series_equal(
        res_is_below_middle.astype(bool),
        pd.Series([False, False, True], index=data.index),
    )

    rules_cross_below_upper = {
        "entry_rules": {
            "conditions": [
                {
                    "indicator": "bbands",
                    "params": {},
                    "condition": {"type": "price_crosses_below_upper_band"},
                }
            ]
        }
    }
    res_cross_below_upper = strategy_engine.process_strategy_rules(
        data, rules_cross_below_upper
    )
    pd.testing.assert_series_equal(
        res_cross_below_upper.astype(bool),
        pd.Series([False, True, False], index=data.index),
    )

    rules_cross_above_lower = {
        "entry_rules": {
            "conditions": [
                {
                    "indicator": "bbands",
                    "params": {},
                    "condition": {"type": "price_crosses_above_lower_band"},
                }
            ]
        }
    }
    res_cross_above_lower = strategy_engine.process_strategy_rules(
        data, rules_cross_above_lower
    )
    pd.testing.assert_series_equal(
        res_cross_above_lower.astype(bool),
        pd.Series([False, True, True], index=data.index),
    )

    # Unknown band falls back to middle
    rules_unknown_band = {
        "entry_rules": {
            "conditions": [
                {
                    "indicator": "bbands",
                    "params": {},
                    "condition": {
                        "type": "price_is_above_indicator",
                        "band": "weird",
                    },
                }
            ]
        }
    }
    res_unknown = strategy_engine.process_strategy_rules(data, rules_unknown_band)
    pd.testing.assert_series_equal(
        res_unknown.astype(bool), pd.Series([True, True, False], index=data.index)
    )


def test_bband_missing_column(monkeypatch):
    data = pd.DataFrame({"Close": [1, 2, 3]}, index=pd.date_range("2020", periods=3))

    def bb_bad(ohlc, period=None, std_dev=None):
        return pd.DataFrame({"XX": [1, 2, 3]}, index=ohlc.index)

    monkeypatch.setattr(indicator_library, "calculate_bbands", bb_bad)
    monkeypatch.setitem(strategy_engine.INDICATOR_MAPPING, "bbands", bb_bad)

    rules = {
        "entry_rules": {
            "conditions": [
                {
                    "indicator": "bbands",
                    "params": {},
                    "condition": {"type": "price_is_above_upper_band"},
                }
            ]
        }
    }
    with pytest.raises(KeyError):
        strategy_engine.process_strategy_rules(data, rules)


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


@pytest.mark.parametrize(
    "macd_output",
    [
        pd.DataFrame({"MACDh_1": [-1, 1], "MACD": [-1, -1]}),
        pd.DataFrame({"MACD": [-1, 1], "MACDs": [-1, -1]}),
        pd.DataFrame({"MACDs": [-1, 1], "Other": [-1, -1]}),
        pd.Series([-1, 1], name="MACDh_1"),
    ],
    ids=["hist", "line", "first", "series"],
)
def test_macd_component_selection(monkeypatch, macd_output):
    data = pd.DataFrame({"Close": [1, 2]}, index=pd.RangeIndex(2))

    def macd_func(df, fast, slow, signal):
        return macd_output

    monkeypatch.setattr(indicator_library, "calculate_macd", macd_func)
    monkeypatch.setitem(strategy_engine.INDICATOR_MAPPING, "macd", macd_func)

    rules = {
        "entry_rules": {
            "conditions": [
                {
                    "indicator": "macd",
                    "params": {"fast": 12, "slow": 26, "signal": 9},
                    "condition": {"type": "indicator_is_above_value", "value": 0},
                }
            ]
        }
    }

    res = strategy_engine.process_strategy_rules(data, rules)
    expected = pd.Series([False, True], index=data.index)
    pd.testing.assert_series_equal(res.astype(bool), expected, check_names=False)


def test_macd_empty_dataframe(monkeypatch):
    data = pd.DataFrame({"Close": [1, 2]}, index=pd.RangeIndex(2))

    def macd_empty(df, fast, slow, signal):
        return pd.DataFrame()

    monkeypatch.setattr(indicator_library, "calculate_macd", macd_empty)
    monkeypatch.setitem(strategy_engine.INDICATOR_MAPPING, "macd", macd_empty)

    rules = {
        "entry_rules": {
            "conditions": [
                {
                    "indicator": "macd",
                    "params": {"fast": 12, "slow": 26, "signal": 9},
                    "condition": {"type": "indicator_is_above_value", "value": 0},
                }
            ]
        }
    }

    with pytest.raises(KeyError, match="available"):
        strategy_engine.process_strategy_rules(data, rules)


def test_macd_column_hint(monkeypatch):
    data = pd.DataFrame({"Close": [1, 2]}, index=pd.RangeIndex(2))

    def macd_alt(df, fast, slow, signal):
        return pd.DataFrame({"MACD_Hist": [-1, 1], "MACD": [-1, 1]})

    monkeypatch.setattr(indicator_library, "calculate_macd", macd_alt)
    monkeypatch.setitem(strategy_engine.INDICATOR_MAPPING, "macd", macd_alt)

    rules_column = {
        "entry_rules": {
            "conditions": [
                {
                    "indicator": "macd",
                    "params": {"fast": 12, "slow": 26, "signal": 9},
                    "condition": {
                        "type": "indicator_is_above_value",
                        "value": 0,
                        "column": "MACD",
                    },
                }
            ]
        }
    }
    res_column = strategy_engine.process_strategy_rules(data, rules_column)
    expected = pd.Series([False, True], index=data.index)
    pd.testing.assert_series_equal(res_column.astype(bool), expected, check_names=False)

    rules_missing = {
        "entry_rules": {
            "conditions": [
                {
                    "indicator": "macd",
                    "params": {"fast": 12, "slow": 26, "signal": 9},
                    "condition": {
                        "type": "indicator_is_above_value",
                        "value": 0,
                        "column": "NOPE",
                    },
                }
            ]
        }
    }
    with pytest.raises(KeyError, match="NOPE"):
        strategy_engine.process_strategy_rules(data, rules_missing)


def test_signal_counts_name_and_nan_policy(monkeypatch):
    data = pd.DataFrame({"Close": [1, 2, 3]})

    def fake_indicator(df, **p):
        return pd.Series([1, 2, 3], index=df.index)

    base_signal = pd.Series([True, pd.NA, True], index=data.index, dtype="boolean")

    def fake_generate(series, condition):
        return base_signal.copy()

    monkeypatch.setitem(strategy_engine.INDICATOR_MAPPING, "foo", fake_indicator)
    monkeypatch.setattr(strategy_engine, "_generate_signal_from_value", fake_generate)

    rules = {
        "entry_rules": {
            "treat_nan_as_false": True,
            "conditions": [
                {
                    "indicator": "foo",
                    "condition": {"type": "indicator_is_above_value", "value": 0},
                }
            ],
        }
    }

    res_true, counts_true = strategy_engine.process_strategy_rules(
        data, rules, collect_counts=True
    )
    assert not res_true.isna().any()
    pd.testing.assert_series_equal(
        res_true.astype(bool), pd.Series([True, False, True], index=data.index)
    )
    assert counts_true == {"foo:indicator_is_above_value": 2}

    rules["entry_rules"]["treat_nan_as_false"] = False
    res_false, counts_false = strategy_engine.process_strategy_rules(
        data, rules, collect_counts=True
    )
    expected_false = pd.Series([True, pd.NA, True], index=data.index, dtype="boolean")
    pd.testing.assert_series_equal(res_false, expected_false)
    assert counts_false == {"foo:indicator_is_above_value": 2}


def test_vote_threshold_normalization(monkeypatch):
    data = pd.DataFrame({"Close": [1, 2, 3]})
    monkeypatch.setitem(
        strategy_engine.INDICATOR_MAPPING,
        "ema",
        lambda df, **p: pd.Series([1, 2, 3], name="ema"),
    )
    rules = {
        "entry_rules": {
            "combination_logic": "VOTE",
            "conditions": [
                {
                    "indicator": "ema",
                    "params": {},
                    "condition": {"type": "indicator_is_above_value", "value": 0},
                }
            ],
        }
    }
    with pytest.warns(
        RuntimeWarning, match="Single active condition; normalized combination_logic"
    ):
        strategy_engine.process_strategy_rules(data, rules)


def test_vote_threshold_clamped(monkeypatch):
    data = pd.DataFrame({"Close": [1, 2, 3]})

    monkeypatch.setitem(
        strategy_engine.INDICATOR_MAPPING,
        "ema",
        lambda df, **p: pd.Series([1, 2, 3], name="ema"),
    )
    rules = {
        "entry_rules": {
            "combination_logic": "VOTE",
            "vote_threshold": 5,
            "conditions": [
                {
                    "indicator": "ema",
                    "params": {},
                    "condition": {"type": "indicator_is_above_value", "value": 0},
                },
                {
                    "indicator": "ema",
                    "params": {},
                    "condition": {"type": "indicator_is_above_value", "value": 0},
                },
            ],
        }
    }
    with pytest.warns(RuntimeWarning, match="vote_threshold exceeds active conditions"):
        res = strategy_engine.process_strategy_rules(data, rules)
    assert res.all()


def test_nan_policy(monkeypatch):
    data = pd.DataFrame({"Close": [1, 2, 3]})

    def base_ind(df, **p):
        return pd.Series([0, 0, 0], index=df.index)

    def fake_gen(series, condition):  # noqa: ANN001, ANN201
        return pd.Series([True, np.nan, True], index=series.index)

    monkeypatch.setitem(strategy_engine.INDICATOR_MAPPING, "nan_ind", base_ind)
    monkeypatch.setattr(strategy_engine, "_generate_signal_from_value", fake_gen)

    rules = {
        "entry_rules": {
            "treat_nan_as_false": True,
            "conditions": [
                {
                    "indicator": "nan_ind",
                    "params": {},
                    "condition": {"type": "indicator_is_above_value", "value": 0},
                }
            ],
        }
    }
    sig, counts = strategy_engine.process_strategy_rules(
        data, rules, collect_counts=True
    )
    assert not sig.isna().any()
    key = next(iter(counts))
    assert counts[key] == 2

    rules["entry_rules"]["treat_nan_as_false"] = False
    sig = strategy_engine.process_strategy_rules(data, rules)
    assert sig.isna().any()
