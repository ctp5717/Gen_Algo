import ast
import logging
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

# Ensure repository root on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Stub optional heavy deps
sys.modules.setdefault("pandas_ta", types.ModuleType("pandas_ta"))
sys.modules.setdefault("vectorbt", types.ModuleType("vectorbt"))

import gene_parser  # noqa: E402
import indicator_library  # noqa: E402,F401
import strategy_engine  # noqa: E402


def _make_cond(name, series):
    def ind(ohlc, period=None):
        return pd.Series(series, index=ohlc.index)

    return ind, {
        "indicator": name,
        "params": {},
        "condition": {"type": "indicator_is_above_value", "value": 0.5},
    }


@pytest.mark.parametrize("comb_logic", ["vote", "Vote"])
def test_vote_threshold_none_and_casing(monkeypatch, comb_logic, caplog):
    data = pd.DataFrame({"Close": [1, 1]}, index=pd.date_range("2020", periods=2))
    ind_a, cond_a = _make_cond("a", [1, 0])
    ind_b, cond_b = _make_cond("b", [0, 1])
    monkeypatch.setitem(strategy_engine.INDICATOR_MAPPING, "a", ind_a)
    monkeypatch.setitem(strategy_engine.INDICATOR_MAPPING, "b", ind_b)

    captured = {}
    orig = strategy_engine._combine_signals

    def spy(
        signals, combination_logic, vote_threshold, nan_policy, ffill_lookback=None
    ):
        captured["combination_logic"] = combination_logic
        captured["vote_threshold"] = vote_threshold
        return orig(signals, combination_logic, vote_threshold, nan_policy)

    monkeypatch.setattr(strategy_engine, "_combine_signals", spy)

    rules = {
        "entry_rules": {
            "combination_logic": comb_logic,
            "conditions": [cond_a, cond_b],
        }
    }

    with caplog.at_level(logging.INFO):
        res = strategy_engine.process_strategy_rules(data, rules)

    record = next(r for r in caplog.records if r.levelno == logging.INFO)
    payload = ast.literal_eval(record.getMessage())
    assert payload == {
        "logic": "VOTE",
        "M": 2,
        "k": 1,
        "nan_policy": "FALSE",
    }

    # Expect majority threshold (ceil(2/2) == 1) and sanitized logic
    assert captured["combination_logic"] == "VOTE"
    assert captured["vote_threshold"] == 1
    pd.testing.assert_series_equal(
        res.astype(bool), pd.Series([True, True], index=data.index)
    )


def test_vote_threshold_clamped_to_one(monkeypatch):
    data = pd.DataFrame({"Close": [1, 1]}, index=pd.date_range("2020", periods=2))
    ind_a, cond = _make_cond("a", [1, 1])
    monkeypatch.setitem(strategy_engine.INDICATOR_MAPPING, "a", ind_a)

    def fail(*args, **kwargs):  # _combine_signals should not be called for single cond
        raise AssertionError("_combine_signals should not be invoked")

    monkeypatch.setattr(strategy_engine, "_combine_signals", fail)

    rules = {
        "entry_rules": {
            "combination_logic": "VOTE",
            "vote_threshold": 5,
            "conditions": [cond],
        }
    }

    with pytest.warns(
        RuntimeWarning, match="Single active condition; normalized combination_logic"
    ):
        res = strategy_engine.process_strategy_rules(data, rules)
    pd.testing.assert_series_equal(
        res.astype(bool), pd.Series([True, True], index=data.index)
    )


def test_vote_threshold_exceeds_active(monkeypatch, caplog):
    strategy_engine.clear_indicator_cache()
    data = pd.DataFrame({"Close": [1, 1]}, index=pd.date_range("2020", periods=2))
    ind_a, cond_a = _make_cond("a", [1, 0])
    ind_b, cond_b = _make_cond("b", [1, 1])
    monkeypatch.setitem(strategy_engine.INDICATOR_MAPPING, "a", ind_a)
    monkeypatch.setitem(strategy_engine.INDICATOR_MAPPING, "b", ind_b)

    captured = {}
    orig = strategy_engine._combine_signals

    def spy(
        signals, combination_logic, vote_threshold, nan_policy, ffill_lookback=None
    ):
        captured["combination_logic"] = combination_logic
        captured["vote_threshold"] = vote_threshold
        return orig(signals, combination_logic, vote_threshold, nan_policy)

    monkeypatch.setattr(strategy_engine, "_combine_signals", spy)

    rules = {
        "entry_rules": {
            "combination_logic": "vote",
            "vote_threshold": 5,
            "conditions": [cond_a, cond_b],
        }
    }

    with caplog.at_level(logging.INFO):
        with pytest.warns(
            RuntimeWarning, match="vote_threshold exceeds active conditions"
        ):
            res = strategy_engine.process_strategy_rules(data, rules)

    record = next(r for r in caplog.records if r.levelno == logging.INFO)
    payload = ast.literal_eval(record.getMessage())
    assert payload == {
        "logic": "VOTE",
        "M": 2,
        "k": 2,
        "nan_policy": "FALSE",
    }

    assert captured["combination_logic"] == "VOTE"
    assert captured["vote_threshold"] == 2
    expected = pd.Series([True, False], index=data.index, dtype=bool)
    pd.testing.assert_series_equal(res.astype(bool), expected)


def test_invalid_threshold_raises(monkeypatch):
    data = pd.DataFrame({"Close": [1, 1]}, index=pd.date_range("2020", periods=2))
    ind_a, cond_a = _make_cond("a", [1, 0])
    ind_b, cond_b = _make_cond("b", [0, 1])
    monkeypatch.setitem(strategy_engine.INDICATOR_MAPPING, "a", ind_a)
    monkeypatch.setitem(strategy_engine.INDICATOR_MAPPING, "b", ind_b)

    rules = {
        "entry_rules": {
            "combination_logic": "AND",
            "vote_threshold": 0,
            "conditions": [cond_a, cond_b],
        }
    }

    with pytest.raises(AssertionError):
        strategy_engine.process_strategy_rules(data, rules)


def test_parse_genes_clamps_vote_threshold():
    rules = {
        "entry_rules": {
            "combination_logic": "VOTE",
            "vote_threshold": {"gene": "vote_threshold", "low": 0, "high": 10},
            "conditions": [
                {"indicator": "a", "params": {}, "condition": {}, "is_active": True}
                for _ in range(3)
            ],
        }
    }
    gs, gm, _ = gene_parser.parse_genes_from_config(rules)
    idx = next(i for i, g in gm.items() if g["name"] == "vote_threshold")
    assert gs[idx]["high"] == 3
    assert gs[idx]["low"] == 1
    vt = rules["entry_rules"]["vote_threshold"]
    assert vt["high"] == 3
    assert vt["low"] == 1
