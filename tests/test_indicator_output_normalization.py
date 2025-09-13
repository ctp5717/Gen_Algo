from collections import namedtuple

import numpy as np
import pandas as pd

import indicator_contracts as contracts
import strategy_engine


def _df():
    idx = pd.date_range("2020", periods=3)
    return pd.DataFrame(
        {
            "Open": [1, 2, 3],
            "High": [1, 2, 3],
            "Low": [1, 2, 3],
            "Close": [1, 2, 3],
        },
        index=idx,
    )


def test_dict_output_normalization(monkeypatch):
    def ind(ohlc):
        s1 = pd.Series([1, 2, 3], index=ohlc.index)
        s2 = pd.Series([4, 5, 6], index=ohlc.index)
        return {"a": s1, "b": s2}

    contracts.CONTRACTS["dictind"] = lambda **_: ["a", "b"]
    strategy_engine.INDICATOR_MAPPING["dictind"] = ind

    rules = {
        "entry_rules": {
            "conditions": [
                {
                    "indicator": "dictind",
                    "condition": {
                        "type": "indicator_is_above_value",
                        "column": "a",
                        "value": 0,
                    },
                    "params": {},
                }
            ]
        }
    }
    sig = strategy_engine.process_strategy_rules(_df(), rules)
    assert isinstance(sig, pd.Series)


def test_namedtuple_output_normalization(monkeypatch):
    NT = namedtuple("NT", ["a", "b"])

    def ind(ohlc):
        s1 = pd.Series([1, 2, 3], index=ohlc.index)
        s2 = pd.Series([4, 5, 6], index=ohlc.index)
        return NT(s1, s2)

    contracts.CONTRACTS["ntind"] = lambda **_: ["a", "b"]
    strategy_engine.INDICATOR_MAPPING["ntind"] = ind

    rules = {
        "entry_rules": {
            "conditions": [
                {
                    "indicator": "ntind",
                    "condition": {
                        "type": "indicator_is_above_value",
                        "column": "a",
                        "value": 0,
                    },
                    "params": {},
                }
            ]
        }
    }
    sig = strategy_engine.process_strategy_rules(_df(), rules)
    assert isinstance(sig, pd.Series)


def test_tuple_arrays_align_to_index(monkeypatch):
    def ind(ohlc):
        a = np.arange(len(ohlc), dtype=float)
        return (a, a, a)

    contracts.CONTRACTS["arrind"] = lambda **_: ["A", "B", "C"]
    strategy_engine.INDICATOR_MAPPING["arrind"] = ind

    rules = {
        "entry_rules": {
            "conditions": [
                {
                    "indicator": "arrind",
                    "condition": {
                        "type": "indicator_is_above_value",
                        "column": "A",
                        "value": 0,
                    },
                    "params": {},
                }
            ]
        }
    }
    df = _df()
    sig = strategy_engine.process_strategy_rules(df, rules)
    assert sig.index.equals(df.index)


def test_dataframe_superset_columns():
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4], "extra": [5, 6]})
    contracts.CONTRACTS["dfsup"] = lambda **_: ["A", "B"]
    out = contracts.normalize_output("dfsup", df, {})
    assert list(out.columns) == ["A", "B"]
