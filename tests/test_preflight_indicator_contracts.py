import pandas as pd
import pytest

import preflight
import strategy_engine


def _df():
    idx = pd.date_range("2020", periods=30)
    return pd.DataFrame(
        {
            "Open": range(30),
            "High": range(30),
            "Low": range(30),
            "Close": range(30),
        },
        index=idx,
    )


def test_preflight_fails_on_bad_columns(monkeypatch):
    def bad_macd(df, **params):
        return pd.DataFrame({"x": df.Close, "y": df.Close})

    monkeypatch.setitem(strategy_engine.INDICATOR_MAPPING, "macd", bad_macd)

    rules = {
        "entry_rules": {
            "conditions": [
                {"indicator": "macd", "params": {"fast": 12, "slow": 26, "signal": 9}}
            ]
        }
    }
    with pytest.raises(preflight.PreflightError):
        preflight.check_indicator_contracts(_df(), rules)


def test_preflight_fails_on_missing_column(monkeypatch):
    def good_macd(df, **params):
        return df.ta.macd(**params)

    monkeypatch.setitem(strategy_engine.INDICATOR_MAPPING, "macd", good_macd)

    rules = {
        "entry_rules": {
            "conditions": [
                {
                    "indicator": "macd",
                    "params": {"fast": 12, "slow": 26, "signal": 9},
                    "condition": {
                        "type": "indicator_is_above_value",
                        "column": "BAD",
                        "value": 0,
                    },
                }
            ]
        }
    }

    with pytest.raises(preflight.PreflightError):
        preflight.check_indicator_contracts(_df(), rules)
