import pandas as pd

import strategy_engine


def test_indicator_output_cached(monkeypatch):
    df = pd.DataFrame(
        {
            "Open": [1, 2, 3],
            "High": [1, 2, 3],
            "Low": [1, 2, 3],
            "Close": [1, 2, 3],
            "Volume": [1, 2, 3],
        }
    )
    calls = {"count": 0}

    def fake_indicator(ohlc_data, **params):
        calls["count"] += 1
        return pd.Series([1, 1, 1], index=ohlc_data.index)

    monkeypatch.setitem(strategy_engine.INDICATOR_MAPPING, "fake", fake_indicator)

    rules = {
        "entry_rules": {
            "conditions": [
                {
                    "indicator": "fake",
                    "params": {"p": 1},
                    "condition": {"type": "price_is_above_indicator"},
                },
                {
                    "indicator": "fake",
                    "params": {"p": 1},
                    "condition": {"type": "price_is_below_indicator"},
                },
            ]
        }
    }

    strategy_engine.process_strategy_rules(df, rules)
    assert calls["count"] == 1

    strategy_engine.process_strategy_rules(df, rules)
    assert calls["count"] == 2
