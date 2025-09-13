import pandas as pd

import indicator_contracts as contracts
import strategy_engine
from fitness import MultiAssetFitnessEvaluator


def _df():
    idx = pd.date_range("2020", periods=5)
    return pd.DataFrame(
        {
            "Open": range(5),
            "High": range(5),
            "Low": range(5),
            "Close": range(5),
        },
        index=idx,
    )


def test_reason_detail_contains_indicator():
    def bad(df, **_):
        s = pd.Series(range(len(df)), index=df.index)
        return (s,)  # wrong length for macd contract

    contracts.CONTRACTS["bad"] = lambda **_: ["x", "y"]
    strategy_engine.INDICATOR_MAPPING["bad"] = bad

    rules = {
        "entry_rules": {
            "conditions": [
                {
                    "indicator": "bad",
                    "params": {},
                    "condition": {
                        "type": "indicator_is_above_value",
                        "column": "x",
                        "value": 0,
                    },
                }
            ]
        }
    }
    evaluator = MultiAssetFitnessEvaluator(
        {"A": _df()}, rules, {}, settings={"zero_trade_policy": "penalize"}
    )
    evaluator(None, [], 0)
    detail = evaluator.last_details["per_asset"].get("A", {}).get("reason_detail", "")
    assert "bad" in detail and "tuple" in detail
