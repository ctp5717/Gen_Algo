import pandas as pd

from fitness import MultiAssetFitnessEvaluator


def _df():
    idx = pd.date_range("2020", periods=5)
    return pd.DataFrame(
        {
            "Open": [1, 2, 3, 4, 5],
            "High": [1, 2, 3, 4, 5],
            "Low": [1, 2, 3, 4, 5],
            "Close": [1, 2, 3, 4, 5],
        },
        index=idx,
    )


def test_integration_smoke():
    data = {"A": _df()}
    rules = {"entry_rules": {"conditions": []}}
    evaluator = MultiAssetFitnessEvaluator(
        data, rules, {}, settings={"zero_trade_policy": "penalize"}
    )
    fitness = evaluator(None, [], 0)
    details = evaluator.last_details
    per_asset = details["per_asset"].get("A", {})
    assert per_asset.get("reason") != "evaluation_error"
    assert details.get("assets_included", 0) >= 1
    assert "exit_reason_breakdown" in details
    assert isinstance(details.get("exit_reason_breakdown"), dict)
    assert "exit_metrics" in details
    assert isinstance(details.get("exit_metrics"), dict)
    assert fitness != -999
