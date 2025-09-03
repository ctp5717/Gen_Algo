import json
import types

import numpy as np
import pandas as pd

import analysis
import config
import fitness


def _make_evaluator(stats_list, settings=None, group_data=None):
    group_data = group_data or {
        "A": pd.DataFrame({"Close": [1, 2, 3]}),
        "B": pd.DataFrame({"Close": [1, 2, 3]}),
        "C": pd.DataFrame({"Close": [1, 2, 3]}),
    }
    base = {
        "per_asset_min_trades": 1,
        "min_included_assets": 1,
        "coverage_penalty": 0.0,
    }
    if settings:
        base.update(settings)
    evaluator = fitness.MultiAssetFitnessEvaluator(group_data, {}, {}, base)
    stats_iter = iter(stats_list)

    def fake_eval(self, ohlc, rules):
        return next(stats_iter)

    evaluator._evaluate_single_asset = types.MethodType(fake_eval, evaluator)
    return evaluator


def test_weighted_aggregation_regression():
    vals = [1.6, 1.0, 0.4]
    w = [1 / 3, 1 / 3, 1 / 3]
    mu, sigma = fitness.weighted_mean_std(vals, w)
    lam = 0.25
    F = mu - lam * sigma
    assert np.isclose(mu, 1.0)
    assert np.isclose(sigma, 0.4899, atol=1e-4)
    assert np.isclose(F, 0.8775, atol=1e-4)


def test_weights_sum_to_one():
    stats = [
        {"total_return": 1.0, "trades": 5},
        {"total_return": 0.5, "trades": 5},
        {"total_return": 0.0, "trades": 5},
    ]
    settings = {
        "metric": "return",
        "lambda_dispersion": 0.0,
        "trade_floor_policy": "hard_floor",
        "min_total_trades": 0,
        "asset_weights": {"A": 3, "B": 1, "C": 1},
    }
    ev = _make_evaluator(stats, settings)
    ev(None, [], 0)
    details = ev.last_details
    inc = [t for t, d in details["per_asset"].items() if d["included"]]
    total = sum(details["per_asset"][t]["asset_weight"] for t in inc)
    assert np.isclose(total, 1.0)


def test_hard_floor_no_coverage_penalty():
    stats = [
        {"total_return": 1.0, "trades": 1},
        {"total_return": 1.0, "trades": 1},
        {"total_return": 1.0, "trades": 1},
    ]
    settings = {
        "metric": "return",
        "lambda_dispersion": 0.0,
        "trade_floor_policy": "hard_floor",
        "min_total_trades": 10,
        "zero_trade_policy": "ignore",
        "coverage_penalty": 0.3,
        "poor_score": -999.0,
    }
    ev = _make_evaluator(stats, settings)
    score = ev(None, [], 0)
    assert score == -999.0
    assert ev.last_details["penalties"]["coverage"] == 0.0


def test_deterministic_order_and_score():
    group = {
        "B": pd.DataFrame({"Close": [1, 2, 3]}),
        "A": pd.DataFrame({"Close": [1, 2, 3]}),
        "C": pd.DataFrame({"Close": [1, 2, 3]}),
    }
    stats = [
        {"total_return": 1.0, "trades": 5},
        {"total_return": 2.0, "trades": 5},
        {"total_return": 3.0, "trades": 5},
    ]
    settings = {
        "metric": "return",
        "lambda_dispersion": 0.0,
        "trade_floor_policy": "hard_floor",
        "min_total_trades": 0,
    }
    ev1 = _make_evaluator(stats, settings, group)
    f1 = ev1(None, [], 0)
    rows1 = pd.DataFrame(
        [
            {"ticker": t, "score": d["score"]}
            for t, d in ev1.last_details["per_asset"].items()
            if d["score"] is not None
        ]
    ).sort_values("score", ascending=False)
    order1 = rows1["ticker"].tolist()

    ev2 = _make_evaluator(stats, settings, group)
    f2 = ev2(None, [], 0)
    rows2 = pd.DataFrame(
        [
            {"ticker": t, "score": d["score"]}
            for t, d in ev2.last_details["per_asset"].items()
            if d["score"] is not None
        ]
    ).sort_values("score", ascending=False)
    order2 = rows2["ticker"].tolist()

    assert f1 == f2
    assert order1 == order2


def test_csv_and_json_include_exclusions(tmp_path, monkeypatch):
    group = {
        "A": pd.DataFrame({"Close": [1, 2, 3]}),
        "B": pd.DataFrame({"Close": [1, 2, 3]}),
    }
    stats = [
        {"total_return": 0.0, "trades": 0},
        {"total_return": 1.0, "trades": 5},
    ]

    stats_iter = iter(stats)

    def fake_eval(self, ohlc, rules):
        return next(stats_iter)

    monkeypatch.setattr(
        fitness.MultiAssetFitnessEvaluator, "_evaluate_single_asset", fake_eval
    )
    monkeypatch.setitem(config.MULTI_ASSET, "metric", "return")
    monkeypatch.setitem(config.MULTI_ASSET, "lambda_dispersion", 0.0)
    monkeypatch.setitem(config.MULTI_ASSET, "trade_floor_policy", "hard_floor")
    monkeypatch.setitem(config.MULTI_ASSET, "min_total_trades", 0)
    monkeypatch.setitem(config.MULTI_ASSET, "min_total_trades_per_year", 0)
    monkeypatch.setitem(config.MULTI_ASSET, "asset_weights", {"A": 1, "B": 1})
    monkeypatch.setitem(config.MULTI_ASSET, "per_asset_min_trades", 1)
    monkeypatch.setitem(config.MULTI_ASSET, "min_included_assets", 1)
    monkeypatch.setitem(config.MULTI_ASSET, "coverage_penalty", 0.0)
    monkeypatch.setattr(analysis, "_plot_multi_asset_overview", lambda *a, **k: None)
    monkeypatch.chdir(tmp_path)

    analysis._run_multi_asset_analysis([], {}, group)

    csv_file = next(tmp_path.glob("multi_asset_stats_*.csv"))
    df = pd.read_csv(csv_file)
    assert set(df["ticker"]) == {"A", "B"}
    assert not bool(df.loc[df["ticker"] == "A", "included"].item())
    assert df.loc[df["ticker"] == "A", "reason"].item() != ""
    assert np.isclose(df[df["included"]]["asset_weight"].sum(), 1.0)

    json_file = next(tmp_path.glob("multi_asset_summary_*.json"))
    summary = json.loads(json_file.read_text())
    assert np.isclose(sum(summary["asset_weights"].values()), 1.0)
    assert set(summary["asset_weights"].keys()) == {"B"}
    assert summary["assets_ignored"] == 1


def test_evaluation_error_reason(tmp_path, monkeypatch):
    group = {
        "A": pd.DataFrame({"Close": [1, 2, 3]}),
        "B": pd.DataFrame({"Close": [1, 2, 3]}),
    }
    stats = [Exception("boom"), {"total_return": 1.0, "trades": 5}]

    stats_iter = iter(stats)

    def fake_eval(self, ohlc, rules):
        val = next(stats_iter)
        if isinstance(val, Exception):
            raise val
        return val

    monkeypatch.setattr(
        fitness.MultiAssetFitnessEvaluator, "_evaluate_single_asset", fake_eval
    )
    monkeypatch.setitem(config.MULTI_ASSET, "metric", "return")
    monkeypatch.setitem(config.MULTI_ASSET, "lambda_dispersion", 0.0)
    monkeypatch.setitem(config.MULTI_ASSET, "trade_floor_policy", "hard_floor")
    monkeypatch.setitem(config.MULTI_ASSET, "min_total_trades", 0)
    monkeypatch.setitem(config.MULTI_ASSET, "min_total_trades_per_year", 0)
    monkeypatch.setitem(config.MULTI_ASSET, "asset_weights", {"A": 1, "B": 1})
    monkeypatch.setitem(config.MULTI_ASSET, "per_asset_min_trades", 1)
    monkeypatch.setitem(config.MULTI_ASSET, "min_included_assets", 1)
    monkeypatch.setitem(config.MULTI_ASSET, "coverage_penalty", 0.0)
    monkeypatch.setattr(analysis, "_plot_multi_asset_overview", lambda *a, **k: None)
    monkeypatch.chdir(tmp_path)

    analysis._run_multi_asset_analysis([], {}, group)

    csv_file = next(tmp_path.glob("multi_asset_stats_*.csv"))
    df = pd.read_csv(csv_file)
    assert df.loc[df["ticker"] == "A", "reason"].item() == "evaluation_error"
    assert np.isclose(df[df["included"]]["asset_weight"].sum(), 1.0)

    json_file = next(tmp_path.glob("multi_asset_summary_*.json"))
    summary = json.loads(json_file.read_text())
    assert set(summary["asset_weights"].keys()) == {"B"}
    assert summary["assets_ignored"] == 1
