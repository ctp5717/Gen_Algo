import sys
import types
from pathlib import Path

import pandas as pd
import pytest

# Ensure repository root is importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Stub optional heavy dependencies before importing project modules
sys.modules.setdefault("pandas_ta", types.ModuleType("pandas_ta"))

vbt = types.ModuleType("vectorbt")
vbt.Portfolio = types.SimpleNamespace()
sys.modules.setdefault("vectorbt", vbt)

import config  # noqa: E402
import fitness  # noqa: E402


def test_dispersion_null_when_metrics_equal():
    m = [1.0, 1.0, 1.0]
    w = [1.0, 1.0, 1.0]
    mu, sigma = fitness.weighted_mean_std(m, w)
    assert mu == 1.0
    assert sigma == 0.0
    for lam in [0.0, 0.25, 2.0]:
        F = mu - lam * sigma
        assert F == pytest.approx(1.0)


def test_dispersion_example_population_stdev():
    m = [1.6, 1.0, 0.4]
    w = [1.0, 1.0, 1.0]
    mu, sigma = fitness.weighted_mean_std(m, w)
    assert mu == pytest.approx(1.0)
    assert sigma == pytest.approx(0.4899, rel=1e-4)
    lam = 0.25
    F = mu - lam * sigma
    assert F == pytest.approx(0.8775, rel=1e-4)


def test_hard_floor_forbids_coverage_penalty(monkeypatch):
    group_data = {
        "A": pd.DataFrame({"Close": [1, 2, 3]}),
        "B": pd.DataFrame({"Close": [1, 2, 3]}),
    }
    settings = {
        "metric": "sortino",
        "lambda_dispersion": 0.0,
        "zero_trade_policy": "ignore",
        "coverage_penalty": 1.0,
        "trade_floor_policy": "hard_floor",
        "min_total_trades": 5,
        "per_asset_min_trades": 1,
        "min_included_assets": 1,
    }
    evaluator = fitness.MultiAssetFitnessEvaluator(group_data, {}, {}, settings)

    stats_seq = [
        {
            "sortino": 1.0,
            "profit_factor": 1.0,
            "max_drawdown": 0.0,
            "trades": 1,
            "total_return": 0.0,
            "equity_curve": pd.Series(dtype=float),
        },
        {
            "sortino": 1.0,
            "profit_factor": 1.0,
            "max_drawdown": 0.0,
            "trades": 0,
            "total_return": 0.0,
            "equity_curve": pd.Series(dtype=float),
        },
    ]

    def fake_eval(self, ohlc, rules):
        return stats_seq.pop(0)

    monkeypatch.setattr(
        fitness.MultiAssetFitnessEvaluator, "_evaluate_single_asset", fake_eval
    )

    score = evaluator(None, [], 0)
    assert score == evaluator.settings.get("poor_score", -999.0)
    details = evaluator.last_details
    assert details["penalties"]["coverage"] == 0.0
    assert details["penalties"]["trade_floor"] == "below_group_floor"


def test_zero_trade_penalize_includes_asset(monkeypatch):
    group_data = {
        "A": pd.DataFrame({"Close": [1, 2, 3]}),
        "B": pd.DataFrame({"Close": [1, 2, 3]}),
    }
    settings = {
        "metric": "sortino",
        "lambda_dispersion": 0.0,
        "zero_trade_policy": "penalize",
        "zero_trade_penalty": -2.0,
        "min_total_trades": 0,
        "per_asset_min_trades": 1,
        "min_included_assets": 1,
        "coverage_penalty": 0.0,
    }
    evaluator = fitness.MultiAssetFitnessEvaluator(group_data, {}, {}, settings)

    stats_seq = [
        {
            "sortino": 1.0,
            "profit_factor": 1.0,
            "max_drawdown": 0.0,
            "trades": 1,
            "total_return": 0.0,
            "equity_curve": pd.Series(dtype=float),
        },
        {
            "sortino": 0.0,
            "profit_factor": 1.0,
            "max_drawdown": 0.0,
            "trades": 0,
            "total_return": 0.0,
            "equity_curve": pd.Series(dtype=float),
        },
    ]

    def fake_eval(self, ohlc, rules):
        return stats_seq.pop(0)

    monkeypatch.setattr(
        fitness.MultiAssetFitnessEvaluator, "_evaluate_single_asset", fake_eval
    )

    score = evaluator(None, [], 0)
    details = evaluator.last_details
    assert details["assets_included"] == 2
    assert details["per_asset"]["B"]["included"] is True
    assert details["per_asset"]["B"]["score"] == -2.0
    assert score == pytest.approx(-0.5)


def test_zero_trade_ignore_excludes_asset(monkeypatch):
    group_data = {
        "A": pd.DataFrame({"Close": [1, 2, 3]}),
        "B": pd.DataFrame({"Close": [1, 2, 3]}),
    }
    settings = {
        "metric": "sortino",
        "lambda_dispersion": 0.0,
        "zero_trade_policy": "ignore",
        "min_total_trades": 0,
        "per_asset_min_trades": 1,
        "min_included_assets": 1,
        "coverage_penalty": 0.3,
    }
    evaluator = fitness.MultiAssetFitnessEvaluator(group_data, {}, {}, settings)

    stats_seq = [
        {
            "sortino": 1.0,
            "profit_factor": 1.0,
            "max_drawdown": 0.0,
            "trades": 1,
            "total_return": 0.0,
            "equity_curve": pd.Series(dtype=float),
        },
        {
            "sortino": 0.0,
            "profit_factor": 1.0,
            "max_drawdown": 0.0,
            "trades": 0,
            "total_return": 0.0,
            "equity_curve": pd.Series(dtype=float),
        },
    ]

    def fake_eval(self, ohlc, rules):
        return stats_seq.pop(0)

    monkeypatch.setattr(
        fitness.MultiAssetFitnessEvaluator, "_evaluate_single_asset", fake_eval
    )

    score = evaluator(None, [], 0)
    details = evaluator.last_details
    assert details["assets_included"] == 1
    assert details["per_asset"]["B"]["included"] is False
    assert details["per_asset"]["B"]["reason"] == "ignored_zero_trades"
    assert details["penalties"]["coverage"] == pytest.approx(0.15)
    assert score == pytest.approx(0.85)


def test_negative_weights_clipped_and_renormalized(monkeypatch):
    group_data = {
        "A": pd.DataFrame({"Close": [1, 2]}),
        "B": pd.DataFrame({"Close": [1, 2]}),
    }
    settings = {
        "metric": "sortino",
        "lambda_dispersion": 0.0,
        "asset_weights": {"A": 2.0, "B": -1.0},
        "per_asset_min_trades": 1,
        "min_included_assets": 1,
        "coverage_penalty": 0.0,
    }
    evaluator = fitness.MultiAssetFitnessEvaluator(group_data, {}, {}, settings)

    stats_seq = [
        {
            "sortino": 1.0,
            "profit_factor": 1.0,
            "max_drawdown": 0.0,
            "trades": 1,
            "total_return": 0.0,
            "equity_curve": pd.Series(dtype=float),
        },
        {
            "sortino": 1.0,
            "profit_factor": 1.0,
            "max_drawdown": 0.0,
            "trades": 1,
            "total_return": 0.0,
            "equity_curve": pd.Series(dtype=float),
        },
    ]

    def fake_eval(self, ohlc, rules):
        return stats_seq.pop(0)

    monkeypatch.setattr(
        fitness.MultiAssetFitnessEvaluator, "_evaluate_single_asset", fake_eval
    )

    evaluator(None, [], 0)
    weights = evaluator.last_details["asset_weights"]
    assert weights["A"] == pytest.approx(1.0)
    assert weights["B"] == pytest.approx(0.0)
    assert pytest.approx(sum(weights.values())) == 1.0


def test_fitness_evaluator_uses_to_pandas_freq(monkeypatch):
    monkeypatch.setattr(config, "TIMEFRAME", "15m", raising=False)
    ohlc = pd.DataFrame({"Close": [1, 2, 3]})
    evaluator = fitness.FitnessEvaluator(ohlc, {}, {})

    def fake_process(data, rules):
        return pd.Series([True, False, False])

    monkeypatch.setattr(fitness.engine, "process_strategy_rules", fake_process)

    captured = {}

    def fake_from_signals(**kwargs):
        captured["freq"] = kwargs.get("freq")

        class PF:
            def __init__(self, trades):
                self._trades = types.SimpleNamespace(count=lambda: trades)

            def stats(self):
                return {
                    "Sortino Ratio": 1.0,
                    "Profit Factor": 1.0,
                    "Max Drawdown [%]": 0.0,
                }

        return PF(trades=1)

    monkeypatch.setattr(
        fitness.vbt.Portfolio, "from_signals", fake_from_signals, raising=False
    )

    evaluator(None, [], 0)
    assert captured["freq"] == "15min"
