import concurrent.futures as cf
import sys
import types
from pathlib import Path

# Ensure repository root is on the import path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Stub heavy optional dependencies
sys.modules.setdefault("pandas_ta", types.ModuleType("pandas_ta"))
try:  # prefer real vectorbt
    import vectorbt  # noqa: F401
except Exception:  # pragma: no cover
    sys.modules.setdefault("vectorbt", types.ModuleType("vectorbt"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pygad  # noqa: E402
import pytest  # noqa: E402

import analysis  # noqa: E402
import config as cfg  # noqa: E402
import data_loader  # noqa: E402
import fitness  # noqa: E402
import tuner  # noqa: E402
from utils.math import weighted_mean_std  # noqa: E402

cfg.initialize_config()


def _make_evaluator(settings=None, stats_list=None):
    """Utility to construct a MultiAssetFitnessEvaluator with patched stats."""
    group_data = {
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

    if stats_list is not None:
        stats_iter = iter(stats_list)

        def fake_eval(self, ohlc, rules):
            return next(stats_iter)

        evaluator._evaluate_single_asset = types.MethodType(fake_eval, evaluator)
    return evaluator


def test_dispersion_math_sanity():
    vals = np.array([1.6, 1.0, 0.4], dtype=float)
    w = np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)
    mu, sigma = weighted_mean_std(vals, w)
    assert np.isclose(mu, 1.0, atol=1e-6)
    assert np.isclose(sigma, 0.4898979, atol=1e-6)
    lam = 0.25
    F = mu - lam * sigma
    assert np.isclose(F, 0.8775255, atol=1e-6)


def test_aggregation_math():
    stats = [
        {"total_return": 1.6, "trades": 5},
        {"total_return": 1.0, "trades": 5},
        {"total_return": 0.4, "trades": 5},
    ]
    settings = {
        "metric": "return",
        "lambda_dispersion": 0.25,
        "trade_floor_policy": "hard_floor",
        "min_total_trades": 0,
        "per_asset_min_trades": 1,
    }
    ev = _make_evaluator(settings, stats)
    score = ev(None, [], 0)
    assert np.isclose(ev.last_details["mu"], 1.0)
    assert np.isclose(ev.last_details["sigma"], 0.4899, atol=1e-4)
    assert np.isclose(score, 0.8775, atol=1e-4)


def test_trade_floor_policies():
    stats = [
        {"total_return": 1.0, "trades": 5},
        {"total_return": 1.0, "trades": 5},
        {"total_return": 1.0, "trades": 5},
    ]
    settings_hard = {
        "metric": "return",
        "trade_floor_policy": "hard_floor",
        "min_total_trades": 20,
        "per_asset_min_trades": 1,
    }
    ev_hard = _make_evaluator(settings_hard, stats)
    assert ev_hard(None, [], 0) == -999.0

    settings_soft = {
        "metric": "return",
        "trade_floor_policy": "soft_penalty",
        "min_total_trades": 30,
        "soft_penalty_strength": 1.0,
        "per_asset_min_trades": 1,
    }
    ev_soft = _make_evaluator(settings_soft, stats)
    assert np.isclose(ev_soft(None, [], 0), 0.5)


def test_unreachable_floor_warns():
    with pytest.warns(UserWarning, match="min_total_trades <"):
        _make_evaluator(
            {
                "min_total_trades": 5,
                "per_asset_min_trades": 3,
                "min_included_assets": 2,
                "trade_floor_policy": "hard_floor",
            }
        )


def test_soft_penalty_auto_adjusts():
    ev = _make_evaluator(
        {
            "min_total_trades": 5,
            "per_asset_min_trades": 3,
            "min_included_assets": 2,
            "trade_floor_policy": "soft_penalty",
        }
    )
    assert ev.settings["min_total_trades"] == 6


def test_min_assets_clamped():
    ev = _make_evaluator({"min_included_assets": 10})
    assert ev.settings["min_included_assets"] == 3


def test_min_total_trades_scaling(monkeypatch):
    monkeypatch.setitem(cfg.MULTI_ASSET, "enabled", True)
    monkeypatch.setitem(cfg.MULTI_ASSET, "min_total_trades_per_year", 24)
    monkeypatch.setitem(cfg.MULTI_ASSET, "trade_floor_policy", "hard_floor")
    monkeypatch.setattr(
        data_loader,
        "get_group_data",
        lambda *a, **k: {"A": pd.DataFrame({"Close": [1, 2]})},
    )
    monkeypatch.setattr(
        pd.DataFrame,
        "ta",
        property(lambda self: None),
        raising=False,
    )
    vbt = sys.modules["vectorbt"]
    monkeypatch.setattr(vbt, "Portfolio", object, raising=False)

    captured = {}

    class DummyEval:
        def __init__(self, group_data, rules, gene_map, settings):
            captured["settings"] = settings

        def __call__(self, ga, sol, idx):
            return 0.0

    monkeypatch.setattr(fitness, "MultiAssetFitnessEvaluator", DummyEval)
    monkeypatch.setattr(
        cfg, "VALIDATION_PERIOD", {"start": "2024-01-01", "end": "2024-02-01"}
    )

    val = {"A": pd.DataFrame({"Close": [1, 2]})}
    tuner._evaluate_on_validation([], {}, val)
    assert captured["settings"]["min_total_trades"] == 3
    assert captured["settings"]["trade_floor_policy"] == "soft_penalty"
    assert captured["settings"]["soft_penalty_mode"] == "multiplicative"


def test_training_floor_scaling(monkeypatch):
    monkeypatch.setitem(cfg.MULTI_ASSET, "enabled", True)
    monkeypatch.setitem(cfg.MULTI_ASSET, "min_total_trades_per_year", 12)
    monkeypatch.setattr(
        cfg, "TRAINING_PERIOD", {"start": "2020-01-01", "end": "2021-01-01"}
    )
    ev = fitness.get_fitness_evaluator({"A": pd.DataFrame({"Close": [1]})}, {}, {})
    assert ev.settings["min_total_trades"] == 18


def test_zero_trade_policy_penalize_vs_ignore():
    stats = [
        {"total_return": 0.0, "trades": 0},
        {"total_return": 1.0, "trades": 5},
        {"total_return": 1.0, "trades": 5},
    ]
    penalize_settings = {
        "metric": "return",
        "zero_trade_policy": "penalize",
        "zero_trade_penalty": -1.0,
        "per_asset_min_trades": 1,
        "trade_floor_policy": "hard_floor",
        "min_total_trades": 0,
        "lambda_dispersion": 0.0,
    }
    ev_pen = _make_evaluator(penalize_settings, stats)
    assert np.isclose(ev_pen(None, [], 0), 1 / 3, atol=1e-6)

    ignore_settings = {
        "metric": "return",
        "zero_trade_policy": "ignore",
        "coverage_penalty": 0.3,
        "per_asset_min_trades": 1,
        "trade_floor_policy": "hard_floor",
        "min_total_trades": 0,
        "lambda_dispersion": 0.0,
    }
    ev_ign = _make_evaluator(ignore_settings, stats)
    assert np.isclose(ev_ign(None, [], 0), 0.9)


def test_stability_regularizer_off(monkeypatch):
    stats = [
        {"total_return": 1.0, "trades": 5},
        {"total_return": 2.0, "trades": 5},
    ]
    settings = {
        "metric": "return",
        "lambda_dispersion": 0.0,
        "trade_floor_policy": "hard_floor",
        "min_total_trades": 0,
        "per_asset_min_trades": 1,
        "param_history": [{"rsi_period": 7}, {"rsi_period": 21}],
    }
    ev = _make_evaluator(settings, stats)
    monkeypatch.setattr(cfg, "ENABLE_STABILITY_REG", False)
    monkeypatch.setattr(cfg, "STABILITY_ALPHA", 1.0)
    monkeypatch.setattr(cfg, "STABILITY_GENES", ["rsi_period"])
    assert np.isclose(ev(None, [], 0), 1.5)


def test_stability_regularizer_penalizes(monkeypatch):
    stats = [
        {"total_return": 1.0, "trades": 5},
        {"total_return": 2.0, "trades": 5},
    ]
    settings = {
        "metric": "return",
        "lambda_dispersion": 0.0,
        "trade_floor_policy": "hard_floor",
        "min_total_trades": 0,
        "per_asset_min_trades": 1,
        "param_history": [{"rsi_period": 7}, {"rsi_period": 21}],
    }
    ev = _make_evaluator(settings, stats)
    monkeypatch.setattr(cfg, "ENABLE_STABILITY_REG", True)
    monkeypatch.setattr(cfg, "STABILITY_ALPHA", 1.0)
    monkeypatch.setattr(cfg, "STABILITY_GENES", ["rsi_period"])
    score = ev(None, [], 0)
    assert np.isclose(score, 1.0)
    assert np.isclose(ev.last_details["penalties"]["stability"], 0.5)


def test_sentinel_ignores_coverage_penalty():
    stats = [
        {"total_return": 1.0, "trades": 1},
        {"total_return": 1.0, "trades": 0},
        {"total_return": 1.0, "trades": 0},
    ]
    settings = {
        "metric": "return",
        "zero_trade_policy": "ignore",
        "coverage_penalty": 0.3,
        "per_asset_min_trades": 1,
        "trade_floor_policy": "hard_floor",
        "min_total_trades": 5,
        "lambda_dispersion": 0.0,
        "poor_score": -999.0,
    }
    ev = _make_evaluator(settings, stats)
    assert ev(None, [], 0) == -999.0


def test_min_included_assets_hard_floor():
    stats = [
        {"total_return": 1.0, "trades": 5},
        {"total_return": 1.0, "trades": 5},
        {"total_return": 1.0, "trades": 0},
    ]
    settings = {
        "metric": "return",
        "zero_trade_policy": "ignore",
        "per_asset_min_trades": 1,
        "trade_floor_policy": "hard_floor",
        "min_total_trades": 0,
        "lambda_dispersion": 0.0,
        "min_included_assets": 3,
    }
    ev = _make_evaluator(settings, stats)
    assert ev(None, [], 0) == -999.0
    assert ev.last_details["penalties"]["trade_floor"] == "below_min_included_assets"
    assert ev.last_details["penalties"]["min_assets"] == "below_min_included_assets"


def test_min_included_assets_soft_penalty():
    stats = [
        {"total_return": 1.0, "trades": 5},
        {"total_return": 1.0, "trades": 5},
        {"total_return": 1.0, "trades": 0},
    ]
    settings = {
        "metric": "return",
        "zero_trade_policy": "ignore",
        "coverage_penalty": 0.0,
        "per_asset_min_trades": 1,
        "trade_floor_policy": "soft_penalty",
        "min_total_trades": 0,
        "lambda_dispersion": 0.0,
        "min_included_assets": 3,
        "soft_penalty_strength": 1.0,
    }
    ev = _make_evaluator(settings, stats)
    expected = 2 / 3
    assert np.isclose(ev(None, [], 0), expected)
    assert np.isclose(ev.last_details["penalties"]["min_assets"]["scale"], expected)


def test_weight_renormalization():
    stats = [
        {"total_return": 1.0, "trades": 5},
        {"total_return": 0.0, "trades": 5},
        {"total_return": 0.0, "trades": 0},
    ]
    settings = {
        "metric": "return",
        "zero_trade_policy": "ignore",
        "coverage_penalty": 0.3,
        "per_asset_min_trades": 1,
        "asset_weights": {"A": 0.6, "B": 0.2, "C": 0.2},
        "trade_floor_policy": "hard_floor",
        "min_total_trades": 0,
        "lambda_dispersion": 0.0,
    }
    ev = _make_evaluator(settings, stats)
    assert np.isclose(ev(None, [], 0), 0.65, atol=1e-6)


def test_soft_penalty_additive():
    stats = [
        {"total_return": 1.0, "trades": 5},
        {"total_return": 1.0, "trades": 5},
        {"total_return": 1.0, "trades": 5},
    ]
    settings = {
        "metric": "return",
        "trade_floor_policy": "soft_penalty",
        "soft_penalty_mode": "additive",
        "soft_penalty_strength": 2.0,
        "min_total_trades": 30,
        "per_asset_min_trades": 1,
    }
    ev = _make_evaluator(settings, stats)
    # Mean = 1.0, total trades = 15 => penalty 2*(1-0.5)=1 => fitness 0
    assert np.isclose(ev(None, [], 0), 0.0)


def test_per_asset_min_trades_threshold():
    stats = [
        {"total_return": 1.0, "trades": 2},
        {"total_return": 1.0, "trades": 5},
        {"total_return": 1.0, "trades": 5},
    ]
    penalize_settings = {
        "metric": "return",
        "per_asset_min_trades": 3,
        "zero_trade_policy": "penalize",
        "zero_trade_penalty": -1.0,
        "trade_floor_policy": "hard_floor",
        "min_total_trades": 0,
        "lambda_dispersion": 0.0,
    }
    ev_pen = _make_evaluator(penalize_settings, stats)
    assert np.isclose(ev_pen(None, [], 0), 1 / 3, atol=1e-6)

    ignore_settings = {
        "metric": "return",
        "per_asset_min_trades": 3,
        "zero_trade_policy": "ignore",
        "coverage_penalty": 0.0,
        "trade_floor_policy": "hard_floor",
        "min_total_trades": 0,
        "lambda_dispersion": 0.0,
    }
    ev_ign = _make_evaluator(ignore_settings, stats)
    assert np.isclose(ev_ign(None, [], 0), 1.0)


def test_diagnostics_and_factory(monkeypatch):
    stats = [
        {"total_return": 1.0, "trades": 5},
        {"total_return": 0.0, "trades": 5},
        {"total_return": 0.0, "trades": 0},
    ]
    settings = {
        "metric": "return",
        "zero_trade_policy": "ignore",
        "coverage_penalty": 0.3,
        "per_asset_min_trades": 1,
        "asset_weights": {"A": 0.6, "B": 0.2, "C": 0.2},
        "trade_floor_policy": "soft_penalty",
        "soft_penalty_mode": "additive",
        "soft_penalty_strength": 1.0,
        "min_total_trades": 30,
        "lambda_dispersion": 0.25,
    }
    ev = _make_evaluator(settings, stats)
    score1 = ev(None, [], 0)
    ev = _make_evaluator(settings, stats)
    score2 = ev(None, [], 0)
    assert np.isclose(score1, score2)
    details = ev.last_details
    assert {
        "per_asset",
        "mu",
        "sigma",
        "lambda_sigma",
        "total_trades",
        "assets_included",
        "assets_ignored",
        "penalties",
    } <= details.keys()
    any_asset = next(iter(details["per_asset"].values()))
    assert "trades" in any_asset

    import config as cfg

    monkeypatch.setitem(cfg.MULTI_ASSET, "enabled", False)
    monkeypatch.setitem(fitness.config.MULTI_ASSET, "enabled", False)
    df = pd.DataFrame({"Close": [1, 2, 3]})
    fe = fitness.get_fitness_evaluator(df, {}, {})
    assert isinstance(fe, fitness.FitnessEvaluator)


def test_metric_options():
    stats = [
        {"sortino": 1.0, "profit_factor": 2.0, "total_return": 3.0, "trades": 5},
        {"sortino": 2.0, "profit_factor": 1.0, "total_return": 6.0, "trades": 5},
        {"sortino": 1.5, "profit_factor": 1.5, "total_return": 4.5, "trades": 5},
    ]
    for metric, expected in {
        "sortino": [1.0, 2.0, 1.5],
        "profit_factor": [2.0, 1.0, 1.5],
        "return": [3.0, 6.0, 4.5],
    }.items():
        ev = _make_evaluator(
            {
                "metric": metric,
                "lambda_dispersion": 0.0,
                "trade_floor_policy": "hard_floor",
                "min_total_trades": 0,
                "per_asset_min_trades": 1,
            },
            stats,
        )
        assert np.isclose(ev(None, [], 0), np.mean(expected))


def test_lambda_with_unequal_weights():
    stats = [
        {"total_return": 1.5, "trades": 5},
        {"total_return": 1.0, "trades": 5},
        {"total_return": 0.5, "trades": 5},
    ]
    settings = {
        "metric": "return",
        "lambda_dispersion": 0.25,
        "trade_floor_policy": "hard_floor",
        "min_total_trades": 0,
        "asset_weights": {"A": 0.6, "B": 0.3, "C": 0.1},
        "per_asset_min_trades": 1,
    }
    ev = _make_evaluator(settings, stats)
    score = ev(None, [], 0)
    mu, sigma = weighted_mean_std([1.5, 1.0, 0.5], [0.6, 0.3, 0.1])
    expected = mu - 0.25 * sigma
    assert np.isclose(score, expected)


def test_profit_factor_capping():
    stats = [
        {
            "sortino": 1.0,
            "profit_factor": 10.0,
            "max_drawdown": 10.0,
            "trades": 5,
            "total_return": 1.0,
        },
        {
            "sortino": 1.0,
            "profit_factor": 2.0,
            "max_drawdown": 10.0,
            "trades": 5,
            "total_return": 1.0,
        },
        {
            "sortino": 1.0,
            "profit_factor": 1.0,
            "max_drawdown": 10.0,
            "trades": 5,
            "total_return": 1.0,
        },
    ]
    settings = {
        "metric": "composite",
        "winsorize_pf_cap": 5.0,
        "trade_floor_policy": "hard_floor",
        "min_total_trades": 0,
        "lambda_dispersion": 0.0,
        "per_asset_min_trades": 1,
    }
    ev = _make_evaluator(settings, stats)
    ev(None, [], 0)
    assert ev.last_details["per_asset"]["A"]["profit_factor_capped"] == 5.0


def test_hard_floor_failure_counts():
    stats = [
        {"total_return": 1.0, "trades": 1},
        {"total_return": 1.0, "trades": 1},
        {"total_return": 1.0, "trades": 1},
    ]
    settings = {
        "metric": "return",
        "trade_floor_policy": "hard_floor",
        "min_total_trades": 10,
        "lambda_dispersion": 0.0,
        "per_asset_min_trades": 1,
    }
    ev = _make_evaluator(settings, stats)
    score = ev(None, [], 0)
    assert ev.floor_failures["below_group_floor"] == 1
    assert ev.last_details["penalties"]["trade_floor"] == "below_group_floor"
    assert score == -999.0


def test_ga_and_tuner_consistency(monkeypatch):
    stats = {
        "total_return": 1.0,
        "trades": 5,
        "sortino": 1.0,
        "profit_factor": 1.0,
        "max_drawdown": 10.0,
        "equity_curve": pd.Series([1, 1.1, 1.2]),
    }

    def fake_eval(self, ohlc, rules):
        return stats

    group_data = {
        "A": pd.DataFrame({"Close": [1, 2, 3]}),
        "B": pd.DataFrame({"Close": [1, 2, 3]}),
    }

    monkeypatch.setattr(
        fitness.MultiAssetFitnessEvaluator,
        "_evaluate_single_asset",
        fake_eval,
        raising=False,
    )
    monkeypatch.setattr(
        data_loader, "get_group_data", lambda *args, **kwargs: group_data
    )
    monkeypatch.setattr(pd.DataFrame, "ta", property(lambda self: None), raising=False)
    vbt = sys.modules["vectorbt"]
    setattr(vbt, "Portfolio", object)
    monkeypatch.setitem(cfg.MULTI_ASSET, "enabled", True)
    monkeypatch.setitem(cfg.MULTI_ASSET, "metric", "return")
    monkeypatch.setitem(cfg.MULTI_ASSET, "lambda_dispersion", 0.0)
    monkeypatch.setitem(cfg.MULTI_ASSET, "min_total_trades", 0)
    monkeypatch.setitem(cfg.MULTI_ASSET, "trade_floor_policy", "hard_floor")
    monkeypatch.setitem(cfg.MULTI_ASSET, "per_asset_min_trades", 0)
    monkeypatch.setitem(cfg.MULTI_ASSET, "min_total_trades_per_year", 0)
    monkeypatch.setitem(cfg.MULTI_ASSET, "min_included_assets", 1)
    monkeypatch.setitem(cfg.MULTI_ASSET, "coverage_penalty", 0.0)

    evaluator = fitness.MultiAssetFitnessEvaluator(group_data, {}, {})
    ga = pygad.GA(
        num_generations=1,
        num_parents_mating=1,
        sol_per_pop=1,
        num_genes=1,
        gene_space=[{"low": 0, "high": 1}],
        gene_type=[float],
        mutation_num_genes=1,
        fitness_func=evaluator.__call__,
    )
    ga.run()
    solution, fit, _ = ga.best_solution()

    score_val = tuner._evaluate_on_validation(solution, {}, group_data)
    assert np.isclose(fit, score_val)


def test_handles_asset_error_gracefully():
    """Evaluator should continue even if one asset raises an error."""
    group_data = {
        "A": pd.DataFrame({"Close": [1, 2, 3]}),
        "B": pd.DataFrame(),  # empty frame triggers error in evaluation
    }

    settings = {
        "metric": "return",
        "zero_trade_policy": "ignore",
        "per_asset_min_trades": 1,
        "trade_floor_policy": "hard_floor",
        "min_total_trades": 0,
        "lambda_dispersion": 0.0,
        "coverage_penalty": 0.0,
        "min_included_assets": 1,
    }

    ev = fitness.MultiAssetFitnessEvaluator(group_data, {}, {}, settings)

    def fake_eval(self, ohlc, rules):
        if ohlc.empty:
            raise IndexError("single positional indexer is out-of-bounds")
        return {"total_return": 1.0, "trades": 5}

    ev._evaluate_single_asset = types.MethodType(fake_eval, ev)
    score = ev(None, [], 0)

    # Only asset A contributes to the score
    assert np.isclose(score, 1.0)
    assert ev.last_details["assets_included"] == 1
    assert ev.last_details["assets_ignored"] == 1


def test_parallel_evaluation_basic():
    group_data = {
        "A": pd.DataFrame({"Close": [1, 2, 3]}),
        "B": pd.DataFrame({"Close": [1, 2, 3]}),
        "C": pd.DataFrame({"Close": [1, 2, 3]}),
    }
    settings = {
        "metric": "return",
        "trade_floor_policy": "hard_floor",
        "min_total_trades": 0,
        "per_asset_min_trades": 1,
        "lambda_dispersion": 0.0,
        "coverage_penalty": 0.0,
        "min_included_assets": 1,
        "min_total_trades_per_year": 0,
        "parallel": {"enabled": True, "backend": "thread", "max_workers": 2},
    }
    ev = fitness.MultiAssetFitnessEvaluator(group_data, {}, {}, settings)
    stats_map = {
        id(group_data["A"]): {"total_return": 5.0, "trades": 2},
        id(group_data["B"]): {"total_return": 5.0, "trades": 2},
        id(group_data["C"]): {"total_return": 5.0, "trades": 2},
    }

    def fake_eval(self, ohlc, rules):
        return stats_map[id(ohlc)]

    ev._evaluate_single_asset = types.MethodType(fake_eval, ev)
    score = ev(None, [], 0)
    assert np.isclose(score, 5.0)
    assert ev.last_details["assets_included"] == 3


def test_parallel_evaluation_handles_exception():
    group_data = {
        "A": pd.DataFrame({"Close": [1, 2, 3]}),
        "B": pd.DataFrame({"Close": [1, 2, 3]}),
    }
    settings = {
        "metric": "return",
        "trade_floor_policy": "hard_floor",
        "min_total_trades": 0,
        "per_asset_min_trades": 1,
        "lambda_dispersion": 0.0,
        "coverage_penalty": 0.0,
        "min_included_assets": 1,
        "min_total_trades_per_year": 0,
        "parallel": {"enabled": True, "backend": "thread", "max_workers": 2},
    }
    ev = fitness.MultiAssetFitnessEvaluator(group_data, {}, {}, settings)

    def fake_eval(self, ohlc, rules):
        if id(ohlc) == id(group_data["B"]):
            raise ValueError("boom")
        return {"total_return": 2.0, "trades": 2}

    ev._evaluate_single_asset = types.MethodType(fake_eval, ev)
    score = ev(None, [], 0)
    assert np.isclose(score, 2.0)
    assert not ev.last_details["per_asset"]["B"]["included"]
    assert ev.last_details["per_asset"]["B"]["reason"] == "evaluation_error"


def test_sequential_evaluation_does_not_create_executor(monkeypatch):
    created = {"thread": 0, "process": 0}

    class SentinelThreadExecutor:
        def __init__(self, *args, **kwargs):
            created["thread"] += 1

        def submit(self, *args, **kwargs):
            raise AssertionError("submit should not be called in sequential mode")

        def shutdown(self, *args, **kwargs):
            created.setdefault("thread_shutdown", 0)
            created["thread_shutdown"] += 1

    class SentinelProcessExecutor:
        def __init__(self, *args, **kwargs):
            created["process"] += 1

        def submit(self, *args, **kwargs):
            raise AssertionError("submit should not be called in sequential mode")

        def shutdown(self, *args, **kwargs):
            created.setdefault("process_shutdown", 0)
            created["process_shutdown"] += 1

    monkeypatch.setattr(fitness.cf, "ThreadPoolExecutor", SentinelThreadExecutor)
    monkeypatch.setattr(fitness.cf, "ProcessPoolExecutor", SentinelProcessExecutor)

    group_data = {
        "A": pd.DataFrame({"Close": [1, 2, 3]}),
        "B": pd.DataFrame({"Close": [4, 5, 6]}),
    }
    settings = {
        "metric": "return",
        "trade_floor_policy": "hard_floor",
        "min_total_trades": 0,
        "per_asset_min_trades": 1,
        "lambda_dispersion": 0.0,
        "coverage_penalty": 0.0,
        "min_included_assets": 1,
        "parallel": {"enabled": False},
    }
    ev = fitness.MultiAssetFitnessEvaluator(group_data, {}, {}, settings)

    def fake_eval(self, ohlc, rules):
        return {"total_return": 1.5, "trades": 2}

    ev._evaluate_single_asset = types.MethodType(fake_eval, ev)

    for _ in range(2):
        score = ev(None, [], 0)
        assert np.isclose(score, 1.5)
        assert ev.last_details["assets_included"] == 2

    ev.close()
    assert created["thread"] == 0
    assert created["process"] == 0


def test_parallel_executor_reused_across_calls(monkeypatch):
    orig_thread = cf.ThreadPoolExecutor

    class CountingThreadPoolExecutor(orig_thread):
        creations = 0
        shutdowns = 0

        def __init__(self, *args, **kwargs):
            type(self).creations += 1
            super().__init__(*args, **kwargs)

        def shutdown(self, *args, **kwargs):
            type(self).shutdowns += 1
            return super().shutdown(*args, **kwargs)

    monkeypatch.setattr(fitness.cf, "ThreadPoolExecutor", CountingThreadPoolExecutor)

    group_data = {
        "A": pd.DataFrame({"Close": [1, 2, 3]}),
        "B": pd.DataFrame({"Close": [4, 5, 6]}),
    }
    settings = {
        "metric": "return",
        "trade_floor_policy": "hard_floor",
        "min_total_trades": 0,
        "per_asset_min_trades": 1,
        "lambda_dispersion": 0.0,
        "coverage_penalty": 0.0,
        "min_included_assets": 1,
        "parallel": {"enabled": True, "backend": "thread", "max_workers": 2},
    }
    ev = fitness.MultiAssetFitnessEvaluator(group_data, {}, {}, settings)

    call_count = 0

    def fake_eval(self, ohlc, rules):
        nonlocal call_count
        call_count += 1
        return {"total_return": 2.0, "trades": 3}

    ev._evaluate_single_asset = types.MethodType(fake_eval, ev)

    first = ev(None, [], 0)
    second = ev(None, [], 0)

    assert np.isclose(first, 2.0)
    assert np.isclose(second, 2.0)
    assert call_count == len(group_data) * 2
    assert CountingThreadPoolExecutor.creations == 1

    ev.close()
    assert CountingThreadPoolExecutor.shutdowns == 1


def test_parallel_executor_reuse_handles_exceptions(monkeypatch):
    orig_thread = cf.ThreadPoolExecutor

    class CountingThreadPoolExecutor(orig_thread):
        creations = 0

        def __init__(self, *args, **kwargs):
            type(self).creations += 1
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(fitness.cf, "ThreadPoolExecutor", CountingThreadPoolExecutor)

    group_data = {
        "A": pd.DataFrame({"Close": [1, 2, 3]}),
        "B": pd.DataFrame({"Close": [4, 5, 6]}),
    }
    settings = {
        "metric": "return",
        "trade_floor_policy": "hard_floor",
        "min_total_trades": 0,
        "per_asset_min_trades": 1,
        "lambda_dispersion": 0.0,
        "coverage_penalty": 0.0,
        "min_included_assets": 1,
        "zero_trade_policy": "ignore",
        "parallel": {"enabled": True, "backend": "thread", "max_workers": 2},
    }
    ev = fitness.MultiAssetFitnessEvaluator(group_data, {}, {}, settings)

    id_map = {id(df): ticker for ticker, df in group_data.items()}
    should_raise = {"value": True}

    def fake_eval(self, ohlc, rules):
        ticker = id_map[id(ohlc)]
        if should_raise["value"] and ticker == "B":
            raise RuntimeError("boom")
        return {"total_return": 3.0, "trades": 3}

    ev._evaluate_single_asset = types.MethodType(fake_eval, ev)

    should_raise["value"] = True
    first = ev(None, [], 0)
    first_details = ev.last_details
    assert np.isclose(first, 3.0)
    assert not first_details["per_asset"]["B"]["included"]
    assert first_details["per_asset"]["B"]["reason"] == "evaluation_error"

    should_raise["value"] = False
    second = ev(None, [], 0)
    second_details = ev.last_details
    assert np.isclose(second, 3.0)
    assert second_details["assets_included"] == 2
    assert second_details["per_asset"]["B"]["included"]
    assert CountingThreadPoolExecutor.creations == 1

    ev.close()


def test_evaluate_assets_collects_errors():
    group_data = {
        "A": pd.DataFrame({"Close": [1, 2, 3]}),
        "B": pd.DataFrame({"Close": [9, 9, 9]}),
        "C": pd.DataFrame(),
    }
    settings = {
        "parallel": {"enabled": True, "backend": "thread", "max_workers": 2},
        "zero_trade_policy": "ignore",
        "per_asset_min_trades": 1,
    }
    ev = fitness.MultiAssetFitnessEvaluator(group_data, {}, {}, settings)

    def fake_eval(self, ohlc, rules):
        if ohlc["Close"].iloc[0] == 9:
            err = RuntimeError("boom")
            err.indicator = "bad_indicator"
            raise err
        return {
            "sortino": 0.5,
            "profit_factor": 1.2,
            "max_drawdown": 10.0,
            "trades": 3,
            "total_return": 4.0,
            "equity_curve": pd.Series([1, 2, 3]),
            "signal_counts": {"x": 1},
        }

    ev._evaluate_single_asset = types.MethodType(fake_eval, ev)
    results, err_counts = ev._evaluate_assets({})

    assert results["A"]["trades"] == 3
    assert results["B"]["evaluation_reason"] == "evaluation_error"
    assert "boom" in results["B"].get("reason_detail", "")
    assert err_counts["bad_indicator"] == 1
    assert results["C"]["evaluation_reason"] == "insufficient_coverage"


def test_score_assets_preserves_reason_details():
    settings = {
        "zero_trade_policy": "penalize",
        "zero_trade_penalty": -2.0,
        "per_asset_min_trades": 2,
    }
    ev = _make_evaluator(settings)

    good_stats = {
        "sortino": 0.4,
        "profit_factor": 1.5,
        "max_drawdown": 20.0,
        "trades": 3,
        "total_return": 5.0,
        "equity_curve": pd.Series([1, 2, 3]),
        "signal_counts": {"x": 1},
    }
    eval_results = {
        "A": ev._build_evaluation_record(stats=dict(good_stats)),
        "B": ev._build_evaluation_record(
            stats=ev._empty_stats(),
            reason="evaluation_error",
            detail="boom",
            trace=("foo", "bar"),
        ),
        "C": ev._build_evaluation_record(stats=dict(good_stats)),
    }

    summary = ev._score_assets(eval_results)

    assert summary["total_trades"] == 6
    assert summary["assets_traded"] == 2
    assert summary["per_asset_metrics"][1] == -2.0
    details_b = summary["per_asset_details"]["B"]
    assert details_b["included"] is True
    assert details_b.get("reason_detail") == "boom"
    assert details_b.get("reason_trace") == "foo | bar"
    assert details_b.get("profit_factor_capped") == 0.0


def test_csv_columns_and_sort(monkeypatch, tmp_path):
    monkeypatch.setitem(cfg.MULTI_ASSET, "enabled", True)
    monkeypatch.setattr(cfg, "CHARTS", {"save_pngs": False, "show_distribution": False})
    monkeypatch.setattr(cfg, "TIMEFRAME", "1d")
    monkeypatch.setattr(
        cfg, "VALIDATION_PERIOD", {"start": "2024-01-01", "end": "2024-01-31"}
    )
    group = {
        "A": pd.DataFrame({"Close": [1]}),
        "B": pd.DataFrame({"Close": [1]}),
    }
    monkeypatch.setattr(
        data_loader,
        "get_group_data",
        lambda *a, **k: group,
    )

    class DummyEval:
        def __init__(self, group, rules, gene_map, settings):
            self.last_details = {
                "per_asset": {
                    "A": {
                        "score": 1.0,
                        "trades": 1,
                        "included": True,
                        "asset_weight": 1.0,
                        "sortino": 0.1,
                        "profit_factor_capped": 1.2,
                        "max_drawdown": 5.0,
                        "equity_curve": pd.Series([1, 2]),
                    },
                    "B": {
                        "score": 2.0,
                        "trades": 1,
                        "included": True,
                        "asset_weight": 1.0,
                        "sortino": 0.2,
                        "profit_factor_capped": 1.3,
                        "max_drawdown": 4.0,
                        "equity_curve": pd.Series([1, 2]),
                    },
                },
                "mu": 0.0,
                "sigma": 0.0,
                "lambda_sigma": 0.0,
                "total_trades": 2,
                "penalties": {"coverage": 0.0, "trade_floor": None, "min_assets": None},
                "assets_included": 2,
                "assets_traded": 2,
                "min_total_trades": 0,
            }

        def __call__(self, ga, sol, idx):
            return 0.5

    monkeypatch.setattr(fitness, "MultiAssetFitnessEvaluator", DummyEval)
    monkeypatch.setattr(analysis, "_plot_multi_asset_overview", lambda *a, **k: None)

    class _VBT:
        __version__ = "0.0.0"
        __file__ = __file__

    import sys

    monkeypatch.setitem(sys.modules, "vectorbt", _VBT)
    monkeypatch.setattr(analysis, "vbt", _VBT)
    monkeypatch.chdir(tmp_path)
    analysis.set_run_dir(tmp_path)

    analysis._run_multi_asset_analysis([], {}, group, [])
    fname = tmp_path / "multi_asset_stats_1d_2024-01-31.csv"
    assert fname.exists()
    assert (tmp_path / "champion_equity.png").exists()
    df = pd.read_csv(fname)
    assert list(df.columns) == [
        "ticker",
        "included",
        "asset_weight",
        "score",
        "trades",
        "sortino",
        "profit_factor_capped",
        "max_drawdown",
        "per_asset_min_trades",
        "reason",
        "reason_detail",
        "reason_trace",
    ]
    scores = df["score"].tolist()
    assert scores == sorted(scores, reverse=True)
