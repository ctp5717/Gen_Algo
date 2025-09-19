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

import concurrent.futures as cf
import copy
import itertools
from collections import Counter

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pygad  # noqa: E402
import pytest  # noqa: E402

import analysis  # noqa: E402
import config as cfg  # noqa: E402
import data_loader  # noqa: E402
import fitness  # noqa: E402
import fitness_worker  # noqa: E402
import tuner  # noqa: E402
from utils.math import weighted_mean_std  # noqa: E402

cfg.initialize_config()


@pytest.fixture(autouse=True)
def _sync_executor(monkeypatch):
    def submit(fn, *args, **kwargs):
        fut = cf.Future()
        try:
            value = fn(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - defensive
            fut.set_exception(exc)
        else:
            fut.set_result(value)
        return fut

    def metrics():
        return {
            "submitted": 0,
            "completed": 0,
            "total_runtime": 0.0,
            "pending": 0,
            "max_pending": 0,
            "in_flight_cap": 0,
            "base_in_flight_cap": 0,
            "bytes_avg": 0.0,
            "worker_count": 0,
            "worker_seeds": [],
        }

    monkeypatch.setattr(fitness.global_executor, "submit", submit)
    monkeypatch.setattr(fitness.global_executor, "metrics", metrics)


def _make_evaluator(settings=None, stats_list=None):
    """Utility to construct a MultiAssetFitnessEvaluator with patched stats."""
    base_frame = pd.DataFrame({"Close": [1, 2, 3]})
    if stats_list is not None:
        tickers = [chr(ord("A") + i) for i in range(len(stats_list))]
        group_data = {ticker: base_frame.copy() for ticker in tickers}
    else:
        group_data = {
            "A": base_frame.copy(),
            "B": base_frame.copy(),
            "C": base_frame.copy(),
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
        stats_cycle = itertools.cycle(stats_list)

        def fake_population(self, solutions, indices):
            per_asset_stats = [next(stats_cycle) for _ in self._sorted_tickers]
            evaluation_results = {
                ticker: self._build_evaluation_record(stats=dict(stat))
                for ticker, stat in zip(self._sorted_tickers, per_asset_stats, strict=False)
            }
            summary = self._score_assets(evaluation_results)
            score = self._aggregate_scores(summary)
            return [score for _ in indices], Counter()

        evaluator._evaluate_population = types.MethodType(fake_population, evaluator)
    return evaluator


def _setup_single_asset_evaluator(monkeypatch, collect=None):
    """Prepare an evaluator for single-asset equity curve tests."""

    index = pd.date_range("2024-01-01", periods=3, freq="D")
    ohlc = pd.DataFrame({"Close": [1.0, 1.1, 1.2]}, index=index)
    equity = pd.Series([100.0, 101.0, 102.0], index=index, dtype=float)
    calls = {"value": 0}

    class DummyPortfolio:
        def __init__(self):
            self.trades = types.SimpleNamespace(count=lambda: 1)

        @classmethod
        def from_signals(cls, *args, **kwargs):
            return cls()

        def value(self):
            calls["value"] += 1
            return equity

        def stats(self, *args, **kwargs):
            return {}

    monkeypatch.setattr(fitness.vbt, "Portfolio", DummyPortfolio)
    monkeypatch.setattr(
        fitness.metrics_contract, "assert_metric_aliases", lambda *a, **k: None
    )
    monkeypatch.setattr(
        fitness.metrics_contract, "_provider_signature", lambda *a, **k: "dummy"
    )

    def fake_metrics(portfolio):
        return (
            {
                "sortino": 0.5,
                "profit_factor": 1.2,
                "max_drawdown": 10.0,
                "total_return": 0.2,
            },
            {},
            [],
        )

    monkeypatch.setattr(fitness.metrics_contract, "evaluate_metrics", fake_metrics)

    def fake_process(ohlc_df, rules, collect_counts=False):
        entries = pd.Series(
            [True] + [False] * (len(ohlc_df.index) - 1),
            index=ohlc_df.index,
            dtype=bool,
        )
        counts = {"entries": int(entries.sum())}
        return (entries, counts) if collect_counts else entries

    monkeypatch.setattr(fitness.engine, "process_strategy_rules", fake_process)

    group_data = {"A": ohlc}
    if collect is None:
        evaluator = fitness.MultiAssetFitnessEvaluator(group_data, {}, {})
    else:
        evaluator = fitness.MultiAssetFitnessEvaluator(
            group_data, {}, {}, {"collect_equity_curve": collect}
        )
    return evaluator, ohlc, equity, calls


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


def test_equity_curve_not_collected_by_default(monkeypatch):
    evaluator, ohlc, _, calls = _setup_single_asset_evaluator(monkeypatch)
    stats = evaluator._evaluate_single_asset(ohlc, {})
    assert calls["value"] == 0
    assert isinstance(stats["equity_curve"], pd.Series)
    assert stats["equity_curve"].empty


def test_equity_curve_collection_opt_in(monkeypatch):
    evaluator, ohlc, equity, calls = _setup_single_asset_evaluator(
        monkeypatch, collect=True
    )
    stats = evaluator._evaluate_single_asset(ohlc, {})
    assert calls["value"] == 1
    pd.testing.assert_series_equal(stats["equity_curve"], equity)


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
            assert settings.get("collect_equity_curve") is True
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
def test_global_executor_batch_dispatch(monkeypatch):
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
    }

    evaluator = fitness.MultiAssetFitnessEvaluator(group_data, {}, {}, settings)

    calls = []

    def fake_submit(fn, *args, **kwargs):
        fut = cf.Future()
        try:
            value = fn(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - defensive
            fut.set_exception(exc)
        else:
            calls.append(value)
            fut.set_result(value)
        return fut

    metrics_calls = {"count": 0}

    def fake_metrics():
        metrics_calls["count"] += 1
        if metrics_calls["count"] == 1:
            return {
                "submitted": 0,
                "completed": 0,
                "total_runtime": 0.0,
                "pending": 0,
                "max_pending": 0,
                "in_flight_cap": 0,
                "base_in_flight_cap": 0,
                "bytes_avg": 0.0,
                "worker_count": 0,
                "worker_seeds": [],
            }
        return {
            "submitted": 0,
            "completed": 0,
            "total_runtime": 1.0,
            "pending": 3,
            "max_pending": 3,
            "in_flight_cap": 0,
            "base_in_flight_cap": 0,
            "bytes_avg": 0.0,
            "worker_count": 0,
            "worker_seeds": [],
        }

    def fake_worker(descriptor, base_rules, gene_map, candidates, worker_settings):
        asset_id = descriptor.get("asset_id")
        return {
            "asset_id": asset_id,
            "results": [
                {
                    "sol_idx": candidate["index"],
                    "stats": {"total_return": 5.0, "trades": 3},
                }
                for candidate in candidates
            ],
            "latency": 0.05,
            "rows": 3,
            "bytes": 128,
        }

    monkeypatch.setattr(fitness_worker, "evaluate_batch", fake_worker)
    monkeypatch.setattr(fitness.global_executor, "submit", fake_submit)
    monkeypatch.setattr(fitness.global_executor, "metrics", fake_metrics)

    score = evaluator(None, np.array([0.1, 0.2]), 0)
    assert np.isclose(score, 5.0)
    assert evaluator.last_details["assets_included"] == 2
    assert len(calls) == 2
    instr = evaluator.instrumentation
    assert instr["evaluations"] == 2
    assert instr["queue_depth"] == 3
    evaluator.close()


def test_worker_error_propagation(monkeypatch):
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
    }

    evaluator = fitness.MultiAssetFitnessEvaluator(group_data, {}, {}, settings)

    metrics_state = {
        "submitted": 0,
        "completed": 0,
        "total_runtime": 0.0,
        "pending": 0,
        "max_pending": 0,
        "in_flight_cap": 0,
        "base_in_flight_cap": 0,
        "bytes_avg": 0.0,
        "worker_count": 0,
        "worker_seeds": [],
    }

    def fake_submit(fn, *args, **kwargs):
        fut = cf.Future()
        metrics_state["submitted"] += 1
        metrics_state["pending"] += 1
        metrics_state["max_pending"] = max(
            metrics_state["max_pending"], metrics_state["pending"]
        )
        try:
            value = fn(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - defensive
            metrics_state["pending"] = max(0, metrics_state["pending"] - 1)
            fut.set_exception(exc)
        else:
            metrics_state["pending"] = max(0, metrics_state["pending"] - 1)
            metrics_state["completed"] += 1
            metrics_state["total_runtime"] += float(value.get("latency", 0.0))
            fut.set_result(value)
        return fut

    def fake_metrics():
        return dict(metrics_state)

    def fake_worker(descriptor, base_rules, gene_map, candidates, worker_settings):
        asset_id = descriptor.get("asset_id")
        if asset_id == "B":
            return {
                "asset_id": asset_id,
                "results": [
                    {
                        "sol_idx": candidate["index"],
                        "error": {
                            "type": "ValueError",
                            "message": "boom",
                            "trace": "traceback",
                            "indicator": "bad_indicator",
                        },
                    }
                    for candidate in candidates
                ],
                "latency": 0.02,
                "rows": 3,
                "bytes": 64,
            }
        return {
            "asset_id": asset_id,
            "results": [
                {
                    "sol_idx": candidate["index"],
                    "stats": {"total_return": 4.0, "trades": 3},
                }
                for candidate in candidates
            ],
            "latency": 0.02,
            "rows": 3,
            "bytes": 64,
        }

    monkeypatch.setattr(fitness_worker, "evaluate_batch", fake_worker)
    monkeypatch.setattr(fitness.global_executor, "submit", fake_submit)
    monkeypatch.setattr(fitness.global_executor, "metrics", fake_metrics)

    score = evaluator(None, np.array([0.1, 0.2]), 0)
    assert np.isclose(score, 4.0)
    details = evaluator.last_details
    asset_b = details["per_asset"]["B"]
    assert asset_b.get("included") is False
    assert asset_b.get("evaluation_reason") == "evaluation_error"
    assert "bad_indicator" in (asset_b.get("reason_detail") or "")
    instr = evaluator.instrumentation
    assert instr["submitted"] == 2
    assert instr["completed"] == 2
    assert instr["pending"] == 0
    assert metrics_state["submitted"] == 2
    assert metrics_state["completed"] == 2
    assert details["per_asset"]["B"]["reason"] == "evaluation_error"
    evaluator.close()


def test_parallel_sequential_parity(monkeypatch):
    group_data = {
        "A": pd.DataFrame({"Close": [1, 2, 3]}),
        "B": pd.DataFrame({"Close": [2, 3, 4]}),
    }
    settings = {
        "metric": "composite",
        "trade_floor_policy": "hard_floor",
        "min_total_trades": 0,
        "per_asset_min_trades": 1,
        "lambda_dispersion": 0.0,
        "coverage_penalty": 0.0,
        "min_included_assets": 1,
    }

    stats_payload = {
        "sortino": 1.25,
        "profit_factor": 1.4,
        "max_drawdown": 12.0,
        "trades": 4,
        "total_return": 0.15,
        "equity_curve": pd.Series([1.0, 1.1, 1.2]),
        "signal_counts": {"entries": 2},
        "metric_sources": {},
        "missing_metrics": [],
    }

    def fake_submit(fn, *args, **kwargs):
        fut = cf.Future()
        try:
            value = fn(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - defensive
            fut.set_exception(exc)
        else:
            fut.set_result(value)
        return fut

    def fake_metrics():
        return {
            "submitted": 0,
            "completed": 0,
            "total_runtime": 0.0,
            "pending": 0,
            "max_pending": 0,
            "in_flight_cap": 2,
            "base_in_flight_cap": 2,
            "bytes_avg": 0.0,
            "worker_count": 2,
            "worker_seeds": [cfg.SEED, cfg.SEED + 1],
        }

    monkeypatch.setattr(fitness.global_executor, "submit", fake_submit)
    monkeypatch.setattr(fitness.global_executor, "metrics", fake_metrics)
    monkeypatch.setattr(
        fitness.global_executor, "record_batch_metrics", lambda *a, **k: 2
    )

    def fake_worker(descriptor, base_rules, gene_map, candidates, worker_settings):
        return {
            "asset_id": descriptor.get("asset_id"),
            "results": [
                {"sol_idx": candidate["index"], "stats": dict(stats_payload)}
                for candidate in candidates
            ],
            "latency": 0.02,
            "rows": 3,
            "bytes": 64,
        }

    def fake_single_asset(self, ohlc, rules):
        return dict(stats_payload)

    monkeypatch.setattr(fitness_worker, "evaluate_batch", fake_worker)
    monkeypatch.setattr(
        fitness.MultiAssetFitnessEvaluator,
        "_evaluate_single_asset",
        fake_single_asset,
    )

    evaluator_parallel = fitness.MultiAssetFitnessEvaluator(
        group_data, {}, {}, settings
    )
    vector = [0.5, 0.25]
    parallel_score = evaluator_parallel(None, vector, 0)
    parallel_details = copy.deepcopy(evaluator_parallel.last_details)

    evaluator_seq = fitness.MultiAssetFitnessEvaluator(group_data, {}, {}, settings)
    results_seq, _ = evaluator_seq._evaluate_assets({"vector": vector})
    assets_map = {
        ticker: results_seq.get(ticker, evaluator_seq._build_evaluation_record())
        for ticker in evaluator_seq._sorted_tickers
    }
    summary = evaluator_seq._score_assets(assets_map)
    sequential_score = evaluator_seq._aggregate_scores(summary)
    sequential_details = copy.deepcopy(evaluator_seq.last_details)

    assert np.isclose(parallel_score, sequential_score)
    for asset in evaluator_parallel._sorted_tickers:
        par = parallel_details["per_asset"].get(asset, {})
        seq = sequential_details["per_asset"].get(asset, {})
        assert par.get("score") == seq.get("score")
        assert par.get("evaluation_reason") == seq.get("evaluation_reason")

    evaluator_parallel.close()
    evaluator_seq.close()
