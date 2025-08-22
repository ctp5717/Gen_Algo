import sys
import types
from pathlib import Path

# Ensure repository root is on the import path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Stub heavy optional dependencies
sys.modules.setdefault('pandas_ta', types.ModuleType('pandas_ta'))
sys.modules.setdefault('vectorbt', types.ModuleType('vectorbt'))

import pandas as pd  # noqa: E402
import numpy as np  # noqa: E402
import pygad  # noqa: E402
import fitness  # noqa: E402
import tuner  # noqa: E402
import data_loader  # noqa: E402
import config as cfg  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_cache():
    fitness._EVAL_CACHE.clear()


def _make_evaluator(settings=None, stats_list=None, group_data=None):
    """Utility to construct a MultiAssetFitnessEvaluator with patched stats."""
    group_data = group_data or {
        'A': pd.DataFrame({'Close': [1, 2, 3]}),
        'B': pd.DataFrame({'Close': [1, 2, 3]}),
        'C': pd.DataFrame({'Close': [1, 2, 3]}),
    }
    evaluator = fitness.MultiAssetFitnessEvaluator(group_data, {}, {}, settings or {})

    if stats_list is not None:
        stats_iter = iter(stats_list)

        def fake_eval(self, ohlc, rules):
            return next(stats_iter)

        evaluator._evaluate_single_asset = types.MethodType(fake_eval, evaluator)
    return evaluator


def test_evaluate_single_asset_handles_zero_trades(monkeypatch):
    df = pd.DataFrame({'Close': [1, 1, 1]}, index=pd.RangeIndex(3))
    evaluator = fitness.MultiAssetFitnessEvaluator({'A': df}, {}, {}, {})

    monkeypatch.setattr(
        fitness.engine,
        'process_strategy_rules',
        lambda *a, **k: pd.Series([False, False, False], index=df.index),
    )

    class DummyPortfolio:
        class trades:
            @staticmethod
            def count():
                return 0

        def stats(self):
            raise RuntimeError('stats should not be called')

        def value(self):
            return pd.Series([1, 1, 1], index=df.index)

    monkeypatch.setattr(
        fitness.vbt,
        'Portfolio',
        types.SimpleNamespace(from_signals=lambda *a, **k: DummyPortfolio()),
        raising=False,
    )

    stats = evaluator._evaluate_single_asset(df, {})
    assert stats['trades'] == 0


def test_no_assets_traded_sets_default_details(monkeypatch):
    df = pd.DataFrame({'Close': [1, 1, 1]}, index=pd.RangeIndex(3))

    monkeypatch.setattr(
        fitness.engine,
        'process_strategy_rules',
        lambda *a, **k: pd.Series([False, False, False], index=df.index),
    )

    class DummyPortfolio:
        class trades:
            @staticmethod
            def count():
                return 0

        def stats(self):
            raise RuntimeError('stats should not be called')

        def value(self):
            return pd.Series([1, 1, 1], index=df.index)

    monkeypatch.setattr(
        fitness.vbt,
        'Portfolio',
        types.SimpleNamespace(from_signals=lambda *a, **k: DummyPortfolio()),
        raising=False,
    )

    monkeypatch.setitem(cfg.MULTI_ASSET, 'min_total_trades', 0)
    monkeypatch.setitem(cfg.MULTI_ASSET, 'poor_score', 0.0)
    monkeypatch.setitem(cfg.MULTI_ASSET, 'coverage_penalty_kappa', 0.0)

    evaluator = fitness.MultiAssetFitnessEvaluator({'A': df}, {}, {}, {})
    score = evaluator(None, [], 0)
    assert score == 0.0
    assert evaluator.last_details['total_trades'] == 0
    assert evaluator.last_details['penalties']['trade_floor'] is None
    assert evaluator.last_details['penalties']['floor_ratio'] == 0.0


def test_aggregation_math():
    stats = [
        {'total_return': 1.6, 'trades': 5},
        {'total_return': 1.0, 'trades': 5},
        {'total_return': 0.4, 'trades': 5},
    ]
    settings = {
        'metric': 'return',
        'lambda_dispersion': 0.25,
        'min_total_trades': 0,
        'soft_penalty_strength': 0,
    }
    ev = _make_evaluator(settings, stats)
    score = ev(None, [], 0)
    assert np.isclose(ev.last_details['mu'], 1.0)
    assert np.isclose(ev.last_details['sigma'], 0.4899, atol=1e-4)
    assert np.isclose(score, 0.8775, atol=1e-4)


def test_weighted_mean_std_deterministic():
    mu, sigma = fitness.weighted_mean_std([1.6, 1.0, 0.4], [1, 1, 1])
    assert np.isclose(mu, 1.0)
    assert np.isclose(sigma, 0.4898979, atol=1e-6)
    lam = 0.25
    F = mu - lam * sigma
    assert np.isclose(F, 0.8775255, atol=1e-6)


def test_all_equal_scores_yield_mean():
    stats = [
        {"total_return": 2.0, "trades": 5},
        {"total_return": 2.0, "trades": 5},
        {"total_return": 2.0, "trades": 5},
    ]
    settings = {
        "metric": "return",
        "lambda_dispersion": 0.5,
        "min_total_trades": 0,
        "soft_penalty_strength": 0,
    }
    ev = _make_evaluator(settings, stats)
    score = ev(None, [], 0)
    assert np.isclose(score, 2.0)
    assert np.isclose(ev.last_details["mu"], 2.0)
    assert np.isclose(ev.last_details["sigma"], 0.0)


def test_trade_floor_policies():
    stats = [
        {'total_return': 1.0, 'trades': 5},
        {'total_return': 1.0, 'trades': 5},
        {'total_return': 1.0, 'trades': 5},
    ]
    settings_hard = {
        'metric': 'return',
        'min_total_trades': 20,
        'mode': 'walk_forward',
    }
    ev_hard = _make_evaluator(settings_hard, stats)
    assert ev_hard(None, [], 0) == -999.0

    settings_soft = {
        'metric': 'return',
        'min_total_trades': 30,
        'soft_penalty_strength': 1.0,
        'mode': 'ga',
    }
    ev_soft = _make_evaluator(settings_soft, stats)
    assert np.isclose(ev_soft(None, [], 0), 0.5)


def test_caches_single_asset_results(monkeypatch):
    calls = {"n": 0}

    def fake_eval(self, ohlc, rules):
        calls["n"] += 1
        return {
            "sortino": 1.0,
            "profit_factor": 1.0,
            "max_drawdown": 10.0,
            "trades": 1,
            "total_return": 1.0,
            "equity_curve": pd.Series([1, 1.1, 1.2]),
        }

    monkeypatch.setattr(
        fitness.MultiAssetFitnessEvaluator,
        "_evaluate_single_asset",
        fake_eval,
        raising=False,
    )
    fitness._EVAL_CACHE.clear()
    group_data = {"A": pd.DataFrame({"Close": [1, 2, 3]})}
    ev1 = fitness.MultiAssetFitnessEvaluator(group_data, {}, {}, {})
    ev1(None, [], 0)
    assert calls["n"] == 1
    # Re-evaluating with the same rules should hit the cache
    ev1(None, [], 0)
    assert calls["n"] == 1
    # A new evaluator instance should also reuse the cache
    ev2 = fitness.MultiAssetFitnessEvaluator(group_data, {}, {}, {})
    ev2(None, [], 0)
    assert calls["n"] == 1
    # Changing the rules invalidates the cache
    ev2.base_rules = {"foo": 1}
    ev2(None, [], 0)
    assert calls["n"] == 2


def test_hard_floor_returns_poor_score_with_zero_trade_penalize():
    stats = [
        {'total_return': 0.0, 'trades': 0},
        {'total_return': 1.0, 'trades': 5},
        {'total_return': 1.0, 'trades': 5},
    ]
    settings = {
        'metric': 'return',
        'per_asset_min_trades': 1,
        'min_total_trades': 30,
        'lambda_dispersion': 0.0,
        'mode': 'walk_forward',
        'zero_trade_policy': 'penalize',
    }
    ev = _make_evaluator(settings, stats)
    assert ev(None, [], 0) == -999.0


def test_hard_floor_returns_poor_score_with_zero_trade_ignore():
    stats = [
        {'total_return': 0.0, 'trades': 0},
        {'total_return': 1.0, 'trades': 5},
        {'total_return': 1.0, 'trades': 5},
    ]
    settings = {
        'metric': 'return',
        'coverage_penalty_kappa': 0.0,
        'per_asset_min_trades': 1,
        'min_total_trades': 30,
        'lambda_dispersion': 0.0,
        'mode': 'walk_forward',
        'zero_trade_policy': 'ignore',
    }
    ev = _make_evaluator(settings, stats)
    assert ev(None, [], 0) == -999.0


def test_zero_trade_assets_shrinkage():
    stats = [
        {'total_return': 0.0, 'trades': 0},
        {'total_return': 1.0, 'trades': 5},
        {'total_return': 1.0, 'trades': 5},
    ]
    settings = {
        'metric': 'return',
        'per_asset_min_trades': 1,
        'min_total_trades': 0,
        'lambda_dispersion': 0.0,
        'soft_penalty_strength': 0,
        'zero_trade_policy': 'penalize',
    }
    ev = _make_evaluator(settings, stats)
    score = ev(None, [], 0)
    assert np.isclose(score, 2/3, atol=1e-6)
    asset = ev.last_details['per_asset']['A']
    assert asset['included'] is True
    assert asset.get('shrinkage_multiplier') is None


def test_zero_trade_assets_no_coverage_penalty():
    stats = [
        {'total_return': 0.0, 'trades': 0},
        {'total_return': 1.0, 'trades': 5},
        {'total_return': 1.0, 'trades': 5},
    ]
    settings = {
        'metric': 'return',
        'coverage_penalty_kappa': 0.3,
        'per_asset_min_trades': 1,
        'min_total_trades': 0,
        'lambda_dispersion': 0.0,
        'soft_penalty_strength': 0,
        'zero_trade_policy': 'penalize',
    }
    ev = _make_evaluator(settings, stats)
    score = ev(None, [], 0)
    assert np.isclose(score, 2/3, atol=1e-6)
    asset = ev.last_details['per_asset']['A']
    assert asset['included'] is True
    assert np.isclose(ev.last_details['penalties']['coverage'], 0.0)


def test_coverage_penalty_formula():
    stats = [
        {'total_return': 0.5, 'trades': 5},
        {'total_return': 1.5, 'trades': 5},
        {'total_return': 0.0, 'trades': 0},
    ]
    kappa = 0.4
    settings = {
        'metric': 'return',
        'coverage_penalty_kappa': kappa,
        'per_asset_min_trades': 1,
        'min_total_trades': 0,
        'lambda_dispersion': 0.0,
        'soft_penalty_strength': 0,
    }
    ev = _make_evaluator(settings, stats)
    score = ev(None, [], 0)
    mu = np.mean([0.5, 1.5])
    coverage_penalty = kappa * (1 - (2 / 3))
    assert np.isclose(ev.last_details['penalties']['coverage'], coverage_penalty)
    assert np.isclose(score, mu - coverage_penalty)


@pytest.mark.parametrize("kappa", [None, 0.0, 0.3])
def test_coverage_penalty_kappa_monotonic(kappa):
    stats = [
        {'total_return': 0.0, 'trades': 0},
        {'total_return': 1.0, 'trades': 5},
        {'total_return': 1.0, 'trades': 5},
    ]
    settings = {
        'metric': 'return',
        'coverage_penalty_kappa': kappa,
        'per_asset_min_trades': 1,
        'min_total_trades': 0,
        'lambda_dispersion': 0.0,
        'soft_penalty_strength': 0,
        'zero_trade_policy': 'penalize',
    }
    ev = _make_evaluator(settings, stats)
    assert np.isclose(ev(None, [], 0), 2/3, atol=1e-6)
    assert np.isclose(ev.last_details['penalties']['coverage'], 0.0)


def test_zero_trade_asset_ignored_triggers_coverage_penalty():
    stats = [
        {'total_return': 1.0, 'trades': 5},
        {'total_return': 0.0, 'trades': 0},
    ]
    settings = {
        'metric': 'return',
        'coverage_penalty_kappa': 0.5,
        'per_asset_min_trades': 1,
        'min_total_trades': 0,
        'lambda_dispersion': 0.0,
        'soft_penalty_strength': 0,
    }
    group_data = {
        'A': pd.DataFrame({'Close': [1, 2, 3]}),
        'B': pd.DataFrame({'Close': [1, 2, 3]}),
    }
    ev = _make_evaluator(settings, stats, group_data)
    score = ev(None, [], 0)
    assert np.isclose(score, 0.75)
    assert ev.last_details['assets_included'] == 1
    assert ev.last_details['assets_ignored'] == 1
    assert np.isclose(ev.last_details['penalties']['coverage'], 0.25)


def test_empty_group_returns_poor_score():
    ev = _make_evaluator({'metric': 'return', 'poor_score': -123.0}, group_data={})
    score = ev(None, [], 0)
    assert score == -123.0
    assert ev.last_details['assets_included'] == 0
    assert ev.last_details['penalties']['coverage'] is None


def test_all_zero_trade_assets_apply_coverage_penalty():
    stats = [
        {'total_return': 0.0, 'trades': 0},
        {'total_return': 0.0, 'trades': 0},
        {'total_return': 0.0, 'trades': 0},
    ]
    settings = {
        'metric': 'return',
        'lambda_dispersion': 0.0,
        'coverage_penalty_kappa': 0.5,
        'min_total_trades': 0,
        'soft_penalty_strength': 0,
        'poor_score': 0.0,
    }
    ev = _make_evaluator(settings, stats)
    score = ev(None, [], 0)
    assert np.isclose(score, -0.5)
    assert np.isclose(ev.last_details['penalties']['coverage'], 0.5)
    assert ev.last_details['assets_included'] == 0
    assert ev.last_details['assets_ignored'] == 3


def test_coverage_penalty_lambda_alias():
    stats = [
        {'total_return': 0.0, 'trades': 0},
        {'total_return': 0.0, 'trades': 0},
        {'total_return': 0.0, 'trades': 0},
    ]
    settings = {
        'metric': 'return',
        'lambda_dispersion': 0.0,
        'coverage_penalty_lambda': 0.5,
        'min_total_trades': 0,
        'soft_penalty_strength': 0,
        'poor_score': 0.0,
    }
    ev = _make_evaluator(settings, stats)
    score = ev(None, [], 0)
    assert np.isclose(score, -0.5)
    assert np.isclose(ev.last_details['penalties']['coverage'], 0.5)
    assert ev.last_details['assets_included'] == 0
    assert ev.last_details['assets_ignored'] == 3


def test_weight_renormalization():
    stats = [
        {'total_return': 1.0, 'trades': 5},
        {'total_return': 0.0, 'trades': 5},
        {'total_return': 0.0, 'trades': 0},
    ]
    settings = {
        'metric': 'return',
        'per_asset_min_trades': 1,
        'asset_weights': {'A': 0.6, 'B': 0.2, 'C': 0.2},
        'min_total_trades': 0,
        'lambda_dispersion': 0.0,
        'soft_penalty_strength': 0,
        'zero_trade_policy': 'penalize',
    }
    ev = _make_evaluator(settings, stats)
    assert np.isclose(ev(None, [], 0), 0.6, atol=1e-6)


def test_weight_renormalization_multiple_exclusions():
    stats = [
        {'total_return': 0.5, 'trades': 5},
        {'total_return': 1.0, 'trades': 0},
        {'total_return': 1.5, 'trades': 0},
    ]
    settings = {
        'metric': 'return',
        'per_asset_min_trades': 1,
        'asset_weights': {'A': 0.2, 'B': 0.3, 'C': 0.5},
        'min_total_trades': 0,
        'lambda_dispersion': 0.0,
        'soft_penalty_strength': 0,
        'zero_trade_policy': 'penalize',
    }
    ev = _make_evaluator(settings, stats)
    score = ev(None, [], 0)
    assert np.isclose(score, 1.15)
    assert ev.last_details['assets_included'] == 3


def test_dynamic_trade_floor_tracks_recent_generations():
    stats = [
        {"total_return": 1.0, "trades": 10},  # gen0 sol1
        {"total_return": 2.0, "trades": 20},  # gen0 sol2 (best)
        {"total_return": 1.0, "trades": 30},  # gen1 sol1
        {"total_return": 1.0, "trades": 5},   # gen2 sol1
    ]
    settings = {
        "metric": "return",
        "min_total_trades": 5,
        "max_total_trades": 25,
        "soft_penalty_strength": 0,
        "trade_floor_window": 5,
    }
    ev = _make_evaluator(settings, stats, {"A": pd.DataFrame({"Close": [1, 2, 3]})})
    ev.base_rules = {"p": 0}
    ev.gene_map = {0: {"path": ["p"]}}

    class GA:
        def __init__(self):
            self.generations_completed = 0

    ga = GA()
    ev(ga, [0], 0)  # gen0 sol1
    ev(ga, [1], 0)  # gen0 sol2
    ga.generations_completed = 1
    ev(ga, [2], 0)  # triggers floor update from gen0 -> 20
    assert ev.settings["min_total_trades"] == 20
    ga.generations_completed = 2
    ev(ga, [3], 0)  # triggers floor update from gen1 -> median([20,30])=25 but clamp to max 25
    assert ev.settings["min_total_trades"] == 25
    assert list(ev._recent_totals) == [20, 30]


def test_floor_strength_scaling():
    stats = [
        {'total_return': 1.0, 'trades': 5},
        {'total_return': 1.0, 'trades': 5},
        {'total_return': 1.0, 'trades': 5},
    ]
    settings = {
        'metric': 'return',
        'soft_penalty_strength': 2.0,
        'min_total_trades': 30,
        'mode': 'ga',
    }
    ev = _make_evaluator(settings, stats)
    # Mean = 1.0, total trades = 15 -> floor_ratio=0.5 -> fitness=0.25
    assert np.isclose(ev(None, [], 0), 0.25)


def test_floor_strength_with_zero_trades():
    stats = [
        {'total_return': 1.0, 'trades': 5},
        {'total_return': 1.0, 'trades': 5},
        {'total_return': 1.0, 'trades': 0},
    ]
    settings = {
        'metric': 'return',
        'per_asset_min_trades': 1,
        'soft_penalty_strength': 2.0,
        'min_total_trades': 20,
        'lambda_dispersion': 0.0,
        'mode': 'ga',
        'zero_trade_policy': 'penalize',
    }
    ev = _make_evaluator(settings, stats)
    # Total trades = 10, floor = 20 -> scale=(0.5)**2=0.25, mean=1.0 => fitness=0.25
    assert np.isclose(ev(None, [], 0), 0.25)


def test_min_total_trades_per_year_scaling():
    stats = [
        {'total_return': 1.0, 'trades': 5},
        {'total_return': 1.0, 'trades': 5},
    ]
    settings = {
        'metric': 'return',
        'min_total_trades_per_year': 12,
        'soft_penalty_strength': 0,
    }
    idx = pd.to_datetime(['2020-01-01', '2020-07-01'])
    group_data = {
        'A': pd.DataFrame({'Close': [1, 2]}, index=idx),
        'B': pd.DataFrame({'Close': [1, 2]}, index=idx),
    }
    ev = _make_evaluator(settings, stats, group_data)
    score = ev(None, [], 0)
    assert np.isclose(score, 1.0)
    assert ev.settings['min_total_trades'] == 6


def test_min_total_trades_per_year_three_month_fold():
    idx = pd.to_datetime(['2020-01-01', '2020-04-01'])
    group_data = {
        'A': pd.DataFrame({'Close': [1, 2]}, index=idx),
        'B': pd.DataFrame({'Close': [1, 2]}, index=idx),
    }
    settings = {'min_total_trades_per_year': 24}
    ev = fitness.MultiAssetFitnessEvaluator(group_data, {}, {}, settings)
    assert ev.settings['min_total_trades'] == 6


def test_per_asset_min_trades_threshold():
    stats = [
        {'total_return': 1.0, 'trades': 2},
        {'total_return': 1.0, 'trades': 5},
        {'total_return': 1.0, 'trades': 5},
    ]
    settings = {
        'metric': 'return',
        'per_asset_min_trades': 3,
        'min_total_trades': 0,
        'lambda_dispersion': 0.0,
        'soft_penalty_strength': 0,
        'mode': None,
        'low_trade_shrink': {'enabled': True, 'k': 3, 's': 1.0},
    }
    ev = _make_evaluator(settings, stats)
    score = ev(None, [], 0)
    expected = ((2/3) + 1.0 + 1.0) / 3
    assert np.isclose(score, expected)
    asset = ev.last_details['per_asset']['A']
    assert np.isclose(asset['shrinkage']['multiplier'], 2/3)


def test_low_trade_scaling_and_total_trades_contribution_legacy_keys():
    stats = [
        {'total_return': 1.0, 'trades': 2},
        {'total_return': 1.0, 'trades': 4},
        {'total_return': 1.0, 'trades': 4},
    ]
    settings = {
        'metric': 'return',
        'min_total_trades': 0,
        'lambda_dispersion': 0.0,
        'soft_penalty_strength': 0,
        'partial_trades_threshold': 4,
        'partial_trades_exponent': 1.0,
    }
    ev = _make_evaluator(settings, stats)
    score = ev(None, [], 0)
    assert np.isclose(score, (0.5 + 1.0 + 1.0) / 3)
    assert ev.last_details['total_trades'] == 10
    assert np.isclose(ev.last_details['per_asset']['A']['score'], 0.5)


def test_per_asset_diagnostics_include_pf_drawdown_and_penalties():
    stats = [
        {
            "sortino": 1.0,
            "profit_factor": 10.0,
            "max_drawdown": 10.0,
            "total_return": 0.0,
            "trades": 5,
        },
        {
            "sortino": 1.0,
            "profit_factor": 2.0,
            "max_drawdown": 20.0,
            "total_return": 0.0,
            "trades": 0,
        },
    ]
    settings = {
        "metric": "composite",
        "per_asset_min_trades": 1,
        "min_total_trades": 0,
        "lambda_dispersion": 0.0,
        "soft_penalty_strength": 0,
    }
    df = pd.DataFrame({"Close": [1, 2, 3]})
    group_data = {"A": df, "B": df}
    ev = _make_evaluator(settings, stats, group_data)
    ev(None, [], 0)
    details = ev.last_details["per_asset"]
    a = details["A"]
    assert np.isclose(a["profit_factor_capped"], 5.0)
    assert np.isclose(a["drawdown_score"], 0.9)
    assert a.get("penalties") in (None, {})
    b = details["B"]
    assert b["included"] is False
    assert b.get("shrinkage_multiplier") is None


def test_caps_logged():
    stats = [
        {"sortino": 12.0, "profit_factor": 8.0, "max_drawdown": 5.0, "trades": 5},
    ]
    settings = {
        "metric": "composite",
        "per_asset_min_trades": 1,
        "min_total_trades": 0,
        "lambda_dispersion": 0.0,
        "soft_penalty_strength": 0,
    }
    df = pd.DataFrame({"Close": [1, 2, 3]})
    ev = _make_evaluator(settings, stats, {"A": df})
    ev(None, [], 0)
    caps = ev.last_details["per_asset"]["A"]["caps"]
    import config as cfg
    assert caps["profit_factor"]["cap"] == cfg.PF_CAP
    assert caps["profit_factor"]["capped"] == cfg.PF_CAP
    assert caps["sortino"]["cap"] == cfg.SORTINO_CAP
    assert caps["sortino"]["capped"] == cfg.SORTINO_CAP


def test_diagnostics_and_factory(monkeypatch):
    stats = [
        {'total_return': 1.0, 'trades': 5},
        {'total_return': 0.0, 'trades': 5},
        {'total_return': 0.0, 'trades': 0},
    ]
    settings = {
        'metric': 'return',
        'coverage_penalty_kappa': 0.3,
        'per_asset_min_trades': 1,
        'asset_weights': {'A': 0.6, 'B': 0.2, 'C': 0.2},
        'min_total_trades': 30,
        'lambda_dispersion': 0.25,
        'soft_penalty_strength': 1.0,
    }
    ev = _make_evaluator(settings, stats)
    score1 = ev(None, [], 0)
    ev = _make_evaluator(settings, stats)
    score2 = ev(None, [], 0)
    assert np.isclose(score1, score2)
    details = ev.last_details
    expected_keys = {
        'per_asset',
        'mu',
        'sigma',
        'lambda_sigma',
        'total_trades',
        'assets_included',
        'assets_ignored',
        'penalties',
    }
    assert expected_keys <= details.keys()
    any_asset = next(iter(details['per_asset'].values()))
    assert {'trades', 'profit_factor_capped', 'drawdown_score', 'penalties'} <= any_asset.keys()

    import config as cfg
    orig = cfg.MULTI_ASSET['enabled']
    cfg.MULTI_ASSET['enabled'] = False
    df = pd.DataFrame({'Close': [1, 2, 3]})
    fe = fitness.get_fitness_evaluator(df, {}, {})
    assert isinstance(fe, fitness.FitnessEvaluator)
    cfg.MULTI_ASSET['enabled'] = orig


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
                "min_total_trades": 0,
                "soft_penalty_strength": 0,
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
        "min_total_trades": 0,
        "asset_weights": {"A": 0.6, "B": 0.3, "C": 0.1},
        "soft_penalty_strength": 0,
    }
    ev = _make_evaluator(settings, stats)
    score = ev(None, [], 0)
    mu, sigma = fitness.weighted_mean_std([1.5, 1.0, 0.5], [0.6, 0.3, 0.1])
    expected = mu - 0.25 * sigma
    assert np.isclose(score, expected)


def test_score_clipping_stabilises_mu_sigma():
    stats = [
        {"total_return": 1.0, "trades": 5},
        {"total_return": 2.0, "trades": 5},
        {"total_return": 1000.0, "trades": 5},
    ]
    settings = {
        "metric": "return",
        "lambda_dispersion": 0.0,
        "min_total_trades": 0,
        "soft_penalty_strength": 0,
        "clip_composite_abs": 20,
    }
    ev = _make_evaluator(settings, stats)
    ev(None, [], 0)
    expected_mu, expected_sigma = fitness.weighted_mean_std([1.0, 2.0, 20.0], [1, 1, 1])
    assert np.isclose(ev.last_details["mu"], expected_mu)
    assert np.isclose(ev.last_details["sigma"], expected_sigma)
    assert ev.last_details["per_asset"]["C"]["score"] == 20.0


def test_ga_and_tuner_consistency(monkeypatch):
    stats = {
        'total_return': 1.0,
        'trades': 5,
        'sortino': 1.0,
        'profit_factor': 1.0,
        'max_drawdown': 10.0,
        'equity_curve': pd.Series([1, 1.1, 1.2]),
    }

    def fake_eval(self, ohlc, rules):
        return stats

    group_data = {
        'A': pd.DataFrame({'Close': [1, 2, 3]}),
        'B': pd.DataFrame({'Close': [1, 2, 3]}),
    }

    monkeypatch.setattr(
        fitness.MultiAssetFitnessEvaluator,
        '_evaluate_single_asset',
        fake_eval,
        raising=False,
    )
    monkeypatch.setattr(data_loader, 'get_group_data', lambda *args, **kwargs: group_data)
    monkeypatch.setattr(pd.DataFrame, 'ta', property(lambda self: None), raising=False)
    vbt = sys.modules['vectorbt']
    setattr(vbt, 'Portfolio', object)
    monkeypatch.setitem(cfg.MULTI_ASSET, 'enabled', True)
    monkeypatch.setitem(cfg.MULTI_ASSET, 'metric', 'return')
    monkeypatch.setitem(cfg.MULTI_ASSET, 'lambda_dispersion', 0.0)
    monkeypatch.setitem(cfg.MULTI_ASSET, 'min_total_trades', 0)
    monkeypatch.setitem(cfg.MULTI_ASSET, 'soft_penalty_strength', 0)

    evaluator = fitness.MultiAssetFitnessEvaluator(group_data, {}, {})
    ga = pygad.GA(
        num_generations=1,
        num_parents_mating=1,
        sol_per_pop=1,
        num_genes=1,
        gene_space=[{'low': 0, 'high': 1}],
        gene_type=[float],
        mutation_num_genes=1,
        fitness_func=evaluator.__call__,
    )
    ga.run()
    solution, fit, _ = ga.best_solution()

    score_val = tuner._evaluate_on_validation(solution, {})
    assert np.isclose(fit, score_val)


def test_last_details_include_config_fields():
    stats = {
        "sortino": 1.0,
        "profit_factor": 2.0,
        "max_drawdown": 10.0,
        "trades": 5,
        "total_return": 0.0,
        "equity_curve": None,
    }
    settings = {
        "pf_cap": 7.0,
        "sortino_cap": 9.0,
        "squash": True,
        "squash_params": {"sortino_c": 2.0, "pf_c": 3.0},
        "lambda_dispersion": 0.5,
        "min_total_trades": 0,
        "mode": "ga",
    }
    evaluator = _make_evaluator(settings=settings, stats_list=[stats, stats, stats])
    evaluator(None, [], 0)
    details = evaluator.last_details
    assert details["pf_cap"] == 7.0
    assert details["sortino_cap"] == 9.0
    assert details["squash"] is True
    assert details["squash_params"] == {"sortino_c": 2.0, "pf_c": 3.0}
    assert details["coverage_threshold"] == cfg.COVERAGE_THRESHOLD
