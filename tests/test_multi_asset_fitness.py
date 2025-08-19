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

    evaluator = fitness.MultiAssetFitnessEvaluator({'A': df}, {}, {}, {})
    score = evaluator(None, [], 0)
    assert score == -999.0
    assert evaluator.last_details['total_trades'] == 0
    assert evaluator.last_details['penalties']['trade_floor'] == 'hard_floor'


def test_aggregation_math():
    stats = [
        {'total_return': 1.6, 'trades': 5},
        {'total_return': 1.0, 'trades': 5},
        {'total_return': 0.4, 'trades': 5},
    ]
    settings = {
        'metric': 'return',
        'lambda_dispersion': 0.25,
        'trade_floor_policy': 'hard_floor',
        'min_total_trades': 0,
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


def test_trade_floor_policies():
    stats = [
        {'total_return': 1.0, 'trades': 5},
        {'total_return': 1.0, 'trades': 5},
        {'total_return': 1.0, 'trades': 5},
    ]
    settings_hard = {
        'metric': 'return',
        'trade_floor_policy': 'hard_floor',
        'min_total_trades': 20,
    }
    ev_hard = _make_evaluator(settings_hard, stats)
    assert ev_hard(None, [], 0) == -999.0

    settings_soft = {
        'metric': 'return',
        'trade_floor_policy': 'soft_penalty',
        'min_total_trades': 30,
        'soft_penalty_strength': 1.0,
    }
    ev_soft = _make_evaluator(settings_soft, stats)
    assert np.isclose(ev_soft(None, [], 0), 0.5)


def test_hard_floor_returns_poor_score_with_zero_trade_penalize():
    stats = [
        {'total_return': 0.0, 'trades': 0},
        {'total_return': 1.0, 'trades': 5},
        {'total_return': 1.0, 'trades': 5},
    ]
    settings = {
        'metric': 'return',
        'zero_trade_policy': 'penalize',
        'zero_trade_penalty': -1.0,
        'per_asset_min_trades': 1,
        'trade_floor_policy': 'hard_floor',
        'min_total_trades': 30,
        'lambda_dispersion': 0.0,
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
        'zero_trade_policy': 'ignore',
        'coverage_penalty_weight': 0.0,
        'per_asset_min_trades': 1,
        'trade_floor_policy': 'hard_floor',
        'min_total_trades': 30,
        'lambda_dispersion': 0.0,
    }
    ev = _make_evaluator(settings, stats)
    assert ev(None, [], 0) == -999.0


def test_zero_trade_policy_penalize_vs_ignore():
    stats = [
        {'total_return': 0.0, 'trades': 0},
        {'total_return': 1.0, 'trades': 5},
        {'total_return': 1.0, 'trades': 5},
    ]
    penalize_settings = {
        'metric': 'return',
        'zero_trade_policy': 'penalize',
        'zero_trade_penalty': -1.0,
        'per_asset_min_trades': 1,
        'trade_floor_policy': 'hard_floor',
        'min_total_trades': 0,
        'lambda_dispersion': 0.0,
    }
    ev_pen = _make_evaluator(penalize_settings, stats)
    assert np.isclose(ev_pen(None, [], 0), 1/3, atol=1e-6)

    ignore_settings = {
        'metric': 'return',
        'zero_trade_policy': 'ignore',
        'coverage_penalty_weight': 0.3,
        'per_asset_min_trades': 1,
        'trade_floor_policy': 'hard_floor',
        'min_total_trades': 0,
        'lambda_dispersion': 0.0,
    }
    ev_ign = _make_evaluator(ignore_settings, stats)
    assert np.isclose(ev_ign(None, [], 0), 0.9)
    ignored = ev_ign.last_details['per_asset']['A']
    assert ignored['included'] is False
    assert ignored.get('ignored_reason') == 'insufficient_trades'


@pytest.mark.parametrize("min_trades", [0, 1])
def test_zero_trade_assets_excluded_with_coverage_penalty(min_trades):
    stats = [
        {'total_return': 0.0, 'trades': 0},
        {'total_return': 1.0, 'trades': 5},
        {'total_return': 1.0, 'trades': 5},
    ]
    settings = {
        'metric': 'return',
        'zero_trade_policy': 'ignore',
        'coverage_penalty_weight': 0.3,
        'per_asset_min_trades': min_trades,
        'trade_floor_policy': 'hard_floor',
        'min_total_trades': 0,
        'lambda_dispersion': 0.0,
    }
    ev = _make_evaluator(settings, stats)
    score = ev(None, [], 0)
    assert np.isclose(score, 0.9)
    per_asset = ev.last_details['per_asset']
    ignored = per_asset['A']
    assert ignored['included'] is False
    assert ignored.get('ignored_reason') == 'insufficient_trades'
    assert np.isclose(ev.last_details['penalties']['coverage'], 0.1)


@pytest.mark.parametrize("weight,expected", [(None, 1.0), (0.0, 1.0), (0.3, 0.9)])
def test_coverage_penalty_weight_monotonic(weight, expected):
    stats = [
        {'total_return': 0.0, 'trades': 0},
        {'total_return': 1.0, 'trades': 5},
        {'total_return': 1.0, 'trades': 5},
    ]
    settings = {
        'metric': 'return',
        'zero_trade_policy': 'ignore',
        'coverage_penalty_weight': weight,
        'per_asset_min_trades': 1,
        'trade_floor_policy': 'hard_floor',
        'min_total_trades': 0,
        'lambda_dispersion': 0.0,
    }
    ev = _make_evaluator(settings, stats)
    assert np.isclose(ev(None, [], 0), expected)


def test_weight_renormalization():
    stats = [
        {'total_return': 1.0, 'trades': 5},
        {'total_return': 0.0, 'trades': 5},
        {'total_return': 0.0, 'trades': 0},
    ]
    settings = {
        'metric': 'return',
        'zero_trade_policy': 'ignore',
        'coverage_penalty_weight': 0.3,
        'per_asset_min_trades': 1,
        'asset_weights': {'A': 0.6, 'B': 0.2, 'C': 0.2},
        'trade_floor_policy': 'hard_floor',
        'min_total_trades': 0,
        'lambda_dispersion': 0.0,
    }
    ev = _make_evaluator(settings, stats)
    assert np.isclose(ev(None, [], 0), 0.65, atol=1e-6)


def test_weight_renormalization_multiple_exclusions():
    stats = [
        {'total_return': 0.5, 'trades': 5},
        {'total_return': 1.0, 'trades': 0},
        {'total_return': 1.5, 'trades': 0},
    ]
    settings = {
        'metric': 'return',
        'zero_trade_policy': 'ignore',
        'coverage_penalty_weight': 0.0,
        'per_asset_min_trades': 1,
        'asset_weights': {'A': 0.2, 'B': 0.3, 'C': 0.5},
        'trade_floor_policy': 'hard_floor',
        'min_total_trades': 0,
        'lambda_dispersion': 0.0,
    }
    ev = _make_evaluator(settings, stats)
    score = ev(None, [], 0)
    assert np.isclose(score, 0.5)
    assert ev.last_details['assets_included'] == 1


def test_soft_penalty_additive():
    stats = [
        {'total_return': 1.0, 'trades': 5},
        {'total_return': 1.0, 'trades': 5},
        {'total_return': 1.0, 'trades': 5},
    ]
    settings = {
        'metric': 'return',
        'trade_floor_policy': 'soft_penalty',
        'soft_penalty_mode': 'additive',
        'soft_penalty_strength': 2.0,
        'min_total_trades': 30,
    }
    ev = _make_evaluator(settings, stats)
    # Mean = 1.0, total trades = 15 => penalty 2*(1-0.5)=1 => fitness 0
    assert np.isclose(ev(None, [], 0), 0.0)


def test_soft_penalty_multiplicative_scaling():
    stats = [
        {'total_return': 1.0, 'trades': 5},
        {'total_return': 1.0, 'trades': 5},
        {'total_return': 1.0, 'trades': 0},
    ]
    settings = {
        'metric': 'return',
        'zero_trade_policy': 'ignore',
        'coverage_penalty_weight': 0.0,
        'per_asset_min_trades': 1,
        'trade_floor_policy': 'soft_penalty',
        'soft_penalty_strength': 2.0,
        'min_total_trades': 20,
        'lambda_dispersion': 0.0,
    }
    ev = _make_evaluator(settings, stats)
    # Total trades = 10, floor = 20 -> scale=(0.5)**2=0.25, mean=1 => fitness=0.25
    assert np.isclose(ev(None, [], 0), 0.25)


def test_min_total_trades_per_year_scaling():
    stats = [
        {'total_return': 1.0, 'trades': 5},
        {'total_return': 1.0, 'trades': 5},
    ]
    settings = {
        'metric': 'return',
        'trade_floor_policy': 'hard_floor',
        'min_total_trades_per_year': 12,
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
    penalize_settings = {
        'metric': 'return',
        'per_asset_min_trades': 3,
        'zero_trade_policy': 'penalize',
        'zero_trade_penalty': -1.0,
        'trade_floor_policy': 'hard_floor',
        'min_total_trades': 0,
        'lambda_dispersion': 0.0,
    }
    ev_pen = _make_evaluator(penalize_settings, stats)
    assert np.isclose(ev_pen(None, [], 0), 1/3, atol=1e-6)

    ignore_settings = {
        'metric': 'return',
        'per_asset_min_trades': 3,
        'zero_trade_policy': 'ignore',
        'coverage_penalty_weight': 0.0,
        'trade_floor_policy': 'hard_floor',
        'min_total_trades': 0,
        'lambda_dispersion': 0.0,
    }
    ev_ign = _make_evaluator(ignore_settings, stats)
    assert np.isclose(ev_ign(None, [], 0), 1.0)


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
        "zero_trade_policy": "penalize",
        "zero_trade_penalty": -1.0,
        "trade_floor_policy": "hard_floor",
        "min_total_trades": 0,
        "lambda_dispersion": 0.0,
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
    assert b.get("penalties", {}).get("zero_trades") == -1.0


def test_diagnostics_and_factory(monkeypatch):
    stats = [
        {'total_return': 1.0, 'trades': 5},
        {'total_return': 0.0, 'trades': 5},
        {'total_return': 0.0, 'trades': 0},
    ]
    settings = {
        'metric': 'return',
        'zero_trade_policy': 'ignore',
        'coverage_penalty_weight': 0.3,
        'per_asset_min_trades': 1,
        'asset_weights': {'A': 0.6, 'B': 0.2, 'C': 0.2},
        'trade_floor_policy': 'soft_penalty',
        'soft_penalty_mode': 'additive',
        'soft_penalty_strength': 1.0,
        'min_total_trades': 30,
        'lambda_dispersion': 0.25,
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
                "trade_floor_policy": "hard_floor",
                "min_total_trades": 0,
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
    }
    ev = _make_evaluator(settings, stats)
    score = ev(None, [], 0)
    mu, sigma = fitness.weighted_mean_std([1.5, 1.0, 0.5], [0.6, 0.3, 0.1])
    expected = mu - 0.25 * sigma
    assert np.isclose(score, expected)


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
    monkeypatch.setitem(cfg.MULTI_ASSET, 'trade_floor_policy', 'hard_floor')

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
