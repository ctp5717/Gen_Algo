import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import config  # noqa: E402
from multi_asset_fitness import MultiAssetFitnessEvaluator  # noqa: E402


class DummyGA:
    generations_completed = 0


def make_data():
    idx = pd.date_range('2020', periods=5, freq='D')
    up = pd.DataFrame({'Open': [1, 2, 3, 4, 5], 'Close': [2, 3, 4, 5, 6]}, index=idx)
    down = pd.DataFrame({'Open': [6, 5, 4, 3, 2], 'Close': [5, 4, 3, 2, 1]}, index=idx)
    return {'up': up, 'down': down}


def patch_engine(monkeypatch, pattern):
    def fake_process(data, rules):
        return pd.Series(pattern, index=data.index)
    monkeypatch.setattr('strategy_engine.process_strategy_rules', fake_process)


def test_asset_dispersion_penalty(monkeypatch):
    data = make_data()
    patch_engine(monkeypatch, [True, False, True, False, True])
    original_lambda = config.ROBUSTNESS['lambda_asset_dispersion']
    original_maxhold = config.MAX_HOLD_PERIOD
    orig_maxcon = config.SCANNER['max_concurrent_trades']
    config.SCANNER['max_concurrent_trades'] = 2
    config.ROBUSTNESS['lambda_asset_dispersion'] = 0.0
    config.MAX_HOLD_PERIOD = 1
    ga = DummyGA()
    evaluator = MultiAssetFitnessEvaluator(data, {}, {})
    base = evaluator(ga, [], 0)
    config.ROBUSTNESS['lambda_asset_dispersion'] = 1.0
    penalised = evaluator(ga, [], 0)
    assert penalised < base
    config.ROBUSTNESS['lambda_asset_dispersion'] = original_lambda
    config.MAX_HOLD_PERIOD = original_maxhold
    config.SCANNER['max_concurrent_trades'] = orig_maxcon


def test_mc_dispersion_penalty(monkeypatch):
    data = make_data()
    patch_engine(monkeypatch, [True]*5)
    orig_lambda = config.ROBUSTNESS['lambda_mc_dispersion']
    orig_policy = config.SCANNER['tie_break_policy']
    orig_runs = config.SCANNER['monte_carlo_runs']
    orig_maxcon = config.SCANNER['max_concurrent_trades']
    config.SCANNER['tie_break_policy'] = 'random'
    config.SCANNER['max_concurrent_trades'] = 1
    config.SCANNER['monte_carlo_runs'] = 2
    config.ROBUSTNESS['lambda_mc_dispersion'] = 0.0
    ga = DummyGA()
    evaluator = MultiAssetFitnessEvaluator(data, {}, {})
    base = evaluator(ga, [], 0)
    config.ROBUSTNESS['lambda_mc_dispersion'] = 1.0
    penalised = evaluator(ga, [], 0)
    assert penalised < base
    config.ROBUSTNESS['lambda_mc_dispersion'] = orig_lambda
    config.SCANNER['tie_break_policy'] = orig_policy
    config.SCANNER['monte_carlo_runs'] = orig_runs
    config.SCANNER['max_concurrent_trades'] = orig_maxcon


def test_minibatch_uses_subset(monkeypatch):
    data = make_data()
    patch_engine(monkeypatch, [True]*5)
    orig_minibatch = config.MINIBATCH.copy()
    config.MINIBATCH['enabled'] = True
    config.MINIBATCH['size'] = 1
    config.MINIBATCH['elite_eval_period'] = 0
    ga = DummyGA()
    evaluator = MultiAssetFitnessEvaluator(data, {}, {})
    evaluator(ga, [], 0)
    assert len(evaluator.last_assets) == 1
    config.MINIBATCH.update(orig_minibatch)


def test_collects_diagnostics(monkeypatch):
    data = make_data()
    patch_engine(monkeypatch, [True]*5)
    ga = DummyGA()
    evaluator = MultiAssetFitnessEvaluator(data, {}, {})
    evaluator(ga, [], 0)
    assert evaluator.last_open_count is not None
    assert "collisions" in evaluator.last_diagnostics
    assert evaluator.last_trade_counts is not None


def test_asset_metrics_aggregated_across_runs(monkeypatch):
    data = make_data()

    def fake_eval(self, solution, seed, assets):
        if not hasattr(fake_eval, "count"):
            fake_eval.count = 0
        metrics_list = [
            {"up": 0.0, "down": 0.0},
            {"up": 4.0, "down": 8.0},
        ]
        result = (
            1.0,
            metrics_list[fake_eval.count],
            pd.Series(dtype=float),
            pd.Series(dtype=float),
            {"collisions": 0, "rejected": 0, "acceptance_rate": 1.0},
            pd.Series(dtype=float),
        )
        fake_eval.count += 1
        return result

    orig_runs = config.SCANNER["monte_carlo_runs"]
    orig_lambda = config.ROBUSTNESS["lambda_asset_dispersion"]
    config.SCANNER["monte_carlo_runs"] = 2
    config.ROBUSTNESS["lambda_asset_dispersion"] = 1.0
    monkeypatch.setattr(MultiAssetFitnessEvaluator, "_evaluate_once", fake_eval, raising=False)
    ga = DummyGA()
    evaluator = MultiAssetFitnessEvaluator(data, {}, {})
    score = evaluator(ga, [], 0)
    assert pytest.approx(score, rel=1e-3) == 0.0
    config.SCANNER["monte_carlo_runs"] = orig_runs
    config.ROBUSTNESS["lambda_asset_dispersion"] = orig_lambda
