import os
import sys
import types
import pandas as pd
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import config  # noqa: E402

# Stub heavy dependency
sys.modules.setdefault('pandas_ta', types.ModuleType('pandas_ta'))
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
            metrics_list[fake_eval.count % len(metrics_list)],
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


def test_exit_parity_with_single_asset(monkeypatch):
    import vectorbt as vbt

    idx = pd.date_range('2020', periods=5, freq='D')
    data = {
        'sl': pd.DataFrame({'Close': [100, 89, 88, 87, 86]}, index=idx),
        'tp': pd.DataFrame({'Close': [100, 111, 112, 113, 114]}, index=idx),
        'tsl': pd.DataFrame({'Close': [100, 105, 104, 103, 99]}, index=idx),
    }

    def fake_process(data, rules):
        return pd.Series([True] + [False] * (len(data.index) - 1), index=data.index)

    monkeypatch.setattr('strategy_engine.process_strategy_rules', fake_process)

    exit_rules = {
        'stop_loss': {'is_active': True, 'type': 'percentage', 'params': {'value': 0.1}},
        'trailing_stop': {'is_active': True, 'type': 'percentage', 'params': {'value': 0.05}},
        'take_profit': {'is_active': True, 'type': 'percentage', 'params': {'value': 0.1}},
    }

    base_rules = {'exit_rules': exit_rules}
    evaluator = MultiAssetFitnessEvaluator(data, base_rules, {})

    orig_hold = config.MAX_HOLD_PERIOD
    orig_maxcon = config.SCANNER['max_concurrent_trades']
    config.MAX_HOLD_PERIOD = 10
    config.SCANNER['max_concurrent_trades'] = 3

    entries_df, exits_df, _scores, sl_stop, tp_stop, sl_trail = evaluator._build_signals([], list(data.keys()))

    for name, df in data.items():
        entries = pd.Series([True] + [False] * (len(df.index) - 1), index=df.index)
        time_exit = entries.shift(config.MAX_HOLD_PERIOD, fill_value=False)
        time_exit = time_exit.reindex(df.index, fill_value=False)
        pf = vbt.Portfolio.from_signals(
            close=df['Close'],
            entries=entries,
            exits=time_exit,
            sl_stop=sl_stop,
            tp_stop=tp_stop,
            sl_trail=sl_trail,
            fees=config.FEES,
            slippage=getattr(config, 'SLIPPAGE', 0.0),
            freq=config.TIMEFRAME,
        )
        sells = pf.orders.records_readable
        sells = sells[sells['Side'] == 'Sell']['Timestamp']
        expected = pd.Series(False, index=df.index, name=name)
        if len(sells):
            expected.loc[sells] = True
        pd.testing.assert_series_equal(exits_df[name], expected)

    config.MAX_HOLD_PERIOD = orig_hold
    config.SCANNER['max_concurrent_trades'] = orig_maxcon
