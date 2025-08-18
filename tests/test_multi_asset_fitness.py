import os
import sys
import types
import pandas as pd
import numpy as np
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import config  # noqa: E402

# Tests operate on tiny synthetic datasets that do not meet the default
# minimum trade threshold used by the evaluator.  Override the global
# configuration so that the penalty does not interfere with unrelated
# assertions.
config.MIN_TRADES = 0

# Stub heavy dependency
sys.modules.setdefault('pandas_ta', types.ModuleType('pandas_ta'))
from multi_asset_fitness import MultiAssetFitnessEvaluator, EvalResult  # noqa: E402


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


def fake_eval_parallel(self, solution, seed, assets):
    """Lightweight stand-in for `_evaluate_once` used in multiprocessing tests."""
    return EvalResult(
        float(seed),
        {},
        pd.Series(dtype=float),
        pd.Series(dtype=float),
        {},
        pd.Series(dtype=float),
        0.0,
    )


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


def test_concentration_penalty(monkeypatch):
    data = make_data()

    def fake_process(data, rules):
        if data['Close'].iloc[0] == 2:  # identify 'up' asset
            return pd.Series([True, False, True, False, True], index=data.index)
        return pd.Series(False, index=data.index)

    monkeypatch.setattr('strategy_engine.process_strategy_rules', fake_process)

    original_lambda = config.ROBUSTNESS['lambda_concentration']
    original_maxhold = config.MAX_HOLD_PERIOD
    config.ROBUSTNESS['lambda_concentration'] = 0.0
    config.MAX_HOLD_PERIOD = 1
    ga = DummyGA()
    evaluator = MultiAssetFitnessEvaluator(data, {}, {})
    base = evaluator(ga, [], 0)
    config.ROBUSTNESS['lambda_concentration'] = 1.0
    penalised = evaluator(ga, [], 0)
    assert penalised < base
    config.ROBUSTNESS['lambda_concentration'] = original_lambda
    config.MAX_HOLD_PERIOD = original_maxhold


def test_monte_carlo_median_with_random_tie_break(monkeypatch, caplog):
    import statistics

    data = make_data()
    patch_engine(monkeypatch, [True] * 5)

    orig_policy = config.SCANNER['tie_break_policy']
    orig_runs = config.SCANNER['monte_carlo_runs']
    orig_seed = config.SCANNER.get('seed', 0)

    config.SCANNER['tie_break_policy'] = 'random'
    config.SCANNER['monte_carlo_runs'] = 5
    config.SCANNER['seed'] = 10

    seeds: list[int] = []

    def fake_eval(self, solution, seed, assets):
        seeds.append(seed)
        return EvalResult(
            float(seed),
            {},
            pd.Series(dtype=float),
            pd.Series(dtype=float),
            {},
            pd.Series(dtype=float),
            0.0,
        )

    monkeypatch.setattr(
        MultiAssetFitnessEvaluator, '_evaluate_once', fake_eval, raising=False
    )

    ga = DummyGA()
    ga.generations_completed = 1
    ga.sol_per_pop = 10
    sol_idx = 3
    evaluator = MultiAssetFitnessEvaluator(data, {}, {})
    with caplog.at_level("DEBUG"):
        score = evaluator(ga, [], sol_idx)

    base = (
        config.SCANNER["seed"]
        + ga.generations_completed * ga.sol_per_pop
        + sol_idx
    )
    expected_seeds = [base + i for i in range(config.SCANNER['monte_carlo_runs'])]
    assert seeds == expected_seeds
    assert score == statistics.median(expected_seeds)
    for s in expected_seeds:
        assert f"using seed {s}" in caplog.text

    config.SCANNER['tie_break_policy'] = orig_policy
    config.SCANNER['monte_carlo_runs'] = orig_runs
    config.SCANNER['seed'] = orig_seed


def test_multiprocessing_backend(monkeypatch):
    """Parallel backend should aggregate the same score as sequential runs."""
    data = make_data()
    monkeypatch.setattr(
        MultiAssetFitnessEvaluator, '_evaluate_once', fake_eval_parallel, raising=False
    )

    class DummyPool:
        def __init__(self, processes=None):
            self.processes = processes

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def starmap(self, func, args):
            return [func(*a) for a in args]

    # Patch the Pool used inside the module to avoid spawning real processes
    monkeypatch.setattr('multi_asset_fitness.mp.Pool', DummyPool)

    orig_runs = config.SCANNER['monte_carlo_runs']
    orig_parallel = config.PARALLEL.copy()

    config.SCANNER['monte_carlo_runs'] = 4
    config.PARALLEL['backend'] = 'multiprocessing'
    config.PARALLEL['workers'] = 2

    ga = DummyGA()
    evaluator = MultiAssetFitnessEvaluator(data, {}, {})
    parallel_score = evaluator(ga, [], 0)

    config.PARALLEL['backend'] = None
    sequential_score = evaluator(ga, [], 0)

    assert parallel_score == sequential_score

    config.SCANNER['monte_carlo_runs'] = orig_runs
    config.PARALLEL.update(orig_parallel)


def test_minibatch_uses_subset(monkeypatch):
    data = make_data()
    patch_engine(monkeypatch, [True] * 5)
    orig_minibatch = config.MINIBATCH.copy()
    config.MINIBATCH['enabled'] = True
    config.MINIBATCH['size'] = 1
    config.MINIBATCH['elite_eval_period'] = 0
    ga = DummyGA()
    evaluator = MultiAssetFitnessEvaluator(data, {}, {})
    evaluator(ga, [], 0)
    first = list(evaluator.last_assets)
    evaluator(ga, [], 1)
    # Same generation => same asset subset
    assert evaluator.last_assets == first
    ga.generations_completed += 1
    evaluator(ga, [], 0)
    assert len(evaluator.last_assets) == 1
    config.MINIBATCH.update(orig_minibatch)


def test_elite_rescored_on_full_assets(monkeypatch):
    data = make_data()
    patch_engine(monkeypatch, [True] * 5)
    orig_minibatch = config.MINIBATCH.copy()
    config.MINIBATCH.update({
        'enabled': True,
        'size': 1,
        'elite_eval_period': 1,
        'elite_count': 1,
    })
    ga = DummyGA()
    ga.generations_completed = 1
    evaluator = MultiAssetFitnessEvaluator(data, {}, {})
    evaluator(ga, [], 0)
    # Elite solution rescored on full asset set
    assert set(evaluator.last_assets) == set(evaluator.assets)
    evaluator(ga, [], 1)
    # Non-elite uses minibatch
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


def test_monte_carlo_diagnostics_saved_and_logged(monkeypatch, caplog):
    data = make_data()

    scores = [1.0, 2.0, 3.0]
    metrics = {"up": 1.0, "down": 2.0}

    def fake_eval(self, solution, seed, assets):
        idx = fake_eval.calls
        fake_eval.calls += 1
        return EvalResult(
            scores[idx],
            metrics,
            pd.Series(dtype=float),
            pd.Series(dtype=float),
            {},
            pd.Series(dtype=float),
            0.0,
        )

    fake_eval.calls = 0

    orig_runs = config.SCANNER["monte_carlo_runs"]
    orig_lambda_asset = config.ROBUSTNESS["lambda_asset_dispersion"]
    orig_lambda_mc = config.ROBUSTNESS["lambda_mc_dispersion"]

    config.SCANNER["monte_carlo_runs"] = 3
    config.ROBUSTNESS["lambda_asset_dispersion"] = 1.0
    config.ROBUSTNESS["lambda_mc_dispersion"] = 1.0

    monkeypatch.setattr(MultiAssetFitnessEvaluator, "_evaluate_once", fake_eval, raising=False)

    ga = DummyGA()
    evaluator = MultiAssetFitnessEvaluator(data, {}, {})

    with caplog.at_level("DEBUG"):
        evaluator(ga, [], 0)

    diag = evaluator.last_diagnostics
    assert diag["run_scores"] == scores
    assert pytest.approx(diag["mc_median"]) == np.median(scores)
    assert pytest.approx(diag["dispersion"]) == np.std(scores)
    assert pytest.approx(diag["asset_dispersion"]) == 0.5
    assert pytest.approx(diag["mc_dispersion"]) == np.std(scores)

    assert "run_scores=[1.0, 2.0, 3.0]" in caplog.text

    config.SCANNER["monte_carlo_runs"] = orig_runs
    config.ROBUSTNESS["lambda_asset_dispersion"] = orig_lambda_asset
    config.ROBUSTNESS["lambda_mc_dispersion"] = orig_lambda_mc


def test_asset_metrics_aggregated_across_runs(monkeypatch):
    data = make_data()

    def fake_eval(self, solution, seed, assets):
        if not hasattr(fake_eval, "count"):
            fake_eval.count = 0
        metrics_list = [
            {"up": 0.0, "down": 0.0},
            {"up": 4.0, "down": 8.0},
        ]
        result = EvalResult(
            1.0,
            metrics_list[fake_eval.count % len(metrics_list)],
            pd.Series(dtype=float),
            pd.Series(dtype=float),
            {"collisions": 0, "rejected": 0, "acceptance_rate": 1.0},
            pd.Series(dtype=float),
            0.0,
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

    (
        _close_df,
        entries_df,
        exits_df,
        _scores,
        sl_stop,
        tp_stop,
        sl_trail,
    ) = evaluator._build_signals([], list(data.keys()))

    for name, df in data.items():
        raw_entries = pd.Series([True] + [False] * (len(df.index) - 1), index=df.index)
        shifted_entries = raw_entries.shift(config.ENTRY_LAG_BARS, fill_value=False)
        time_exit = shifted_entries.shift(config.MAX_HOLD_PERIOD, fill_value=False)
        pf = vbt.Portfolio.from_signals(
            close=df['Close'],
            entries=shifted_entries,
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


def test_shifted_entries_and_next_bar_returns(monkeypatch):
    """Entries use data from ``t`` but trades execute on ``t+1``.

    The evaluator should shift signals forward one bar so the first trade
    occurs on the bar following the signal.  Returns are therefore realised
    starting from that next bar.
    """

    import vectorbt as vbt

    idx = pd.date_range('2020', periods=4, freq='D')
    data = {'a': pd.DataFrame({'Close': [100, 110, 120, 130]}, index=idx)}

    def fake_process(df, rules):
        # Signal on the first bar only (time ``t``)
        return pd.Series([True, False, False, False], index=df.index)

    monkeypatch.setattr('strategy_engine.process_strategy_rules', fake_process)

    captured: list[pd.Series] = []
    orig_from_signals = vbt.Portfolio.from_signals

    def capture_entries(*args, **kwargs):
        entries = kwargs.get('entries')
        if entries is None and len(args) >= 2:
            entries = args[1]
        captured.append(entries.copy())
        return orig_from_signals(*args, **kwargs)

    monkeypatch.setattr(vbt.Portfolio, 'from_signals', capture_entries)

    orig_hold = config.MAX_HOLD_PERIOD
    config.MAX_HOLD_PERIOD = 1

    evaluator = MultiAssetFitnessEvaluator(data, {}, {})
    evaluator._evaluate_once([], seed=None, assets=['a'])

    # Second call to ``from_signals`` is during portfolio evaluation
    used_entries = captured[1]
    assert used_entries.index[used_entries].tolist() == [idx[1]]

    # Returns should start one bar after execution
    time_exit = used_entries.shift(config.MAX_HOLD_PERIOD, fill_value=False)
    pf = orig_from_signals(
        close=data['a']['Close'],
        entries=used_entries,
        exits=time_exit,
        freq='D',
    )
    non_zero = pf.returns().loc[pf.returns() != 0]
    assert non_zero.index[0] == idx[2]

    config.MAX_HOLD_PERIOD = orig_hold


def test_single_asset_matches_single_eval(monkeypatch):
    data = make_data()
    patch_engine(monkeypatch, [True, False, True, False, True])
    orig_hold = config.MAX_HOLD_PERIOD
    config.MAX_HOLD_PERIOD = 1

    evaluator = MultiAssetFitnessEvaluator({'up': data['up']}, {}, {})
    res = evaluator._evaluate_once(
        [], seed=0, assets=['up']
    )
    portfolio_returns = res.portfolio_returns
    trade_counts = res.trade_counts

    import vectorbt as vbt  # local import to avoid heavy dependency at module load

    entries = pd.Series([True, False, True, False, True], index=data['up'].index)
    shifted = entries.shift(config.ENTRY_LAG_BARS, fill_value=False)
    time_exit = shifted.shift(config.MAX_HOLD_PERIOD, fill_value=False)
    pf = vbt.Portfolio.from_signals(
        close=data['up']['Close'],
        entries=shifted,
        exits=time_exit,
        fees=config.FEES,
        slippage=getattr(config, 'SLIPPAGE', 0.0),
        freq=config.TIMEFRAME,
    )
    expected_returns = pf.returns()
    expected_trades = pf.trades.count()

    pd.testing.assert_series_equal(
        portfolio_returns, expected_returns, rtol=1e-6, atol=1e-8, check_names=False
    )
    assert trade_counts['up'] == expected_trades

    config.MAX_HOLD_PERIOD = orig_hold
