import sys
import types
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Stub optional dependencies
sys.modules.setdefault('pandas_ta', types.ModuleType('pandas_ta'))

import fitness  # noqa: E402


def test_fitness_uses_aggregated_stats(monkeypatch):
    ohlc = pd.DataFrame(
        {
            ('A', 'Close'): [1],
            ('B', 'Close'): [1],
        },
        index=pd.date_range('2020-01-01', periods=1),
    )
    ohlc.columns = pd.MultiIndex.from_tuples(ohlc.columns)

    monkeypatch.setattr(
        fitness.engine,
        'process_strategy_rules',
        lambda *a, **k: pd.DataFrame(
            [[True, True]],
            index=ohlc.index,
            columns=['A', 'B'],
        ),
    )
    monkeypatch.setattr(
        fitness.config,
        'FITNESS_WEIGHTS',
        {
            'sortino_ratio': 1,
            'profit_factor': 0,
            'max_drawdown': 0,
            'min_trades': 0,
        },
        raising=False,
    )

    agg = pd.Series(
        {
            'Sortino Ratio': 2.0,
            'Profit Factor': 1.0,
            'Max Drawdown [%]': 5.0,
        }
    )
    per_asset = pd.DataFrame(
        {'A': [1, 1, 1], 'B': [1, 1, 1]},
        index=['Sortino Ratio', 'Profit Factor', 'Max Drawdown [%]'],
    )
    monkeypatch.setattr(
        fitness,
        'run_portfolio_backtest',
        lambda *a, **k: (None, None, agg, per_asset),
    )

    evaluator = fitness.FitnessEvaluator(ohlc, {}, {})
    score = evaluator(None, [], 0)
    assert score == 2.0


def test_run_portfolio_backtest_weights(monkeypatch):
    import numpy as np

    ohlc = pd.DataFrame(
        {
            ('A', 'Close'): [1, 2],
            ('B', 'Close'): [1000, 1002],
        },
        index=pd.date_range('2020-01-01', periods=2),
    )
    ohlc.columns = pd.MultiIndex.from_tuples(ohlc.columns)
    entries = pd.DataFrame(
        [[True, True], [False, False]],
        index=ohlc.index,
        columns=['A', 'B'],
    )

    captured = {}

    class DummyAgg:
        def stats(self):
            return pd.Series({'Total Return [%]': 0})

    class DummyPortfolio:
        def stats(self):
            return pd.DataFrame({'A': [0], 'B': [0]}, index=['Total Return [%]'])

        def total(self):
            return DummyAgg()

    monkeypatch.setattr(
        fitness.vbt.Portfolio,
        'from_signals',
        lambda **k: (captured.update(k) or DummyPortfolio()),
    )

    _, agg_pf, agg_stats, per_asset = fitness.run_portfolio_backtest(ohlc, entries)
    assert np.allclose(captured['weights'], [0.5, 0.5])
    assert isinstance(agg_pf, DummyAgg)
    assert isinstance(agg_stats, pd.Series)
    assert list(per_asset.columns) == ['A', 'B']

    captured.clear()
    _, _, _, _ = fitness.run_portfolio_backtest(
        ohlc,
        entries,
        weights=[0.7, 0.3],
    )
    assert np.allclose(captured['weights'], [0.7, 0.3])
