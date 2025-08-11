import sys
import types
from pathlib import Path
import pandas as pd
import numpy as np

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
            'Volatility': 1.0,
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

    class DummyPortfolio:
        def stats(self):
            return pd.DataFrame({'A': [0], 'B': [0]}, index=['Total Return [%]'])

        def value(self):
            # provide constant value series for each asset
            return pd.DataFrame(
                {
                    'A': [100, 100],
                    'B': [100, 100],
                },
                index=ohlc.index,
            )

    orig_from_signals = fitness.vbt.Portfolio.from_signals

    def fake_from_signals(*a, **k):
        # Calls originating from from_holding pass boolean entries/exits
        if isinstance(k.get('entries'), bool):
            return orig_from_signals(*a, **k)
        captured.update(k)
        return DummyPortfolio()

    monkeypatch.setattr(
        fitness.vbt.Portfolio,
        'from_signals',
        fake_from_signals,
    )

    _, agg_pf, agg_stats, per_asset = fitness.run_portfolio_backtest(ohlc, entries)
    assert 'weights' not in captured
    assert isinstance(agg_stats, pd.Series)
    assert list(per_asset.columns) == ['A', 'B']

    captured.clear()
    _, _, _, _ = fitness.run_portfolio_backtest(
        ohlc,
        entries,
        weights=[0.7, 0.3],
    )
    assert 'weights' not in captured


def test_run_portfolio_backtest_aggregates_equity_and_trades(monkeypatch):
    ohlc = pd.DataFrame(
        {
            ('A', 'Close'): [1, 2, 3],
            ('B', 'Close'): [4, 5, 6],
        },
        index=pd.date_range('2020-01-01', periods=3),
    )
    ohlc.columns = pd.MultiIndex.from_tuples(ohlc.columns)
    entries = pd.DataFrame(
        [[True, False], [False, True], [False, False]],
        index=ohlc.index,
        columns=['A', 'B'],
    )

    class DummyPortfolio:
        def stats(self, column=None, silence_warnings=None):
            return pd.Series({'Total Trades': 1, 'Win Rate [%]': np.nan})

        def value(self):
            return pd.DataFrame(
                {'A': [100, 110, 110], 'B': [200, 200, 210]}, index=ohlc.index
            )

        def returns(self):
            return pd.DataFrame(
                {'A': [0.1, 0.0, 0.0], 'B': [0.0, 0.0, 0.05]}, index=ohlc.index
            )

        @property
        def trades(self):
            class T:
                @property
                def records_readable(self):
                    return pd.DataFrame({'PnL': [10, -5], 'Column': ['A', 'B']})

            return T()

    monkeypatch.setattr(
        fitness.vbt.Portfolio, 'from_signals', lambda *a, **k: DummyPortfolio()
    )

    portfolio, agg_equity, agg_stats, per_asset_stats = fitness.run_portfolio_backtest(
        ohlc, entries, weights=[0.6, 0.4]
    )

    expected = (portfolio.value() * np.array([0.6, 0.4])).sum(axis=1)
    assert agg_equity.equals(expected)
    assert agg_stats['Total Trades'] == 2 and isinstance(
        agg_stats['Total Trades'], (int, np.integer)
    )
    assert per_asset_stats.loc['Win Rate [%]'].tolist() == [0.0, 0.0]
    assert all(isinstance(v, float) for v in per_asset_stats.loc['Win Rate [%]'])
    assert not per_asset_stats.isna().any().any()
    assert not agg_stats.isna().any()


def test_fitness_penalizes_zero_volatility(monkeypatch):
    ohlc = pd.DataFrame({('A', 'Close'): [1]}, index=pd.date_range('2020-01-01', periods=1))
    ohlc.columns = pd.MultiIndex.from_tuples(ohlc.columns)

    monkeypatch.setattr(
        fitness.engine,
        'process_strategy_rules',
        lambda *a, **k: pd.DataFrame([[True]], index=ohlc.index, columns=['A']),
    )
    monkeypatch.setattr(
        fitness.config,
        'FITNESS_WEIGHTS',
        {'sortino_ratio': 1, 'profit_factor': 0, 'max_drawdown': 0, 'min_trades': 0},
        raising=False,
    )

    agg = pd.Series(
        {
            'Sortino Ratio': 1.0,
            'Profit Factor': 1.0,
            'Max Drawdown [%]': 5.0,
            'Volatility': 0.0,
        }
    )
    monkeypatch.setattr(
        fitness,
        'run_portfolio_backtest',
        lambda *a, **k: (None, None, agg, None),
    )

    evaluator = fitness.FitnessEvaluator(ohlc, {}, {})
    score = evaluator(None, [], 0)
    assert np.isfinite(score) and score == -999.0


def test_fitness_is_finite_when_no_trades(monkeypatch):
    ohlc = pd.DataFrame({'Close': [1, 2]}, index=pd.date_range('2020-01-01', periods=2))

    monkeypatch.setattr(
        fitness.engine,
        'process_strategy_rules',
        lambda *a, **k: pd.DataFrame([[False], [False]], index=ohlc.index, columns=['Close']),
    )
    monkeypatch.setattr(
        fitness.config,
        'FITNESS_WEIGHTS',
        {'sortino_ratio': 1, 'profit_factor': 0, 'max_drawdown': 0, 'min_trades': 1},
        raising=False,
    )

    evaluator = fitness.FitnessEvaluator(ohlc, {}, {})
    score = evaluator(None, [], 0)
    assert np.isfinite(score)
