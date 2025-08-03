import sys
import types
from pathlib import Path

# Ensure repository root is on the import path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Stub heavy optional dependencies before importing modules that use them
sys.modules.setdefault('pandas_ta', types.ModuleType('pandas_ta'))
sys.modules.setdefault('vectorbt', types.ModuleType('vectorbt'))

import pandas as pd  # noqa: E402
import fitness  # noqa: E402


def test_exception_logging(capsys, monkeypatch):
    """FitnessEvaluator prints exception messages"""
    ohlc = pd.DataFrame({'Close': [1, 2, 3]})
    evaluator = fitness.FitnessEvaluator(ohlc, {}, {})

    def raise_error(*args, **kwargs):
        raise ValueError("boom")

    monkeypatch.setattr(fitness.engine, 'process_strategy_rules', raise_error)

    score = evaluator(None, [], 0)

    captured = capsys.readouterr()
    assert "boom" in captured.out
    assert score == -999.0


def test_aggregates_multi_asset_portfolio(monkeypatch):
    """Aggregates value and computes metrics without using stats()."""
    # Minimal OHLC data
    ohlc = pd.DataFrame({'Close': [1, 2, 3]})
    evaluator = fitness.FitnessEvaluator(ohlc, {}, {})

    # Provide entries with enough trades
    entries = pd.Series([True, True, True])
    monkeypatch.setattr(
        fitness.engine, 'process_strategy_rules', lambda *a, **k: entries
    )

    # Simplify config weights
    monkeypatch.setattr(
        fitness.config,
        'FITNESS_WEIGHTS',
        {'min_trades': 0, 'sortino_ratio': 1, 'profit_factor': 1, 'max_drawdown': 1},
        raising=False,
    )
    monkeypatch.setattr(fitness.config, 'MAX_HOLD_PERIOD', 1, raising=False)
    monkeypatch.setattr(fitness.config, 'TIMEFRAME', '1d', raising=False)

    class DummyPnl:
        def __init__(self, data):
            self.data = data

        def to_pd(self):
            return self.data

    class DummyTrades:
        pnl = DummyPnl(pd.DataFrame({'a': [1.0, -0.5], 'b': [0.5, -1.0]}))

    class DummyPortfolio:
        def __init__(self):
            self.value_called = False

        def value(self):
            self.value_called = True
            return pd.DataFrame({'a': [100, 105, 110], 'b': [200, 198, 202]})

        def stats(self):  # Should never be called
            raise AssertionError("stats should not be called")

        @property
        def trades(self):
            return DummyTrades()

    holder = {}

    class DummyPortfolioClass:
        @staticmethod
        def from_signals(**kwargs):
            holder['p'] = DummyPortfolio()
            return holder['p']

    monkeypatch.setattr(fitness.vbt, 'Portfolio', DummyPortfolioClass, raising=False)

    score = evaluator(None, [], 0)

    # With proper aggregation the evaluator returns a finite score and uses value()
    assert score > 0
    assert holder['p'].value_called


def test_handles_single_asset_portfolio(monkeypatch):
    """Handles portfolios whose value and PnL are Series."""
    ohlc = pd.DataFrame({'Close': [1, 2, 3]})
    evaluator = fitness.FitnessEvaluator(ohlc, {}, {})

    entries = pd.Series([True, True, True])
    monkeypatch.setattr(
        fitness.engine, 'process_strategy_rules', lambda *a, **k: entries
    )

    monkeypatch.setattr(
        fitness.config,
        'FITNESS_WEIGHTS',
        {'min_trades': 0, 'sortino_ratio': 1, 'profit_factor': 1, 'max_drawdown': 1},
        raising=False,
    )
    monkeypatch.setattr(fitness.config, 'MAX_HOLD_PERIOD', 1, raising=False)
    monkeypatch.setattr(fitness.config, 'TIMEFRAME', '1d', raising=False)

    class DummyPnl:
        def __init__(self, data):
            self.data = data

        def to_pd(self):
            return self.data

    class DummyTrades:
        pnl = DummyPnl(pd.Series([1.0, -0.5]))

    class DummyPortfolio:
        def value(self):
            return pd.Series([100, 105, 110])

        @property
        def trades(self):
            return DummyTrades()

    class DummyPortfolioClass:
        @staticmethod
        def from_signals(**kwargs):
            return DummyPortfolio()

    monkeypatch.setattr(fitness.vbt, 'Portfolio', DummyPortfolioClass, raising=False)

    score = evaluator(None, [], 0)
    assert score > 0
