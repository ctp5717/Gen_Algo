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
    """Uses total().stats() to avoid pandas multi-column warnings."""
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

    # Dummy vectorbt Portfolio that requires calling total().stats()
    class DummyTotal:
        def stats(self):
            return pd.Series(
                {
                    'Sortino Ratio': 1.0,
                    'Profit Factor': 1.0,
                    'Max Drawdown [%]': 10.0,
                }
            )

    class DummyPortfolio:
        def stats(self):
            raise ValueError("stats should not be called directly")

        def total(self):
            return DummyTotal()

    class DummyPortfolioClass:
        @staticmethod
        def from_signals(**kwargs):
            return DummyPortfolio()

    monkeypatch.setattr(fitness.vbt, 'Portfolio', DummyPortfolioClass, raising=False)

    score = evaluator(None, [], 0)

    # With proper aggregation the evaluator returns a finite score
    assert score > 0
