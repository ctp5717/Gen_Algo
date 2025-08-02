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
import pytest  # noqa: E402
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


def test_multi_column_stats_are_reduced(monkeypatch):
    """FitnessEvaluator handles DataFrame stats returned by vectorbt"""
    ohlc = pd.DataFrame({'Close': [1, 2, 3]})
    evaluator = fitness.FitnessEvaluator(ohlc, {'exit_rules': {}}, {})

    # Engine stub returning a DataFrame of entries
    entries = pd.DataFrame({0: [True, False, False]})
    monkeypatch.setattr(fitness.engine, 'process_strategy_rules', lambda *a, **k: entries)

    # vectorbt stub returning stats for two columns
    stats_df = pd.DataFrame({
        'A': {
            'Sortino Ratio': 1.0,
            'Profit Factor': 2.0,
            'Max Drawdown [%]': 10.0,
        },
        'B': {
            'Sortino Ratio': 2.0,
            'Profit Factor': 3.0,
            'Max Drawdown [%]': 20.0,
        },
    })

    class DummyPortfolio:
        def stats(self):
            return stats_df

    portfolio_ns = types.SimpleNamespace(from_signals=lambda *a, **k: DummyPortfolio())
    monkeypatch.setattr(fitness.vbt, 'Portfolio', portfolio_ns)

    monkeypatch.setattr(
        fitness.config,
        'FITNESS_WEIGHTS',
        {'sortino_ratio': 1.0, 'profit_factor': 1.0, 'max_drawdown': 1.0, 'min_trades': 0},
        raising=False,
    )
    monkeypatch.setattr(fitness.config, 'MAX_HOLD_PERIOD', 1, raising=False)
    monkeypatch.setattr(fitness.config, 'TIMEFRAME', '1d', raising=False)

    score = evaluator(None, [], 0)

    expected = (1.5 + 2.5 + (1 - 15 / 100))
    assert score == pytest.approx(expected)
