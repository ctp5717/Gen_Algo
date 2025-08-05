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
import numpy as np  # noqa: E402
import logging  # noqa: E402


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

    # vectorbt stub returning stats for two numeric columns and one non-numeric column
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
        'C': {
            'Sortino Ratio': 'x',
            'Profit Factor': 'y',
            'Max Drawdown [%]': 'z',
        },
    })

    class DummyPortfolio:
        def stats(self, *args, **kwargs):
            return stats_df

    portfolio_ns = types.SimpleNamespace(from_signals=lambda *a, **k: DummyPortfolio())
    monkeypatch.setattr(fitness.vbt, 'Portfolio', portfolio_ns, raising=False)

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


def test_inject_genes_casts_int():
    base = {'rule': {'period': 0}}
    gene_map = {0: {'path': ['rule', 'period'], 'type': int}}
    injected = fitness._inject_genes_into_rules(base, gene_map, [5.7])
    assert injected['rule']['period'] == 5
    assert isinstance(injected['rule']['period'], int)


def test_portfolio_stats_called_without_agg(monkeypatch):
    ohlc = pd.DataFrame({'Close': [1, 2, 3]})
    evaluator = fitness.FitnessEvaluator(ohlc, {'exit_rules': {}}, {})

    entries = pd.Series([True, False, False])
    monkeypatch.setattr(fitness.engine, 'process_strategy_rules', lambda *a, **k: entries)

    called = {}

    class DummyPortfolio:
        def stats(self, *args, **kwargs):
            called.update(kwargs)
            return pd.Series({'Sortino Ratio': 1.0, 'Profit Factor': 1.0, 'Max Drawdown [%]': 10.0})

    portfolio_ns = types.SimpleNamespace(from_signals=lambda *a, **k: DummyPortfolio())
    monkeypatch.setattr(fitness.vbt, 'Portfolio', portfolio_ns, raising=False)

    monkeypatch.setattr(
        fitness.config,
        'FITNESS_WEIGHTS',
        {'sortino_ratio': 1.0, 'profit_factor': 1.0, 'max_drawdown': 1.0, 'min_trades': 0},
        raising=False,
    )
    monkeypatch.setattr(fitness.config, 'MAX_HOLD_PERIOD', 1, raising=False)
    monkeypatch.setattr(fitness.config, 'TIMEFRAME', '1d', raising=False)

    evaluator(None, [], 0)

    assert 'agg_func' in called and called['agg_func'] is None


@pytest.mark.parametrize(
    "stats_series,expected",
    [
        (pd.Series({'Sortino Ratio': np.nan, 'Profit Factor': 1.0, 'Max Drawdown [%]': 10.0}), 'Sortino Ratio'),
        (pd.Series({'Sortino Ratio': 1.0, 'Profit Factor': 0.0, 'Max Drawdown [%]': 10.0}), 'Profit Factor'),
        (pd.Series({'Sortino Ratio': 1.0, 'Profit Factor': 1.0, 'Max Drawdown [%]': 100.0}), 'Drawdown Score'),
    ],
)
def test_invalid_metrics_penalize(monkeypatch, caplog, stats_series, expected):
    ohlc = pd.DataFrame({'Close': [1, 2, 3]})
    evaluator = fitness.FitnessEvaluator(ohlc, {'exit_rules': {}}, {})

    entries = pd.Series([True, False, False])
    monkeypatch.setattr(fitness.engine, 'process_strategy_rules', lambda *a, **k: entries)

    class DummyPortfolio:
        def stats(self, *args, **kwargs):
            return stats_series

    portfolio_ns = types.SimpleNamespace(from_signals=lambda *a, **k: DummyPortfolio())
    monkeypatch.setattr(fitness.vbt, 'Portfolio', portfolio_ns, raising=False)

    monkeypatch.setattr(
        fitness.config,
        'FITNESS_WEIGHTS',
        {'sortino_ratio': 1.0, 'profit_factor': 1.0, 'max_drawdown': 1.0, 'min_trades': 0},
        raising=False,
    )
    monkeypatch.setattr(fitness.config, 'MAX_HOLD_PERIOD', 1, raising=False)
    monkeypatch.setattr(fitness.config, 'TIMEFRAME', '1d', raising=False)

    with caplog.at_level(logging.DEBUG):
        score = evaluator(None, [], 0)

    assert score == -999.0
    assert expected in caplog.text
