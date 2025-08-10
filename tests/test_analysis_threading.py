import sys
import types
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Stub heavy deps
sys.modules.setdefault('pandas_ta', types.ModuleType('pandas_ta'))

import analysis  # noqa: E402


def test_run_champion_analysis_non_blocking(monkeypatch):
    df = pd.DataFrame({
        'Open': [1, 2],
        'High': [1, 2],
        'Low': [1, 2],
        'Close': [1, 2],
        'Volume': [1, 1],
    }, index=pd.date_range('2020-01-01', periods=2))

    monkeypatch.setattr(analysis.data_loader, 'get_data', lambda *a, **k: df)
    monkeypatch.setattr(analysis.config, 'MAX_HOLD_PERIOD', 1, raising=False)
    monkeypatch.setattr(analysis.config, 'TIMEFRAME', '1d', raising=False)
    monkeypatch.setattr(
        analysis.config,
        'VALIDATION_PERIOD',
        {'start': '2020-01-01', 'end': '2020-01-02'},
        raising=False,
    )
    monkeypatch.setattr(analysis.config, 'TICKER', 'TEST', raising=False)
    monkeypatch.setattr(analysis.config, 'SELECTED_ASSET_NAME', 'Test', raising=False)
    monkeypatch.setattr(analysis.config, 'STRATEGY_RULES', {}, raising=False)

    monkeypatch.setattr(
        analysis.fitness,
        '_inject_genes_into_rules',
        lambda *a, **k: {'exit_rules': {}}
    )
    monkeypatch.setattr(
        analysis.engine,
        'process_strategy_rules',
        lambda *a, **k: pd.Series([True, False], index=df.index)
    )

    metrics = [
        'Start', 'End', 'Period', 'Total Return [%]', 'Benchmark Return [%]',
        'Max Drawdown [%]', 'Sortino Ratio', 'Sharpe Ratio', 'Profit Factor',
        'Win Rate [%]', 'Total Trades', 'Avg Winning Trade [%]',
        'Avg Losing Trade [%]', 'Volatility', 'Calmar Ratio', 'Max Consecutive Losses'
    ]

    class DummyPortfolio:
        pass

    agg = pd.Series({m: 0 for m in metrics})
    per_asset = pd.DataFrame({0: [0] * len(metrics)}, index=metrics)

    calls = []

    agg_series = pd.Series([1, 2], index=df.index)

    class DummyFig:
        def __init__(self):
            self.figure = self

        def show(self):
            calls.append('show')

    agg_series.plot = lambda *a, **k: DummyFig()

    monkeypatch.setattr(
        analysis.fitness,
        'run_portfolio_backtest',
        lambda *a, **k: (DummyPortfolio(), agg_series, agg, per_asset),
    )
    ion_called = {}

    monkeypatch.setattr(
        analysis,
        'plt',
        types.SimpleNamespace(
            ion=lambda: ion_called.setdefault('ion', True),
        ),
    )

    analysis.run_champion_analysis(
        [0],
        {0: {'name': 'x', 'path': [], 'type': float}},
    )

    assert ion_called['ion']
    assert calls == ['show']


def test_run_champion_analysis_prints_both_stats(monkeypatch, capsys):
    df = pd.DataFrame(
        {
            'Open': [1],
            'High': [1],
            'Low': [1],
            'Close': [1],
            'Volume': [1],
        },
        index=pd.date_range('2020-01-01', periods=1),
    )
    monkeypatch.setattr(analysis.data_loader, 'get_data', lambda *a, **k: df)
    monkeypatch.setattr(analysis.config, 'MAX_HOLD_PERIOD', 1, raising=False)
    monkeypatch.setattr(analysis.config, 'TIMEFRAME', '1d', raising=False)
    monkeypatch.setattr(
        analysis.config,
        'VALIDATION_PERIOD',
        {'start': '2020-01-01', 'end': '2020-01-02'},
        raising=False,
    )
    monkeypatch.setattr(
        analysis.config,
        'PORTFOLIO_OPTIMIZATION_ENABLED',
        True,
        raising=False,
    )
    monkeypatch.setattr(analysis.config, 'ASSET_BASKET', ['A', 'B'], raising=False)

    monkeypatch.setattr(
        analysis.fitness,
        '_inject_genes_into_rules',
        lambda *a, **k: {'exit_rules': {}},
    )
    monkeypatch.setattr(
        analysis.engine,
        'process_strategy_rules',
        lambda *a, **k: pd.Series([True], index=df.index),
    )

    metrics = [
        'Start', 'End', 'Period', 'Total Return [%]', 'Benchmark Return [%]',
        'Max Drawdown [%]', 'Sortino Ratio', 'Sharpe Ratio', 'Profit Factor',
        'Win Rate [%]', 'Total Trades', 'Avg Winning Trade [%]', 'Avg Losing Trade [%]',
        'Volatility', 'Calmar Ratio', 'Max Consecutive Losses'
    ]
    agg = pd.Series({m: 0 for m in metrics})
    per_asset = pd.DataFrame(
        {'A': [0] * len(metrics), 'B': [0] * len(metrics)},
        index=metrics,
    )

    class DummyPortfolio:
        pass

    agg_series = pd.Series([1], index=df.index)

    class DummyFig:
        def __init__(self):
            self.figure = self

        def show(self):
            pass

    agg_series.plot = lambda *a, **k: DummyFig()

    monkeypatch.setattr(
        analysis.fitness,
        'run_portfolio_backtest',
        lambda *a, **k: (DummyPortfolio(), agg_series, agg, per_asset),
    )

    analysis.run_champion_analysis(
        [0],
        {0: {'name': 'x', 'path': [], 'type': float}},
    )
    out = capsys.readouterr().out
    assert 'Per-Asset Breakdown' in out
    assert 'Total Return [%]' in out
