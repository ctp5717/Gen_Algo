import sys
import types
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Stub heavy deps
sys.modules.setdefault('pandas_ta', types.ModuleType('pandas_ta'))

import analysis  # noqa: E402


def test_run_champion_analysis_plots_without_typeerror(monkeypatch):
    df = pd.DataFrame(
        {
            'Open': [1, 2],
            'High': [1, 2],
            'Low': [1, 2],
            'Close': [1, 2],
            'Volume': [1, 1],
        },
        index=pd.date_range('2020-01-01', periods=2),
    )

    entries = pd.DataFrame(
        {
            'A': [True, False],
            'B': [False, False],
        },
        index=df.index,
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
    monkeypatch.setattr(analysis.config, 'TICKER', 'TEST', raising=False)
    monkeypatch.setattr(analysis.config, 'SELECTED_ASSET_NAME', 'Test', raising=False)
    monkeypatch.setattr(analysis.config, 'STRATEGY_RULES', {}, raising=False)

    monkeypatch.setattr(
        analysis.fitness,
        '_inject_genes_into_rules',
        lambda *a, **k: {'exit_rules': {}},
    )
    monkeypatch.setattr(
        analysis.engine,
        'process_strategy_rules',
        lambda *a, **k: entries,
    )

    metrics = [
        'Start', 'End', 'Period', 'Total Return [%]', 'Benchmark Return [%]',
        'Max Drawdown [%]', 'Sortino Ratio', 'Sharpe Ratio', 'Profit Factor',
        'Win Rate [%]', 'Total Trades', 'Avg Winning Trade [%]',
        'Avg Losing Trade [%]'
    ]

    calls = []

    class DummyAggregatedPortfolio:
        def __init__(self):
            self.wrapper = types.SimpleNamespace(columns=[0])

        def stats(self):
            data = {m: [0] for m in metrics}
            return pd.DataFrame(data)

        def plot(self, *a, **k):
            if k.get('column') != 0:
                raise TypeError('incorrect column passed')

            class DummyFig:
                def show(self):
                    calls.append('show')

            return DummyFig()

    class DummySelectedPortfolio:
        def agg(self, how):
            return DummyAggregatedPortfolio()

    class DummyLoc:
        def __getitem__(self, key):
            _, mask = key
            assert mask.tolist() == [True, False]
            return DummySelectedPortfolio()

    class DummyPortfolio:
        trades = types.SimpleNamespace(
            count=lambda: pd.Series({'A': 1, 'B': 0})
        )
        loc = DummyLoc()

    monkeypatch.setattr(
        analysis.vbt,
        'Portfolio',
        types.SimpleNamespace(from_signals=lambda *a, **k: DummyPortfolio()),
        raising=False,
    )

    monkeypatch.setattr(analysis.plt, 'ion', lambda: None)

    analysis.run_champion_analysis(
        [0],
        {0: {'name': 'x', 'path': [], 'type': float}},
    )

    assert calls == ['show']
