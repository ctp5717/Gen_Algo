import sys
import types
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

sys.modules.setdefault('pandas_ta', types.ModuleType('pandas_ta'))

import analysis  # noqa: E402


def test_run_champion_analysis_multigroup(monkeypatch):
    idx = pd.date_range('2020-01-01', periods=5, freq='D')
    assets = ['AAA', 'BBB']
    fields = ['Open', 'High', 'Low', 'Close']
    columns = pd.MultiIndex.from_product([assets, fields])
    data = pd.DataFrame(
        np.random.rand(len(idx), len(columns)), index=idx, columns=columns
    )

    monkeypatch.setattr(analysis.data_loader, 'get_data', lambda *a, **k: data)
    monkeypatch.setattr(
        analysis.config,
        'VALIDATION_PERIOD',
        {'start': '2020-01-01', 'end': '2020-01-05'},
        raising=False,
    )
    monkeypatch.setattr(analysis.config, 'PORTFOLIO_OPTIMIZATION_ENABLED', True, raising=False)
    monkeypatch.setattr(analysis.config, 'ASSET_BASKET', assets, raising=False)
    monkeypatch.setattr(analysis.config, 'TIMEFRAME', '1d', raising=False)
    monkeypatch.setattr(analysis.config, 'STRATEGY_RULES', {}, raising=False)
    monkeypatch.setattr(analysis.config, 'PORTFOLIO_WEIGHTS', None, raising=False)
    monkeypatch.setattr(analysis.config, 'TICKER', 'AAA', raising=False)
    monkeypatch.setattr(analysis.config, 'SELECTED_ASSET_NAME', 'Test', raising=False)

    monkeypatch.setattr(
        analysis.fitness, '_inject_genes_into_rules', lambda *a, **k: {'exit_rules': {}}
    )
    close_cols = data.xs('Close', level=1, axis=1).columns
    entries = pd.DataFrame(False, index=data.index, columns=close_cols)
    entries.iloc[0] = True
    monkeypatch.setattr(
        analysis.engine, 'process_strategy_rules', lambda *a, **k: entries
    )

    class DummyFig:
        def show(self):
            pass

    class DummyPortfolio:
        def __init__(self):
            self.wrapper = SimpleNamespace(grouper=True, ndim=2)
            self.columns = close_cols

        def plot(self, *a, **k):
            return DummyFig()

    class DummyAggPortfolio:
        def __init__(self):
            self.wrapper = SimpleNamespace(grouper=None, ndim=1)

        def plot(self, *a, **k):
            return DummyFig()

    agg_stats = pd.Series({'Total Return [%]': 0.0})
    per_asset_stats = pd.DataFrame({'AAA': agg_stats, 'BBB': agg_stats})

    monkeypatch.setattr(
        analysis.fitness,
        'run_portfolio_backtest',
        lambda *a, **k: (DummyPortfolio(), DummyAggPortfolio(), agg_stats, per_asset_stats),
    )

    monkeypatch.setattr(analysis, 'plt', types.SimpleNamespace(ion=lambda: None))

    analysis.run_champion_analysis([], {})
