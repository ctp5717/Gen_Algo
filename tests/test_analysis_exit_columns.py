import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# provide dummy pandas_ta module if missing
sys.modules.setdefault('pandas_ta', types.ModuleType('pandas_ta'))

import analysis  # noqa: E402


class DummyFig:
    def __init__(self):
        self.figure = self

    def show(self):
        pass

    def scatter(self, *args, **kwargs):
        pass

    def legend(self, *args, **kwargs):
        pass


class DummyPortfolio:
    def __init__(self, trades_df, index, column):
        self.trades = SimpleNamespace(records_readable=trades_df)
        self.wrapper = SimpleNamespace(grouper=None, ndim=1)
        # used by run_champion_analysis for weight mapping
        self._value = pd.DataFrame({column: np.ones(len(index))}, index=index)

    def plot(self, *args, **kwargs):
        return DummyFig()

    def value(self):
        return self._value


@pytest.mark.parametrize("exit_col", ["Exit Time", "Exit Price"])
def test_run_champion_analysis_accepts_exit_variants(monkeypatch, exit_col):
    idx = pd.date_range('2020-01-01', periods=3, freq='D')
    column = 'AAA'
    data = pd.DataFrame({'Close': np.arange(len(idx))}, index=idx)

    monkeypatch.setattr(analysis.data_loader, 'get_data', lambda *a, **k: data)
    monkeypatch.setattr(
        analysis.config,
        'VALIDATION_PERIOD',
        {'start': '2020-01-01', 'end': '2020-01-03'},
        raising=False,
    )
    monkeypatch.setattr(analysis.config, 'TIMEFRAME', '1d', raising=False)
    monkeypatch.setattr(analysis.config, 'STRATEGY_RULES', {}, raising=False)
    monkeypatch.setattr(analysis.config, 'PORTFOLIO_OPTIMIZATION_ENABLED', False, raising=False)
    monkeypatch.setattr(analysis.config, 'PORTFOLIO_WEIGHTS', None, raising=False)
    monkeypatch.setattr(analysis.config, 'TICKER', column, raising=False)
    monkeypatch.setattr(analysis.config, 'SELECTED_ASSET_NAME', column, raising=False)

    monkeypatch.setattr(
        analysis.fitness,
        '_inject_genes_into_rules',
        lambda *a, **k: {'exit_rules': {}}
    )
    entries = pd.DataFrame(False, index=idx, columns=[column])
    entries.iloc[0] = True
    monkeypatch.setattr(
        analysis.engine,
        'process_strategy_rules',
        lambda *a, **k: entries
    )

    trades_df = pd.DataFrame({
        exit_col: [idx[0]],
        'PnL': [1.0],
        'Column': [column],
    })

    agg_series = pd.Series(np.ones(len(idx)), index=idx)
    agg_series.plot = lambda *a, **k: DummyFig()
    stats = pd.Series({'Total Return [%]': 0.0})

    monkeypatch.setattr(
        analysis.fitness,
        'run_portfolio_backtest',
        lambda *a, **k: (DummyPortfolio(trades_df, idx, column), agg_series, stats, stats)
    )

    monkeypatch.setattr(analysis, 'plt', types.SimpleNamespace(ion=lambda: None))

    # Should not raise even if exit column is not named "Exit"
    analysis.run_champion_analysis([], {})
