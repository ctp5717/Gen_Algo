import sys
import types
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Stub optional deps
sys.modules.setdefault('pandas_ta', types.ModuleType('pandas_ta'))

import fitness  # noqa: E402
import analysis  # noqa: E402


def test_run_portfolio_backtest_includes_trade_metrics(monkeypatch):
    index = pd.date_range('2020-01-01', periods=4)
    ohlc = pd.DataFrame({
        ('A', 'Close'): [1, 2, 3, 4],
        ('B', 'Close'): [4, 3, 2, 1],
    }, index=index)
    ohlc.columns = pd.MultiIndex.from_tuples(ohlc.columns)
    entries = pd.DataFrame(
        [[True, True], [False, False], [False, False], [False, False]],
        index=index,
        columns=['A', 'B'],
    )
    monkeypatch.setattr(fitness.config, 'TIMEFRAME', '1d', raising=False)
    _, _, agg_stats, _ = fitness.run_portfolio_backtest(ohlc, entries)
    assert 'Volatility' in agg_stats.index
    assert 'Max Consecutive Losses' in agg_stats.index


def test_run_champion_analysis_handles_missing_metrics(monkeypatch, capsys):
    df = pd.DataFrame({
        'Open': [1],
        'High': [1],
        'Low': [1],
        'Close': [1],
        'Volume': [1],
    }, index=pd.date_range('2020-01-01', periods=1))
    monkeypatch.setattr(analysis.data_loader, 'get_data', lambda *a, **k: df)
    monkeypatch.setattr(analysis.config, 'MAX_HOLD_PERIOD', 1, raising=False)
    monkeypatch.setattr(analysis.config, 'TIMEFRAME', '1d', raising=False)
    monkeypatch.setattr(
        analysis.config,
        'VALIDATION_PERIOD',
        {'start': '2020-01-01', 'end': '2020-01-01'},
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
        lambda *a, **k: pd.Series([True], index=df.index),
    )

    class DummyPortfolio:
        def plot(self, *a, **k):
            class DummyFig:
                def show(self):
                    pass

            return DummyFig()

    agg = pd.Series({'Total Return [%]': 0})
    per_asset = pd.DataFrame({'A': [0]}, index=['Total Return [%]'])
    monkeypatch.setattr(
        analysis.fitness,
        'run_portfolio_backtest',
        lambda *a, **k: (DummyPortfolio(), DummyPortfolio(), agg, per_asset),
    )
    monkeypatch.setattr(analysis, 'plt', types.SimpleNamespace(ion=lambda: None))

    analysis.run_champion_analysis([0], {0: {'name': 'x', 'path': [], 'type': float}})
    out = capsys.readouterr().out
    assert 'Total Return [%]' in out
    assert 'Volatility' in out

