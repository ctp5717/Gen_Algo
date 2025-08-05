import sys
import types
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Stub heavy deps
sys.modules.setdefault('pandas_ta', types.ModuleType('pandas_ta'))

import analysis  # noqa: E402


def _stub_portfolio(monkeypatch):
    metrics = [
        'Start',
        'End',
        'Period',
        'Total Return [%]',
        'Benchmark Return [%]',
        'Max Drawdown [%]',
        'Sortino Ratio',
        'Sharpe Ratio',
        'Profit Factor',
        'Win Rate [%]',
        'Total Trades',
        'Avg Winning Trade [%]',
        'Avg Losing Trade [%]'
    ]

    class DummyPortfolio:
        def stats(self):
            return pd.DataFrame({m: [0] for m in metrics})

        def plot(self, *a, **k):
            class DummyFig:
                def show(self):
                    pass

            return DummyFig()

    monkeypatch.setattr(
        analysis.vbt,
        'Portfolio',
        types.SimpleNamespace(from_signals=lambda *a, **k: DummyPortfolio()),
        raising=False,
    )
    monkeypatch.setattr(analysis, 'plt', types.SimpleNamespace(ion=lambda: None))


def _base_config(monkeypatch, assets):
    monkeypatch.setattr(analysis.config, 'MAX_HOLD_PERIOD', 1, raising=False)
    monkeypatch.setattr(analysis.config, 'TIMEFRAME', '1d', raising=False)
    monkeypatch.setattr(
        analysis.config,
        'VALIDATION_PERIOD',
        {'start': '2020-01-01', 'end': '2020-01-02'},
        raising=False,
    )
    monkeypatch.setattr(analysis.config, 'PORTFOLIO_OPTIMIZATION_ENABLED', True, raising=False)
    monkeypatch.setattr(analysis.config, 'ASSET_BASKET', assets, raising=False)
    monkeypatch.setattr(analysis.config, 'STRATEGY_RULES', {}, raising=False)
    monkeypatch.setattr(
        analysis.fitness,
        '_inject_genes_into_rules',
        lambda *a, **k: {'exit_rules': {}},
    )


def test_run_champion_analysis_warns_and_drops_missing(monkeypatch, capsys):
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

    multi = pd.concat({'A': df}, axis=1)
    monkeypatch.setattr(analysis.data_loader, 'get_data', lambda *a, **k: multi)
    _base_config(monkeypatch, ['A', 'B'])
    _stub_portfolio(monkeypatch)

    called = {}

    def fake_process_strategy_rules(data, rules):
        called['called'] = True
        assert list(data.columns.get_level_values(0).unique()) == ['A']
        return pd.DataFrame({'A': [True, False]}, index=df.index)

    monkeypatch.setattr(
        analysis.engine,
        'process_strategy_rules',
        fake_process_strategy_rules,
    )

    analysis.run_champion_analysis([0], {0: {'name': 'x', 'path': [], 'type': float}})
    out = capsys.readouterr().out
    assert 'Missing data for tickers: B' in out
    assert called.get('called')


def test_run_champion_analysis_aborts_on_excess_missing(monkeypatch):
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

    multi = pd.concat({'A': df}, axis=1)
    monkeypatch.setattr(analysis.data_loader, 'get_data', lambda *a, **k: multi)
    _base_config(monkeypatch, ['A', 'B', 'C'])
    _stub_portfolio(monkeypatch)

    called = {}

    def fake_process_strategy_rules(*a, **k):
        called['called'] = True
        return pd.DataFrame({'A': [True, False]}, index=df.index)

    monkeypatch.setattr(
        analysis.engine,
        'process_strategy_rules',
        fake_process_strategy_rules,
    )

    analysis.run_champion_analysis([0], {0: {'name': 'x', 'path': [], 'type': float}})
    assert 'called' not in called
