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
        'Avg Losing Trade [%]'
    ]

    class DummyPortfolio:
        def stats(self, *args, **kwargs):
            return pd.Series({m: 0 for m in metrics})

        def plot(self, *a, **k):
            class DummyFig:
                def show(self):
                    calls.append('show')

            return DummyFig()

    monkeypatch.setattr(
        analysis.vbt,
        'Portfolio',
        types.SimpleNamespace(from_signals=lambda *a, **k: DummyPortfolio()),
        raising=False
    )

    calls = []
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


def test_run_champion_analysis_asset_breakdown(monkeypatch, capsys):
    df = pd.DataFrame(
        {
            "Open": [1, 2],
            "High": [1, 2],
            "Low": [1, 2],
            "Close": [1, 2],
            "Volume": [1, 1],
        },
        index=pd.date_range("2020-01-01", periods=2),
    )

    multi = pd.concat({"A": df, "B": df}, axis=1)

    monkeypatch.setattr(analysis.data_loader, "get_data", lambda *a, **k: multi)
    monkeypatch.setattr(analysis.config, "MAX_HOLD_PERIOD", 1, raising=False)
    monkeypatch.setattr(analysis.config, "TIMEFRAME", "1d", raising=False)
    monkeypatch.setattr(
        analysis.config,
        "VALIDATION_PERIOD",
        {"start": "2020-01-01", "end": "2020-01-02"},
        raising=False,
    )
    monkeypatch.setattr(analysis.config, "PORTFOLIO_OPTIMIZATION_ENABLED", True, raising=False)
    monkeypatch.setattr(analysis.config, "ASSET_BASKET", ["A", "B"], raising=False)
    monkeypatch.setattr(analysis.config, "STRATEGY_RULES", {}, raising=False)

    monkeypatch.setattr(
        analysis.fitness,
        "_inject_genes_into_rules",
        lambda *a, **k: {"exit_rules": {}},
    )
    monkeypatch.setattr(
        analysis.engine,
        "process_strategy_rules",
        lambda *a, **k: pd.DataFrame({"A": [True, False], "B": [True, False]}, index=df.index),
    )

    plot_calls = []
    total_called = []

    class DummyPortfolio:
        def __init__(self, name="agg"):
            self.name = name

        def stats(self, *args, **kwargs):
            if self.name == "agg":
                return pd.DataFrame(
                    {
                        "A": {
                            "Start": 0,
                            "End": 0,
                            "Period": 0,
                            "Total Return [%]": 10.0,
                            "Benchmark Return [%]": 0.0,
                            "Max Drawdown [%]": 5.0,
                            "Sortino Ratio": 1.0,
                            "Sharpe Ratio": 1.0,
                            "Profit Factor": 1.0,
                            "Win Rate [%]": 60.0,
                            "Total Trades": 5,
                            "Avg Winning Trade [%]": 1.0,
                            "Avg Losing Trade [%]": -1.0,
                        },
                        "B": {
                            "Start": 0,
                            "End": 0,
                            "Period": 0,
                            "Total Return [%]": 20.0,
                            "Benchmark Return [%]": 0.0,
                            "Max Drawdown [%]": 15.0,
                            "Sortino Ratio": 2.0,
                            "Sharpe Ratio": 2.0,
                            "Profit Factor": 2.0,
                            "Win Rate [%]": 40.0,
                            "Total Trades": 7,
                            "Avg Winning Trade [%]": 2.0,
                            "Avg Losing Trade [%]": -2.0,
                        },
                    }
                )
            else:
                data = {
                    "Start": 0,
                    "End": 0,
                    "Period": 0,
                    "Total Return [%]": 10.0 if self.name == "A" else 20.0,
                    "Benchmark Return [%]": 0.0,
                    "Max Drawdown [%]": 5.0 if self.name == "A" else 15.0,
                    "Sortino Ratio": 1.0 if self.name == "A" else 2.0,
                    "Sharpe Ratio": 1.0 if self.name == "A" else 2.0,
                    "Profit Factor": 1.0 if self.name == "A" else 2.0,
                    "Win Rate [%]": 60.0 if self.name == "A" else 40.0,
                    "Total Trades": 5 if self.name == "A" else 7,
                    "Avg Winning Trade [%]": 1.0 if self.name == "A" else 2.0,
                    "Avg Losing Trade [%]": -1.0 if self.name == "A" else -2.0,
                }
                return pd.Series(data)

        def plot(self, *a, **k):
            plot_calls.append({"name": self.name, "column": k.get("column")})

            class DummyFig:
                def show(self):
                    pass

            return DummyFig()

        def total(self):
            total_called.append(True)
            return DummyPortfolio("total")

        def __getitem__(self, key):
            return DummyPortfolio(key)

    monkeypatch.setattr(
        analysis.vbt,
        "Portfolio",
        types.SimpleNamespace(from_signals=lambda *a, **k: DummyPortfolio()),
        raising=False,
    )

    analysis.run_champion_analysis([0], {0: {"name": "x", "path": [], "type": float}})

    out = capsys.readouterr().out
    assert "Per-Asset Performance Breakdown" in out
    assert "Asset: A" in out
    assert "Asset: B" in out
    assert "Total Return [%]" in out
    assert "15.0" in out  # mean of 10 and 20
    assert "12" in out  # total trades summed
    assert total_called
    assert plot_calls[-1]["name"] == "total"
