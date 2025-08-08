import sys
import types
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Stub heavy deps
sys.modules.setdefault("pandas_ta", types.ModuleType("pandas_ta"))

import analysis  # noqa: E402


def test_run_champion_analysis_missing_metrics(monkeypatch, capsys):
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

    class DummyPortfolio:
        def __init__(self, name="agg"):
            self.name = name

        def stats(self, *args, **kwargs):
            if self.name == "agg":
                return pd.DataFrame(
                    {
                        "A": {"Total Return [%]": 1.0},
                        "B": {"Total Return [%]": 2.0},
                    }
                )
            else:
                return pd.Series({"Total Return [%]": 1.0})

        def plot(self, *a, **k):
            class DummyFig:
                def show(self):
                    pass

            return DummyFig()

        def total(self):
            return DummyPortfolio("total")

        def __getitem__(self, key):
            return DummyPortfolio(str(key))

    monkeypatch.setattr(
        analysis.vbt,
        "Portfolio",
        types.SimpleNamespace(from_signals=lambda *a, **k: DummyPortfolio()),
        raising=False,
    )
    monkeypatch.setattr(analysis, "plt", types.SimpleNamespace(ion=lambda: None))

    analysis.run_champion_analysis(
        [0],
        {0: {"name": "x", "path": [], "type": float}},
    )

    out = capsys.readouterr().out
    assert "Missing metrics:" in out
    assert "Missing metrics for A:" in out
    assert "Missing metrics for B:" in out
