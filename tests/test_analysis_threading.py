import sys
import types
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Stub heavy deps
sys.modules.setdefault("pandas_ta", types.ModuleType("pandas_ta"))
try:  # use real vectorbt if available
    import vectorbt  # noqa: F401
except Exception:  # pragma: no cover - fallback to stub
    sys.modules.setdefault("vectorbt", types.ModuleType("vectorbt"))

import analysis  # noqa: E402


def test_run_champion_analysis_non_blocking(monkeypatch):
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

    monkeypatch.setattr(analysis.config, "MAX_HOLD_PERIOD", 1, raising=False)
    monkeypatch.setattr(analysis.config, "TIMEFRAME", "1d", raising=False)
    monkeypatch.setattr(
        analysis.config,
        "VALIDATION_PERIOD",
        {"start": "2020-01-01", "end": "2020-01-02"},
        raising=False,
    )
    monkeypatch.setattr(analysis.config, "TICKER", "TEST", raising=False)
    monkeypatch.setattr(analysis.config, "SELECTED_ASSET_NAME", "Test", raising=False)
    monkeypatch.setattr(analysis.config, "STRATEGY_RULES", {}, raising=False)
    monkeypatch.setitem(analysis.config.MULTI_ASSET, "enabled", False)

    monkeypatch.setattr(
        analysis.fitness, "_inject_genes_into_rules", lambda *a, **k: {"exit_rules": {}}
    )
    monkeypatch.setattr(
        analysis.engine,
        "process_strategy_rules",
        lambda *a, **k: pd.Series([True, False], index=df.index),
    )

    metrics = [
        "Start",
        "End",
        "Period",
        "Total Return [%]",
        "Benchmark Return [%]",
        "Max Drawdown [%]",
        "Sortino Ratio",
        "Sharpe Ratio",
        "Profit Factor",
        "Win Rate [%]",
        "Total Trades",
        "Avg Winning Trade [%]",
        "Avg Losing Trade [%]",
    ]

    class DummyPortfolio:
        def stats(self):
            return pd.DataFrame({m: [0] for m in metrics})

        def plot(self, *a, **k):
            class DummyFig:
                def show(self):
                    calls.append("show")

            return DummyFig()

    monkeypatch.setattr(
        analysis.vbt,
        "Portfolio",
        types.SimpleNamespace(from_signals=lambda *a, **k: DummyPortfolio()),
        raising=False,
    )
    monkeypatch.setattr(analysis, "ensure_real_vectorbt", lambda *a, **k: None)

    calls = []
    ion_called = {}

    monkeypatch.setattr(
        analysis,
        "plt",
        types.SimpleNamespace(
            ion=lambda: ion_called.setdefault("ion", True),
            savefig=lambda *a, **k: None,
            close=lambda *a, **k: None,
        ),
    )

    analysis.run_champion_analysis(
        [0],
        {0: {"name": "x", "path": [], "type": float}},
        df,
    )

    assert ion_called["ion"]
    assert calls == ["show"]
