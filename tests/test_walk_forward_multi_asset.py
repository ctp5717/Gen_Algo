import sys
import types
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Stub heavy optional dependencies
sys.modules.setdefault("pandas_ta", types.ModuleType("pandas_ta"))
sys.modules.setdefault("vectorbt", types.ModuleType("vectorbt"))

warnings.filterwarnings("ignore", category=DeprecationWarning)

import walk_forward  # noqa: E402


def test_walk_forward_multi_asset_stats_finite(monkeypatch):
    idx = pd.date_range("2020-01-01", periods=3)
    columns = pd.MultiIndex.from_product([
        ["A", "B"],
        ["Open", "High", "Low", "Close", "Volume"],
    ])
    df = pd.DataFrame(1.0, index=idx, columns=columns)

    monkeypatch.setattr(walk_forward.data_loader, "get_data", lambda *a, **k: df)
    monkeypatch.setattr(
        walk_forward,
        "_generate_periods",
        lambda *a, **k: [
            {
                "train_start": idx[0],
                "train_end": idx[1],
                "test_start": idx[1],
                "test_end": idx[2],
            }
        ],
    )
    monkeypatch.setattr(walk_forward, "parse_genes_from_config", lambda *a, **k: ([], {}, []))

    class DummyEvaluator:
        def __init__(self, *a, **k):
            pass

        def __call__(self, *a, **k):
            return 1.0

    monkeypatch.setattr(walk_forward.fitness, "FitnessEvaluator", DummyEvaluator)

    class DummyGA:
        def __init__(self, *a, **k):
            pass

        def run(self):
            pass

        def best_solution(self, *a, **k):
            return [], 1.0, None

    monkeypatch.setattr(walk_forward.pygad, "GA", DummyGA)

    test_index = df.index[1:]
    entries = pd.DataFrame({"A": [True, False], "B": [False, False]}, index=test_index)
    monkeypatch.setattr(
        walk_forward.engine,
        "process_strategy_rules",
        lambda *a, **k: entries,
    )

    class DummyTrades:
        def __init__(self, counts):
            self._counts = counts

        def count(self):
            return self._counts

    class DummyLoc:
        def __init__(self, portfolio):
            self.portfolio = portfolio

        def __getitem__(self, item):
            _, cols_mask = item
            counts = self.portfolio.trades.count()[cols_mask]
            return DummyPortfolio(counts)

    class DummyPortfolio:
        def __init__(self, counts):
            self.trades = DummyTrades(counts)
            self.wrapper = types.SimpleNamespace(columns=list(counts.index))
            self.loc = DummyLoc(self)

        def agg(self, how):
            agg_counts = pd.Series([self.trades.count().sum()], index=[0])
            return DummyPortfolio(agg_counts)

        def stats(self, column=None):
            if column is None:
                warnings.warn("mean aggregation", UserWarning)
            return {
                "Total Return [%]": 0.0,
                "Max Drawdown [%]": 0.0,
                "Sharpe Ratio": np.inf,
                "Sortino Ratio": -np.inf,
                "Win Rate [%]": 50.0,
            }

        def plot(self, *a, **k):
            return None

    def from_signals_stub(*a, **k):
        counts = pd.Series([1, 0], index=["A", "B"])
        return DummyPortfolio(counts)

    monkeypatch.setattr(
        walk_forward.vbt,
        "Portfolio",
        types.SimpleNamespace(from_signals=from_signals_stub),
        raising=False,
    )
    monkeypatch.setattr(walk_forward.config, "FITNESS_WEIGHTS", {"min_trades": 0}, raising=False)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        summary = walk_forward.run_walk_forward_validation()
    assert len(w) == 0
    assert summary is not None
    folds = summary["folds"]
    assert np.isfinite(folds["Sharpe Ratio"].iloc[0])
    assert np.isfinite(folds["Sortino Ratio"].iloc[0])
