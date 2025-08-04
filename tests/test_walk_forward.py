import sys
import types
from pathlib import Path
from datetime import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Stub heavy optional dependencies
sys.modules.setdefault('pandas_ta', types.ModuleType('pandas_ta'))
sys.modules.setdefault('vectorbt', types.ModuleType('vectorbt'))
warnings.filterwarnings("ignore", category=DeprecationWarning)

import walk_forward  # noqa: E402



def test_generate_periods_produces_windows():
    start = datetime(2020, 1, 1)
    end = datetime(2021, 6, 1)
    periods = walk_forward._generate_periods(start, end, train_months=12, test_months=3)
    assert len(periods) > 0


def test_generate_periods_insufficient_data():
    start = datetime(2020, 1, 1)
    end = datetime(2020, 3, 1)
    periods = walk_forward._generate_periods(start, end, train_months=3, test_months=3)
    assert periods == []


def test_generate_periods_window_consistency():
    start = datetime(2020, 1, 1)
    end = datetime(2020, 12, 31)
    train_months = 6
    test_months = 2
    periods = walk_forward._generate_periods(start, end, train_months, test_months)
    assert periods
    for idx, p in enumerate(periods):
        assert p['train_end'] == p['train_start'] + relativedelta(months=train_months)
        assert p['test_start'] == p['train_end']
        assert p['test_end'] == p['test_start'] + relativedelta(months=test_months)
        if idx > 0:
            expected_start = periods[idx - 1]['train_start'] + relativedelta(months=test_months)
            assert p['train_start'] == expected_start


def test_three_year_history_yields_more_windows():
    start = datetime(2020, 1, 1)
    end = datetime(2023, 1, 1)
    periods = walk_forward._generate_periods(start, end, train_months=12, test_months=3)
    assert len(periods) == 8


def test_config_walk_forward_start_date():
    expected_start = (walk_forward.config.today - relativedelta(years=3)).strftime("%Y-%m-%d")
    assert walk_forward.config.WALK_FORWARD_SETTINGS["total_data_range"]["start"] == expected_start


def test_walk_forward_uses_all_cores(monkeypatch):
    """GA in walk-forward should leverage all available CPU cores"""
    import os
    import pandas as pd
    import types

    captured = {}

    class DummyGA:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

        def run(self):
            return None

        def best_solution(self, **kwargs):
            return [0], 1.0, None

    monkeypatch.setattr(walk_forward.pygad, "GA", DummyGA)

    df = pd.DataFrame(
        {
            "Open": [1, 1],
            "High": [1, 1],
            "Low": [1, 1],
            "Close": [1, 1],
            "Volume": [1, 1],
        },
        index=pd.date_range("2020-01-01", periods=2),
    )

    monkeypatch.setattr(walk_forward.data_loader, "get_data", lambda *a, **k: df)

    monkeypatch.setattr(
        walk_forward,
        "_generate_periods",
        lambda *a, **k: [
            {
                "train_start": df.index[0],
                "train_end": df.index[1],
                "test_start": df.index[0],
                "test_end": df.index[1],
            }
        ],
    )

    class DummyEvaluator:
        def __init__(self, *a, **k):
            pass

        def __call__(self, *a, **k):
            return 1.0

    monkeypatch.setattr(walk_forward.fitness, "FitnessEvaluator", DummyEvaluator)

    monkeypatch.setattr(
        walk_forward.engine,
        "process_strategy_rules",
        lambda *a, **k: pd.Series([True, False], index=df.index),
    )

    class DummyLoc:
        def __getitem__(self, key):
            return DummyPortfolio()

    class DummyPortfolio:
        def __init__(self, *a, **k):
            self.trades = types.SimpleNamespace(count=lambda: pd.Series([1], index=[0]))
            self.loc = DummyLoc()
            self.wrapper = types.SimpleNamespace(columns=[0])

        def agg(self, how):
            return self

        def stats(self):
            return {"Total Return [%]": 0, "Max Drawdown [%]": 0}

        def plot(self, *a, **k):
            return None

    monkeypatch.setattr(
        walk_forward.vbt,
        "Portfolio",
        types.SimpleNamespace(from_signals=lambda *a, **k: DummyPortfolio()),
        raising=False,
    )

    monkeypatch.setattr(walk_forward.config, "FITNESS_WEIGHTS", {"min_trades": 0}, raising=False)

    walk_forward.run_walk_forward_validation()

    assert captured["parallel_processing"][1] == os.cpu_count()


def test_walk_forward_returns_summary(monkeypatch):
    """run_walk_forward_validation should return aggregate metrics"""
    import pandas as pd
    import types

    df = pd.DataFrame(
        {
            "Open": [1, 1],
            "High": [1, 1],
            "Low": [1, 1],
            "Close": [1, 1],
            "Volume": [1, 1],
        },
        index=pd.date_range("2020-01-01", periods=2),
    )

    monkeypatch.setattr(walk_forward.data_loader, "get_data", lambda *a, **k: df)
    monkeypatch.setattr(
        walk_forward,
        "_generate_periods",
        lambda *a, **k: [
            {
                "train_start": df.index[0],
                "train_end": df.index[1],
                "test_start": df.index[0],
                "test_end": df.index[1],
            }
        ],
    )

    # Simplify gene parsing and fitness evaluation
    monkeypatch.setattr(walk_forward, "parse_genes_from_config", lambda *a, **k: ([], {}, []))

    class DummyEvaluator:
        def __init__(self, *a, **k):
            pass

        def __call__(self, *a, **k):
            return 1.0

    monkeypatch.setattr(walk_forward.fitness, "FitnessEvaluator", DummyEvaluator)

    class DummyGA:
        def __init__(self, *args, **kwargs):
            pass

        def run(self):
            return None

        def best_solution(self, **kwargs):
            return [], 1.0, None

    monkeypatch.setattr(walk_forward.pygad, "GA", DummyGA)

    monkeypatch.setattr(
        walk_forward.engine,
        "process_strategy_rules",
        lambda *a, **k: pd.DataFrame({'A': [True, False], 'B': [False, False]}, index=df.index),
    )

    class DummyLoc:
        def __getitem__(self, key):
            return DummyPortfolio()

    class DummyPortfolio:
        def __init__(self, *a, **k):
            self.trades = types.SimpleNamespace(count=lambda: pd.Series([1], index=[0]))
            self.loc = DummyLoc()
            self.wrapper = types.SimpleNamespace(columns=[0])

        def agg(self, how):
            return self

        def stats(self):
            return {
                "Total Return [%]": 1.0,
                "Max Drawdown [%]": 0.0,
                "Sharpe Ratio": np.inf,
                "Sortino Ratio": -np.inf,
                "Win Rate [%]": 50.0,
            }

        def plot(self, *a, **k):
            return None

    monkeypatch.setattr(
        walk_forward.vbt,
        "Portfolio",
        types.SimpleNamespace(from_signals=lambda *a, **k: DummyPortfolio()),
        raising=False,
    )

    monkeypatch.setattr(walk_forward.config, "FITNESS_WEIGHTS", {"min_trades": 0}, raising=False)

    summary = walk_forward.run_walk_forward_validation()

    assert isinstance(summary, dict)
    for key in ["average_return", "total_compounded_return", "folds"]:
        assert key in summary
    assert np.isfinite(summary["average_sharpe"])
    assert np.isfinite(summary["average_sortino"])


def test_walk_forward_portfolio_basket(monkeypatch):
    import pandas as pd

    df = pd.DataFrame({'Close': [1]}, index=pd.date_range('2020-01-01', periods=1))

    captured = {}

    def loader_stub(ticker, *a, **k):
        captured.setdefault('ticker', ticker)
        return df

    monkeypatch.setattr(walk_forward.data_loader, 'get_data', loader_stub)
    monkeypatch.setattr(walk_forward, '_generate_periods', lambda *a, **k: [])
    monkeypatch.setattr(walk_forward.config, 'PORTFOLIO_OPTIMIZATION_ENABLED', True, raising=False)
    monkeypatch.setattr(walk_forward.config, 'ASSET_BASKET', ['A', 'B'], raising=False)
    monkeypatch.setattr(walk_forward.config, 'TICKER', 'A', raising=False)

    walk_forward.run_walk_forward_validation()
    assert captured['ticker'] == ['A', 'B']


def test_update_champion_pool_logic(monkeypatch, capsys):
    settings = {
        "survival_threshold": 0.5,
        "cloning_threshold": 1.0,
        "num_clones": 2,
        "clone_mutation_rate": 0.0,
    }
    gene_space = [{"low": 0, "high": 1, "step": 1}]

    pool = []
    # Discard case
    pool = walk_forward._update_champion_pool(pool, [0], 0.1, gene_space, settings)
    assert pool == []
    assert "discarded" in capsys.readouterr().out.lower()

    # Keep case
    pool = walk_forward._update_champion_pool(pool, [0], 0.7, gene_space, settings)
    assert len(pool) == 1
    assert "kept" in capsys.readouterr().out.lower()

    # Clone case
    pool = walk_forward._update_champion_pool(pool, [1], 1.2, gene_space, settings)
    assert len(pool) == 1 + 1 + settings["num_clones"]
    out = capsys.readouterr().out.lower()
    assert "cloning" in out
