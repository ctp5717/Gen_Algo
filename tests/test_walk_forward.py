import sys
import types
from pathlib import Path
from datetime import datetime
from dateutil.relativedelta import relativedelta

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Stub heavy optional dependencies
sys.modules.setdefault('pandas_ta', types.ModuleType('pandas_ta'))
sys.modules.setdefault('vectorbt', types.ModuleType('vectorbt'))

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

    class DummyPortfolio:
        def __init__(self, *a, **k):
            pass

        def stats(self):
            return {"Total Return [%]": 0, "Max Drawdown [%]": 0}

    monkeypatch.setattr(
        walk_forward.vbt,
        "Portfolio",
        types.SimpleNamespace(from_signals=lambda *a, **k: DummyPortfolio()),
        raising=False,
    )

    monkeypatch.setattr(walk_forward.config, "FITNESS_WEIGHTS", {"min_trades": 0}, raising=False)

    walk_forward.run_walk_forward_validation()

    assert captured["parallel_processing"][1] == os.cpu_count()
