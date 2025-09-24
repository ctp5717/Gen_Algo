import sys
import types
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from dateutil.relativedelta import relativedelta

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Stub heavy optional dependencies
sys.modules.setdefault("pandas_ta", types.ModuleType("pandas_ta"))
try:  # prefer real vectorbt when installed
    import vectorbt  # noqa: F401
except Exception:  # pragma: no cover - fallback stub
    sys.modules.setdefault("vectorbt", types.ModuleType("vectorbt"))

import walk_forward  # noqa: E402

walk_forward.config.initialize_config()


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
        assert p["train_end"] == p["train_start"] + relativedelta(months=train_months)
        assert p["test_start"] == p["train_end"]
        assert p["test_end"] == p["test_start"] + relativedelta(months=test_months)
        if idx > 0:
            expected_start = periods[idx - 1]["train_start"] + relativedelta(
                months=test_months
            )
            assert p["train_start"] == expected_start


def test_three_year_history_yields_more_windows():
    start = datetime(2020, 1, 1)
    end = datetime(2023, 1, 1)
    periods = walk_forward._generate_periods(start, end, train_months=12, test_months=3)
    assert len(periods) == 8


def test_config_walk_forward_start_date():
    expected_start = (walk_forward.config.today - relativedelta(years=3)).strftime(
        "%Y-%m-%d"
    )
    assert (
        walk_forward.config.WALK_FORWARD_SETTINGS["total_data_range"]["start"]
        == expected_start
    )


def test_walk_forward_uses_all_cores(monkeypatch, tmp_path):
    """GA in walk-forward should leverage all available CPU cores"""
    import os
    import types

    import pandas as pd
    import vectorbt as vbt

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

    monkeypatch.setattr(
        walk_forward.data_loader, "get_data", lambda *a, **k: (df, "cache")
    )

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
            self.last_exit_params = {}

        def __call__(self, *a, **k):
            return 1.0

    fitness_stub = types.SimpleNamespace(
        FitnessEvaluator=DummyEvaluator,
        get_fitness_evaluator=lambda *a, **k: DummyEvaluator(),
        print_floor_failures=lambda *a, **k: None,
    )
    engine_stub = types.SimpleNamespace(
        process_strategy_rules=lambda *a, **k: pd.Series([True, False], index=df.index)
    )
    monkeypatch.setitem(sys.modules, "fitness", fitness_stub)
    monkeypatch.setitem(sys.modules, "strategy_engine", engine_stub)
    monkeypatch.setattr(walk_forward, "inject_genes_into_rules", lambda *a, **k: {})

    class DummyPortfolio:
        def __init__(self, *a, **k):
            pass

        def stats(self):
            return {"Total Return [%]": 0, "Max Drawdown [%]": 0}

    monkeypatch.setattr(
        vbt,
        "Portfolio",
        types.SimpleNamespace(from_signals=lambda *a, **k: DummyPortfolio()),
        raising=False,
    )
    monkeypatch.setattr(walk_forward, "ensure_real_vectorbt", lambda *a, **k: None)

    monkeypatch.setattr(
        walk_forward.config, "FITNESS_WEIGHTS", {"min_trades": 0}, raising=False
    )
    monkeypatch.setitem(walk_forward.config.MULTI_ASSET, "enabled", False)

    walk_forward.run_walk_forward_validation(tmp_path)

    assert captured["parallel_processing"][1] == os.cpu_count()


def test_walk_forward_returns_summary(monkeypatch, tmp_path):
    """run_walk_forward_validation should return aggregate metrics"""
    import types

    import pandas as pd
    import vectorbt as vbt

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

    monkeypatch.setattr(
        walk_forward.data_loader, "get_data", lambda *a, **k: (df, "cache")
    )
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
    monkeypatch.setattr(
        walk_forward, "parse_genes_from_config", lambda *a, **k: ([], {}, [])
    )

    class DummyEvaluator:
        def __init__(self, *a, **k):
            self.last_exit_params = {}

        def __call__(self, *a, **k):
            return 1.0

    fitness_stub = types.SimpleNamespace(
        FitnessEvaluator=DummyEvaluator,
        get_fitness_evaluator=lambda *a, **k: DummyEvaluator(),
        print_floor_failures=lambda *a, **k: None,
    )
    monkeypatch.setitem(sys.modules, "fitness", fitness_stub)

    class DummyGA:
        def __init__(self, *args, **kwargs):
            pass

        def run(self):
            return None

        def best_solution(self, **kwargs):
            return [], 1.0, None

    monkeypatch.setattr(walk_forward.pygad, "GA", DummyGA)

    engine_stub = types.SimpleNamespace(
        process_strategy_rules=lambda *a, **k: pd.Series([True, False], index=df.index)
    )
    monkeypatch.setitem(sys.modules, "strategy_engine", engine_stub)
    monkeypatch.setattr(walk_forward, "inject_genes_into_rules", lambda *a, **k: {})

    class DummyPortfolio:
        def stats(self):
            return {
                "Total Return [%]": 1.0,
                "Max Drawdown [%]": 0.0,
                "Sharpe Ratio": 1.0,
                "Sortino Ratio": 1.0,
                "Win Rate [%]": 50.0,
            }

    monkeypatch.setattr(
        vbt,
        "Portfolio",
        types.SimpleNamespace(from_signals=lambda *a, **k: DummyPortfolio()),
        raising=False,
    )
    monkeypatch.setattr(walk_forward, "ensure_real_vectorbt", lambda *a, **k: None)

    monkeypatch.setattr(
        walk_forward.config, "FITNESS_WEIGHTS", {"min_trades": 0}, raising=False
    )
    monkeypatch.setitem(walk_forward.config.MULTI_ASSET, "enabled", False)

    summary = walk_forward.run_walk_forward_validation(tmp_path)

    assert isinstance(summary, dict)
    for key in ["average_return", "total_compounded_return", "folds"]:
        assert key in summary
    results_path = tmp_path / "walk_forward" / "walk_forward_results.csv"
    assert results_path.exists()


def test_round_floats_helper():
    obj = {"a": 0.05500000000000001, "b": [0.3333333], "c": {"d": 1.234567}}
    rounded = walk_forward._round_floats(obj, ndigits=3)
    assert rounded == {"a": 0.055, "b": [0.333], "c": {"d": 1.235}}


def test_file_sha256_handles_present_and_missing(tmp_path):
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"walk-forward")

    digest = walk_forward._file_sha256(path)
    assert digest == walk_forward._file_sha256(path), "hash should be deterministic"

    missing = walk_forward._file_sha256(tmp_path / "missing.bin")
    assert missing is None


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
    pool, status = walk_forward._update_champion_pool(
        pool, [0], 0.1, gene_space, settings
    )
    assert pool == [] and status == "Discarded"
    assert "discarded" in capsys.readouterr().out.lower()

    # Keep case
    pool, status = walk_forward._update_champion_pool(
        pool, [0], 0.7, gene_space, settings
    )
    assert len(pool) == 1 and status == "Viable"
    assert "kept" in capsys.readouterr().out.lower()

    # Clone case
    pool, status = walk_forward._update_champion_pool(
        pool, [1], 1.2, gene_space, settings
    )
    assert len(pool) == 1 + 1 + settings["num_clones"] and status == "Elite"
    out = capsys.readouterr().out.lower()
    assert "cloning" in out


def test_update_champion_pool_mutation_preserves_types():
    settings = {
        "survival_threshold": 0.0,
        "cloning_threshold": 1.0,
        "num_clones": 4,
        "clone_mutation_rate": 1.0,
    }
    gene_space = [
        {"low": 0, "high": 10, "step": 1},
        [False, True],
        ["hold", "breakeven", "follow_tp"],
    ]
    best = [2, False, "hold"]

    np.random.seed(0)
    pool, status = walk_forward._update_champion_pool(
        [], best, 2.5, gene_space, settings
    )

    assert status == "Elite"
    assert len(pool) == 1 + settings["num_clones"]
    # First entry is the original champion
    assert pool[0] == best

    clones = pool[1:]
    # Mutation should keep integer genes as ints and boolean genes as bools
    assert {type(clone[0]) for clone in clones} == {int}
    assert {type(clone[1]) for clone in clones} == {bool}
    assert all(isinstance(clone[2], str) for clone in clones)
    # At least one clone should differ on discrete option genes
    assert any(clone[1] is not best[1] for clone in clones)
    assert any(clone[2] != best[2] for clone in clones)


def test_update_champion_pool_mutation_handles_float_like():
    settings = {
        "survival_threshold": 0.0,
        "cloning_threshold": 0.5,
        "num_clones": 3,
        "clone_mutation_rate": 1.0,
    }
    gene_space = [{"low": 0.5, "high": 2.5}]
    best = [np.float64(1.5)]

    np.random.seed(1)
    pool, status = walk_forward._update_champion_pool(
        [], best, 0.75, gene_space, settings
    )

    assert status == "Elite"
    clones = pool[1:]
    assert clones, "expected clones to be generated"
    assert all(isinstance(clone[0], np.floating) for clone in clones)
    assert any(not np.isclose(clone[0], best[0]) for clone in clones)


def test_normalize_trade_floor_penalty_shapes():
    mapping_penalty = {"reason": "soft_penalty", "scale": 0.8}
    assert (
        walk_forward._normalize_trade_floor_penalty(mapping_penalty) == mapping_penalty
    )

    assert walk_forward._normalize_trade_floor_penalty("lack_of_trades") == {
        "reason": "lack_of_trades"
    }
    assert walk_forward._normalize_trade_floor_penalty(None) == {}


def test_write_run_metadata_collects_artifacts(tmp_path, monkeypatch):
    run_dir = tmp_path
    inside = run_dir / "chart.png"
    inside.write_text("artifact")
    outside = tmp_path.parent / "global.txt"
    outside.write_text("external")

    captured = {}

    monkeypatch.setattr(
        walk_forward,
        "_get_cache_hashes",
        lambda start, end: {"cache.parquet": "abc123"},
    )

    def fake_merge(path, data):
        captured["path"] = path
        captured["data"] = data

    monkeypatch.setattr(walk_forward, "merge_run_metadata", fake_merge)

    fake_vbt = types.SimpleNamespace(
        __version__="1.2.3", __file__=str(run_dir / "vbt.py")
    )
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    walk_forward._write_run_metadata(
        run_dir,
        start,
        "2025-01-01",
        "2025-02-01",
        [inside, outside, run_dir / "missing.csv"],
        fake_vbt,
    )

    assert captured["path"].name == "run_metadata.json"
    metadata = captured["data"]
    assert metadata["artifacts"][0] == inside.name
    assert any(outside.resolve().as_posix() in item for item in metadata["artifacts"])
    assert metadata["library_versions"]["vectorbt"]["version"] == "1.2.3"
    assert metadata["cache_files"] == {"cache.parquet": "abc123"}


def _run_walk_forward_with_penalty(monkeypatch, penalty, mode=None, run_dir=None):
    import types

    import pandas as pd

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

    monkeypatch.setattr(
        walk_forward.data_loader, "get_group_data", lambda *a, **k: {"AAA": df}
    )
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
    monkeypatch.setattr(
        walk_forward, "parse_genes_from_config", lambda *a, **k: ([], {}, [])
    )

    class DummyGA:
        def __init__(self, *args, **kwargs):
            pass

        def run(self):
            return None

        def best_solution(self, **kwargs):
            return [], 1.0, None

    monkeypatch.setattr(walk_forward.pygad, "GA", DummyGA)

    class DummyEvaluator:
        def __init__(self, *a, **k):
            self.last_details = {}

        def __call__(self, *a, **k):
            self.last_details = {
                "penalties": {"trade_floor": penalty},
                "min_total_trades": 10,
                "total_trades": 0,
                "per_asset": {},
                "mu": 0.0,
                "lambda_sigma": 0.0,
            }
            return -999.0

    fitness_stub = types.SimpleNamespace(
        MultiAssetFitnessEvaluator=lambda *a, **k: DummyEvaluator(),
        print_floor_failures=lambda *a, **k: None,
    )
    monkeypatch.setitem(sys.modules, "fitness", fitness_stub)
    monkeypatch.setattr(walk_forward, "inject_genes_into_rules", lambda *a, **k: {})
    monkeypatch.setattr(walk_forward, "ensure_real_vectorbt", lambda *a, **k: None)
    monkeypatch.setattr(walk_forward, "_write_run_metadata", lambda *a, **k: None)
    monkeypatch.setattr(
        walk_forward.trade_floor,
        "scale_floor",
        lambda rate, s, e, td=252: (
            0,
            {
                "base_floor": rate,
                "window_days": 0,
                "trading_days_per_year": td,
                "years": 0,
                "raw": 0,
                "ceil": 0,
            },
        ),
    )
    if mode:
        monkeypatch.setitem(walk_forward.config.MULTI_ASSET, "soft_penalty_mode", mode)

    run_dir = Path(run_dir or Path("."))
    return walk_forward.run_walk_forward_validation(run_dir)


@pytest.mark.parametrize(
    "penalty, mode, expected",
    [
        ("no_assets", None, ["reason=no_assets"]),
        (
            {"penalty": -0.25, "reason": "low_trades"},
            "additive",
            ["additive", "reason=low_trades"],
        ),
        (None, None, []),
    ],
)
def test_walk_forward_trade_floor_penalties(
    monkeypatch, capsys, tmp_path, penalty, mode, expected
):
    summary = _run_walk_forward_with_penalty(monkeypatch, penalty, mode, tmp_path)
    assert isinstance(summary, dict)
    out = capsys.readouterr().out
    for token in expected:
        assert token in out
    if not expected:
        assert "reason=" not in out
