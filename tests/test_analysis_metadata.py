import importlib.machinery
import json
import sys
import types
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

vbt_stub = types.ModuleType("vectorbt")
vbt_stub.__spec__ = importlib.machinery.ModuleSpec("vectorbt", loader=None)
sys.modules.setdefault("vectorbt", vbt_stub)

import analysis  # noqa: E402
import config  # noqa: E402
import data_loader  # noqa: E402
import fitness  # noqa: E402
from schemas import load_wf_summary  # noqa: E402
from exits_nb import ExitReason  # noqa: E402

config.initialize_config()


def test_cache_hashes_use_cache_helper(monkeypatch):
    monkeypatch.setattr(config, "DATA_SOURCE", "yfinance", raising=False)
    monkeypatch.setattr(config, "TIMEFRAME", "1d", raising=False)
    monkeypatch.setattr(config, "TICKER", "SOL-USD", raising=False)
    monkeypatch.setattr(
        config,
        "TRAINING_PERIOD",
        {"start": "2024-01-01", "end": "2024-01-10"},
        raising=False,
    )
    monkeypatch.setattr(
        config,
        "VALIDATION_PERIOD",
        {"start": "2024-01-11", "end": "2024-01-20"},
        raising=False,
    )
    monkeypatch.setattr(config, "ENABLE_WALK_FORWARD_VALIDATION", False, raising=False)
    monkeypatch.setattr(config, "WALK_FORWARD_SETTINGS", {}, raising=False)
    monkeypatch.setitem(config.MULTI_ASSET, "enabled", False)

    hashes = analysis._get_cache_hashes()

    expected_stem = data_loader.build_cache_stem(
        "SOL-USD", "2024-01-01", "2024-01-20", "1d", source="yfinance"
    )
    expected_files = {
        f"{expected_stem}{data_loader.CACHE_EXTENSION}",
        f"{expected_stem}{data_loader.LEGACY_CACHE_EXTENSION}",
    }
    assert set(hashes) == expected_files


def test_write_run_metadata_extra(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(analysis, "_get_cache_hashes", lambda: {})
    analysis.set_run_dir(tmp_path)

    class _VBT:
        __version__ = "0.0.0"
        __file__ = __file__

    monkeypatch.setattr(analysis, "vbt", _VBT)
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    extra = {
        "combination_logic": "AND",
        "vote_threshold": None,
        "per_asset_signal_counts": {"A": {"rule": 3}},
    }
    (tmp_path / "foo.png").write_text("x")
    analysis._write_run_metadata(start, ["foo.png"], extra)
    with open("run_metadata.json") as fh:
        meta = json.load(fh)
    assert meta["combination_logic"] == "AND"
    assert meta["per_asset_signal_counts"]["A"]["rule"] == 3
    assert "foo.png" in meta["artifacts"]


def test_write_run_metadata_skips_missing_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(analysis, "_get_cache_hashes", lambda: {})
    analysis.set_run_dir(tmp_path)

    class _VBT:
        __version__ = "0.0.0"
        __file__ = __file__

    monkeypatch.setattr(analysis, "vbt", _VBT)
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)

    existing = tmp_path / "exists.png"
    existing.write_text("x")
    analysis._write_run_metadata(start, [str(existing), str(tmp_path / "missing.png")])
    with open("run_metadata.json") as fh:
        meta = json.load(fh)
    assert meta["artifacts"] == ["exists.png"]


def test_write_run_metadata_dedupes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(analysis, "_get_cache_hashes", lambda: {})
    analysis.set_run_dir(tmp_path)

    class _VBT:
        __version__ = "0.0.0"
        __file__ = __file__

    monkeypatch.setattr(analysis, "vbt", _VBT)
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    file1 = tmp_path / "a.png"
    file1.write_text("x")
    analysis._write_run_metadata(start, [str(file1), str(file1), str(file1)])
    with open("run_metadata.json") as fh:
        meta = json.load(fh)
    assert meta["artifacts"] == ["a.png"]


def test_champion_equity_in_metadata(tmp_path, monkeypatch):
    monkeypatch.setitem(config.MULTI_ASSET, "enabled", True)
    monkeypatch.setattr(
        config,
        "CHARTS",
        {"save_pngs": False, "show_distribution": False, "save_csv": False},
    )
    monkeypatch.setattr(config, "TIMEFRAME", "1d")  # for stable file names
    monkeypatch.setattr(
        config,
        "VALIDATION_PERIOD",
        {"start": "2024-01-01", "end": "2024-01-31"},
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(analysis, "_get_cache_hashes", lambda: {})
    analysis.set_run_dir(tmp_path)

    class _VBT:
        __version__ = "0.0.0"
        __file__ = __file__

    monkeypatch.setattr(analysis, "vbt", _VBT)

    group = {
        "A": pd.DataFrame({"Close": [1, 2]}),
        "B": pd.DataFrame({"Close": [1, 2]}),
    }

    class DummyEval:
        def __init__(self, group, rules, gene_map, settings):
            assert settings.get("collect_equity_curve") is True
            self.last_details = {
                "per_asset": {
                    "A": {"score": 1.0, "trades": 1, "equity_curve": pd.Series([1, 2])},
                    "B": {"score": 2.0, "trades": 1, "equity_curve": pd.Series([1, 2])},
                },
                "mu": 0.0,
                "sigma": 0.0,
                "lambda_sigma": 0.0,
                "total_trades": 2,
                "penalties": {"coverage": 0.0, "trade_floor": None},
                "assets_included": 2,
                "assets_traded": 2,
                "min_total_trades": 0,
            }

        def __call__(self, ga, sol, idx):
            return 0.5

    monkeypatch.setattr(fitness, "MultiAssetFitnessEvaluator", DummyEval)
    monkeypatch.setattr(analysis, "_plot_multi_asset_overview", lambda *a, **k: None)

    analysis._run_multi_asset_analysis([], {}, group, [])
    assert (tmp_path / "champion_equity.png").exists()
    meta = json.loads((tmp_path / "run_metadata.json").read_text())
    assert "champion_equity.png" in meta["artifacts"]
    champ_entry = [a for a in meta["artifacts"] if a.endswith("champion_equity.png")][0]
    assert not Path(champ_entry).is_absolute()


def test_write_run_metadata_external_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(analysis, "_get_cache_hashes", lambda: {})
    analysis.set_run_dir(tmp_path)

    class _VBT:
        __version__ = "0.0.0"
        __file__ = __file__

    monkeypatch.setattr(analysis, "vbt", _VBT)
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    external = tmp_path.parent / "external.png"
    external.write_text("x")
    analysis._write_run_metadata(start, [str(external)])
    meta = json.loads((tmp_path / "run_metadata.json").read_text())
    assert meta["artifacts"] == [str(external.resolve())]


def test_run_champion_analysis_records_exit_metadata(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(analysis, "_get_cache_hashes", lambda: {})
    analysis.set_run_dir(tmp_path)
    monkeypatch.setattr(config, "TIMEFRAME", "1d", raising=False)
    monkeypatch.setattr(config, "SELECTED_ASSET_NAME", "Test", raising=False)
    monkeypatch.setattr(config, "TICKER", "TEST", raising=False)
    monkeypatch.setattr(config, "MAX_HOLD_PERIOD", 5, raising=False)
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    validation = pd.DataFrame(
        {
            "Close": [100.0, 100.6, 100.0],
            "High": [100.0, 100.6, 100.0],
            "Low": [100.0, 100.0, 100.0],
        },
        index=index,
    )
    entries = pd.Series([True, False, False], index=index)
    monkeypatch.setattr(
        analysis.engine,
        "process_strategy_rules",
        lambda *args, **kwargs: (entries, {}),
    )

    last_from_signals: dict[str, object] = {}

    class DummyPortfolio:
        def __init__(self):
            self.trades = types.SimpleNamespace(count=lambda: 1)

        @classmethod
        def from_signals(cls, *args, **kwargs):
            last_from_signals.clear()
            last_from_signals.update(kwargs)
            return cls()

        def stats(self):
            return pd.Series(
                {
                    "Start": index[0],
                    "End": index[-1],
                    "Period": "3D",
                    "Total Return [%]": 5.0,
                    "Benchmark Return [%]": 1.0,
                    "Max Drawdown [%]": 1.0,
                    "Sortino Ratio": 1.0,
                    "Sharpe Ratio": 1.0,
                    "Profit Factor": 2.0,
                    "Win Rate [%]": 50.0,
                    "Total Trades": 1,
                    "Avg Winning Trade [%]": 1.0,
                    "Avg Losing Trade [%]": -1.0,
                }
            )

        def plot(self, *args, **kwargs):
            class _Fig:
                def show(self_inner):
                    return None

            return _Fig()

    monkeypatch.setattr(analysis.vbt, "Portfolio", DummyPortfolio)

    def fake_build_dynamic_exit_orders(**kwargs):
        fake_build_dynamic_exit_orders.called = True
        return kwargs["entries"], kwargs["exits_series"], kwargs["exit_size_series"]

    fake_build_dynamic_exit_orders.called = False
    monkeypatch.setattr(
        analysis.fitness,
        "build_dynamic_exit_orders",
        fake_build_dynamic_exit_orders,
    )

    def fake_savefig(path, *args, **kwargs):
        Path(path).write_text("fig")

    monkeypatch.setattr(analysis.plt, "savefig", fake_savefig)
    monkeypatch.setattr(analysis.plt, "close", lambda *a, **k: None)
    monkeypatch.setattr(analysis, "ensure_real_vectorbt", lambda *a, **k: None)

    analysis.run_champion_analysis([], {}, validation)
    meta_path = tmp_path / "run_metadata.json"
    assert meta_path.exists()
    metadata = json.loads(meta_path.read_text())
    exit_meta = metadata.get("exit", {})
    assert "params" in exit_meta
    params_snapshot = exit_meta["params"]
    assert params_snapshot.get("sl_break_even_mode") in {
        "none",
        "breakeven",
        "follow_tp",
    }
    assert params_snapshot.get("timeframe") == "1d"
    cap_expected = config.get_tp_cap_for_timeframe("1d")
    assert params_snapshot.get("tp_cap") == pytest.approx(cap_expected)
    breakdown = exit_meta.get("reason_breakdown", {})
    assert breakdown.get(ExitReason.TP1.name, {}).get("fraction")
    metrics = exit_meta.get("metrics", {})
    assert "avg_tp_level_reached" in metrics
    assert "avg_sl_timeout_bars" in metrics
    assert "breakeven_touch_rate" in metrics
    kpi = exit_meta.get("kpi_strip", {})
    assert "breakeven_touch_pct" in kpi
    csv_path = tmp_path / "exit_reason_breakdown.csv"
    assert csv_path.exists()
    assert any(str(csv_path.name) in art for art in metadata.get("artifacts", []))
    kpi_path = tmp_path / "exit_kpi_strip.csv"
    assert kpi_path.exists()
    assert any(kpi_path.name in art for art in metadata.get("artifacts", []))
    assert fake_build_dynamic_exit_orders.called is True
    assert last_from_signals.get("accumulate") is getattr(
        config, "DYNAMIC_EXIT_ACCUMULATE", False
    )
    assert last_from_signals.get("size_type") == "amount"


def test_load_wf_summary_preserves_resolved_params(tmp_path):
    wf_dir = tmp_path / "walk_forward"
    wf_dir.mkdir()
    payload = {
        "metadata": {
            "schema_version": "1.0",
            "num_folds": 1,
            "asset_universe": ["TEST"],
        },
        "folds": [
            {
                "Window": 1,
                "Fitness": 1.23,
                "Params": "{'tp_pct_1': 0.05}",
                "Resolved Params": "{'tp_pct_1': 0.05, 'tp_cap': 0.5}",
            }
        ],
    }
    summary_path = wf_dir / "walk_forward_summary.json"
    summary_path.write_text(json.dumps(payload))

    summary = load_wf_summary(summary_path)
    assert summary.folds[0].params == {"tp_pct_1": 0.05}
    assert summary.folds[0].resolved_params == {
        "tp_pct_1": 0.05,
        "tp_cap": 0.5,
    }
