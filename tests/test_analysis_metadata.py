import importlib.machinery
import json
import sys
import types
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

vbt_stub = types.ModuleType("vectorbt")
vbt_stub.__spec__ = importlib.machinery.ModuleSpec("vectorbt", loader=None)
sys.modules.setdefault("vectorbt", vbt_stub)

import analysis  # noqa: E402
import config  # noqa: E402
import data_loader  # noqa: E402
import fitness  # noqa: E402

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
