import logging
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

# Ensure repository root is on the import path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Stub heavy optional dependencies before importing modules
sys.modules.setdefault("pandas_ta", types.ModuleType("pandas_ta"))
sys.modules.setdefault("vectorbt", types.ModuleType("vectorbt"))

# Stub binance client to avoid import errors
binance_mod = types.ModuleType("binance")
client_mod = types.ModuleType("binance.client")
client_mod.Client = object
binance_mod.client = client_mod
sys.modules.setdefault("binance", binance_mod)
sys.modules.setdefault("binance.client", client_mod)

import data_loader  # noqa: E402


def test_get_data_uses_cache(monkeypatch):
    df = pd.DataFrame(
        {"Close": [1, 2], "Volume": [1, 1]},
        index=pd.date_range("2020-01-01", periods=2),
    )

    # Force cache path to exist and return our dataframe
    monkeypatch.setattr(data_loader.os.path, "exists", lambda path: True)
    monkeypatch.setattr(data_loader.pd, "read_parquet", lambda *a, **k: df)
    monkeypatch.setattr(data_loader, "_get_binance_data", lambda *a, **k: None)
    monkeypatch.setattr(data_loader.yf, "download", lambda *a, **k: None)
    monkeypatch.setattr(data_loader.config, "DATA_SOURCE", "yfinance")

    result, src = data_loader.get_data("TEST", "2020-01-01", "2020-01-02")

    pd.testing.assert_frame_equal(result, df)
    assert src == "cache"


def test_get_data_warns_when_volume_missing(monkeypatch, caplog):
    df = pd.DataFrame({"Close": [1, 2]}, index=pd.date_range("2020-01-01", periods=2))
    monkeypatch.setattr(data_loader.os.path, "exists", lambda path: True)
    monkeypatch.setattr(data_loader.pd, "read_parquet", lambda *a, **k: df)
    with caplog.at_level(logging.WARNING):
        result, src = data_loader.get_data(
            "TEST", "2020-01-01", "2020-01-02", verbose=True
        )
    pd.testing.assert_frame_equal(result, df)
    assert src == "cache"
    assert "Volume column missing" in caplog.text


def test_get_data_raises_when_volume_invalid(monkeypatch):
    df = pd.DataFrame(
        {"Close": [1, 2], "Volume": [1, -1]},
        index=pd.date_range("2020-01-01", periods=2),
    )
    monkeypatch.setattr(data_loader.os.path, "exists", lambda path: True)
    monkeypatch.setattr(data_loader.pd, "read_parquet", lambda *a, **k: df)
    with pytest.raises(KeyError):
        data_loader.get_data("TEST", "2020-01-01", "2020-01-02", verbose=False)


def test_warn_missing_volume_helper_emits_once(monkeypatch, caplog):
    engine_stub = types.ModuleType("strategy_engine")
    engine_stub.VOLUME_INDICATORS = {"mock"}
    monkeypatch.setitem(sys.modules, "strategy_engine", engine_stub)

    df = pd.DataFrame({"Close": [1, 2]}, index=pd.date_range("2020-01-01", periods=2))
    logger = logging.getLogger("data_loader_test.warn_helper")

    with caplog.at_level(logging.WARNING):
        data_loader._warn_missing_volume(df, True, logger)
        data_loader._warn_missing_volume(df, True, logger)

    assert caplog.text.count("Volume column missing") == 1


def test_load_legacy_cache_uses_warning_helper(monkeypatch):
    df = pd.DataFrame({"Close": [1]}, index=pd.date_range("2020-01-01", periods=1))
    monkeypatch.setattr(data_loader.pd, "read_csv", lambda *a, **k: df)

    calls: list[tuple[pd.DataFrame, bool]] = []

    def fake_warn(frame, verbose, _logger):
        calls.append((frame, verbose))

    monkeypatch.setattr(data_loader, "_warn_missing_volume", fake_warn)

    logger = logging.getLogger("data_loader_test.load_legacy")
    result = data_loader._load_legacy_cache(
        "legacy.csv",
        ticker="TEST",
        cache_filepath="cache.parquet",
        cache_filename="cache.parquet",
        verbose=True,
        logger=logger,
    )

    assert calls == [(df, True)]
    assert result is df


def test_get_data_calls_warning_helper(monkeypatch):
    df = pd.DataFrame({"Close": [1]}, index=pd.date_range("2020-01-01", periods=1))
    monkeypatch.setattr(data_loader.os.path, "exists", lambda path: True)
    monkeypatch.setattr(data_loader.pd, "read_parquet", lambda *a, **k: df)
    monkeypatch.setattr(data_loader.config, "DATA_SOURCE", "yfinance")

    calls: list[tuple[pd.DataFrame, bool]] = []

    def fake_warn(frame, verbose, _logger):
        calls.append((frame, verbose))

    monkeypatch.setattr(data_loader, "_warn_missing_volume", fake_warn)

    result, src = data_loader.get_data("TEST", "2020-01-01", "2020-01-02")

    pd.testing.assert_frame_equal(result, df)
    assert src == "cache"
    assert calls == [(df, True)]


def test_get_data_builds_paths_with_helper(monkeypatch):
    df = pd.DataFrame(
        {
            "Open": [1],
            "High": [1],
            "Low": [1],
            "Close": [1],
            "Volume": [1],
        },
        index=pd.date_range("2024-01-01", periods=1),
    )

    called: list[tuple[str, str, str, str, str | None, bool]] = []

    def fake_build_cache_stem(
        ticker: str,
        start: str,
        end: str,
        interval: str,
        *,
        source: str | None = None,
        normalize: bool = True,
    ) -> str:
        called.append((ticker, start, end, interval, source, normalize))
        return "custom-stem"

    seen_paths: list[str] = []

    def fake_exists(path: str) -> bool:
        seen_paths.append(path)
        return False

    monkeypatch.setattr(data_loader, "build_cache_stem", fake_build_cache_stem)
    monkeypatch.setattr(data_loader.os.path, "exists", fake_exists)
    monkeypatch.setattr(data_loader.os, "makedirs", lambda *a, **k: None)
    monkeypatch.setattr(data_loader.pd.DataFrame, "to_parquet", lambda self, path: None)
    monkeypatch.setattr(data_loader, "_warn_missing_volume", lambda *a, **k: None)
    monkeypatch.setattr(data_loader.config, "DATA_SOURCE", "yfinance")
    monkeypatch.setattr(data_loader.yf, "download", lambda *a, **k: df.copy())

    result, src = data_loader.get_data(
        "ETH-USD", "2024-01-01", "2024-01-20", interval="1d", verbose=False
    )

    pd.testing.assert_frame_equal(result, df)
    assert src == "API"
    assert called == [("ETH-USD", "2024-01-01", "2024-01-20", "1d", "yfinance", False)]
    expected_cache = data_loader.os.path.join(
        data_loader.CACHE_DIR, "custom-stem" + data_loader.CACHE_EXTENSION
    )
    expected_legacy = data_loader.os.path.join(
        data_loader.CACHE_DIR, "custom-stem" + data_loader.LEGACY_CACHE_EXTENSION
    )
    assert expected_cache in seen_paths
    assert expected_legacy in seen_paths


def test_get_group_data_calls_warning_helper(monkeypatch):
    df = pd.DataFrame({"Close": [1]}, index=pd.date_range("2020-01-01", periods=1))

    def fake_get_data(*args, **kwargs):
        return df.copy(), "cache"

    calls: list[tuple[pd.DataFrame, bool]] = []

    def fake_warn(frame, verbose, _logger):
        calls.append((frame, verbose))

    monkeypatch.setattr(data_loader, "get_data", fake_get_data)
    monkeypatch.setattr(data_loader, "_warn_missing_volume", fake_warn)
    monkeypatch.setattr(data_loader.config, "DATA_LOADER_MAX_WORKERS", 1, raising=False)

    result = data_loader.get_group_data(
        [("Asset", "AAA")],
        start_date="2020-01-01",
        end_date="2020-01-02",
        interval="1d",
        verbose=True,
        logger=logging.getLogger("data_loader_test.group"),
    )

    assert list(result.keys()) == ["AAA"]
    assert len(calls) == 1
    pd.testing.assert_frame_equal(calls[0][0], df)
    assert calls[0][1] is True


def test_get_group_data_summary_respects_verbose_false(monkeypatch, caplog):
    df = pd.DataFrame(
        {"Close": [1], "Volume": [1]}, index=pd.date_range("2020-01-01", periods=1)
    )

    def fake_get_data(*, ticker, start_date, end_date, interval, verbose, logger):
        assert verbose is False
        return df.copy(), "cache"

    monkeypatch.setattr(data_loader, "get_data", fake_get_data)
    monkeypatch.setattr(data_loader, "_warn_missing_volume", lambda *a, **k: None)
    monkeypatch.setattr(data_loader.config, "DATA_LOADER_MAX_WORKERS", 1, raising=False)

    logger = logging.getLogger("data_loader_test.summary_false")
    caplog.clear()
    with caplog.at_level(logging.INFO, logger=logger.name):
        result = data_loader.get_group_data(
            [("Asset", "AAA")],
            start_date="2020-01-01",
            end_date="2020-01-02",
            interval="1d",
            verbose=False,
            logger=logger,
        )

    assert list(result.keys()) == ["AAA"]
    info_records = [
        record for record in caplog.records if record.levelno == logging.INFO
    ]
    assert not info_records


def test_get_group_data_summary_logs_when_verbose_true(monkeypatch, caplog):
    df = pd.DataFrame(
        {"Close": [1], "Volume": [1]}, index=pd.date_range("2020-01-01", periods=1)
    )

    def fake_get_data(*, ticker, start_date, end_date, interval, verbose, logger):
        assert verbose is True
        return df.copy(), "cache"

    monkeypatch.setattr(data_loader, "get_data", fake_get_data)
    monkeypatch.setattr(data_loader, "_warn_missing_volume", lambda *a, **k: None)
    monkeypatch.setattr(data_loader.config, "DATA_LOADER_MAX_WORKERS", 1, raising=False)

    logger = logging.getLogger("data_loader_test.summary_true")
    caplog.clear()
    with caplog.at_level(logging.INFO, logger=logger.name):
        result = data_loader.get_group_data(
            [("Asset", "AAA")],
            start_date="2020-01-01",
            end_date="2020-01-02",
            interval="1d",
            verbose=True,
            logger=logger,
        )

    assert list(result.keys()) == ["AAA"]
    info_records = [
        record for record in caplog.records if record.levelno == logging.INFO
    ]
    assert len(info_records) == 1
    assert "Loaded 1 assets" in info_records[0].getMessage()
