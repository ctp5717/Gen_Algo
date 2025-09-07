import sys
import types
from pathlib import Path

import pandas as pd

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
    df = pd.DataFrame({"Close": [1, 2]}, index=pd.date_range("2020-01-01", periods=2))

    # Force cache path to exist and return our dataframe
    monkeypatch.setattr(data_loader.os.path, "exists", lambda path: True)
    monkeypatch.setattr(data_loader.pd, "read_parquet", lambda *a, **k: df)
    monkeypatch.setattr(data_loader, "_get_binance_data", lambda *a, **k: None)
    monkeypatch.setattr(data_loader.yf, "download", lambda *a, **k: None)
    monkeypatch.setattr(data_loader.config, "DATA_SOURCE", "yfinance")

    result, src = data_loader.get_data("TEST", "2020-01-01", "2020-01-02")

    pd.testing.assert_frame_equal(result, df, check_freq=False)
    assert src == "cache"


def test_get_data_falls_back_to_csv(monkeypatch, tmp_path):
    df = pd.DataFrame(
        {"Open": [1], "High": [1], "Low": [1], "Close": [1], "Volume": [1]},
        index=pd.date_range("2020-01-01", periods=1),
    )

    # Force download path and use yfinance
    monkeypatch.setattr(data_loader.config, "DATA_SOURCE", "yfinance")
    monkeypatch.setattr(data_loader.yf, "download", lambda *a, **k: df)
    monkeypatch.setattr(data_loader, "CACHE_DIR", tmp_path)

    # Simulate missing parquet engine
    monkeypatch.setattr(
        pd.DataFrame,
        "to_parquet",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("no engine")),
    )

    saved = {}
    orig_to_csv = pd.DataFrame.to_csv

    def track_to_csv(self, path, index=True):
        saved["path"] = path
        return orig_to_csv(self, path, index=index)

    monkeypatch.setattr(pd.DataFrame, "to_csv", track_to_csv)

    # First call should download and save as CSV
    result, src = data_loader.get_data("TEST", "2020-01-01", "2020-01-02")
    pd.testing.assert_frame_equal(result, df, check_freq=False)
    assert src == "API"
    assert saved["path"].endswith(".csv")

    # Second call should load from the CSV cache
    result2, src2 = data_loader.get_data("TEST", "2020-01-01", "2020-01-02")
    pd.testing.assert_frame_equal(result2, df, check_freq=False)
    assert src2 == "cache"
