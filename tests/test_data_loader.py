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
        {"Close": [1, 2], "Volume": [1, 1]}, index=pd.date_range("2020-01-01", periods=2)
    )

    # Force cache path to exist and return our dataframe
    monkeypatch.setattr(data_loader.os.path, "exists", lambda path: True)
    monkeypatch.setattr(data_loader.pd, "read_csv", lambda *a, **k: df)
    monkeypatch.setattr(data_loader, "_get_binance_data", lambda *a, **k: None)
    monkeypatch.setattr(data_loader.yf, "download", lambda *a, **k: None)
    monkeypatch.setattr(data_loader.config, "DATA_SOURCE", "yfinance")

    result, src = data_loader.get_data("TEST", "2020-01-01", "2020-01-02")

    pd.testing.assert_frame_equal(result, df)
    assert src == "cache"


def test_get_data_warns_when_volume_missing(monkeypatch, capsys):
    df = pd.DataFrame({"Close": [1, 2]}, index=pd.date_range("2020-01-01", periods=2))
    monkeypatch.setattr(data_loader.os.path, "exists", lambda path: True)
    monkeypatch.setattr(data_loader.pd, "read_csv", lambda *a, **k: df)
    result, src = data_loader.get_data("TEST", "2020-01-01", "2020-01-02", verbose=True)
    pd.testing.assert_frame_equal(result, df)
    assert src == "cache"
    out = capsys.readouterr().out
    assert "Volume column missing" in out


def test_get_data_raises_when_volume_invalid(monkeypatch):
    df = pd.DataFrame(
        {"Close": [1, 2], "Volume": [1, -1]}, index=pd.date_range("2020-01-01", periods=2)
    )
    monkeypatch.setattr(data_loader.os.path, "exists", lambda path: True)
    monkeypatch.setattr(data_loader.pd, "read_csv", lambda *a, **k: df)
    with pytest.raises(KeyError):
        data_loader.get_data("TEST", "2020-01-01", "2020-01-02", verbose=False)
