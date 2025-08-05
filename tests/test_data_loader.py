import sys
import types
from pathlib import Path
import pandas as pd

# Ensure repository root is on the import path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Stub heavy optional dependencies before importing modules
sys.modules.setdefault('pandas_ta', types.ModuleType('pandas_ta'))
sys.modules.setdefault('vectorbt', types.ModuleType('vectorbt'))

# Stub binance client to avoid import errors
binance_mod = types.ModuleType('binance')
client_mod = types.ModuleType('binance.client')
client_mod.Client = object
binance_mod.client = client_mod
sys.modules.setdefault('binance', binance_mod)
sys.modules.setdefault('binance.client', client_mod)

import data_loader  # noqa: E402


def test_get_data_uses_cache(monkeypatch, capsys):
    df = pd.DataFrame({'Close': [1, 2]}, index=pd.date_range('2020-01-01', periods=2))

    # Force cache path to exist and return our dataframe
    monkeypatch.setattr(data_loader.os.path, 'exists', lambda path: True)
    monkeypatch.setattr(data_loader.pd, 'read_csv', lambda *a, **k: df)
    monkeypatch.setattr(data_loader, '_get_binance_data', lambda *a, **k: None)
    monkeypatch.setattr(data_loader.yf, 'download', lambda *a, **k: None)
    monkeypatch.setattr(data_loader.config, 'DATA_SOURCE', 'yfinance')

    result = data_loader.get_data('TEST', '2020-01-01', '2020-01-02')
    out = capsys.readouterr().out
    assert "[1/1] TEST from cache" in out

    pd.testing.assert_frame_equal(result, df)


def test_get_data_handles_asset_list(monkeypatch, capsys):
    df_a = pd.DataFrame(
        {
            "Open": [1, 2],
            "High": [1, 2],
            "Low": [1, 2],
            "Close": [1, 2],
            "Volume": [1, 2],
        },
        index=pd.date_range("2020-01-01", periods=2),
    )
    df_b = df_a * 2

    def fake_dl(ticker, *a, **k):
        return df_a if ticker == "A" else df_b

    monkeypatch.setattr(data_loader.os.path, "exists", lambda *a, **k: False)
    monkeypatch.setattr(data_loader, "_get_binance_data", fake_dl)
    monkeypatch.setattr(data_loader.config, "DATA_SOURCE", "binance", raising=False)

    result = data_loader.get_data(["A", "B"], "2020-01-01", "2020-01-02")
    out = capsys.readouterr().out
    assert "Assets: A, B" in out
    assert "[1/2] A from Binance" in out
    assert "[2/2] B from Binance" in out

    assert isinstance(result.columns, pd.MultiIndex)
    assert ("A", "Close") in result.columns
    assert ("B", "Close") in result.columns


def test_get_data_logs_skipped_assets(monkeypatch, capsys):
    df = pd.DataFrame(
        {
            "Open": [1, 2],
            "High": [1, 2],
            "Low": [1, 2],
            "Close": [1, 2],
            "Volume": [1, 2],
        },
        index=pd.date_range("2020-01-01", periods=2),
    )

    def fake_loader(ticker, *a, **k):
        if ticker == "A":
            return df
        raise RuntimeError("fail")

    monkeypatch.setattr(data_loader, "_load_single_asset", fake_loader)

    result = data_loader.get_data(["A", "B"], "2020-01-01", "2020-01-02")
    out = capsys.readouterr().out
    assert "Skipped assets: B" in out
    assert ("A", "Close") in result.columns
    assert ("B", "Close") not in result.columns
