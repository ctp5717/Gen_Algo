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


def test_get_data_uses_cache(monkeypatch):
    df = pd.DataFrame({'Close': [1, 2]}, index=pd.date_range('2020-01-01', periods=2))

    # Force cache path to exist and return our dataframe
    monkeypatch.setattr(data_loader.os.path, 'exists', lambda path: True)
    monkeypatch.setattr(data_loader.pd, 'read_csv', lambda *a, **k: df)
    monkeypatch.setattr(data_loader, '_get_binance_data', lambda *a, **k: None)
    monkeypatch.setattr(data_loader.yf, 'download', lambda *a, **k: None)
    monkeypatch.setattr(data_loader.config, 'DATA_SOURCE', 'yfinance')

    result = data_loader.get_data('TEST', '2020-01-01', '2020-01-02')

    pd.testing.assert_frame_equal(result, df)


def test_get_data_handles_multiple_tickers(monkeypatch):
    df = pd.DataFrame(
        {
            'Open': [1],
            'High': [1],
            'Low': [1],
            'Close': [1],
            'Volume': [1],
        },
        index=pd.date_range('2020-01-01', periods=1),
    )

    monkeypatch.setattr(data_loader.os.path, 'exists', lambda path: False)
    monkeypatch.setattr(data_loader.os, 'makedirs', lambda *a, **k: None)
    monkeypatch.setattr(pd.DataFrame, 'to_csv', lambda *a, **k: None, raising=False)
    monkeypatch.setattr(data_loader.config, 'DATA_SOURCE', 'yfinance', raising=False)

    def fake_download(ticker, *a, **k):
        return df

    monkeypatch.setattr(data_loader.yf, 'download', fake_download)

    result = data_loader.get_data(['A', 'B'], '2020-01-01', '2020-01-02')
    assert isinstance(result.columns, pd.MultiIndex)
    assert set(result.columns.get_level_values(0)) == {'A', 'B'}


def test_get_data_handles_three_tickers(monkeypatch):
    df = pd.DataFrame(
        {
            'Open': [1],
            'High': [1],
            'Low': [1],
            'Close': [1],
            'Volume': [1],
        },
        index=pd.date_range('2020-01-01', periods=1),
    )

    monkeypatch.setattr(data_loader.os.path, 'exists', lambda path: False)
    monkeypatch.setattr(data_loader.os, 'makedirs', lambda *a, **k: None)
    monkeypatch.setattr(pd.DataFrame, 'to_csv', lambda *a, **k: None, raising=False)
    monkeypatch.setattr(data_loader.config, 'DATA_SOURCE', 'yfinance', raising=False)

    def fake_download(ticker, *a, **k):
        return df

    monkeypatch.setattr(data_loader.yf, 'download', fake_download)

    result = data_loader.get_data(['A', 'B', 'C'], '2020-01-01', '2020-01-02')
    assert isinstance(result.columns, pd.MultiIndex)
    assert set(result.columns.get_level_values(0)) == {'A', 'B', 'C'}


def test_get_data_handles_missing_asset(monkeypatch):
    df = pd.DataFrame(
        {
            'Open': [1],
            'High': [1],
            'Low': [1],
            'Close': [1],
            'Volume': [1],
        },
        index=pd.date_range('2020-01-01', periods=1),
    )

    monkeypatch.setattr(data_loader.os.path, 'exists', lambda path: False)
    monkeypatch.setattr(data_loader.os, 'makedirs', lambda *a, **k: None)
    monkeypatch.setattr(pd.DataFrame, 'to_csv', lambda *a, **k: None, raising=False)
    monkeypatch.setattr(data_loader.config, 'DATA_SOURCE', 'yfinance', raising=False)

    def fake_download(ticker, *a, **k):
        return df if ticker == 'A' else pd.DataFrame()

    monkeypatch.setattr(data_loader.yf, 'download', fake_download)

    result = data_loader.get_data(['A', 'B'], '2020-01-01', '2020-01-02')
    assert 'A' in result.columns.get_level_values(0)
    assert 'B' not in result.columns.get_level_values(0)
