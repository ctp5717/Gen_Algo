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


def test_get_data_return_source(monkeypatch):
    df = pd.DataFrame({'Close': [1]}, index=pd.date_range('2020-01-01', periods=1))

    monkeypatch.setattr(data_loader.os.path, 'exists', lambda path: True)
    monkeypatch.setattr(data_loader.pd, 'read_csv', lambda *a, **k: df)
    monkeypatch.setattr(data_loader.config, 'DATA_SOURCE', 'yfinance')

    data, source = data_loader.get_data(
        'TEST', '2020-01-01', '2020-01-02', return_source=True
    )

    assert source == 'cache'
    pd.testing.assert_frame_equal(data, df)


def test_get_data_verbose_false_suppresses_output(monkeypatch, capsys):
    df = pd.DataFrame({'Close': [1]}, index=pd.date_range('2020-01-01', periods=1))

    monkeypatch.setattr(data_loader.os.path, 'exists', lambda path: False)
    monkeypatch.setattr(data_loader.os, 'makedirs', lambda *a, **k: None)
    monkeypatch.setattr(data_loader.pd.DataFrame, 'to_csv', lambda self, *a, **k: None)
    monkeypatch.setattr(data_loader.yf, 'download', lambda *a, **k: df)
    monkeypatch.setattr(data_loader.config, 'DATA_SOURCE', 'yfinance')

    data_loader.get_data(
        'TEST', '2020-01-01', '2020-01-02', verbose=False
    )

    out = capsys.readouterr().out
    assert out == ''


def test_get_group_data_summary(monkeypatch, capsys):
    data_loader._group_load_count = 0
    df = pd.DataFrame({'Close': [1]}, index=pd.date_range('2020-01-01', periods=1))

    def fake_get_data(*args, **kwargs):
        return df, 'cache'

    monkeypatch.setattr(data_loader, 'get_data', fake_get_data)

    assets = [('A', 'A'), ('B', 'B')]

    data_loader.get_group_data(assets, '2020-01-01', '2020-01-02', '1d')
    capsys.readouterr()  # clear output from first call
    data_loader.get_group_data(assets, '2020-01-01', '2020-01-02', '1d')
    out = capsys.readouterr().out

    assert 'Loading asset data for 2 assets (2020-01-01–2020-01-02) from cache' in out
    assert 'A:' not in out
    assert 'B:' not in out


def test_get_group_data_excluded_assets(monkeypatch, capsys):
    data_loader._group_load_count = 0

    df_full = pd.DataFrame({'Close': [1, 2]}, index=pd.date_range('2020-01-01', periods=2))
    df_short = pd.DataFrame({'Close': [1]}, index=pd.date_range('2020-01-01', periods=1))

    def fake_get_data(ticker, *args, **kwargs):
        return (df_full if ticker == 'A' else df_short), 'cache'

    monkeypatch.setattr(data_loader, 'get_data', fake_get_data)

    assets = [('A', 'A'), ('B', 'B')]

    aligned, excluded = data_loader.get_group_data(
        assets, '2020-01-01', '2020-01-02', '1d', coverage_threshold=0.75
    )
    out = capsys.readouterr().out

    assert 'Excluded: B (50%)' in out
    assert 'B' not in aligned
    assert excluded and excluded[0]['ticker'] == 'B'
    assert excluded[0]['reason'] == 'low_coverage'
