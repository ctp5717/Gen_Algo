import sys
import types
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Stub heavy optional dependencies
sys.modules.setdefault('pandas_ta', types.ModuleType('pandas_ta'))
sys.modules.setdefault('vectorbt', types.ModuleType('vectorbt'))

import indicator_library  # noqa: E402


def _multi_ohlc():
    base = pd.DataFrame(
        {
            "Open": [1, 2, 3],
            "High": [1, 2, 3],
            "Low": [1, 2, 3],
            "Close": [1, 2, 3],
            "Volume": [1, 1, 1],
        },
        index=pd.date_range("2020-01-01", periods=3),
    )
    return pd.concat({"A": base, "B": base}, axis=1)


def test_calculate_ema_multiindex():
    df = _multi_ohlc()
    result = indicator_library.calculate_ema(df, period=2)
    assert isinstance(result, pd.DataFrame)
    assert isinstance(result.columns, pd.MultiIndex)


def test_calculate_atr_multiindex():
    df = _multi_ohlc()
    result = indicator_library.calculate_atr(df, period=2)
    assert isinstance(result, pd.DataFrame)
    assert isinstance(result.columns, pd.MultiIndex)


def test_calculate_rsi_multiindex():
    df = _multi_ohlc()
    result = indicator_library.calculate_rsi(df, period=2)
    assert isinstance(result, pd.DataFrame)
    assert isinstance(result.columns, pd.MultiIndex)


def test_calculate_macd_multiindex():
    df = _multi_ohlc()
    result = indicator_library.calculate_macd(df, fast=2, slow=3, signal=1)
    assert isinstance(result, pd.DataFrame)
    assert isinstance(result.columns, pd.MultiIndex)


def test_calculate_bbands_multiindex():
    df = _multi_ohlc()
    result = indicator_library.calculate_bbands(df, period=2, std_dev=2)
    assert isinstance(result, pd.DataFrame)
    assert isinstance(result.columns, pd.MultiIndex)
