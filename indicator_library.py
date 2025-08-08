# indicator_library.py

"""
Indicator Library
=================

This script serves as the central "toolbox" for all technical indicators
used in the trading framework.
Each function in this file is responsible for calculating a specific indicator and returning its
values as a pandas Series or DataFrame.

Design Philosophy:
- Each function is self-contained and only responsible for one indicator.
- We use the high-performance 'pandas-ta' library to ensure calculations are fast and accurate.
- Functions are designed to be called by the 'strategy_engine.py' module.
- The function signatures are standardized for easy integration: they take a pandas DataFrame
  of price data (ohlc) and the indicator's specific parameters as arguments.

To Add a New Indicator:
1. Find the desired indicator in the 'pandas-ta' library documentation or implement your own.
2. Create a new function in this file (e.g., `calculate_macd(...)`).
3. Ensure it takes the ohlc DataFrame and necessary parameters as arguments.
4. Call the 'pandas-ta' function and return the resulting series/dataframe.
5. The new indicator can now be called from the `strategy_engine.py` by referencing
   the function name in the `config.py` file.
"""

import pandas as pd
import numpy as np

# -- Compatibility shim -------------------------------------------------------
# Some versions of pandas_ta expect ``numpy.NaN`` to be defined, but newer
# numpy releases expose only ``numpy.nan``. Importing pandas_ta without this
# attribute raises ``ImportError: cannot import name 'NaN'``. To keep the
# library working across numpy versions, ensure ``np.NaN`` exists before
# importing pandas_ta.
if not hasattr(np, "NaN"):
    np.NaN = np.nan

import pandas_ta as ta
try:  # Optional dependency used for portfolio‑wide indicator computation
    import vectorbt as vbt
except Exception:  # pragma: no cover - vectorbt may be unavailable in tests
    vbt = None


def _select_price(ohlc_data: pd.DataFrame, field: str) -> pd.DataFrame:
    """Return the requested price field from single or multi-asset data."""
    if isinstance(ohlc_data.columns, pd.MultiIndex):
        return ohlc_data.xs(field, axis=1, level=1)
    return ohlc_data[field]

def calculate_ema(ohlc_data: pd.DataFrame, period: int) -> pd.DataFrame:
    """Calculate the Exponential Moving Average (EMA)."""
    if period is None:
        raise ValueError("EMA 'period' parameter cannot be None.")

    close = _select_price(ohlc_data, 'Close')

    if vbt is not None and hasattr(vbt, 'EMA'):
        return vbt.EMA.run(close, window=period).ema

    return close.ewm(span=period, adjust=False).mean()

def calculate_atr(ohlc_data: pd.DataFrame, period: int) -> pd.DataFrame:
    """Calculate the Average True Range (ATR)."""
    if period is None:
        raise ValueError("ATR 'period' parameter cannot be None.")

    high = _select_price(ohlc_data, 'High')
    low = _select_price(ohlc_data, 'Low')
    close = _select_price(ohlc_data, 'Close')

    if vbt is not None and hasattr(vbt, 'ATR'):
        return vbt.ATR.run(high, low, close, window=period).atr

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1)
    if isinstance(tr.columns, pd.MultiIndex):
        tr = tr.groupby(level=0, axis=1).max()
    else:
        tr = tr.max(axis=1)
    return tr.rolling(period).mean()

def calculate_rsi(ohlc_data: pd.DataFrame, period: int) -> pd.DataFrame:
    """Calculate the Relative Strength Index (RSI)."""
    if period is None:
        raise ValueError("RSI 'period' parameter cannot be None.")

    close = _select_price(ohlc_data, 'Close')

    if vbt is not None and hasattr(vbt, 'RSI'):
        return vbt.RSI.run(close, window=period).rsi

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_macd(ohlc_data: pd.DataFrame, fast: int, slow: int, signal: int) -> pd.DataFrame:
    """Calculate MACD and return a DataFrame with columns ['macd','signal','hist']."""
    if not all([fast, slow, signal]):
        raise ValueError("MACD 'fast', 'slow', and 'signal' parameters cannot be None.")

    close = _select_price(ohlc_data, 'Close')

    if vbt is not None and hasattr(vbt, 'MACD'):
        macd = vbt.MACD.run(close, fast_window=fast, slow_window=slow, signal_window=signal)
        out = pd.concat({
            'macd': macd.macd,
            'signal': macd.signal,
            'hist': macd.macd - macd.signal,
        }, axis=1)
        return out

    fast_ema = close.ewm(span=fast, adjust=False).mean()
    slow_ema = close.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return pd.concat({'macd': macd_line, 'signal': signal_line, 'hist': hist}, axis=1)

def calculate_bbands(ohlc_data: pd.DataFrame, period: int, std_dev: float) -> pd.DataFrame:
    """Calculate Bollinger Bands returning columns ['upper','middle','lower']."""
    if period is None or std_dev is None:
        raise ValueError("BBands 'period' and 'std_dev' parameters cannot be None.")

    close = _select_price(ohlc_data, 'Close')

    if vbt is not None and hasattr(vbt, 'BollingerBands'):
        bb = vbt.BollingerBands.run(close, window=period, std=std_dev)
        return pd.concat({'upper': bb.upper, 'middle': bb.middle, 'lower': bb.lower}, axis=1)

    ma = close.rolling(window=period).mean()
    sd = close.rolling(window=period).std(ddof=0)
    upper = ma + sd * std_dev
    lower = ma - sd * std_dev
    return pd.concat({'upper': upper, 'middle': ma, 'lower': lower}, axis=1)

# --- Future Indicators Will Be Added Below ---
# Example of how we would add another indicator, like RSI.
# We will implement this in a later version.
#
# def calculate_rsi(ohlc_data: pd.DataFrame, period: int) -> pd.Series:
#     """
#     Calculates the Relative Strength Index (RSI).
#     """
#     if period is None:
#         raise ValueError("RSI 'period' parameter cannot be None.")
#
#     rsi_series = ohlc_data.ta.rsi(length=period)
#
#     if rsi_series is None:
#         raise ValueError("Failed to calculate RSI. Check input data and parameters.")
#
#     return rsi_series
