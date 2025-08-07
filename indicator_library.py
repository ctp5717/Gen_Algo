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
import sys
import types
import importlib.metadata as importlib_metadata

# -- Compatibility shim -------------------------------------------------------
# Some versions of pandas_ta expect ``numpy.NaN`` to be defined, but newer
# numpy releases expose only ``numpy.nan``. Importing pandas_ta without this
# attribute raises ``ImportError: cannot import name 'NaN'``. To keep the
# library working across numpy versions, ensure ``np.NaN`` exists before
# importing pandas_ta.
if not hasattr(np, "NaN"):
    np.NaN = np.nan

# -- pkg_resources shim ------------------------------------------------------
# ``pandas_ta`` imports ``pkg_resources`` from ``setuptools`` to detect its
# version.  Newer ``setuptools`` releases emit a deprecation warning each time
# ``pkg_resources`` is imported.  To keep the console output clean (and to be
# forward compatible once ``pkg_resources`` is removed), provide a very small
# stub that offers only the pieces ``pandas_ta`` needs.  The stub relies on
# ``importlib.metadata`` which is part of the standard library.

if "pkg_resources" not in sys.modules:
    pkg_resources_stub = types.ModuleType("pkg_resources")

    class DistributionNotFound(Exception):
        """Replacement for the real pkg_resources exception."""

    def get_distribution(name: str):
        """Lightweight replacement for pkg_resources.get_distribution.

        ``pandas_ta`` expects the returned object to expose both ``version`` and
        ``location`` attributes.  The previous implementation only provided the
        version which caused ``AttributeError`` when pandas_ta accessed
        ``dist.location`` during import.  Here we mirror the minimal interface
        using ``importlib.metadata`` and supply the package installation path as
        ``location``.
        """

        try:
            # ``importlib.metadata`` uses hyphenated package names.  Handle both
            # styles by normalising underscores to hyphens.
            dist = importlib_metadata.distribution(name.replace("_", "-"))
            # ``locate_file('')`` returns the distribution root path
            location = str(dist.locate_file(""))
            return types.SimpleNamespace(version=dist.version, location=location)
        except importlib_metadata.PackageNotFoundError as exc:  # pragma: no cover - unlikely
            raise DistributionNotFound from exc

    pkg_resources_stub.get_distribution = get_distribution
    pkg_resources_stub.DistributionNotFound = DistributionNotFound
    sys.modules["pkg_resources"] = pkg_resources_stub

import pandas_ta as ta

def calculate_ema(ohlc_data: pd.DataFrame, period: int) -> pd.Series:
    """
    Calculates the Exponential Moving Average (EMA).

    Args:
        ohlc_data (pd.DataFrame): DataFrame with columns 'Open', 'High', 'Low', 'Close'.
        period (int): The lookback period for the EMA.

    Returns:
        pd.Series: A pandas Series containing the EMA values.
    """
    if period is None:
        raise ValueError("EMA 'period' parameter cannot be None.")

    if isinstance(ohlc_data.columns, pd.MultiIndex):
        frames = []
        for tk in ohlc_data.columns.get_level_values(0).unique():
            close = ohlc_data[tk]['Close']
            ema = close.ewm(span=period, adjust=False).mean()
            frames.append(pd.DataFrame({(tk, 'EMA'): ema}))
        ema_series = pd.concat(frames, axis=1)
    elif hasattr(ohlc_data, "ta"):
        ema_series = ohlc_data.ta.ema(length=period)
    else:
        close = ohlc_data['Close'] if 'Close' in ohlc_data else ohlc_data.xs('Close', level=-1, axis=1)
        ema_series = close.ewm(span=period, adjust=False).mean()
    
    if ema_series is None:
        raise ConnectionError("Failed to calculate EMA. Check input data and parameters.")
        
    return ema_series

def calculate_atr(ohlc_data: pd.DataFrame, period: int) -> pd.Series:
    """
    Calculates the Average True Range (ATR).

    Args:
        ohlc_data (pd.DataFrame): DataFrame with columns 'Open', 'High', 'Low', 'Close'.
        period (int): The lookback period for the ATR.

    Returns:
        pd.Series: A pandas Series containing the ATR values.
    """
    if period is None:
        raise ValueError("ATR 'period' parameter cannot be None.")

    if isinstance(ohlc_data.columns, pd.MultiIndex):
        frames = []
        for tk in ohlc_data.columns.get_level_values(0).unique():
            high = ohlc_data[tk]['High']
            low = ohlc_data[tk]['Low']
            close = ohlc_data[tk]['Close']
            tr1 = high - low
            tr2 = (high - close.shift()).abs()
            tr3 = (low - close.shift()).abs()
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=period).mean()
            frames.append(pd.DataFrame({(tk, 'ATR'): atr}))
        atr_series = pd.concat(frames, axis=1)
    elif hasattr(ohlc_data, "ta"):
        atr_series = ohlc_data.ta.atr(length=period)
    else:
        high = ohlc_data['High'] if 'High' in ohlc_data else ohlc_data.xs('High', level=-1, axis=1)
        low = ohlc_data['Low'] if 'Low' in ohlc_data else ohlc_data.xs('Low', level=-1, axis=1)
        close = ohlc_data['Close'] if 'Close' in ohlc_data else ohlc_data.xs('Close', level=-1, axis=1)
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_series = true_range.rolling(window=period).mean()
    
    if atr_series is None:
        raise ConnectionError("Failed to calculate ATR. Check input data and parameters.")
        
    return atr_series

def calculate_rsi(ohlc_data: pd.DataFrame, period: int) -> pd.Series:
    """
    Calculates the Relative Strength Index (RSI).
    """
    if period is None:
        raise ValueError("RSI 'period' parameter cannot be None.")

    if isinstance(ohlc_data.columns, pd.MultiIndex):
        frames = []
        for tk in ohlc_data.columns.get_level_values(0).unique():
            close = ohlc_data[tk]['Close']
            delta = close.diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            frames.append(pd.DataFrame({(tk, 'RSI'): rsi}))
        rsi_series = pd.concat(frames, axis=1)
    elif hasattr(ohlc_data, "ta"):
        rsi_series = ohlc_data.ta.rsi(length=period)
    else:
        close = ohlc_data['Close'] if 'Close' in ohlc_data else ohlc_data.xs('Close', level=-1, axis=1)
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        rsi_series = 100 - (100 / (1 + rs))
    
    if rsi_series is None:
        raise ConnectionError("Failed to calculate RSI. Check input data and parameters.")
        
    return rsi_series

def calculate_macd(ohlc_data: pd.DataFrame, fast: int, slow: int, signal: int) -> pd.DataFrame:
    """
    Calculates the Moving Average Convergence Divergence (MACD).
    
    Returns:
        pd.DataFrame: A DataFrame containing MACD line, histogram, and signal line.
    """
    if not all([fast, slow, signal]):
        raise ValueError("MACD 'fast', 'slow', and 'signal' parameters cannot be None.")

    if isinstance(ohlc_data.columns, pd.MultiIndex):
        frames = []
        for tk in ohlc_data.columns.get_level_values(0).unique():
            close = ohlc_data[tk]['Close']
            exp1 = close.ewm(span=fast, adjust=False).mean()
            exp2 = close.ewm(span=slow, adjust=False).mean()
            macd_line = exp1 - exp2
            signal_line = macd_line.ewm(span=signal, adjust=False).mean()
            data = {
                (tk, 'MACD'): macd_line,
                (tk, 'MACDh'): macd_line - signal_line,
                (tk, 'MACDs'): signal_line,
            }
            frames.append(pd.DataFrame(data))
        macd_df = pd.concat(frames, axis=1)
    elif hasattr(ohlc_data, "ta"):
        macd_df = ohlc_data.ta.macd(fast=fast, slow=slow, signal=signal)
    else:
        close = ohlc_data['Close'] if 'Close' in ohlc_data else ohlc_data.xs('Close', level=-1, axis=1)
        exp1 = close.ewm(span=fast, adjust=False).mean()
        exp2 = close.ewm(span=slow, adjust=False).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        macd_df = pd.DataFrame({
            'MACD': macd_line,
            'MACDh': macd_line - signal_line,
            'MACDs': signal_line,
        })

    if macd_df is None:
        raise ConnectionError("Failed to calculate MACD. Check input data and parameters.")
    
    return macd_df

def calculate_bbands(ohlc_data: pd.DataFrame, period: int, std_dev: float) -> pd.DataFrame:
    """
    Calculates Bollinger Bands (BBands).

    Returns:
        pd.DataFrame: A DataFrame containing the upper, middle, and lower bands.
    """
    if period is None or std_dev is None:
        raise ValueError("BBands 'period' and 'std_dev' parameters cannot be None.")

    if isinstance(ohlc_data.columns, pd.MultiIndex):
        frames = []
        for tk in ohlc_data.columns.get_level_values(0).unique():
            close = ohlc_data[tk]['Close']
            ma = close.rolling(window=period).mean()
            std = close.rolling(window=period).std()
            upper = ma + std * std_dev
            lower = ma - std * std_dev
            data = {
                (tk, 'BBL'): lower,
                (tk, 'BBM'): ma,
                (tk, 'BBU'): upper,
            }
            frames.append(pd.DataFrame(data))
        bbands_df = pd.concat(frames, axis=1)
    elif hasattr(ohlc_data, "ta"):
        bbands_df = ohlc_data.ta.bbands(length=period, std=std_dev)
    else:
        close = ohlc_data['Close'] if 'Close' in ohlc_data else ohlc_data.xs('Close', level=-1, axis=1)
        ma = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        upper = ma + std * std_dev
        lower = ma - std * std_dev
        bbands_df = pd.concat([lower.rename('BBL'), ma.rename('BBM'), upper.rename('BBU')], axis=1)

    if bbands_df is None:
        raise ConnectionError("Failed to calculate BBands. Check input data and parameters.")

    return bbands_df

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
