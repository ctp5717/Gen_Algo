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
import warnings

# -- Compatibility shim -------------------------------------------------------
# Some versions of pandas_ta expect ``numpy.NaN`` to be defined, but newer
# numpy releases expose only ``numpy.nan``. Importing pandas_ta without this
# attribute raises ``ImportError: cannot import name 'NaN'``. To keep the
# library working across numpy versions, ensure ``np.NaN`` exists before
# importing pandas_ta.
if not hasattr(np, "NaN"):
    np.NaN = np.nan

# ``pandas_ta`` emits a deprecation warning on import due to its use of
# ``pkg_resources``.  The library is scheduled to remove this dependency but in
# the interim the warning is extremely noisy during optimisation where the
# module may be imported many times.  Suppress it here to keep logs readable.
warnings.filterwarnings(
    "ignore",
    message=r"pkg_resources is deprecated as an API",
    category=UserWarning,
    module=r"pandas_ta",
)

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
    
    # pandas-ta automatically finds the 'Close' column to perform the calculation
    ema_series = ohlc_data.ta.ema(length=period)
    
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
    
    # pandas-ta uses the high, low, and close columns for the ATR calculation
    atr_series = ohlc_data.ta.atr(length=period)
    
    if atr_series is None:
        raise ConnectionError("Failed to calculate ATR. Check input data and parameters.")
        
    return atr_series

def calculate_rsi(ohlc_data: pd.DataFrame, period: int) -> pd.Series:
    """
    Calculates the Relative Strength Index (RSI).
    """
    if period is None:
        raise ValueError("RSI 'period' parameter cannot be None.")
    
    rsi_series = ohlc_data.ta.rsi(length=period)
    
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
    
    # The .ta.macd() function returns a DataFrame with multiple columns
    macd_df = ohlc_data.ta.macd(fast=fast, slow=slow, signal=signal)

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
    
    bbands_df = ohlc_data.ta.bbands(length=period, std=std_dev)

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
