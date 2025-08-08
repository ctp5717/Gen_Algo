import pandas as pd
import pandas_ta as ta  # type: ignore

def calculate_ema(df: pd.DataFrame, period: int = 20) -> pd.Series:
    return df['Close'].ta.ema(length=int(period))

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    atr = ta.atr(high=df['High'], low=df['Low'], close=df['Close'], length=int(period))
    return atr

def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return df['Close'].ta.rsi(length=int(period))

def calculate_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    macd = ta.macd(df['Close'], fast=int(fast), slow=int(slow), signal=int(signal))
    return macd['MACD_12_26_9']

def calculate_bbands(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
    bb = ta.bbands(df['Close'], length=int(period), std=float(std_dev))
    return bb.rename(columns={c: c.replace('BOLL', 'BB').replace('L', 'L').replace('M', 'M').replace('U','U')})
