"""Score functions for scanner tie-break policies."""
from __future__ import annotations

from typing import Callable, Dict

import numpy as np
import pandas as pd


def pct_change(data: pd.DataFrame) -> pd.Series:
    """Simple percentage change of the close price."""
    return data["Close"].pct_change()


def momentum_3(data: pd.DataFrame) -> pd.Series:
    """Three-bar momentum based on percentage change."""
    return data["Close"].pct_change().rolling(3).sum()


def atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range for volatility scaling."""
    high_low = data["High"] - data["Low"]
    high_close = (data["High"] - data["Close"].shift()).abs()
    low_close = (data["Low"] - data["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def apply_score_scaling(score: pd.Series, data: pd.DataFrame, method: str | None) -> pd.Series:
    """Scale raw scores to reduce micro-cap bias."""
    if method == "atr":
        scaled = score / atr(data)
    elif method == "dollar_volume":
        scaled = score * (data["Close"] * data["Volume"])
    else:
        scaled = score
    return scaled.replace([np.inf, -np.inf], np.nan)


SCORE_FUNCTIONS: Dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "pct_change": pct_change,
    "momentum_3": momentum_3,
}
