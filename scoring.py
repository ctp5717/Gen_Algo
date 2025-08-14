"""Score functions for scanner tie-break policies."""
from __future__ import annotations

from typing import Callable, Dict

import pandas as pd


def pct_change(data: pd.DataFrame) -> pd.Series:
    """Simple percentage change of the close price."""
    return data["Close"].pct_change()


def momentum_3(data: pd.DataFrame) -> pd.Series:
    """Three-bar momentum based on percentage change."""
    return data["Close"].pct_change().rolling(3).sum()


SCORE_FUNCTIONS: Dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "pct_change": pct_change,
    "momentum_3": momentum_3,
}
