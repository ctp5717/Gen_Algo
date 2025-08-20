"""Utility functions for the project."""
from typing import Union
import random
import numpy as np


def set_global_seed(seed: int) -> None:
    """Seed Python and NumPy RNGs for reproducible behaviour."""
    random.seed(seed)
    np.random.seed(seed)


def _norm_freq(freq: Union[str, None]) -> Union[str, None]:
    """Normalize timeframe strings to pandas frequency aliases.

    Examples
    --------
    "15m" -> "15min"
    "1h"  -> "1H"
    "1wk" -> "1W"
    "1mo" -> "1M"
    """
    if not isinstance(freq, str):
        return freq
    f = freq.strip().lower()
    mapping = {
        "m": "min",
        "h": "H",
        "d": "D",
        "wk": "W",
        "mo": "M",
    }
    for suffix, repl in mapping.items():
        if f.endswith(suffix):
            return f[:-len(suffix)] + repl
    return f.upper()
