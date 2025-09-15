"""Formatting helpers for numbers and percentages."""

from __future__ import annotations

import math


def _is_bad(x) -> bool:
    return x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))


def _norm_zero(val: float) -> float:
    """Return +0.0 for any representation of negative zero."""

    return 0.0 if val == 0.0 else val


def fmt_num(x: float, digits: int = 2) -> str:
    """Format a float with a fixed number of decimal places."""
    if _is_bad(x):
        return "—"
    value = _norm_zero(float(x))
    return f"{value:.{digits}f}"


def fmt_pct(x: float, digits: int = 1) -> str:
    """Format a percentage with a fixed number of decimal places and % sign."""
    if _is_bad(x):
        return "—"
    value = _norm_zero(float(x))
    return f"{value:.{digits}f}%"
