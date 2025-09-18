"""Utility helpers for portfolio construction and exit parameters."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd


def _get_active_value(rules: Mapping[str, Any], key: str) -> Any:
    """Return the ``params.value`` for an active exit rule if present."""

    rule = rules.get(key)
    if not isinstance(rule, Mapping) or not rule.get("is_active", False):
        return None

    params = rule.get("params", {})
    if not isinstance(params, Mapping):
        return None

    return params.get("value")


def extract_exit_params(
    entries: pd.Series,
    exit_rules: Mapping[str, Any] | None,
    hold_period: int,
):
    """Extract exit parameters for vectorbt from strategy rules.

    Parameters
    ----------
    entries : pd.Series
        Entry signal series.
    exit_rules : Mapping[str, Any] | None
        Mapping of exit rules from strategy configuration.
    hold_period : int
        Maximum holding period in bars for time-based exits.

    Returns
    -------
    tuple[pd.Series, float | None, float | None, float | None]
        ``(time_based_exit, sl_stop, sl_trail, tp_stop)``
    """

    rules = exit_rules if isinstance(exit_rules, Mapping) else {}
    time_based_exit = entries.shift(hold_period, fill_value=False)

    sl_stop, sl_trail, tp_stop = (
        _get_active_value(rules, key)
        for key in ("stop_loss", "trailing_stop", "take_profit")
    )

    return time_based_exit, sl_stop, sl_trail, tp_stop
