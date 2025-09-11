"""Utility helpers for portfolio construction and exit parameters."""

from __future__ import annotations

import pandas as pd


def extract_exit_params(
    entries: pd.Series,
    exit_rules: dict | None,
    hold_period: int,
):
    """Extract exit parameters for vectorbt from strategy rules.

    Parameters
    ----------
    entries : pd.Series
        Entry signal series.
    exit_rules : dict | None
        Dictionary of exit rules from strategy configuration.
    hold_period : int
        Maximum holding period in bars for time-based exits.

    Returns
    -------
    tuple[pd.Series, float | None, float | None, float | None]
        ``(time_based_exit, sl_stop, sl_trail, tp_stop)``
    """

    exit_rules = exit_rules or {}

    sl_rule = exit_rules.get("stop_loss", {})
    tsl_rule = exit_rules.get("trailing_stop", {})
    tp_rule = exit_rules.get("take_profit", {})

    sl_stop = (
        sl_rule.get("params", {}).get("value")
        if sl_rule.get("is_active", False)
        else None
    )
    sl_trail = (
        tsl_rule.get("params", {}).get("value")
        if tsl_rule.get("is_active", False)
        else None
    )
    tp_stop = (
        tp_rule.get("params", {}).get("value")
        if tp_rule.get("is_active", False)
        else None
    )

    time_based_exit = entries.shift(hold_period, fill_value=False)
    time_based_exit = time_based_exit.reindex(entries.index, fill_value=False)

    return time_based_exit, sl_stop, sl_trail, tp_stop
