"""Preflight checks for indicator contracts."""

from __future__ import annotations

import pandas as pd

import indicator_contracts as contracts
import strategy_engine


class PreflightError(Exception):
    """Raised when an indicator violates its declared contract."""


def check_indicator_contracts(ohlc: pd.DataFrame, rules: dict) -> None:
    """Validate that configured indicators adhere to their column contracts."""
    entry = rules.get("entry_rules", {})
    for idx, cond in enumerate(entry.get("conditions", []), start=1):
        name = (cond.get("indicator") or "").lower()
        func = strategy_engine.INDICATOR_MAPPING.get(name)
        if func is None:
            continue
        params = cond.get("params", {})
        canonical = strategy_engine.INDICATOR_CANONICAL.get(name, name)
        norm_params = strategy_engine._normalize_indicator_params(canonical, params)
        try:
            output = func(ohlc, **norm_params)
            norm = contracts.normalize_output(
                canonical, output, norm_params, index=ohlc.index
            )
            condition = cond.get("condition", {})
            try:
                strategy_engine.select_indicator_series(
                    canonical,
                    norm,
                    condition,
                    strict_column=True,
                    indicator_params=norm_params,
                )
            except KeyError as ke:
                col = condition.get("column")
                raise PreflightError(
                    f"[cond#{idx}] {name}: missing column {col}"
                ) from ke
        except Exception as e:
            raise PreflightError(f"[cond#{idx}] {name}: {e}") from e
