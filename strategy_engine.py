# strategy_engine.py

"""
Strategy Engine
===============

This script is the core logic engine responsible for interpreting the strategy rules
defined in `config.py`, generating trading signals, and preparing data for the
backtester.

Design Philosophy:
- Agnostic & Dynamic: The engine is designed to be completely agnostic of the strategy
  being tested. It does not contain any hardcoded references to specific indicators
  like 'RSI' or 'EMA'.
- Rule Interpreter: It dynamically parses the `STRATEGY_RULES` dictionary, calls the
  appropriate functions from the `indicator_library`, evaluates the specified conditions
  (e.g., 'price_is_above_indicator'), and combines the resulting boolean signals.
- Scalability: The architecture allows for new indicators and conditions to be added
  with minimal changes to the engine itself. The primary work happens in the
  `indicator_library.py` and `config.py`.
"""

import logging
import math
import warnings
from functools import reduce
from typing import Callable

import numpy as np
import pandas as pd

import indicator_library as ind_lib  # Import our toolbox of indicators
from gene_parser import resolve_gene_value

logger = logging.getLogger(__name__)

# A mapping dictionary to dynamically call indicator functions.
# This makes the engine scalable. To add a new indicator, you just add an entry here
# and a corresponding function in the indicator_library.
INDICATOR_MAPPING = {
    "ema": ind_lib.calculate_ema,
    "atr": ind_lib.calculate_atr,
    "rsi": ind_lib.calculate_rsi,
    "macd": ind_lib.calculate_macd,
    "bbands": ind_lib.calculate_bbands,
}


# Mapping of condition strings to comparison or vectorbt crossover functions
# for price/indicator relationships, including Bollinger Band variants.
BASE_CONDITION_FUNCTIONS: dict[str, Callable[[pd.Series, pd.Series], pd.Series]] = {
    "price_is_above_indicator": lambda price, ind: price > ind,
    "price_is_below_indicator": lambda price, ind: price < ind,
    "price_crosses_above_indicator": lambda price, ind: price.vbt.crossed_above(ind),
    "price_crosses_below_indicator": lambda price, ind: price.vbt.crossed_below(ind),
}

BOLLINGER_CONDITION_FUNCTIONS: dict[
    str, Callable[[pd.Series, pd.Series], pd.Series]
] = {
    "price_crosses_above_upper_band": lambda price, band: price.vbt.crossed_above(band),
    "price_crosses_below_lower_band": lambda price, band: price.vbt.crossed_below(band),
    "price_crosses_below_upper_band": lambda price, band: price.vbt.crossed_below(band),
    "price_crosses_above_lower_band": lambda price, band: price.vbt.crossed_above(band),
    "price_is_above_upper_band": lambda price, band: price > band,
    "price_is_below_lower_band": lambda price, band: price < band,
    "price_is_below_upper_band": lambda price, band: price < band,
    "price_is_above_lower_band": lambda price, band: price > band,
    "price_is_above_middle_band": lambda price, band: price > band,
    "price_is_below_middle_band": lambda price, band: price < band,
}

CONDITION_FUNCTIONS: dict[str, Callable[[pd.Series, pd.Series], pd.Series]] = {
    **BASE_CONDITION_FUNCTIONS,
    **BOLLINGER_CONDITION_FUNCTIONS,
}


def canonical_rule_label(rule: dict) -> str:
    """Return a stable label for a rule used in counts and metadata."""
    name = rule.get("rule_name")
    if not name:
        indicator_name = rule.get("indicator")
        condition = rule.get("condition", {})
        condition_type = condition.get("type")
        name = f"{indicator_name}:{condition_type}"
        column = condition.get("column")
        if column:
            name += f":{column}"
    return name


def _generate_signal(
    ohlc_data: pd.DataFrame, indicator_series: pd.Series, condition: dict
) -> pd.Series:
    """
    Generates a boolean signal series based on a condition between price and an indicator.
    (This version is updated to handle Bollinger Band conditions).
    """
    condition_type = condition.get("type")
    close_price = ohlc_data["Close"]
    func = CONDITION_FUNCTIONS.get(condition_type)
    if func is not None:
        return func(close_price, indicator_series)
    warnings.warn(f"Unknown condition type '{condition_type}'.", stacklevel=2)
    return pd.Series(False, index=ohlc_data.index)


def _generate_signal_from_value(
    indicator_series: pd.Series, condition: dict
) -> pd.Series:
    """
    Generates a boolean signal based on a condition between an indicator and a static value.
    """
    value = condition.get("value")
    if value is None:
        raise ValueError("'value' must be provided for value comparison conditions")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(
            "'value' must be an int or float for value comparison conditions"
        )
    condition_type = condition.get("type")

    if condition_type == "indicator_is_above_value":
        return indicator_series > value
    elif condition_type == "indicator_is_below_value":
        return indicator_series < value
    elif condition_type == "indicator_crosses_above_value":
        return indicator_series.vbt.crossed_above(value)
    elif condition_type == "indicator_crosses_below_value":
        return indicator_series.vbt.crossed_below(value)
    else:
        warnings.warn(
            f"Unknown condition type '{condition_type}' for value comparison.",
            stacklevel=2,
        )
        return pd.Series(False, index=indicator_series.index)


def _combine_signals(
    signals: list[pd.Series],
    combination_logic: str,
    vote_threshold: int | None = None,
    treat_nan_as_false: bool = True,
) -> pd.Series:
    """Combine individual condition signals into a final entry signal.

    Parameters
    ----------
    signals : list[pd.Series]
        Boolean Series for each active condition.
    combination_logic : str
        Logical operator to use ("AND", "OR", "VOTE").
    vote_threshold : int, optional
        Minimum number of conditions that must be true when using
        ``combination_logic="VOTE"``.  ``None`` defaults to a majority.
    treat_nan_as_false : bool, optional
        Whether to replace NaNs with ``False`` before combining.  Defaults to
        ``True``; set to ``False`` to allow NaNs to propagate to the result.
    """

    if not signals:
        return pd.Series(dtype=bool)

    prepared = (
        [s.fillna(False) for s in signals]
        if treat_nan_as_false
        else [s.astype("boolean") for s in signals]
    )

    if combination_logic == "AND":
        return reduce(lambda x, y: x & y, prepared)

    if combination_logic == "OR":
        return reduce(lambda x, y: x | y, prepared)

    if combination_logic == "VOTE":
        threshold = (
            vote_threshold
            if vote_threshold is not None
            else math.ceil(len(prepared) / 2)
        )
        if threshold < 1 or threshold > len(prepared):
            raise ValueError(
                "vote_threshold must be between 1 and the number of active conditions"
            )
        df = pd.DataFrame([s.astype(float) for s in prepared])
        arr = df.to_numpy()
        if treat_nan_as_false:
            signal_sum = np.sum(arr, axis=0)
            return pd.Series(signal_sum >= threshold, index=df.columns)
        nan_mask = np.isnan(arr).any(axis=0)
        arr = np.nan_to_num(arr)
        signal_sum = np.sum(arr, axis=0)
        result = pd.Series(signal_sum >= threshold, index=df.columns, dtype="boolean")
        result[nan_mask] = pd.NA
        return result

    raise ValueError(
        f"Invalid combination_logic '{combination_logic}'. Expected AND, OR, or VOTE."
    )


def process_strategy_rules(
    ohlc_data: pd.DataFrame, rules: dict, collect_counts: bool = False
) -> pd.Series | tuple[pd.Series, dict]:
    """Generate entry signals based on configured rules.

    Parameters
    ----------
    ohlc_data : pd.DataFrame
        OHLCV data.
    rules : dict
        Strategy rules from ``config.STRATEGY_RULES``.
    collect_counts : bool, optional
        If ``True``, also return per-condition true counts.

    Returns
    -------
    pd.Series or (pd.Series, dict)
        Combined entry signals and optionally per-condition counts.
    """
    entry_rules = rules.get("entry_rules", {})
    conditions = entry_rules.get("conditions", [])
    combination_logic = entry_rules.get("combination_logic", "AND")
    combination_logic = str(resolve_gene_value(combination_logic)).upper()
    if combination_logic not in {"AND", "OR", "VOTE"}:
        raise ValueError(
            f"Invalid combination_logic '{combination_logic}'. Expected AND, OR, or VOTE."
        )

    vote_threshold = resolve_gene_value(entry_rules.get("vote_threshold"))
    treat_nan_as_false = entry_rules.get("treat_nan_as_false", True)
    active_conds = [c for c in conditions if c.get("is_active", True)]
    n = len(active_conds)
    requested_k = vote_threshold

    if n == 1:
        if combination_logic != "AND" or vote_threshold not in (None, 1):
            warnings.warn(
                "Single active condition; normalized combination_logic to 'AND' and vote_threshold to 1",
                RuntimeWarning,
                stacklevel=2,
            )
        combination_logic = "AND"
        vote_threshold = 1
    elif combination_logic == "VOTE":
        if vote_threshold is None:
            vote_threshold = math.ceil(n / 2)
        elif vote_threshold < 1:
            vote_threshold = 1
            warnings.warn(
                "Normalized vote_threshold to 1 for VOTE combination",
                RuntimeWarning,
                stacklevel=2,
            )
        if vote_threshold > n:
            warnings.warn(
                "vote_threshold exceeds active conditions; clamped to n",
                RuntimeWarning,
                stacklevel=2,
            )
            vote_threshold = n
        logger.debug(
            {
                "logic": "VOTE",
                "M": n,
                "requested_k": requested_k,
                "final_k": vote_threshold,
                "treat_nan_as_false": treat_nan_as_false,
            }
        )

    if vote_threshold is not None and not isinstance(vote_threshold, int):
        raise TypeError("vote_threshold must be an integer or None")
    if not isinstance(treat_nan_as_false, bool):
        raise TypeError("treat_nan_as_false must be a boolean")

    if vote_threshold is not None:
        assert (
            1 <= vote_threshold <= n
        ), "vote_threshold must be between 1 and the number of active conditions"

    signals = []
    counts = {} if collect_counts else None

    for rule in conditions:
        if not rule.get("is_active", True):
            continue

        indicator_name = rule.get("indicator")
        params = rule.get("params", {})
        condition_logic = rule.get("condition", {})
        indicator_func = INDICATOR_MAPPING.get(indicator_name)

        if not indicator_func:
            warnings.warn(
                f"Indicator '{indicator_name}' not found. Skipping rule.",
                stacklevel=2,
            )
            continue

        indicator_output = indicator_func(ohlc_data, **params)
        condition_type = condition_logic.get("type")

        # --- Intelligent Column Selection ---
        target_series = indicator_output
        if isinstance(indicator_output, pd.DataFrame):
            col_hint = condition_logic.get("column")
            if col_hint:
                if col_hint in indicator_output.columns:
                    target_series = indicator_output[col_hint]
                else:
                    df = indicator_output.filter(regex=col_hint)
                    if df.shape[1] == 0:
                        raise KeyError(
                            f"Requested column '{col_hint}' not found; available: {list(indicator_output.columns)}"
                        )
                    target_series = df.iloc[:, 0]
            elif "bbands" in indicator_name:
                band = condition_logic.get("band")
                if band:
                    band = band.lower()
                    if band == "upper":
                        df = indicator_output.filter(like="BBU")
                        if df.shape[1] == 0:
                            raise KeyError(
                                "Upper band not found in BBands output; expected columns like 'BBU_*'",
                            )
                        target_series = df.iloc[:, 0]
                    elif band == "lower":
                        df = indicator_output.filter(like="BBL")
                        if df.shape[1] == 0:
                            raise KeyError(
                                "Lower band not found in BBands output; expected columns like 'BBL_*'",
                            )
                        target_series = df.iloc[:, 0]
                    else:
                        if band not in {"middle", "mid", "basis"}:
                            warnings.warn(
                                f"Unknown band '{band}' for Bollinger Bands; defaulting to middle",
                                stacklevel=2,
                            )
                        df = indicator_output.filter(like="BBM")
                        if df.shape[1] == 0:
                            raise KeyError(
                                "Middle band not found in BBands output; expected columns like 'BBM_*'",
                            )
                        target_series = df.iloc[:, 0]
                else:
                    if "upper" in condition_type:
                        df = indicator_output.filter(like="BBU")
                        if df.shape[1] == 0:
                            raise KeyError(
                                "Upper band not found in BBands output; expected columns like 'BBU_*'",
                            )
                        target_series = df.iloc[:, 0]
                    elif "lower" in condition_type:
                        df = indicator_output.filter(like="BBL")
                        if df.shape[1] == 0:
                            raise KeyError(
                                "Lower band not found in BBands output; expected columns like 'BBL_*'",
                            )
                        target_series = df.iloc[:, 0]
                    else:
                        df = indicator_output.filter(like="BBM")
                        if df.shape[1] == 0:
                            raise KeyError(
                                "Middle band not found in BBands output; expected columns like 'BBM_*'",
                            )
                        target_series = df.iloc[:, 0]
            elif "macd" in indicator_name:
                if isinstance(indicator_output, pd.Series):
                    target_series = indicator_output
                else:
                    hist = indicator_output.filter(
                        regex=r"(?i)macdh(?:\b|_)|macd[_-]?hist(?:ogram)?"
                    )
                    if hist.shape[1]:
                        target_series = hist.iloc[:, 0]
                    else:
                        macd_line = indicator_output.filter(
                            regex=r"(?i)^macd(?:\b|_)(?!h|s)"
                        )
                        if macd_line.shape[1]:
                            target_series = macd_line.iloc[:, 0]
                        elif indicator_output.shape[1]:
                            target_series = indicator_output.iloc[:, 0]
                        else:
                            raise KeyError(
                                "No MACD columns found; available: "
                                f"{list(indicator_output.columns)}"
                            )
            else:
                if isinstance(indicator_output, pd.Series):
                    target_series = indicator_output
                elif indicator_output.shape[1]:
                    target_series = indicator_output.iloc[:, 0]
                else:
                    raise KeyError(
                        "Indicator produced no columns; available: "
                        f"{list(indicator_output.columns)}"
                    )

        individual_signal = pd.Series(False, index=ohlc_data.index)
        if "price" in condition_type:
            individual_signal = _generate_signal(
                ohlc_data, target_series, condition_logic
            )
        elif "indicator" in condition_type:
            individual_signal = _generate_signal_from_value(
                target_series, condition_logic
            )

        signals.append(individual_signal)
        if collect_counts:
            name = canonical_rule_label(rule)
            val = (
                individual_signal.fillna(False)
                if treat_nan_as_false
                else individual_signal
            )
            counts[name] = int(pd.Series(val, dtype="boolean").sum(skipna=True))

    if not signals:
        empty = pd.Series(False, index=ohlc_data.index)
        return (empty, {}) if collect_counts else empty

    if len(signals) == 1:
        single = signals[0]
        result = (
            single.fillna(False) if treat_nan_as_false else single.astype("boolean")
        )
        return (result, counts) if collect_counts else result

    combined = _combine_signals(
        signals, combination_logic, vote_threshold, treat_nan_as_false
    )
    return (combined, counts) if collect_counts else combined
