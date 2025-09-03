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

import math
from functools import reduce

import pandas as pd

import indicator_library as ind_lib  # Import our toolbox of indicators

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


def _generate_signal(
    ohlc_data: pd.DataFrame, indicator_series: pd.Series, condition: dict
) -> pd.Series:
    """
    Generates a boolean signal series based on a condition between price and an indicator.
    (This version is updated to handle Bollinger Band conditions).
    """
    condition_type = condition.get("type")
    close_price = ohlc_data["Close"]

    if condition_type == "price_is_above_indicator":
        return close_price > indicator_series
    elif condition_type == "price_is_below_indicator":
        return close_price < indicator_series
    elif condition_type == "price_crosses_above_indicator":
        return close_price.vbt.crossed_above(indicator_series)
    elif condition_type == "price_crosses_below_indicator":
        return close_price.vbt.crossed_below(indicator_series)

    # --- NEW: Added logic for Bollinger Band breakout conditions ---
    elif condition_type == "price_crosses_above_upper_band":
        # In this case, 'indicator_series' will be the upper band
        return close_price.vbt.crossed_above(indicator_series)
    elif condition_type == "price_crosses_below_lower_band":
        # In this case, 'indicator_series' will be the lower band
        return close_price.vbt.crossed_below(indicator_series)

    else:
        print(f"Warning: Unknown condition type '{condition_type}'.")
        return pd.Series(False, index=ohlc_data.index)


def _generate_signal_from_value(
    indicator_series: pd.Series, condition: dict
) -> pd.Series:
    """
    Generates a boolean signal based on a condition between an indicator and a static value.
    """
    condition_type = condition.get("type")
    value = condition.get("value")

    if condition_type == "indicator_is_above_value":
        return indicator_series > value
    elif condition_type == "indicator_is_below_value":
        return indicator_series < value
    elif condition_type == "indicator_crosses_above_value":
        return indicator_series.vbt.crossed_above(value)
    elif condition_type == "indicator_crosses_below_value":
        return indicator_series.vbt.crossed_below(value)
    else:
        print(
            f"Warning: Unknown condition type '{condition_type}' for value comparison."
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
        if treat_nan_as_false:
            signal_sum = pd.concat(prepared, axis=1).astype(int).sum(axis=1)
            return signal_sum >= threshold
        signal_df = pd.concat(prepared, axis=1)
        signal_sum = signal_df.astype("Int64").sum(axis=1, min_count=len(prepared))
        return signal_sum >= threshold

    raise ValueError(
        f"Invalid combination_logic '{combination_logic}'. Expected AND, OR, or VOTE."
    )


def process_strategy_rules(ohlc_data: pd.DataFrame, rules: dict) -> pd.Series:
    """Generate entry signals based on configured rules.

    Parameters
    ----------
    ohlc_data : pd.DataFrame
        OHLCV data.
    rules : dict
        Strategy rules from ``config.STRATEGY_RULES``.

    Returns
    -------
    pd.Series
        Boolean Series representing combined entry signals.
    """
    entry_rules = rules.get("entry_rules", {})
    conditions = entry_rules.get("conditions", [])
    combination_logic = entry_rules.get("combination_logic", "AND").upper()
    vote_threshold = entry_rules.get("vote_threshold")
    treat_nan_as_false = entry_rules.get("treat_nan_as_false", True)

    if vote_threshold is not None and not isinstance(vote_threshold, int):
        raise TypeError("vote_threshold must be an integer or None")
    if not isinstance(treat_nan_as_false, bool):
        raise TypeError("treat_nan_as_false must be a boolean")

    if combination_logic not in {"AND", "OR", "VOTE"}:
        raise ValueError(
            f"Invalid combination_logic '{combination_logic}'. Expected AND, OR, or VOTE."
        )

    signals = []

    for rule in conditions:
        if not rule.get("is_active", True):
            continue

        indicator_name = rule.get("indicator")
        params = rule.get("params", {})
        condition_logic = rule.get("condition", {})
        indicator_func = INDICATOR_MAPPING.get(indicator_name)

        if not indicator_func:
            print(f"Warning: Indicator '{indicator_name}' not found. Skipping rule.")
            continue

        indicator_output = indicator_func(ohlc_data, **params)
        condition_type = condition_logic.get("type")

        # --- Intelligent Column Selection ---
        target_series = indicator_output
        if isinstance(indicator_output, pd.DataFrame):
            if "bbands" in indicator_name:
                if "upper" in condition_type:
                    target_series = indicator_output.filter(like="BBU").iloc[:, 0]
                elif "lower" in condition_type:
                    target_series = indicator_output.filter(like="BBL").iloc[:, 0]
                else:
                    target_series = indicator_output.filter(like="BBM").iloc[:, 0]
            elif "macd" in indicator_name:
                target_series = indicator_output.filter(like="MACDh").iloc[:, 0]
            else:
                target_series = indicator_output.iloc[:, 0]

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

    if not signals:
        return pd.Series(False, index=ohlc_data.index)

    if len(signals) == 1:
        if combination_logic == "VOTE":
            threshold = vote_threshold if vote_threshold is not None else 1
            if threshold < 1 or threshold > 1:
                raise ValueError(
                    "vote_threshold must be between 1 and the number of active conditions"
                )
        single = signals[0]
        return single.fillna(False) if treat_nan_as_false else single.astype("boolean")

    return _combine_signals(
        signals, combination_logic, vote_threshold, treat_nan_as_false
    )
