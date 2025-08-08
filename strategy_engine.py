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

import pandas as pd
import indicator_library as ind_lib  # Import our toolbox of indicators

# A mapping dictionary to dynamically call indicator functions.
# This makes the engine scalable. To add a new indicator, you just add an entry here
# and a corresponding function in the indicator_library.
INDICATOR_MAPPING = {
    'ema': ind_lib.calculate_ema,
    'atr': ind_lib.calculate_atr,
    'rsi': ind_lib.calculate_rsi,
    'macd': ind_lib.calculate_macd,
    'bbands': ind_lib.calculate_bbands,
}

def _select_price(ohlc_data: pd.DataFrame, field: str) -> pd.DataFrame:
    """Utility to extract a price field from single or multi-asset data."""
    if isinstance(ohlc_data.columns, pd.MultiIndex):
        return ohlc_data.xs(field, axis=1, level=1)
    return ohlc_data[field]


def _generate_signal(ohlc_data: pd.DataFrame, indicator_series, condition: dict):
    """
    Generates a boolean signal series based on a condition between price and an indicator.
    Works with both Series and DataFrames to support portfolio-level operations.
    """
    condition_type = condition.get('type')
    close_price = _select_price(ohlc_data, 'Close')

    if condition_type == 'price_is_above_indicator':
        return close_price > indicator_series
    elif condition_type == 'price_is_below_indicator':
        return close_price < indicator_series
    elif condition_type == 'price_crosses_above_indicator':
        return close_price.vbt.crossed_above(indicator_series)
    elif condition_type == 'price_crosses_below_indicator':
        return close_price.vbt.crossed_below(indicator_series)
    
    # --- NEW: Added logic for Bollinger Band breakout conditions ---
    elif condition_type == 'price_crosses_above_upper_band':
        # In this case, 'indicator_series' will be the upper band
        return close_price.vbt.crossed_above(indicator_series)
    elif condition_type == 'price_crosses_below_lower_band':
        # In this case, 'indicator_series' will be the lower band
        return close_price.vbt.crossed_below(indicator_series)
    
    else:
        print(f"Warning: Unknown condition type '{condition_type}'.")
        return pd.Series(False, index=ohlc_data.index)

def _generate_signal_from_value(indicator_series, condition: dict):
    """Generate a boolean signal comparing an indicator to a static value."""
    condition_type = condition.get('type')
    value = condition.get('value')

    if condition_type == 'indicator_is_above_value':
        return indicator_series > value
    elif condition_type == 'indicator_is_below_value':
        return indicator_series < value
    elif condition_type == 'indicator_crosses_above_value':
        return indicator_series.vbt.crossed_above(value)
    elif condition_type == 'indicator_crosses_below_value':
        return indicator_series.vbt.crossed_below(value)
    else:
        print(f"Warning: Unknown condition type '{condition_type}' for value comparison.")
        return pd.Series(False, index=indicator_series.index)

def process_strategy_rules(ohlc_data: pd.DataFrame, rules: dict):
    """Process the full strategy rules dictionary to generate entry signals.

    The function now supports both single-asset Series and multi-asset DataFrames.
    """
    entry_rules = rules.get('entry_rules', {})
    conditions = entry_rules.get('conditions', [])
    combination_logic = entry_rules.get('combination_logic', 'AND').upper()

    if not conditions:
        return pd.Series(False, index=ohlc_data.index)

    close_price = _select_price(ohlc_data, 'Close')
    if isinstance(close_price, pd.DataFrame):
        final_entry_signal = pd.DataFrame(True, index=ohlc_data.index, columns=close_price.columns)
    else:
        final_entry_signal = pd.Series(True, index=ohlc_data.index)

    for rule in conditions:
        if not rule.get('is_active', True):
            continue

        indicator_name = rule.get('indicator')
        params = rule.get('params', {})
        condition_logic = rule.get('condition', {})
        indicator_func = INDICATOR_MAPPING.get(indicator_name)

        if not indicator_func:
            print(f"Warning: Indicator '{indicator_name}' not found. Skipping rule.")
            continue

        indicator_output = indicator_func(ohlc_data, **params)
        condition_type = condition_logic.get('type')

        target_series = indicator_output
        if isinstance(indicator_output, pd.DataFrame) and indicator_output.columns.nlevels > 1:
            # Multi-output indicator, select the appropriate component
            if 'bbands' in indicator_name:
                if 'upper' in condition_type:
                    target_series = indicator_output.xs('upper', level=1, axis=1)
                elif 'lower' in condition_type:
                    target_series = indicator_output.xs('lower', level=1, axis=1)
                else:
                    target_series = indicator_output.xs('middle', level=1, axis=1)
            elif 'macd' in indicator_name:
                target_series = indicator_output.xs('hist', level=1, axis=1)

        if isinstance(close_price, pd.DataFrame):
            individual_signal = pd.DataFrame(False, index=ohlc_data.index, columns=close_price.columns)
        else:
            individual_signal = pd.Series(False, index=ohlc_data.index)

        if 'price' in condition_type:
            individual_signal = _generate_signal(ohlc_data, target_series, condition_logic)
        elif 'indicator' in condition_type:
            individual_signal = _generate_signal_from_value(target_series, condition_logic)

        if combination_logic == 'AND':
            final_entry_signal &= individual_signal
        
    return final_entry_signal
