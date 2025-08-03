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
import indicator_library as ind_lib # Import our toolbox of indicators

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

def _generate_signal(ohlc_data: pd.DataFrame, indicator_series: pd.Series, condition: dict) -> pd.Series:
    """
    Generates a boolean signal series based on a condition between price and an indicator.
    (This version is updated to handle Bollinger Band conditions).
    """
    condition_type = condition.get('type')
    close_price = ohlc_data['Close']

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

def _generate_signal_from_value(indicator_series: pd.Series, condition: dict) -> pd.Series:
    """
    Generates a boolean signal based on a condition between an indicator and a static value.
    """
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

def process_strategy_rules(ohlc_data: pd.DataFrame, rules: dict) -> pd.Series:
    """
    Processes the full strategy rules dictionary to generate final entry signals.
    (This version can intelligently select columns from multi-output indicators).
    """
    # If the data contains multiple assets stacked in a MultiIndex, process each
    # asset individually and concatenate the resulting signals.  This keeps the
    # existing single-asset logic intact while enabling portfolio mode.
    if isinstance(ohlc_data.columns, pd.MultiIndex):
        signals = {}
        for asset in ohlc_data.columns.get_level_values(0).unique():
            signals[asset] = process_strategy_rules(ohlc_data[asset], rules)
        return pd.concat(signals, axis=1)

    entry_rules = rules.get('entry_rules', {})
    conditions = entry_rules.get('conditions', [])
    combination_logic = entry_rules.get('combination_logic', 'AND').upper()

    if not conditions:
        return pd.Series(False, index=ohlc_data.index)

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
        
        # --- NEW: Intelligent Column Selection ---
        target_series = indicator_output
        if isinstance(indicator_output, pd.DataFrame):
            # If the output is a DataFrame, find the correct column to use.
            if 'bbands' in indicator_name:
                if 'upper' in condition_type: target_series = indicator_output.filter(like='BBU').iloc[:, 0]
                elif 'lower' in condition_type: target_series = indicator_output.filter(like='BBL').iloc[:, 0]
                else: target_series = indicator_output.filter(like='BBM').iloc[:, 0] # Default to middle band
            elif 'macd' in indicator_name:
                target_series = indicator_output.filter(like='MACDh').iloc[:, 0] # Default to histogram
            else:
                # Fallback for other multi-column indicators if not specified
                target_series = indicator_output.iloc[:, 0]

        individual_signal = pd.Series(False, index=ohlc_data.index)
        if 'price' in condition_type:
            individual_signal = _generate_signal(ohlc_data, target_series, condition_logic)
        elif 'indicator' in condition_type:
            individual_signal = _generate_signal_from_value(target_series, condition_logic)

        if combination_logic == 'AND':
            final_entry_signal &= individual_signal
        
    return final_entry_signal
