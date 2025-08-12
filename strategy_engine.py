import pandas as pd
from typing import Dict
import indicator_library as ind_lib

INDICATOR_MAPPING = {
    'ema': ind_lib.calculate_ema,
    'atr': ind_lib.calculate_atr,
    'rsi': ind_lib.calculate_rsi,
    'macd': ind_lib.calculate_macd,
    'bbands': ind_lib.calculate_bbands,
}

def _generate_signal(ohlc_data: pd.DataFrame, indicator_series: pd.Series, condition: dict) -> pd.Series:
    t = condition.get('type')
    if t == 'price_is_above_indicator':
        return (ohlc_data['Close'] > indicator_series).fillna(False)
    if t == 'indicator_is_above_value':
        v = condition.get('value', 0)
        return (indicator_series > v).fillna(False)
    if t == 'indicator_crosses_above_value':
        v = condition.get('value', 0)
        return (indicator_series.shift(1) <= v) & (indicator_series > v)
    if t == 'price_crosses_above_upper_band':
        col = condition.get('column')
        upper = indicator_series if hasattr(indicator_series, 'name') and indicator_series.name == col else None
        if isinstance(indicator_series, pd.DataFrame):
            upper = indicator_series.get(col)
        if upper is None:
            return pd.Series(False, index=ohlc_data.index)
        return (ohlc_data['Close'].shift(1) <= upper.shift(1)) & (ohlc_data['Close'] > upper)
    return pd.Series(False, index=ohlc_data.index)

def _process_single_asset(df: pd.DataFrame, rules: dict) -> pd.Series:
    logic = (rules.get('entry_rules', {}) or {}).get('combination_logic', 'AND').upper()
    conds = (rules.get('entry_rules', {}) or {}).get('conditions', [])
    signals = []
    for c in conds:
        if c.get('is_active') is False:
            continue
        ind = c.get('indicator')
        fn = INDICATOR_MAPPING.get(ind)
        if fn is None:
            continue
        params = c.get('params', {}) or {}
        ind_series = fn(df, **{k: (v if not isinstance(v, dict) else v.get('value', None) or v) for k, v in params.items()})
        if isinstance(ind_series, pd.DataFrame):
            col = (c.get('condition', {}) or {}).get('column')
            if col and col in ind_series.columns:
                target = ind_series[col]
            else:
                target = ind_series.iloc[:, 1] if ind_series.shape[1] > 1 else ind_series.iloc[:, 0]
        else:
            target = ind_series
        signals.append(_generate_signal(df, target, c.get('condition', {}) or {}))
    if not signals:
        return pd.Series(False, index=df.index)
    if logic == 'AND':
        out = signals[0]
        for s in signals[1:]:
            out = out & s
        return out.fillna(False)
    else:
        out = signals[0]
        for s in signals[1:]:
            out = out | s
        return out.fillna(False)

def process_strategy_rules(ohlc_data: pd.DataFrame, rules: dict) -> pd.DataFrame | pd.Series:
    if isinstance(ohlc_data.columns, pd.MultiIndex):
        assets = sorted(set(ohlc_data.columns.get_level_values(0)))
        out = {}
        for a in assets:
            df = ohlc_data[a].copy()
            out[a] = _process_single_asset(df, rules)
        return pd.DataFrame(out).reindex(ohlc_data.index)
    return _process_single_asset(ohlc_data, rules)
