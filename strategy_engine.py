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
import re
import warnings
import weakref
from typing import Any, Callable

import numpy as np
import pandas as pd

import config
import indicator_library as ind_lib  # Import our toolbox of indicators

logger = logging.getLogger(__name__)

# Auto-registered indicator mapping from ``indicator_library``
INDICATOR_MAPPING = {k.lower(): v for k, v in ind_lib.INDICATOR_REGISTRY.items()}

# Common shorthand aliases for indicators
ALIASES = {
    "uo": "ultimate_oscillator",
    "willr": "williams_r",
    "kc": "keltner",
    "dc": "donchian",
    "dmi": "adx",
    "bb": "bbands",
    "bollinger": "bbands",
    "keltner_channels": "keltner",
}
for alias, target in ALIASES.items():
    func = INDICATOR_MAPPING.get(target.lower())
    if func:
        INDICATOR_MAPPING[alias] = func

# Indicators that require a ``Volume`` column
VOLUME_INDICATORS = {"obv", "mfi", "adl", "cmf"}

# Cached column prefixes for common multi-output indicators to avoid repeated
# DataFrame.filter calls.
INDICATOR_COLUMN_PREFIXES = {
    "bbands": {"upper": "BBU", "middle": "BBM", "lower": "BBL"},
    "keltner": {"upper": "KCU", "middle": "KCM", "lower": "KCL"},
    "donchian": {"upper": "DCU", "middle": "DCM", "lower": "DCL"},
    "ma_envelope": {"upper": "MAE_U", "middle": "MAE_M", "lower": "MAE_L"},
    "adx": {"main": "ADX"},
    "stoch": {"k": "STOCHk"},
    "ichimoku": {"baseline": "IKS"},
}

MACD_HIST_PATTERN = re.compile(r"(?i)macdh(?:\b|_)|macd[_-]?hist(?:ogram)?")
MACD_LINE_PATTERN = re.compile(r"(?i)^macd(?:\b|_)(?!h|s)")
TRIX_LINE_PATTERN = re.compile(r"(?i)^TRIX(?!s)")


# Global cache for indicator outputs keyed by
# (indicator_name, params_tuple, data_id, columns). Each cache entry stores a
# weak reference to the original DataFrame so we can verify that the cached
# result was produced for the same object even if Python reuses ``id`` values
# after garbage collection.
CacheKey = tuple[str, tuple[tuple[str, Any], ...], int, tuple[str, ...]]
CacheVal = tuple[weakref.ReferenceType[pd.DataFrame], pd.Series | pd.DataFrame]
_INDICATOR_CACHE: dict[CacheKey, CacheVal] = {}
_CACHE_HITS = 0
_CACHE_MISSES = 0


def clear_indicator_cache() -> None:
    """Clear the indicator output cache and reset counters."""
    _INDICATOR_CACHE.clear()
    global _CACHE_HITS, _CACHE_MISSES
    _CACHE_HITS = 0
    _CACHE_MISSES = 0


def cache_stats() -> dict[str, int]:
    return {"hits": _CACHE_HITS, "misses": _CACHE_MISSES, "size": len(_INDICATOR_CACHE)}


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


def _band_resolver(prefixes: dict[str, str], label: str) -> Callable:
    """Create a resolver for band-based indicators."""

    def resolver(
        output: pd.DataFrame,
        condition_logic: dict,
        condition_type: str,
        choose_first: Callable,
        _df_from_prefix: Callable[[str], pd.DataFrame],
        _df_from_regex: Callable[[str, re.Pattern], pd.DataFrame],
    ) -> pd.Series:
        band = condition_logic.get("band")
        if not band:
            if "upper" in condition_type:
                band = "upper"
            elif "lower" in condition_type:
                band = "lower"
            else:
                band = "middle"
        else:
            band = str(band).lower()

        if band not in {"upper", "lower", "middle", "mid", "basis"}:
            warnings.warn(
                f"Unknown band '{band}' for {label}; defaulting to middle",
                stacklevel=2,
            )
            band = "middle"

        key = band
        if band in {"middle", "mid", "basis"}:
            key = "middle"

        df = _df_from_prefix(prefixes[key])
        msg = (
            f"{key.capitalize()} band not found in {label} output; expected columns like "
            f"'{prefixes[key]}_*'"
        )
        return choose_first(df, msg, fallback=False)

    return resolver


def _macd_resolver(
    output: pd.DataFrame,
    condition_logic: dict,
    condition_type: str,
    choose_first: Callable,
    _df_from_prefix: Callable[[str], pd.DataFrame],
    _df_from_regex: Callable[[str, re.Pattern], pd.DataFrame],
) -> pd.Series:
    hist = _df_from_regex("macd_hist", MACD_HIST_PATTERN)
    if hist.shape[1]:
        return hist.iloc[:, 0]
    macd_line = _df_from_regex("macd_line", MACD_LINE_PATTERN)
    if macd_line.shape[1]:
        return macd_line.iloc[:, 0]
    if output.shape[1]:
        return output.iloc[:, 0]
    raise KeyError("No MACD columns found; available: " f"{list(output.columns)}")


def _adx_resolver(
    output: pd.DataFrame,
    condition_logic: dict,
    condition_type: str,
    choose_first: Callable,
    _df_from_prefix: Callable[[str], pd.DataFrame],
    _df_from_regex: Callable[[str, re.Pattern], pd.DataFrame],
) -> pd.Series:
    df = _df_from_prefix(INDICATOR_COLUMN_PREFIXES["adx"]["main"])
    return choose_first(df, "ADX column not found", fallback=False)


def _stoch_resolver(
    output: pd.DataFrame,
    condition_logic: dict,
    condition_type: str,
    choose_first: Callable,
    _df_from_prefix: Callable[[str], pd.DataFrame],
    _df_from_regex: Callable[[str, re.Pattern], pd.DataFrame],
) -> pd.Series:
    df = _df_from_prefix(INDICATOR_COLUMN_PREFIXES["stoch"]["k"])
    msg = "%K column not found in Stochastic output; expected columns like 'STOCHk_*'"
    return choose_first(df, msg, fallback=False)


def _ichimoku_resolver(
    output: pd.DataFrame,
    condition_logic: dict,
    condition_type: str,
    choose_first: Callable,
    _df_from_prefix: Callable[[str], pd.DataFrame],
    _df_from_regex: Callable[[str, re.Pattern], pd.DataFrame],
) -> pd.Series:
    df = _df_from_prefix(INDICATOR_COLUMN_PREFIXES["ichimoku"]["baseline"])
    msg = "Baseline column not found in Ichimoku output; expected columns like 'IKS_*'"
    return choose_first(df, msg, fallback=False)


def _pivot_resolver(
    output: pd.DataFrame,
    condition_logic: dict,
    condition_type: str,
    choose_first: Callable,
    _df_from_prefix: Callable[[str], pd.DataFrame],
    _df_from_regex: Callable[[str, re.Pattern], pd.DataFrame],
) -> pd.Series:
    if "P" in output.columns:
        return output["P"]
    if output.shape[1]:
        return output.iloc[:, 0]
    raise KeyError(
        "Pivot Points output produced no columns; available: " f"{list(output.columns)}"
    )


def _trix_resolver(
    output: pd.DataFrame,
    condition_logic: dict,
    condition_type: str,
    choose_first: Callable,
    _df_from_prefix: Callable[[str], pd.DataFrame],
    _df_from_regex: Callable[[str, re.Pattern], pd.DataFrame],
) -> pd.Series:
    df = _df_from_regex("trix_line", TRIX_LINE_PATTERN)
    return choose_first(
        df, "TRIX line not found; expected columns like 'TRIX_*'", fallback=False
    )


INDICATOR_SERIES_RESOLVERS: dict[str, Callable] = {
    "bbands": _band_resolver(INDICATOR_COLUMN_PREFIXES["bbands"], "BBands"),
    "keltner": _band_resolver(INDICATOR_COLUMN_PREFIXES["keltner"], "Keltner"),
    "donchian": _band_resolver(INDICATOR_COLUMN_PREFIXES["donchian"], "Donchian"),
    "ma_envelope": _band_resolver(
        INDICATOR_COLUMN_PREFIXES["ma_envelope"], "MA Envelope"
    ),
    "macd": _macd_resolver,
    "adx": _adx_resolver,
    "stoch": _stoch_resolver,
    "ichimoku": _ichimoku_resolver,
    "pivot_points": _pivot_resolver,
    "pivots": _pivot_resolver,
    "trix": _trix_resolver,
}


def select_indicator_series(
    indicator_name: str,
    indicator_output: pd.Series | pd.DataFrame,
    condition_logic: dict,
    strict_column: bool,
) -> pd.Series:
    """Return the appropriate Series from ``indicator_output``."""

    if isinstance(indicator_output, pd.Series):
        return indicator_output

    col_hint = condition_logic.get("column")
    rule_strict = condition_logic.get("strict_column", strict_column)
    if not isinstance(rule_strict, bool):
        raise TypeError("strict_column must be a boolean")

    columns = list(indicator_output.columns)
    prefix_cache: dict[str, str | None] = {}
    regex_cache: dict[str, str | None] = {}

    def _df_from_prefix(prefix: str, cols=columns) -> pd.DataFrame:
        col = prefix_cache.get(prefix)
        if col is None:
            col = next((c for c in cols if c.startswith(prefix)), None)
            prefix_cache[prefix] = col
        return indicator_output[[col]] if col else indicator_output.iloc[:, 0:0]

    def _df_from_regex(name: str, pattern: re.Pattern, cols=columns) -> pd.DataFrame:
        col = regex_cache.get(name)
        if col is None:
            col = next((c for c in cols if pattern.search(c)), None)
            regex_cache[name] = col
        return indicator_output[[col]] if col else indicator_output.iloc[:, 0:0]

    def choose_first(
        df: pd.DataFrame,
        msg: str,
        output: pd.DataFrame = indicator_output,
        strict: bool = rule_strict,
        fallback: bool = True,
    ) -> pd.Series:
        avail = list(output.columns)
        msg = f"{msg}; available: {avail}"
        if df.shape[1] == 0:
            if strict:
                raise KeyError(msg + "; set strict_column=False to allow fallback")
            warnings.warn(msg + "; using first available column", stacklevel=2)
            if output.shape[1] == 0:
                raise KeyError(msg + "; set strict_column=False to allow fallback")
            return output.iloc[:, 0]
        if not fallback:
            return df.iloc[:, 0]
        if strict:
            raise KeyError(msg + "; set strict_column=False to allow fallback")
        warnings.warn(msg + "; using first available column", stacklevel=2)
        return df.iloc[:, 0]

    if col_hint:
        if col_hint in indicator_output.columns:
            return indicator_output[col_hint]
        df = _df_from_regex(col_hint, re.compile(col_hint))
        return choose_first(df, f"Requested column '{col_hint}' not found")

    condition_type = condition_logic.get("type", "")

    for key, resolver in INDICATOR_SERIES_RESOLVERS.items():
        if key in indicator_name:
            return resolver(
                indicator_output,
                condition_logic,
                condition_type,
                choose_first,
                _df_from_prefix,
                _df_from_regex,
            )

    if indicator_output.shape[1]:
        return indicator_output.iloc[:, 0]

    raise KeyError(
        "Indicator produced no columns; available: " f"{list(indicator_output.columns)}"
    )


def _combine_signals(
    signals: list[pd.Series],
    combination_logic: str,
    vote_threshold: int | None = None,
    nan_policy: str = "FALSE",
    ffill_lookback: int | None = None,
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
    nan_policy : str, optional
        "FALSE" replaces NaNs with ``False``; "PROPAGATE" allows NaNs to
        propagate; "FORWARD_FILL" forward-fills each series (respecting
        ``ffill_lookback``) then replaces remaining NaNs with ``False``.
    ffill_lookback : int, optional
        Maximum bars to forward-fill when ``nan_policy`` is "FORWARD_FILL".
    """

    if not signals:
        return pd.Series(dtype=bool)

    idx = signals[0].index
    name = signals[0].name

    policy = nan_policy.upper()
    limit = None if ffill_lookback == 0 else ffill_lookback
    if policy == "FORWARD_FILL":
        filled = [s.ffill(limit=limit).fillna(False) for s in signals]
        arr = np.stack([s.to_numpy(dtype=bool, copy=False) for s in filled])
        treat_nan_as_false = True
    elif policy == "FALSE":
        arr = np.stack(
            [s.fillna(False).to_numpy(dtype=bool, copy=False) for s in signals]
        )
        treat_nan_as_false = True
    elif policy == "PROPAGATE":
        arr = np.stack([s.to_numpy(dtype=float, copy=False) for s in signals])
        treat_nan_as_false = False
    else:
        raise ValueError("nan_policy must be FALSE, PROPAGATE, or FORWARD_FILL")
    if treat_nan_as_false:
        if combination_logic == "AND":
            combined = np.logical_and.reduce(arr, axis=0)
            return pd.Series(combined, index=idx, name=name)
        if combination_logic == "OR":
            combined = np.logical_or.reduce(arr, axis=0)
            return pd.Series(combined, index=idx, name=name)
        if combination_logic == "VOTE":
            M = arr.shape[0]
            threshold = (
                vote_threshold if vote_threshold is not None else math.ceil(M / 2)
            )
            if threshold < 1 or threshold > M:
                raise ValueError(
                    "vote_threshold must be between 1 and the number of active conditions"
                )
            payload = {"logic": "VOTE", "M": M, "k": threshold, "nan_policy": policy}
            logger.info(payload)
            votes = np.sum(arr, axis=0)
            return pd.Series(votes >= threshold, index=idx)

    has_nan = np.isnan(arr).any(axis=0)
    if combination_logic == "AND":
        has_false = (arr == 0).any(axis=0)
        combined = np.where(has_false, False, np.where(has_nan, np.nan, True))
        return pd.Series(combined, index=idx, dtype="boolean", name=name)
    if combination_logic == "OR":
        has_true = (arr == 1).any(axis=0)
        combined = np.where(has_true, True, np.where(has_nan, np.nan, False))
        return pd.Series(combined, index=idx, dtype="boolean", name=name)
    if combination_logic == "VOTE":
        M = arr.shape[0]
        threshold = vote_threshold if vote_threshold is not None else math.ceil(M / 2)
        if threshold < 1 or threshold > M:
            raise ValueError(
                "vote_threshold must be between 1 and the number of active conditions"
            )
        payload = {"logic": "VOTE", "M": M, "k": threshold, "nan_policy": policy}
        logger.info(payload)
        votes = np.sum(arr, axis=0)
        series = pd.Series(votes, index=idx, dtype="Float64")
        return series.ge(threshold)

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
        Strategy rules from ``strategy_rules.STRATEGY_RULES``.
    collect_counts : bool, optional
        If ``True``, also return per-condition true counts.

    Returns
    -------
    pd.Series or (pd.Series, dict)
        Combined entry signals and optionally per-condition counts.
    """
    global _CACHE_HITS, _CACHE_MISSES
    if config.CACHE_GUARDRAILS.get("clear_cache_between_assets"):
        clear_indicator_cache()
    entry_rules = rules.get("entry_rules", {})
    conditions = entry_rules.get("conditions", [])
    combination_logic = entry_rules.get("combination_logic", "AND")
    if isinstance(combination_logic, dict):
        combination_logic = combination_logic.get(
            "low",
            combination_logic.get("high", combination_logic.get("options", [None])[0]),
        )
    combination_logic = str(combination_logic).upper()
    if combination_logic not in {"AND", "OR", "VOTE"}:
        raise ValueError(
            f"Invalid combination_logic '{combination_logic}'. Expected AND, OR, or VOTE."
        )

    vote_threshold = entry_rules.get("vote_threshold")
    if isinstance(vote_threshold, dict):
        vote_threshold = vote_threshold.get("low", vote_threshold.get("high"))
    nan_policy = entry_rules.get("nan_policy", config.NAN_POLICY)
    ffill_lookback = entry_rules.get("ffill_lookback", config.NAN_FFILL_LOOKBACK)
    strict_column = entry_rules.get("strict_column", True)
    if not isinstance(strict_column, bool):
        raise TypeError("strict_column must be a boolean")
    if not isinstance(nan_policy, str):
        raise TypeError("nan_policy must be a string")
    nan_policy_u = nan_policy.upper()
    if nan_policy_u not in {"FALSE", "PROPAGATE", "FORWARD_FILL"}:
        raise ValueError("nan_policy must be FALSE, PROPAGATE, or FORWARD_FILL")
    if not isinstance(ffill_lookback, int):
        raise TypeError("ffill_lookback must be an integer")
    ffill_limit = None if ffill_lookback == 0 else ffill_lookback

    active_conds = [c for c in conditions if c.get("is_active", True)]
    used_inds = {c.get("indicator", "").lower() for c in active_conds}
    missing_inds = VOLUME_INDICATORS.intersection(used_inds)
    if missing_inds and "Volume" not in ohlc_data.columns:
        affected = [
            canonical_rule_label(c)
            for c in active_conds
            if c.get("indicator", "").lower() in missing_inds
        ]
        raise ValueError(
            f"Volume column required for indicators: {sorted(missing_inds)}; "
            f"affected rules: {affected}"
        )
    n = len(active_conds)

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

    if vote_threshold is not None and not isinstance(vote_threshold, int):
        raise TypeError("vote_threshold must be an integer or None")

    if vote_threshold is not None:
        assert (
            1 <= vote_threshold <= n
        ), "vote_threshold must be between 1 and the number of active conditions"

    signals = []
    counts = {} if collect_counts else None

    for rule in conditions:
        if not rule.get("is_active", True):
            continue

        indicator_name_raw = rule.get("indicator", "")
        indicator_name = indicator_name_raw.lower()
        params = rule.get("params", {})
        condition_logic = rule.get("condition", {})
        indicator_func = INDICATOR_MAPPING.get(indicator_name)

        if not indicator_func:
            warnings.warn(
                f"Indicator '{indicator_name}' not found. Skipping rule.",
                stacklevel=2,
            )
            continue

        norm_params: dict[str, float | int | str] = {}
        for k, v in params.items():
            if (
                indicator_name == "ma_envelope"
                and k == "percent"
                and isinstance(v, (int, float))
                and not isinstance(v, bool)
            ):
                v = v / 100 if v > 1 else v
            if isinstance(v, float):
                v = round(v, 10)
            norm_params[k] = v
        params_tuple = tuple(sorted(norm_params.items()))
        key = (
            indicator_name,
            params_tuple,
            id(ohlc_data),  # different data -> different cache bucket
            tuple(ohlc_data.columns),
        )
        cache_entry = _INDICATOR_CACHE.get(key)
        if cache_entry is not None:
            _CACHE_HITS += 1
            data_ref, indicator_output = cache_entry
            if data_ref() is not ohlc_data:
                indicator_output = indicator_func(ohlc_data, **norm_params)
                _INDICATOR_CACHE[key] = (weakref.ref(ohlc_data), indicator_output)
        else:
            _CACHE_MISSES += 1
            indicator_output = indicator_func(ohlc_data, **norm_params)
            guard = config.CACHE_GUARDRAILS
            if (
                len(_INDICATOR_CACHE) < guard["MAX_CACHE_KEYS"]
                and len(ohlc_data) <= guard["MAX_CACHE_ROWS"]
            ):
                _INDICATOR_CACHE[key] = (weakref.ref(ohlc_data), indicator_output)
        condition_type = condition_logic.get("type")

        target_series = select_indicator_series(
            indicator_name, indicator_output, condition_logic, strict_column
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

        if nan_policy_u == "FORWARD_FILL":
            individual_signal = individual_signal.ffill(limit=ffill_limit)
        signals.append(
            individual_signal.fillna(False)
            if nan_policy_u == "FALSE" or nan_policy_u == "FORWARD_FILL"
            else individual_signal.astype("boolean")
        )
        if collect_counts:
            name = canonical_rule_label(rule)
            val = (
                individual_signal.fillna(False)
                if nan_policy_u != "PROPAGATE"
                else individual_signal
            )
            counts[name] = int(pd.Series(val, dtype="boolean").sum(skipna=True))

    if not signals:
        empty = pd.Series(False, index=ohlc_data.index)
        return (empty, {}) if collect_counts else empty

    if len(signals) == 1:
        single = signals[0]
        if nan_policy_u == "FORWARD_FILL":
            single = single.ffill(limit=ffill_limit).fillna(False)
        elif nan_policy_u == "FALSE":
            single = single.fillna(False)
        else:
            single = single.astype("boolean")
        return (single, counts) if collect_counts else single

    combined = _combine_signals(
        signals, combination_logic, vote_threshold, nan_policy_u, ffill_lookback
    )
    return (combined, counts) if collect_counts else combined
