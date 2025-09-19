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
import sys
import warnings
import weakref
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

import config

try:  # Preserve original config module across reloads
    _ORIGINAL_CONFIG  # type: ignore[used-before-def]
except NameError:  # pragma: no cover - executed only once
    _ORIGINAL_CONFIG = config
import indicator_contracts as contracts
import indicator_library as ind_lib  # Import our toolbox of indicators

logger = logging.getLogger(__name__)
_VOTE_LOG_SEEN = False


def _log_vote_payload(payload: Mapping[str, Any]) -> None:
    """Emit vote diagnostics without flooding INFO logs."""

    global _VOTE_LOG_SEEN
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(payload)
        _VOTE_LOG_SEEN = True
        return
    if not _VOTE_LOG_SEEN and logger.isEnabledFor(logging.INFO):
        logger.info(payload)
        _VOTE_LOG_SEEN = True


# Auto-registered indicator mapping from ``indicator_library``
INDICATOR_MAPPING = {k.lower(): v for k, v in ind_lib.INDICATOR_REGISTRY.items()}
INDICATOR_CANONICAL = {k.lower(): k.lower() for k in ind_lib.INDICATOR_REGISTRY}

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
        INDICATOR_CANONICAL[alias] = target.lower()

# Ensure direct names map to themselves for canonical lookups.
for name in list(INDICATOR_MAPPING):
    INDICATOR_CANONICAL.setdefault(name, name)

# Indicators that require a ``Volume`` column
VOLUME_INDICATORS = {"obv", "mfi", "adl", "cmf"}

# Global cache for indicator outputs keyed by
# (indicator_name, params_tuple, data_id, columns). Each cache entry stores a
# weak reference to the original DataFrame so we can verify that the cached
# result was produced for the same object even if Python reuses ``id`` values
# after garbage collection.
CacheKey = tuple[str, tuple[tuple[str, Any], ...], int, tuple[str, ...]]
CacheVal = tuple[weakref.ReferenceType[pd.DataFrame], pd.Series | pd.DataFrame]
_INDICATOR_CACHE: OrderedDict[CacheKey, CacheVal] = OrderedDict()
_CACHE_HITS = 0
_CACHE_MISSES = 0


@dataclass(frozen=True)
class CacheGuardrails:
    max_keys: int
    max_rows: int
    clear_between: bool


@dataclass(frozen=True)
class EntrySettings:
    combination_logic: str
    vote_threshold: int | None
    nan_policy: str
    nan_policy_u: str
    ffill_lookback: int
    ffill_limit: int | None
    strict_column: bool
    conditions: list[dict]


class IndicatorCallError(Exception):
    """Wrapper for errors raised during indicator evaluation."""

    def __init__(self, indicator: str, original: Exception):
        self.indicator = indicator
        msg = f"{indicator} call failed: {original.__class__.__name__}: {original}"
        super().__init__(msg)


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


def _resolve_cache_guardrails() -> CacheGuardrails:
    guardrails: dict[str, Any] = {}
    cfg_sys = sys.modules.get("config")
    if cfg_sys and hasattr(cfg_sys, "CACHE_GUARDRAILS"):
        guardrails.update(cfg_sys.CACHE_GUARDRAILS)  # noqa: B009
    if hasattr(_ORIGINAL_CONFIG, "CACHE_GUARDRAILS"):
        guardrails.update(_ORIGINAL_CONFIG.CACHE_GUARDRAILS)  # noqa: B009
    max_cache_keys = max(0, int(guardrails.get("MAX_CACHE_KEYS", 0)))
    max_cache_rows = max(0, int(guardrails.get("MAX_CACHE_ROWS", 0)))
    clear_between = bool(guardrails.get("clear_cache_between_assets"))
    return CacheGuardrails(max_cache_keys, max_cache_rows, clear_between)


def _resolve_entry_settings(rules: dict) -> EntrySettings:
    entry_rules = rules.get("entry_rules", {}) or {}
    conditions = entry_rules.get("conditions", []) or []
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

    cfg_params = sys.modules.get("config") or _ORIGINAL_CONFIG
    nan_policy = entry_rules.get(
        "nan_policy",
        (
            cfg_params.NAN_POLICY
            if hasattr(cfg_params, "NAN_POLICY")
            else config.NAN_POLICY
        ),
    )
    ffill_lookback = entry_rules.get(
        "ffill_lookback",
        (
            cfg_params.NAN_FFILL_LOOKBACK
            if hasattr(cfg_params, "NAN_FFILL_LOOKBACK")
            else config.NAN_FFILL_LOOKBACK
        ),
    )
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

    active_conditions = [c for c in conditions if c.get("is_active", True)]
    n_active = len(active_conditions)

    if n_active == 1:
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
            vote_threshold = math.ceil(n_active / 2)
        elif vote_threshold < 1:
            vote_threshold = 1
            warnings.warn(
                "Normalized vote_threshold to 1 for VOTE combination",
                RuntimeWarning,
                stacklevel=2,
            )
        if vote_threshold > n_active:
            warnings.warn(
                "vote_threshold exceeds active conditions; clamped to n",
                RuntimeWarning,
                stacklevel=2,
            )
            vote_threshold = n_active

    if vote_threshold is not None and not isinstance(vote_threshold, int):
        raise TypeError("vote_threshold must be an integer or None")
    if vote_threshold is not None:
        assert (
            1 <= vote_threshold <= max(1, n_active)
        ), "vote_threshold must be between 1 and the number of active conditions"

    return EntrySettings(
        combination_logic=combination_logic,
        vote_threshold=vote_threshold,
        nan_policy=nan_policy,
        nan_policy_u=nan_policy_u,
        ffill_lookback=ffill_lookback,
        ffill_limit=ffill_limit,
        strict_column=strict_column,
        conditions=active_conditions,
    )


def _normalize_indicator_params(
    indicator_name: str, params: Mapping[str, Any]
) -> dict[str, Any]:
    norm: dict[str, Any] = {}
    for key, value in params.items():
        val = value
        if (
            indicator_name == "ma_envelope"
            and key == "percent"
            and isinstance(val, (int, float))
            and not isinstance(val, bool)
        ):
            val = val / 100 if val > 1 else val
        if isinstance(val, float):
            val = round(val, 10)
        norm[key] = val
    return norm


def _invoke_indicator(
    indicator_func: Callable[..., Any],
    indicator_key: str,
    contract_name: str,
    ohlc_data: pd.DataFrame,
    params: Mapping[str, Any],
) -> pd.Series | pd.DataFrame:
    try:
        raw_output = indicator_func(ohlc_data, **params)
    except Exception as exc:  # pragma: no cover - defensive
        raise IndicatorCallError(indicator_key, exc) from exc
    return contracts.normalize_output(
        contract_name, raw_output, params, index=ohlc_data.index
    )


def _compute_indicator_output(
    indicator_key: str,
    contract_name: str,
    indicator_func: Callable[..., Any],
    ohlc_data: pd.DataFrame,
    params: Mapping[str, Any],
    guardrails: CacheGuardrails,
) -> pd.Series | pd.DataFrame:
    global _CACHE_HITS, _CACHE_MISSES

    params_tuple = tuple(sorted(params.items()))
    cache_key: CacheKey = (
        indicator_key,
        params_tuple,
        id(ohlc_data),
        tuple(ohlc_data.columns),
    )
    use_cache = guardrails.max_keys > 0 and not guardrails.clear_between
    cache_entry = _INDICATOR_CACHE.get(cache_key) if use_cache else None
    if cache_entry is not None:
        _CACHE_HITS += 1
        data_ref, cached_output = cache_entry
        if data_ref() is ohlc_data:
            indicator_output = cached_output
        else:
            indicator_output = _invoke_indicator(
                indicator_func, indicator_key, contract_name, ohlc_data, params
            )
            if use_cache and len(ohlc_data) <= guardrails.max_rows:
                _INDICATOR_CACHE[cache_key] = (weakref.ref(ohlc_data), indicator_output)
            else:
                _INDICATOR_CACHE.pop(cache_key, None)
        if use_cache:
            _INDICATOR_CACHE.move_to_end(cache_key)
        return indicator_output

    _CACHE_MISSES += 1
    indicator_output = _invoke_indicator(
        indicator_func, indicator_key, contract_name, ohlc_data, params
    )
    if use_cache and len(ohlc_data) <= guardrails.max_rows:
        _INDICATOR_CACHE[cache_key] = (weakref.ref(ohlc_data), indicator_output)
        _INDICATOR_CACHE.move_to_end(cache_key)
        if len(_INDICATOR_CACHE) > guardrails.max_keys:
            _INDICATOR_CACHE.popitem(last=False)
    return indicator_output


def _normalise_band_hint(band: Any, condition_type: str | None) -> str | None:
    if band is not None:
        value = str(band).lower()
    else:
        cond = (condition_type or "").lower()
        if "upper" in cond:
            value = "upper"
        elif "lower" in cond:
            value = "lower"
        elif "middle" in cond or "basis" in cond:
            value = "middle"
        else:
            value = None
    if value in {"mid", "basis"}:
        value = "middle"
    if value in {"upper", "lower", "middle"}:
        return value
    return None


def _select_with_fallback(
    output: pd.DataFrame, strict: bool, message: str
) -> pd.Series:
    columns = list(output.columns)
    msg = f"{message}; available: {columns}"
    if output.shape[1] == 0:
        raise KeyError(msg + "; set strict_column=False to allow fallback")
    if strict:
        raise KeyError(msg + "; set strict_column=False to allow fallback")
    warnings.warn(msg + "; using first available column", stacklevel=2)
    return output.iloc[:, 0]


def _build_condition_signal(
    ohlc_data: pd.DataFrame,
    indicator_series: pd.Series,
    condition_logic: Mapping[str, Any],
    nan_policy_u: str,
    ffill_limit: int | None,
) -> pd.Series:
    condition_type = str(condition_logic.get("type") or "")
    if "price" in condition_type:
        signal = _generate_signal(ohlc_data, indicator_series, condition_logic)
    elif "indicator" in condition_type:
        signal = _generate_signal_from_value(indicator_series, condition_logic)
    else:
        signal = pd.Series(False, index=ohlc_data.index)
    if nan_policy_u == "FORWARD_FILL":
        signal = signal.ffill(limit=ffill_limit)
    return signal


def _evaluate_rule(
    rule: dict,
    ohlc_data: pd.DataFrame,
    settings: EntrySettings,
    guardrails: CacheGuardrails,
    collect_counts: bool,
) -> tuple[pd.Series | None, tuple[str, int] | None]:
    indicator_raw = str(rule.get("indicator", "")).lower()
    indicator_func = INDICATOR_MAPPING.get(indicator_raw)
    if indicator_func is None:
        warnings.warn(
            f"Indicator '{indicator_raw}' not found. Skipping rule.",
            stacklevel=2,
        )
        return None, None

    canonical = INDICATOR_CANONICAL.get(indicator_raw, indicator_raw)
    params = rule.get("params", {}) or {}
    norm_params = _normalize_indicator_params(canonical, params)
    indicator_output = _compute_indicator_output(
        indicator_raw,
        canonical,
        indicator_func,
        ohlc_data,
        norm_params,
        guardrails,
    )

    condition_logic = rule.get("condition", {}) or {}
    target_series = select_indicator_series(
        canonical,
        indicator_output,
        condition_logic,
        settings.strict_column,
        norm_params,
    )
    signal = _build_condition_signal(
        ohlc_data,
        target_series,
        condition_logic,
        settings.nan_policy_u,
        settings.ffill_limit,
    )

    if not collect_counts:
        return signal, None

    name = canonical_rule_label(rule)
    values = signal.to_numpy(dtype=float, copy=False)
    if settings.nan_policy_u != "PROPAGATE":
        values = np.nan_to_num(values, nan=0.0)
    count = int(np.nansum(values))
    return signal, (name, count)


def select_indicator_series(
    indicator_name: str,
    indicator_output: pd.Series | pd.DataFrame,
    condition_logic: Mapping[str, Any],
    strict_column: bool,
    indicator_params: Mapping[str, Any] | None = None,
) -> pd.Series:
    """Return the appropriate Series from ``indicator_output``."""

    if isinstance(indicator_output, pd.Series):
        return indicator_output

    rule_strict = condition_logic.get("strict_column", strict_column)
    if not isinstance(rule_strict, bool):
        raise TypeError("strict_column must be a boolean")

    columns = list(indicator_output.columns)
    if not columns:
        raise KeyError("Indicator produced no columns; available: []")

    col_hint = condition_logic.get("column")
    if col_hint:
        if col_hint in columns:
            return indicator_output[col_hint]
        try:
            pattern = re.compile(str(col_hint))
        except re.error:
            pattern = None
        if pattern is not None:
            for col in columns:
                if pattern.search(col):
                    return indicator_output[col]
        return _select_with_fallback(
            indicator_output,
            rule_strict,
            f"Requested column '{col_hint}' not found",
        )

    schema = contracts.describe_output(indicator_name, indicator_params or {})
    band = _normalise_band_hint(
        condition_logic.get("band"), condition_logic.get("type")
    )
    if band:
        target = schema.roles.get(band)
        if target and target in columns:
            return indicator_output[target]
        if target:
            return _select_with_fallback(
                indicator_output,
                rule_strict,
                f"{band.capitalize()} band not found in {indicator_name} output; expected column '{target}'",
            )
        return _select_with_fallback(
            indicator_output,
            rule_strict,
            f"{band.capitalize()} band not supported for {indicator_name}",
        )

    preferred: list[str] = []
    if schema.default and schema.default in columns:
        preferred.append(schema.default)
    for candidate in schema.priority:
        if candidate in columns and candidate not in preferred:
            preferred.append(candidate)
    if preferred:
        return indicator_output[preferred[0]]

    if columns:
        if rule_strict:
            warnings.warn(
                f"No default column defined for {indicator_name}; using first available column",
                stacklevel=2,
            )
        return indicator_output.iloc[:, 0]

    return _select_with_fallback(
        indicator_output,
        rule_strict,
        f"No columns returned for {indicator_name}",
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
        prepared: list[np.ndarray] = []
        for series in signals:
            filled = series.ffill(limit=limit)
            values = filled.to_numpy(dtype=float, copy=False)
            if np.isnan(values).any():
                values = np.nan_to_num(values, nan=0.0)
            prepared.append(values.astype(bool))
        treat_nan_as_false = True
    elif policy == "FALSE":
        prepared = []
        for series in signals:
            values = series.to_numpy(dtype=float, copy=False)
            if np.isnan(values).any():
                values = np.nan_to_num(values, nan=0.0)
            prepared.append(values.astype(bool))
        treat_nan_as_false = True
    elif policy == "PROPAGATE":
        prepared = [series.to_numpy(dtype=float, copy=False) for series in signals]
        treat_nan_as_false = False
    else:
        raise ValueError("nan_policy must be FALSE, PROPAGATE, or FORWARD_FILL")

    if treat_nan_as_false:
        if combination_logic == "AND":
            combined = prepared[0].copy()
            for arr in prepared[1:]:
                np.logical_and(combined, arr, out=combined)
            return pd.Series(combined, index=idx, name=name)
        if combination_logic == "OR":
            combined = prepared[0].copy()
            for arr in prepared[1:]:
                np.logical_or(combined, arr, out=combined)
            return pd.Series(combined, index=idx, name=name)
        if combination_logic == "VOTE":
            M = len(prepared)
            threshold = (
                vote_threshold if vote_threshold is not None else math.ceil(M / 2)
            )
            if threshold < 1 or threshold > M:
                raise ValueError(
                    "vote_threshold must be between 1 and the number of active conditions"
                )
            payload = {"logic": "VOTE", "M": M, "k": threshold, "nan_policy": policy}
            _log_vote_payload(payload)
            votes = np.zeros(prepared[0].shape, dtype=np.int16)
            for arr in prepared:
                votes += arr
            return pd.Series(votes >= threshold, index=idx)

    arrays = prepared
    if combination_logic == "AND":
        has_nan = np.zeros(arrays[0].shape, dtype=bool)
        has_false = np.zeros_like(has_nan)
        for arr in arrays:
            np.logical_or(has_nan, np.isnan(arr), out=has_nan)
            np.logical_or(has_false, arr <= 0.0, out=has_false)
        outcome = np.full(arrays[0].shape, True, dtype=object)
        outcome[has_false] = False
        nan_mask = ~has_false & has_nan
        outcome[nan_mask] = pd.NA
        return pd.Series(pd.array(outcome, dtype="boolean"), index=idx, name=name)
    if combination_logic == "OR":
        has_nan = np.zeros(arrays[0].shape, dtype=bool)
        has_true = np.zeros_like(has_nan)
        for arr in arrays:
            np.logical_or(has_nan, np.isnan(arr), out=has_nan)
            np.logical_or(has_true, arr > 0.5, out=has_true)
        outcome = np.full(arrays[0].shape, False, dtype=object)
        outcome[has_true] = True
        nan_mask = ~has_true & has_nan
        outcome[nan_mask] = pd.NA
        return pd.Series(pd.array(outcome, dtype="boolean"), index=idx, name=name)
    if combination_logic == "VOTE":
        M = len(arrays)
        threshold = vote_threshold if vote_threshold is not None else math.ceil(M / 2)
        if threshold < 1 or threshold > M:
            raise ValueError(
                "vote_threshold must be between 1 and the number of active conditions"
            )
        payload = {"logic": "VOTE", "M": M, "k": threshold, "nan_policy": policy}
        _log_vote_payload(payload)
        votes = np.zeros(arrays[0].shape, dtype=float)
        for arr in arrays:
            votes += arr
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
    guardrails = _resolve_cache_guardrails()
    if guardrails.clear_between:
        clear_indicator_cache()

    settings = _resolve_entry_settings(rules)
    active_conditions = settings.conditions
    used_inds = {c.get("indicator", "").lower() for c in active_conditions}
    missing_inds = VOLUME_INDICATORS.intersection(used_inds)
    if missing_inds and "Volume" not in ohlc_data.columns:
        affected = [
            canonical_rule_label(c)
            for c in active_conditions
            if c.get("indicator", "").lower() in missing_inds
        ]
        raise ValueError(
            f"Volume column required for indicators: {sorted(missing_inds)}; "
            f"affected rules: {affected}"
        )

    signals: list[pd.Series] = []
    counts: dict[str, int] | None = {} if collect_counts else None

    for rule in active_conditions:
        signal, count_info = _evaluate_rule(
            rule, ohlc_data, settings, guardrails, collect_counts
        )
        if signal is None:
            continue
        signals.append(signal)
        if collect_counts and count_info is not None and counts is not None:
            counts[count_info[0]] = count_info[1]

    if not signals:
        empty = pd.Series(False, index=ohlc_data.index)
        result = (empty, counts or {}) if collect_counts else empty
    elif len(signals) == 1:
        single = signals[0]
        if settings.nan_policy_u in {"FORWARD_FILL", "FALSE"}:
            single = single.fillna(False)
        else:
            single = single.astype("boolean")
        result = (single, counts) if collect_counts else single
    else:
        combined = _combine_signals(
            signals,
            settings.combination_logic,
            settings.vote_threshold,
            settings.nan_policy_u,
            settings.ffill_lookback,
        )
        result = (combined, counts) if collect_counts else combined

    if guardrails.clear_between:
        clear_indicator_cache()

    return result
