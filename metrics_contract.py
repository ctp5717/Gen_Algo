"""Metric aliasing and fallback computations for portfolio statistics."""

from __future__ import annotations

import logging
import re
import threading
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)

__all__ = [
    "METRIC_ALIASES",
    "MetricsAliasError",
    "assert_metric_aliases",
    "compute_fallbacks",
    "evaluate_metrics",
    "format_mapping",
    "resolve_metrics",
    "reset_cache",
]

# Canonical metrics and the aliases exposed by vectorbt/QuantStats across versions.
METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "sortino": (
        "sortino",
        "sortino_ratio",
        "Sortino Ratio",
        "Sortino",
        "QS Sortino Ratio",
    ),
    "profit_factor": (
        "profit_factor",
        "Profit Factor",
        "PF",
        "ProfitFactor",
    ),
    "max_drawdown": (
        "max_drawdown",
        "Max Drawdown",
        "Max Drawdown [%]",
        "Max Drawdown %",
        "Max Drawdown ( % )",
    ),
    "total_return": (
        "total_return",
        "Total Return",
        "Total Return [%]",
        "Return [%]",
        "Cumulative Returns [%]",
    ),
}

_KEY_SANITISE_RE = re.compile(r"[^0-9a-zA-Z]+")


def _key_norm(key: Any) -> str:
    """Return a normalised representation for safe alias comparison."""

    text = str(key)
    collapsed = _KEY_SANITISE_RE.sub("_", text.lower())
    collapsed = re.sub(r"_+", "_", collapsed)
    return collapsed.strip("_")


def _merge_alias_extension() -> None:
    """Merge user-provided alias patches from :mod:`config`."""

    extra = getattr(config, "METRIC_ALIASES_EXT", None)
    if not extra:
        return
    if isinstance(extra, Mapping):
        items = extra.items()
    else:  # pragma: no cover - defensive fallback for bad input
        try:
            items = dict(extra).items()
        except Exception:
            return
    for metric, aliases in items:
        if not aliases:
            continue
        existing = list(METRIC_ALIASES.get(metric, ()))
        values = [aliases] if isinstance(aliases, str) else list(aliases)
        for alias in values:
            alias_str = str(alias)
            if alias_str not in existing:
                existing.append(alias_str)
        METRIC_ALIASES[metric] = tuple(existing)


_merge_alias_extension()

_PERCENTAGE_METRICS = {"max_drawdown", "total_return"}
_CANONICAL_ORDER = tuple(METRIC_ALIASES.keys())
_PREFERRED_ALIASES = {key: METRIC_ALIASES[key][0] for key in _CANONICAL_ORDER}
# ``_ALIAS_CACHE`` is keyed by a provider signature so that alias discovery is
# isolated per stats source.  The cache is guarded by ``_ALIAS_CACHE_LOCK`` for
# thread safety; multiprocessing workers obtain a copy-on-write view which is
# inherently safe because each process has its own module state.
_ALIAS_CACHE_LOCK = threading.Lock()
_ALIAS_CACHE: dict[str, dict[str, str | None]] = {}
_DISCOVERY_LOGGED: set[str] = set()
_MAPPING_LOGGED: set[str] = set()


class MetricsAliasError(RuntimeError):
    """Raised when required metric aliases are missing in preflight."""


def reset_cache() -> None:
    """Reset cached alias selections (primarily for unit tests)."""

    with _ALIAS_CACHE_LOCK:
        _ALIAS_CACHE.clear()
        _DISCOVERY_LOGGED.clear()
        _MAPPING_LOGGED.clear()


def _provider_signature(portfolio: Any) -> str:
    """Return a simple signature identifying the stats provider."""

    cls = type(portfolio)
    module = getattr(cls, "__module__", "") or ""
    name = getattr(cls, "__qualname__", getattr(cls, "__name__", "")) or str(cls)
    return f"{module}:{name}"


def _normalise_stats(stats: Any) -> tuple[dict[str, Any], dict[str, str]]:
    """Convert stats output into a mapping and a lookup table."""

    raw: dict[str, Any]
    if isinstance(stats, pd.Series):
        raw = stats.to_dict()
    elif isinstance(stats, Mapping):
        raw = dict(stats)
    elif hasattr(stats, "to_dict"):
        try:
            raw = dict(stats.to_dict())
        except Exception:  # pragma: no cover - defensive
            raw = {}
    else:
        try:
            raw = dict(stats)
        except Exception:  # pragma: no cover - defensive
            raw = {}

    result: dict[str, Any] = {}
    lookup: dict[str, str] = {}
    for key, value in raw.items():
        key_str = str(key)
        collapsed = value
        if isinstance(value, pd.Series):
            cleaned = value.dropna()
            if cleaned.size == 1:
                collapsed = cleaned.iloc[0]
        elif isinstance(value, Mapping):
            try:
                if len(value) == 1:
                    collapsed = next(iter(value.values()))
            except TypeError:  # pragma: no cover - defensive
                pass
        result[key_str] = collapsed
        lookup.setdefault(key_str, key_str)
        norm = _key_norm(key_str)
        if norm:
            lookup.setdefault(norm, key_str)
        if isinstance(key, (tuple, list)):
            for part in key:
                part_norm = _key_norm(part)
                if part_norm:
                    lookup.setdefault(part_norm, key_str)
    return result, lookup


def _coerce_series(data: Any) -> pd.Series | None:
    """Convert arbitrary iterables into a clean pandas Series of floats."""

    if data is None:
        return None
    if isinstance(data, pd.Series):
        series = data
    else:
        try:
            series = pd.Series(data)
        except Exception:
            return None
    series = pd.to_numeric(series, errors="coerce")
    series = series.replace([np.inf, -np.inf], np.nan).dropna()
    return series if not series.empty else None


def _get_equity_series(portfolio: Any) -> pd.Series | None:
    value_attr = getattr(portfolio, "value", None)
    equity = None
    if callable(value_attr):
        try:
            equity = value_attr()
        except Exception:  # pragma: no cover - defensive
            equity = None
    elif value_attr is not None:
        equity = value_attr
    return _coerce_series(equity)


def _get_returns_series(portfolio: Any) -> pd.Series | None:
    candidates: list[Any] = []
    returns_attr = getattr(portfolio, "returns", None)
    if callable(returns_attr):
        try:
            candidates.append(returns_attr())
        except Exception:  # pragma: no cover - defensive
            pass
    elif returns_attr is not None:
        candidates.append(returns_attr)
    returns_method = getattr(portfolio, "get_returns", None)
    if callable(returns_method):  # pragma: no cover - compatibility
        try:
            candidates.append(returns_method())
        except Exception:
            pass
    for candidate in candidates:
        series = _coerce_series(candidate)
        if series is not None:
            return series
    equity = _get_equity_series(portfolio)
    if equity is not None:
        returns = equity.pct_change().dropna()
        return returns if not returns.empty else None
    return None


def _to_pct(value: Any) -> Any:
    """Normalise fractional inputs into percentage units."""

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return value
    if np.isnan(numeric):
        return numeric
    if abs(numeric) <= 1:
        return numeric * 100.0
    return numeric


def _build_metric_dict(
    alias_map: Mapping[str, str | None],
    stats_dict: Mapping[str, Any],
) -> dict[str, Any]:
    """Translate provider-specific stats into canonical keys."""

    result: dict[str, Any] = {}
    for metric in _CANONICAL_ORDER:
        alias = alias_map.get(metric)
        value = stats_dict.get(alias) if alias else None
        if metric in _PERCENTAGE_METRICS and value is not None:
            value = _to_pct(value)
        result[metric] = value
    return result


def _discover_aliases(
    portfolio: Any,
    signature: str | None = None,
) -> tuple[dict[str, str | None], dict[str, Any], dict[str, str]]:
    log_level = None
    if signature:
        with _ALIAS_CACHE_LOCK:
            first_log = signature not in _DISCOVERY_LOGGED
            if first_log:
                _DISCOVERY_LOGGED.add(signature)
        if signature:
            log_level = logging.INFO if first_log else logging.DEBUG
            logger.log(
                log_level,
                "Discovering metric aliases via full stats() for provider %s",
                signature,
            )

    stats_raw = portfolio.stats()
    stats_all, lookup = _normalise_stats(stats_raw)
    alias_map: dict[str, str | None] = {}
    for metric, aliases in METRIC_ALIASES.items():
        resolved = None
        for candidate in aliases:
            resolved = lookup.get(_key_norm(candidate))
            if resolved:
                break
        alias_map[metric] = resolved
    return alias_map, stats_all, lookup


def resolve_metrics(portfolio: Any) -> tuple[dict[str, Any], dict[str, str | None]]:
    """Resolve canonical metrics from a portfolio, handling alias drift."""

    signature = _provider_signature(portfolio)
    with _ALIAS_CACHE_LOCK:
        alias_map = dict(_ALIAS_CACHE.get(signature, {}))
    stats_dict: dict[str, Any] | None = None
    stats_lookup: dict[str, str] | None = None

    if not alias_map:
        preferred = [_PREFERRED_ALIASES[m] for m in _CANONICAL_ORDER]
        try:
            stats_obj = portfolio.stats(metrics=preferred)
        except KeyError as exc:
            logger.debug(
                "Provider %s reported missing metrics during preferred fetch: %s",
                signature,
                exc,
            )
            alias_map, stats_dict, stats_lookup = _discover_aliases(
                portfolio, signature
            )
        except Exception as exc:
            logger.warning(
                "Provider %s stats(metrics=%s) raised %s; rediscovering aliases",
                signature,
                preferred,
                exc,
            )
            alias_map, stats_dict, stats_lookup = _discover_aliases(
                portfolio, signature
            )
        else:
            stats_dict, stats_lookup = _normalise_stats(stats_obj)
            if all(alias in stats_lookup for alias in preferred if alias):
                alias_map = {
                    metric: stats_lookup[preferred[idx]]
                    for idx, metric in enumerate(_CANONICAL_ORDER)
                }
            else:
                alias_map, stats_dict, stats_lookup = _discover_aliases(
                    portfolio, signature
                )
        with _ALIAS_CACHE_LOCK:
            _ALIAS_CACHE[signature] = dict(alias_map)

    requested = [
        alias for alias in (alias_map.get(m) for m in _CANONICAL_ORDER) if alias
    ]

    if stats_dict is None or stats_lookup is None:
        stats_dict = {}
        stats_lookup = {}
    if requested and not all(alias in stats_lookup for alias in requested):
        stats_dict, stats_lookup, alias_map = _refresh_stats(
            portfolio, signature, alias_map, requested
        )

    metrics = _build_metric_dict(alias_map, stats_dict)
    return metrics, dict(alias_map)


def _refresh_stats(
    portfolio: Any,
    signature: str,
    alias_map: dict[str, str | None],
    requested: list[str],
) -> tuple[dict[str, Any], dict[str, str], dict[str, str | None]]:
    """Fetch provider stats, re-discovering aliases if required."""

    stats_dict: dict[str, Any] = {}
    stats_lookup: dict[str, str] = {}
    current_alias_map = dict(alias_map)
    current_requested = list(requested)
    attempts = 0

    while True:
        attempts += 1
        discovery_needed = False
        try:
            stats_obj = (
                portfolio.stats(metrics=current_requested)
                if current_requested
                else {}
            )
        except KeyError as exc:
            discovery_needed = True
            logger.debug(
                "Provider %s reported missing metrics during refresh: %s",
                signature,
                exc,
            )
        except Exception as exc:
            discovery_needed = True
            logger.warning(
                "Provider %s stats(metrics=%s) raised %s during refresh; rediscovering",
                signature,
                current_requested,
                exc,
            )
        else:
            stats_dict, stats_lookup = _normalise_stats(stats_obj)
            if all(alias in stats_lookup for alias in current_requested):
                return stats_dict, stats_lookup, current_alias_map
            discovery_needed = True

        if attempts >= 2 or not discovery_needed:
            break

        with _ALIAS_CACHE_LOCK:
            _ALIAS_CACHE.pop(signature, None)
        current_alias_map, stats_dict, stats_lookup = _discover_aliases(
            portfolio, signature
        )
        with _ALIAS_CACHE_LOCK:
            _ALIAS_CACHE[signature] = dict(current_alias_map)
        current_requested = [
            alias
            for alias in (current_alias_map.get(m) for m in _CANONICAL_ORDER)
            if alias
        ]
        if not current_requested:
            return stats_dict, stats_lookup, current_alias_map
        if stats_lookup and all(alias in stats_lookup for alias in current_requested):
            return stats_dict, stats_lookup, current_alias_map

    return stats_dict, stats_lookup, current_alias_map


def _needs_metric(stats: Mapping[str, Any], metric: str) -> bool:
    value = stats.get(metric)
    if value is None:
        return True
    try:
        return bool(np.isnan(value))
    except TypeError:
        return False


def compute_fallbacks(
    portfolio: Any,
    stats: Mapping[str, Any],
    rf: float = 0.0,
    ann_factor: int = 252,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Fill in missing metrics using raw returns when available."""

    resolved = dict(stats)
    computed: dict[str, str] = {}

    returns = _get_returns_series(portfolio)
    if returns is None or returns.empty:
        return resolved, computed

    excess = returns - (rf / float(ann_factor) if ann_factor else rf)

    if _needs_metric(resolved, "sortino"):
        downside = excess[excess < 0]
        downside_sq = (downside**2).mean()
        if downside_sq and not np.isnan(downside_sq):
            downside_dev = float(np.sqrt(downside_sq))
            if downside_dev > 0:
                mean_excess = float(excess.mean())
                scale = np.sqrt(float(ann_factor)) if ann_factor else 1.0
                resolved["sortino"] = mean_excess / downside_dev * scale
                computed["sortino"] = "computed"

    if _needs_metric(resolved, "profit_factor"):
        gains = float(returns[returns > 0].sum())
        losses = float(-returns[returns < 0].sum())
        if losses > 0:
            resolved["profit_factor"] = gains / losses if losses else np.inf
            computed["profit_factor"] = "computed"
        elif gains > 0:
            resolved["profit_factor"] = np.inf
            computed["profit_factor"] = "computed"

    cumulative = (1 + returns).cumprod()

    if _needs_metric(resolved, "total_return"):
        total_return = float(cumulative.iloc[-1] - 1)
        resolved["total_return"] = _to_pct(total_return)
        computed["total_return"] = "computed"

    if _needs_metric(resolved, "max_drawdown"):
        running_max = cumulative.cummax()
        drawdown = cumulative / running_max - 1.0
        min_dd = float(drawdown.min())
        if not np.isnan(min_dd):
            resolved["max_drawdown"] = _to_pct(abs(min_dd))
            computed["max_drawdown"] = "computed"

    return resolved, computed


def _missing_metrics(stats: Mapping[str, Any]) -> list[str]:
    missing: list[str] = []
    for metric in _CANONICAL_ORDER:
        value = stats.get(metric)
        if value is None:
            missing.append(metric)
            continue
        try:
            if np.isnan(value):
                missing.append(metric)
        except TypeError:
            continue
    return missing


def format_mapping(
    metric_sources: Mapping[str, str | None],
) -> str:
    """Return a stable string describing metric sourcing."""

    parts: list[str] = []
    for metric in _CANONICAL_ORDER:
        source = metric_sources.get(metric)
        if not source:
            source = "missing"
        parts.append(f"{metric}→{source}")
    return ", ".join(parts)


def assert_metric_aliases(portfolio: Any) -> dict[str, str | None]:
    """Validate that each canonical metric is obtainable from the portfolio.

    When ``config.METRICS_PREFLIGHT`` is absent the defaults are
    ``mode="warn"`` and ``missing_threshold=0``.
    """

    signature = _provider_signature(portfolio)
    alias_map, _, _ = _discover_aliases(portfolio, signature)
    with _ALIAS_CACHE_LOCK:
        _ALIAS_CACHE[signature] = dict(alias_map)

    missing = [metric for metric, alias in alias_map.items() if not alias]
    mapping_summary = format_mapping(
        {metric: alias or "missing" for metric, alias in alias_map.items()}
    )

    should_log = False
    with _ALIAS_CACHE_LOCK:
        if signature not in _MAPPING_LOGGED:
            _MAPPING_LOGGED.add(signature)
            should_log = True
    if should_log:
        logger.info("Metrics mapping for %s: %s", signature, mapping_summary)

    settings = getattr(config, "METRICS_PREFLIGHT", {})
    mode = str(settings.get("mode", "warn")).lower()
    threshold = int(settings.get("missing_threshold", 0))

    if missing and len(missing) > threshold:
        message = f"Missing metric aliases for {signature}: {', '.join(missing)}"
        if mode == "fail":
            raise MetricsAliasError(message)
        logger.warning(message)
    elif missing:
        logger.warning(
            "Missing metric aliases for %s: %s",
            signature,
            ", ".join(missing),
        )

    return alias_map


def evaluate_metrics(
    portfolio: Any,
) -> tuple[dict[str, Any], dict[str, str], list[str]]:
    """Convenience wrapper returning metrics, sources and missing keys."""

    metrics, alias_map = resolve_metrics(portfolio)
    metrics, computed = compute_fallbacks(portfolio, metrics)
    missing = _missing_metrics(metrics)
    sources: dict[str, str] = {}
    for metric in _CANONICAL_ORDER:
        if metric in computed:
            sources[metric] = "computed"
        else:
            alias = alias_map.get(metric)
            if alias:
                sources[metric] = alias
            elif metric in missing:
                sources[metric] = "missing"
    return metrics, sources, missing
