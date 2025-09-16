"""Metric aliasing and fallback computations for portfolio statistics."""

from __future__ import annotations

import logging
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
        "Sortino Ratio",
        "sortino",
        "sortino_ratio",
        "Sortino",
        "QS Sortino Ratio",
    ),
    "profit_factor": (
        "Profit Factor",
        "profit_factor",
        "PF",
        "ProfitFactor",
    ),
    "max_drawdown": (
        "Max Drawdown [%]",
        "Max Drawdown",
        "max_drawdown",
        "Max Drawdown %",
        "Max Drawdown ( % )",
    ),
    "total_return": (
        "Total Return [%]",
        "Total Return",
        "Return [%]",
        "total_return",
        "Cumulative Returns [%]",
    ),
}

_PERCENTAGE_METRICS = {"max_drawdown", "total_return"}
_CANONICAL_ORDER = tuple(METRIC_ALIASES.keys())
_PREFERRED_ALIASES = {key: METRIC_ALIASES[key][0] for key in _CANONICAL_ORDER}
_ALIAS_CACHE: dict[str, str | None] | None = None


class MetricsAliasError(RuntimeError):
    """Raised when required metric aliases are missing in preflight."""


def reset_cache() -> None:
    """Reset cached alias selections (primarily for unit tests)."""

    global _ALIAS_CACHE
    _ALIAS_CACHE = None


def _normalise_stats(stats: Any) -> dict[str, Any]:
    """Convert stats output (Series, dict, DataFrame) into a mapping."""

    if isinstance(stats, pd.Series):
        return stats.to_dict()
    if isinstance(stats, Mapping):
        return dict(stats)
    if hasattr(stats, "to_dict"):
        try:
            return dict(stats.to_dict())
        except Exception:  # pragma: no cover - defensive
            pass
    try:
        return dict(stats)
    except Exception:  # pragma: no cover - defensive
        return {}


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


def _discover_aliases(portfolio: Any) -> tuple[dict[str, str | None], dict[str, Any]]:
    stats_all = _normalise_stats(portfolio.stats())
    alias_map: dict[str, str | None] = {}
    for metric, aliases in METRIC_ALIASES.items():
        alias = next((name for name in aliases if name in stats_all), None)
        alias_map[metric] = alias
    return alias_map, stats_all


def resolve_metrics(portfolio: Any) -> tuple[dict[str, Any], dict[str, str | None]]:
    """Resolve canonical metrics from a portfolio, handling alias drift."""

    global _ALIAS_CACHE
    stats_dict: dict[str, Any] | None = None

    if _ALIAS_CACHE is None:
        preferred = [_PREFERRED_ALIASES[m] for m in _CANONICAL_ORDER]
        try:
            stats_obj = portfolio.stats(metrics=preferred)
        except Exception:
            alias_map, stats_dict = _discover_aliases(portfolio)
        else:
            stats_dict = _normalise_stats(stats_obj)
            alias_map = {
                metric: preferred[idx] for idx, metric in enumerate(_CANONICAL_ORDER)
            }
        _ALIAS_CACHE = alias_map
    else:
        alias_map = dict(_ALIAS_CACHE)

    requested = [
        alias for alias in (alias_map.get(m) for m in _CANONICAL_ORDER) if alias
    ]
    if stats_dict is None:
        stats_obj = portfolio.stats(metrics=requested) if requested else {}
        stats_dict = _normalise_stats(stats_obj)
    metrics = _build_metric_dict(alias_map, stats_dict)
    return metrics, dict(alias_map)


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
    """Validate that each canonical metric is obtainable from the portfolio."""

    global _ALIAS_CACHE
    alias_map, _ = _discover_aliases(portfolio)
    _ALIAS_CACHE = alias_map

    missing = [metric for metric, alias in alias_map.items() if not alias]
    mapping_summary = format_mapping(
        {metric: alias or "missing" for metric, alias in alias_map.items()}
    )
    logger.info("Metrics mapping: %s", mapping_summary)

    settings = getattr(config, "METRICS_PREFLIGHT", {})
    mode = str(settings.get("mode", "warn")).lower()
    threshold = int(settings.get("missing_threshold", 0))

    if missing and len(missing) > threshold:
        message = f"Missing metric aliases: {', '.join(missing)}"
        if mode == "fail":
            raise MetricsAliasError(message)
        logger.warning(message)
    elif missing:
        logger.warning("Missing metric aliases: %s", ", ".join(missing))

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
