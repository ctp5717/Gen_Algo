"""Worker-side evaluation helpers for multi-asset fitness."""

from __future__ import annotations

import logging
import traceback
import time
from typing import Any

import pandas as pd

try:  # pragma: no cover - optional heavy dependency
    import vectorbt as vbt
except Exception:  # pragma: no cover - fallback to stub
    from vbt_stub import Portfolio
    from vbt_stub import __file__ as _vbt_file
    from vbt_stub import __version__ as _vbt_ver

    vbt = type("vectorbt", (), {})()
    vbt.Portfolio = Portfolio
    vbt.__file__ = _vbt_file
    vbt.__version__ = _vbt_ver

import config
import metrics_contract
import strategy_engine as engine
from data_registry import registry
from params_resolver import inject_genes_into_rules
from portfolio_utils import extract_exit_params

logger = logging.getLogger(__name__)

_WORKER_STATE = {
    "metrics_preflight_done": False,
    "metric_mapping_logged": False,
}


def warm_up() -> None:
    """Best-effort warm-up executed in worker initialisation."""

    try:
        config.initialize_config()
    except Exception:  # pragma: no cover - best effort
        pass

    try:
        _ = vbt.Portfolio
    except Exception:  # pragma: no cover - guard only
        pass


def _evaluate_candidate(
    ohlc: pd.DataFrame,
    rules: dict[str, Any],
    settings: dict[str, Any],
) -> dict[str, Any]:
    entries, signal_counts = engine.process_strategy_rules(
        ohlc, rules, collect_counts=True
    )

    exit_rules = rules.get("exit_rules", {})
    time_exit, sl_stop, sl_trail, tp_stop = extract_exit_params(
        entries, exit_rules, config.MAX_HOLD_PERIOD
    )

    portfolio = vbt.Portfolio.from_signals(
        close=ohlc["Close"],
        entries=entries,
        exits=time_exit,
        sl_stop=sl_stop,
        tp_stop=tp_stop,
        sl_trail=sl_trail,
        fees=config.FEES,
        freq=config.to_pandas_freq(config.TIMEFRAME),
    )

    signature = metrics_contract._provider_signature(portfolio)

    if not _WORKER_STATE["metrics_preflight_done"]:
        try:
            metrics_contract.assert_metric_aliases(portfolio)
        except Exception as exc:  # pragma: no cover - warning path
            logger.warning(
                "Metric alias preflight failed for %s: %s", signature, exc
            )
        _WORKER_STATE["metrics_preflight_done"] = True

    try:
        metrics, sources, missing = metrics_contract.evaluate_metrics(portfolio)
    except Exception as exc:  # pragma: no cover - warning path
        logger.warning("Metric evaluation failed for %s: %s", signature, exc)
        canonical = list(metrics_contract.METRIC_ALIASES)
        metrics = dict.fromkeys(canonical)
        sources = dict.fromkeys(canonical, "missing")
        missing = list(canonical)

    trades = int(portfolio.trades.count())

    equity_curve = pd.Series(dtype=float)
    if settings.get("collect_equity_curve"):
        value_fn = getattr(portfolio, "value", None)
        if callable(value_fn):
            try:
                equity_curve = value_fn()
            except Exception:  # pragma: no cover - best effort conversion
                equity_curve = pd.Series(dtype=float)
            else:
                if not isinstance(equity_curve, pd.Series):
                    try:
                        equity_curve = pd.Series(equity_curve)
                    except Exception:
                        equity_curve = pd.Series(dtype=float)

    if sources and not _WORKER_STATE["metric_mapping_logged"]:
        logger.info("Metrics mapping for %s: %s", signature, metrics_contract.format_mapping(sources))
        _WORKER_STATE["metric_mapping_logged"] = True

    return {
        "sortino": metrics.get("sortino"),
        "profit_factor": metrics.get("profit_factor"),
        "max_drawdown": metrics.get("max_drawdown"),
        "trades": trades,
        "total_return": metrics.get("total_return"),
        "equity_curve": equity_curve,
        "signal_counts": signal_counts,
        "metric_sources": sources,
        "missing_metrics": list(missing),
        "metric_provider": signature,
    }


def evaluate_batch(
    descriptor: dict[str, Any],
    base_rules: dict[str, Any],
    gene_map: dict[int, dict[str, Any]],
    candidates: list[dict[str, Any]],
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate a micro-batch of candidates for a specific asset."""

    asset_id = descriptor.get("asset_id", "")
    start = time.perf_counter()
    try:
        ohlc = registry.attach(descriptor)
    except Exception as exc:  # pragma: no cover - attachment issues are rare
        err = {
            "type": type(exc).__name__,
            "message": str(exc),
            "trace": traceback.format_exc(),
        }
        latency = time.perf_counter() - start
        return {
            "asset_id": asset_id,
            "results": [
                {
                    "sol_idx": candidate["index"],
                    "error": err,
                }
                for candidate in candidates
            ],
            "latency": latency,
            "rows": 0,
            "bytes": 0,
        }

    if ohlc.empty:
        latency = time.perf_counter() - start
        return {
            "asset_id": asset_id,
            "results": [
                {
                    "sol_idx": candidate["index"],
                    "stats": None,
                    "reason": "insufficient_coverage",
                }
                for candidate in candidates
            ],
            "latency": latency,
            "rows": 0,
            "bytes": 0,
        }

    ohlc = ohlc.set_flags(allows_duplicate_labels=False)

    results: list[dict[str, Any]] = []
    for candidate in candidates:
        sol_idx = candidate["index"]
        vector = candidate["vector"]
        try:
            rules = inject_genes_into_rules(base_rules, gene_map, vector)
            stats = _evaluate_candidate(ohlc, rules, settings)
        except Exception as exc:  # noqa: BLE001 - propagate diagnostic payload
            trace = traceback.format_exception(exc.__class__, exc, exc.__traceback__)
            results.append(
                {
                    "sol_idx": sol_idx,
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "trace": "".join(trace),
                        "indicator": getattr(exc, "indicator", None),
                    },
                }
            )
            continue

        results.append({"sol_idx": sol_idx, "stats": stats})

    latency = time.perf_counter() - start
    bytes_estimate = int(ohlc.memory_usage(deep=True).sum())
    return {
        "asset_id": asset_id,
        "results": results,
        "latency": latency,
        "rows": int(len(ohlc)),
        "bytes": bytes_estimate,
    }

