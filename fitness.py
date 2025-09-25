# fitness.py

"""
Fitness Function for Genetic Algorithm
(This version uses the correct pandas .shift() method for time-based exits)
"""
import concurrent.futures as cf
import copy
import logging
import traceback
import types
import warnings
import weakref
from collections import Counter
from collections.abc import Mapping
from typing import Literal

import numpy as np
import pandas as pd

try:  # pragma: no cover - import guard for optional heavy dependency
    import vectorbt as vbt
except Exception:  # pragma: no cover - fallback to stub
    from vbt_stub import Portfolio
    from vbt_stub import __file__ as _vbt_file
    from vbt_stub import __version__ as _vbt_ver

    vbt = types.ModuleType("vectorbt")
    vbt.Portfolio = Portfolio
    vbt.__version__ = _vbt_ver
    vbt.__file__ = _vbt_file
else:  # pragma: no cover - inject stub attributes if minimal module present
    if not hasattr(vbt, "Portfolio"):
        from vbt_stub import Portfolio
        from vbt_stub import __file__ as _vbt_file
        from vbt_stub import __version__ as _vbt_ver

        vbt.Portfolio = Portfolio
        vbt.__version__ = getattr(vbt, "__version__", _vbt_ver)
        vbt.__file__ = getattr(vbt, "__file__", _vbt_file)

import config
import metrics_contract
import strategy_engine as engine
import trade_floor
from exits_nb import (
    coerce_exit_params,
    compute_exit_metrics,
    generate_dynamic_exit_signals_nb,
    summarise_exit_reasons,
)
from params_resolver import inject_genes_into_rules
from portfolio_utils import extract_exit_params
from utils.math import weighted_mean_std

PenaltyDetail = str | dict[str, float | str] | None

logger = logging.getLogger(__name__)

DynamicExitSizeMode = Literal["fraction_base", "fraction_current", "absolute"]


ConfigError = getattr(config, "ConfigurationError", RuntimeError)


def build_dynamic_exit_orders(
    *,
    entries: pd.Series,
    exits_series: pd.Series,
    exit_size_series: pd.Series,
    base_entry_size: float,
    mode: DynamicExitSizeMode,
    asset_label: str,
    tol: float = 1e-8,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Convert raw entry/exit signals into mutually exclusive orders.

    ``generate_dynamic_exit_signals_nb`` initialises its internal ``open_qty`` to
    ``1.0`` and emits ``exit_size`` as the fraction of that baseline.  The
    ``mode`` parameter allows alternative semantics:

    - ``"fraction_base"`` (default): treat ``exit_size`` as a fraction of the
      configured ``base_entry_size``.
    - ``"fraction_current"``: treat ``exit_size`` as a fraction of the
      *current* ``open_qty`` (the simulator's behaviour if it re-computes on the
      live position).
    - ``"absolute"``: use ``exit_size`` directly as an absolute quantity.

    The helper enforces that entries and exits never overlap except for an
    intentional forced close on the last bar, clamps overshoots with capped exit
    sizes, and drops zero-quantity exits to avoid submitting no-op orders.
    """

    if base_entry_size <= 0:
        raise ValueError("BASE_ENTRY_SIZE must be positive")

    valid_modes: set[str] = {"fraction_base", "fraction_current", "absolute"}
    if mode not in valid_modes:
        raise ValueError(f"Unsupported dynamic exit sizing mode: {mode!r}")

    entries_index = entries.index
    exits_series = exits_series.reindex(entries_index, fill_value=False)
    exit_size_series = exit_size_series.reindex(entries_index, fill_value=0.0)

    entries_bool = entries.astype(bool)
    exits_bool = exits_series.astype(bool)
    exit_size_clean = exit_size_series.astype(float).fillna(0.0)

    if mode in ("fraction_base", "fraction_current"):
        above_one = exit_size_clean > 1.0 + tol
        below_zero = exit_size_clean < -tol
        if above_one.any() or below_zero.any():
            logger.warning(
                "Exit fractions for %s fell outside [0, 1]; clipping", asset_label
            )
        exit_size_clean = exit_size_clean.clip(lower=0.0, upper=1.0)
    else:
        if (exit_size_clean < -tol).any():
            logger.warning(
                "Negative absolute exit sizes for %s detected; clipping", asset_label
            )
        exit_size_clean = exit_size_clean.clip(lower=0.0)

    entries_array = entries_bool.to_numpy(dtype=bool, copy=False)
    exits_array = exits_bool.to_numpy(dtype=bool, copy=False)
    exit_size_array = exit_size_clean.to_numpy(dtype=float, copy=False)

    entries_active_array = np.zeros_like(entries_array, dtype=bool)
    exits_clean_array = exits_array.copy()
    size_array = np.zeros_like(exit_size_array, dtype=float)

    open_qty = 0.0
    max_open_qty = 0.0
    warning_counts: dict[str, int] = {}
    warning_limit = 5

    def _warn(key: str, message: str, *args) -> None:
        count = warning_counts.get(key, 0) + 1
        warning_counts[key] = count
        if count <= warning_limit:
            logger.warning(message, *args)

    entry_logs: list[str] = []
    exit_logs: list[str] = []

    for idx in range(entries_array.size):
        if entries_array[idx] and open_qty <= tol:
            entries_active_array[idx] = True
            size_array[idx] = base_entry_size
            open_qty += base_entry_size
            max_open_qty = max(max_open_qty, open_qty)
            if len(entry_logs) < 5:
                entry_logs.append(f"{entries_index[idx]}:{base_entry_size:.6f}")
            continue

        if exits_array[idx]:
            if mode == "fraction_base":
                requested_exit = exit_size_array[idx] * base_entry_size
            elif mode == "fraction_current":
                requested_exit = exit_size_array[idx] * open_qty
            else:
                requested_exit = exit_size_array[idx]

            if requested_exit < -tol:
                _warn(
                    "negative_exit",
                    "Negative exit size requested for %s at %s; clamping to zero",
                    asset_label,
                    entries_index[idx],
                )
                requested_exit = 0.0

            if requested_exit > open_qty + tol:
                _warn(
                    "overshoot_exit",
                    "Exit qty > open qty at %s for %s (capping): exit=%.6f, open=%.6f",
                    entries_index[idx],
                    asset_label,
                    requested_exit,
                    open_qty,
                )

            exit_qty = min(requested_exit, open_qty)
            exit_qty = max(0.0, exit_qty)
            if exit_qty <= tol:
                exits_clean_array[idx] = False
                size_array[idx] = 0.0
            else:
                size_array[idx] = exit_qty
                open_qty = max(0.0, open_qty - exit_qty)

                if len(exit_logs) < 5:
                    exit_logs.append(f"{entries_index[idx]}:{exit_qty:.6f}")

        max_open_qty = max(max_open_qty, open_qty)

    if max_open_qty > base_entry_size * (1.0 + 1e-8):
        raise AssertionError("Dynamic exit state machine opened more than one position")

    forced_close = False
    if open_qty > tol:
        forced_close = True
        forced_index = entries_array.size - 1
        exits_clean_array[forced_index] = True
        size_array[forced_index] = size_array[forced_index] + open_qty
        if len(exit_logs) < 5 and size_array[forced_index] > tol:
            exit_logs.append(
                f"{entries_index[forced_index]}:{size_array[forced_index]:.6f}"
            )
        open_qty = 0.0
    else:
        forced_index = -1

    exits_clean_array[entries_active_array] = False
    if forced_close and forced_index >= 0 and entries_active_array[forced_index]:
        # Intentional EOF overlap: forced close flattens any residual on the last bar.
        exits_clean_array[forced_index] = True

    entries_active = pd.Series(entries_active_array, index=entries_index, dtype=bool)
    exits_clean = pd.Series(exits_clean_array, index=entries_index, dtype=bool)
    size_series = pd.Series(size_array, index=entries_index, dtype=float)

    if (entries_active & (size_series <= 0)).any():
        raise AssertionError("Entries must have strictly positive order size")

    logger.debug(
        "Dynamic exit orders for %s entries: %s",
        asset_label,
        ", ".join(entry_logs) if entry_logs else "none",
    )
    logger.debug(
        "Dynamic exit orders for %s exits: %s",
        asset_label,
        ", ".join(exit_logs) if exit_logs else "none",
    )
    if forced_close:
        logger.debug(
            "Dynamic exit orders for %s inserted forced close at %s",
            asset_label,
            entries_index[forced_index],
        )

    summary_labels = {
        "overshoot_exit": "exit overshoot",
        "negative_exit": "negative exit size",
    }
    for key, count in warning_counts.items():
        if count > warning_limit:
            label = summary_labels.get(key, key)
            logger.warning(
                "%s warnings for %s exceeded %d occurrences (total=%d); further messages suppressed",
                label,
                asset_label,
                warning_limit,
                count,
            )

    return entries_active, exits_clean, size_series


def _sanitize_metric(value: float | int | None, fallback: float) -> float:
    if value is None or pd.isna(value):
        return float(fallback)
    return float(value)


def _sanitize_profit_factor(
    value: float | int | None, *, cap: float, fallback: float
) -> float:
    pf = _sanitize_metric(value, fallback)
    if np.isinf(pf) or pf > cap:
        return float(cap)
    return pf


def _composite_score(
    sortino: float | int | None,
    profit_factor: float | int | None,
    max_drawdown: float | int | None,
    *,
    weights: Mapping[str, float],
    pf_cap: float,
    nan_fallback: float,
    max_drawdown_fallback: float,
    exit_usage: Mapping[str, float] | None = None,
    exit_weights: Mapping[str, float] | None = None,
) -> float:
    sortino_val = _sanitize_metric(sortino, nan_fallback)
    pf_val = _sanitize_profit_factor(profit_factor, cap=pf_cap, fallback=nan_fallback)
    drawdown_val = _sanitize_metric(max_drawdown, max_drawdown_fallback)
    drawdown_score = 1 - (drawdown_val / 100.0)
    base_score = (
        sortino_val * weights["sortino_ratio"]
        + pf_val * weights["profit_factor"]
        + drawdown_score * weights["max_drawdown"]
    )
    if exit_usage and exit_weights:
        trades = float(exit_usage.get("trades_evaluated", 0.0) or 0.0)
        tp_events = float(exit_usage.get("tp_trades_evaluated", 0.0) or 0.0)
        tp_rate = tp_events / trades if trades > 0 else 0.0
        timeout_rate = float(exit_usage.get("sl_timeout_usage_rate", 0.0) or 0.0)
        avg_tp_level = float(exit_usage.get("avg_tp_level_reached", 0.0) or 0.0)
        trailing_rate = float(exit_usage.get("trailing_tp_hit_rate", 0.0) or 0.0)

        timeout_penalty = max(
            0.0,
            timeout_rate - float(exit_weights.get("timeout_target", 1.0) or 0.0),
        )
        timeout_penalty *= float(exit_weights.get("timeout_weight", 0.0) or 0.0)

        tp_shortfall = max(
            0.0,
            float(exit_weights.get("tp_hit_target", 0.0) or 0.0) - tp_rate,
        )
        tp_penalty = tp_shortfall * float(exit_weights.get("tp_hit_weight", 0.0) or 0.0)

        level_shortfall = max(
            0.0,
            float(exit_weights.get("avg_tp_level_target", 0.0) or 0.0) - avg_tp_level,
        )
        level_penalty = level_shortfall * float(
            exit_weights.get("avg_tp_level_weight", 0.0) or 0.0
        )

        trailing_shortfall = max(
            0.0,
            float(exit_weights.get("trailing_tp_target", 0.0) or 0.0) - trailing_rate,
        )
        trailing_penalty = trailing_shortfall * float(
            exit_weights.get("trailing_tp_weight", 0.0) or 0.0
        )

        base_score -= timeout_penalty + tp_penalty + level_penalty + trailing_penalty

    return base_score


def print_floor_failures(counter: Counter):
    """Utility to print a consistent hard-floor failure summary."""
    if not counter or sum(counter.values()) == 0:
        print("Hard-floor failures: none")
    else:
        print(f"Hard-floor failures: {dict(counter)}")


class FitnessEvaluator:
    def __init__(self, ohlc_data: pd.DataFrame, base_rules: dict, gene_map: dict):
        self.ohlc_data = ohlc_data
        self.base_rules = base_rules
        self.gene_map = gene_map
        self._metrics_preflight_done = False
        self._metric_mapping_logged = False
        self.last_exit_params: dict | None = None
        self.last_exit_summary: dict | None = None
        self.last_exit_metrics: dict | None = None

    def __call__(self, ga_instance, solution, sol_idx):
        config.initialize_config()
        try:
            rules = inject_genes_into_rules(self.base_rules, self.gene_map, solution)
            entries = engine.process_strategy_rules(self.ohlc_data, rules)

            if entries.sum() < config.FITNESS_WEIGHTS["min_trades"]:
                return -1.0

            exit_rules = rules.get("exit_rules", {})
            self.last_exit_params = None
            self.last_exit_summary = None
            self.last_exit_metrics = None
            exit_usage_metrics: Mapping[str, float] | None = None
            if config.USE_DYNAMIC_EXIT_SIMULATOR:
                base_entry_size = float(getattr(config, "BASE_ENTRY_SIZE", 1.0))
                size_mode: DynamicExitSizeMode = getattr(
                    config, "DYNAMIC_EXIT_SIZE_MODE", "fraction_base"
                )
                raw_label = getattr(config, "TICKER", "asset")
                asset_label = str(raw_label or "asset")
                try:
                    exit_params = coerce_exit_params(
                        exit_rules,
                        config.MAX_HOLD_PERIOD,
                        getattr(config, "TIMEFRAME", None),
                    )
                    price_map = {
                        "close": self.ohlc_data["Close"].to_numpy(dtype=float),
                        "high": self.ohlc_data.get(
                            "High", self.ohlc_data["Close"]
                        ).to_numpy(dtype=float),
                        "low": self.ohlc_data.get(
                            "Low", self.ohlc_data["Close"]
                        ).to_numpy(dtype=float),
                    }
                    entry_array = entries.to_numpy(dtype=bool)
                    telemetry_cfg = getattr(config, "EXIT_TELEMETRY", {})
                    collect_traces = telemetry_cfg.get(
                        "collect_traces", telemetry_cfg.get("enabled", True)
                    )
                    exit_result = generate_dynamic_exit_signals_nb(
                        entry_array,
                        price_map,
                        exit_params,
                        seed=getattr(config, "SEED", None),
                        collect_traces=collect_traces,
                    )
                except ValueError as exc:
                    logger.debug(
                        "Invalid exit configuration for %s: %s", asset_label, exc
                    )
                    self.last_exit_summary = None
                    self.last_exit_metrics = None
                    self.last_exit_params = None
                    return -999.0
                exits_mask = np.asarray(exit_result.exits, dtype=bool)
                residual_size = exit_result.exit_size[~exits_mask]
                if residual_size.size and not np.allclose(
                    residual_size, 0.0, atol=1e-12
                ):
                    raise AssertionError(
                        "exit_size must be zero on bars without exit signals"
                    )
                exits_series = pd.Series(exit_result.exits, index=entries.index)
                exit_size_series = pd.Series(
                    exit_result.exit_size, index=entries.index, dtype=float
                )
                entries_active, exits_series, size_series = build_dynamic_exit_orders(
                    entries=entries,
                    exits_series=exits_series,
                    exit_size_series=exit_size_series,
                    base_entry_size=base_entry_size,
                    mode=size_mode,
                    asset_label=asset_label,
                )
                summary = summarise_exit_reasons(exit_result, [asset_label])
                exit_metrics_map = compute_exit_metrics(exit_result, [asset_label])
                self.last_exit_summary = summary
                self.last_exit_metrics = exit_metrics_map
                self.last_exit_params = exit_params.as_dict()
                exit_usage_metrics = exit_metrics_map.get(asset_label)
                accumulate_flag = bool(
                    getattr(config, "DYNAMIC_EXIT_ACCUMULATE", False)
                )
                if not accumulate_flag:
                    raise ConfigError(
                        "Dynamic exit simulator requires"
                        " config.DYNAMIC_EXIT_ACCUMULATE=True for staged exits"
                    )
                portfolio = vbt.Portfolio.from_signals(
                    close=self.ohlc_data["Close"],
                    entries=entries_active,
                    exits=exits_series,
                    size=size_series,
                    accumulate=accumulate_flag,
                    size_type="amount",
                    fees=config.FEES,
                    freq=config.to_pandas_freq(config.TIMEFRAME),
                )
            else:
                time_based_exit, sl_stop, sl_trail, tp_stop = extract_exit_params(
                    entries, exit_rules, config.MAX_HOLD_PERIOD
                )
                portfolio = vbt.Portfolio.from_signals(
                    close=self.ohlc_data["Close"],
                    entries=entries,
                    exits=time_based_exit,
                    sl_stop=sl_stop,
                    tp_stop=tp_stop,
                    sl_trail=sl_trail,
                    fees=config.FEES,
                    freq=config.to_pandas_freq(config.TIMEFRAME),
                )

            signature = metrics_contract._provider_signature(portfolio)

            if not self._metrics_preflight_done:
                try:
                    metrics_contract.assert_metric_aliases(portfolio)
                except Exception as exc:
                    logger.warning(
                        "Metric alias preflight failed for %s: %s",
                        signature,
                        exc,
                    )
                self._metrics_preflight_done = True

            try:
                metrics, sources, _ = metrics_contract.evaluate_metrics(portfolio)
            except Exception as exc:
                logger.warning(
                    "Metric evaluation failed for %s: %s",
                    signature,
                    exc,
                )
                canonical = list(metrics_contract.METRIC_ALIASES)
                metrics = dict.fromkeys(canonical)
                sources = dict.fromkeys(canonical, "missing")
            if not self._metric_mapping_logged and sources:
                logger.info(
                    "Metrics mapping for %s: %s",
                    signature,
                    metrics_contract.format_mapping(sources),
                )
                self._metric_mapping_logged = True

            weights = config.FITNESS_WEIGHTS
            cap = getattr(config, "MULTI_ASSET", {}).get("winsorize_pf_cap", 5.0)
            exit_weights = getattr(config, "FITNESS_EXIT_USAGE", None)
            score = _composite_score(
                metrics.get("sortino"),
                metrics.get("profit_factor"),
                metrics.get("max_drawdown"),
                weights=weights,
                pf_cap=cap,
                nan_fallback=0.0,
                max_drawdown_fallback=100.0,
                exit_usage=exit_usage_metrics,
                exit_weights=exit_weights,
            )

            return score if np.isfinite(score) else -1.0

        except Exception as e:
            print(f"Error in fitness evaluation: {e}")
            return -999.0


class MultiAssetFitnessEvaluator:
    """Evaluate a candidate solution across multiple assets.

    The evaluator computes the per-asset composite metric using the same
    recipe as :class:`FitnessEvaluator` and then aggregates the results using a
    dispersion penalty.  The behaviour is governed by ``config.MULTI_ASSET``
    but can be overridden by passing a custom ``settings`` dictionary.
    """

    def __init__(
        self,
        group_data: dict,
        base_rules: dict,
        gene_map: dict,
        settings: dict | None = None,
    ):
        self.group_data = group_data  # dict[ticker -> OHLCV DataFrame]
        self._sorted_tickers = sorted(group_data)
        self.base_rules = base_rules
        self.gene_map = gene_map
        defaults = getattr(config, "MULTI_ASSET", {})
        self.settings = copy.deepcopy(defaults)
        if settings:
            self.settings.update(settings)
        self.settings["collect_equity_curve"] = bool(
            self.settings.get("collect_equity_curve", False)
        )
        # Clamp min_included_assets to available data after alignment
        mia = self.settings.get("min_included_assets", 1)
        self.settings["min_included_assets"] = min(mia, len(group_data))
        self.last_details = {}
        self.floor_failures = Counter()
        self._metrics_preflight_done = False
        self._metric_mapping_logged = False
        self._executor = None
        self._executor_signature = None
        self._executor_finalizer = None

        # Validate key configuration values to catch misconfiguration early.
        assert (
            self.settings.get("lambda_dispersion", 0.0) >= 0
        ), "lambda_dispersion must be >= 0"
        assert (
            self.settings.get("winsorize_pf_cap", 1.0) >= 1
        ), "winsorize_pf_cap must be >= 1"
        assert (
            self.settings.get("soft_penalty_strength", 0.0) >= 0
        ), "soft_penalty_strength must be >= 0"
        assert (
            self.settings.get("min_total_trades", 0) >= 0
        ), "min_total_trades must be >= 0"

        # Warn if the configured floors are unreachable
        min_group = self.settings.get("min_total_trades", 0)
        need = self.settings.get("min_included_assets", 0) * self.settings.get(
            "per_asset_min_trades", 0
        )
        if min_group and need > min_group:
            if self.settings.get("trade_floor_policy") == "soft_penalty":
                self.settings["min_total_trades"] = need
            else:
                warnings.warn(
                    "min_total_trades < min_included_assets * per_asset_min_trades; run may be infeasible.",
                    stacklevel=2,
                )

    # ------------------------------------------------------------------
    @staticmethod
    def _shutdown_executor_static(executor) -> None:
        """Best-effort shutdown helper that tolerates older signatures."""

        if executor is None:
            return
        try:
            executor.shutdown(wait=True)
        except TypeError:
            executor.shutdown()
        except Exception:  # pragma: no cover - defensive logging only
            logger.debug("Failed to shutdown executor", exc_info=True)

    # ------------------------------------------------------------------
    def _shutdown_executor(self) -> None:
        """Shut down and detach from the cached executor if present."""

        executor = self._executor
        if executor is None:
            return
        self._executor = None
        self._executor_signature = None
        finalizer = self._executor_finalizer
        self._executor_finalizer = None
        if finalizer is not None:
            try:
                finalizer.detach()
            except Exception:  # pragma: no cover - best effort cleanup
                logger.debug("Failed to detach executor finalizer", exc_info=True)
        self._shutdown_executor_static(executor)

    # ------------------------------------------------------------------
    def close(self) -> None:
        """Public method to release the cached executor."""

        self._shutdown_executor()

    # ------------------------------------------------------------------
    def shutdown(self) -> None:
        """Alias for :meth:`close` to match executor semantics."""

        self.close()

    # ------------------------------------------------------------------
    def _get_executor(self):
        """Return the cached executor, creating it lazily when required."""

        parallel_cfg = self.settings.get("parallel", {}) or {}
        if not parallel_cfg.get("enabled"):
            if self._executor is not None:
                self._shutdown_executor()
            return None

        backend = parallel_cfg.get("backend", "process")
        max_workers = parallel_cfg.get("max_workers")
        signature = (backend, max_workers)
        if self._executor is not None and self._executor_signature != signature:
            self._shutdown_executor()

        if self._executor is None:
            Executor = (
                cf.ProcessPoolExecutor
                if backend == "process"
                else cf.ThreadPoolExecutor
            )
            executor = Executor(max_workers=max_workers)
            self._executor = executor
            self._executor_signature = signature
            if self._executor_finalizer is not None:
                try:
                    self._executor_finalizer.detach()
                except Exception:  # pragma: no cover - best effort cleanup
                    logger.debug(
                        "Failed to detach stale executor finalizer", exc_info=True
                    )
            self._executor_finalizer = weakref.finalize(
                self, MultiAssetFitnessEvaluator._shutdown_executor_static, executor
            )
        return self._executor

    # ------------------------------------------------------------------
    @staticmethod
    def _empty_stats() -> dict:
        """Return a copy-safe container for assets without valid results."""

        return {
            "sortino": None,
            "profit_factor": None,
            "max_drawdown": None,
            "trades": 0,
            "total_return": None,
            "equity_curve": pd.Series(dtype=float),
            "signal_counts": {},
            "metric_sources": {},
            "missing_metrics": [],
        }

    # ------------------------------------------------------------------
    def _build_evaluation_record(
        self,
        stats: dict | None = None,
        reason: str | None = None,
        detail: str | None = None,
        trace: tuple | str | None = None,
    ) -> dict:
        """Normalise evaluation output into a consistent mapping."""

        record = dict(stats) if stats is not None else self._empty_stats()
        if reason is not None:
            record["evaluation_reason"] = reason
        if detail is not None:
            record["reason_detail"] = detail
        if trace:
            record["reason_trace"] = trace
        return record

    # ------------------------------------------------------------------
    @staticmethod
    def _normalise_reason_trace(trace: tuple | str | None) -> str | None:
        """Convert verbose trace tuples into a printable string."""

        if not trace:
            return None
        if isinstance(trace, str):
            return trace
        try:
            return " | ".join(str(part) for part in trace)
        except TypeError:
            return str(trace)

    # ------------------------------------------------------------------
    def _log_metric_mapping(
        self, metric_sources: Mapping[str, str], signature: str | None = None
    ) -> None:
        """Log the resolved metric mapping exactly once."""

        if self._metric_mapping_logged or not metric_sources:
            return
        mapping_summary = metrics_contract.format_mapping(metric_sources)
        if signature:
            logger.info("Metrics mapping for %s: %s", signature, mapping_summary)
        else:
            logger.info("Metrics mapping: %s", mapping_summary)
        self._metric_mapping_logged = True

    # ------------------------------------------------------------------
    @staticmethod
    def _collect_exit_overview(
        per_asset_details: Mapping[str, Mapping[str, object]],
    ) -> tuple[dict[str, dict], dict[str, dict], dict[str, dict]]:
        """Extract exit summaries, metrics, and parameter snapshots for metadata."""

        summary: dict[str, dict] = {}
        params: dict[str, dict] = {}
        metrics: dict[str, dict] = {}
        for ticker, detail in per_asset_details.items():
            if not isinstance(detail, Mapping):
                continue
            exit_summary = detail.get("exit_summary")
            if isinstance(exit_summary, Mapping) and exit_summary:
                summary[ticker] = dict(exit_summary)
            exit_params = detail.get("exit_params")
            if isinstance(exit_params, Mapping) and exit_params:
                params[ticker] = dict(exit_params)
            exit_metrics = detail.get("exit_metrics")
            if isinstance(exit_metrics, Mapping) and exit_metrics:
                metrics[ticker] = dict(exit_metrics)
        return summary, params, metrics

    # ------------------------------------------------------------------
    def _prepare_metrics_record(
        self, stats: dict
    ) -> tuple[dict, str | None, str | None]:
        """Normalise returned stats and derive evaluation metadata."""

        record = dict(stats)
        metric_sources = record.get("metric_sources") or {}
        signature = record.get("metric_provider")
        if isinstance(metric_sources, Mapping):
            self._log_metric_mapping(metric_sources, signature)
        missing_metrics = list(record.get("missing_metrics") or [])
        trades = int(record.get("trades", 0) or 0)
        reason = None
        detail = None
        if missing_metrics and trades > 0:
            reason = "metrics_missing"
            metrics_list = ", ".join(sorted(missing_metrics))
            if signature:
                detail = f"{signature}: {metrics_list}"
            else:
                detail = metrics_list
        return record, reason, detail

    # ------------------------------------------------------------------
    def _evaluate_single_asset(
        self, ohlc: pd.DataFrame, rules: dict, ticker: str | None = None
    ) -> dict:
        """Run the strategy on a single asset and return raw statistics."""
        # Empty or very short dataframes can cause downstream libraries to
        # raise ``IndexError`` when statistics are requested.  In walk forward
        # validation some assets may have no data for a given window.  Handle
        # this case early and return a stub result that indicates zero trades
        # so that the caller can decide whether to ignore or penalise it.
        if ohlc is None or ohlc.empty:
            return self._empty_stats()

        entries, signal_counts = engine.process_strategy_rules(
            ohlc, rules, collect_counts=True
        )

        raw_label = ticker if ticker is not None else getattr(ohlc, "name", "asset")
        asset_label = str(raw_label or "asset")
        exit_rules = rules.get("exit_rules", {})
        exit_summary: dict[str, dict[str, float]] = {}
        exit_metrics: dict[str, float] = {}
        exit_params_dict: dict | None = None
        if config.USE_DYNAMIC_EXIT_SIMULATOR:
            try:
                exit_params = coerce_exit_params(
                    exit_rules,
                    config.MAX_HOLD_PERIOD,
                    getattr(config, "TIMEFRAME", None),
                )
            except ValueError as exc:
                logger.debug("Invalid exit configuration for %s: %s", asset_label, exc)
                return self._build_evaluation_record(
                    reason="invalid_exit_config", detail=str(exc)
                )
            price_map = {
                "close": ohlc["Close"].to_numpy(dtype=float),
                "high": ohlc.get("High", ohlc["Close"]).to_numpy(dtype=float),
                "low": ohlc.get("Low", ohlc["Close"]).to_numpy(dtype=float),
            }
            telemetry_cfg = getattr(config, "EXIT_TELEMETRY", {})
            collect_traces = telemetry_cfg.get(
                "collect_traces", telemetry_cfg.get("enabled", True)
            )
            exit_result = generate_dynamic_exit_signals_nb(
                entries.to_numpy(dtype=bool),
                price_map,
                exit_params,
                seed=getattr(config, "SEED", None),
                collect_traces=collect_traces,
            )
            exits_mask = np.asarray(exit_result.exits, dtype=bool)
            residual_size = exit_result.exit_size[~exits_mask]
            if residual_size.size and not np.allclose(residual_size, 0.0, atol=1e-12):
                raise AssertionError(
                    "exit_size must be zero on bars without exit signals"
                )
            exits_series = pd.Series(exit_result.exits, index=entries.index)
            exit_size_series = pd.Series(
                exit_result.exit_size, index=entries.index, dtype=float
            )
            base_entry_size = float(getattr(config, "BASE_ENTRY_SIZE", 1.0))
            size_mode: DynamicExitSizeMode = getattr(
                config, "DYNAMIC_EXIT_SIZE_MODE", "fraction_base"
            )
            entries_active, exits_series, size_series = build_dynamic_exit_orders(
                entries=entries,
                exits_series=exits_series,
                exit_size_series=exit_size_series,
                base_entry_size=base_entry_size,
                mode=size_mode,
                asset_label=asset_label,
            )
            summary_map = summarise_exit_reasons(exit_result, [asset_label])
            metrics_map = compute_exit_metrics(exit_result, [asset_label])
            exit_summary = summary_map.get(asset_label, {})
            exit_metrics = metrics_map.get(asset_label, {})
            exit_params_dict = exit_params.as_dict()
            accumulate_flag = bool(getattr(config, "DYNAMIC_EXIT_ACCUMULATE", False))
            if not accumulate_flag:
                raise ConfigError(
                    "Dynamic exit simulator requires"
                    " config.DYNAMIC_EXIT_ACCUMULATE=True for staged exits"
                )
            portfolio = vbt.Portfolio.from_signals(
                close=ohlc["Close"],
                entries=entries_active,
                exits=exits_series,
                size=size_series,
                accumulate=accumulate_flag,
                size_type="amount",
                fees=config.FEES,
                freq=config.to_pandas_freq(config.TIMEFRAME),
            )
        else:
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

        if not self._metrics_preflight_done:
            try:
                metrics_contract.assert_metric_aliases(portfolio)
            except Exception as exc:
                logger.warning(
                    "Metric alias preflight failed for %s: %s",
                    signature,
                    exc,
                )
            self._metrics_preflight_done = True

        try:
            metrics, sources, missing = metrics_contract.evaluate_metrics(portfolio)
        except Exception as exc:
            logger.warning(
                "Metric evaluation failed for %s: %s",
                signature,
                exc,
            )
            canonical = list(metrics_contract.METRIC_ALIASES)
            metrics = dict.fromkeys(canonical)
            sources = dict.fromkeys(canonical, "missing")
            missing = list(canonical)
        trades = int(portfolio.trades.count())

        equity_curve = pd.Series(dtype=float)
        if self.settings.get("collect_equity_curve"):
            value_fn = getattr(portfolio, "value", None)
            if callable(value_fn):
                try:
                    equity_curve = value_fn()
                except Exception:
                    equity_curve = pd.Series(dtype=float)
                else:
                    if not isinstance(equity_curve, pd.Series):
                        try:
                            equity_curve = pd.Series(equity_curve)
                        except Exception:
                            equity_curve = pd.Series(dtype=float)

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
            "exit_summary": exit_summary,
            "exit_metrics": exit_metrics,
            "exit_params": exit_params_dict,
        }

    # ------------------------------------------------------------------
    def _evaluate_assets(self, rules: dict) -> tuple[dict[str, dict], Counter]:
        """Evaluate every asset and return their raw statistics."""

        results: dict[str, dict] = {}
        err_counts: Counter = Counter()
        verbose = bool(self.settings.get("verbose_asset_errors"))

        executor = self._get_executor()
        if executor is not None:
            future_map = {}
            for ticker in self._sorted_tickers:
                ohlc = self.group_data[ticker]
                if ohlc is None or ohlc.empty:
                    results[ticker] = self._build_evaluation_record(
                        reason="insufficient_coverage"
                    )
                    continue
                future_map[
                    executor.submit(self._evaluate_single_asset, ohlc, rules, ticker)
                ] = ticker
            for fut in cf.as_completed(future_map):
                ticker = future_map[fut]
                try:
                    raw_stats = fut.result()
                    stats, reason, detail = self._prepare_metrics_record(raw_stats)
                    results[ticker] = self._build_evaluation_record(
                        stats, reason=reason, detail=detail
                    )
                except Exception as e:
                    if verbose:
                        print(f"Error evaluating asset {ticker}: {e}")
                        tb = traceback.format_exception(e.__class__, e, e.__traceback__)
                        trace = (tb[0].strip(), tb[-1].strip())
                    else:
                        trace = None
                    results[ticker] = self._build_evaluation_record(
                        reason="evaluation_error",
                        detail=repr(e),
                        trace=trace,
                    )
                    ind = getattr(e, "indicator", None)
                    if ind:
                        err_counts[ind] += 1
        else:
            for ticker in self._sorted_tickers:
                ohlc = self.group_data[ticker]
                if ohlc is None or ohlc.empty:
                    results[ticker] = self._build_evaluation_record(
                        reason="insufficient_coverage"
                    )
                    continue
                try:
                    raw_stats = self._evaluate_single_asset(ohlc, rules, ticker)
                    stats, reason, detail = self._prepare_metrics_record(raw_stats)
                    results[ticker] = self._build_evaluation_record(
                        stats, reason=reason, detail=detail
                    )
                except Exception as e:
                    if verbose:
                        print(f"Error evaluating asset {ticker}: {e}")
                        tb = traceback.format_exception(e.__class__, e, e.__traceback__)
                        trace = (tb[0].strip(), tb[-1].strip())
                    else:
                        trace = None
                    results[ticker] = self._build_evaluation_record(
                        reason="evaluation_error",
                        detail=repr(e),
                        trace=trace,
                    )
                    ind = getattr(e, "indicator", None)
                    if ind:
                        err_counts[ind] += 1

        for ticker in self._sorted_tickers:
            results.setdefault(ticker, self._build_evaluation_record())
        return results, err_counts

    # ------------------------------------------------------------------
    def _score_assets(self, evaluation_results: dict[str, dict]) -> dict:
        """Compute per-asset scores and bookkeeping from evaluation stats."""

        per_asset_metrics: list[float] = []
        included_assets: list[str] = []
        per_asset_details: dict[str, dict] = {}
        total_trades = 0
        assets_traded = 0
        asset_weights_cfg = self.settings.get("asset_weights") or {}
        per_asset_min = self.settings.get("per_asset_min_trades", 1)
        metric_type = self.settings.get("metric", "composite")
        nan_fallback = self.settings.get("nan_fallback", 0.0)
        cap = self.settings.get("winsorize_pf_cap", 5.0)
        sources_recorded = False
        run_metric_sources: dict[str, str] | None = None
        exit_weights = getattr(config, "FITNESS_EXIT_USAGE", None)

        for ticker in self._sorted_tickers:
            stats_raw = evaluation_results.get(ticker, self._empty_stats())
            stats = dict(stats_raw)
            metric_sources = stats.pop("metric_sources", None)
            reason = stats.pop("evaluation_reason", None)
            reason_detail = stats.pop("reason_detail", None)
            reason_trace = stats.pop("reason_trace", None)
            trace_str = self._normalise_reason_trace(reason_trace)
            trades = int(stats.get("trades", 0) or 0)
            total_trades += trades
            weight = asset_weights_cfg.get(ticker, 1.0)
            pf_raw = stats.get("profit_factor")
            pf_capped = _sanitize_profit_factor(pf_raw, cap=cap, fallback=nan_fallback)
            exit_usage = stats.get("exit_metrics")
            if not isinstance(exit_usage, Mapping):
                exit_usage = None

            if trades < per_asset_min:
                if self.settings.get("zero_trade_policy") == "penalize":
                    val = self.settings.get("zero_trade_penalty", -1.0)
                    per_asset_metrics.append(val)
                    included_assets.append(ticker)
                    if trades > 0:
                        assets_traded += 1
                    details = {
                        **stats,
                        "score": val,
                        "included": True,
                        "asset_weight": weight,
                        "profit_factor_capped": pf_capped,
                    }
                    if reason_detail is not None:
                        details["reason_detail"] = reason_detail
                    if trace_str:
                        details["reason_trace"] = trace_str
                    if metric_sources and not sources_recorded:
                        details["metric_sources"] = metric_sources
                        sources_recorded = True
                        if run_metric_sources is None:
                            run_metric_sources = dict(metric_sources)
                    per_asset_details[ticker] = details
                else:
                    reason_str = reason or (
                        "ignored_zero_trades"
                        if trades == 0
                        else "below_per_asset_min_trades"
                    )
                    info = self.settings.get("per_asset_floor_info")
                    if info:
                        reason_str += (
                            "; Per-asset floor: base="
                            f"{info['base_floor']} → scaled={info['ceil']} "
                            f"(window={info['window_days']}d, base={info['trading_days_per_year']}d)"
                        )
                    details = {
                        **stats,
                        "score": None,
                        "included": False,
                        "asset_weight": weight,
                        "profit_factor_capped": pf_capped,
                        "reason": reason_str,
                    }
                    if reason_detail is not None:
                        details["reason_detail"] = reason_detail
                    if trace_str:
                        details["reason_trace"] = trace_str
                    if metric_sources and not sources_recorded:
                        details["metric_sources"] = metric_sources
                        sources_recorded = True
                        if run_metric_sources is None:
                            run_metric_sources = dict(metric_sources)
                    per_asset_details[ticker] = details
                continue

            if metric_type == "sortino":
                val = _sanitize_metric(stats.get("sortino"), nan_fallback)
            elif metric_type == "profit_factor":
                val = pf_capped
            elif metric_type == "return":
                total_return = stats.get("total_return")
                val = _sanitize_metric(total_return, nan_fallback)
            else:
                w = config.FITNESS_WEIGHTS
                val = _composite_score(
                    stats.get("sortino"),
                    pf_raw,
                    stats.get("max_drawdown"),
                    weights=w,
                    pf_cap=cap,
                    nan_fallback=nan_fallback,
                    max_drawdown_fallback=100.0,
                    exit_usage=exit_usage,
                    exit_weights=exit_weights,
                )

            per_asset_metrics.append(val)
            included_assets.append(ticker)
            if trades > 0:
                assets_traded += 1
            details = {
                **stats,
                "score": val,
                "included": True,
                "asset_weight": weight,
                "profit_factor_capped": pf_capped,
            }
            if reason_detail is not None:
                details["reason_detail"] = reason_detail
            if trace_str:
                details["reason_trace"] = trace_str
            if metric_sources and not sources_recorded:
                details["metric_sources"] = metric_sources
                sources_recorded = True
                if run_metric_sources is None:
                    run_metric_sources = dict(metric_sources)
            per_asset_details[ticker] = details

        return {
            "per_asset_metrics": per_asset_metrics,
            "included_assets": included_assets,
            "per_asset_details": per_asset_details,
            "total_trades": total_trades,
            "assets_traded": assets_traded,
            "metric_sources": run_metric_sources or {},
        }

    # ------------------------------------------------------------------
    def _aggregate_scores(self, summary: dict) -> float:
        """Combine per-asset scores into the final fitness value."""

        per_asset_metrics = summary["per_asset_metrics"]
        included_assets = summary["included_assets"]
        per_asset_details = summary["per_asset_details"]
        total_trades = summary["total_trades"]
        assets_traded = summary["assets_traded"]
        metric_sources = summary.get("metric_sources") or {}
        (
            exit_summary_map,
            exit_params_map,
            exit_metrics_map,
        ) = self._collect_exit_overview(per_asset_details)

        if not per_asset_metrics:
            poor_score = self.settings.get("poor_score", -999.0)
            reason = "no_assets"
            self.floor_failures[reason] += 1
            self.last_details = {
                "per_asset": per_asset_details,
                "mu": 0.0,
                "sigma": 0.0,
                "lambda_sigma": 0.0,
                "total_trades": total_trades,
                "assets_included": 0,
                "assets_traded": assets_traded,
                "assets_ignored": len(self.group_data),
                "penalties": {
                    "trade_floor": reason,
                    "coverage": 0.0,
                    "min_assets": reason,
                    "stability": 0.0,
                },
                "min_total_trades": self.settings.get("min_total_trades", 0),
                "fitness": poor_score,
                "asset_weights": {},
                "metric_sources": metric_sources,
                "exit_reason_breakdown": exit_summary_map,
                "exit_params": exit_params_map,
                "exit_metrics": exit_metrics_map,
            }
            return poor_score

        asset_weights_cfg = self.settings.get("asset_weights") or {}
        raw_weights = []
        neg_seen = False
        for ticker in included_assets:
            w = asset_weights_cfg.get(ticker, 1.0)
            if w < 0:
                neg_seen = True
                w = 0.0
            raw_weights.append(w)
        if neg_seen:
            print("Warning: negative asset weights clipped to zero")
        weight_sum = sum(raw_weights)
        if weight_sum == 0:
            if raw_weights:
                print(
                    "Warning: all asset weights were zero; reverting to equal weights"
                )
            weights = [1.0 / len(per_asset_metrics)] * len(per_asset_metrics)
        else:
            weights = [w / weight_sum for w in raw_weights]

        w_map = {}
        for ticker, weight in zip(included_assets, weights):
            per_asset_details[ticker]["asset_weight"] = weight
            w_map[ticker] = weight

        m_arr = np.array(per_asset_metrics, dtype=float)
        w_arr = np.array(weights, dtype=float)
        mu, sigma = weighted_mean_std(m_arr, w_arr)

        lam = self.settings.get("lambda_dispersion", 0.0)
        F = mu - lam * sigma

        stability_penalty = 0.0
        if config.ENABLE_STABILITY_REG:
            history = self.settings.get("param_history") or []
            covs = []
            for g in config.STABILITY_GENES:
                vals = [
                    float(p[g]) for p in history if isinstance(p.get(g), (int, float))
                ]
                if len(vals) > 1:
                    mean = float(np.mean(vals))
                    if mean != 0:
                        std = float(np.std(vals))
                        if std > 0 and np.isfinite(std):
                            cov = std / abs(mean)
                            if np.isfinite(cov):
                                covs.append(cov)
            if covs:
                mean_cov = float(np.mean(covs))
                stability_penalty = config.STABILITY_ALPHA * mean_cov
                F -= stability_penalty

        policy = self.settings.get("trade_floor_policy", "hard_floor")
        poor_score = self.settings.get("poor_score", -999.0)
        min_trades = self.settings.get("min_total_trades", 0)
        min_assets = self.settings.get("min_included_assets", 1)
        trade_penalty: PenaltyDetail = None
        min_assets_penalty: PenaltyDetail = None

        assets_count = len(included_assets)
        if assets_count < min_assets:
            if policy == "hard_floor":
                F = poor_score
                reason = "below_min_included_assets"
                trade_penalty = reason
                min_assets_penalty = reason
                self.floor_failures[reason] += 1
                self.last_details = {
                    "per_asset": per_asset_details,
                    "mu": mu,
                    "sigma": sigma,
                    "lambda_sigma": lam * sigma,
                    "total_trades": total_trades,
                    "assets_included": assets_count,
                    "assets_traded": assets_traded,
                    "assets_ignored": len(self.group_data) - assets_count,
                    "penalties": {
                        "trade_floor": trade_penalty,
                        "coverage": 0.0,
                        "min_assets": min_assets_penalty,
                        "stability": stability_penalty,
                    },
                    "min_total_trades": min_trades,
                    "fitness": F,
                    "asset_weights": w_map,
                    "metric_sources": metric_sources,
                    "exit_reason_breakdown": exit_summary_map,
                    "exit_params": exit_params_map,
                    "exit_metrics": exit_metrics_map,
                }
                return F
            else:
                strength = self.settings.get("soft_penalty_strength", 1.0)
                scale = (assets_count / max(1, min_assets)) ** strength
                F *= scale
                min_assets_penalty = {"scale": scale}

        if policy == "hard_floor" and total_trades < min_trades:
            F = poor_score
            reason = "below_group_floor"
            trade_penalty = reason
            self.floor_failures[reason] += 1
            self.last_details = {
                "per_asset": per_asset_details,
                "mu": mu,
                "sigma": sigma,
                "lambda_sigma": lam * sigma,
                "total_trades": total_trades,
                "assets_included": assets_count,
                "assets_traded": assets_traded,
                "assets_ignored": len(self.group_data) - assets_count,
                "penalties": {
                    "trade_floor": trade_penalty,
                    "coverage": 0.0,
                    "min_assets": min_assets_penalty,
                    "stability": stability_penalty,
                },
                "min_total_trades": min_trades,
                "fitness": F,
                "asset_weights": w_map,
                "metric_sources": metric_sources,
                "exit_reason_breakdown": exit_summary_map,
                "exit_params": exit_params_map,
                "exit_metrics": exit_metrics_map,
            }
            return F
        elif policy == "soft_penalty" and total_trades < min_trades:
            mode = self.settings.get("soft_penalty_mode", "multiplicative")
            strength = self.settings.get("soft_penalty_strength", 1.0)
            if mode == "additive":
                penalty = strength * (1 - total_trades / max(1, min_trades))
                F -= penalty
                trade_penalty = {"mode": "additive", "penalty": penalty}
            else:
                scale = (total_trades / max(1, min_trades)) ** strength
                F *= scale
                trade_penalty = {"mode": "multiplicative", "scale": scale}

        coverage_penalty = 0.0
        if self.settings.get("zero_trade_policy") == "ignore":
            kappa = self.settings.get("coverage_penalty", 0.0)
            coverage = assets_count / max(1, len(self.group_data))
            coverage_penalty = kappa * (1 - coverage)
            F -= coverage_penalty

        self.last_details = {
            "per_asset": per_asset_details,
            "mu": mu,
            "sigma": sigma,
            "lambda_sigma": lam * sigma,
            "total_trades": total_trades,
            "assets_included": assets_count,
            "assets_traded": assets_traded,
            "assets_ignored": len(self.group_data) - assets_count,
            "penalties": {
                "trade_floor": trade_penalty,
                "coverage": coverage_penalty,
                "min_assets": min_assets_penalty,
                "stability": stability_penalty,
            },
            "min_total_trades": min_trades,
            "fitness": F,
            "asset_weights": w_map,
            "metric_sources": metric_sources,
            "exit_reason_breakdown": exit_summary_map,
            "exit_params": exit_params_map,
            "exit_metrics": exit_metrics_map,
        }

        return F

    # ------------------------------------------------------------------
    def __call__(self, ga_instance, solution, sol_idx):
        config.initialize_config()
        try:
            rules = inject_genes_into_rules(self.base_rules, self.gene_map, solution)

            evaluation_results, err_counts = self._evaluate_assets(rules)
            if err_counts:
                logger.info("evaluation error counts: %s", dict(err_counts))

            summary = self._score_assets(evaluation_results)
            return self._aggregate_scores(summary)

        except Exception as e:
            print(f"Error in multi-asset fitness evaluation: {e}")
            poor = self.settings.get("poor_score", -999.0)
            self.last_details = {
                "per_asset": {},
                "mu": 0.0,
                "sigma": 0.0,
                "lambda_sigma": 0.0,
                "total_trades": 0,
                "assets_included": 0,
                "assets_ignored": len(self.group_data),
                "penalties": {
                    "trade_floor": None,
                    "coverage": 0.0,
                    "min_assets": None,
                    "stability": 0.0,
                },
                "min_total_trades": self.settings.get("min_total_trades", 0),
                "fitness": poor,
            }
            return poor


def get_fitness_evaluator(ohlc_data, base_rules, gene_map):
    """Factory returning the appropriate fitness evaluator.

    Parameters
    ----------
    ohlc_data : pd.DataFrame or dict
        If ``config.MULTI_ASSET['enabled']`` is True, ``ohlc_data`` should be a
        mapping of ticker -> DataFrame.  Otherwise it is a single DataFrame.
    """

    config.initialize_config()
    settings = copy.deepcopy(getattr(config, "MULTI_ASSET", {}))
    if settings.get("enabled"):
        start = pd.to_datetime(config.TRAINING_PERIOD["start"])
        end = pd.to_datetime(config.TRAINING_PERIOD["end"])

        per_asset_base = settings.get("per_asset_min_trades")
        if per_asset_base:
            floor_pa, info_pa = trade_floor.scale_floor(
                per_asset_base,
                start,
                end,
                settings.get("trading_days_per_year", 252),
            )
            settings["per_asset_min_trades"] = floor_pa
            settings["per_asset_floor_info"] = info_pa
            print(
                "Per-asset floor: base="
                f"{per_asset_base} → scaled={floor_pa} "
                f"(window={info_pa['window_days']}d, base={info_pa['trading_days_per_year']}d)"
            )

        rate = settings.get("min_total_trades_per_year")
        if rate:
            floor, info = trade_floor.scale_floor(
                rate, start, end, settings.get("trading_days_per_year", 252)
            )
            settings["min_total_trades"] = floor
            settings["group_floor_info"] = info
        return MultiAssetFitnessEvaluator(ohlc_data, base_rules, gene_map, settings)
    return FitnessEvaluator(ohlc_data, base_rules, gene_map)
