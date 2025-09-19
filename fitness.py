# fitness.py

"""
Fitness Function for Genetic Algorithm
(This version uses the correct pandas .shift() method for time-based exits)
"""
import concurrent.futures as cf
import copy
import logging
import types
import warnings
import itertools
import time
import traceback
import math
from collections import Counter
from collections.abc import Mapping
from typing import Any

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
import global_executor
import metrics_contract
import strategy_engine as engine
import trade_floor
from data_registry import registry as data_registry
import fitness_worker
from params_resolver import inject_genes_into_rules
from portfolio_utils import extract_exit_params
from utils.math import weighted_mean_std

PenaltyDetail = str | dict[str, float | str] | None

logger = logging.getLogger(__name__)

_WINDOW_COUNTER = itertools.count()


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

    def __call__(self, ga_instance, solution, sol_idx):
        config.initialize_config()
        try:
            rules = inject_genes_into_rules(self.base_rules, self.gene_map, solution)
            entries = engine.process_strategy_rules(self.ohlc_data, rules)

            if entries.sum() < config.FITNESS_WEIGHTS["min_trades"]:
                return -1.0

            # Extract parameters for exits and stop rules
            exit_rules = rules.get("exit_rules", {})
            time_based_exit, sl_stop, sl_trail, tp_stop = extract_exit_params(
                entries, exit_rules, config.MAX_HOLD_PERIOD
            )

            portfolio = vbt.Portfolio.from_signals(
                close=self.ohlc_data["Close"],
                entries=entries,
                exits=time_based_exit,
                sl_stop=sl_stop,
                tp_stop=tp_stop,
                sl_trail=sl_trail,  # Pass the trailing stop value to the backtester
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

            sortino = metrics.get("sortino")
            profit_factor = metrics.get("profit_factor")
            max_drawdown = metrics.get("max_drawdown")

            cap = getattr(config, "MULTI_ASSET", {}).get("winsorize_pf_cap", 5.0)
            if profit_factor is None or pd.isna(profit_factor):
                profit_factor = 0.0
            else:
                profit_factor = float(profit_factor)
                if np.isinf(profit_factor) or profit_factor > cap:
                    profit_factor = cap
            if sortino is None or pd.isna(sortino):
                sortino = 0.0
            else:
                sortino = float(sortino)
            if max_drawdown is None or pd.isna(max_drawdown):
                max_drawdown = 100.0
            else:
                max_drawdown = float(max_drawdown)

            drawdown_score = 1 - (max_drawdown / 100.0)
            weights = config.FITNESS_WEIGHTS

            fitness_score = (
                (sortino * weights["sortino_ratio"])
                + (profit_factor * weights["profit_factor"])
                + (drawdown_score * weights["max_drawdown"])
            )

            return fitness_score if not np.isnan(fitness_score) else -1.0

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
        self._window_id = self.settings.get("window_id") or f"window-{next(_WINDOW_COUNTER)}"
        self._window_released = False
        self._descriptors = data_registry.register_window(self._window_id, group_data)
        self._assets_with_data = [
            ticker for ticker, desc in self._descriptors.items() if not desc.get("empty")
        ]
        self._assets_without_data = [
            ticker for ticker, desc in self._descriptors.items() if desc.get("empty")
        ]
        self.instrumentation: dict[str, Any] = {}
        self._generation_records: list[dict[str, Any]] = []
        exec_cfg = dict(getattr(config, "GLOBAL_EXECUTOR", {}))
        self._batch_size = max(1, int(exec_cfg.get("batch_size") or 1))
        self._min_batch_size = max(1, int(exec_cfg.get("min_batch_size") or 1))
        self._max_batch_size = max(self._min_batch_size, int(exec_cfg.get("max_batch_size") or self._batch_size))
        self._batch_adjustment_rate = float(exec_cfg.get("batch_step_ratio", 0.25) or 0.25)
        self._batch_adjustment_rate = min(0.5, max(0.05, self._batch_adjustment_rate))
        self._batch_cooldown_submissions = max(
            1, int(exec_cfg.get("batch_cooldown_submissions", 6) or 6)
        )
        self._submissions_since_adjustment = self._batch_cooldown_submissions
        self._latency_target = max(0.01, float(exec_cfg.get("latency_target_ms", 200)) / 1000.0)
        self._queue_high_watermark = float(exec_cfg.get("queue_high_watermark", 0.85) or 0.85)
        self._queue_low_watermark = float(exec_cfg.get("queue_low_watermark", 0.35) or 0.35)
        self._queue_high_watermark = min(0.99, max(0.0, self._queue_high_watermark))
        self._queue_low_watermark = max(0.0, min(self._queue_high_watermark, self._queue_low_watermark))
        self._reducer_timeout = float(exec_cfg.get("reducer_timeout", 30.0) or 30.0)
        self._latency_ema = 0.0
        self._bytes_ema = 0.0
        self.batch_details: dict[int, dict[str, Any]] = {}

        registry_backend = getattr(config, "DATA_REGISTRY", {}).get("backend", "auto")
        logger.info(
            "Window %s registered (%d assets: %d with data, %d without) initial_batch=%d registry_backend=%s",
            self._window_id,
            len(self._descriptors),
            len(self._assets_with_data),
            len(self._assets_without_data),
            self._batch_size,
            registry_backend,
        )

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
    def close(self) -> None:
        """Release shared resources (registry descriptors)."""

        if not self._window_released:
            logger.debug(
                "Releasing registry window %s (%d assets: %d with data, %d without)",
                self._window_id,
                len(self._descriptors),
                len(self._assets_with_data),
                len(self._assets_without_data),
            )
            data_registry.release_window(self._window_id)
            self._window_released = True

    # ------------------------------------------------------------------
    def shutdown(self) -> None:
        """Alias for :meth:`close` to match executor semantics."""

        self.close()

    # ------------------------------------------------------------------
    def __del__(self):  # pragma: no cover - defensive cleanup
        try:
            self.close()
        except Exception:
            pass

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
    @staticmethod
    def _compose_reason_detail(
        message: str | None, indicator: str | None = None
    ) -> str | None:
        """Combine indicator metadata and error text into a detail string."""

        parts: list[str] = []
        if indicator:
            parts.append(str(indicator))
        if message:
            msg = str(message)
            if msg:
                parts.append(msg)
        if not parts:
            return None
        return " | ".join(parts)

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
    def _adjust_batching(
        self,
        latencies: list[float],
        max_pending: int,
        avg_batch_bytes: float,
        submitted_batches: int,
    ) -> None:
        """Adapt batch size based on latency, queue pressure, and cooldowns."""

        self._submissions_since_adjustment += max(0, submitted_batches)

        if latencies:
            mean_latency = sum(latencies) / len(latencies)
            if self._latency_ema <= 0:
                self._latency_ema = mean_latency
            else:
                self._latency_ema = 0.6 * self._latency_ema + 0.4 * mean_latency
        else:
            self._latency_ema *= 0.9

        if avg_batch_bytes > 0:
            if self._bytes_ema <= 0:
                self._bytes_ema = avg_batch_bytes
            else:
                self._bytes_ema = 0.7 * self._bytes_ema + 0.3 * avg_batch_bytes
        else:
            self._bytes_ema *= 0.9

        in_flight_cap = max(1, global_executor.current_in_flight_cap())
        high_water = max(1, int(self._queue_high_watermark * in_flight_cap))
        low_water = max(0, int(self._queue_low_watermark * in_flight_cap))

        desired = self._batch_size
        direction = 0
        step = max(1, math.ceil(self._batch_size * self._batch_adjustment_rate))

        if max_pending >= high_water or self._latency_ema > self._latency_target * 1.5:
            desired = max(self._min_batch_size, self._batch_size - step)
            direction = -1 if desired < self._batch_size else 0
        elif max_pending <= low_water and self._latency_ema < self._latency_target * 0.7:
            desired = min(self._max_batch_size, self._batch_size + step)
            direction = 1 if desired > self._batch_size else 0
        elif (
            self._latency_ema < self._latency_target * 0.5
            and self._batch_size < self._max_batch_size
        ):
            desired = min(self._max_batch_size, self._batch_size + step)
            direction = 1 if desired > self._batch_size else 0

        if direction and desired != self._batch_size:
            if self._submissions_since_adjustment < self._batch_cooldown_submissions:
                logger.debug(
                    "Skipping batch adjustment (cooldown %d/%d, pending=%d/%d)",
                    self._submissions_since_adjustment,
                    self._batch_cooldown_submissions,
                    max_pending,
                    in_flight_cap,
                )
                return
            logger.debug(
                "Adjusting batch size from %d to %d (latency=%.3fs pending=%d/%d)",
                self._batch_size,
                desired,
                self._latency_ema,
                max_pending,
                in_flight_cap,
            )
            self._batch_size = desired
            self._submissions_since_adjustment = 0

    # ------------------------------------------------------------------
    def collect_generation_report(self) -> dict[str, Any]:
        """Aggregate instrumentation collected since the last generation."""

        if not self._generation_records:
            return {}

        records = self._generation_records
        self._generation_records = []

        latencies: list[float] = []
        total_latency = 0.0
        for record in records:
            sample = record.get("latency") or []
            if sample:
                latencies.extend(sample)
                total_latency += sum(sample)

        total_evaluations = sum(r.get("evaluations", 0) for r in records)
        total_cpu = sum(r.get("cpu_time", 0.0) for r in records)
        submitted = sum(r.get("submitted", 0) for r in records)
        completed = sum(r.get("completed", 0) for r in records)
        max_pending = max((r.get("max_pending", 0) for r in records), default=0)
        serialization_bytes = sum(r.get("serialization_bytes", 0) for r in records)
        rows_processed = sum(r.get("rows_processed", 0) for r in records)
        reducer_timeouts = sum(r.get("reducer_timeouts", 0) for r in records)

        mean_latency = sum(latencies) / len(latencies) if latencies else 0.0
        p95_latency = float(np.percentile(latencies, 95)) if latencies else 0.0
        throughput = (
            total_evaluations / total_latency if total_latency > 0 else 0.0
        )
        occupancy = total_cpu / total_latency if total_latency > 0 else 0.0

        error_counts = Counter()
        for record in records:
            error_counts.update(record.get("error_counts", {}))

        last = records[-1]
        return {
            "submitted": submitted,
            "completed": completed,
            "queue_depth": max_pending,
            "max_pending": max_pending,
            "pending": records[-1].get("pending"),
            "mean_latency": mean_latency,
            "p95_latency": p95_latency,
            "throughput": throughput,
            "cpu_time": total_cpu,
            "occupancy": occupancy,
            "evaluations": total_evaluations,
            "serialization_bytes": serialization_bytes,
            "rows_processed": rows_processed,
            "latency_samples": len(latencies),
            "batch_size": last.get("batch_size"),
            "next_batch_size": last.get("next_batch_size"),
            "in_flight_cap": last.get("in_flight_cap"),
            "base_in_flight_cap": last.get("base_in_flight_cap"),
            "bytes_avg": last.get("bytes_avg"),
            "worker_count": last.get("worker_count"),
            "worker_seeds": list(last.get("worker_seeds", [])),
            "latency_ema": last.get("latency_ema"),
            "bytes_ema": last.get("bytes_ema"),
            "reducer_timeouts": reducer_timeouts,
            "error_top": error_counts.most_common(5),
        }

    # ------------------------------------------------------------------
    def _evaluate_population(
        self, solutions: list[list[Any]], indices: list[int]
    ) -> tuple[list[float], Counter]:
        """Evaluate a batch of candidate solutions across all assets."""

        verbose = bool(self.settings.get("verbose_asset_errors"))
        candidate_state: dict[int, dict[str, dict]] = {}
        for idx in indices:
            candidate_state[idx] = {"assets": {}}
            for asset in self._assets_without_data:
                candidate_state[idx]["assets"][asset] = self._build_evaluation_record(
                    reason="insufficient_coverage"
                )

        payload = [
            {"index": idx, "vector": sol}
            for idx, sol in zip(indices, solutions, strict=False)
        ]

        current_batch_size = self._batch_size
        futures: set[cf.Future] = set()
        future_meta: dict[Any, dict[str, Any]] = {}
        tasks_submitted = 0
        serialization_volume = 0
        latencies: list[float] = []
        rows_processed = 0
        evaluations_processed = 0
        err_counts: Counter = Counter()
        reducer_timeouts = 0

        before_metrics = global_executor.metrics()
        before_submitted = int(before_metrics.get("submitted", 0))
        before_completed = int(before_metrics.get("completed", 0))
        before_runtime = float(before_metrics.get("total_runtime", 0.0))

        for asset in self._assets_with_data:
            descriptor = self._descriptors[asset]
            for start in range(0, len(payload), current_batch_size):
                chunk = payload[start : start + current_batch_size]
                if not chunk:
                    continue
                serialization_volume += sum(len(item["vector"]) for item in chunk) * 8
                future = global_executor.submit(
                    fitness_worker.evaluate_batch,
                    descriptor,
                    self.base_rules,
                    self.gene_map,
                    chunk,
                    {"collect_equity_curve": self.settings.get("collect_equity_curve")},
                )
                futures.add(future)
                future_meta[future] = {
                    "asset": asset,
                    "indices": [item["index"] for item in chunk],
                }
                tasks_submitted += 1

        pending = set(futures)
        while pending:
            done, pending = cf.wait(
                pending,
                timeout=self._reducer_timeout,
                return_when=cf.FIRST_COMPLETED,
            )
            if not done:
                reducer_timeouts += 1
                log_fn = logger.warning if reducer_timeouts == 1 else logger.error
                log_fn(
                    "Reducer timeout waiting for %d batches (window=%s, batch=%d, attempt=%d)",
                    len(pending),
                    self._window_id,
                    current_batch_size,
                    reducer_timeouts,
                )
                continue

            for future in done:
                meta = future_meta.get(future, {})
                asset = meta.get("asset")
                indices_chunk = meta.get("indices", [])
                try:
                    result = future.result()
                except Exception as exc:  # pragma: no cover - executor exceptions
                    detail = repr(exc)
                    for idx in indices_chunk:
                        candidate_state[idx]["assets"][asset] = self._build_evaluation_record(
                            reason="evaluation_error",
                            detail=detail,
                        )
                        if verbose:
                            print(f"Error evaluating asset {asset}: {exc}")
                    continue

                latency_val = float(result.get("latency", 0.0))
                if latency_val:
                    latencies.append(latency_val)
                rows_processed += int(result.get("rows", 0))
                batch_bytes = int(result.get("bytes", 0))
                serialization_volume += batch_bytes
                global_executor.record_batch_metrics(batch_bytes)

                for entry in result.get("results", []):
                    sol_idx = entry.get("sol_idx")
                    if sol_idx not in candidate_state:
                        continue
                    evaluations_processed += 1
                    assets_map = candidate_state[sol_idx]["assets"]
                    if entry.get("reason") == "insufficient_coverage":
                        assets_map[asset] = self._build_evaluation_record(
                            reason="insufficient_coverage"
                        )
                        continue
                    error_payload = entry.get("error")
                    if error_payload:
                        if verbose:
                            msg = error_payload.get("message", "")
                            print(f"Error evaluating asset {asset}: {msg}")
                        indicator = error_payload.get("indicator")
                        if indicator:
                            err_counts[indicator] += 1
                        detail = self._compose_reason_detail(
                            error_payload.get("message"), indicator
                        )
                        assets_map[asset] = self._build_evaluation_record(
                            reason="evaluation_error",
                            detail=detail,
                            trace=error_payload.get("trace"),
                        )
                        continue

                    stats_raw = entry.get("stats") or self._empty_stats()
                    stats, reason, detail = self._prepare_metrics_record(stats_raw)
                    assets_map[asset] = self._build_evaluation_record(
                        stats,
                        reason=reason,
                        detail=detail,
                    )

        after_metrics = global_executor.metrics()
        submitted = int(after_metrics.get("submitted", 0)) - before_submitted
        completed = int(after_metrics.get("completed", 0)) - before_completed
        cpu_time = float(after_metrics.get("total_runtime", 0.0)) - before_runtime
        total_latency = sum(latencies)
        throughput = (
            evaluations_processed / total_latency if total_latency > 0 else 0.0
        )
        occupancy = cpu_time / total_latency if total_latency > 0 else 0.0
        max_pending = int(after_metrics.get("max_pending", 0))
        in_flight_cap = int(after_metrics.get("in_flight_cap", 0))
        base_in_flight_cap = int(after_metrics.get("base_in_flight_cap", in_flight_cap))
        pending_now = int(after_metrics.get("pending", max_pending))
        bytes_avg = float(after_metrics.get("bytes_avg", 0.0))
        mean_latency = sum(latencies) / len(latencies) if latencies else 0.0
        p95_latency = float(np.percentile(latencies, 95)) if latencies else 0.0

        self.instrumentation = {
            "tasks_submitted": tasks_submitted,
            "evaluations": evaluations_processed,
            "latency": list(latencies),
            "latency_mean": mean_latency,
            "latency_p95": p95_latency,
            "latency_target": self._latency_target,
            "throughput": throughput,
            "cpu_time": cpu_time,
            "occupancy": occupancy,
            "pending": pending_now,
            "max_pending": max_pending,
            "queue_depth": max_pending,
            "queue_ratio": max_pending / max(1, in_flight_cap),
            "serialization_bytes": serialization_volume,
            "rows_processed": rows_processed,
            "submitted": submitted,
            "completed": completed,
            "batch_size": current_batch_size,
            "next_batch_size": self._batch_size,
            "in_flight_cap": in_flight_cap,
            "base_in_flight_cap": base_in_flight_cap,
            "bytes_avg": bytes_avg,
            "worker_count": after_metrics.get("worker_count"),
            "worker_seeds": list(after_metrics.get("worker_seeds", [])),
            "reducer_timeouts": reducer_timeouts,
            "error_counts": dict(err_counts),
            "error_top": err_counts.most_common(5),
        }

        self._adjust_batching(latencies, max_pending, bytes_avg, tasks_submitted)
        self.instrumentation["next_batch_size"] = self._batch_size
        self.instrumentation["latency_ema"] = self._latency_ema
        self.instrumentation["bytes_ema"] = self._bytes_ema
        record = dict(self.instrumentation)
        record["latency"] = list(latencies)
        self._generation_records.append(record)

        fitness_values: list[float] = []
        for idx in indices:
            assets_map = candidate_state[idx]["assets"]
            for ticker in self._sorted_tickers:
                assets_map.setdefault(ticker, self._build_evaluation_record())
            summary = self._score_assets(assets_map)
            score = self._aggregate_scores(summary)
            fitness_values.append(score)
            self.batch_details[idx] = copy.deepcopy(self.last_details)

        return fitness_values, err_counts

    # ------------------------------------------------------------------
    def _evaluate_assets(
        self, overrides: dict | None = None
    ) -> tuple[dict[str, dict], Counter]:
        """Sequential compatibility layer for per-asset evaluation.

        This method mirrors the pre-executor behaviour and is primarily used by
        tests that patch :meth:`_evaluate_single_asset` directly.
        """

        overrides = overrides or {}
        vector = overrides.get("vector")
        if vector is None and "solution" in overrides:
            vector = overrides["solution"]
        rules_override = overrides.get("rules")

        if vector is not None:
            vector_list = np.asarray(vector).tolist()
            rules = inject_genes_into_rules(
                self.base_rules, self.gene_map, vector_list
            )
        elif rules_override is not None:
            rules = rules_override
        else:
            rules = self.base_rules

        results: dict[str, dict] = {}
        err_counts: Counter = Counter()

        for asset in self._sorted_tickers:
            descriptor = self._descriptors.get(asset)
            if descriptor and descriptor.get("empty"):
                results[asset] = self._build_evaluation_record(
                    reason="insufficient_coverage"
                )
                continue

            try:
                if descriptor:
                    ohlc = data_registry.attach(descriptor)
                else:
                    ohlc = self.group_data.get(asset)
            except Exception as exc:  # pragma: no cover - defensive path
                trace = traceback.format_exception(
                    exc.__class__, exc, exc.__traceback__
                )
                results[asset] = self._build_evaluation_record(
                    reason="evaluation_error",
                    detail=str(exc),
                    trace="".join(trace),
                )
                continue

            try:
                stats = self._evaluate_single_asset(ohlc, rules)
            except Exception as exc:  # noqa: BLE001 - propagate indicator info
                indicator = getattr(exc, "indicator", None)
                if indicator:
                    err_counts[indicator] += 1
                detail = self._compose_reason_detail(str(exc), indicator)
                trace = traceback.format_exception(
                    exc.__class__, exc, exc.__traceback__
                )
                results[asset] = self._build_evaluation_record(
                    reason="evaluation_error",
                    detail=detail,
                    trace="".join(trace),
                )
                continue

            stats, reason, detail = self._prepare_metrics_record(stats)
            results[asset] = self._build_evaluation_record(
                stats, reason=reason, detail=detail
            )

        return results, err_counts

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
    def _evaluate_single_asset(self, ohlc: pd.DataFrame, rules: dict) -> dict:
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

        # Record the actual executed trades using vectorbt.
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
        }

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
            if pf_raw is None or pd.isna(pf_raw):
                pf_capped = nan_fallback
            else:
                pf_capped = cap if np.isinf(pf_raw) else min(cap, float(pf_raw))

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
                    if reason is not None:
                        details["evaluation_reason"] = reason
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
                    if reason is not None:
                        details["evaluation_reason"] = reason
                    if metric_sources and not sources_recorded:
                        details["metric_sources"] = metric_sources
                        sources_recorded = True
                        if run_metric_sources is None:
                            run_metric_sources = dict(metric_sources)
                    per_asset_details[ticker] = details
                continue

            if metric_type == "sortino":
                val = stats.get("sortino")
                if val is None or pd.isna(val):
                    val = nan_fallback
            elif metric_type == "profit_factor":
                val = pf_capped
            elif metric_type == "return":
                total_return = stats.get("total_return")
                val = (
                    nan_fallback
                    if total_return is None or pd.isna(total_return)
                    else total_return
                )
            else:
                sortino_val = stats.get("sortino")
                if sortino_val is None or pd.isna(sortino_val):
                    sortino_val = nan_fallback
                max_dd = stats.get("max_drawdown")
                if max_dd is None or pd.isna(max_dd):
                    max_dd = 100.0
                drawdown_score = 1 - (max_dd / 100.0)
                w = config.FITNESS_WEIGHTS
                val = (
                    sortino_val * w["sortino_ratio"]
                    + pf_capped * w["profit_factor"]
                    + drawdown_score * w["max_drawdown"]
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
            if reason is not None:
                details["evaluation_reason"] = reason
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
        }

        return F

    # ------------------------------------------------------------------
    def __call__(self, ga_instance, solution, sol_idx):
        config.initialize_config()
        try:
            if isinstance(sol_idx, (list, tuple, np.ndarray)):
                indices = list(np.asarray(sol_idx).astype(int))
                solutions_arr = np.asarray(solution)
                solutions_list = [np.asarray(row).tolist() for row in solutions_arr]
                scores, err_counts = self._evaluate_population(solutions_list, indices)
                if err_counts:
                    logger.info("evaluation error counts: %s", dict(err_counts))
                return scores

            idx = int(sol_idx)
            vector = np.asarray(solution).tolist()
            scores, err_counts = self._evaluate_population([vector], [idx])
            if err_counts:
                logger.info("evaluation error counts: %s", dict(err_counts))
            return scores[0]

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
        if isinstance(ohlc_data, pd.DataFrame):
            group_data = {"asset_0": ohlc_data}
        elif isinstance(ohlc_data, Mapping):
            group_data = dict(ohlc_data)
        else:
            raise TypeError(
                "Multi-asset fitness requires a DataFrame or mapping of DataFrames"
            )
        return MultiAssetFitnessEvaluator(group_data, base_rules, gene_map, settings)
    return FitnessEvaluator(ohlc_data, base_rules, gene_map)
