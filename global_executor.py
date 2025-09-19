"""Single, process-based executor shared across the application.

This module is responsible for provisioning a single global
``concurrent.futures.ProcessPoolExecutor`` instance that is reused across
fitness evaluation calls.  It enforces deterministic seeding, constrains the
number of native threads used by numerical libraries, and provides simple
back-pressure so the queue of in-flight tasks never grows without bound.

The public API intentionally mirrors a subset of ``concurrent.futures`` so the
rest of the codebase can submit tasks without knowing about the underlying
implementation details.  The executor is lazily constructed on first use using
the configuration exposed via :mod:`config` under ``GLOBAL_EXECUTOR``.
"""

from __future__ import annotations

import concurrent.futures as cf
import logging
import multiprocessing as mp
import os
import threading
import time
from collections.abc import Callable
from typing import Any, TypedDict

import config

_THREAD_ENV_VARS = {
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
}

logger = logging.getLogger(__name__)

_executor: cf.ProcessPoolExecutor | None = None
_executor_settings: dict[str, Any] | None = None
_executor_lock = threading.Lock()
_pending_cond = threading.Condition(threading.Lock())
_pending_tasks = 0
_in_flight_cap = 0
_base_in_flight_cap = 0
_memory_target_bytes = 0
_avg_batch_bytes = 0.0
_future_starts: dict[cf.Future, float] = {}


class ExecutorMetrics(TypedDict):
    submitted: int
    completed: int
    total_runtime: float
    pending: int
    max_pending: int
    in_flight_cap: int
    base_in_flight_cap: int
    bytes_avg: float
    worker_count: int
    worker_seeds: list[int]


_metrics: ExecutorMetrics = {
    "submitted": 0,
    "completed": 0,
    "total_runtime": 0.0,
    "pending": 0,
    "max_pending": 0,
    "in_flight_cap": 0,
    "base_in_flight_cap": 0,
    "bytes_avg": 0.0,
    "worker_count": 0,
    "worker_seeds": [],
}
_cpu_affinity: set[int] | None = None


def _apply_thread_env() -> None:
    """Force heavy numerical libraries to use a single native thread."""

    for var in _THREAD_ENV_VARS:
        os.environ[var] = "1"


def _worker_initializer(seed: int) -> None:
    """Initialise worker processes with deterministic state."""

    _apply_thread_env()
    if _cpu_affinity:
        try:
            os.sched_setaffinity(0, _cpu_affinity)
            logger.info(
                "worker %s affinity=%s",
                mp.current_process().name,
                sorted(_cpu_affinity),
            )
        except Exception:  # pragma: no cover - affinity is best effort
            logger.debug("CPU affinity unavailable on this platform", exc_info=True)
    try:  # pragma: no cover - defensive seeding path
        import random

        import numpy as np
    except Exception:  # pragma: no cover - minimal fallback
        return

    proc = mp.current_process()
    identity = getattr(proc, "identity", None)
    offset = int(identity[0]) if identity else 0
    derived_seed = int(seed) + offset
    logger.info("worker %s seed=%d", getattr(proc, "name", "unknown"), derived_seed)
    random.seed(derived_seed)
    np.random.seed(derived_seed)

    try:
        config.initialize_config()
    except Exception:  # pragma: no cover - best effort initialisation
        pass

    try:
        from fitness_worker import warm_up

        warm_up()
    except Exception:  # pragma: no cover - warm-up is best effort
        pass


def _load_settings(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    config.initialize_config()
    base = dict(getattr(config, "GLOBAL_EXECUTOR", {}))
    if overrides:
        base.update(overrides)
    return base


def _on_future_done(fut: cf.Future) -> None:
    end = time.perf_counter()
    start = _future_starts.pop(fut, None)
    duration = end - start if start is not None else 0.0
    with _pending_cond:
        global _pending_tasks
        _pending_tasks = max(0, _pending_tasks - 1)
        _metrics["completed"] += 1
        _metrics["total_runtime"] += duration
        _metrics["pending"] = _pending_tasks
        _pending_cond.notify_all()


def create(
    overrides: dict[str, Any] | None = None, *, force: bool = False
) -> cf.ProcessPoolExecutor:
    """Create (or return) the global executor.

    Parameters
    ----------
    overrides:
        Optional overrides applied on top of ``config.GLOBAL_EXECUTOR``.
    force:
        When ``True`` the executor is re-created even if the settings have not
        changed.  Primarily used by tests.
    """

    global _executor, _executor_settings, _in_flight_cap, _cpu_affinity
    global _memory_target_bytes, _avg_batch_bytes, _base_in_flight_cap

    settings = _load_settings(overrides)
    start_method = settings.get("start_method") or "spawn"
    cpu_total = os.cpu_count() or 1
    os_reserve = int(settings.get("os_reserve") or 0)
    configured_workers = settings.get("max_workers")
    max_workers = int(configured_workers or max(1, cpu_total - os_reserve))
    if max_workers <= 0:
        max_workers = 1
    seed = int(settings.get("seed") or config.SEED)
    affinity_opt = settings.get("cpu_affinity")
    affinity_set: set[int] | None = None
    if affinity_opt:
        if isinstance(affinity_opt, int):
            affinity_set = {int(affinity_opt)}
        else:
            try:
                affinity_set = {int(core) for core in affinity_opt}
            except TypeError:
                affinity_set = None

    with _executor_lock:
        if _executor is not None and not force and settings == _executor_settings:
            return _executor

        if _executor is not None:
            shutdown(wait=True)

        _apply_thread_env()
        _cpu_affinity = affinity_set

        try:
            ctx = mp.get_context(start_method)
        except ValueError:
            ctx = mp.get_context("spawn")

        executor = cf.ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=ctx,
            initializer=_worker_initializer,
            initargs=(seed,),
        )
        seeds = [int(seed) + i for i in range(max_workers)]
        _in_flight_cap = int(settings.get("in_flight_cap") or max_workers * 8)
        _base_in_flight_cap = max(1, _in_flight_cap)
        _memory_target_bytes = int(
            float(settings.get("memory_target_gib") or 0) * (1024**3)
        )
        _avg_batch_bytes = 0.0
        with _pending_cond:
            global _pending_tasks
            _pending_tasks = 0
            _metrics["submitted"] = 0
            _metrics["completed"] = 0
            _metrics["total_runtime"] = 0.0
            _metrics["pending"] = 0
            _metrics["max_pending"] = 0
            _metrics["in_flight_cap"] = _in_flight_cap
            _metrics["base_in_flight_cap"] = _base_in_flight_cap
            _metrics["bytes_avg"] = 0.0
            _metrics["worker_count"] = max_workers
            _metrics["worker_seeds"] = list(seeds)
            _future_starts.clear()

        _executor = executor
        _executor_settings = settings
        logger.info(
            "Global executor initialised with %d workers (cpu=%d reserve=%d, start_method=%s)",
            max_workers,
            cpu_total,
            os_reserve,
            start_method,
        )
        if affinity_set:
            logger.info("Executor affinity pinned to cores: %s", sorted(affinity_set))
        logger.info("Executor worker seeds: %s", ", ".join(str(s) for s in seeds))
        return executor


def submit(
    fn: Callable[..., Any],
    *args: Any,
    overrides: dict[str, Any] | None = None,
    **kwargs: Any,
) -> cf.Future:
    """Submit ``fn`` to the global executor with basic back-pressure."""

    executor = create(overrides)
    with _pending_cond:
        global _pending_tasks
        while _pending_tasks >= _in_flight_cap:
            _pending_cond.wait()
        future = executor.submit(fn, *args, **kwargs)
        _pending_tasks += 1
        _metrics["submitted"] += 1
        _metrics["pending"] = _pending_tasks
        _metrics["max_pending"] = max(_metrics["max_pending"], _pending_tasks)
        _metrics["in_flight_cap"] = _in_flight_cap
        _future_starts[future] = time.perf_counter()
        future.add_done_callback(_on_future_done)
        return future


def pending() -> int:
    """Return the number of in-flight futures."""

    with _pending_cond:
        return _pending_tasks


def metrics() -> dict[str, Any]:
    """Return a snapshot of executor instrumentation."""

    with _pending_cond:
        return dict(_metrics)


def current_in_flight_cap() -> int:
    """Expose the current in-flight cap for adaptive schedulers."""

    with _pending_cond:
        return _in_flight_cap


def record_batch_metrics(bytes_count: int) -> int:
    """Update moving averages and adjust the in-flight cap if needed."""

    if bytes_count < 0:
        bytes_count = 0

    with _pending_cond:
        global _avg_batch_bytes, _in_flight_cap

        if bytes_count > 0:
            if _avg_batch_bytes <= 0:
                _avg_batch_bytes = float(bytes_count)
            else:
                _avg_batch_bytes = 0.8 * _avg_batch_bytes + 0.2 * float(bytes_count)
        else:
            _avg_batch_bytes *= 0.95

        target_cap = _base_in_flight_cap or _in_flight_cap or 1
        if _memory_target_bytes and _avg_batch_bytes > 0:
            budget_cap = max(1, int(_memory_target_bytes / max(_avg_batch_bytes, 1.0)))
            target_cap = min(target_cap, budget_cap)

        if target_cap < _in_flight_cap:
            logger.info(
                "Reducing in-flight cap from %d to %d (avg batch=%.2f MiB, target=%.2f GiB)",
                _in_flight_cap,
                target_cap,
                _avg_batch_bytes / (1024**2),
                (_memory_target_bytes or 0) / (1024**3),
            )
            _in_flight_cap = max(1, target_cap)
        elif target_cap > _in_flight_cap and _in_flight_cap < (
            _base_in_flight_cap or target_cap
        ):
            proposed = min((_base_in_flight_cap or target_cap), _in_flight_cap + 1)
            if not _memory_target_bytes or (_avg_batch_bytes * proposed) < max(
                _memory_target_bytes * 0.8, 1.0
            ):
                _in_flight_cap = max(1, proposed)

        _metrics["bytes_avg"] = _avg_batch_bytes
        _metrics["in_flight_cap"] = _in_flight_cap
        _metrics["base_in_flight_cap"] = _base_in_flight_cap
        return _in_flight_cap


def shutdown(wait: bool = True) -> None:
    """Shut down the executor and reset shared state."""

    global _executor, _executor_settings, _cpu_affinity
    with _executor_lock:
        executor = _executor
        _executor = None
        _executor_settings = None
        _cpu_affinity = None
    if executor is not None:
        executor.shutdown(wait=wait)
    with _pending_cond:
        global _pending_tasks
        _pending_tasks = 0
        global _avg_batch_bytes, _base_in_flight_cap, _in_flight_cap, _memory_target_bytes
        _avg_batch_bytes = 0.0
        _base_in_flight_cap = 0
        _in_flight_cap = 0
        _memory_target_bytes = 0
        _metrics["submitted"] = 0
        _metrics["completed"] = 0
        _metrics["total_runtime"] = 0.0
        _metrics["pending"] = 0
        _metrics["max_pending"] = 0
        _metrics["in_flight_cap"] = 0
        _metrics["base_in_flight_cap"] = 0
        _metrics["bytes_avg"] = 0.0
        _metrics["worker_count"] = 0
        _metrics["worker_seeds"] = []
        _future_starts.clear()
