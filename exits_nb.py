"""Event-driven exit simulator for trade management genes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Mapping, Sequence

import numpy as np

import config

try:  # pragma: no cover - numba is optional in some environments
    from numba import njit
except Exception:  # pragma: no cover - provide a no-op decorator

    def njit(*args, **kwargs):  # type: ignore[override]
        def decorator(func):
            return func

        return decorator


class ExitReason(IntEnum):
    """Enumerated exit reasons returned by the simulator."""

    NONE = 0
    TP1 = 1
    TP2 = 2
    TP3 = 3
    TP4 = 4
    SL = 5
    SL_TIMEOUT = 6
    TSL = 7
    TTP = 8


class BreakEvenMode(IntEnum):
    """Supported stop adjustment modes."""

    NONE = 0
    BREAKEVEN = 1
    FOLLOW_TP = 2


MAX_TP_LEVELS = 4
MAX_REASON_BUCKETS = int(max(reason.value for reason in ExitReason)) + 1

_BREAK_EVEN_MODE_TO_NAME = {
    BreakEvenMode.NONE: "none",
    BreakEvenMode.BREAKEVEN: "breakeven",
    BreakEvenMode.FOLLOW_TP: "follow_tp",
}

_BREAK_EVEN_NAME_TO_MODE = {
    "none": BreakEvenMode.NONE,
    "breakeven": BreakEvenMode.BREAKEVEN,
    "follow_tp": BreakEvenMode.FOLLOW_TP,
}

_BREAK_EVEN_NUMERIC_VALUES = {mode.value for mode in BreakEvenMode}

REASON_NONE = int(ExitReason.NONE)
REASON_TP1 = int(ExitReason.TP1)
REASON_TP2 = int(ExitReason.TP2)
REASON_TP3 = int(ExitReason.TP3)
REASON_TP4 = int(ExitReason.TP4)
REASON_SL = int(ExitReason.SL)
REASON_SL_TIMEOUT = int(ExitReason.SL_TIMEOUT)
REASON_TSL = int(ExitReason.TSL)
REASON_TTP = int(ExitReason.TTP)

BE_MODE_NONE = int(BreakEvenMode.NONE)
BE_MODE_BREAKEVEN = int(BreakEvenMode.BREAKEVEN)
BE_MODE_FOLLOW_TP = int(BreakEvenMode.FOLLOW_TP)


@dataclass(slots=True)
class ExitParams:
    """Resolved trade-management parameters for a single strategy run."""

    stop_loss_pct: float
    num_tp_levels: int
    tp_pcts: Sequence[float]
    tp_trailing_enabled: bool
    tp_trailing_pct: float
    sl_timeout_enabled: bool
    sl_timeout_bars: int
    sl_break_even_mode: BreakEvenMode
    sl_trailing_enabled: bool
    sl_trailing_pct: float
    max_hold_bars: int
    tp_cap: float
    timeframe: str | None = None

    def as_dict(self) -> dict:
        """Serialise the parameter payload for metadata snapshots."""

        payload = {
            "stop_loss_pct": float(self.stop_loss_pct),
            "num_tp_levels": int(self.num_tp_levels),
            "tp_pcts": [float(v) for v in self.tp_pcts[:MAX_TP_LEVELS]],
            "tp_trailing_enabled": bool(self.tp_trailing_enabled),
            "tp_trailing_pct": float(self.tp_trailing_pct),
            "sl_timeout_enabled": bool(self.sl_timeout_enabled),
            "sl_timeout_bars": int(self.sl_timeout_bars),
            "sl_break_even_mode": _BREAK_EVEN_MODE_TO_NAME.get(
                self.sl_break_even_mode, "none"
            ),
            "sl_trailing_enabled": bool(self.sl_trailing_enabled),
            "sl_trailing_pct": float(self.sl_trailing_pct),
            "max_hold_bars": int(self.max_hold_bars),
            "tp_cap": float(self.tp_cap),
        }
        payload["timeframe"] = self.timeframe
        return payload


@dataclass(slots=True)
class ExitResult:
    """Container for simulator outputs and optional telemetry traces."""

    exits: np.ndarray
    exit_size: np.ndarray
    exit_reason: np.ndarray
    reason_fractions: np.ndarray | None
    current_stop: np.ndarray | None
    trailing_tp_trigger: np.ndarray | None
    sl_breach_bar: np.ndarray | None
    tp_progress: np.ndarray | None
    reason_counts: np.ndarray
    reason_totals: np.ndarray
    timeout_breach_bars_sum: np.ndarray
    timeout_event_count: np.ndarray
    tp_level_reached_sum: np.ndarray
    tp_level_reached_count: np.ndarray
    breakeven_trade_count: np.ndarray
    trailing_tp_trade_count: np.ndarray
    trade_count: np.ndarray


@njit(cache=True)
def _simulate_dynamic_exits(
    entries: np.ndarray,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    stop_loss_pct: float,
    tp_pcts: np.ndarray,
    num_tp_levels: int,
    tp_trailing_enabled: bool,
    tp_trailing_pct: float,
    sl_timeout_enabled: bool,
    sl_timeout_bars: int,
    sl_break_even_mode: int,
    sl_trailing_enabled: bool,
    sl_trailing_pct: float,
    max_hold_bars: int,
    collect_traces: bool,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    n_rows, n_assets = entries.shape
    exits = np.zeros((n_rows, n_assets), dtype=np.bool_)
    exit_size = np.zeros((n_rows, n_assets), dtype=np.float64)
    exit_reason = np.zeros((n_rows, n_assets), dtype=np.int64)

    if collect_traces:
        reason_fractions = np.zeros(
            (n_rows, n_assets, MAX_REASON_BUCKETS), dtype=np.float64
        )
        current_stop_trace = np.full((n_rows, n_assets), np.nan, dtype=np.float64)
        trailing_tp_trace = np.full((n_rows, n_assets), np.nan, dtype=np.float64)
        sl_breach_trace = np.full((n_rows, n_assets), -1.0, dtype=np.float64)
        tp_progress_trace = np.zeros((n_rows, n_assets), dtype=np.int64)
    else:
        reason_fractions = np.empty((0, 0, 0), dtype=np.float64)
        current_stop_trace = np.empty((0, 0), dtype=np.float64)
        trailing_tp_trace = np.empty((0, 0), dtype=np.float64)
        sl_breach_trace = np.empty((0, 0), dtype=np.float64)
        tp_progress_trace = np.empty((0, 0), dtype=np.int64)

    reason_count_totals = np.zeros((n_assets, MAX_REASON_BUCKETS), dtype=np.float64)
    reason_volume_totals = np.zeros((n_assets, MAX_REASON_BUCKETS), dtype=np.float64)
    timeout_duration_sum = np.zeros(n_assets, dtype=np.float64)
    timeout_event_count = np.zeros(n_assets, dtype=np.float64)
    tp_level_sum = np.zeros(n_assets, dtype=np.float64)
    tp_level_count = np.zeros(n_assets, dtype=np.float64)
    breakeven_trade_totals = np.zeros(n_assets, dtype=np.float64)
    trailing_tp_trade_totals = np.zeros(n_assets, dtype=np.float64)
    trade_count_totals = np.zeros(n_assets, dtype=np.float64)

    tp_targets = np.zeros((n_assets, MAX_TP_LEVELS), dtype=np.float64)
    tp_consumed = np.zeros((n_assets, MAX_TP_LEVELS), dtype=np.bool_)

    tp_fraction = 1.0 / num_tp_levels if num_tp_levels > 0 else 1.0
    stop_loss_discount = 1.0 - stop_loss_pct
    sl_trailing_discount = 1.0 - sl_trailing_pct
    tp_trailing_discount = 1.0 - tp_trailing_pct

    for asset in range(n_assets):
        in_trade = False
        entry_price = 0.0
        entry_index = -1
        open_qty = 0.0
        last_tp_hit = 0
        current_stop = np.nan
        current_stop_is_trailing = False
        trailing_tp_active = False
        trailing_tp_trigger = np.nan
        sl_breach_index = -1
        reason_recorded = np.zeros(MAX_REASON_BUCKETS, dtype=np.bool_)
        breakeven_engaged = False

        for row in range(n_rows):
            for reason_idx in range(MAX_REASON_BUCKETS):
                reason_recorded[reason_idx] = False

            trade_closed = False
            signal = entries[row, asset]
            price_close = close[row, asset]
            price_high = high[row, asset]
            price_low = low[row, asset]

            exit_fraction = 0.0
            reason_code = REASON_NONE

            if signal and not in_trade:
                in_trade = True
                entry_price = price_close
                entry_index = row
                open_qty = 1.0
                last_tp_hit = 0
                sl_breach_index = -1
                trailing_tp_active = False
                trailing_tp_trigger = entry_price
                current_stop_is_trailing = False
                breakeven_engaged = False
                for level in range(MAX_TP_LEVELS):
                    tp_consumed[asset, level] = False
                    if level < num_tp_levels:
                        tp_targets[asset, level] = entry_price * (1.0 + tp_pcts[level])
                    else:
                        tp_targets[asset, level] = entry_price * (
                            1.0 + tp_pcts[num_tp_levels - 1]
                        )
                if stop_loss_pct > 0.0:
                    current_stop = entry_price * stop_loss_discount
                else:
                    current_stop = -math.inf
                if collect_traces:
                    current_stop_trace[row, asset] = current_stop
                    trailing_tp_trace[row, asset] = np.nan
                    sl_breach_trace[row, asset] = -1.0
                    tp_progress_trace[row, asset] = 0
                continue

            if not in_trade:
                if collect_traces:
                    current_stop_trace[row, asset] = np.nan
                    trailing_tp_trace[row, asset] = np.nan
                    sl_breach_trace[row, asset] = -1.0
                    tp_progress_trace[row, asset] = 0
                continue

            # Step 1: process static take-profits in ascending order
            for level in range(num_tp_levels):
                if tp_consumed[asset, level]:
                    continue
                target = tp_targets[asset, level]
                if price_high >= target:
                    if tp_trailing_enabled and level == num_tp_levels - 1:
                        tp_consumed[asset, level] = True
                        trailing_tp_active = True
                        trailing_tp_trigger = target * tp_trailing_discount
                        if trailing_tp_trigger < entry_price:
                            trailing_tp_trigger = entry_price
                        last_tp_hit = level + 1
                        reason_code = REASON_NONE
                    else:
                        fraction = tp_fraction
                        if fraction > open_qty:
                            fraction = open_qty
                        open_qty -= fraction
                        tp_consumed[asset, level] = True
                        last_tp_hit = level + 1
                        exit_fraction += fraction
                        reason_code = REASON_TP1 + level
                        if collect_traces:
                            reason_fractions[row, asset, reason_code] += fraction
                        reason_volume_totals[asset, reason_code] += fraction
                        if not reason_recorded[reason_code]:
                            reason_count_totals[asset, reason_code] += 1.0
                            reason_recorded[reason_code] = True
                        if open_qty <= 1e-12:
                            in_trade = False
                            open_qty = 0.0
                            sl_breach_index = -1
                            trailing_tp_active = False
                            trailing_tp_trigger = np.nan
                            current_stop = np.nan
                            current_stop_is_trailing = False
                            trade_closed = True
                            break
            if collect_traces:
                tp_progress_trace[row, asset] = last_tp_hit

            if not in_trade:
                exits[row, asset] = exit_fraction > 0.0
                exit_size[row, asset] = exit_fraction
                exit_reason[row, asset] = reason_code
                if trade_closed:
                    tp_level_sum[asset] += last_tp_hit
                    tp_level_count[asset] += 1.0
                    if breakeven_engaged:
                        breakeven_trade_totals[asset] += 1.0
                    if reason_code == REASON_TTP:
                        trailing_tp_trade_totals[asset] += 1.0
                    trade_count_totals[asset] += 1.0
                    breakeven_engaged = False
                if collect_traces:
                    current_stop_trace[row, asset] = np.nan
                    trailing_tp_trace[row, asset] = np.nan
                    sl_breach_trace[row, asset] = -1.0
                continue

            # Step 2: break-even / follow-TP stop adjustments
            be_adjusted = False
            if last_tp_hit > 0:
                if sl_break_even_mode == BE_MODE_BREAKEVEN:
                    if current_stop < entry_price:
                        current_stop = entry_price
                        current_stop_is_trailing = False
                        be_adjusted = True
                elif sl_break_even_mode == BE_MODE_FOLLOW_TP:
                    follow_idx = last_tp_hit - 1
                    follow_price = tp_targets[asset, follow_idx]
                    if current_stop < follow_price:
                        current_stop = follow_price
                        current_stop_is_trailing = False
                        be_adjusted = True
            if be_adjusted and not breakeven_engaged:
                breakeven_engaged = True

            # Step 3: trailing stop and trailing TP ratchets
            if sl_trailing_enabled:
                candidate = price_high * sl_trailing_discount
                if candidate > current_stop:
                    current_stop = candidate
                    current_stop_is_trailing = True
            if tp_trailing_enabled and trailing_tp_active:
                candidate = price_high * tp_trailing_discount
                if candidate < entry_price:
                    candidate = entry_price
                if math.isnan(trailing_tp_trigger) or candidate > trailing_tp_trigger:
                    trailing_tp_trigger = candidate

            # Step 4: hard stop-loss logic with optional timeout
            if price_low <= current_stop:
                if sl_timeout_enabled:
                    if sl_breach_index == -1:
                        sl_breach_index = row
                    elif row >= sl_breach_index + sl_timeout_bars:
                        fraction = open_qty
                        if fraction > 0.0:
                            exit_fraction += fraction
                            reason = REASON_SL_TIMEOUT
                            reason_code = reason
                            if collect_traces:
                                reason_fractions[row, asset, reason] += fraction
                            reason_volume_totals[asset, reason] += fraction
                            if not reason_recorded[reason]:
                                reason_count_totals[asset, reason] += 1.0
                                reason_recorded[reason] = True
                            if sl_breach_index >= 0:
                                timeout_duration_sum[asset] += row - sl_breach_index
                                timeout_event_count[asset] += 1.0
                            open_qty = 0.0
                            in_trade = False
                            trailing_tp_active = False
                            trailing_tp_trigger = np.nan
                            current_stop = np.nan
                            current_stop_is_trailing = False
                            sl_breach_index = -1
                            trade_closed = True
                else:
                    fraction = open_qty
                    if fraction > 0.0:
                        exit_fraction += fraction
                        if current_stop_is_trailing:
                            reason = REASON_TSL
                        else:
                            reason = REASON_SL
                        reason_code = reason
                        if collect_traces:
                            reason_fractions[row, asset, reason] += fraction
                        reason_volume_totals[asset, reason] += fraction
                        if not reason_recorded[reason]:
                            reason_count_totals[asset, reason] += 1.0
                            reason_recorded[reason] = True
                        open_qty = 0.0
                        in_trade = False
                        trailing_tp_active = False
                        trailing_tp_trigger = np.nan
                        current_stop = np.nan
                        current_stop_is_trailing = False
                        sl_breach_index = -1
                        trade_closed = True
            else:
                sl_breach_index = -1

            # Step 5: trailing take-profit guard
            if in_trade and tp_trailing_enabled and trailing_tp_active:
                if price_low <= trailing_tp_trigger:
                    fraction = open_qty
                    if fraction > 0.0:
                        exit_fraction += fraction
                        reason = REASON_TTP
                        reason_code = reason
                        if collect_traces:
                            reason_fractions[row, asset, reason] += fraction
                        reason_volume_totals[asset, reason] += fraction
                        if not reason_recorded[reason]:
                            reason_count_totals[asset, reason] += 1.0
                            reason_recorded[reason] = True
                        open_qty = 0.0
                        in_trade = False
                        trailing_tp_active = False
                        trailing_tp_trigger = np.nan
                        current_stop = np.nan
                        current_stop_is_trailing = False
                        sl_breach_index = -1
                        trade_closed = True

            # Step 6: max-hold timeout (if enabled)
            if in_trade and max_hold_bars > 0 and entry_index >= 0:
                if row >= entry_index + max_hold_bars:
                    fraction = open_qty
                    if fraction > 0.0:
                        exit_fraction += fraction
                        reason = REASON_SL
                        reason_code = reason
                        if collect_traces:
                            reason_fractions[row, asset, reason] += fraction
                        reason_volume_totals[asset, reason] += fraction
                        if not reason_recorded[reason]:
                            reason_count_totals[asset, reason] += 1.0
                            reason_recorded[reason] = True
                        open_qty = 0.0
                        in_trade = False
                        trailing_tp_active = False
                        trailing_tp_trigger = np.nan
                        current_stop = np.nan
                        current_stop_is_trailing = False
                        sl_breach_index = -1
                        trade_closed = True

            if not in_trade:
                entry_index = -1

            exits[row, asset] = exit_fraction > 1e-12
            exit_size[row, asset] = exit_fraction
            exit_reason[row, asset] = reason_code
            if trade_closed:
                tp_level_sum[asset] += last_tp_hit
                tp_level_count[asset] += 1.0
                if breakeven_engaged:
                    breakeven_trade_totals[asset] += 1.0
                if reason_code == REASON_TTP:
                    trailing_tp_trade_totals[asset] += 1.0
                trade_count_totals[asset] += 1.0
                breakeven_engaged = False
            if collect_traces:
                current_stop_trace[row, asset] = current_stop
                trailing_tp_trace[row, asset] = (
                    trailing_tp_trigger if trailing_tp_active else np.nan
                )
                sl_breach_trace[row, asset] = float(sl_breach_index)

    return (
        exits,
        exit_size,
        exit_reason,
        reason_fractions,
        current_stop_trace,
        trailing_tp_trace,
        sl_breach_trace,
        tp_progress_trace,
        reason_count_totals,
        reason_volume_totals,
        timeout_duration_sum,
        timeout_event_count,
        tp_level_sum,
        tp_level_count,
        breakeven_trade_totals,
        trailing_tp_trade_totals,
        trade_count_totals,
    )


def _ensure_2d(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


def _normalise_price_array(values: np.ndarray | Sequence[float] | None) -> np.ndarray:
    if values is None:
        raise ValueError("OHLC mapping is missing required price arrays")
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


def generate_dynamic_exit_signals_nb(
    open_entries: np.ndarray,
    ohlc: Mapping[str, np.ndarray],
    params: ExitParams,
    fees_slippage: Mapping[str, float] | None = None,
    seed: int | None = None,
    *,
    collect_traces: bool = True,
) -> ExitResult:
    """Generate exit arrays for ``vectorbt`` style backtests.

    Parameters
    ----------
    open_entries:
        Boolean array (1D or 2D) indicating entry signals.
    ohlc:
        Mapping containing ``close``, ``high``, and ``low`` price arrays. The
        arrays may be 1D or 2D and will be aligned with ``open_entries``.
    params:
        Resolved exit parameters.
    fees_slippage:
        Reserved for future use; the simulator itself is price-only and
        deterministic so the payload is ignored today.
    seed:
        Included for forward compatibility; the simulator is deterministic and
        does not use randomness but the argument keeps the call signature stable.
    collect_traces:
        When ``False`` the simulator skips recording per-bar telemetry traces
        (stop levels, trailing triggers, and reason fractions) to reduce memory
        usage. Aggregated statistics are always produced so metadata remains
        available.

    Notes
    -----
    The simulator initialises ``open_qty`` to ``1.0`` for every new position and
    reports ``exit_size`` as the fraction of that baseline quantity.  Callers can
    reinterpret the fractions as a portion of the base order, the remaining
    position, or absolute units depending on their chosen sizing mode.
    """

    entries_arr = np.asarray(open_entries, dtype=np.bool_)
    squeeze = entries_arr.ndim == 1
    entries_2d = _ensure_2d(entries_arr)

    def _get_price(key: str):
        for variant in (key, key.capitalize(), key.upper(), key.title()):
            if variant in ohlc:
                return ohlc[variant]
        return None

    close_arr = _normalise_price_array(_get_price("close"))
    high_arr = _normalise_price_array(_get_price("high"))
    low_arr = _normalise_price_array(_get_price("low"))

    tp_array = np.zeros(MAX_TP_LEVELS, dtype=np.float64)
    for idx in range(MAX_TP_LEVELS):
        if idx < len(params.tp_pcts):
            tp_array[idx] = float(params.tp_pcts[idx])
        else:
            tp_array[idx] = float(params.tp_pcts[-1]) if params.tp_pcts else 0.0

    (
        exits,
        exit_size,
        exit_reason,
        reason_fractions,
        current_stop,
        trailing_tp,
        sl_breach,
        tp_progress,
        reason_counts,
        reason_totals,
        timeout_sum,
        timeout_count,
        tp_level_sum,
        tp_level_count,
        breakeven_totals,
        trailing_tp_totals,
        trade_count_totals,
    ) = _simulate_dynamic_exits(
        entries_2d,
        close_arr,
        high_arr,
        low_arr,
        float(params.stop_loss_pct),
        tp_array,
        int(params.num_tp_levels),
        bool(params.tp_trailing_enabled),
        float(params.tp_trailing_pct),
        bool(params.sl_timeout_enabled),
        int(params.sl_timeout_bars),
        int(params.sl_break_even_mode),
        bool(params.sl_trailing_enabled),
        float(params.sl_trailing_pct),
        int(params.max_hold_bars),
        bool(collect_traces),
    )

    if squeeze:
        exits = exits.reshape(-1)
        exit_size = exit_size.reshape(-1)
        exit_reason = exit_reason.reshape(-1)
        if collect_traces:
            current_stop = current_stop.reshape(-1)
            trailing_tp = trailing_tp.reshape(-1)
            sl_breach = sl_breach.reshape(-1)
            tp_progress = tp_progress.reshape(-1)
            reason_fractions = reason_fractions.reshape(-1, MAX_REASON_BUCKETS)

    if not collect_traces:
        reason_fractions = None
        current_stop = None
        trailing_tp = None
        sl_breach = None
        tp_progress = None

    return ExitResult(
        exits=exits,
        exit_size=exit_size,
        exit_reason=exit_reason,
        reason_fractions=reason_fractions,
        current_stop=current_stop,
        trailing_tp_trigger=trailing_tp,
        sl_breach_bar=sl_breach,
        tp_progress=tp_progress,
        reason_counts=reason_counts,
        reason_totals=reason_totals,
        timeout_breach_bars_sum=timeout_sum,
        timeout_event_count=timeout_count,
        tp_level_reached_sum=tp_level_sum,
        tp_level_reached_count=tp_level_count,
        breakeven_trade_count=breakeven_totals,
        trailing_tp_trade_count=trailing_tp_totals,
        trade_count=trade_count_totals,
    )


def summarise_exit_reasons(
    result: ExitResult,
    asset_names: Sequence[str],
) -> dict[str, dict[str, dict[str, float]]]:
    """Summarise exit fractions and event counts per reason for metadata."""

    summary: dict[str, dict[str, dict[str, float]]] = {}
    fractions = result.reason_fractions
    counts_totals = getattr(result, "reason_counts", None)
    volume_totals = getattr(result, "reason_totals", None)

    def _label(reason: int) -> tuple[str, int]:
        try:
            enum_val = ExitReason(reason)
        except ValueError:
            return f"UNKNOWN_{reason}", int(reason)
        return enum_val.name, int(enum_val.value)

    if fractions is not None and fractions.size:
        if fractions.ndim == 2:  # single asset squeezed output
            fractions = fractions.reshape(fractions.shape[0], 1, fractions.shape[1])
        asset_count = min(len(asset_names), fractions.shape[1])
        for asset_idx in range(asset_count):
            name = asset_names[asset_idx]
            reason_stats_dynamic: dict[str, dict[str, float]] = {}
            asset_matrix = fractions[:, asset_idx, :]
            for reason in range(1, MAX_REASON_BUCKETS):
                values = asset_matrix[:, reason]
                if np.allclose(values, 0.0):
                    continue
                count = float(np.count_nonzero(values > 0))
                volume = float(values.sum())
                reason_name, code_int = _label(reason)
                reason_stats_dynamic[reason_name] = {
                    "code": code_int,
                    "count": count,
                    "fraction": volume,
                }
            summary[name] = reason_stats_dynamic
    elif counts_totals is not None and volume_totals is not None:
        asset_count = min(len(asset_names), counts_totals.shape[0])
        for asset_idx in range(asset_count):
            name = asset_names[asset_idx]
            reason_stats_aggregated: dict[str, dict[str, float]] = {}
            for reason in range(1, MAX_REASON_BUCKETS):
                count = float(counts_totals[asset_idx, reason])
                volume = float(volume_totals[asset_idx, reason])
                if abs(count) <= 1e-12 and abs(volume) <= 1e-12:
                    continue
                reason_name, code_int = _label(reason)
                reason_stats_aggregated[reason_name] = {
                    "code": code_int,
                    "count": count,
                    "fraction": volume,
                }
            summary[name] = reason_stats_aggregated
    else:
        for name in asset_names:
            summary[name] = {}
    return summary


def compute_exit_metrics(
    result: ExitResult, asset_names: Sequence[str]
) -> dict[str, dict[str, float]]:
    """Compute aggregated telemetry metrics derived from simulator traces."""

    metrics: dict[str, dict[str, float]] = {}
    timeout_sum = getattr(result, "timeout_breach_bars_sum", np.array([], dtype=float))
    timeout_count = getattr(result, "timeout_event_count", np.array([], dtype=float))
    tp_sum = getattr(result, "tp_level_reached_sum", np.array([], dtype=float))
    tp_count = getattr(result, "tp_level_reached_count", np.array([], dtype=float))
    breakeven_trades = getattr(
        result, "breakeven_trade_count", np.array([], dtype=float)
    )
    trailing_tp_trades = getattr(
        result, "trailing_tp_trade_count", np.array([], dtype=float)
    )
    trade_totals = getattr(result, "trade_count", np.array([], dtype=float))

    asset_count = min(
        len(asset_names),
        max(
            timeout_sum.shape[0] if timeout_sum.ndim else 0,
            timeout_count.shape[0] if timeout_count.ndim else 0,
            tp_sum.shape[0] if tp_sum.ndim else 0,
            tp_count.shape[0] if tp_count.ndim else 0,
            breakeven_trades.shape[0] if breakeven_trades.ndim else 0,
            trailing_tp_trades.shape[0] if trailing_tp_trades.ndim else 0,
            trade_totals.shape[0] if trade_totals.ndim else 0,
        ),
    )

    for idx in range(asset_count):
        name = asset_names[idx]
        avg_timeout = 0.0
        timeout_events = 0.0
        if timeout_count.shape[0] > idx and timeout_count[idx] > 0:
            timeout_events = float(timeout_count[idx])
            avg_timeout = float(timeout_sum[idx] / timeout_count[idx])

        avg_tp_level = 0.0
        tp_events = 0.0
        if tp_count.shape[0] > idx and tp_count[idx] > 0:
            tp_events = float(tp_count[idx])
            avg_tp_level = float(tp_sum[idx] / tp_count[idx])

        trades = float(trade_totals[idx]) if trade_totals.shape[0] > idx else 0.0
        be_rate = 0.0
        if trades > 0 and breakeven_trades.shape[0] > idx:
            be_rate = float(breakeven_trades[idx] / trades)
        timeout_rate = 0.0
        if trades > 0 and timeout_count.shape[0] > idx:
            timeout_rate = float(timeout_count[idx] / trades)
        trailing_rate = 0.0
        if trades > 0 and trailing_tp_trades.shape[0] > idx:
            trailing_rate = float(trailing_tp_trades[idx] / trades)

        metrics[name] = {
            "avg_sl_timeout_bars": avg_timeout,
            "sl_timeout_event_count": timeout_events,
            "avg_tp_level_reached": avg_tp_level,
            "tp_trades_evaluated": tp_events,
            "breakeven_touch_rate": be_rate,
            "sl_timeout_usage_rate": timeout_rate,
            "trailing_tp_hit_rate": trailing_rate,
            "trades_evaluated": trades,
        }

    for idx in range(asset_count, len(asset_names)):
        metrics[asset_names[idx]] = {
            "avg_sl_timeout_bars": 0.0,
            "sl_timeout_event_count": 0.0,
            "avg_tp_level_reached": 0.0,
            "tp_trades_evaluated": 0.0,
            "breakeven_touch_rate": 0.0,
            "sl_timeout_usage_rate": 0.0,
            "trailing_tp_hit_rate": 0.0,
            "trades_evaluated": 0.0,
        }

    return metrics


def coerce_exit_params(
    exit_rules: Mapping[str, object], max_hold_bars: int, timeframe: str | None = None
) -> ExitParams:
    """Resolve trade-management gene outputs into ``ExitParams``."""

    tm = exit_rules or {}
    stop_loss_cfg = tm.get("stop_loss", {})
    stop_loss_pct = float(
        stop_loss_cfg.get("params", {}).get("value", stop_loss_cfg.get("value", 0.0))
    )

    tm_cfg = tm.get("trade_management", {})
    num_tp_levels = int(tm_cfg.get("num_tp_levels", 1))
    num_tp_levels = max(1, min(MAX_TP_LEVELS, num_tp_levels))

    tp_defaults = [0.02, 0.03, 0.04, 0.05]
    any_tp_override = False
    for idx in range(MAX_TP_LEVELS):
        key = f"tp_pct_{idx + 1}"
        if key in tm_cfg:
            tp_defaults[idx] = float(tm_cfg[key])
            any_tp_override = True
    if not any_tp_override and tm_cfg.get("tp_pct_1") is not None:
        tp_defaults[0] = float(tm_cfg.get("tp_pct_1", 0.02))

    repaired_tp: list[float] = []
    min_gap_abs = float(getattr(config, "TP_MIN_GAP", 0.005))
    if not math.isfinite(min_gap_abs) or min_gap_abs <= 0.0:
        min_gap_abs = 0.005
    tf_source = (
        timeframe if timeframe is not None else getattr(config, "TIMEFRAME", None)
    )
    cap_resolver = getattr(config, "get_tp_cap_for_timeframe", None)
    if callable(cap_resolver):
        cap_default = float(cap_resolver(tf_source))
    else:
        cap_default = float(getattr(config, "MAX_TP_PCT", 0.8))
    try:
        max_tp_cap = tm_cfg.get("tp_pct_cap")
    except AttributeError:  # pragma: no cover - defensive
        max_tp_cap = None
    cap_for_timeframe = cap_default
    if max_tp_cap is None:
        max_tp_cap = cap_for_timeframe
    else:
        try:
            max_tp_cap = float(max_tp_cap)
        except (TypeError, ValueError):
            max_tp_cap = cap_for_timeframe
        if not math.isfinite(max_tp_cap) or max_tp_cap <= 0:
            max_tp_cap = cap_for_timeframe
        else:
            max_tp_cap = min(max_tp_cap, cap_for_timeframe)
    min_required_cap = min_gap_abs * num_tp_levels
    if min_required_cap > max_tp_cap + 1e-12:
        raise ValueError(
            "Cannot allocate TP ladder: cap=%.3f, levels=%d, min_gap=%.3f"
            % (max_tp_cap, num_tp_levels, min_gap_abs)
        )

    for idx, base in enumerate(tp_defaults):
        if idx == 0:
            val = float(base)
            if val <= 0:
                val = min_gap_abs
            val = max(val, min_gap_abs)
        else:
            prev = repaired_tp[-1]
            if idx < num_tp_levels:
                headroom = max_tp_cap - prev
                if headroom <= min_gap_abs - 1e-12:
                    raise ValueError(
                        "Insufficient TP cap headroom: prev=%.3f, remaining=%.3f, min_gap=%.3f"
                        % (prev, headroom, min_gap_abs)
                    )
                desired = max(float(base), prev + min_gap_abs)
                val = min(desired, max_tp_cap)
                if val <= prev:
                    val = min(max_tp_cap, prev + min_gap_abs)
            else:
                desired = max(float(base), prev)
                val = min(desired, max_tp_cap)
        if val > max_tp_cap:
            val = max_tp_cap
        repaired_tp.append(val)

    tp_vals = tuple(repaired_tp)

    tp_trailing_flag = tm_cfg.get("tp_trailing_enabled")
    tp_trailing_pct = float(tm_cfg.get("tp_trailing_pct", 0.0))
    if tp_trailing_pct < 0.0:
        tp_trailing_pct = 0.0
    if tp_trailing_flag is None:
        tp_trailing_enabled = tp_trailing_pct > 0.0
    else:
        tp_trailing_enabled = bool(tp_trailing_flag)
    if num_tp_levels <= 1:
        tp_trailing_enabled = False
    if tp_trailing_enabled:
        if tp_trailing_pct <= 0.0:
            tp_trailing_pct = 0.001
    else:
        tp_trailing_pct = 0.0

    sl_timeout_flag = tm_cfg.get("sl_timeout_enabled")
    sl_timeout_bars_raw = tm_cfg.get("sl_timeout_bars", 0)
    try:
        sl_timeout_bars = int(sl_timeout_bars_raw)
    except (TypeError, ValueError):
        sl_timeout_bars = 0
    if sl_timeout_flag is None:
        sl_timeout_enabled = sl_timeout_bars > 0
    else:
        sl_timeout_enabled = bool(sl_timeout_flag)
    if sl_timeout_enabled:
        if sl_timeout_bars <= 0:
            sl_timeout_bars = 1
        else:
            sl_timeout_bars = max(1, min(12, sl_timeout_bars))
    else:
        sl_timeout_bars = 0

    be_mode_raw = tm_cfg.get("sl_break_even_mode", "none")
    if isinstance(be_mode_raw, BreakEvenMode):
        be_mode = be_mode_raw
    else:
        numeric_mode = None
        try:
            numeric_mode = int(be_mode_raw)
        except (TypeError, ValueError):
            numeric_mode = None
        if numeric_mode in _BREAK_EVEN_NUMERIC_VALUES:
            be_mode = BreakEvenMode(numeric_mode)
        else:
            be_mode_key = str(be_mode_raw).lower().split(".")[-1]
            be_mode = _BREAK_EVEN_NAME_TO_MODE.get(be_mode_key, BreakEvenMode.NONE)

    sl_trailing_flag = tm_cfg.get("sl_trailing_enabled")
    sl_trailing_pct = float(tm_cfg.get("sl_trailing_pct", 0.0))
    if sl_trailing_pct < 0.0:
        sl_trailing_pct = 0.0
    if sl_trailing_flag is None:
        sl_trailing_enabled = sl_trailing_pct > 0.0
    else:
        sl_trailing_enabled = bool(sl_trailing_flag)
    if sl_trailing_enabled:
        if sl_trailing_pct <= 0.0:
            sl_trailing_pct = 0.001
    else:
        sl_trailing_pct = 0.0

    timeframe_serialised = str(tf_source) if tf_source is not None else None
    return ExitParams(
        stop_loss_pct=stop_loss_pct,
        num_tp_levels=num_tp_levels,
        tp_pcts=tuple(tp_vals),
        tp_trailing_enabled=tp_trailing_enabled,
        tp_trailing_pct=tp_trailing_pct,
        sl_timeout_enabled=sl_timeout_enabled,
        sl_timeout_bars=sl_timeout_bars,
        sl_break_even_mode=be_mode,
        sl_trailing_enabled=sl_trailing_enabled,
        sl_trailing_pct=sl_trailing_pct,
        max_hold_bars=int(max_hold_bars or 0),
        tp_cap=float(max_tp_cap),
        timeframe=timeframe_serialised,
    )
