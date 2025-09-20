import time

import numpy as np
import pytest

from exits_nb import (
    BreakEvenMode,
    ExitParams,
    ExitReason,
    coerce_exit_params,
    compute_exit_metrics,
    generate_dynamic_exit_signals_nb,
    summarise_exit_reasons,
)


def _make_params(**overrides):
    base = {
        "stop_loss_pct": 0.05,
        "num_tp_levels": 2,
        "tp_pcts": (0.01, 0.02, 0.03, 0.04),
        "tp_trailing_enabled": False,
        "tp_trailing_pct": 0.02,
        "sl_timeout_enabled": False,
        "sl_timeout_bars": 0,
        "sl_break_even_mode": BreakEvenMode.NONE,
        "sl_trailing_enabled": False,
        "sl_trailing_pct": 0.0,
        "max_hold_bars": 0,
    }
    base.update(overrides)
    return ExitParams(**base)


def _price_map(close, high=None, low=None):
    close = np.asarray(close, dtype=float)
    high = np.asarray(high if high is not None else close, dtype=float)
    low = np.asarray(low if low is not None else close, dtype=float)
    return {"close": close, "high": high, "low": low}


def test_tp_ladder_multiple_same_bar():
    entries = np.array([True, False, False])
    price = _price_map([100, 103, 103], high=[100, 103, 103], low=[100, 100, 103])
    params = _make_params()
    result = generate_dynamic_exit_signals_nb(entries, price, params)
    assert result.exit_size[1] == pytest.approx(1.0)
    assert result.exit_reason[1] == ExitReason.TP2
    fractions = result.reason_fractions[1]
    assert fractions[ExitReason.TP1] == pytest.approx(0.5)
    assert fractions[ExitReason.TP2] == pytest.approx(0.5)


def test_break_even_stop_moves_to_entry():
    entries = np.array([True, False, False, False])
    price = _price_map(
        [100, 101.5, 100, 100],
        high=[100, 101.5, 100.5, 100],
        low=[100, 100.2, 100, 100],
    )
    params = _make_params(sl_break_even_mode=BreakEvenMode.BREAKEVEN)
    result = generate_dynamic_exit_signals_nb(entries, price, params)
    assert result.current_stop[1] == pytest.approx(100.0)
    assert result.exit_size[2] == pytest.approx(0.5)
    assert result.exit_reason[2] == ExitReason.SL
    metrics = compute_exit_metrics(result, ["asset"])["asset"]
    assert metrics["breakeven_touch_rate"] == pytest.approx(1.0)


def test_follow_tp_ratchets_stop():
    entries = np.array([True, False, False, False])
    price = _price_map(
        [100, 101.2, 102.4, 101.9],
        high=[100, 101.2, 102.4, 102.0],
        low=[100, 101.1, 102.1, 101.8],
    )
    params = _make_params(
        num_tp_levels=3,
        tp_pcts=(0.01, 0.02, 0.03, 0.04),
        sl_break_even_mode=BreakEvenMode.FOLLOW_TP,
    )
    result = generate_dynamic_exit_signals_nb(entries, price, params)
    # After TP2 the stop should move to the TP2 price (~102.0)
    assert result.current_stop[2] == pytest.approx(102.0, rel=1e-3)
    assert result.exit_size[3] == pytest.approx(1 / 3)
    assert result.exit_reason[3] == ExitReason.SL
    metrics = compute_exit_metrics(result, ["asset"])["asset"]
    assert metrics["breakeven_touch_rate"] == pytest.approx(1.0)
    assert metrics["trailing_tp_hit_rate"] == pytest.approx(0.0)


def test_trailing_stop_never_looses_ground():
    entries = np.array([True, False, False])
    price = _price_map(
        [100, 110, 115],
        high=[100, 110, 115],
        low=[100, 105, 110.5],
    )
    params = _make_params(
        sl_trailing_enabled=True,
        sl_trailing_pct=0.05,
        num_tp_levels=1,
        tp_pcts=(0.5, 0.6, 0.7, 0.8),
    )
    result = generate_dynamic_exit_signals_nb(entries, price, params)
    assert result.current_stop[1] == pytest.approx(104.5)
    assert result.current_stop[2] == pytest.approx(109.25)


def test_trailing_take_profit_activates_after_last_tp():
    entries = np.array([True, False, False, False])
    price = _price_map(
        [100, 108, 110, 108],
        high=[100, 108, 112, 112],
        low=[100, 106, 108, 107],
    )
    params = _make_params(
        num_tp_levels=2,
        tp_pcts=(0.02, 0.04, 0.04, 0.04),
        tp_trailing_enabled=True,
        tp_trailing_pct=0.02,
    )
    result = generate_dynamic_exit_signals_nb(entries, price, params)
    # First bar exits 50% at TP1 and activates trailing protection for the remainder
    assert result.exit_size[1] == pytest.approx(0.5)
    fractions = result.reason_fractions[1]
    assert fractions[ExitReason.TP1] == pytest.approx(0.5)
    # Trailing trigger ratchets with new highs after TP activation
    assert result.trailing_tp_trigger[1] == pytest.approx(108 * (1 - 0.02))
    # Remaining size exits when price pulls back to the trailing trigger on the next bar
    assert np.isnan(result.trailing_tp_trigger[2])
    assert result.exit_reason[2] == ExitReason.TTP
    assert result.exit_size[2] == pytest.approx(0.5)
    metrics = compute_exit_metrics(result, ["asset"])["asset"]
    assert metrics["trailing_tp_hit_rate"] == pytest.approx(1.0)
    assert metrics["breakeven_touch_rate"] == pytest.approx(0.0)


def test_stop_timeout_exits_after_window():
    entries = np.array([True, False, False, False, False])
    price = _price_map(
        [100, 95, 95, 95, 95],
        high=[100, 95, 95, 95, 95],
        low=[100, 94, 94, 94, 94],
    )
    params = _make_params(sl_timeout_enabled=True, sl_timeout_bars=2)
    result = generate_dynamic_exit_signals_nb(entries, price, params)
    assert result.exit_reason[3] == ExitReason.SL_TIMEOUT
    assert result.exit_size[3] == pytest.approx(1.0)
    assert result.exit_size[1] == pytest.approx(0.0)
    metrics = compute_exit_metrics(result, ["asset"])["asset"]
    assert metrics["sl_timeout_usage_rate"] == pytest.approx(1.0)


def test_stop_timeout_resets_when_price_recovers():
    entries = np.array([True, False, False, False, False, False])
    price = _price_map(
        [100, 95, 99, 95, 95, 95],
        high=[100, 96, 99, 96, 96, 96],
        low=[100, 94, 96, 94, 94, 94],
    )
    params = _make_params(sl_timeout_enabled=True, sl_timeout_bars=2)
    result = generate_dynamic_exit_signals_nb(entries, price, params)
    # The first breach starts a timer that resets on recovery (bar 2)
    assert result.exit_size[1] == pytest.approx(0.0)
    assert result.exit_size[2] == pytest.approx(0.0)
    # Second breach exits only after the full timeout window elapses
    assert result.exit_size[4] == pytest.approx(0.0)
    assert result.exit_reason[5] == ExitReason.SL_TIMEOUT
    assert result.exit_size[5] == pytest.approx(1.0)
    metrics = compute_exit_metrics(result, ["asset"])["asset"]
    assert metrics["sl_timeout_usage_rate"] == pytest.approx(1.0)


def test_timeout_no_lookahead_behaviour():
    entries = np.array([True, False, False])
    price = _price_map(
        [100, 95, 101],
        high=[100, 95, 100.2],
        low=[100, 94, 100],
    )
    params = _make_params(sl_timeout_enabled=True, sl_timeout_bars=2)
    result = generate_dynamic_exit_signals_nb(entries, price, params)
    assert np.all(result.exits == np.array([False, False, False]))
    metrics = compute_exit_metrics(result, ["asset"])["asset"]
    assert metrics["sl_timeout_usage_rate"] == pytest.approx(0.0)


def test_same_bar_tp_precedes_stop():
    entries = np.array([True, False])
    price = _price_map(
        [100, 100],
        high=[100, 101.2],
        low=[100, 94],
    )
    params = _make_params(stop_loss_pct=0.05)
    result = generate_dynamic_exit_signals_nb(entries, price, params)
    fractions = result.reason_fractions[1]
    assert fractions[ExitReason.TP1] == pytest.approx(0.5)
    assert fractions[ExitReason.SL] == pytest.approx(0.5)
    assert result.exit_size[1] == pytest.approx(1.0)
    assert np.count_nonzero(result.exits) == 1
    assert result.exit_reason[1] == ExitReason.SL
    summary = summarise_exit_reasons(result, ["asset"])["asset"]
    assert summary[ExitReason.TP1.name]["count"] == pytest.approx(1.0)
    assert summary[ExitReason.SL.name]["count"] == pytest.approx(1.0)


def test_max_hold_timeout_exits_on_configured_bar():
    entries = np.array([True, False, False, False, False])
    price = _price_map([100, 100, 100, 100, 100])
    params = _make_params(max_hold_bars=3)
    result = generate_dynamic_exit_signals_nb(entries, price, params)
    assert result.exit_size[1] == pytest.approx(0.0)
    assert result.exit_size[2] == pytest.approx(0.0)
    assert result.exit_reason[3] == ExitReason.SL
    assert result.exit_size[3] == pytest.approx(1.0)


def test_coerce_exit_params_enforces_hierarchy():
    base_rules = {
        "stop_loss": {"params": {"value": 0.05}},
        "trade_management": {
            "num_tp_levels": 1,
            "tp_pct_1": 0.02,
            "tp_trailing_enabled": 1,
            "tp_trailing_pct": 0.02,
            "sl_break_even_mode": "follow_tp",
            "sl_timeout_enabled": 0,
            "sl_timeout_bars": 6,
            "sl_trailing_enabled": 0,
            "sl_trailing_pct": 0.1,
        },
    }
    params_single = coerce_exit_params(base_rules, max_hold_bars=4)
    assert params_single.tp_trailing_enabled is False
    assert params_single.tp_trailing_pct == pytest.approx(0.0)
    assert params_single.sl_timeout_bars == 0
    assert params_single.sl_trailing_pct == pytest.approx(0.0)
    assert params_single.sl_break_even_mode == BreakEvenMode.FOLLOW_TP
    assert params_single.max_hold_bars == 4

    tm_multi = dict(base_rules["trade_management"])
    tm_multi.update(
        {
            "num_tp_levels": 2,
            "tp_trailing_enabled": 1,
            "sl_timeout_enabled": 1,
            "sl_timeout_bars": 0,
        }
    )
    params_multi = coerce_exit_params(
        {"stop_loss": base_rules["stop_loss"], "trade_management": tm_multi},
        max_hold_bars=0,
    )
    assert params_multi.tp_trailing_enabled is True
    assert params_multi.tp_trailing_pct == pytest.approx(0.02)
    assert params_multi.sl_timeout_bars == 1
    assert params_multi.tp_pcts[0] < params_multi.tp_pcts[1]
    assert params_multi.tp_pcts[1] <= params_multi.tp_pcts[2]
    assert params_multi.tp_pcts[2] <= params_multi.tp_pcts[3]


def test_deterministic_output():
    entries = np.array([True, False, False])
    price = _price_map([100, 103, 103], high=[100, 103, 103], low=[100, 100, 103])
    params = _make_params()
    first = generate_dynamic_exit_signals_nb(entries, price, params)
    second = generate_dynamic_exit_signals_nb(entries, price, params)
    assert np.array_equal(first.exits, second.exits)
    assert np.array_equal(first.exit_size, second.exit_size)
    assert np.array_equal(first.exit_reason, second.exit_reason)


def test_dynamic_exit_simulator_performance_smoke():
    rng = np.random.default_rng(42)
    rows, assets = 2000, 3
    base = np.linspace(100, 130, rows, dtype=float).reshape(-1, 1)
    noise = rng.normal(scale=0.5, size=(rows, assets))
    close = np.clip(base + noise, 10.0, None)
    high = close + rng.uniform(0.0, 1.0, size=(rows, assets))
    low = close - rng.uniform(0.0, 1.0, size=(rows, assets))
    entries = rng.random((rows, assets)) < 0.05
    entries[0, :] = True
    params = _make_params(
        tp_trailing_enabled=True,
        sl_timeout_enabled=True,
        sl_timeout_bars=3,
        sl_trailing_enabled=True,
    )
    price_map = {"close": close, "high": high, "low": low}
    generate_dynamic_exit_signals_nb(entries, price_map, params, collect_traces=False)
    iterations = 5
    start = time.perf_counter()
    for _ in range(iterations):
        generate_dynamic_exit_signals_nb(
            entries, price_map, params, collect_traces=False
        )
    avg_runtime = (time.perf_counter() - start) / iterations
    assert avg_runtime < 0.1


def test_summary_available_without_traces():
    entries = np.array([True, False, False, True, False, False, False])
    price = _price_map(
        [100, 105, 105, 100, 94, 94, 93],
        high=[100, 105, 105, 100, 94, 94, 93],
        low=[100, 105, 105, 100, 94, 94, 93],
    )
    params = _make_params(sl_timeout_enabled=True, sl_timeout_bars=2)
    result = generate_dynamic_exit_signals_nb(
        entries, price, params, collect_traces=False
    )
    assert result.current_stop is None
    assert result.reason_fractions is None

    summary = summarise_exit_reasons(result, ["asset"])
    asset_summary = summary["asset"]
    tp1 = asset_summary[ExitReason.TP1.name]
    assert tp1["code"] == ExitReason.TP1
    assert tp1["fraction"] == pytest.approx(0.5)
    tp2 = asset_summary[ExitReason.TP2.name]
    assert tp2["code"] == ExitReason.TP2
    assert tp2["fraction"] == pytest.approx(0.5)
    timeout_stats = asset_summary[ExitReason.SL_TIMEOUT.name]
    assert timeout_stats["code"] == ExitReason.SL_TIMEOUT
    assert timeout_stats["fraction"] == pytest.approx(1.0)

    metrics = compute_exit_metrics(result, ["asset"])["asset"]
    assert metrics["avg_tp_level_reached"] == pytest.approx(1.0)
    assert metrics["tp_trades_evaluated"] == pytest.approx(2.0)
    assert metrics["avg_sl_timeout_bars"] == pytest.approx(2.0)
    assert metrics["sl_timeout_event_count"] == pytest.approx(1.0)
    assert metrics["sl_timeout_usage_rate"] == pytest.approx(0.5)
    assert metrics["breakeven_touch_rate"] == pytest.approx(0.0)
    assert metrics["trailing_tp_hit_rate"] == pytest.approx(0.0)
    assert metrics["trades_evaluated"] == pytest.approx(2.0)
