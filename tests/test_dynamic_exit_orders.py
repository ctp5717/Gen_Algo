import copy
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

sys.modules.setdefault("pandas_ta", types.ModuleType("pandas_ta"))
try:  # prefer real vectorbt if available
    import vectorbt  # noqa: F401
except Exception:  # pragma: no cover
    sys.modules.setdefault("vectorbt", types.ModuleType("vectorbt"))

import config  # noqa: E402
import fitness  # noqa: E402
import params_resolver  # noqa: E402
import strategy_rules  # noqa: E402
from exits_nb import coerce_exit_params, generate_dynamic_exit_signals_nb  # noqa: E402


def test_dynamic_exit_partial_accounting(monkeypatch):
    monkeypatch.setattr(
        config, "DYNAMIC_EXIT_SIZE_MODE", "fraction_base", raising=False
    )
    base_size = 10.0
    index = pd.RangeIndex(6)
    entries = pd.Series([True] + [False] * 5, index=index, dtype=bool)

    # Scenario with same-bar exit flag, partial legs, and an overshoot capped at open qty.
    exits_raw = pd.Series(
        [True, True, True, False, False, False], index=index, dtype=bool
    )
    exit_sizes = pd.Series(
        [0.05, 0.4, 0.6000000001, 0.0, 0.0, 0.0], index=index, dtype=float
    )
    entries_active, exits_clean, size_series = fitness.build_dynamic_exit_orders(
        entries=entries,
        exits_series=exits_raw,
        exit_size_series=exit_sizes,
        base_entry_size=base_size,
        mode=config.DYNAMIC_EXIT_SIZE_MODE,
        asset_label="unit-overshoot",
    )

    assert entries_active.sum() == 1
    assert not exits_clean.iloc[0]
    assert exits_clean.sum() == 2
    assert size_series[entries_active].iloc[0] == pytest.approx(base_size)
    exit_totals = size_series[exits_clean]
    assert (exit_totals >= 0).all()
    assert exit_totals.sum() == pytest.approx(base_size)

    # Scenario where exit legs sum to < 1 so a forced terminal close is inserted.
    exits_residual = pd.Series(
        [False, True, False, False, False, False], index=index, dtype=bool
    )
    residual_sizes = pd.Series([0.0, 0.3, 0.0, 0.0, 0.0, 0.0], index=index, dtype=float)
    entries_active2, exits_clean2, size_series2 = fitness.build_dynamic_exit_orders(
        entries=entries,
        exits_series=exits_residual,
        exit_size_series=residual_sizes,
        base_entry_size=base_size,
        mode=config.DYNAMIC_EXIT_SIZE_MODE,
        asset_label="unit-residual",
    )

    assert entries_active2.sum() == 1
    assert exits_clean2.sum() == 2
    assert exits_clean2.iloc[-1]
    assert size_series2.iloc[-1] == pytest.approx(base_size * 0.7)
    assert size_series2[exits_clean2].sum() == pytest.approx(base_size)


def _simulate_positions_and_pnl(
    close: pd.Series,
    entries: pd.Series,
    exits: pd.Series,
    sizes: pd.Series,
    accumulate: bool,
) -> tuple[list[float], float]:
    position = 0.0
    avg_entry_price = 0.0
    pnl = 0.0
    positions: list[float] = []
    for idx, price in enumerate(close.tolist()):
        if bool(entries.iloc[idx]):
            qty = float(sizes.iloc[idx])
            if accumulate and position > 0:
                total_qty = position + qty
                avg_entry_price = (avg_entry_price * position + price * qty) / max(
                    total_qty, 1e-12
                )
                position = total_qty
            else:
                position = qty
                avg_entry_price = price
        if bool(exits.iloc[idx]):
            qty = float(sizes.iloc[idx])
            pnl += qty * (price - avg_entry_price)
            position = max(0.0, position - qty)
        positions.append(position)
    return positions, pnl


@pytest.mark.parametrize("accumulate_flag", [False, True])
def test_partial_tp_position_path(monkeypatch, accumulate_flag):
    configured_accumulate = getattr(config, "DYNAMIC_EXIT_ACCUMULATE", False)
    if accumulate_flag != configured_accumulate:
        pytest.skip(
            f"accumulate={accumulate_flag} is inactive (configured={configured_accumulate})"
        )

    monkeypatch.setattr(config, "TIMEFRAME", "4h", raising=False)

    index = pd.RangeIndex(3)
    entries = pd.Series([True, False, False], index=index, dtype=bool)
    close = pd.Series([100.0, 105.0, 110.0], index=index, dtype=float)
    price_map = {
        "close": close.to_numpy(dtype=float),
        "high": close.to_numpy(dtype=float),
        "low": close.to_numpy(dtype=float),
    }
    exit_rules = {
        "stop_loss": {"params": {"value": 0.05}},
        "trade_management": {
            "num_tp_levels": 2,
            "tp_pct_1": 0.05,
            "tp_pct_2": 0.10,
        },
    }

    params = coerce_exit_params(exit_rules, max_hold_bars=10, timeframe="4h")
    exit_result = generate_dynamic_exit_signals_nb(
        entries.to_numpy(dtype=bool), price_map, params, collect_traces=False
    )
    exits_series = pd.Series(exit_result.exits, index=index)
    exit_size_series = pd.Series(exit_result.exit_size, index=index, dtype=float)

    base_entry_size = 1.0
    mode = getattr(config, "DYNAMIC_EXIT_SIZE_MODE", "fraction_base")
    entries_active, exits_clean, size_series = fitness.build_dynamic_exit_orders(
        entries=entries,
        exits_series=exits_series,
        exit_size_series=exit_size_series,
        base_entry_size=base_entry_size,
        mode=mode,
        asset_label="partial-path",
    )

    assert entries_active.tolist() == [True, False, False]
    assert size_series[entries_active].iloc[0] == pytest.approx(base_entry_size)
    assert size_series[exits_clean].tolist() == pytest.approx(
        [base_entry_size * 0.5, base_entry_size * 0.5]
    )

    positions, pnl = _simulate_positions_and_pnl(
        close, entries_active, exits_clean, size_series, accumulate_flag
    )
    assert positions == pytest.approx([1.0, 0.5, 0.0])

    entry_idx = entries_active[entries_active].index[0]
    entry_price = float(close.loc[entry_idx])
    expected_pnl = 0.0
    for idx in exits_clean[exits_clean].index:
        exit_price = float(close.loc[idx])
        qty = float(size_series.loc[idx])
        expected_pnl += qty * (exit_price - entry_price)
    assert pnl == pytest.approx(expected_pnl)


@pytest.mark.parametrize("accumulate_flag", [False, True])
def test_dynamic_exit_round_trip_consistency(monkeypatch, accumulate_flag):
    monkeypatch.setattr(
        config, "DYNAMIC_EXIT_ACCUMULATE", accumulate_flag, raising=False
    )
    monkeypatch.setattr(
        config, "DYNAMIC_EXIT_SIZE_MODE", "fraction_base", raising=False
    )
    monkeypatch.setattr(config, "TIMEFRAME", "4h", raising=False)
    config.initialize_config(force=True)

    base_rules = copy.deepcopy(strategy_rules.STRATEGY_RULES)
    gene_map = {
        0: {"path": ["exit_rules", "trade_management", "num_tp_levels"]},
        1: {"path": ["exit_rules", "trade_management", "tp_pct_1"]},
        2: {"path": ["exit_rules", "trade_management", "tp_pct_2"]},
        3: {"path": ["exit_rules", "trade_management", "tp_pct_3"]},
        4: {"path": ["exit_rules", "trade_management", "tp_pct_4"]},
    }
    solution = [2, 0.05, 0.05, 0.05, 0.05]
    resolved_rules = params_resolver.resolve_effective_rules(
        base_rules, gene_map, solution
    )
    exit_rules = resolved_rules.get("exit_rules", {})
    exit_params = coerce_exit_params(
        exit_rules,
        max_hold_bars=getattr(config, "MAX_HOLD_PERIOD", 0) or 0,
        timeframe=config.TIMEFRAME,
    )

    index = pd.RangeIndex(3)
    entries = pd.Series([True, False, False], index=index, dtype=bool)
    close = pd.Series([100.0, 105.0, 110.0], index=index, dtype=float)
    price_map = {
        "close": close.to_numpy(dtype=float),
        "high": close.to_numpy(dtype=float),
        "low": close.to_numpy(dtype=float),
    }
    exit_result = generate_dynamic_exit_signals_nb(
        entries.to_numpy(dtype=bool),
        price_map,
        exit_params,
        collect_traces=False,
    )
    exits_series = pd.Series(exit_result.exits, index=index)
    exit_size_series = pd.Series(exit_result.exit_size, index=index, dtype=float)

    entries_active, exits_clean, size_series = fitness.build_dynamic_exit_orders(
        entries=entries,
        exits_series=exits_series,
        exit_size_series=exit_size_series,
        base_entry_size=1.0,
        mode=config.DYNAMIC_EXIT_SIZE_MODE,
        asset_label="round-trip",
    )

    expected_positions, expected_pnl = _simulate_positions_and_pnl(
        close, entries_active, exits_clean, size_series, accumulate_flag
    )

    class _TrackingTrades:
        def __init__(self, count: int) -> None:
            self._count = int(count)

        def count(self) -> int:
            return self._count

    class _TrackingPortfolio:
        def __init__(self, positions, pnl, trade_count):
            self.positions = positions
            self.final_pnl = pnl
            self._stats = {"Total Profit": pnl}
            self.trades = _TrackingTrades(trade_count)

        def stats(self):
            return self._stats

    def fake_from_signals(cls, close, entries, exits, size, accumulate, **kwargs):
        assert bool(accumulate) is accumulate_flag
        close_series = (
            close if isinstance(close, pd.Series) else pd.Series(close, index=index)
        )
        entries_series = (
            entries
            if isinstance(entries, pd.Series)
            else pd.Series(entries, index=index, dtype=bool)
        )
        exits_series_local = (
            exits
            if isinstance(exits, pd.Series)
            else pd.Series(exits, index=index, dtype=bool)
        )
        size_series_local = (
            size
            if isinstance(size, pd.Series)
            else pd.Series(size, index=index, dtype=float)
        )
        positions, pnl = _simulate_positions_and_pnl(
            close_series,
            entries_series,
            exits_series_local,
            size_series_local,
            accumulate_flag,
        )
        trade_count = int(exits_series_local.astype(bool).sum())
        return _TrackingPortfolio(positions, pnl, trade_count)

    monkeypatch.setattr(
        fitness.vbt.Portfolio,
        "from_signals",
        classmethod(fake_from_signals),
        raising=False,
    )

    portfolio = fitness.vbt.Portfolio.from_signals(
        close=close,
        entries=entries_active,
        exits=exits_clean,
        size=size_series,
        accumulate=accumulate_flag,
        fees=config.FEES,
        freq=config.to_pandas_freq(config.TIMEFRAME),
    )

    assert portfolio.positions == pytest.approx(expected_positions)
    assert portfolio.final_pnl == pytest.approx(expected_pnl)


def test_dynamic_exit_fraction_base_semantics(monkeypatch):
    monkeypatch.setattr(
        config, "DYNAMIC_EXIT_SIZE_MODE", "fraction_base", raising=False
    )
    base_size = 5.0
    index = pd.RangeIndex(3)
    entries = pd.Series([True, False, False], index=index, dtype=bool)
    exits_raw = pd.Series([False, True, True], index=index, dtype=bool)
    exit_sizes = pd.Series([0.0, 0.5, 0.5], index=index, dtype=float)

    entries_active, exits_clean, size_series = fitness.build_dynamic_exit_orders(
        entries=entries,
        exits_series=exits_raw,
        exit_size_series=exit_sizes,
        base_entry_size=base_size,
        mode=config.DYNAMIC_EXIT_SIZE_MODE,
        asset_label="unit-fraction",
    )

    assert entries_active.sum() == 1
    assert exits_clean.sum() == 2
    sold_sizes = size_series[exits_clean].to_list()
    assert sold_sizes == [
        pytest.approx(base_size * 0.5),
        pytest.approx(base_size * 0.5),
    ]
    assert sum(sold_sizes) == pytest.approx(base_size)


def test_dynamic_exit_zero_fraction_drops_exit(monkeypatch):
    monkeypatch.setattr(
        config, "DYNAMIC_EXIT_SIZE_MODE", "fraction_base", raising=False
    )
    base_size = 3.0
    index = pd.RangeIndex(4)
    entries = pd.Series([True, False, False, False], index=index, dtype=bool)
    exits_raw = pd.Series([False, True, False, False], index=index, dtype=bool)
    exit_sizes = pd.Series([0.0, 0.0, 0.0, 0.0], index=index, dtype=float)

    entries_active, exits_clean, size_series = fitness.build_dynamic_exit_orders(
        entries=entries,
        exits_series=exits_raw,
        exit_size_series=exit_sizes,
        base_entry_size=base_size,
        mode=config.DYNAMIC_EXIT_SIZE_MODE,
        asset_label="unit-zero-exit",
    )

    assert entries_active.sum() == 1
    assert exits_clean.sum() == 1
    assert bool(exits_clean.iloc[1]) is False
    assert size_series.iloc[1] == pytest.approx(0.0)
    assert exits_clean.iloc[-1]
    assert size_series.iloc[-1] == pytest.approx(base_size)


def test_dynamic_exit_fraction_current_semantics(monkeypatch):
    monkeypatch.setattr(
        config, "DYNAMIC_EXIT_SIZE_MODE", "fraction_current", raising=False
    )
    base_size = 8.0
    index = pd.RangeIndex(4)
    entries = pd.Series([True, False, False, False], index=index, dtype=bool)
    exits_raw = pd.Series([False, True, True, False], index=index, dtype=bool)
    exit_sizes = pd.Series([0.0, 0.5, 0.5, 0.0], index=index, dtype=float)

    entries_active, exits_clean, size_series = fitness.build_dynamic_exit_orders(
        entries=entries,
        exits_series=exits_raw,
        exit_size_series=exit_sizes,
        base_entry_size=base_size,
        mode=config.DYNAMIC_EXIT_SIZE_MODE,
        asset_label="unit-fraction-current",
    )

    assert entries_active.sum() == 1
    assert exits_clean.sum() == 3
    sold_sizes = size_series[exits_clean].to_list()
    assert sold_sizes[0] == pytest.approx(base_size * 0.5)
    assert sold_sizes[1] == pytest.approx(base_size * 0.25)
    assert sold_sizes[2] == pytest.approx(base_size * 0.25)
    assert sum(sold_sizes) == pytest.approx(base_size)


def test_dynamic_exit_absolute_semantics(monkeypatch):
    monkeypatch.setattr(config, "DYNAMIC_EXIT_SIZE_MODE", "absolute", raising=False)
    base_size = 6.0
    index = pd.RangeIndex(4)
    entries = pd.Series([True, False, False, False], index=index, dtype=bool)
    exits_raw = pd.Series([False, True, True, False], index=index, dtype=bool)
    exit_sizes = pd.Series([0.0, 2.0, 1.5, 0.0], index=index, dtype=float)

    entries_active, exits_clean, size_series = fitness.build_dynamic_exit_orders(
        entries=entries,
        exits_series=exits_raw,
        exit_size_series=exit_sizes,
        base_entry_size=base_size,
        mode=config.DYNAMIC_EXIT_SIZE_MODE,
        asset_label="unit-absolute",
    )

    assert entries_active.sum() == 1
    assert exits_clean.sum() == 3
    sold_sizes = size_series[exits_clean].to_list()
    assert sold_sizes[0] == pytest.approx(2.0)
    assert sold_sizes[1] == pytest.approx(1.5)
    assert sold_sizes[2] == pytest.approx(base_size - 3.5)
    assert sum(sold_sizes) == pytest.approx(base_size)
