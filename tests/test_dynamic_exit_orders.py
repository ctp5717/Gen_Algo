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
