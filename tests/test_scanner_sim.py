import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import scanner_sim


def _make_series(values):
    return pd.Series(values, index=pd.date_range("2020", periods=len(values), freq="D"))


def test_k_ge_assets_admits_all_entries():
    entries = pd.concat({
        "A": _make_series([1, 0, 1]),
        "B": _make_series([0, 1, 0])
    }, axis=1).astype(bool)
    exits = pd.concat({
        "A": _make_series([0, 1, 0]),
        "B": _make_series([0, 0, 1])
    }, axis=1).astype(bool)
    gated, open_count, diag = scanner_sim.gate_entries(entries, exits, max_concurrent=2)
    assert_frame_equal(gated, entries)
    assert (open_count <= 2).all()
    assert diag["rejected"] == 0


def test_k1_single_entry_per_timestamp():
    entries = pd.concat({
        "A": _make_series([1, 0, 1, 0]),
        "B": _make_series([1, 0, 0, 1])
    }, axis=1).astype(bool)
    exits = pd.concat({
        "A": _make_series([0, 1, 0, 1]),
        "B": _make_series([0, 1, 1, 0])
    }, axis=1).astype(bool)
    gated, open_count, _ = scanner_sim.gate_entries(entries, exits, max_concurrent=1)
    # At any timestamp, at most one entry
    assert (gated.sum(axis=1) <= 1).all()
    assert (open_count <= 1).all()


def test_random_policy_same_seed_deterministic():
    entries = pd.concat({
        "A": _make_series([1, 0]),
        "B": _make_series([1, 0])
    }, axis=1).astype(bool)
    exits = pd.concat({
        "A": _make_series([0, 1]),
        "B": _make_series([0, 1])
    }, axis=1).astype(bool)
    gated1, _, _ = scanner_sim.gate_entries(entries, exits, 1, tie_break_policy="random", seed=42)
    gated2, _, _ = scanner_sim.gate_entries(entries, exits, 1, tie_break_policy="random", seed=42)
    assert_frame_equal(gated1, gated2)


def test_no_asset_reenters_before_exit():
    entries = pd.concat(
        {"A": _make_series([1, 1, 0]), "B": _make_series([0, 0, 0])}, axis=1
    ).astype(bool)
    exits = pd.concat(
        {"A": _make_series([0, 0, 1]), "B": _make_series([0, 0, 0])}, axis=1
    ).astype(bool)
    gated, _, _ = scanner_sim.gate_entries(entries, exits, max_concurrent=1)
    assert not bool(gated.loc[gated.index[1], "A"])


def test_capacity_monotonicity():
    entries = pd.concat(
        {"A": _make_series([1, 0, 0, 0]), "B": _make_series([0, 1, 0, 0])}, axis=1
    ).astype(bool)
    exits = pd.concat(
        {"A": _make_series([0, 0, 1, 0]), "B": _make_series([0, 0, 0, 1])}, axis=1
    ).astype(bool)
    gated_k1, _, _ = scanner_sim.gate_entries(entries, exits, max_concurrent=1)
    gated_k2, _, _ = scanner_sim.gate_entries(entries, exits, max_concurrent=2)
    assert not ((gated_k1 & ~gated_k2).any().any())


def test_allocation_sums_to_one():
    entries = pd.concat(
        {"A": _make_series([1, 0, 0, 0]), "B": _make_series([0, 1, 0, 0])}, axis=1
    ).astype(bool)
    exits = pd.concat(
        {"A": _make_series([0, 0, 1, 0]), "B": _make_series([0, 0, 0, 1])}, axis=1
    ).astype(bool)
    gated, open_count, _ = scanner_sim.gate_entries(entries, exits, max_concurrent=2)
    positions = pd.DataFrame(False, index=gated.index, columns=gated.columns)
    open_set: set[str] = set()
    for ts in gated.index:
        for asset in list(open_set):
            if exits.at[ts, asset]:
                open_set.remove(asset)
        for asset in gated.columns:
            if gated.at[ts, asset]:
                open_set.add(asset)
        for asset in open_set:
            positions.at[ts, asset] = True
    alloc = positions.astype(float).div(open_count.replace(0, pd.NA), axis=0).fillna(0.0)
    sums = alloc.sum(axis=1)
    assert sums[open_count > 0].to_list() == pytest.approx([1.0] * int((open_count > 0).sum()))
    assert (sums[open_count == 0] == 0.0).all()


def test_fifo_policy_deterministic_across_runs():
    entries = pd.concat({"A": _make_series([1]), "B": _make_series([1])}, axis=1).astype(bool)
    exits = pd.concat({"A": _make_series([0]), "B": _make_series([0])}, axis=1).astype(bool)
    gated1, _, _ = scanner_sim.gate_entries(entries, exits, max_concurrent=1, tie_break_policy="fifo")
    gated2, _, _ = scanner_sim.gate_entries(entries, exits, max_concurrent=1, tie_break_policy="fifo")
    assert_frame_equal(gated1, gated2)


def test_random_policy_differs_across_seeds():
    entries = pd.concat({"A": _make_series([1]), "B": _make_series([1])}, axis=1).astype(bool)
    exits = pd.concat({"A": _make_series([0]), "B": _make_series([0])}, axis=1).astype(bool)
    gated1, _, _ = scanner_sim.gate_entries(entries, exits, 1, tie_break_policy="random", seed=0)
    gated2, _, _ = scanner_sim.gate_entries(entries, exits, 1, tie_break_policy="random", seed=1)
    assert not gated1.equals(gated2)


def test_capacity_zero_rejects_all():
    entries = pd.concat(
        {"A": _make_series([1, 0]), "B": _make_series([1, 1])}, axis=1
    ).astype(bool)
    exits = pd.concat(
        {"A": _make_series([0, 0]), "B": _make_series([0, 0])}, axis=1
    ).astype(bool)
    gated, open_count, diag = scanner_sim.gate_entries(entries, exits, max_concurrent=0)
    assert not gated.any().any()
    assert (open_count == 0).all()
    assert diag["collisions"] == 2
    assert diag["rejected"] == 3
    assert diag["accepted"] == 0
    assert diag["total_candidates"] == 3
    assert diag["accepted"] + diag["rejected"] == diag["total_candidates"]
    assert diag["avg_n_open"] == 0
    assert diag["max_n_open"] == 0
    assert diag["per_asset"] == {
        "A": {"candidates": 1, "accepted": 0, "rejected": 1},
        "B": {"candidates": 2, "accepted": 0, "rejected": 2},
    }


def test_zero_capacity_counts_single_timestamp():
    entries = pd.concat(
        {
            "A": _make_series([1]),
            "B": _make_series([1]),
            "C": _make_series([1]),
        },
        axis=1,
    ).astype(bool)
    exits = pd.concat(
        {
            "A": _make_series([0]),
            "B": _make_series([0]),
            "C": _make_series([0]),
        },
        axis=1,
    ).astype(bool)
    _, _, diag = scanner_sim.gate_entries(entries, exits, max_concurrent=0)
    assert diag["collisions"] == 1
    assert diag["rejected"] == 3
    assert diag["accepted"] == 0


def test_per_asset_tallies():
    entries = pd.concat(
        {"A": _make_series([1, 0]), "B": _make_series([1, 0])}, axis=1
    ).astype(bool)
    exits = pd.concat(
        {"A": _make_series([0, 1]), "B": _make_series([0, 1])}, axis=1
    ).astype(bool)
    gated, _, diag = scanner_sim.gate_entries(entries, exits, max_concurrent=1)
    expected = pd.concat(
        {"A": _make_series([1, 0]), "B": _make_series([0, 0])}, axis=1
    ).astype(bool)
    assert_frame_equal(gated, expected)
    assert diag["collisions"] == 1
    assert diag["accepted"] == 1
    assert diag["rejected"] == 1
    assert diag["total_candidates"] == 2
    assert diag["per_asset"]["A"] == {
        "candidates": 1,
        "accepted": 1,
        "rejected": 0,
    }
    assert diag["per_asset"]["B"] == {
        "candidates": 1,
        "accepted": 0,
        "rejected": 1,
    }


def test_collision_histogram():
    entries = pd.concat(
        {"A": _make_series([1, 0]), "B": _make_series([1, 0])}, axis=1
    ).astype(bool)
    exits = pd.concat(
        {"A": _make_series([0, 1]), "B": _make_series([0, 1])}, axis=1
    ).astype(bool)
    _, _, diag = scanner_sim.gate_entries(
        entries, exits, max_concurrent=1, collect_collision_histogram=True
    )
    assert diag["collisions_by_asset"] == {"A": 1, "B": 1}


def test_plot_helper_runs():
    entries = pd.concat(
        {"A": _make_series([1]), "B": _make_series([0])}, axis=1
    ).astype(bool)
    exits = pd.concat(
        {"A": _make_series([0]), "B": _make_series([0])}, axis=1
    ).astype(bool)
    _, _, diag = scanner_sim.gate_entries(entries, exits, max_concurrent=1)
    scanner_sim.plot_admitted_trade_skew(diag)
