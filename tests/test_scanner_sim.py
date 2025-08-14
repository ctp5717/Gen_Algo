import pandas as pd
from pandas.testing import assert_frame_equal

import scanner_sim


def _make_series(values):
    return pd.Series(values, index=pd.date_range("2020", periods=len(values), freq="D"))


def test_no_gating_when_capacity_high():
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


def test_k1_limits_entries():
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


def test_random_policy_deterministic_with_seed():
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


def test_no_reentry_while_open():
    entries = pd.concat(
        {"A": _make_series([1, 1, 0]), "B": _make_series([0, 0, 0])}, axis=1
    ).astype(bool)
    exits = pd.concat(
        {"A": _make_series([0, 0, 1]), "B": _make_series([0, 0, 0])}, axis=1
    ).astype(bool)
    gated, _, _ = scanner_sim.gate_entries(entries, exits, max_concurrent=1)
    assert not bool(gated.loc[gated.index[1], "A"])
