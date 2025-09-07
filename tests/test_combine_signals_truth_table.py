import sys
import types
from pathlib import Path

import pandas as pd

# Ensure repository root on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Stub optional heavy deps
sys.modules.setdefault("pandas_ta", types.ModuleType("pandas_ta"))
sys.modules.setdefault("vectorbt", types.ModuleType("vectorbt"))

import strategy_engine  # noqa: E402


def test_truth_table_odd_signals():
    idx = pd.date_range("2020", periods=4)
    signals = [
        pd.Series([True, True, False, False], index=idx),
        pd.Series([True, False, True, False], index=idx),
        pd.Series([True, False, False, True], index=idx),
    ]

    expected_and = pd.Series([True, False, False, False], index=idx)
    pd.testing.assert_series_equal(
        strategy_engine._combine_signals(signals, "AND"),
        expected_and,
    )

    expected_or = pd.Series([True, True, True, True], index=idx)
    pd.testing.assert_series_equal(
        strategy_engine._combine_signals(signals, "OR"),
        expected_or,
    )

    expected_majority = pd.Series([True, False, False, False], index=idx)
    pd.testing.assert_series_equal(
        strategy_engine._combine_signals(signals, "VOTE"),
        expected_majority,
    )

    pd.testing.assert_series_equal(
        strategy_engine._combine_signals(signals, "VOTE", vote_threshold=2),
        expected_majority,
    )

    pd.testing.assert_series_equal(
        strategy_engine._combine_signals(signals, "VOTE", vote_threshold=1),
        expected_or,
    )

    pd.testing.assert_series_equal(
        strategy_engine._combine_signals(signals, "VOTE", vote_threshold=len(signals)),
        expected_and,
    )


def test_truth_table_even_signals():
    idx = pd.date_range("2020", periods=4)
    signals = [
        pd.Series([True, True, False, False], index=idx),
        pd.Series([True, False, True, False], index=idx),
        pd.Series([False, True, True, False], index=idx),
        pd.Series([True, False, False, True], index=idx),
    ]

    expected_and = pd.Series([False, False, False, False], index=idx)
    pd.testing.assert_series_equal(
        strategy_engine._combine_signals(signals, "AND"),
        expected_and,
    )

    expected_or = pd.Series([True, True, True, True], index=idx)
    pd.testing.assert_series_equal(
        strategy_engine._combine_signals(signals, "OR"),
        expected_or,
    )

    expected_majority = pd.Series([True, True, True, False], index=idx)
    pd.testing.assert_series_equal(
        strategy_engine._combine_signals(signals, "VOTE"),
        expected_majority,
    )

    expected_three = pd.Series([True, False, False, False], index=idx)
    pd.testing.assert_series_equal(
        strategy_engine._combine_signals(signals, "VOTE", vote_threshold=3),
        expected_three,
    )

    pd.testing.assert_series_equal(
        strategy_engine._combine_signals(signals, "VOTE", vote_threshold=1),
        expected_or,
    )

    pd.testing.assert_series_equal(
        strategy_engine._combine_signals(signals, "VOTE", vote_threshold=len(signals)),
        expected_and,
    )
