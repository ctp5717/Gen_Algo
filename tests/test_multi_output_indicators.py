import sys
import types
from pathlib import Path

import pandas as pd
import pytest

# ensure repo root on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# stub heavy deps
sys.modules.setdefault("pandas_ta", types.ModuleType("pandas_ta"))
sys.modules.setdefault("vectorbt", types.ModuleType("vectorbt"))

import indicator_library  # noqa: E402
import strategy_engine  # noqa: E402


def _base_df():
    return pd.DataFrame(
        {
            "Open": [1, 2, 3, 4],
            "High": [2, 3, 4, 5],
            "Low": [0, 1, 2, 3],
            "Close": [1, 2, 3, 4],
            "Volume": [1, 1, 1, 1],
        },
        index=pd.date_range("2020-01-01", periods=4, freq="D"),
    )


def _run_rule(df, indicator_name, output_df, condition, entry_extra=None):
    def func(data, **p):
        return output_df

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(indicator_library, f"calculate_{indicator_name}", func)
    monkeypatch.setitem(strategy_engine.INDICATOR_MAPPING, indicator_name, func)
    monkeypatch.setitem(indicator_library.INDICATOR_REGISTRY, indicator_name, func)
    rules = {
        "entry_rules": {
            **(entry_extra or {}),
            "conditions": [
                {
                    "indicator": indicator_name,
                    "params": {},
                    "condition": condition,
                }
            ],
        }
    }
    try:
        entries = strategy_engine.process_strategy_rules(df, rules)
    finally:
        monkeypatch.undo()
    return entries


def test_adx_defaults_to_adx_line():
    df = _base_df()
    output = pd.DataFrame(
        {
            "ADX_14": [10, 20, 30, 40],
            "DMP_14": [1, 1, 1, 1],
            "DMN_14": [1, 1, 1, 1],
        },
        index=df.index,
    )
    entries = _run_rule(
        df,
        "adx",
        output,
        {"type": "indicator_is_above_value", "value": 15},
    )
    assert entries.iloc[-1]


def test_stoch_defaults_to_k():
    df = _base_df()
    output = pd.DataFrame(
        {
            "STOCHk_14": [10, 20, 30, 40],
            "STOCHd_14": [1, 1, 1, 1],
        },
        index=df.index,
    )
    entries = _run_rule(
        df,
        "stoch",
        output,
        {"type": "indicator_is_above_value", "value": 15},
    )
    assert entries.iloc[-1]


def test_keltner_defaults_to_middle_band():
    df = _base_df()
    output = pd.DataFrame(
        {
            "KCU_20": [5, 5, 5, 5],
            "KCM_20": [0, 1, 2, 3],
            "KCL_20": [0, 0, 0, 0],
        },
        index=df.index,
    )
    entries = _run_rule(
        df,
        "keltner",
        output,
        {"type": "price_is_above_indicator"},
    )
    assert entries.iloc[-1]


def test_donchian_defaults_to_middle_band():
    df = _base_df()
    output = pd.DataFrame(
        {
            "DCU_20": [5, 5, 5, 5],
            "DCM_20": [0, 1, 2, 3],
            "DCL_20": [0, 0, 0, 0],
        },
        index=df.index,
    )
    entries = _run_rule(
        df,
        "donchian",
        output,
        {"type": "price_is_above_indicator"},
    )
    assert entries.iloc[-1]


def test_ma_envelope_defaults_to_middle_band():
    df = _base_df()
    output = pd.DataFrame(
        {
            "MAE_U_2_2.0": [5, 5, 5, 5],
            "MAE_M_2_2.0": [0, 1, 2, 3],
            "MAE_L_2_2.0": [0, 0, 0, 0],
        },
        index=df.index,
    )
    entries = _run_rule(
        df,
        "ma_envelope",
        output,
        {"type": "price_is_above_indicator"},
    )
    assert entries.iloc[-1]


def test_ichimoku_defaults_to_baseline():
    df = _base_df()
    output = pd.DataFrame(
        {
            "ITS_9": [0, 0, 0, 0],
            "IKS_26": [0, 0, 0, 5],
            "ISA_9": [0, 0, 0, 0],
        },
        index=df.index,
    )
    entries = _run_rule(
        df,
        "ichimoku",
        output,
        {"type": "indicator_is_above_value", "value": 1},
    )
    assert entries.iloc[-1]


def test_pivot_points_defaults_to_pivot():
    df = _base_df()
    output = pd.DataFrame(
        {
            "P": [0, 1, 2, 3],
            "R1": [10, 10, 10, 10],
            "S1": [0, 0, 0, 0],
        },
        index=df.index,
    )
    entries = _run_rule(
        df,
        "pivot_points",
        output,
        {"type": "indicator_is_above_value", "value": 2},
    )
    assert entries.iloc[-1]


def test_trix_defaults_to_trix_line():
    df = _base_df()
    output = pd.DataFrame(
        {
            "TRIX_15": [0, 0, 1, 2],
            "TRIXs_15": [0, 0, 0, 0],
        },
        index=df.index,
    )
    entries = _run_rule(
        df,
        "trix",
        output,
        {"type": "indicator_is_above_value", "value": 1},
    )
    assert entries.iloc[-1]


@pytest.mark.parametrize(
    "indicator_name,prefixes",
    [
        ("bbands", strategy_engine.INDICATOR_COLUMN_PREFIXES["bbands"]),
        ("keltner", strategy_engine.INDICATOR_COLUMN_PREFIXES["keltner"]),
        ("donchian", strategy_engine.INDICATOR_COLUMN_PREFIXES["donchian"]),
    ],
)
def test_band_hint_and_strict_column(indicator_name, prefixes):
    df = pd.DataFrame(
        {
            f"{prefixes['upper']}_20": [5, 5, 5, 5],
            f"{prefixes['middle']}_20": [0, 1, 2, 3],
            f"{prefixes['lower']}_20": [0, 0, 0, 0],
        },
        index=pd.date_range("2020-01-01", periods=4, freq="D"),
    )
    series = strategy_engine.select_indicator_series(
        indicator_name,
        df,
        {"type": "price_is_above_indicator", "band": "lower"},
        True,
    )
    pd.testing.assert_series_equal(series, df[f"{prefixes['lower']}_20"])

    with pytest.raises(KeyError):
        strategy_engine.select_indicator_series(
            indicator_name,
            df.drop(columns=f"{prefixes['upper']}_20"),
            {"type": "price_is_above_indicator", "band": "upper"},
            True,
        )

    with pytest.warns(UserWarning):
        series_fb = strategy_engine.select_indicator_series(
            indicator_name,
            df.drop(columns=f"{prefixes['upper']}_20"),
            {"type": "price_is_above_indicator", "band": "upper"},
            False,
        )
    pd.testing.assert_series_equal(
        series_fb, df.drop(columns=f"{prefixes['upper']}_20").iloc[:, 0]
    )


def test_macd_hist_default_and_strict_column():
    df = pd.DataFrame(
        {
            "MACDh_12_26_9": [0, 1, 2, 3],
            "MACD_12_26_9": [0, 0, 0, 0],
        },
        index=pd.date_range("2020-01-01", periods=4, freq="D"),
    )
    series = strategy_engine.select_indicator_series(
        "macd", df, {"type": "indicator_is_above_value"}, True
    )
    pd.testing.assert_series_equal(series, df["MACDh_12_26_9"])

    with pytest.raises(KeyError):
        strategy_engine.select_indicator_series(
            "macd",
            df,
            {
                "type": "indicator_is_above_value",
                "column": "bogus",
            },
            True,
        )

    with pytest.warns(UserWarning):
        series_fb = strategy_engine.select_indicator_series(
            "macd",
            df,
            {
                "type": "indicator_is_above_value",
                "column": "bogus",
            },
            False,
        )
    pd.testing.assert_series_equal(series_fb, df.iloc[:, 0])


def test_strict_column_false_falls_back():
    df = _base_df()
    output = pd.DataFrame({"ADX_14": [10, 20, 30, 40]}, index=df.index)
    with pytest.warns(UserWarning):
        entries = _run_rule(
            df,
            "adx",
            output,
            {"type": "indicator_is_above_value", "value": 15, "column": "DMX"},
            {"strict_column": False},
        )
    assert entries.iloc[-1]


def test_strict_column_true_raises():
    df = _base_df()
    output = pd.DataFrame({"ADX_14": [10, 20, 30, 40]}, index=df.index)
    with pytest.raises(KeyError):
        _run_rule(
            df,
            "adx",
            output,
            {"type": "indicator_is_above_value", "value": 15, "column": "DMX"},
            {"strict_column": True},
        )


def test_condition_override_falls_back():
    df = _base_df()
    output = pd.DataFrame({"ADX_14": [10, 20, 30, 40]}, index=df.index)
    with pytest.warns(UserWarning):
        entries = _run_rule(
            df,
            "adx",
            output,
            {
                "type": "indicator_is_above_value",
                "value": 15,
                "column": "DMX",
                "strict_column": False,
            },
            {"strict_column": True},
        )
    assert entries.iloc[-1]


def test_condition_override_raises():
    df = _base_df()
    output = pd.DataFrame({"ADX_14": [10, 20, 30, 40]}, index=df.index)
    with pytest.raises(KeyError):
        _run_rule(
            df,
            "adx",
            output,
            {
                "type": "indicator_is_above_value",
                "value": 15,
                "column": "DMX",
                "strict_column": True,
            },
            {"strict_column": False},
        )
