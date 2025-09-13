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

import indicator_contracts as contracts  # noqa: E402
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
    cols = contracts.CONTRACTS["adx"]()
    output = pd.DataFrame(
        {
            cols[0]: [10, 20, 30, 40],
            cols[1]: [0, 0, 0, 0],
            cols[2]: [1, 1, 1, 1],
            cols[3]: [1, 1, 1, 1],
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
    cols = contracts.CONTRACTS["stoch"]()
    output = pd.DataFrame(
        {
            cols[0]: [10, 20, 30, 40],
            cols[1]: [1, 1, 1, 1],
            cols[2]: [0, 0, 0, 0],
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
    cols = contracts.CONTRACTS["keltner"](period=20, multiplier=2.0)
    output = pd.DataFrame(
        {
            cols[2]: [5, 5, 5, 5],
            cols[1]: [0, 1, 2, 3],
            cols[0]: [0, 0, 0, 0],
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
    cols = contracts.CONTRACTS["donchian"](period=20)
    output = pd.DataFrame(
        {
            cols[2]: [5, 5, 5, 5],
            cols[1]: [0, 1, 2, 3],
            cols[0]: [0, 0, 0, 0],
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
    cols = contracts.CONTRACTS["ichimoku"]()
    output = pd.DataFrame(
        {
            cols[1]: [0, 0, 0, 0],
            cols[0]: [0, 0, 0, 5],
            cols[2]: [0, 0, 0, 0],
            cols[3]: [0, 0, 0, 0],
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
    cols = contracts.CONTRACTS["trix"](period=15, signal=9)
    output = pd.DataFrame(
        {
            cols[0]: [0, 0, 1, 2],
            cols[1]: [0, 0, 0, 0],
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
    "indicator_name,params",
    [
        ("bbands", {"period": 20, "std_dev": 2.0}),
        ("keltner", {"period": 20, "multiplier": 2.0}),
        ("donchian", {"period": 20}),
    ],
)
def test_band_hint_and_strict_column(indicator_name, params):
    cols = contracts.CONTRACTS[indicator_name](**params)
    df = pd.DataFrame(
        {
            cols[2]: [5, 5, 5, 5],
            cols[1]: [0, 1, 2, 3],
            cols[0]: [0, 0, 0, 0],
        },
        index=pd.date_range("2020-01-01", periods=4, freq="D"),
    )
    series = strategy_engine.select_indicator_series(
        indicator_name,
        df,
        {"type": "price_is_above_indicator", "band": "lower"},
        True,
    )
    pd.testing.assert_series_equal(series, df[cols[0]])

    with pytest.raises(KeyError):
        strategy_engine.select_indicator_series(
            indicator_name,
            df.drop(columns=cols[2]),
            {"type": "price_is_above_indicator", "band": "upper"},
            True,
        )

    with pytest.warns(UserWarning):
        series_fb = strategy_engine.select_indicator_series(
            indicator_name,
            df.drop(columns=cols[2]),
            {"type": "price_is_above_indicator", "band": "upper"},
            False,
        )
    pd.testing.assert_series_equal(series_fb, df.drop(columns=cols[2]).iloc[:, 0])


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


def test_strict_column_false_fails():
    df = _base_df()
    output = pd.DataFrame({"ADX_14": [10, 20, 30, 40]}, index=df.index)
    with pytest.raises(contracts.IndicatorContractError):
        _run_rule(
            df,
            "adx",
            output,
            {"type": "indicator_is_above_value", "value": 15, "column": "DMX"},
            {"strict_column": False},
        )


def test_strict_column_true_raises_contract_error():
    df = _base_df()
    output = pd.DataFrame({"ADX_14": [10, 20, 30, 40]}, index=df.index)
    with pytest.raises(contracts.IndicatorContractError):
        _run_rule(
            df,
            "adx",
            output,
            {"type": "indicator_is_above_value", "value": 15, "column": "DMX"},
            {"strict_column": True},
        )


def test_condition_override_missing_column_errors():
    df = _base_df()
    output = pd.DataFrame({"ADX_14": [10, 20, 30, 40]}, index=df.index)
    with pytest.raises(contracts.IndicatorContractError):
        _run_rule(
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


def test_condition_override_raises_contract_error():
    df = _base_df()
    output = pd.DataFrame({"ADX_14": [10, 20, 30, 40]}, index=df.index)
    with pytest.raises(contracts.IndicatorContractError):
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
