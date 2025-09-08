import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ensure repo root on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# stub optional heavy deps
sys.modules.setdefault("vectorbt", types.ModuleType("vectorbt"))
sys.modules.setdefault("pandas_ta", types.ModuleType("pandas_ta"))

import indicator_library  # noqa: E402


def _base_df():
    return pd.DataFrame(
        {
            "Open": [1, 2, 3, 4],
            "High": [1, 2, 3, 4],
            "Low": [1, 2, 3, 4],
            "Close": [1, 2, 3, 4],
            "Volume": [1, 1, 1, 1],
        },
        index=pd.date_range("2020-01-01", periods=4, freq="D"),
    )


INDICATOR_CASES = [
    ("calculate_sma", "sma", "series", None, {"period": 2}),
    ("calculate_wma", "wma", "series", None, {"period": 2}),
    ("calculate_hma", "hma", "series", None, {"period": 2}),
    (
        "calculate_stoch",
        "stoch",
        "df",
        ["STOCHk_14", "STOCHd_14"],
        {"k": 14, "d": 3, "smooth_k": 3},
    ),
    ("calculate_cci", "cci", "series", None, {"period": 2}),
    ("calculate_williams_r", "willr", "series", None, {"period": 2}),
    ("calculate_tsi", "tsi", "series", None, {"long": 25, "short": 13}),
    (
        "calculate_ultimate_oscillator",
        "uo",
        "series",
        None,
        {"short": 7, "medium": 14, "long": 28},
    ),
    (
        "calculate_adx",
        "adx",
        "df",
        ["ADX_14", "DMP_14", "DMN_14"],
        {"period": 14},
    ),
    ("calculate_psar", "psar", "series", None, {}),
    (
        "calculate_keltner",
        "kc",
        "df",
        ["KCU_20", "KCM_20", "KCL_20"],
        {"period": 20, "multiplier": 2.0},
    ),
    (
        "calculate_donchian",
        "donchian",
        "df",
        ["DCU_20", "DCM_20", "DCL_20"],
        {"period": 20},
    ),
    ("calculate_stdev_channel", "stdev", "series", None, {"period": 5}),
    ("calculate_cmo", "cmo", "series", None, {"period": 5}),
    ("calculate_obv", "obv", "series", None, {}),
    ("calculate_mfi", "mfi", "series", None, {"period": 5}),
    ("calculate_adl", "ad", "series", None, {}),
    ("calculate_cmf", "cmf", "series", None, {"period": 5}),
    (
        "calculate_ma_envelope",
        "maenvelope",
        "df",
        ["UPPER", "LOWER"],
        {"period": 20, "percent": 2.5},
    ),
    (
        "calculate_ichimoku",
        "ichimoku",
        "df",
        ["ITS_9", "IKS_26", "ISA_9", "ISB_26", "ICS_26"],
        {"tenkan": 9, "kijun": 26, "senkou": 52},
    ),
    (
        "calculate_pivot_points",
        "pivot_points",
        "df",
        ["P", "R1", "S1"],
        {},
    ),
    ("calculate_trix", "trix", "series", None, {"period": 15}),
    ("calculate_roc", "roc", "series", None, {"period": 10}),
]


@pytest.mark.parametrize(
    "func_name, ta_method, out_kind, cols, params", INDICATOR_CASES
)
def test_indicator_output_shapes(
    func_name, ta_method, out_kind, cols, params, monkeypatch
):
    df = _base_df()
    if out_kind == "series":
        output = pd.Series(range(len(df)), index=df.index)
    else:
        cols = cols or ["col1"]
        output = pd.DataFrame({c: range(len(df)) for c in cols}, index=df.index)
    df.ta = types.SimpleNamespace(**{ta_method: lambda **kwargs: output})
    func = getattr(indicator_library, func_name)
    result = func(df, **params)
    assert len(result) == len(df)
    if isinstance(result, pd.DataFrame) and cols is not None:
        assert list(result.columns) == cols


BAD_PARAMS = [
    (indicator_library.calculate_sma, {"period": 0}),
    (indicator_library.calculate_sma, {"period": -1}),
    (indicator_library.calculate_wma, {"period": 0}),
    (indicator_library.calculate_hma, {"period": 0}),
    (indicator_library.calculate_stoch, {"k": 0, "d": 1, "smooth_k": 1}),
    (indicator_library.calculate_cci, {"period": 0}),
    (indicator_library.calculate_williams_r, {"period": 0}),
    (indicator_library.calculate_tsi, {"long": 0, "short": 1}),
    (indicator_library.calculate_tsi, {"long": 10, "short": 10}),
    (indicator_library.calculate_tsi, {"long": 10, "short": 12}),
    (
        indicator_library.calculate_ultimate_oscillator,
        {"short": 0, "medium": 1, "long": 2},
    ),
    (
        indicator_library.calculate_ultimate_oscillator,
        {"short": 14, "medium": 7, "long": 28},
    ),
    (indicator_library.calculate_adx, {"period": 0}),
    (indicator_library.calculate_psar, {"acceleration": 0.0}),
    (indicator_library.calculate_keltner, {"period": 0, "multiplier": 2.0}),
    (indicator_library.calculate_donchian, {"period": 0}),
    (indicator_library.calculate_stdev_channel, {"period": 0}),
    (indicator_library.calculate_cmo, {"period": 0}),
    (indicator_library.calculate_mfi, {"period": 0}),
    (indicator_library.calculate_cmf, {"period": 0}),
    (
        indicator_library.calculate_ma_envelope,
        {"period": 0, "percent": 2.5},
    ),
    (
        indicator_library.calculate_ichimoku,
        {"tenkan": 0, "kijun": 26, "senkou": 52},
    ),
    (indicator_library.calculate_trix, {"period": 0}),
    (indicator_library.calculate_trix, {"period": 10, "signal": 20}),
    (indicator_library.calculate_roc, {"period": 0}),
]


@pytest.mark.parametrize("func, params", BAD_PARAMS)
def test_indicator_invalid_params_raise(func, params):
    df = _base_df()
    df.ta = types.SimpleNamespace()
    with pytest.raises((ValueError, TypeError)):
        func(df, **params)


def test_volume_column_required():
    df = pd.DataFrame(
        {
            "Open": [1, 2],
            "High": [1, 2],
            "Low": [1, 2],
            "Close": [1, 2],
        }
    )
    df.ta = types.SimpleNamespace(
        obv=lambda **k: pd.Series([1, 2]),
        mfi=lambda **k: pd.Series([1, 2]),
        ad=lambda **k: pd.Series([1, 2]),
        cmf=lambda **k: pd.Series([1, 2]),
    )
    with pytest.raises(ValueError):
        indicator_library.calculate_obv(df)
    with pytest.raises(ValueError):
        indicator_library.calculate_mfi(df, period=2)
    with pytest.raises(ValueError):
        indicator_library.calculate_adl(df)
    with pytest.raises(ValueError):
        indicator_library.calculate_cmf(df, period=2)


def test_sma_expected_values():
    df = pd.DataFrame(
        {
            "Open": [1, 2, 3],
            "High": [1, 2, 3],
            "Low": [1, 2, 3],
            "Close": [1, 2, 3],
            "Volume": [1, 1, 1],
        }
    )

    def sma(length):
        s = df["Close"].rolling(length).mean()
        s.name = f"SMA_{length}"
        return s

    df.ta = types.SimpleNamespace(sma=sma)
    result = indicator_library.calculate_sma(df, period=2)
    expected = pd.Series([np.nan, 1.5, 2.5], name="SMA_2")
    pd.testing.assert_series_equal(result, expected)


def test_roc_expected_values():
    df = pd.DataFrame(
        {
            "Open": [1, 2, 3, 4],
            "High": [1, 2, 3, 4],
            "Low": [1, 2, 3, 4],
            "Close": [1, 2, 3, 4],
            "Volume": [1, 1, 1, 1],
        }
    )

    def roc(length):
        s = df["Close"].pct_change(length) * 100
        s.name = f"ROC_{length}"
        return s

    df.ta = types.SimpleNamespace(roc=roc)
    result = indicator_library.calculate_roc(df, period=1)
    expected = pd.Series([np.nan, 100.0, 50.0, 33.33333333333333], name="ROC_1")
    pd.testing.assert_series_equal(result, expected)
