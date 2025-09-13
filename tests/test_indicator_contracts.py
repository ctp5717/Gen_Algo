import pandas as pd
import pytest

import indicator_contracts as contracts
import indicator_library as lib


def _sample_df():
    idx = pd.date_range("2020", periods=120)
    vals = range(len(idx))
    return pd.DataFrame(
        {
            "Open": vals,
            "High": [x + 1 for x in vals],
            "Low": [x - 1 for x in vals],
            "Close": [x + 0.5 for x in vals],
        },
        index=idx,
    )


@pytest.mark.parametrize(
    "name,func,params",
    [
        ("macd", lib.calculate_macd, {"fast": 12, "slow": 26, "signal": 9}),
        ("stoch", lib.calculate_stoch, {"k": 14, "d": 3, "smooth_k": 3}),
        ("adx", lib.calculate_adx, {"period": 14}),
        ("bbands", lib.calculate_bbands, {"period": 20, "std_dev": 2}),
        ("psar", lib.calculate_psar, {"acceleration": 0.02, "maximum": 0.2}),
        ("keltner", lib.calculate_keltner, {"period": 20, "multiplier": 2}),
        ("donchian", lib.calculate_donchian, {"period": 20}),
        ("trix", lib.calculate_trix, {"period": 9, "signal": 3}),
        ("ichimoku", lib.calculate_ichimoku, {"tenkan": 9, "kijun": 26, "senkou": 52}),
    ],
)
def test_contract_columns(name, func, params):
    df = _sample_df()
    out = func(df, **params)
    expected = contracts.CONTRACTS[name](**params)
    assert list(out.columns) == expected
