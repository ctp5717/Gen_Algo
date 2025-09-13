import importlib
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

# Ensure repository root is on the import path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_indicator_library_imports_pandas_ta(monkeypatch):
    stub = types.ModuleType("pandas_ta")
    monkeypatch.setitem(sys.modules, "pandas_ta", stub)
    import indicator_library

    importlib.reload(indicator_library)
    assert hasattr(
        indicator_library, "ta"
    ), "indicator_library should import pandas_ta as 'ta'"

    # The module should ensure numpy.NaN exists for pandas_ta compatibility
    import numpy as np

    assert hasattr(np, "NaN"), "indicator_library should define numpy.NaN"
    assert np.NaN is np.nan


@pytest.mark.parametrize(
    "alias,target,kwargs",
    [
        ("uo", "ultimate_oscillator", {"short": 2, "medium": 3, "long": 4}),
        ("willr", "williams_r", {"period": 2}),
        ("kc", "keltner", {"period": 2, "multiplier": 1.5}),
        ("dmi", "adx", {"period": 2}),
        ("bb", "bbands", {"period": 2, "std_dev": 2}),
        ("bollinger", "bbands", {"period": 2, "std_dev": 2}),
        ("keltner_channels", "keltner", {"period": 2, "multiplier": 1.5}),
    ],
)
def test_indicator_aliases(alias, target, kwargs):
    df = pd.DataFrame(
        {
            "Open": range(1, 6),
            "High": range(1, 6),
            "Low": range(1, 6),
            "Close": range(1, 6),
            "Volume": [1] * 5,
        }
    )
    import importlib

    import indicator_library as il
    import strategy_engine as se

    importlib.reload(il)
    importlib.reload(se)

    res_alias = se.INDICATOR_MAPPING[alias](df, **kwargs)
    res_full = se.INDICATOR_MAPPING[target](df, **kwargs)
    if isinstance(res_alias, pd.DataFrame):
        pd.testing.assert_frame_equal(res_alias, res_full)
    else:
        pd.testing.assert_series_equal(res_alias, res_full)


def test_indicator_lookup_case_insensitive():
    import strategy_engine as se

    df = pd.DataFrame(
        {
            "Open": range(1, 6),
            "High": range(1, 6),
            "Low": range(1, 6),
            "Close": range(1, 6),
            "Volume": [1] * 5,
        }
    )
    rules_upper = {
        "entry_rules": {
            "conditions": [
                {
                    "indicator": "UO",
                    "params": {"short": 2, "medium": 3, "long": 4},
                    "condition": {
                        "type": "indicator_is_above_value",
                        "value": 50,
                    },
                }
            ]
        }
    }
    rules_full = {
        "entry_rules": {
            "conditions": [
                {
                    "indicator": "ultimate_oscillator",
                    "params": {"short": 2, "medium": 3, "long": 4},
                    "condition": {
                        "type": "indicator_is_above_value",
                        "value": 50,
                    },
                }
            ]
        }
    }
    sig1 = se.process_strategy_rules(df, rules_upper)
    sig2 = se.process_strategy_rules(df, rules_full)
    pd.testing.assert_series_equal(sig1, sig2)
