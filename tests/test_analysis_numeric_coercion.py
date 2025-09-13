import sys
import types
import warnings

import pandas as pd

sys.modules.setdefault("vectorbt", types.ModuleType("vectorbt"))
import analysis  # noqa: E402


def test_numeric_coercion_only_numeric_cols():
    df = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "score": ["1.0", "bad"],
            "trades": [5, "oops"],
            "note": ["x", "y"],
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        out = analysis._coerce_numeric_cols(df.copy(), ["score", "trades"])
        out[["score", "trades"]] = out[["score", "trades"]].fillna(0)
    assert list(out["ticker"]) == ["AAA", "BBB"]
    assert list(out["note"]) == ["x", "y"]
    assert list(out["score"]) == [1.0, 0.0]
    assert list(out["trades"]) == [5.0, 0.0]
