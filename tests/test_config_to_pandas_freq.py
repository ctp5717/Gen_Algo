import importlib
import sys
import types
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Stub optional heavy dependencies
sys.modules.setdefault("pandas_ta", types.ModuleType("pandas_ta"))
sys.modules.setdefault("vectorbt", types.ModuleType("vectorbt"))


def test_to_pandas_freq_uses_min_no_future_warning():
    sys.modules.pop("config", None)
    import config

    importlib.reload(config)
    freq = config.to_pandas_freq("15m")
    assert freq == "15min"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", FutureWarning)
        pd.Timedelta(freq)
        assert not any("deprecated" in str(warn.message) for warn in w)


def test_to_pandas_freq_hour_uppercase(monkeypatch):
    sys.modules.pop("config", None)
    import config

    monkeypatch.setattr(config, "TIMEFRAME", "1h", raising=False)
    importlib.reload(config)
    assert config.to_pandas_freq("1h") == "1H"
