import importlib
import sys
import types
from pathlib import Path

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
