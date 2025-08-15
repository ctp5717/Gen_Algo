import sys
import types
from pathlib import Path
import importlib

# Ensure repository root is on the import path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_indicator_library_lazily_imports_pandas_ta(monkeypatch):
    stub = types.ModuleType('pandas_ta')
    monkeypatch.setitem(sys.modules, 'pandas_ta', stub)

    import indicator_library
    importlib.reload(indicator_library)

    assert indicator_library.ta is None
    ta_mod = indicator_library._get_ta()
    assert ta_mod is stub
    assert indicator_library.ta is stub

    # The module should ensure numpy.NaN exists for pandas_ta compatibility
    import numpy as np
    assert hasattr(np, "NaN"), "indicator_library should define numpy.NaN"
    assert np.NaN is np.nan
