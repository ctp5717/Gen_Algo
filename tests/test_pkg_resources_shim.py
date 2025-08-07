import sys
import types
import importlib
from pathlib import Path

# Ensure repo root on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_indicator_library_provides_pkg_resources_stub(monkeypatch):
    # Remove modules to force indicator_library to set up the shim
    for mod in ["pkg_resources", "pandas_ta", "indicator_library"]:
        monkeypatch.delitem(sys.modules, mod, raising=False)
    # Stub pandas_ta so importing indicator_library does not require the package
    monkeypatch.setitem(sys.modules, "pandas_ta", types.ModuleType("pandas_ta"))

    import indicator_library
    importlib.reload(indicator_library)

    pkg = sys.modules.get("pkg_resources")
    assert pkg is not None
    assert hasattr(pkg, "get_distribution")
    assert hasattr(pkg, "DistributionNotFound")
    # The shim should be able to report versions using importlib.metadata
    dist = pkg.get_distribution("pytest")
    assert isinstance(dist.version, str)
    # pandas_ta also expects distributions to expose the installation path via
    # the ``location`` attribute.  Ensure our shim provides it.
    assert hasattr(dist, "location"), "distribution should expose location"
    assert Path(dist.location).exists()
