import importlib
import os
import sys
import types
from importlib import metadata
from importlib.metadata import PackageNotFoundError
from pathlib import Path

import pandas as pd
import pytest

import deps
import run_metadata
import vbt_stub
from deps import ensure_real_vectorbt

sys.modules.setdefault("vectorbt", vbt_stub)


@pytest.fixture(autouse=True)
def _vectorbt_stub():
    sys.modules["vectorbt"] = vbt_stub
    yield
    sys.modules["vectorbt"] = vbt_stub


def test_ensure_real_vectorbt_requires_accessor():
    orig_accessor = getattr(pd.Series, "vbt", None)
    if orig_accessor is not None:
        delattr(pd.Series, "vbt")
    try:
        with pytest.raises(ImportError):
            ensure_real_vectorbt()
    finally:
        if orig_accessor is not None:
            pd.Series.vbt = orig_accessor


def test_ensure_real_vectorbt_detects_stub_path(monkeypatch):
    monkeypatch.setattr(pd.Series, "vbt", object, raising=False)
    fake_path = Path(__file__).resolve().parent / "vectorbt.py"
    monkeypatch.setattr(sys.modules["vectorbt"], "__file__", str(fake_path))
    with pytest.raises(ImportError) as exc:
        ensure_real_vectorbt(Path(__file__).resolve().parents[1])
    assert "inside the repository" in str(exc.value)


def test_ensure_real_vectorbt_accepts_real_package():
    root = Path(__file__).resolve().parents[1]
    stub_module = sys.modules["vectorbt"]
    orig_accessor = getattr(pd.Series, "vbt", None)
    sys.modules.pop("vectorbt", None)
    sys_path = list(sys.path)
    sys.path = [p for p in sys.path if p != str(root)]
    try:
        real_vbt = importlib.import_module("vectorbt")
    except ModuleNotFoundError:
        sys.path = sys_path
        sys.modules["vectorbt"] = stub_module
        if orig_accessor is None and hasattr(pd.Series, "vbt"):
            delattr(pd.Series, "vbt")
        elif orig_accessor is not None:
            pd.Series.vbt = orig_accessor
        pytest.skip("real vectorbt not installed")
    sys.path = sys_path
    sys.modules["vectorbt"] = real_vbt
    try:
        ensure_real_vectorbt()
    finally:
        sys.modules["vectorbt"] = stub_module
        if orig_accessor is None and hasattr(pd.Series, "vbt"):
            delattr(pd.Series, "vbt")
        elif orig_accessor is not None:
            pd.Series.vbt = orig_accessor


def test_ensure_real_vectorbt_version_guard(monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(pd.Series, "vbt", object, raising=False)
    fake_path = tmp_path / "vectorbt.py"
    monkeypatch.setattr(vbt_stub, "__file__", str(fake_path))
    monkeypatch.setattr(sys.modules["vectorbt"], "__file__", str(fake_path))
    monkeypatch.setattr(
        Path,
        "is_relative_to",
        lambda self, other: (_ for _ in ()).throw(AttributeError),
    )

    def fake_version(name):
        raise PackageNotFoundError

    monkeypatch.setattr(deps, "version", fake_version)
    monkeypatch.setattr(sys.modules["vectorbt"], "__version__", "0.0.0", raising=False)
    with pytest.raises(ImportError) as exc:
        ensure_real_vectorbt(root)
    assert "Suspicious 'vectorbt' version" in str(exc.value)


def test_ensure_real_vectorbt_requires_portfolio(monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(pd.Series, "vbt", object, raising=False)
    fake_path = tmp_path / "vectorbt.py"
    monkeypatch.setattr(vbt_stub, "__file__", str(fake_path))
    monkeypatch.setattr(sys.modules["vectorbt"], "__file__", str(fake_path))
    monkeypatch.setattr(deps, "version", lambda name: "1.2.3")
    monkeypatch.setattr(sys.modules["vectorbt"], "__version__", "1.2.3", raising=False)
    monkeypatch.delattr(sys.modules["vectorbt"], "Portfolio", raising=False)
    with pytest.raises(ImportError) as exc:
        ensure_real_vectorbt(root)
    assert "vectorbt.Portfolio not found" in str(exc.value)


def test_ensure_real_vectorbt_is_relative_to_fallback(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(pd.Series, "vbt", object, raising=False)
    monkeypatch.setattr(
        Path,
        "is_relative_to",
        lambda self, other: (_ for _ in ()).throw(AttributeError),
    )
    fake_path = Path(__file__).resolve().parent / "vectorbt.py"
    monkeypatch.setattr(sys.modules["vectorbt"], "__file__", str(fake_path))
    with pytest.raises(ImportError):
        ensure_real_vectorbt(root)


def test_ensure_real_vectorbt_allows_site_packages(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(pd.Series, "vbt", object, raising=False)
    monkeypatch.setattr(vbt_stub, "__version__", "1.2.3", raising=False)
    fake_path = (
        root
        / "env"
        / "lib"
        / "python3.11"
        / "site-packages"
        / "vectorbt"
        / "__init__.py"
    )
    monkeypatch.setattr(sys.modules["vectorbt"], "__file__", str(fake_path))
    ensure_real_vectorbt(root)


def test_ensure_pandas_ta_requires_accessor(monkeypatch):
    stub = types.ModuleType("pandas_ta")
    monkeypatch.setitem(sys.modules, "pandas_ta", stub)
    orig = getattr(pd.DataFrame, "ta", None)
    if orig is not None:
        delattr(pd.DataFrame, "ta")
    try:
        with pytest.raises(ImportError):
            deps.ensure_pandas_ta()
    finally:
        if orig is not None:
            pd.DataFrame.ta = orig
        sys.modules.pop("pandas_ta", None)


def test_dependency_versions_pinned():
    allow_drift = os.getenv("ALLOW_DEP_DRIFT")
    for pkg, expected in deps.PINNED_DEPENDENCIES.items():
        try:
            ver = metadata.version(pkg)
        except PackageNotFoundError:
            pytest.skip(f"{pkg} not installed")
        if allow_drift and ver != expected:
            pytest.skip(f"{pkg} version {ver} != {expected}")
        assert ver == expected


def test_warn_if_deps_diverge_records(monkeypatch):
    recorded = {}

    def fake_write(meta):  # noqa: ANN001
        recorded.update(meta)

    monkeypatch.setattr(run_metadata, "_write_run_metadata", fake_write)

    def fake_version(name):  # noqa: ANN001
        return "0.0.0"

    monkeypatch.setattr(deps.importlib_metadata, "version", fake_version)
    deps.warn_if_deps_diverge()
    assert "dependency_mismatches" in recorded
