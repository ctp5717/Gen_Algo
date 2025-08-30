import importlib
import sys
from pathlib import Path

import pandas as pd
import pytest

from importlib.metadata import PackageNotFoundError

import deps
from deps import ensure_real_vectorbt
import vbt_stub


def test_ensure_real_vectorbt_requires_accessor():
    with pytest.raises(ImportError):
        ensure_real_vectorbt()


def test_ensure_real_vectorbt_detects_stub_path(monkeypatch):
    monkeypatch.setattr(pd.Series, "vbt", object, raising=False)
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
    finally:
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
    monkeypatch.setattr(Path, "is_relative_to", lambda self, other: (_ for _ in ()).throw(AttributeError))

    def fake_version(name):
        raise PackageNotFoundError

    monkeypatch.setattr(deps, "version", fake_version)
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
    monkeypatch.setattr(vbt_stub, "__version__", "1.2.3", raising=False)
    monkeypatch.delattr(vbt_stub, "Portfolio", raising=False)
    with pytest.raises(ImportError) as exc:
        ensure_real_vectorbt(root)
    assert "vectorbt.Portfolio not found" in str(exc.value)


def test_ensure_real_vectorbt_is_relative_to_fallback(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(pd.Series, "vbt", object, raising=False)
    monkeypatch.setattr(Path, "is_relative_to", lambda self, other: (_ for _ in ()).throw(AttributeError))
    with pytest.raises(ImportError):
        ensure_real_vectorbt(root)
