import importlib
import sys
from importlib.metadata import PackageNotFoundError
from pathlib import Path

import pandas as pd
import pytest

import deps
import vbt_stub
from deps import ensure_real_vectorbt


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
    """ensure_real_vectorbt should succeed when the real package is installed."""
    sys.modules.pop("vectorbt", None)
    real_vbt = importlib.import_module("vectorbt")
    sys.modules["vectorbt"] = real_vbt
    # Our virtual environment lives inside the repository; pass a dummy
    # path to avoid false positives when validating the install location.
    ensure_real_vectorbt(Path(__file__).resolve().parents[1] / "dummy")


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
