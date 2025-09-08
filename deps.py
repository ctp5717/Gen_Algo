import importlib.metadata as importlib_metadata
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd

if sys.version_info >= (3, 12):
    PINNED_DEPENDENCIES = {
        "pandas": "2.2.2",
        "numpy": "1.26.4",
        "pandas-ta": "0.3.14b0",
    }
else:
    PINNED_DEPENDENCIES = {
        "pandas": "2.0.3",
        "numpy": "1.24.4",
        "pandas-ta": "0.3.14b0",
    }

import warnings

import run_metadata


def ensure_real_vectorbt(repo_root: Path | None = None) -> None:
    """Ensure the real vectorbt package is installed and not shadowed."""
    import vectorbt as vbt

    if not hasattr(pd.Series, "vbt"):
        raise ImportError(
            "Pandas accessor Series.vbt is missing. Install the real 'vectorbt'."
        )

    vbt_path = Path(getattr(vbt, "__file__", "")).resolve()
    if repo_root is None:
        repo_root = Path(__file__).resolve().parent

    try:
        inside_repo = vbt_path.is_relative_to(repo_root)
    except AttributeError:
        inside_repo = str(vbt_path).startswith(str(repo_root))

    if inside_repo:
        allowed = {"site-packages", "dist-packages"}
        if not any(part in allowed for part in vbt_path.parts):
            raise ImportError(
                f"Loaded 'vectorbt' from {vbt_path}, which is inside the repository. "
                "Remove/rename any local stubs."
            )

    try:
        pkg_ver = version("vectorbt")
    except PackageNotFoundError:
        pkg_ver = getattr(vbt, "__version__", "unknown")
    if pkg_ver in ("0.0.0", "unknown"):
        raise ImportError(
            "Suspicious 'vectorbt' version "
            f"'{pkg_ver}' at {vbt_path}. The real package must be installed."
        )

    if not hasattr(vbt, "Portfolio"):
        raise ImportError(
            "vectorbt.Portfolio not found; ensure the full package is installed."
        )


def ensure_pandas_ta() -> None:
    """Ensure pandas-ta is installed and DataFrame.ta accessor exists."""
    try:
        import pandas_ta  # noqa: F401
    except Exception as exc:
        raise ImportError(
            "Missing optional dependency 'pandas-ta'. Install it to enable the "
            "indicator library."
        ) from exc

    if not hasattr(pd.DataFrame, "ta"):
        raise ImportError(
            "pandas-ta is installed but DataFrame.ta accessor is missing."
        )


def warn_if_deps_diverge() -> None:
    """Warn and record if runtime deps deviate from pinned versions."""
    mismatches: dict[str, dict[str, str | None]] = {}
    for pkg, expected in PINNED_DEPENDENCIES.items():
        try:
            actual = importlib_metadata.version(pkg)
            if actual != expected:
                mismatches[pkg] = {"expected": expected, "actual": actual}
        except Exception:
            mismatches[pkg] = {"expected": expected, "actual": None}
    if mismatches:
        try:
            run_metadata._write_run_metadata({"dependency_mismatches": mismatches})
        except Exception:
            pass
        warnings.warn(f"Dependency mismatches detected: {mismatches}", stacklevel=2)
