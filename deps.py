from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd


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
