import sys
import types
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Stub heavy optional dependencies to keep import light in tests
sys.modules.setdefault('pandas_ta', types.ModuleType('pandas_ta'))
sys.modules.setdefault('vectorbt', types.ModuleType('vectorbt'))


def test_main_sets_info_logging(monkeypatch):
    """Importing main should configure the root logger at INFO level."""
    # Ensure previous imports don't influence the logger state
    sys.modules.pop('main', None)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.NOTSET)
    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)

    import main  # noqa: F401

    assert logging.getLogger().getEffectiveLevel() == logging.INFO
