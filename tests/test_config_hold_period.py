import importlib
import sys
import types
from pathlib import Path

# Ensure repository root is on the import path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Stub heavy optional dependencies before importing config indirectly
sys.modules.setdefault("pandas_ta", types.ModuleType("pandas_ta"))
sys.modules.setdefault("vectorbt", types.ModuleType("vectorbt"))


def test_hold_period_converts_days_to_bars(monkeypatch):
    sys.modules.pop("config", None)
    import config

    monkeypatch.setattr(config, "TIMEFRAME", "15m", raising=False)
    importlib.reload(config)
    assert config.MAX_HOLD_PERIOD == 14 * 96


def test_hold_period_hour_timeframe(monkeypatch):
    sys.modules.pop("config", None)
    import config

    monkeypatch.setattr(config, "TIMEFRAME", "4h", raising=False)
    importlib.reload(config)
    assert config.MAX_HOLD_PERIOD == 14 * 6
