import sys
import types
from pathlib import Path

# Ensure repository root is on the import path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.modules.setdefault("pandas_ta", types.ModuleType("pandas_ta"))
sys.modules.setdefault("vectorbt", types.ModuleType("vectorbt"))

import config  # noqa: E402
import strategy_rules  # noqa: E402

config.initialize_config()


def _rsi_cond():
    return next(
        c
        for c in strategy_rules.STRATEGY_RULES["entry_rules"].get("conditions", [])
        if c.get("rule_name") == "RSI_Momentum_Filter"
    )


def test_rsi_period_bounds_configurable(monkeypatch):
    cond = _rsi_cond()
    assert cond["params"]["period"]["low"] == config.RSI_PERIOD_BOUNDS[0]
    assert cond["params"]["period"]["high"] == config.RSI_PERIOD_BOUNDS[1]

    monkeypatch.setattr(config, "RSI_PERIOD_BOUNDS", (9, 19), raising=False)
    config._apply_rsi_bounds()
    cond2 = _rsi_cond()
    assert cond2["params"]["period"]["low"] == 9
    assert cond2["params"]["period"]["high"] == 19

    # restore
    monkeypatch.setattr(config, "RSI_PERIOD_BOUNDS", (7, 21), raising=False)
    config._apply_rsi_bounds()
