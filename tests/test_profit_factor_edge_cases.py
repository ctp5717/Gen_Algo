import sys
import types
from pathlib import Path

# Ensure repository root is on the import path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Stub heavy optional dependencies
sys.modules.setdefault("pandas_ta", types.ModuleType("pandas_ta"))
sys.modules.setdefault("vectorbt", types.ModuleType("vectorbt"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import fitness  # noqa: E402


def _make_evaluator(stats, settings=None):
    """Utility to construct a MultiAssetFitnessEvaluator with patched stats."""
    group_data = {
        "A": pd.DataFrame({"Close": [1, 2, 3]}),
    }
    evaluator = fitness.MultiAssetFitnessEvaluator(group_data, {}, {}, settings or {})

    def fake_eval(self, ohlc, rules):
        return stats

    evaluator._evaluate_single_asset = types.MethodType(fake_eval, evaluator)
    return evaluator


def test_near_zero_losses_winsorized():
    profit = 1000.0
    loss = 1e-9
    pf_raw = profit / loss
    stats = {
        "sortino": 1.0,
        "profit_factor": pf_raw,
        "max_drawdown": 10.0,
        "trades": 5,
        "total_return": 1.0,
    }
    settings = {
        "winsorize_pf_cap": 5.0,
        "trade_floor_policy": "hard_floor",
        "min_total_trades": 0,
        "lambda_dispersion": 0.0,
    }
    ev = _make_evaluator(stats, settings)
    ev(None, [], 0)
    details = ev.last_details["per_asset"]["A"]
    assert details["profit_factor"] == pf_raw
    assert details["profit_factor_capped"] == 5.0


def test_negative_profit_factor_not_capped():
    profit = -100.0
    loss = 10.0
    pf_raw = profit / loss
    stats = {
        "sortino": 1.0,
        "profit_factor": pf_raw,
        "max_drawdown": 10.0,
        "trades": 5,
        "total_return": 1.0,
    }
    settings = {
        "winsorize_pf_cap": 5.0,
        "trade_floor_policy": "hard_floor",
        "min_total_trades": 0,
        "lambda_dispersion": 0.0,
    }
    ev = _make_evaluator(stats, settings)
    ev(None, [], 0)
    details = ev.last_details["per_asset"]["A"]
    assert details["profit_factor"] == pf_raw
    assert details["profit_factor_capped"] == pf_raw


def test_nan_profit_factor_fallback():
    pf_raw = float("nan")
    stats = {
        "sortino": 1.0,
        "profit_factor": pf_raw,
        "max_drawdown": 10.0,
        "trades": 5,
        "total_return": 1.0,
    }
    settings = {
        "winsorize_pf_cap": 5.0,
        "nan_fallback": -1.0,
        "trade_floor_policy": "hard_floor",
        "min_total_trades": 0,
        "lambda_dispersion": 0.0,
    }
    ev = _make_evaluator(stats, settings)
    ev(None, [], 0)
    details = ev.last_details["per_asset"]["A"]
    assert np.isnan(details["profit_factor"])
    assert details["profit_factor_capped"] == -1.0
