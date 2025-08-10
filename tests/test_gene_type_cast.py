import sys
import importlib
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_float_solution_casts_to_int_and_executes(monkeypatch):
    # Ensure the real vectorbt library is imported
    monkeypatch.delitem(sys.modules, "vectorbt", raising=False)
    import vectorbt  # noqa: F401

    # Reload modules that depend on vectorbt
    for module_name in ["indicator_library", "strategy_engine", "fitness"]:
        monkeypatch.delitem(sys.modules, module_name, raising=False)
    _ = importlib.import_module("indicator_library")
    strategy_engine = importlib.import_module("strategy_engine")
    fitness = importlib.import_module("fitness")

    df = pd.DataFrame(
        {
            "Open": [1, 2, 3],
            "High": [1, 2, 3],
            "Low": [1, 2, 3],
            "Close": [1, 2, 3],
            "Volume": [1, 1, 1],
        },
        index=pd.date_range("2020-01-01", periods=3),
    )

    base_rules = {
        "entry_rules": {
            "combination_logic": "AND",
            "conditions": [
                {
                    "indicator": "rsi",
                    "params": {"period": 2},
                    "condition": {
                        "type": "indicator_is_above_value",
                        "value": 50,
                    },
                    "is_active": True,
                }
            ],
        }
    }

    gene_map = {
        0: {
            "name": "rsi_period",
            "path": ["entry_rules", "conditions", 0, "params", "period"],
            "type": int,
        }
    }

    # Provide float solution for integer gene; should be cast to int
    rules = fitness._inject_genes_into_rules(base_rules, gene_map, [2.0])
    assert isinstance(
        rules["entry_rules"]["conditions"][0]["params"]["period"], int
    )

    # Should execute without raising NumbaTypeError
    signal = strategy_engine.process_strategy_rules(df, rules)
    assert not signal.isna().any()
