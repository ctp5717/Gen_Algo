import sys
import types
from pathlib import Path

# Ensure repository root is on the import path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Stub heavy optional dependencies before importing modules that use them
sys.modules.setdefault('pandas_ta', types.ModuleType('pandas_ta'))
sys.modules.setdefault('vectorbt', types.ModuleType('vectorbt'))

import pandas as pd  # noqa: E402
import fitness  # noqa: E402
import logging  # noqa: E402


def test_exception_logging(caplog, monkeypatch):
    """FitnessEvaluator logs exception messages"""
    ohlc = pd.DataFrame({'Close': [1, 2, 3]}, index=pd.date_range('2020', periods=3, freq='D'))
    evaluator = fitness.FitnessEvaluator(ohlc, {}, {})

    def raise_error(*args, **kwargs):
        raise ValueError("boom")

    monkeypatch.setattr(fitness.engine, 'process_strategy_rules', raise_error)

    with caplog.at_level(logging.ERROR):
        score = evaluator(None, [], 0)

    assert "Fitness evaluation failed" in caplog.text
    assert score == -999.0
