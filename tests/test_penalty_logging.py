import logging
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

sys.modules.setdefault('pandas_ta', types.ModuleType('pandas_ta'))
sys.modules.setdefault('vectorbt', types.ModuleType('vectorbt'))

import fitness  # noqa: E402
import config  # noqa: E402


def test_penalty_logging_deduplicates(monkeypatch, caplog):
    monkeypatch.setattr(config, "VERBOSE_FITNESS_LOGS", False, raising=False)
    fitness._penalty_counts.clear()
    caplog.set_level(logging.WARNING)
    fitness._log_penalty_metrics({"vol": 0}, 1, "reason", 0)
    fitness._log_penalty_metrics({"vol": 0}, 1, "reason", 0)
    assert caplog.text.count("fitness_penalty") == 1


def test_penalty_logging_verbose(monkeypatch, caplog):
    monkeypatch.setattr(config, "VERBOSE_FITNESS_LOGS", True, raising=False)
    fitness._penalty_counts.clear()
    caplog.set_level(logging.WARNING)
    fitness._log_penalty_metrics({"vol": 0}, 1, "reason", 0)
    fitness._log_penalty_metrics({"vol": 0}, 1, "reason", 0)
    assert caplog.text.count("fitness_penalty") == 2
