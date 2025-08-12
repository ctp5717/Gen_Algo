import logging
import sys
import types
from pathlib import Path
import multiprocessing
import logging.handlers

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

sys.modules.setdefault('pandas_ta', types.ModuleType('pandas_ta'))
sys.modules.setdefault('vectorbt', types.ModuleType('vectorbt'))

import fitness  # noqa: E402
import config  # noqa: E402


def test_penalty_logging_deduplicates(monkeypatch, caplog):
    monkeypatch.setattr(config, "VERBOSE_FITNESS_LOGS", False, raising=False)
    fitness.reset_fitness_run()
    caplog.set_level(logging.WARNING)
    fitness._log_penalty_metrics({"vol": 0}, 1, "reason", 0)
    fitness._log_penalty_metrics({"vol": 0}, 1, "reason", 0)
    assert caplog.text.count("fitness_penalty") == 1


def test_penalty_logging_verbose(monkeypatch, caplog):
    monkeypatch.setattr(config, "VERBOSE_FITNESS_LOGS", True, raising=False)
    fitness.reset_fitness_run()
    caplog.set_level(logging.WARNING)
    fitness._log_penalty_metrics({"vol": 0}, 1, "reason", 0)
    fitness._log_penalty_metrics({"vol": 0}, 1, "reason", 0)
    assert caplog.text.count("fitness_penalty") == 2


def _worker(queue):
    logger = logging.getLogger()
    logger.handlers = [logging.handlers.QueueHandler(queue)]
    fitness._log_penalty_metrics({"vol": 0}, 1, "reason", 0)


def test_penalty_logging_multiprocess(monkeypatch):
    monkeypatch.setattr(config, "VERBOSE_FITNESS_LOGS", False, raising=False)
    fitness.reset_fitness_run()
    queue: multiprocessing.Queue = multiprocessing.Queue()
    p1 = multiprocessing.Process(target=_worker, args=(queue,))
    p2 = multiprocessing.Process(target=_worker, args=(queue,))
    p1.start(); p2.start(); p1.join(); p2.join()
    records = []
    while not queue.empty():
        records.append(queue.get().getMessage())
    assert sum("fitness_penalty" in r for r in records) == 1
