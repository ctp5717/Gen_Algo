import pickle

import ga_utils
import config


def test_restart_policy_resets_population(monkeypatch):
    monkeypatch.setattr(config, "GA_STAGNATION_THRESHOLD", 2, raising=False)
    monkeypatch.setattr(config, "GA_RESTART_POLICY", "restart", raising=False)
    cb = ga_utils.make_stagnation_callback()

    class DummyGA:
        def __init__(self):
            self.calls = 0
            self.gene_space = [{"low": 0, "high": 1}]
            self.last_generation_fitness = [-999]

        def best_solution(self, pop_fitness=None):
            return None, -999, None

        def initialize_population(self):
            self.calls += 1

    ga = DummyGA()
    cb(ga)
    cb(ga)
    assert ga.calls == 1


def test_expand_policy_expands_ranges(monkeypatch):
    monkeypatch.setattr(config, "GA_STAGNATION_THRESHOLD", 1, raising=False)
    monkeypatch.setattr(config, "GA_RESTART_POLICY", "expand", raising=False)
    monkeypatch.setattr(config, "GA_GENE_RANGE_EXPANSION", 0.5, raising=False)
    cb = ga_utils.make_stagnation_callback()

    class DummyGA:
        def __init__(self):
            self.gene_space = [{"low": 0, "high": 10}]
            self.last_generation_fitness = [-999]
            self.init_called = 0

        def best_solution(self, pop_fitness=None):
            return None, -999, None

        def initialize_population(self):
            self.init_called += 1

    ga = DummyGA()
    cb(ga)
    assert ga.init_called == 1
    assert ga.gene_space[0]["low"] < 0
    assert ga.gene_space[0]["high"] > 10


def test_stagnation_callback_picklable():
    """Regression test ensuring the callback can be pickled for multiprocessing."""
    cb = ga_utils.make_stagnation_callback()
    pickle.dumps(cb)
