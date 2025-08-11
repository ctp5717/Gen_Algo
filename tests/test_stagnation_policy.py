import pickle

import ga_utils
import config


def test_mutation_escalates_after_stagnation(monkeypatch):
    monkeypatch.setattr(config, "GA_STAGNATION_THRESHOLD", 2, raising=False)
    monkeypatch.setattr(config, "GA_MUTATION_ESCALATION_FACTOR", 2.0, raising=False)
    cb = ga_utils.make_stagnation_callback()

    class DummyGA:
        def __init__(self):
            self.gene_space = [{"low": 0, "high": 1}]
            self.last_generation_fitness = [-999]
            self.mutation_num_genes = 1
            self.num_genes = 10

        def best_solution(self, pop_fitness=None):
            return None, -999, None

    ga = DummyGA()
    cb(ga)
    assert ga.mutation_num_genes == 1
    cb(ga)
    assert ga.mutation_num_genes == 2


def test_expand_policy_expands_ranges(monkeypatch):
    monkeypatch.setattr(config, "GA_STAGNATION_THRESHOLD", 1, raising=False)
    monkeypatch.setattr(config, "GA_RESTART_POLICY", "expand", raising=False)
    monkeypatch.setattr(config, "GA_GENE_RANGE_EXPANSION", 0.5, raising=False)
    cb = ga_utils.make_stagnation_callback()

    class DummyGA:
        def __init__(self):
            self.gene_space = [{"low": 0, "high": 10}]
            self.last_generation_fitness = [-999]
            self.mutation_num_genes = 1
            self.num_genes = 10

        def best_solution(self, pop_fitness=None):
            return None, -999, None

    ga = DummyGA()
    cb(ga)
    assert ga.gene_space[0]["low"] < 0
    assert ga.gene_space[0]["high"] > 10
    assert ga.mutation_num_genes > 1


def test_stagnation_callback_picklable():
    """Regression test ensuring the callback can be pickled for multiprocessing."""
    cb = ga_utils.make_stagnation_callback()
    pickle.dumps(cb)


def test_callback_compatible_with_pygad():
    """The returned callback should be directly usable with ``pygad.GA``.

    A regression test for the issue where ``StagnationCallback`` instances were
    passed to ``on_generation`` causing ``AttributeError`` due to missing
    ``__code__``. ``make_stagnation_callback`` now returns a bound method which
    ``pygad`` accepts.
    """

    import pygad

    cb = ga_utils.make_stagnation_callback()

    ga = pygad.GA(
        num_generations=1,
        num_parents_mating=1,
        sol_per_pop=2,
        num_genes=1,
        gene_space=[{"low": 0, "high": 1}],
        gene_type=int,
        fitness_func=lambda ga, sol, idx: 0,
        on_generation=cb,
    )

    ga.run()
    assert ga.generations_completed == 1
