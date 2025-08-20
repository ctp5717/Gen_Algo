import sys
import types
from pathlib import Path

# Ensure repository root on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Stub heavy optional dependencies
sys.modules.setdefault('pandas_ta', types.ModuleType('pandas_ta'))
sys.modules.setdefault('vectorbt', types.ModuleType('vectorbt'))

import pygad  # noqa: E402
import tuner  # noqa: E402


class DummyEvaluator:
    def __init__(self):
        self.last_details = {}

    def __call__(self, ga_instance, solution, idx):
        return 1.0


def test_on_generation_callback_picklable():
    evaluator = DummyEvaluator()
    cb = tuner._make_on_generation(evaluator, evaluator.__call__)

    ga = pygad.GA(
        num_generations=1,
        num_parents_mating=1,
        sol_per_pop=2,
        num_genes=1,
        gene_space=[0, 1],
        gene_type=int,
        mutation_num_genes=1,
        fitness_func=evaluator.__call__,
        parallel_processing=["process", 1],
        on_generation=cb,
    )

    ga.run()
