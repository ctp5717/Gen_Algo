import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pygad  # noqa: E402
import numpy as np  # noqa: E402
import config  # noqa: E402
from utils import set_global_seed  # noqa: E402


def _run_ga():
    seed = config.RANDOM_SEED
    set_global_seed(seed)
    ga = pygad.GA(
        num_generations=2,
        num_parents_mating=2,
        sol_per_pop=4,
        num_genes=3,
        init_range_low=0.0,
        init_range_high=1.0,
        fitness_func=lambda ga_inst, sol, idx: np.sum(sol),
        random_seed=seed,
    )
    ga.run()
    return ga.best_solution()[1]


def test_deterministic_ga_produces_repeatable_fitness():
    """Running the GA twice with the same seed should yield the same fitness."""
    first = _run_ga()
    second = _run_ga()
    assert first == second
