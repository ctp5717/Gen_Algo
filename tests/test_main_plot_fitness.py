import sys
from pathlib import Path

import pandas as pd

# Ensure repository root is on the import path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import main  # noqa: E402


def test_plot_fitness_called_without_legend(monkeypatch):
    """main.main should call GA.plot_fitness without unsupported args"""
    # Simplify configuration to keep the run lightweight and deterministic.
    monkeypatch.setattr(main.config, "PORTFOLIO_OPTIMIZATION_ENABLED", False)
    monkeypatch.setattr(main.config, "AUTO_TUNE_ENABLED", False)
    monkeypatch.setattr(main.config, "WALK_FORWARD_SETTINGS", {"enabled": False})
    monkeypatch.setattr(main.config, "GA_NUM_GENERATIONS", 1)
    monkeypatch.setattr(main.config, "GA_POPULATION_SIZE", 1)
    monkeypatch.setattr(main.config, "GA_PARENTS_MATING", 1)
    monkeypatch.setattr(main.config, "GA_MUTATION_NUM_GENES", 1)

    # Stub data loader to avoid network calls.
    ohlc = pd.DataFrame({"Close": [1, 2, 3]})
    monkeypatch.setattr(main.data_loader, "get_data", lambda **_: ohlc)

    # Provide a minimal gene configuration.
    gene_space = [range(2)]
    gene_map = {0: {"name": "ema_period", "path": [], "type": int}}
    gene_types = [int]

    def fake_parser(_):
        return gene_space, gene_map, gene_types

    monkeypatch.setattr(main, "parse_genes_from_config", fake_parser)

    # Dummy fitness evaluator that always returns a constant fitness.
    class DummyEvaluator:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, ga_instance, solution, sol_idx):
            return 1.0

    monkeypatch.setattr(main.fitness, "FitnessEvaluator", DummyEvaluator)

    # Dummy GA implementation with a plot_fitness method that rejects kwargs.
    class DummyGA:
        def __init__(self, *args, **kwargs):
            self.num_generations = 1
            self.generations_completed = 1
            self.last_generation_fitness = [1.0]

        def run(self):
            pass

        def best_solution(self, pop_fitness=None):
            return [1], 1.0, None

        def plot_fitness(self, *args, **kwargs):
            if kwargs:
                raise TypeError("unexpected kwargs")

    monkeypatch.setattr(main.pygad, "GA", DummyGA)

    # Analysis step is not relevant for this test.
    monkeypatch.setattr(main.analysis, "run_champion_analysis", lambda *a, **k: None)

    # Should execute without raising a TypeError.
    main.main()
