import sys
import types
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

sys.modules.setdefault('pandas_ta', types.ModuleType('pandas_ta'))
sys.modules.setdefault('vectorbt', types.ModuleType('vectorbt'))

import tuner  # noqa: E402


def test_find_best_hyperparameters_selects_best(monkeypatch):
    df = pd.DataFrame(
        {
            'Open': [1, 1],
            'High': [1, 1],
            'Low': [1, 1],
            'Close': [1, 1],
            'Volume': [1, 1],
        },
        index=pd.date_range('2020-01-01', periods=2),
    )

    gene_space = [{'low': 0, 'high': 1}]
    gene_map = {0: {'name': 'x', 'path': [], 'type': float}}
    gene_types = [float]

    search = [
        {'sol_per_pop': 1, 'num_parents_mating': 1, 'mutation_num_genes': 1},
        {'sol_per_pop': 2, 'num_parents_mating': 1, 'mutation_num_genes': 1},
    ]
    monkeypatch.setattr(tuner.config, 'HYPERPARAMETER_SEARCH_SPACE', search, raising=False)
    monkeypatch.setattr(tuner.config, 'GENERATIONS_PER_TUNE', 1, raising=False)

    scores = [0.1, 0.2]

    class DummyGA:
        def __init__(self, *a, **k):
            self.score = scores.pop(0)

        def run(self):
            pass

        def best_solution(self, **kwargs):
            return [self.score], self.score, None

    monkeypatch.setattr(tuner.pygad, 'GA', DummyGA)
    monkeypatch.setattr(tuner, '_evaluate_on_validation', lambda sol, gm: sol[0])

    best = tuner.find_best_hyperparameters(df, gene_space, gene_map, gene_types)
    assert best == search[1]


def test_find_best_hyperparameters_preserves_gene_types(monkeypatch):
    df = pd.DataFrame({
        'Open': [1],
        'High': [1],
        'Low': [1],
        'Close': [1],
        'Volume': [1],
    }, index=pd.date_range('2020-01-01', periods=1))

    gene_space = [{'low': 0, 'high': 1}]
    gene_map = {0: {'name': 'x', 'path': [], 'type': float}}
    gene_types = [float]

    search = [{
        'sol_per_pop': 1,
        'num_parents_mating': 1,
        'mutation_num_genes': 1,
    }, {
        'sol_per_pop': 1,
        'num_parents_mating': 1,
        'mutation_num_genes': 1,
    }]

    monkeypatch.setattr(tuner.config, 'HYPERPARAMETER_SEARCH_SPACE', search, raising=False)
    monkeypatch.setattr(tuner.config, 'GENERATIONS_PER_TUNE', 1, raising=False)

    class DummyGA:
        def __init__(self, *a, **k):
            # Simulate PyGAD mutating the list in-place
            gt = k.get('gene_type')
            if isinstance(gt, list) and gt:
                gt[0] = [gt[0], None]

        def run(self):
            pass

        def best_solution(self, **kwargs):
            return [0], 0, None

    monkeypatch.setattr(tuner.pygad, 'GA', DummyGA)
    monkeypatch.setattr(tuner, '_evaluate_on_validation', lambda sol, gm: 0)

    original = list(gene_types)
    tuner.find_best_hyperparameters(df, gene_space, gene_map, gene_types)
    assert gene_types == original


def test_find_best_hyperparameters_updates_lambda(monkeypatch):
    df = pd.DataFrame(
        {
            'Open': [1, 1],
            'High': [1, 1],
            'Low': [1, 1],
            'Close': [1, 1],
            'Volume': [1, 1],
        },
        index=pd.date_range('2020-01-01', periods=2),
    )

    gene_space = [{'low': 0, 'high': 1}]
    gene_map = {0: {'name': 'x', 'path': [], 'type': float}}
    gene_types = [float]

    search = [
        {'sol_per_pop': 1, 'num_parents_mating': 1, 'mutation_num_genes': 1, 'lambda_dispersion': 0.0},
        {'sol_per_pop': 1, 'num_parents_mating': 1, 'mutation_num_genes': 1, 'lambda_dispersion': 0.5},
    ]
    monkeypatch.setattr(tuner.config, 'HYPERPARAMETER_SEARCH_SPACE', search, raising=False)
    monkeypatch.setattr(tuner.config, 'GENERATIONS_PER_TUNE', 1, raising=False)

    scores = [0.1, 0.2]

    class DummyGA:
        def __init__(self, *a, **k):
            self.score = scores.pop(0)

        def run(self):
            pass

        def best_solution(self, **kwargs):
            return [self.score], self.score, None

    monkeypatch.setattr(tuner.pygad, 'GA', DummyGA)
    monkeypatch.setattr(tuner, '_evaluate_on_validation', lambda sol, gm: sol[0])

    tuner.config.MULTI_ASSET['lambda_dispersion'] = 0.25
    tuner.find_best_hyperparameters(df, gene_space, gene_map, gene_types)
    assert tuner.config.MULTI_ASSET['lambda_dispersion'] == 0.5
