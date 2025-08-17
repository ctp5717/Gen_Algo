import sys
import types
from pathlib import Path
import pandas as pd
import pickle

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


def test_find_best_hyperparameters_pickleable_callback(monkeypatch):
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
    }]

    monkeypatch.setattr(tuner.config, 'HYPERPARAMETER_SEARCH_SPACE', search, raising=False)
    monkeypatch.setattr(tuner.config, 'GENERATIONS_PER_TUNE', 1, raising=False)

    class DummyGA:
        def __init__(self, *a, **k):
            pickle.dumps(k['on_generation'])

        def run(self):
            pass

        def best_solution(self, **kwargs):
            return [0], 0, None

    monkeypatch.setattr(tuner.pygad, 'GA', DummyGA)
    monkeypatch.setattr(tuner, '_evaluate_on_validation', lambda sol, gm: 0)

    tuner.find_best_hyperparameters(df, gene_space, gene_map, gene_types)


def test_validation_elevates_mc_runs(monkeypatch):
    # Ensure heavy deps appear to be present
    monkeypatch.setattr(pd.DataFrame, 'ta', property(lambda self: None), raising=False)
    monkeypatch.setattr(tuner, 'vbt', types.SimpleNamespace(Portfolio=object()), raising=False)

    idx = pd.date_range('2020', periods=2, freq='D')
    sample = pd.DataFrame({'Open': [1, 1], 'Close': [1, 1]}, index=idx)

    def fake_load_group(group, start, end, tf):
        return {name: sample for name, _ in group}

    monkeypatch.setattr(tuner.data_loader, 'load_group_data', fake_load_group)

    called = []

    def fake_eval_once(self, solution, seed, assets):
        called.append(seed)
        return tuner.multi_asset_fitness.EvalResult(
            0.0,
            {},
            pd.Series([0.0, 0.0], index=idx),
            pd.Series([0, 0], index=idx),
            {'accepted': 10},
            pd.Series([10], index=['a']),
            0.0,
        )

    monkeypatch.setattr(
        tuner.multi_asset_fitness.MultiAssetFitnessEvaluator,
        '_evaluate_once',
        fake_eval_once,
    )
    monkeypatch.setattr(
        tuner.multi_asset_fitness.MultiAssetFitnessEvaluator,
        '_calc_stats',
        lambda self, r: (0.5, 1.0, 1.0),
    )

    orig_policy = tuner.config.SCANNER['tie_break_policy']
    orig_runs = tuner.config.SCANNER['monte_carlo_runs']
    tuner.config.SCANNER['tie_break_policy'] = 'random'
    tuner.config.SCANNER['monte_carlo_runs'] = 1

    score = tuner._evaluate_on_validation([], {})
    assert len(called) == 3
    assert score == 0.5

    tuner.config.SCANNER['tie_break_policy'] = orig_policy
    tuner.config.SCANNER['monte_carlo_runs'] = orig_runs
