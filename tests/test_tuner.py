import sys
import types
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

sys.modules.setdefault("pandas_ta", types.ModuleType("pandas_ta"))
sys.modules.setdefault("vectorbt", types.ModuleType("vectorbt"))

import tuner  # noqa: E402

tuner.config.initialize_config()


def test_find_best_hyperparameters_selects_best(monkeypatch):
    df = pd.DataFrame(
        {
            "Open": [1, 1],
            "High": [1, 1],
            "Low": [1, 1],
            "Close": [1, 1],
            "Volume": [1, 1],
        },
        index=pd.date_range("2020-01-01", periods=2),
    )

    gene_space = [{"low": 0, "high": 1}]
    gene_map = {0: {"name": "x", "path": [], "type": float}}
    gene_types = [float]

    search = [
        {"sol_per_pop": 1, "num_parents_mating": 1, "mutation_num_genes": 1},
        {"sol_per_pop": 2, "num_parents_mating": 1, "mutation_num_genes": 1},
    ]
    monkeypatch.setattr(
        tuner.config, "HYPERPARAMETER_SEARCH_SPACE", search, raising=False
    )
    monkeypatch.setattr(tuner.config, "GENERATIONS_PER_TUNE", 1, raising=False)
    monkeypatch.setitem(tuner.config.MULTI_ASSET, "lambda_grid", None)

    scores = [0.1, 0.2]

    class DummyGA:
        def __init__(self, *a, **k):
            self.score = scores.pop(0)

        def run(self):
            pass

        def best_solution(self, **kwargs):
            return [self.score], self.score, None

    monkeypatch.setattr(tuner.pygad, "GA", DummyGA)
    monkeypatch.setattr(tuner, "_evaluate_on_validation", lambda sol, gm, val: sol[0])

    best = tuner.find_best_hyperparameters(df, gene_space, gene_map, gene_types, df)
    assert best == search[1]


def test_find_best_hyperparameters_preserves_gene_types(monkeypatch):
    df = pd.DataFrame(
        {
            "Open": [1],
            "High": [1],
            "Low": [1],
            "Close": [1],
            "Volume": [1],
        },
        index=pd.date_range("2020-01-01", periods=1),
    )

    gene_space = [{"low": 0, "high": 1}]
    gene_map = {0: {"name": "x", "path": [], "type": float}}
    gene_types = [float]

    search = [
        {
            "sol_per_pop": 1,
            "num_parents_mating": 1,
            "mutation_num_genes": 1,
        },
        {
            "sol_per_pop": 1,
            "num_parents_mating": 1,
            "mutation_num_genes": 1,
        },
    ]

    monkeypatch.setattr(
        tuner.config, "HYPERPARAMETER_SEARCH_SPACE", search, raising=False
    )
    monkeypatch.setattr(tuner.config, "GENERATIONS_PER_TUNE", 1, raising=False)
    monkeypatch.setitem(tuner.config.MULTI_ASSET, "lambda_grid", None)

    class DummyGA:
        def __init__(self, *a, **k):
            # Simulate PyGAD mutating the list in-place
            gt = k.get("gene_type")
            if isinstance(gt, list) and gt:
                gt[0] = [gt[0], None]

        def run(self):
            pass

        def best_solution(self, **kwargs):
            return [0], 0, None

    monkeypatch.setattr(tuner.pygad, "GA", DummyGA)
    monkeypatch.setattr(tuner, "_evaluate_on_validation", lambda sol, gm, val: 0)

    original = list(gene_types)
    tuner.find_best_hyperparameters(df, gene_space, gene_map, gene_types, df)
    assert gene_types == original


def test_evaluate_on_validation_uses_multi_asset(monkeypatch):
    sentinel = object()

    class DummyEval:
        def __call__(self, ga, sol, idx):
            return sentinel

    monkeypatch.setitem(tuner.config.MULTI_ASSET, "enabled", True)
    monkeypatch.setattr(pd.DataFrame, "ta", None, raising=False)
    monkeypatch.setattr(tuner.vbt, "Portfolio", object, raising=False)
    df = pd.DataFrame(
        {
            "Open": [1],
            "High": [1],
            "Low": [1],
            "Close": [1],
            "Volume": [1],
        },
        index=pd.date_range("2020-01-01", periods=1),
    )
    monkeypatch.setattr(
        tuner.fitness, "MultiAssetFitnessEvaluator", lambda *a, **k: DummyEval()
    )
    res = tuner._evaluate_on_validation([0], {}, {"X": df})
    assert res is sentinel


def test_lambda_grid_rescoring(monkeypatch):
    df = pd.DataFrame(
        {
            "Open": [1],
            "High": [1],
            "Low": [1],
            "Close": [1],
            "Volume": [1],
        },
        index=pd.date_range("2020-01-01", periods=1),
    )

    gene_space = [{"low": 0, "high": 1}]
    gene_map = {0: {"name": "x", "path": [], "type": float}}
    gene_types = [float]

    monkeypatch.setitem(tuner.config.MULTI_ASSET, "lambda_grid", [0.1, 0.2, 0.3])
    monkeypatch.setitem(tuner.config.MULTI_ASSET, "lambda_top_k", 2)
    monkeypatch.setitem(tuner.config.MULTI_ASSET, "lambda_rescore_seeds", [1, 2])
    monkeypatch.setattr(tuner.config, "HYPERPARAMETER_SEARCH_SPACE", [], raising=False)

    scores = {
        (0.1, 42): 1.0,
        (0.2, 42): 0.8,
        (0.3, 42): 0.7,
        (0.1, 1): 1.0,
        (0.1, 2): 0.5,
        (0.2, 1): 2.0,
        (0.2, 2): 1.5,
    }

    current_lam = {"value": None}

    class DummyEval:
        def __init__(self, *a, **k):
            settings = a[3]
            current_lam["value"] = settings["lambda_dispersion"]

        def __call__(self, ga, sol, idx):
            return 0

    class DummyGA:
        def __init__(self, *a, **k):
            seed = k.get("random_seed")
            lam = current_lam["value"]
            self.score = scores[(lam, seed)]

        def run(self):
            pass

        def best_solution(self, **kwargs):
            return [self.score], self.score, None

    monkeypatch.setattr(
        tuner.fitness,
        "MultiAssetFitnessEvaluator",
        DummyEval,
    )
    monkeypatch.setattr(tuner.pygad, "GA", DummyGA)
    monkeypatch.setattr(
        tuner.fitness,
        "get_fitness_evaluator",
        lambda *a, **k: types.SimpleNamespace(__call__=lambda *a, **k: 0),
    )
    monkeypatch.setattr(tuner, "_evaluate_on_validation", lambda sol, gm, val: 0)

    tuner.find_best_hyperparameters(df, gene_space, gene_map, gene_types, df)
    assert tuner.config.MULTI_ASSET["lambda_dispersion"] == 0.2


def test_lambda_rescore_disables_mutation(monkeypatch):
    df = pd.DataFrame(
        {
            "Open": [1],
            "High": [1],
            "Low": [1],
            "Close": [1],
            "Volume": [1],
        },
        index=pd.date_range("2020-01-01", periods=1),
    )

    gene_space = [{"low": 0, "high": 1}]
    gene_map = {0: {"name": "x", "path": [], "type": float}}
    gene_types = [float]

    monkeypatch.setitem(tuner.config.MULTI_ASSET, "lambda_grid", [0.1])
    monkeypatch.setitem(tuner.config.MULTI_ASSET, "lambda_top_k", 1)
    monkeypatch.setitem(tuner.config.MULTI_ASSET, "lambda_rescore_seeds", [1])
    monkeypatch.setattr(tuner.config, "HYPERPARAMETER_SEARCH_SPACE", [], raising=False)

    class DummyEval:
        def __init__(self, *a, **k):
            pass

        def __call__(self, ga, sol, idx):
            return 0

    calls = []

    class DummyGA:
        def __init__(self, *a, **k):
            if k.get("mutation_num_genes") == 0:
                calls.append((k.get("mutation_type"), k.get("mutation_probability")))

        def run(self):
            pass

        def best_solution(self, **kwargs):
            return [0], 0, None

    monkeypatch.setattr(tuner.fitness, "MultiAssetFitnessEvaluator", DummyEval)
    monkeypatch.setattr(
        tuner.fitness,
        "get_fitness_evaluator",
        lambda *a, **k: types.SimpleNamespace(__call__=lambda *a, **k: 0),
    )
    monkeypatch.setattr(tuner.pygad, "GA", DummyGA)
    monkeypatch.setattr(tuner, "_evaluate_on_validation", lambda sol, gm, val: 0)

    tuner.find_best_hyperparameters(df, gene_space, gene_map, gene_types, df)
    assert calls == [(None, 0.0)]


def test_lambda_grid_generations(monkeypatch):
    df = pd.DataFrame(
        {
            "Open": [1],
            "High": [1],
            "Low": [1],
            "Close": [1],
            "Volume": [1],
        },
        index=pd.date_range("2020-01-01", periods=1),
    )

    gene_space = [{"low": 0, "high": 1}]
    gene_map = {0: {"name": "x", "path": [], "type": float}}
    gene_types = [float]

    monkeypatch.setitem(tuner.config.MULTI_ASSET, "lambda_grid", [0.1])
    monkeypatch.setitem(tuner.config.MULTI_ASSET, "lambda_top_k", 1)
    monkeypatch.setitem(tuner.config.MULTI_ASSET, "lambda_rescore_seeds", [1])
    monkeypatch.setitem(tuner.config.MULTI_ASSET, "lambda_grid_generations", 3)
    monkeypatch.setattr(tuner.config, "HYPERPARAMETER_SEARCH_SPACE", [], raising=False)

    class DummyEval:
        def __init__(self, *a, **k):
            pass

        def __call__(self, ga, sol, idx):
            return 0

    gens = []

    class DummyGA:
        def __init__(self, *a, **k):
            gens.append(k.get("num_generations"))

        def run(self):
            pass

        def best_solution(self, **kwargs):
            return [0], 0, None

    monkeypatch.setattr(tuner.fitness, "MultiAssetFitnessEvaluator", DummyEval)
    monkeypatch.setattr(
        tuner.fitness,
        "get_fitness_evaluator",
        lambda *a, **k: types.SimpleNamespace(__call__=lambda *a, **k: 0),
    )
    monkeypatch.setattr(tuner.pygad, "GA", DummyGA)
    monkeypatch.setattr(tuner, "_evaluate_on_validation", lambda sol, gm, val: 0)

    tuner.find_best_hyperparameters(df, gene_space, gene_map, gene_types, df)
    assert gens == [3, 3]
