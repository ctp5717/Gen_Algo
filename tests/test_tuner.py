import sys
import types
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

sys.modules.setdefault("pandas_ta", types.ModuleType("pandas_ta"))
sys.modules.setdefault("vectorbt", types.ModuleType("vectorbt"))

import tuner  # noqa: E402


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


def test_lambda_grid_calls_selector(monkeypatch):
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

    monkeypatch.setitem(tuner.config.MULTI_ASSET, "enabled", True)
    monkeypatch.setitem(tuner.config.MULTI_ASSET, "lambda_grid", [0.1, 0.2])
    monkeypatch.setitem(tuner.config.MULTI_ASSET, "lambda_seeds", [1, 2])
    monkeypatch.setitem(tuner.config.MULTI_ASSET, "lambda_coverage_min", 0.1)
    monkeypatch.setattr(tuner.config, "HYPERPARAMETER_SEARCH_SPACE", [], raising=False)

    calls = []
    seed_calls = []
    selector_kwargs = {}

    def fake_selector(rows, **kwargs):
        calls.append(rows)
        selector_kwargs.update(kwargs)
        table = pd.DataFrame(
            {
                "lambda": [0.1, 0.2],
                "mu_val_mean": [0, 0],
                "mu_val_std": [0, 0],
                "sigma_val_mean": [0, 0],
                "sigma_val_std": [0, 0],
                "mu_train_mean": [0, 0],
                "F_train_mean": [0, 0],
                "coverage_mean": [0, 0],
                "gap": [0, 0],
                "elbow_dist": [0, 0],
            }
        )
        return 0.2, table, table

    class DummyEval:
        def __init__(self, *a, **k):
            pass

        def __call__(self, ga, sol, idx):
            return 0

    class DummyGA:
        def __init__(self, *a, **k):
            seed_calls.append(k.get("random_seed"))

        def run(self):
            pass

        def best_solution(self, **kwargs):
            return [0], 0, None

    monkeypatch.setattr(
        tuner.lambda_selector, "select_lambda_with_elbow", fake_selector
    )
    monkeypatch.setattr(tuner.fitness, "MultiAssetFitnessEvaluator", DummyEval)
    monkeypatch.setattr(tuner.pygad, "GA", DummyGA)
    monkeypatch.setattr(
        tuner.fitness,
        "get_fitness_evaluator",
        lambda *a, **k: types.SimpleNamespace(__call__=lambda *a, **k: 0),
    )
    monkeypatch.setattr(tuner, "_evaluate_on_validation", lambda sol, gm, val: 0)

    tuner.find_best_hyperparameters(df, gene_space, gene_map, gene_types, df)
    assert tuner.config.MULTI_ASSET["lambda_dispersion"] == 0.2
    assert calls and len(calls[0]) == 4
    assert seed_calls == [1, 2, 1, 2]
    assert (
        selector_kwargs["shortlist_size"]
        == tuner.config.MULTI_ASSET["lambda_shortlist_size"]
    )
    assert (
        selector_kwargs["sigma_pct_threshold"]
        == tuner.config.MULTI_ASSET["lambda_sigma_pctl"]
    )
    assert selector_kwargs["coverage_min"] == 0.1
