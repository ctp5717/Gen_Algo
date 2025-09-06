import json
import sys
import types
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

sys.modules.setdefault("pandas_ta", types.ModuleType("pandas_ta"))
sys.modules.setdefault("vectorbt", types.ModuleType("vectorbt"))

import tuner  # noqa: E402


def test_lambda_sweep_artifacts(tmp_path, monkeypatch):
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

    monkeypatch.chdir(tmp_path)

    gene_space = [{"low": 0, "high": 1}]
    gene_map = {0: {"name": "x", "path": [], "type": float}}
    gene_types = [float]

    monkeypatch.setitem(tuner.config.MULTI_ASSET, "enabled", True)
    monkeypatch.setitem(tuner.config.MULTI_ASSET, "lambda_grid", [0.1])
    monkeypatch.setitem(tuner.config.MULTI_ASSET, "lambda_seeds", [1])
    monkeypatch.setitem(
        tuner.config.MULTI_ASSET, "lambda_probe_round2_on_duplicate", False
    )
    monkeypatch.setattr(tuner.config, "HYPERPARAMETER_SEARCH_SPACE", [], raising=False)

    class DummyGA:
        def __init__(self, *a, **k):
            pass

        def run(self):
            pass

        def best_solution(self, **kwargs):
            return [0], 0, None

    class DummyEval:
        def __init__(self, *a, **k):
            pass

        def __call__(self, ga, sol, idx):
            return 0

    monkeypatch.setattr(tuner.pygad, "GA", DummyGA)
    monkeypatch.setattr(tuner.fitness, "MultiAssetFitnessEvaluator", DummyEval)
    monkeypatch.setattr(
        tuner.fitness,
        "get_fitness_evaluator",
        lambda *a, **k: types.SimpleNamespace(__call__=lambda *a, **k: 0),
    )
    monkeypatch.setattr(tuner, "_evaluate_on_validation", lambda sol, gm, val: 0)

    tuner.find_best_hyperparameters(df, gene_space, gene_map, gene_types, df)

    json_path = tmp_path / "lambda_sweep.json"
    csv_path = tmp_path / "lambda_sweep.csv"
    png_path = tmp_path / "lambda_frontier.png"

    assert json_path.exists()
    assert csv_path.exists()
    assert png_path.exists()

    data = json.loads(json_path.read_text())
    assert {
        "rows_all",
        "rows_final",
        "rows_agg",
        "shortlist",
        "nan_summary",
        "elbow_AB",
        "chosen",
        "probe",
    } <= data.keys()
    assert data["rows_all"][0]["solution_hash"]
    assert data["nan_summary"] == []
