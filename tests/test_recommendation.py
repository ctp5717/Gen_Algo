import copy
import json
from pathlib import Path

import pandas as pd
import pytest

import config
import recommendation
from schemas import (
    Fold,
    PerAssetRow,
    WalkForwardPerAssetV1,
    WalkForwardSummaryV1,
    load_wf_per_asset,
    load_wf_summary,
)


def _write_sample_files(tmp_path: Path) -> None:
    wf = tmp_path / "walk_forward"
    wf.mkdir()
    summary = {
        "metadata": {
            "schema_version": "1.0",
            "num_folds": 3,
            "asset_universe": ["AAA"],
        },
        "folds": [
            {
                "fold_id": 0,
                "validation_fitness": 1.2,
                "params": {"x": 1},
                "champion_status": "Elite",
            },
            {
                "fold_id": 1,
                "validation_fitness": 0.5,
                "params": {"x": 1.1},
                "champion_status": "Viable",
            },
            {
                "fold_id": 2,
                "validation_fitness": -0.3,
                "params": {"x": 0.9},
                "champion_status": "Discarded",
            },
        ],
    }
    (wf / "walk_forward_summary.json").write_text(json.dumps(summary))
    per_asset = pd.DataFrame(
        {
            "fold": [0, 1, 2, 0, 1, 2],
            "ticker": ["AAA", "AAA", "AAA", "BBB", "BBB", "BBB"],
            "score": [1.2, 0.5, -0.3, 1.2, 1.3, 1.1],
            "trades": [5, 5, 5, 5, 5, 5],
            "included": [True] * 6,
        }
    )
    per_asset.to_csv(wf / "walk_forward_per_asset.csv", index=False)
    (tmp_path / "run_metadata.json").write_text("{}")


def test_load_wf_summary_and_per_asset(tmp_path):
    _write_sample_files(tmp_path)
    summary = load_wf_summary(tmp_path / "walk_forward" / "walk_forward_summary.json")
    assert isinstance(summary, WalkForwardSummaryV1)
    per_asset = load_wf_per_asset(
        tmp_path / "walk_forward" / "walk_forward_per_asset.csv"
    )
    assert isinstance(per_asset, WalkForwardPerAssetV1)


def test_load_wf_summary_missing_key(tmp_path):
    bad = {"metadata": {"schema_version": "1.0", "num_folds": 0, "asset_universe": []}}
    wf = tmp_path / "walk_forward"
    wf.mkdir()
    (wf / "walk_forward_summary.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        load_wf_summary(wf / "walk_forward_summary.json")


def test_use_return_as_fitness(monkeypatch, tmp_path):
    wf = tmp_path / "walk_forward"
    wf.mkdir()
    summary = {
        "metadata": {"schema_version": "1.0", "num_folds": 1, "asset_universe": []},
        "folds": [
            {"fold_id": 0, "Total Return [%]": 2.0, "Params": {}},
        ],
    }
    (wf / "walk_forward_summary.json").write_text(json.dumps(summary))
    monkeypatch.setitem(config.RECOMMENDATION, "USE_RETURN_AS_FITNESS", True)
    obj = load_wf_summary(wf / "walk_forward_summary.json")
    assert obj.folds[0].validation_fitness == 2.0


def test_confidence_scoring_categories():
    high = recommendation._compute_confidence([1.5, 1.6, 1.4])
    assert high["category"] == "High"
    low = recommendation._compute_confidence([-1.0, -0.5, -0.2])
    assert low["category"] == "Low"
    medium = recommendation._compute_confidence([0.2, 0.0, 0.1])
    assert medium["category"] == "Medium"


def test_asset_matrix_classification():
    rows = WalkForwardPerAssetV1(
        rows=[
            PerAssetRow(fold=i, ticker="STAR", score=1.2, trades=5, included=True)
            for i in range(3)
        ]
        + [PerAssetRow(fold=3, ticker="STAR", score=None, trades=5, included=True)]
        + [
            PerAssetRow(fold=i, ticker="STAL", score=0.5, trades=5, included=True)
            for i in range(3)
        ]
        + [
            PerAssetRow(
                fold=i, ticker="DRAG", score=-0.5 if i else 0.2, trades=5, included=True
            )
            for i in range(3)
        ]
        + [
            PerAssetRow(
                fold=i,
                ticker="BORD",
                score=[0.0, -0.1, 1.5][i],
                trades=5,
                included=True,
            )
            for i in range(3)
        ]
        + [
            PerAssetRow(fold=i, ticker="INSU", score=1.0, trades=5, included=True)
            for i in range(2)
        ]
    )
    matrix = recommendation._build_asset_matrix(rows)
    assert matrix["STAR"]["class"] == "Stars"
    assert matrix["STAL"]["class"] == "Stalwarts"
    assert matrix["DRAG"]["class"] == "Drags"
    assert matrix["BORD"]["class"] == "Borderline"
    assert matrix["INSU"]["class"] == "Insufficient Data"


def test_asset_class_threshold_override(monkeypatch):
    rows = WalkForwardPerAssetV1(
        rows=[
            PerAssetRow(fold=i, ticker="X", score=0.8, trades=5, included=True)
            for i in range(3)
        ]
    )
    overrides = copy.deepcopy(config.RECOMMENDATION["ASSET_CLASS_THRESHOLDS"])
    overrides["star"]["performance"] = 0.5
    monkeypatch.setitem(config.RECOMMENDATION, "ASSET_CLASS_THRESHOLDS", overrides)
    matrix = recommendation._build_asset_matrix(rows)
    assert matrix["X"]["class"] == "Stars"


def test_param_stability_detection():
    vals_a = [1, 2, 3, 4, 0]
    vals_b = [1.0, 1.5, 0.5, 1.4, 0.6]
    folds = [
        Fold(
            fold_id=i,
            validation_fitness=1.0,
            params={"a": vals_a[i], "b": vals_b[i]},
            champion_status=None,
        )
        for i in range(5)
    ]
    cov, unstable, watch = recommendation._param_stability(folds)
    assert cov["a"] > cov["b"]
    assert "a" in unstable
    assert "b" in watch


def test_param_stability_champion_only():
    folds = [
        Fold(
            fold_id=i,
            validation_fitness=1.0,
            params={"x": v},
            champion_status=s,
        )
        for i, (v, s) in enumerate(
            [(1.0, "Elite"), (1.0, "Viable"), (100.0, "Discarded")]
        )
    ]
    cov, _, _ = recommendation._param_stability(folds)
    assert cov["x"] == 0.0
    cov_all, _, _ = recommendation._param_stability(
        [
            Fold(
                fold_id=f.fold_id,
                validation_fitness=f.validation_fitness,
                params=f.params,
                champion_status=None,
            )
            for f in folds
        ]
    )
    assert cov_all["x"] > 1


def test_schema_validation_failure(tmp_path):
    wf = tmp_path / "walk_forward"
    wf.mkdir()
    (wf / "walk_forward_summary.json").write_text("{}")
    (wf / "walk_forward_per_asset.csv").write_text(
        "fold,ticker,score,trades,included\n"
    )
    (tmp_path / "run_metadata.json").write_text("{}")
    out = recommendation.generate_recommendation({"run_dir": tmp_path})
    assert out["error"] == "schema_validation_failed"
    meta = json.loads((tmp_path / "run_metadata.json").read_text())
    assert meta["recommendation"]["error"] == "schema_validation_failed"


def test_load_wf_per_asset_bad_column_type(tmp_path):
    wf = tmp_path / "walk_forward"
    wf.mkdir()
    bad_csv = "fold,ticker,score,trades,included\n0,AAA,1.0,notint,True\n"
    (wf / "walk_forward_per_asset.csv").write_text(bad_csv)
    with pytest.raises(ValueError):
        load_wf_per_asset(wf / "walk_forward_per_asset.csv")


def test_generate_recommendation_determinism(tmp_path):
    _write_sample_files(tmp_path)
    out1 = recommendation.generate_recommendation({"run_dir": tmp_path})
    md1 = (tmp_path / "strategy_recommendation.md").read_text()
    meta1 = json.loads((tmp_path / "run_metadata.json").read_text())
    snap = Path(__file__).with_name("snapshots") / "strategy_recommendation.md"
    assert md1.strip() == snap.read_text().strip()
    # fresh directory with same inputs
    tmp2 = tmp_path / "b"
    tmp2.mkdir()
    _write_sample_files(tmp2)
    out2 = recommendation.generate_recommendation({"run_dir": tmp2})
    md2 = (tmp2 / "strategy_recommendation.md").read_text()
    meta2 = json.loads((tmp2 / "run_metadata.json").read_text())
    assert out1 == out2
    assert md1 == md2
    assert meta1["recommendation"] == meta2["recommendation"]
