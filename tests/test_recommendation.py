import copy
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import config
import recommendation
from schemas import (
    Fold,
    PerAssetRow,
    SchemaCsvError,
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


def test_asset_matrix_classification(monkeypatch):
    overrides = copy.deepcopy(config.RECOMMENDATION["ASSET_CLASS_THRESHOLDS"])
    overrides["gamble"]["consistency"] = 80
    monkeypatch.setitem(config.RECOMMENDATION, "ASSET_CLASS_THRESHOLDS", overrides)
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
        + [PerAssetRow(fold=3, ticker="STAL", score=np.nan, trades=5, included=True)]
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
            PerAssetRow(
                fold=i,
                ticker="GAMB",
                score=[1.2, 1.1, -0.5, -0.6, 1.3][i],
                trades=5,
                included=True,
            )
            for i in range(5)
        ]
        + [
            PerAssetRow(fold=i, ticker="INSU", score=1.0, trades=5, included=True)
            for i in range(2)
        ]
        + [
            PerAssetRow(fold=i, ticker="LOWT", score=1.0, trades=2, included=True)
            for i in range(3)
        ]
        + [
            PerAssetRow(fold=i, ticker="EXCL", score=1.0, trades=5, included=False)
            for i in range(3)
        ]
    )
    matrix = recommendation._build_asset_matrix(rows)
    assert matrix["STAR"]["class"] == "Stars"
    assert matrix["STAL"]["class"] == "Stalwarts" and matrix["STAL"]["samples"] == 3
    assert matrix["DRAG"]["class"] == "Drags"
    assert matrix["BORD"]["class"] == "Borderline"
    assert matrix["GAMB"]["class"] == "Gambles"
    assert matrix["INSU"]["class"] == "Insufficient Data"
    assert (
        matrix["LOWT"]["class"] == "Insufficient Data"
        and matrix["LOWT"]["samples"] == 0
    )
    assert "EXCL" not in matrix


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
    vals_c = [2, 5, 8, 11, 14]
    vals_d = [1, 2, 3, 4, 0]
    folds = [
        Fold(
            fold_id=i,
            validation_fitness=1.0,
            params={
                "a": vals_a[i],
                "b": vals_b[i],
                "c": vals_c[i],
                "d": vals_d[i],
                "const": 1.0,
            },
            champion_status=None,
        )
        for i in range(5)
    ]
    cov, unstable, watch = recommendation._param_stability(folds)
    assert cov["a"] == pytest.approx(0.71, abs=0.01)
    assert cov["c"] == pytest.approx(0.53, abs=0.01)
    assert cov["d"] == pytest.approx(0.71, abs=0.01)
    assert "const" not in cov
    assert unstable == ["a", "d", "c"]
    assert watch == ["b"]


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
    assert "x" not in cov
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


def test_param_stability_rounding_boundary():
    folds = [
        Fold(
            fold_id=0,
            validation_fitness=1.0,
            params={"e": 2.99},
            champion_status=None,
        ),
        Fold(
            fold_id=1,
            validation_fitness=1.0,
            params={"e": 1.01},
            champion_status=None,
        ),
    ]
    cov, unstable, watch = recommendation._param_stability(folds)
    assert cov["e"] == 0.5
    assert unstable == []
    assert watch == ["e"]


def test_schema_validation_failure_summary(tmp_path):
    wf = tmp_path / "walk_forward"
    wf.mkdir()
    bad_summary = {
        "metadata": {"schema_version": "1.0", "num_folds": 1, "asset_universe": []},
        "folds": [{"fold_id": 1}],
    }
    (wf / "walk_forward_summary.json").write_text(json.dumps(bad_summary))
    (wf / "walk_forward_per_asset.csv").write_text(
        "fold,ticker,score,trades,included\n1,BTC,1,1,True\n"
    )
    (tmp_path / "run_metadata.json").write_text("{}")
    out = recommendation.generate_recommendation({"run_dir": tmp_path})
    assert out["error"] == "schema_validation_failed"
    meta = json.loads((tmp_path / "run_metadata.json").read_text())
    assert meta["recommendation"]["error"] == "schema_validation_failed"
    md = (tmp_path / "strategy_recommendation.md").read_text()
    assert "schema validation failed" in md.lower()
    assert "summary" in md
    assert "validation_fitness" in md
    assert "strategy_recommendation.md" in meta["artifacts"]


def test_schema_validation_failure_per_asset(tmp_path):
    wf = tmp_path / "walk_forward"
    wf.mkdir()
    summary = {
        "metadata": {"schema_version": "1.0", "num_folds": 1, "asset_universe": []},
        "folds": [
            {
                "fold_id": 0,
                "validation_fitness": 1.0,
                "params": {},
            }
        ],
    }
    (wf / "walk_forward_summary.json").write_text(json.dumps(summary))
    (wf / "walk_forward_per_asset.csv").write_text(
        "fold,ticker,score,trades,included,foo\n0,AAA,1.0,notint,True,bar\n"
    )
    (tmp_path / "run_metadata.json").write_text("{}")
    out = recommendation.generate_recommendation({"run_dir": tmp_path})
    assert out["error"] == "schema_validation_failed"
    assert "foo" in out.get("diagnostics", "")
    md = (tmp_path / "strategy_recommendation.md").read_text()
    assert "Diagnostics" in md and "foo" in md


def test_load_wf_per_asset_bad_column_type(tmp_path):
    wf = tmp_path / "walk_forward"
    wf.mkdir()
    bad_csv = "fold,ticker,score,trades,included,extra\n0,AAA,1.0,notint,True,x\n"
    (wf / "walk_forward_per_asset.csv").write_text(bad_csv)
    with pytest.raises(SchemaCsvError) as e:
        load_wf_per_asset(wf / "walk_forward_per_asset.csv")
    assert "invalid trades" in str(e.value)
    assert e.value.unknown_columns == ["extra"]


def test_load_wf_per_asset_header_case_and_whitespace(tmp_path):
    wf = tmp_path / "walk_forward"
    wf.mkdir()
    csv = "Fold ,Ticker ,Score ,Trades ,Included \n0,AAA,1.0,5,True\n"
    (wf / "walk_forward_per_asset.csv").write_text(csv)
    obj = load_wf_per_asset(wf / "walk_forward_per_asset.csv")
    assert obj.rows[0].ticker == "AAA"


def test_generate_recommendation_determinism(tmp_path):
    _write_sample_files(tmp_path)
    out1 = recommendation.generate_recommendation({"run_dir": tmp_path})
    md1 = (tmp_path / "strategy_recommendation.md").read_text()
    meta1 = json.loads((tmp_path / "run_metadata.json").read_text())
    assert "strategy_recommendation.md" in meta1["artifacts"]
    tmp2 = tmp_path / "b"
    tmp2.mkdir()
    _write_sample_files(tmp2)
    out2 = recommendation.generate_recommendation({"run_dir": tmp2})
    md2 = (tmp2 / "strategy_recommendation.md").read_text()
    assert out1 == out2 and md1 == md2
    assert re.search(r"Folds: median", md1)


def test_asset_summary_and_legend_full_text(tmp_path):
    _write_sample_files(tmp_path)
    recommendation.generate_recommendation({"run_dir": tmp_path})
    md_path = tmp_path / "strategy_recommendation.md"
    lines = md_path.read_text(encoding="utf-8").splitlines()

    asset_idx = lines.index("## Asset Summary") + 1
    asset_line = lines[asset_idx]
    assert "..." not in asset_line
    expected_asset = (
        "Stars: BBB; Stalwarts: AAA; All assets have \u22653 qualifying fold(s)."
    )
    assert asset_line == expected_asset
    assert "\n" not in asset_line

    legend_line = next(line for line in lines if line.startswith("Legend:"))
    assert "..." not in legend_line
    th = config.RECOMMENDATION["ASSET_CLASS_THRESHOLDS"]
    expected_legend = (
        "Legend: "
        f"Stars \u2265{th['star']['performance']} perf & "
        f"\u2265{th['star']['consistency']}% consistency; "
        f"Stalwarts {th['stalwart']['performance_low']}\u2013"
        f"{th['stalwart']['performance_high']} perf & \u2265"
        f"{th['stalwart']['consistency']}% consistency; "
        f"Gambles \u2265{th['gamble']['performance']} perf & "
        f"<{th['gamble']['consistency']}% consistency; "
        f"Drags <{th['drag']['performance']} perf & <"
        f"{th['drag']['consistency']}% consistency"
    )
    assert legend_line == expected_legend
    assert "\n" not in legend_line


def test_markdown_artifact_deduped(tmp_path):
    _write_sample_files(tmp_path)
    recommendation.generate_recommendation({"run_dir": tmp_path})
    recommendation.generate_recommendation({"run_dir": tmp_path})
    meta = json.loads((tmp_path / "run_metadata.json").read_text())
    assert meta["artifacts"].count("strategy_recommendation.md") == 1


def test_narrative_borderline_and_samples():
    conf = {
        "category": "Low",
        "score": 0,
        "factors": {
            "median_fitness": 0,
            "positive_fold_pct": 0,
            "worst_fold_fitness": 0,
            "downside_deviation": 0,
        },
    }
    assets = {
        "AAA": {
            "class": "Borderline",
            "performance": 0.0,
            "consistency": 0.0,
            "samples": 3,
        },
        "BBB": {
            "class": "Insufficient Data",
            "performance": 0.0,
            "consistency": 0.0,
            "samples": 1,
        },
    }
    out = recommendation._build_narrative(conf, assets, [], [])
    assert "Borderline: AAA" in out["assets"]
    assert "Insufficient Data: BBB" in out["assets"]
    assets2 = {
        "CCC": {
            "class": "Stars",
            "performance": 1.0,
            "consistency": 100.0,
            "samples": 3,
        }
    }
    out2 = recommendation._build_narrative(conf, assets2, [], [])
    assert "All assets have" in out2["assets"]


def test_narrative_no_assets_no_sample_line():
    conf = {
        "category": "Low",
        "score": 0,
        "factors": {
            "median_fitness": 0,
            "positive_fold_pct": 0,
            "worst_fold_fitness": 0,
            "downside_deviation": 0,
        },
    }
    out = recommendation._build_narrative(conf, {}, [], [])
    assert "All assets have" not in out["assets"]


def test_infinite_cov_rendered(tmp_path):
    payload = {
        "confidence": {
            "category": "Low",
            "score": 0,
            "scores": {"median": 0, "consistency": 0, "tail": 0, "downside": 0},
            "factors": {
                "median_fitness": 0,
                "positive_fold_pct": 0,
                "worst_fold_fitness": 0,
                "downside_deviation": 0,
            },
        },
        "assets": {},
        "param_stability": {
            "cov_by_gene": {"x": float("inf")},
            "unstable_genes": ["x"],
            "watchlist_genes": [],
        },
        "narrative": {"overall": "", "assets": "", "params": ""},
        "schema_version": "1.0",
    }
    md_path = tmp_path / "md.md"
    recommendation._write_markdown(md_path, payload)
    assert "∞" in md_path.read_text()


def test_asset_table_handles_nan_and_none(tmp_path):
    payload = {
        "confidence": {
            "category": "Low",
            "score": 0,
            "scores": {"median": 0, "consistency": 0, "tail": 0, "downside": 0},
            "factors": {
                "median_fitness": 0,
                "positive_fold_pct": 0,
                "worst_fold_fitness": 0,
                "downside_deviation": 0,
            },
        },
        "assets": {
            "AAA": {
                "performance": None,
                "consistency": float("nan"),
                "class": "Stars",
                "samples": 1,
            }
        },
        "param_stability": {"cov_by_gene": {}, "unstable_genes": []},
        "narrative": {"overall": "", "assets": "", "params": ""},
        "schema_version": "1.0",
    }
    md_path = tmp_path / "md.md"
    recommendation._write_markdown(md_path, payload)
    text = md_path.read_text()
    assert "| AAA | — | — | Stars | 1 |" in text
