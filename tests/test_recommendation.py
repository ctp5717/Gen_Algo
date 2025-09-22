import copy
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

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
from strings import PARAM_STABILITY_IMPLICATION
from utils.format import fmt_num, fmt_pct


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
    per_asset, unknown = load_wf_per_asset(
        tmp_path / "walk_forward" / "walk_forward_per_asset.csv"
    )
    assert isinstance(per_asset, WalkForwardPerAssetV1)
    assert unknown == []


def test_load_wf_summary_missing_key(tmp_path):
    bad = {"metadata": {"schema_version": "1.0", "num_folds": 0, "asset_universe": []}}
    wf = tmp_path / "walk_forward"
    wf.mkdir()
    (wf / "walk_forward_summary.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        load_wf_summary(wf / "walk_forward_summary.json")


def test_load_wf_summary_with_string_param(tmp_path):
    wf = tmp_path / "walk_forward"
    wf.mkdir()
    summary = {
        "metadata": {"schema_version": "1.0", "num_folds": 1, "asset_universe": []},
        "folds": [
            {
                "fold_id": 0,
                "validation_fitness": 1.0,
                "params": {"sl_break_even_mode": "none", "tp_trailing_enabled": 1},
            }
        ],
    }
    (wf / "walk_forward_summary.json").write_text(json.dumps(summary))
    result = load_wf_summary(wf / "walk_forward_summary.json")
    assert result.folds[0].params["sl_break_even_mode"] == "none"


def test_load_wf_summary_bad_champion_status(tmp_path):
    wf = tmp_path / "walk_forward"
    wf.mkdir()
    summary = {
        "metadata": {"schema_version": "1.0", "num_folds": 1, "asset_universe": []},
        "folds": [
            {
                "fold_id": 0,
                "validation_fitness": 1.0,
                "params": {},
                "champion_status": "Unknown",
            }
        ],
    }
    (wf / "walk_forward_summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValidationError) as exc:
        load_wf_summary(wf / "walk_forward_summary.json")
    msg = str(exc.value)
    assert "Fold.champion_status must be one of: Elite | Viable | Discarded" in msg


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


def test_confidence_downside_deviation_ddof():
    single = recommendation._compute_confidence([0.5, -0.5, 1.0])
    assert single["factors"]["downside_deviation"] == 0.0
    two_negatives = recommendation._compute_confidence([1.0, -1.0, -2.0])
    assert two_negatives["factors"]["downside_deviation"] == pytest.approx(np.sqrt(0.5))


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


def test_asset_sort_key_ordering():
    assets = {
        # Stars: ties broken by consistency then ticker
        "AAA": {"class": "Stars", "performance": 2.0, "consistency": 95},
        "AAB": {"class": "Stars", "performance": 2.0, "consistency": 95},
        "AAC": {"class": "Stars", "performance": 1.5, "consistency": 97},
        "AAD": {"class": "Stars", "performance": 1.5, "consistency": 90},
        # Stalwarts: tie on performance -> consistency desc
        "BBA": {"class": "Stalwarts", "performance": 1.0, "consistency": 70},
        "BBB": {"class": "Stalwarts", "performance": 1.0, "consistency": 60},
        "BBC": {"class": "Stalwarts", "performance": 0.8, "consistency": 65},
        # Gambles and below follow class priority
        "CCA": {"class": "Gambles", "performance": 1.3, "consistency": 45},
        "CCB": {"class": "Gambles", "performance": 1.1, "consistency": 40},
        "DDA": {"class": "Borderline", "performance": 0.2, "consistency": 55},
        "EEA": {"class": "Drags", "performance": -0.5, "consistency": 30},
        "FFA": {"class": "Insufficient Data", "performance": 0.0, "consistency": 0},
    }
    ordered = [t for t, _ in sorted(assets.items(), key=recommendation._asset_sort_key)]
    assert ordered == [
        "AAA",
        "AAB",
        "AAC",
        "AAD",
        "BBA",
        "BBB",
        "BBC",
        "CCA",
        "CCB",
        "DDA",
        "EEA",
        "FFA",
    ]


def test_asset_sort_tiebreakers_exact():
    assets = {
        "AAA": {"class": "Stars", "performance": 2.0, "consistency": 95},
        "AAB": {"class": "Stars", "performance": 2.0, "consistency": 94},
        "AAC": {"class": "Stars", "performance": 1.9, "consistency": 99},
    }
    ordered = [t for t, _ in sorted(assets.items(), key=recommendation._asset_sort_key)]
    assert ordered == ["AAA", "AAB", "AAC"]


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


def test_param_stability_all_discarded_uses_all_folds():
    folds = [
        Fold(
            fold_id=i,
            validation_fitness=1.0,
            params={"x": v},
            champion_status="Discarded",
        )
        for i, v in enumerate([1.0, 1.0, 100.0])
    ]
    cov_discarded, _, _ = recommendation._param_stability(folds)
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
    assert cov_discarded == cov_all
    assert cov_discarded["x"] > 1


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


def test_param_stability_ignores_non_finite_values():
    folds = [
        Fold(
            fold_id=0,
            validation_fitness=1.0,
            params={"x": 1.0, "y": float("nan")},
            champion_status=None,
        ),
        Fold(
            fold_id=1,
            validation_fitness=1.0,
            params={"x": float("inf"), "y": 2.0},
            champion_status=None,
        ),
        Fold(
            fold_id=2,
            validation_fitness=1.0,
            params={"x": 2.0, "y": float("-inf")},
            champion_status=None,
        ),
    ]
    cov, unstable, watch = recommendation._param_stability(folds)
    assert cov["x"] == pytest.approx(0.33, abs=0.01)
    assert "y" not in cov
    assert unstable == []
    assert watch == []


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
    assert out["failed_file"] == "summary"
    meta = json.loads((tmp_path / "run_metadata.json").read_text())
    assert meta["recommendation"]["error"] == "schema_validation_failed"
    assert meta["recommendation"]["failed_file"] == "summary"
    md = (tmp_path / "strategy_recommendation.md").read_text()
    assert "schema validation failed" in md.lower()
    # Ensure the markdown explicitly labels the failing schema section
    assert "## Failed file: summary" in md
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
    assert out["failed_file"] == "per_asset"
    diag = out.get("diagnostics", "")
    # Diagnostics should report unknown column headers
    assert diag.startswith("Unknown columns:") and "foo" in diag
    md = (tmp_path / "strategy_recommendation.md").read_text()
    assert "schema validation failed (per_asset)" in md
    assert "## Failed file: per_asset" in md
    assert "Diagnostics" in md and "Unknown columns:" in md and "foo" in md


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
    obj, unknown = load_wf_per_asset(wf / "walk_forward_per_asset.csv")
    assert obj.rows[0].ticker == "AAA"
    assert unknown == []


def test_load_wf_per_asset_accepts_reason_column(tmp_path):
    wf = tmp_path / "walk_forward"
    wf.mkdir()
    csv = (
        "Fold,Ticker,Score,Trades,Included,Reason\n"
        "0,AAA,1.0,5,False,timeout dominated\n"
    )
    (wf / "walk_forward_per_asset.csv").write_text(csv)
    obj, unknown = load_wf_per_asset(wf / "walk_forward_per_asset.csv")
    assert obj.rows[0].reason == "timeout dominated"
    assert unknown == []


def test_load_wf_per_asset_snapshot_with_reasons():
    csv_path = (
        Path(__file__).parent / "snapshots" / "walk_forward_per_asset_with_reasons.csv"
    )
    obj, unknown = load_wf_per_asset(csv_path)
    assert unknown == []
    rows = [
        {
            "fold": row.fold,
            "ticker": row.ticker,
            "score": row.score,
            "trades": row.trades,
            "included": row.included,
            "reason": row.reason,
        }
        for row in obj.rows
    ]
    assert rows == [
        {
            "fold": 0,
            "ticker": "BTCUSDT",
            "score": 1.23,
            "trades": 6,
            "included": True,
            "reason": None,
        },
        {
            "fold": 0,
            "ticker": "ETHUSDT",
            "score": 0.87,
            "trades": 4,
            "included": False,
            "reason": "timeout dominated",
        },
        {
            "fold": 1,
            "ticker": "BTCUSDT",
            "score": 1.05,
            "trades": 5,
            "included": True,
            "reason": None,
        },
        {
            "fold": 1,
            "ticker": "ETHUSDT",
            "score": 0.92,
            "trades": 5,
            "included": True,
            "reason": "breakeven_follow_tp",
        },
    ]


def test_unknown_columns_logged_on_success(tmp_path, monkeypatch):
    _write_sample_files(tmp_path)
    wf = tmp_path / "walk_forward"
    df = pd.read_csv(wf / "walk_forward_per_asset.csv")
    df["foo"] = 1
    df.to_csv(wf / "walk_forward_per_asset.csv", index=False)
    monkeypatch.setitem(config.RECOMMENDATION, "LOG_UNKNOWN_COLUMNS_ON_SUCCESS", True)
    recommendation.generate_recommendation({"run_dir": tmp_path})
    meta = json.loads((tmp_path / "run_metadata.json").read_text())
    assert {
        "source": "walk_forward_per_asset.csv",
        "unknown_columns": ["foo"],
    } in meta.get("diagnostics", [])


def test_logging_origin_env_override(tmp_path, monkeypatch):
    _write_sample_files(tmp_path)
    wf = tmp_path / "walk_forward"
    df = pd.read_csv(wf / "walk_forward_per_asset.csv")
    df["foo"] = 1
    df.to_csv(wf / "walk_forward_per_asset.csv", index=False)
    monkeypatch.setattr(config, "IS_PROD", True)
    monkeypatch.setattr(config, "ENV_NAME", "production")
    monkeypatch.setenv("SRE_LOG_UNKNOWN_COLS", "1")
    monkeypatch.setitem(config.RECOMMENDATION, "LOG_UNKNOWN_COLUMNS_ON_SUCCESS", True)
    recommendation.generate_recommendation({"run_dir": tmp_path})
    md = (tmp_path / "strategy_recommendation.md").read_text(encoding="utf-8")
    assert "LOG_UNKNOWN_COLUMNS_ON_SUCCESS: True (env override (1))" in md
    meta = json.loads((tmp_path / "run_metadata.json").read_text())
    assert {
        "source": "walk_forward_per_asset.csv",
        "unknown_columns": ["foo"],
    } in meta.get("diagnostics", [])


def test_generate_recommendation_determinism(tmp_path):
    _write_sample_files(tmp_path)
    out1 = recommendation.generate_recommendation({"run_dir": tmp_path})
    md1 = (tmp_path / "strategy_recommendation.md").read_text()
    meta1 = json.loads((tmp_path / "run_metadata.json").read_text())
    assert "strategy_recommendation.md" in meta1["artifacts"]
    digest1 = (
        meta1.get("artifacts_meta", {})
        .get("strategy_recommendation.md", {})
        .get("sha256")
    )
    assert isinstance(digest1, str) and len(digest1) == 64
    tmp2 = tmp_path / "b"
    tmp2.mkdir()
    _write_sample_files(tmp2)
    out2 = recommendation.generate_recommendation({"run_dir": tmp2})
    md2 = (tmp2 / "strategy_recommendation.md").read_text()
    meta2 = json.loads((tmp2 / "run_metadata.json").read_text())
    digest2 = (
        meta2.get("artifacts_meta", {})
        .get("strategy_recommendation.md", {})
        .get("sha256")
    )
    assert out1 == out2 and md1 == md2
    assert digest1 == digest2
    assert re.search(r"Folds: median", md1)


def test_markdown_sections_are_complete(tmp_path):
    _write_sample_files(tmp_path)
    payload = recommendation.generate_recommendation({"run_dir": tmp_path})
    md_path = tmp_path / "strategy_recommendation.md"
    text = md_path.read_text(encoding="utf-8")
    assert "..." not in text and "…" not in text

    headings = [
        "## Overall Confidence",
        "### Confidence Factors",
        "## Asset Summary",
        "## Parameter Summary",
        "## Asset Performance Matrix",
        "## Parameter Stability",
        "## SRE Config",
    ]
    for heading in headings:
        assert text.count(heading) == 1

    issues = recommendation._audit_markdown(md_path, text=text)
    assert issues == []

    lines = text.splitlines()
    asset_line = lines[lines.index("## Asset Summary") + 1]
    param_line = lines[lines.index("## Parameter Summary") + 1]
    assert asset_line.endswith(".")
    assert param_line.endswith(".")

    header = "| Ticker | Performance | Consistency | Class | Samples |"
    assert header in text
    idx = lines.index(header)
    rows = []
    for line in lines[idx + 2 :]:
        if not line.startswith("|"):
            break
        rows.append(line)
    assert len(rows) == len(payload["assets"])

    assert "No unstable parameters detected." in text
    assert "- ENV: —" in text
    assert "- IS_PROD: False" in text
    assert "- LOG_UNKNOWN_COLUMNS_ON_SUCCESS: True (dev default)" in text


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
    assert legend_line.startswith("Legend: ")
    th = config.RECOMMENDATION["ASSET_CLASS_THRESHOLDS"]
    segments = [
        (
            f"Stars \u2265{th['star']['performance']} perf & "
            f"\u2265{th['star']['consistency']}% consistency"
        ),
        (
            f"Stalwarts {th['stalwart']['performance_low']}\u2013"
            f"{th['stalwart']['performance_high']} perf & \u2265"
            f"{th['stalwart']['consistency']}% consistency"
        ),
        (
            f"Gambles \u2265{th['gamble']['performance']} perf & "
            f"<{th['gamble']['consistency']}% consistency"
        ),
        (
            f"Drags <{th['drag']['performance']} perf & <"
            f"{th['drag']['consistency']}% consistency"
        ),
    ]
    for seg in segments:
        assert seg in legend_line
    assert "\n" not in legend_line


def test_markdown_asset_summary_includes_drag_stance(tmp_path):
    assets = {
        "AVAXUSDT": {
            "performance": -0.4,
            "consistency": 35.0,
            "class": "Drags",
            "samples": 3,
        },
        "BTCUSDT": {
            "performance": -0.9,
            "consistency": 25.0,
            "class": "Drags",
            "samples": 3,
        },
    }
    payload = _mk_payload(assets=assets)
    md_path = tmp_path / "drag.md"
    recommendation._write_markdown(md_path, payload)
    text = md_path.read_text(encoding="utf-8")
    assert "Drags should be underweighted or avoided (e.g., AVAXUSDT, BTCUSDT)." in text


def test_markdown_matrix_header_even_when_empty(tmp_path):
    payload = _mk_payload(assets={})
    md_path = tmp_path / "empty.md"
    recommendation._write_markdown(md_path, payload)
    lines = md_path.read_text(encoding="utf-8").splitlines()
    header = "| Ticker | Performance | Consistency | Class | Samples |"
    idx = lines.index(header)
    assert lines[idx + 1] == "|---|---|---|---|---|"
    assert lines[idx + 2] == ""


def test_markdown_parameter_implication_present(tmp_path):
    assets = {
        "STAR": {
            "performance": 1.5,
            "consistency": 85.0,
            "class": "Stars",
            "samples": 3,
        }
    }
    payload = _mk_payload(
        assets=assets,
        cov_by_gene={"stop_loss_pct": float("inf"), "cci_period": 0.4},
        unstable=["stop_loss_pct"],
        watchlist=["cci_period"],
    )
    md_path = tmp_path / "params.md"
    recommendation._write_markdown(md_path, payload)
    text = md_path.read_text(encoding="utf-8")
    assert text.count(PARAM_STABILITY_IMPLICATION) == 2
    lines = text.splitlines()
    summary_line = lines[lines.index("## Parameter Summary") + 1]
    assert PARAM_STABILITY_IMPLICATION in summary_line
    stability_idx = lines.index("## Parameter Stability")
    trailing = lines[stability_idx + 1 :]
    assert any(line == PARAM_STABILITY_IMPLICATION for line in trailing)


def test_markdown_confidence_factors_line_format(tmp_path):
    conf = {
        "category": "Medium",
        "score": 55,
        "scores": {
            "median": 33.3,
            "consistency": 66.6,
            "tail": 45.5,
            "downside": 88.8,
        },
        "factors": {
            "median_fitness": 1.2345,
            "positive_fold_pct": 62.345,
            "worst_fold_fitness": -0.9876,
            "downside_deviation": 0.0049,
        },
    }
    payload = _mk_payload(confidence=conf)
    md_path = tmp_path / "confidence.md"
    recommendation._write_markdown(md_path, payload)
    lines = md_path.read_text(encoding="utf-8").splitlines()
    idx = lines.index("### Confidence Factors") + 1
    assert lines[idx] == "Folds: median 1.23, worst -0.99, positive 62.3%."


def test_asset_matrix_markdown_sort_order(tmp_path):
    payload = {
        "confidence": {
            "category": "Low",
            "score": 0,
            "scores": {
                "median": 0,
                "consistency": 0,
                "tail": 0,
                "downside": 0,
            },
            "factors": {
                "median_fitness": 0,
                "positive_fold_pct": 0,
                "worst_fold_fitness": 0,
                "downside_deviation": 0,
            },
        },
        "assets": {
            "STAA": {
                "performance": 2.0,
                "consistency": 95.0,
                "class": "Stars",
                "samples": 4,
            },
            "STAB": {
                "performance": 2.0,
                "consistency": 95.0,
                "class": "Stars",
                "samples": 4,
            },
            "STAC": {
                "performance": 1.5,
                "consistency": 97.0,
                "class": "Stars",
                "samples": 4,
            },
            "STAD": {
                "performance": 1.5,
                "consistency": 90.0,
                "class": "Stars",
                "samples": 4,
            },
            "STL1": {
                "performance": 1.0,
                "consistency": 70.0,
                "class": "Stalwarts",
                "samples": 3,
            },
            "STL2": {
                "performance": 1.0,
                "consistency": 60.0,
                "class": "Stalwarts",
                "samples": 3,
            },
            "STL3": {
                "performance": 0.8,
                "consistency": 65.0,
                "class": "Stalwarts",
                "samples": 3,
            },
            "GMB1": {
                "performance": 1.3,
                "consistency": 45.0,
                "class": "Gambles",
                "samples": 5,
            },
            "GMB2": {
                "performance": 1.1,
                "consistency": 40.0,
                "class": "Gambles",
                "samples": 5,
            },
            "BRD1": {
                "performance": 0.2,
                "consistency": 55.0,
                "class": "Borderline",
                "samples": 2,
            },
            "DRG1": {
                "performance": -0.5,
                "consistency": 30.0,
                "class": "Drags",
                "samples": 3,
            },
            "INS1": {
                "performance": 0.0,
                "consistency": 0.0,
                "class": "Insufficient Data",
                "samples": 1,
            },
        },
        "param_stability": {"cov_by_gene": {}, "unstable_genes": []},
        "narrative": {"overall": "", "assets": "", "params": ""},
        "schema_version": "1.0",
    }
    md_path = tmp_path / "report.md"
    recommendation._write_markdown(md_path, payload)
    lines = md_path.read_text(encoding="utf-8").splitlines()
    header = "| Ticker | Performance | Consistency | Class | Samples |"
    idx = lines.index(header)
    rows = []
    for line in lines[idx + 2 :]:
        if not line.startswith("|"):
            break
        rows.append(line.split("|")[1].strip())
    assert rows == [
        "STAA",
        "STAB",
        "STAC",
        "STAD",
        "STL1",
        "STL2",
        "STL3",
        "GMB1",
        "GMB2",
        "BRD1",
        "DRG1",
        "INS1",
    ]


def test_markdown_artifact_deduped(tmp_path):
    _write_sample_files(tmp_path)
    recommendation.generate_recommendation({"run_dir": tmp_path})
    recommendation.generate_recommendation({"run_dir": tmp_path})
    meta = json.loads((tmp_path / "run_metadata.json").read_text())
    assert meta["artifacts"].count("strategy_recommendation.md") == 1


def test_markdown_snapshot_matches(tmp_path):
    _write_sample_files(tmp_path)
    recommendation.generate_recommendation({"run_dir": tmp_path})
    md_text = (tmp_path / "strategy_recommendation.md").read_text(encoding="utf-8")
    snapshot_path = Path(__file__).parent / "snapshots" / "strategy_recommendation.md"
    expected = snapshot_path.read_text(encoding="utf-8")
    assert md_text == expected


def test_audit_markdown_detects_truncation(tmp_path):
    bad = tmp_path / "bad.md"
    bad.write_text("## Overall Confidence\n...\n", encoding="utf-8")
    issues = recommendation._audit_markdown(bad)
    assert any("ellipses" in issue for issue in issues)
    assert any("Required anchor missing" in issue for issue in issues)


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


def _mk_confidence():
    return {
        "category": "Low",
        "score": 0,
        "scores": {"median": 0, "consistency": 0, "tail": 0, "downside": 0},
        "factors": {
            "median_fitness": 0,
            "positive_fold_pct": 0,
            "worst_fold_fitness": 0,
            "downside_deviation": 0,
        },
    }


def _base_conf():
    return _mk_confidence()


def _mk_payload(
    *,
    assets: dict[str, dict[str, object]] | None = None,
    cov_by_gene: dict[str, float] | None = None,
    unstable: list[str] | None = None,
    watchlist: list[str] | None = None,
    confidence: dict[str, object] | None = None,
):
    conf = confidence or _mk_confidence()
    asset_map = assets or {}
    unstable_genes = list(unstable or [])
    watchlist_genes = list(watchlist or [])
    cov = dict(cov_by_gene or {})
    for gene in unstable_genes + watchlist_genes:
        cov.setdefault(gene, 0.0)
    narrative = recommendation._build_narrative(
        conf, asset_map, unstable_genes, watchlist_genes
    )
    return {
        "confidence": conf,
        "assets": asset_map,
        "param_stability": {
            "cov_by_gene": cov,
            "unstable_genes": unstable_genes,
            "watchlist_genes": watchlist_genes,
        },
        "narrative": narrative,
        "schema_version": "1.0",
    }


def test_narrative_includes_drag_stance():
    conf = _base_conf()
    assets = {
        "AAA": {
            "class": "Drags",
            "performance": -1.0,
            "consistency": 0.0,
            "samples": 3,
        }
    }
    out = recommendation._build_narrative(conf, assets, [], [])
    assert "Drags should be underweighted or avoided" in out["assets"]


def test_params_narrative_implication_when_unstable():
    conf = _base_conf()
    out = recommendation._build_narrative(conf, {}, ["x"], [])
    assert PARAM_STABILITY_IMPLICATION in out["params"]


def test_params_narrative_implication_when_watchlist():
    conf = _base_conf()
    out = recommendation._build_narrative(conf, {}, [], ["y"])
    assert PARAM_STABILITY_IMPLICATION in out["params"]


def test_params_narrative_no_implication_when_stable():
    conf = _base_conf()
    out = recommendation._build_narrative(conf, {}, [], [])
    assert PARAM_STABILITY_IMPLICATION not in out["params"]


def test_infinite_cov_rendered(tmp_path):
    payload = _mk_payload(cov_by_gene={"x": float("inf")}, unstable=["x"])
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


def test_fmt_helpers_normalize_negative_zero():
    assert fmt_num(-0.0) == "0.00"
    assert fmt_pct(-0.0) == "0.0%"
