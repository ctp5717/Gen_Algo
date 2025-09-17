import copy
import json
import logging
import math
from pathlib import Path

import pytest

import config
import final_strategy
from final_strategy import WeightedFold
from schemas import Fold, Metadata, PerAssetRow, WalkForwardPerAssetV1, WalkForwardSummaryV1


@pytest.fixture
def cfg_copy():
    return copy.deepcopy(config.FINAL_STRATEGY)


def _make_weighted_fold(fold_id: int, params: dict, weight: float) -> WeightedFold:
    fold = Fold.model_construct(
        fold_id=fold_id,
        validation_fitness=1.0,
        params=params,
        champion_status="Elite",
    )
    return WeightedFold(fold=fold, base_weight=1.0, decay_factor=1.0, weight=weight)


def test_parameter_aggregation_weighted_median_and_mode(cfg_copy):
    folds = [
        _make_weighted_fold(0, {"alpha": 10, "mode": "A"}, 0.7),
        _make_weighted_fold(1, {"alpha": 12, "mode": "B"}, 0.3),
    ]
    params, summaries = final_strategy._aggregate_parameters(folds, cfg_copy)
    assert params["alpha"] == 10
    assert summaries["alpha"].stability == "Stable"
    assert params["mode"] == "A"
    assert summaries["mode"].stability == "Consensus"


def test_parameter_multimodal_detection(cfg_copy):
    folds = [
        _make_weighted_fold(0, {"beta": 10}, 0.5),
        _make_weighted_fold(1, {"beta": 90}, 0.5),
        _make_weighted_fold(2, {"beta": 95}, 0.0001),
    ]
    params, summaries = final_strategy._aggregate_parameters(folds, cfg_copy)
    assert params["beta"] in {10, 90, 95}
    assert summaries["beta"].multi_modal
    assert "Multi-modal" in summaries["beta"].stability


def test_parameter_precision_overrides(cfg_copy):
    cfg_copy["PARAM_VALUE_DECIMALS"] = {"default": 2, "alpha": 4}
    folds = [
        _make_weighted_fold(0, {"alpha": 1.23456}, 0.6),
        _make_weighted_fold(1, {"alpha": 1.23454}, 0.4),
    ]
    params, summaries = final_strategy._aggregate_parameters(folds, cfg_copy)
    assert params["alpha"] == pytest.approx(1.2346, abs=1e-9)
    assert summaries["alpha"].precision == 4


def test_parameter_near_zero_rcv_triggers_note(cfg_copy):
    folds = [
        _make_weighted_fold(0, {"gamma": 0.0}, 0.6),
        _make_weighted_fold(1, {"gamma": 0.0}, 0.4),
    ]
    _, summaries = final_strategy._aggregate_parameters(folds, cfg_copy)
    assert math.isinf(summaries["gamma"].rcv)
    assert summaries["gamma"].median_near_zero
    notes = final_strategy._notes_from_summaries(summaries)
    assert any("median ≈ 0" in note for note in notes)


@pytest.mark.parametrize(
    ("decay", "expected_half_life"),
    [(math.log(2), 1.0), (math.log(2) / 3.0, 3.0)],
)
def test_recency_weighting_applies_decay(cfg_copy, caplog, decay, expected_half_life):
    cfg_copy["USE_RECENCY_WEIGHTING"] = True
    cfg_copy["FOLD_DECAY_RATE"] = decay
    folds = [
        Fold(fold_id=0, validation_fitness=1.0, params={}, champion_status="Elite"),
        Fold(fold_id=1, validation_fitness=1.0, params={}, champion_status="Elite"),
        Fold(fold_id=2, validation_fitness=1.0, params={}, champion_status="Elite"),
    ]
    with caplog.at_level(logging.INFO):
        _, mapping = final_strategy._compute_fold_weights(folds, cfg_copy)
    assert pytest.approx(sum(mapping.values()), rel=1e-6) == 1.0
    assert mapping[2] > mapping[0]
    assert f"half-life ≈ {expected_half_life:.2f} folds" in caplog.text


def test_fold_weights_all_negative_fitness(cfg_copy):
    folds = [
        Fold(fold_id=idx, validation_fitness=-0.1, params={}, champion_status="Elite")
        for idx in range(3)
    ]
    _, mapping = final_strategy._compute_fold_weights(folds, cfg_copy)
    expected = 1.0 / len(folds)
    for fold in folds:
        assert pytest.approx(mapping[fold.fold_id], rel=1e-6) == expected


def test_asset_weighting_risk_adjusted(cfg_copy):
    cfg_copy["WEIGHTING_SCHEME"] = "risk_adjusted"
    cfg_copy["MAX_WEIGHT_CAP"] = 1.0
    cfg_copy["MIN_WEIGHT_FLOOR"] = 0.0
    cfg_copy["SHRINK_TO_EQUAL"] = 0.0
    sre_assets = {
        "AAA": {"class": "Stars", "performance": 1.2, "consistency": 80.0},
        "BBB": {"class": "Stalwarts", "performance": 0.6, "consistency": 70.0},
    }
    per_asset = WalkForwardPerAssetV1(
        rows=[
            PerAssetRow(fold=0, ticker="AAA", score=1.0, trades=5, included=True),
            PerAssetRow(fold=1, ticker="AAA", score=1.2, trades=5, included=True),
            PerAssetRow(fold=0, ticker="BBB", score=0.4, trades=5, included=True),
            PerAssetRow(fold=1, ticker="BBB", score=0.5, trades=5, included=True),
        ]
    )
    assets, derivation, exclusions, notes = final_strategy._compute_asset_allocation(
        sre_assets,
        per_asset,
        {0: 0.6, 1: 0.4},
        cfg_copy,
    )
    assert not exclusions
    assert not notes
    assert assets["AAA"]["weight"] > assets["BBB"]["weight"]
    assert pytest.approx(sum(a["weight"] for a in assets.values()), rel=1e-6) == 1.0
    assert derivation["AAA"].raw_weight > derivation["BBB"].raw_weight


def test_asset_weighting_override_mismatch(cfg_copy):
    cfg_copy["WEIGHTING_SCHEME"] = "override"
    cfg_copy["ASSET_WEIGHTS_OVERRIDE"] = {"AAA": 1.0}
    sre_assets = {
        "AAA": {"class": "Stars", "performance": 1.2, "consistency": 80.0},
        "BBB": {"class": "Stalwarts", "performance": 0.6, "consistency": 75.0},
    }
    per_asset = WalkForwardPerAssetV1(
        rows=[PerAssetRow(fold=0, ticker="AAA", score=1.0, trades=5, included=True)]
    )
    with pytest.raises(final_strategy.FinalStrategyError):
        final_strategy._compute_asset_allocation(sre_assets, per_asset, {0: 1.0}, cfg_copy)


def test_asset_weighting_relaxes_floor(cfg_copy):
    cfg_copy["MIN_WEIGHT_FLOOR"] = 0.6
    cfg_copy["MAX_WEIGHT_CAP"] = 0.8
    cfg_copy["SHRINK_TO_EQUAL"] = 0.0
    sre_assets = {
        "AAA": {"class": "Stars", "performance": 1.0, "consistency": 80.0},
        "BBB": {"class": "Stalwarts", "performance": 0.9, "consistency": 70.0},
    }
    per_asset = WalkForwardPerAssetV1(
        rows=[
            PerAssetRow(fold=0, ticker="AAA", score=1.0, trades=5, included=True),
            PerAssetRow(fold=0, ticker="BBB", score=0.8, trades=5, included=True),
        ]
    )
    assets, _, _, notes = final_strategy._compute_asset_allocation(
        sre_assets, per_asset, {0: 1.0}, cfg_copy
    )
    assert pytest.approx(sum(a["weight"] for a in assets.values()), rel=1e-6) == 1.0
    assert any("Weight floor relaxed" in note for note in notes)


def test_asset_selection_case_insensitive(cfg_copy):
    cfg_copy["INCLUDE_CLASSES"] = ["stars"]
    sre_assets = {
        "AAA": {"class": "Stars", "performance": 1.0, "consistency": 75.0},
        "BBB": {"class": "Stalwarts", "performance": 0.8, "consistency": 80.0},
    }
    per_asset = WalkForwardPerAssetV1(
        rows=[
            PerAssetRow(fold=0, ticker="AAA", score=1.0, trades=5, included=True),
            PerAssetRow(fold=0, ticker="BBB", score=0.9, trades=5, included=True),
        ]
    )
    assets, _, exclusions, _ = final_strategy._compute_asset_allocation(
        sre_assets, per_asset, {0: 1.0}, cfg_copy
    )
    assert set(assets.keys()) == {"AAA"}
    assert any("BBB" in reason and "INCLUDE_CLASSES" in reason for reason in exclusions)


def test_jackknife_notes_include_thresholds(cfg_copy):
    cfg_copy["PARAM_SENSITIVITY_THRESHOLD"] = 0.05
    cfg_copy["WEIGHT_SENSITIVITY_THRESHOLD"] = 0.02
    cfg_copy["WEIGHT_SENSITIVITY_RATIO_THRESHOLD"] = 0.2
    cfg_copy["MAX_WEIGHT_CAP"] = 1.0
    cfg_copy["MIN_WEIGHT_FLOOR"] = 0.0
    cfg_copy["SHRINK_TO_EQUAL"] = 0.0
    folds = [
        Fold(fold_id=0, validation_fitness=0.2, params={"alpha": 1.0}, champion_status="Elite"),
        Fold(fold_id=1, validation_fitness=0.9, params={"alpha": 10.0}, champion_status="Elite"),
        Fold(fold_id=2, validation_fitness=0.8, params={"alpha": 20.0}, champion_status="Elite"),
    ]
    weighted_folds, mapping = final_strategy._compute_fold_weights(folds, cfg_copy)
    _, summaries = final_strategy._aggregate_parameters(weighted_folds, cfg_copy)
    sre_assets = {
        "AAA": {"class": "Stars", "performance": 1.0, "consistency": 80.0},
        "BBB": {"class": "Stalwarts", "performance": 1.0, "consistency": 80.0},
    }
    per_asset = WalkForwardPerAssetV1(
        rows=[
            PerAssetRow(fold=0, ticker="AAA", score=3.0, trades=5, included=True),
            PerAssetRow(fold=1, ticker="AAA", score=3.0, trades=5, included=True),
            PerAssetRow(fold=2, ticker="AAA", score=-3.0, trades=5, included=True),
            PerAssetRow(fold=0, ticker="BBB", score=0.5, trades=5, included=True),
            PerAssetRow(fold=1, ticker="BBB", score=0.5, trades=5, included=True),
            PerAssetRow(fold=2, ticker="BBB", score=0.5, trades=5, included=True),
        ]
    )
    assets, _, _, _ = final_strategy._compute_asset_allocation(
        sre_assets,
        per_asset,
        mapping,
        cfg_copy,
    )
    notes = final_strategy._jackknife_sensitivity(
        folds,
        cfg_copy,
        sre_assets,
        per_asset,
        summaries,
        assets,
    )
    assert any("> 0.05 threshold" in note for note in notes)
    assert any("0.02" in note and "abs threshold" in note for note in notes)
    assert any("0.20" in note and "ratio threshold" in note for note in notes)


def test_weight_sensitivity_ratio_guard(cfg_copy):
    cfg_copy["PARAM_SENSITIVITY_THRESHOLD"] = 1.0
    cfg_copy["WEIGHT_SENSITIVITY_THRESHOLD"] = 0.01
    cfg_copy["WEIGHT_SENSITIVITY_RATIO_THRESHOLD"] = 10.0
    cfg_copy["MAX_WEIGHT_CAP"] = 1.0
    cfg_copy["MIN_WEIGHT_FLOOR"] = 0.0
    cfg_copy["SHRINK_TO_EQUAL"] = 0.0
    folds = [
        Fold(fold_id=0, validation_fitness=1.0, params={"alpha": 1.0}, champion_status="Elite"),
        Fold(fold_id=1, validation_fitness=0.8, params={"alpha": 1.2}, champion_status="Elite"),
        Fold(fold_id=2, validation_fitness=0.7, params={"alpha": 1.4}, champion_status="Elite"),
    ]
    weighted_folds, mapping = final_strategy._compute_fold_weights(folds, cfg_copy)
    _, summaries = final_strategy._aggregate_parameters(weighted_folds, cfg_copy)
    sre_assets = {
        "AAA": {"class": "Stars", "performance": 1.0, "consistency": 80.0},
    }
    per_asset = WalkForwardPerAssetV1(
        rows=[
            PerAssetRow(fold=0, ticker="AAA", score=1.0, trades=5, included=True),
            PerAssetRow(fold=1, ticker="AAA", score=1.1, trades=5, included=True),
            PerAssetRow(fold=2, ticker="AAA", score=1.05, trades=5, included=True),
        ]
    )
    assets, _, _, _ = final_strategy._compute_asset_allocation(
        sre_assets,
        per_asset,
        mapping,
        cfg_copy,
    )
    notes = final_strategy._jackknife_sensitivity(
        folds,
        cfg_copy,
        sre_assets,
        per_asset,
        summaries,
        assets,
    )
    assert all("Weight for" not in note for note in notes)


def test_candidate_folds_empty_summary():
    summary = WalkForwardSummaryV1.model_construct(
        metadata=Metadata.model_construct(
            schema_version="1.0", num_folds=0, asset_universe=[]
        ),
        folds=[],
    )
    with pytest.raises(final_strategy.FinalStrategyError):
        final_strategy._candidate_folds(summary)


def test_confidence_gate_blocks_strategy(tmp_path):
    run_dir = tmp_path
    wf = run_dir / "walk_forward"
    wf.mkdir()
    summary = {
        "metadata": {"schema_version": "1.0", "num_folds": 1, "asset_universe": ["AAA"]},
        "folds": [
            {
                "fold_id": 0,
                "validation_fitness": 1.0,
                "params": {"alpha": 1},
                "champion_status": "Elite",
            }
        ],
    }
    (wf / "walk_forward_summary.json").write_text(json.dumps(summary))
    per_asset = "Fold,Ticker,Score,Trades,Included\n0,AAA,1.0,5,True\n"
    (wf / "walk_forward_per_asset.csv").write_text(per_asset)
    run_meta = {
        "recommendation": {
            "schema_version": "1.0",
            "confidence": {"score": 30, "category": "Low"},
            "assets": {},
        }
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(run_meta))
    payload = final_strategy.generate_final_strategy({"run_dir": run_dir})
    assert payload["parameters"] == {}
    assert payload["assets"] == {}
    md = (run_dir / "final_strategy.md").read_text()
    assert "Confidence gate blocked" in md


def _write_integration_files(base: Path) -> None:
    wf = base / "walk_forward"
    wf.mkdir()
    summary = {
        "metadata": {"schema_version": "1.0", "num_folds": 3, "asset_universe": ["AAA", "BBB", "CCC"]},
        "folds": [
            {
                "fold_id": 0,
                "validation_fitness": 1.2,
                "params": {"alpha": 10, "beta": 1.5},
                "champion_status": "Elite",
            },
            {
                "fold_id": 1,
                "validation_fitness": 0.8,
                "params": {"alpha": 11, "beta": 1.6},
                "champion_status": "Viable",
            },
            {
                "fold_id": 2,
                "validation_fitness": 0.4,
                "params": {"alpha": 9, "beta": 1.7},
                "champion_status": "Discarded",
            },
        ],
    }
    (wf / "walk_forward_summary.json").write_text(json.dumps(summary))
    per_asset = "\n".join(
        [
            "Fold,Ticker,Score,Trades,Included",
            "0,AAA,1.1,5,True",
            "1,AAA,1.0,5,True",
            "0,BBB,0.6,5,True",
            "1,BBB,0.7,5,True",
            "0,CCC,-0.5,5,True",
            "1,CCC,-0.6,5,True",
        ]
    )
    (wf / "walk_forward_per_asset.csv").write_text(per_asset)
    run_meta = {
        "recommendation": {
            "schema_version": "1.0",
            "confidence": {"score": 71, "category": "Medium"},
            "assets": {
                "AAA": {"class": "Stars", "performance": 1.1, "consistency": 80.0},
                "BBB": {"class": "Stalwarts", "performance": 0.65, "consistency": 70.0},
                "CCC": {"class": "Gambles", "performance": -0.5, "consistency": 45.0},
            },
        }
    }
    (base / "run_metadata.json").write_text(json.dumps(run_meta))


def test_generate_final_strategy_integration_snapshot(tmp_path, monkeypatch):
    _write_integration_files(tmp_path)
    monkeypatch.setitem(config.FINAL_STRATEGY, "SHRINK_TO_EQUAL", 0.0)
    result = final_strategy.generate_final_strategy({"run_dir": tmp_path})
    assert result["assets"]
    md_path = tmp_path / "final_strategy.md"
    snapshot_path = Path(__file__).parent / "snapshots" / "final_strategy.md"
    assert md_path.read_text() == snapshot_path.read_text()
    meta = json.loads((tmp_path / "run_metadata.json").read_text())
    assert "final_strategy" in meta
    assert "final_strategy.md" in meta.get("artifacts", [])


def test_validate_final_strategy_config_override_sum_error():
    bad = copy.deepcopy(config.FINAL_STRATEGY)
    bad["WEIGHTING_SCHEME"] = "override"
    bad["ASSET_WEIGHTS_OVERRIDE"] = {"AAA": 0.5, "BBB": 0.3}
    with pytest.raises(config.ConfigurationError):
        config.validate_final_strategy_config(bad)


def test_validate_final_strategy_config_negative_param_threshold():
    bad = copy.deepcopy(config.FINAL_STRATEGY)
    bad["PARAM_SENSITIVITY_THRESHOLD"] = -0.1
    with pytest.raises(config.ConfigurationError):
        config.validate_final_strategy_config(bad)


def test_validate_final_strategy_config_param_threshold_upper_bound():
    bad = copy.deepcopy(config.FINAL_STRATEGY)
    bad["PARAM_SENSITIVITY_THRESHOLD"] = 1.5
    with pytest.raises(config.ConfigurationError):
        config.validate_final_strategy_config(bad)


def test_validate_final_strategy_config_weight_threshold_bounds():
    bad = copy.deepcopy(config.FINAL_STRATEGY)
    bad["WEIGHT_SENSITIVITY_THRESHOLD"] = 1.5
    with pytest.raises(config.ConfigurationError):
        config.validate_final_strategy_config(bad)


def test_validate_final_strategy_config_unknown_classes_warn():
    bad = copy.deepcopy(config.FINAL_STRATEGY)
    bad["INCLUDE_CLASSES"] = ["Stars", "Aliens"]
    with pytest.warns(UserWarning, match="Unknown FINAL_STRATEGY INCLUDE_CLASSES"):
        config.validate_final_strategy_config(bad)


def test_validate_final_strategy_config_negative_ratio_threshold():
    bad = copy.deepcopy(config.FINAL_STRATEGY)
    bad["WEIGHT_SENSITIVITY_RATIO_THRESHOLD"] = -0.1
    with pytest.raises(config.ConfigurationError):
        config.validate_final_strategy_config(bad)


def test_validate_final_strategy_config_param_decimals_invalid():
    bad = copy.deepcopy(config.FINAL_STRATEGY)
    bad["PARAM_VALUE_DECIMALS"] = {"default": "two"}
    with pytest.raises(config.ConfigurationError):
        config.validate_final_strategy_config(bad)
