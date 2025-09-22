import copy
import hashlib
import json
import logging
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

# Ensure repository root is on the import path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Stub heavy optional dependencies
sys.modules.setdefault("pandas_ta", types.ModuleType("pandas_ta"))
sys.modules.setdefault(
    "vectorbt", types.SimpleNamespace(__version__="0", __file__=__file__)
)

import final_strategy  # noqa: E402
import main  # noqa: E402
from run_metadata import merge_run_metadata  # noqa: E402

main.config.initialize_config()


def _stub_matplotlib():
    axis = types.SimpleNamespace(
        plot=lambda *a, **k: None,
        get_legend_handles_labels=lambda: ([], []),
        legend=lambda *a, **k: None,
        set_title=lambda *a, **k: None,
        set_xlabel=lambda *a, **k: None,
        set_ylabel=lambda *a, **k: None,
    )
    fig = types.SimpleNamespace()
    return types.SimpleNamespace(
        ion=lambda: None,
        plot=lambda *a, **k: None,
        gca=lambda: axis,
        legend=lambda *a, **k: None,
        xlabel=lambda *a, **k: None,
        ylabel=lambda *a, **k: None,
        title=lambda *a, **k: None,
        show=lambda *a, **k: None,
        savefig=lambda *a, **k: None,
        close=lambda *a, **k: None,
        subplots=lambda *a, **k: (fig, axis),
    )


def _configure_minimal_main(monkeypatch):
    vb = types.SimpleNamespace(__version__="0", __file__=__file__)
    monkeypatch.setitem(sys.modules, "vectorbt", vb)

    df = pd.DataFrame(
        {
            "Open": [1, 2, 3],
            "High": [1, 2, 3],
            "Low": [1, 2, 3],
            "Close": [1, 2, 3],
            "Volume": [100, 100, 100],
        },
        index=pd.date_range("2020-01-01", periods=3),
    )

    monkeypatch.setattr(main.data_loader, "get_data", lambda *a, **k: (df, "cache"))
    monkeypatch.setitem(main.config.MULTI_ASSET, "enabled", False)

    gene_space = [{"low": 0, "high": 1}]
    gene_map = {0: {"name": "x", "path": [], "type": float}}
    gene_types = [float]

    def parser_stub(*_args, **_kwargs):
        return gene_space, gene_map, gene_types

    monkeypatch.setattr(main, "parse_genes_from_config", parser_stub)

    class DummyGA:
        def __init__(self, *args, **kwargs):
            self.num_generations = 1
            self.generations_completed = 1
            self.last_generation_fitness = [1.0]
            self.best_solutions_fitness = [1.0]

        def run(self):
            return None

        def best_solution(self, **kwargs):
            return [0], 1.0, None

    monkeypatch.setattr(main.pygad, "GA", DummyGA)

    monkeypatch.setattr(main.analysis, "run_champion_analysis", lambda *a, **k: None)
    monkeypatch.setattr(main, "ensure_real_vectorbt", lambda *a, **k: None)
    monkeypatch.setattr(main, "indicator_preflight", lambda *a, **k: None)
    monkeypatch.setattr(
        main.analysis, "_write_run_metadata", lambda *a, **k: None, raising=False
    )
    monkeypatch.setattr(main, "plt", _stub_matplotlib())

    monkeypatch.setattr(
        main,
        "STRATEGY_RULES",
        {"entry_rules": {"combination_logic": "AND", "conditions": []}},
        raising=False,
    )
    monkeypatch.setattr(
        main.config,
        "STRATEGY_RULES",
        {"entry_rules": {"combination_logic": "AND", "conditions": []}},
        raising=False,
    )

    class DummyEvaluator:
        def __init__(self, *a, **k):
            pass

        def __call__(self, *a, **k):
            return 1.0

    monkeypatch.setattr(main.fitness, "FitnessEvaluator", DummyEvaluator)

    monkeypatch.setattr(
        main.config, "ENABLE_WALK_FORWARD_VALIDATION", False, raising=False
    )
    monkeypatch.setattr(
        main.config, "WALK_FORWARD_SETTINGS", {"enabled": False}, raising=False
    )
    monkeypatch.setattr(
        main.config, "FITNESS_WEIGHTS", {"min_trades": 0}, raising=False
    )
    monkeypatch.setattr(main.config, "GA_NUM_GENERATIONS", 1, raising=False)
    monkeypatch.setattr(main.config, "GA_POPULATION_SIZE", 1, raising=False)
    monkeypatch.setattr(main.config, "GA_PARENTS_MATING", 1, raising=False)
    monkeypatch.setattr(main.config, "GA_MUTATION_NUM_GENES", 1, raising=False)

    train_period = {"start": "2020-01-01", "end": "2020-01-02"}
    valid_period = {"start": "2020-01-02", "end": "2020-01-03"}
    monkeypatch.setattr(main.config, "TRAINING_PERIOD", train_period, raising=False)
    monkeypatch.setattr(main.config, "VALIDATION_PERIOD", valid_period, raising=False)
    monkeypatch.setattr(main.config, "SELECTED_ASSET_NAME", "Test", raising=False)
    monkeypatch.setattr(main.config, "TICKER", "TEST", raising=False)
    monkeypatch.setattr(main.config, "TIMEFRAME", "1d", raising=False)
    monkeypatch.setattr(main.config, "AUTO_TUNE_ENABLED", False, raising=False)

    return df


DEFAULT_WALK_FORWARD_SUMMARY = {
    "metadata": {
        "schema_version": "1.0",
        "num_folds": 2,
        "asset_universe": ["AAA", "BBB"],
    },
    "folds": [
        {
            "fold_id": 0,
            "validation_fitness": 1.2,
            "params": {"alpha": 10},
            "champion_status": "Elite",
        },
        {
            "fold_id": 1,
            "validation_fitness": 0.8,
            "params": {"alpha": 11},
            "champion_status": "Viable",
        },
    ],
}


DEFAULT_PER_ASSET_ROWS = [
    {"fold": 0, "ticker": "AAA", "score": 1.2, "trades": 5, "included": True},
    {"fold": 1, "ticker": "AAA", "score": 1.1, "trades": 5, "included": True},
    {"fold": 0, "ticker": "BBB", "score": 0.7, "trades": 5, "included": True},
    {"fold": 1, "ticker": "BBB", "score": 0.6, "trades": 5, "included": True},
]


DEFAULT_RECOMMENDATION = {
    "schema_version": "1.0",
    "confidence": {"score": 80, "category": "High"},
    "assets": {
        "AAA": {
            "class": "Stars",
            "performance": 1.1,
            "consistency": 75.0,
        },
        "BBB": {
            "class": "Stalwarts",
            "performance": 0.7,
            "consistency": 70.0,
        },
    },
}


def _install_walk_forward_stub(
    monkeypatch,
    *,
    summary: dict | None = None,
    per_asset_rows: list[dict] | None = None,
    omit_summary: bool = False,
):
    state: dict[str, object] = {}
    summary_payload = copy.deepcopy(summary or DEFAULT_WALK_FORWARD_SUMMARY)
    per_asset_payload = copy.deepcopy(per_asset_rows or DEFAULT_PER_ASSET_ROWS)

    def stub_walk_forward(run_dir, initial_champions, data):
        run_dir = Path(run_dir)
        state["run_dir"] = run_dir
        wf_dir = run_dir / "walk_forward"
        wf_dir.mkdir(parents=True, exist_ok=True)
        if not omit_summary:
            (wf_dir / "walk_forward_summary.json").write_text(
                json.dumps(summary_payload), encoding="utf-8"
            )
        csv_lines = ["fold,ticker,score,trades,included"]
        for row in per_asset_payload:
            csv_lines.append(
                ",".join(
                    [
                        str(row["fold"]),
                        row["ticker"],
                        f"{row['score']}",
                        f"{row['trades']}",
                        str(row["included"]).lower(),
                    ]
                )
            )
        (wf_dir / "walk_forward_per_asset.csv").write_text(
            "\n".join(csv_lines),
            encoding="utf-8",
        )
        return {"status": "ok"}

    walk_module = types.SimpleNamespace(run_walk_forward_validation=stub_walk_forward)
    monkeypatch.setitem(sys.modules, "walk_forward", walk_module)
    return state


def _install_recommendation_stub(
    monkeypatch,
    *,
    payload_mutator=None,
    post_write_hook=None,
):
    state: dict[str, object] = {}

    def stub_recommendation(ctx):
        run_dir = Path(ctx["run_dir"])
        payload = copy.deepcopy(DEFAULT_RECOMMENDATION)
        if payload_mutator is not None:
            payload_mutator(payload)
        merge_run_metadata(
            run_dir / "run_metadata.json",
            {
                "recommendation": payload,
                "artifacts": ["strategy_recommendation.md"],
            },
        )
        (run_dir / "strategy_recommendation.md").write_text(
            "# Strategy Recommendation\n",
            encoding="utf-8",
        )
        if post_write_hook is not None:
            post_write_hook(run_dir)
        state["run_dir"] = run_dir
        state["payload"] = payload
        return payload

    recommendation_module = types.SimpleNamespace(
        generate_recommendation=stub_recommendation
    )
    monkeypatch.setitem(sys.modules, "recommendation", recommendation_module)
    return state


def _relax_final_strategy_requirements(monkeypatch):
    monkeypatch.setitem(main.config.FINAL_STRATEGY, "MIN_CONFIDENCE_FOR_FINAL", 0)
    monkeypatch.setitem(main.config.FINAL_STRATEGY, "MIN_ASSET_CONSISTENCY", 0.0)
    monkeypatch.setitem(
        main.config.FINAL_STRATEGY, "INCLUDE_CLASSES", ["Stars", "Stalwarts"]
    )
    monkeypatch.setitem(main.config.FINAL_STRATEGY, "MAX_WEIGHT_CAP", 1.0)
    monkeypatch.setitem(main.config.FINAL_STRATEGY, "MIN_WEIGHT_FLOOR", 0.0)
    monkeypatch.setitem(main.config.FINAL_STRATEGY, "SHRINK_TO_EQUAL", 0.0)


def _run_main_with_walk_forward(
    monkeypatch,
    tmp_path,
    *,
    summary: dict | None = None,
    per_asset_rows: list[dict] | None = None,
    recommendation_mutator=None,
    recommendation_post_hook=None,
    omit_summary: bool = False,
    argv: list[str] | None = None,
):
    monkeypatch.chdir(tmp_path)
    _configure_minimal_main(monkeypatch)
    monkeypatch.setattr(
        main.config, "ENABLE_WALK_FORWARD_VALIDATION", True, raising=False
    )
    monkeypatch.setattr(
        main.config, "WALK_FORWARD_SETTINGS", {"enabled": True}, raising=False
    )
    _relax_final_strategy_requirements(monkeypatch)
    wf_state = _install_walk_forward_stub(
        monkeypatch,
        summary=summary,
        per_asset_rows=per_asset_rows,
        omit_summary=omit_summary,
    )
    recommendation_state = _install_recommendation_stub(
        monkeypatch,
        payload_mutator=recommendation_mutator,
        post_write_hook=recommendation_post_hook,
    )
    argv_list = [] if argv is None else list(argv)
    main.main(argv_list)
    run_dir = wf_state.get("run_dir")
    assert run_dir is not None
    return {
        "run_dir": Path(run_dir),
        "walk_forward": wf_state,
        "recommendation": recommendation_state,
    }


def test_main_runs(monkeypatch):
    _configure_minimal_main(monkeypatch)
    # Execute main and ensure no exception is raised
    main.main()


def test_main_generates_final_strategy(tmp_path, monkeypatch):
    result = _run_main_with_walk_forward(monkeypatch, tmp_path)
    run_dir = result["run_dir"]
    md_path = run_dir / "final_strategy.md"
    assert md_path.exists()
    meta = json.loads((run_dir / "run_metadata.json").read_text())
    assert "final_strategy" in meta
    assert "final_strategy.md" in meta.get("artifacts", [])
    digest = meta["artifacts_meta"]["final_strategy.md"]["sha256"]
    with md_path.open("rb") as fh:
        first_bytes = fh.read()
    assert hashlib.sha256(first_bytes).hexdigest() == digest
    assert meta["final_strategy"]["schema_version"] == "1.0"

    # Idempotency: rerun FSS and ensure the digest remains stable.
    final_strategy.generate_final_strategy({"run_dir": run_dir})
    meta_second = json.loads((run_dir / "run_metadata.json").read_text())
    digest_second = meta_second["artifacts_meta"]["final_strategy.md"]["sha256"]
    with md_path.open("rb") as fh:
        second_bytes = fh.read()
    assert hashlib.sha256(second_bytes).hexdigest() == digest_second
    assert digest_second == digest


def test_final_strategy_schema_version_guard(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.ERROR, logger=main.LOGGER.name)
    result = _run_main_with_walk_forward(
        monkeypatch,
        tmp_path,
        recommendation_mutator=lambda payload: payload.update(
            {"schema_version": "2.0"}
        ),
    )
    run_dir = result["run_dir"]
    assert not (run_dir / "final_strategy.md").exists()
    meta = json.loads((run_dir / "run_metadata.json").read_text())
    assert "final_strategy" not in meta
    artifacts_meta = meta.get("artifacts_meta", {})
    assert "final_strategy.md" not in artifacts_meta
    log_text = caplog.text
    assert "Final strategy synthesizer failed" in log_text
    assert (
        "Unsupported run_metadata.recommendation schema_version '2.0'; expected '1.0'"
        in log_text
    )


def test_final_strategy_requires_summary(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.ERROR, logger=main.LOGGER.name)
    result = _run_main_with_walk_forward(
        monkeypatch,
        tmp_path,
        omit_summary=True,
    )
    run_dir = result["run_dir"]
    summary_path = run_dir / "walk_forward" / "walk_forward_summary.json"
    assert not summary_path.exists()
    assert not (run_dir / "final_strategy.md").exists()
    meta = json.loads((run_dir / "run_metadata.json").read_text())
    assert "final_strategy" not in meta
    artifacts = meta.get("artifacts", [])
    assert "final_strategy.md" not in artifacts
    artifacts_meta = meta.get("artifacts_meta", {})
    assert "final_strategy.md" not in artifacts_meta
    log_text = caplog.text
    assert "Final strategy synthesizer failed for run at" in log_text
    assert "walk_forward_summary.json not found; run walk_forward.py first" in log_text


def test_final_strategy_markdown_snapshot(tmp_path, monkeypatch):
    result = _run_main_with_walk_forward(monkeypatch, tmp_path)
    run_dir = result["run_dir"]
    md_path = run_dir / "final_strategy.md"
    assert md_path.exists()
    md_text = md_path.read_text()
    expected_lines = [
        "# Final Strategy",
        "## Overview",
        "Confidence: High (80)",
        "Fold selection: Elite/Viable",
        "Recency weighting: disabled",
        (
            "Weighting scheme: risk_adjusted — weights ∝ (performance / volatility) × "
            "consistency (cap 1.00, floor 0.00)"
        ),
        "## Recommended Parameters",
        "| Gene | Value | Stability | Distribution |",
        "| --- | --- | --- | --- |",
        "| alpha | 10 | Stable |  10.000 ┤ 10.000 ┼ 10.000 ┼ 11.000 ┤  11.000 |",
        "## Asset Allocation",
        "| Ticker | Class | Performance | Consistency | Volatility | Weight |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
        "| AAA | Stars | 1.100 | 75.0% | 0.1000 | 1.0000 |",
        "| BBB | Stalwarts | 0.700 | 70.0% | 0.1000 | 0.0000 |",
        "| **Total** | | | | | 1.000000 |",
        "",
        (
            "Note: displayed weights are rounded for readability; "
            "the internal sum remains exactly 1.0."
        ),
        "### Derivation",
        "| Ticker | Raw Weight | Performance | Consistency | Volatility |",
        "| --- | ---: | ---: | ---: | ---: |",
        "| AAA | 825.000000 | 1.100 | 75.0% | 0.1000 |",
        "| BBB | 490.000000 | 0.700 | 70.0% | 0.1000 |",
        "## Excluded Assets",
        "- None",
        "",
        "## Exit Behaviour",
        "_No exit telemetry available._",
        "",
        "## Confidence & SRE Summary",
        "Inherited confidence: High (80).",
        (
            "FSS stability classifications use relative coefficient of variation "
            "(RCV; IQR/median) while SRE reports coefficient of variation (CoV), "
            "so labels may diverge."
        ),
        "## Notes",
        "No additional notes.",
        "## Configuration",
        "```json",
        "{",
        '  "ASSET_WEIGHTS_OVERRIDE": {},',
        '  "FOLD_DECAY_RATE": 0.0,',
        '  "INCLUDE_CLASSES": [',
        '    "Stars",',
        '    "Stalwarts"',
        "  ],",
        '  "MAX_WEIGHT_CAP": 1.0,',
        '  "MIN_ASSET_CONSISTENCY": 0.0,',
        '  "MIN_CONFIDENCE_FOR_FINAL": 0,',
        '  "MIN_WEIGHT_FLOOR": 0.0,',
        '  "MULTIMODAL_MIN_CLUSTER_WEIGHT": 0.2,',
        '  "MULTIMODAL_MIN_SEPARATION": 0.75,',
        '  "PARAM_RCV_DDOF": 0,',
        '  "PARAM_RCV_UNSTABLE": 0.5,',
        '  "PARAM_RCV_WATCHLIST": 0.35,',
        '  "PARAM_SENSITIVITY_THRESHOLD": 0.15,',
        '  "PARAM_VALUE_DECIMALS": {',
        '    "default": 3,',
        '    "sl_trailing_pct": 3,',
        '    "tp_pct_1": 3,',
        '    "tp_pct_2": 3,',
        '    "tp_pct_3": 3,',
        '    "tp_pct_4": 3,',
        '    "tp_trailing_pct": 3',
        "  },",
        '  "SHOW_PARAM_DISTS": true,',
        '  "SHOW_RECENCY_HALFLIFE": true,',
        '  "SHRINK_TO_EQUAL": 0.0,',
        '  "USE_RECENCY_WEIGHTING": false,',
        '  "WEIGHTING_SCHEME": "risk_adjusted",',
        '  "WEIGHT_SENSITIVITY_RATIO_THRESHOLD": 0.25,',
        '  "WEIGHT_SENSITIVITY_THRESHOLD": 0.05',
        "}",
        "```",
    ]
    expected_md = "\n".join(expected_lines) + "\n"
    assert md_text == expected_md


def test_final_strategy_skips_on_unreadable_metadata(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.ERROR, logger=main.LOGGER.name)

    def corrupt_metadata(run_dir: Path) -> None:
        (run_dir / "run_metadata.json").write_text("{not json", encoding="utf-8")

    result = _run_main_with_walk_forward(
        monkeypatch,
        tmp_path,
        recommendation_post_hook=corrupt_metadata,
    )
    run_dir = result["run_dir"]
    meta_path = run_dir / "run_metadata.json"
    assert meta_path.exists()
    assert meta_path.read_text() == "{not json"
    assert not (run_dir / "final_strategy.md").exists()
    log_text = caplog.text
    assert "run_metadata.json unreadable at" in log_text
    assert "Final strategy synthesizer failed for run at" not in log_text


def test_main_no_fss_flag(tmp_path, monkeypatch):
    result = _run_main_with_walk_forward(
        monkeypatch,
        tmp_path,
        argv=["--no-fss"],
    )
    run_dir = result["run_dir"]
    assert not (run_dir / "final_strategy.md").exists()
    meta = json.loads((run_dir / "run_metadata.json").read_text())
    assert "final_strategy" not in meta
    artifacts = meta.get("artifacts", [])
    assert "final_strategy.md" not in artifacts


def test_main_uses_tuner(monkeypatch):
    df = pd.DataFrame(
        {
            "Open": [1, 2, 3],
            "High": [1, 2, 3],
            "Low": [1, 2, 3],
            "Close": [1, 2, 3],
            "Volume": [100, 100, 100],
        },
        index=pd.date_range("2020-01-01", periods=3),
    )

    monkeypatch.setattr(main.data_loader, "get_data", lambda *a, **k: (df, "cache"))
    monkeypatch.setitem(main.config.MULTI_ASSET, "enabled", False)

    gene_space = [{"low": 0, "high": 1}]
    gene_map = {0: {"name": "x", "path": [], "type": float}}
    gene_types = [float]

    def parser_stub(*_args, **_kwargs):
        return gene_space, gene_map, gene_types

    monkeypatch.setattr(main, "parse_genes_from_config", parser_stub)

    captured = {}

    class DummyGA:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)
            self.num_generations = 1
            self.generations_completed = 1
            self.last_generation_fitness = [1.0]
            self.best_solutions_fitness = [1.0]

        def run(self):
            return None

        def best_solution(self, **kwargs):
            return [0], 1.0, None

    monkeypatch.setattr(main.pygad, "GA", DummyGA)
    monkeypatch.setattr(main.analysis, "run_champion_analysis", lambda *a, **k: None)
    monkeypatch.setattr(main, "ensure_real_vectorbt", lambda *a, **k: None)
    monkeypatch.setattr(
        main,
        "plt",
        types.SimpleNamespace(
            ion=lambda: None,
            plot=lambda *a, **k: None,
            gca=lambda: types.SimpleNamespace(
                get_legend_handles_labels=lambda: ([], [])
            ),
            legend=lambda *a, **k: None,
            xlabel=lambda *a, **k: None,
            ylabel=lambda *a, **k: None,
            title=lambda *a, **k: None,
            show=lambda *a, **k: None,
            savefig=lambda *a, **k: None,
            close=lambda *a, **k: None,
            subplots=lambda *a, **k: (
                types.SimpleNamespace(),
                types.SimpleNamespace(
                    plot=lambda *a, **k: None,
                    get_legend_handles_labels=lambda: ([], []),
                    legend=lambda *a, **k: None,
                    set_title=lambda *a, **k: None,
                    set_xlabel=lambda *a, **k: None,
                    set_ylabel=lambda *a, **k: None,
                ),
            ),
        ),
    )

    monkeypatch.setattr(
        main.config,
        "STRATEGY_RULES",
        {"entry_rules": {"combination_logic": "AND", "conditions": []}},
        raising=False,
    )

    class DummyEvaluator:
        def __init__(self, *a, **k):
            pass

        def __call__(self, *a, **k):
            return 1.0

    monkeypatch.setattr(main.fitness, "FitnessEvaluator", DummyEvaluator)

    monkeypatch.setattr(
        main.config, "ENABLE_WALK_FORWARD_VALIDATION", False, raising=False
    )
    monkeypatch.setattr(
        main.config, "WALK_FORWARD_SETTINGS", {"enabled": False}, raising=False
    )
    monkeypatch.setattr(
        main.config, "FITNESS_WEIGHTS", {"min_trades": 0}, raising=False
    )
    monkeypatch.setattr(main.config, "GA_NUM_GENERATIONS", 1, raising=False)
    monkeypatch.setattr(main.config, "GA_POPULATION_SIZE", 1, raising=False)
    monkeypatch.setattr(main.config, "GA_PARENTS_MATING", 1, raising=False)
    monkeypatch.setattr(main.config, "GA_MUTATION_NUM_GENES", 1, raising=False)
    monkeypatch.setattr(main.config, "AUTO_TUNE_ENABLED", True, raising=False)
    monkeypatch.setattr(
        main,
        "STRATEGY_RULES",
        {"entry_rules": {"combination_logic": "AND", "conditions": []}},
        raising=False,
    )
    monkeypatch.setattr(
        main.config,
        "STRATEGY_RULES",
        {"entry_rules": {"combination_logic": "AND", "conditions": []}},
        raising=False,
    )
    train_period = {"start": "2020-01-01", "end": "2020-01-02"}
    valid_period = {"start": "2020-01-02", "end": "2020-01-03"}
    monkeypatch.setattr(main.config, "TRAINING_PERIOD", train_period, raising=False)
    monkeypatch.setattr(main.config, "VALIDATION_PERIOD", valid_period, raising=False)
    monkeypatch.setattr(main.config, "SELECTED_ASSET_NAME", "Test", raising=False)
    monkeypatch.setattr(main.config, "TICKER", "TEST", raising=False)
    monkeypatch.setattr(main.config, "TIMEFRAME", "1d", raising=False)

    tuned_params = {
        "sol_per_pop": 3,
        "num_parents_mating": 2,
        "mutation_num_genes": 1,
    }
    monkeypatch.setattr(
        main.tuner,
        "find_best_hyperparameters",
        lambda *a, **k: tuned_params,
    )

    main.main()

    assert captured["sol_per_pop"] == 3
    assert captured["num_parents_mating"] == 2
    assert captured["mutation_num_genes"] == 1


@pytest.mark.skip(reason="skipped after enabling additional entry indicators")
def test_fitness_plot_non_blocking(monkeypatch):
    df = pd.DataFrame(
        {
            "Open": [1, 2],
            "High": [1, 2],
            "Low": [1, 2],
            "Close": [1, 2],
            "Volume": [100, 100],
        },
        index=pd.date_range("2020-01-01", periods=2),
    )

    monkeypatch.setattr(main.data_loader, "get_data", lambda *a, **k: (df, "cache"))
    monkeypatch.setitem(main.config.MULTI_ASSET, "enabled", False)

    gene_space = [{"low": 0, "high": 1}]
    gene_map = {0: {"name": "x", "path": [], "type": float}}
    gene_types = [float]

    monkeypatch.setattr(
        main,
        "parse_genes_from_config",
        lambda *a, **k: (gene_space, gene_map, gene_types),
    )

    events = {}

    class DummyGA:
        def __init__(self, *args, **kwargs):
            self.num_generations = 1
            self.generations_completed = 1
            self.last_generation_fitness = [1.0]
            self.best_solutions_fitness = [1.0]

        def run(self):
            return None

        def best_solution(self, **kwargs):
            return [0], 1.0, None

    monkeypatch.setattr(main.pygad, "GA", DummyGA)
    monkeypatch.setattr(main.analysis, "run_champion_analysis", lambda *a, **k: None)
    monkeypatch.setattr(main, "ensure_real_vectorbt", lambda *a, **k: None)

    class DummyEvaluator:
        def __init__(self, *a, **k):
            pass

        def __call__(self, *a, **k):
            return 1.0

    monkeypatch.setattr(main.fitness, "FitnessEvaluator", DummyEvaluator)

    monkeypatch.setattr(
        main.config, "ENABLE_WALK_FORWARD_VALIDATION", False, raising=False
    )
    monkeypatch.setattr(
        main.config, "WALK_FORWARD_SETTINGS", {"enabled": False}, raising=False
    )
    monkeypatch.setattr(
        main.config, "FITNESS_WEIGHTS", {"min_trades": 0}, raising=False
    )
    monkeypatch.setattr(main.config, "GA_NUM_GENERATIONS", 1, raising=False)
    monkeypatch.setattr(main.config, "GA_POPULATION_SIZE", 1, raising=False)
    monkeypatch.setattr(main.config, "GA_PARENTS_MATING", 1, raising=False)
    monkeypatch.setattr(main.config, "GA_MUTATION_NUM_GENES", 1, raising=False)
    train_period = {"start": "2020-01-01", "end": "2020-01-02"}
    valid_period = {"start": "2020-01-02", "end": "2020-01-03"}
    monkeypatch.setattr(main.config, "TRAINING_PERIOD", train_period, raising=False)
    monkeypatch.setattr(main.config, "VALIDATION_PERIOD", valid_period, raising=False)
    monkeypatch.setattr(main.config, "SELECTED_ASSET_NAME", "Test", raising=False)
    monkeypatch.setattr(main.config, "TICKER", "TEST", raising=False)
    monkeypatch.setattr(main.config, "TIMEFRAME", "1d", raising=False)

    class FakePlt:
        def ion(self):
            events["ion"] = True

        def plot(self, *a, **k):
            events["plot_called"] = True

        def gca(self):
            return types.SimpleNamespace(get_legend_handles_labels=lambda: ([], []))

        def legend(self, *a, **k):
            pass

        def xlabel(self, *a, **k):
            pass

        def ylabel(self, *a, **k):
            pass

        def title(self, *a, **k):
            pass

        def show(self, *a, **k):
            pass

        def savefig(self, *a, **k):
            pass

        def close(self, *a, **k):
            pass

        def subplots(self, *a, **k):
            def _plot(*a, **k):
                events["plot_called"] = True

            return (
                types.SimpleNamespace(),
                types.SimpleNamespace(
                    plot=_plot,
                    get_legend_handles_labels=lambda: ([], []),
                    legend=lambda *a, **k: None,
                    set_title=lambda *a, **k: None,
                    set_xlabel=lambda *a, **k: None,
                    set_ylabel=lambda *a, **k: None,
                ),
            )

    monkeypatch.setattr(main, "plt", FakePlt())

    main.main()


def test_ga_banner_reports_repaired_tp_levels(monkeypatch, capsys):
    _configure_minimal_main(monkeypatch)

    exit_rules = {
        "stop_loss": {
            "is_active": True,
            "type": "percentage",
            "params": {"value": {"gene": "stop_loss_pct", "low": 0.01, "high": 0.05}},
        },
        "trade_management": {
            "num_tp_levels": {"gene": "num_tp_levels", "low": 1, "high": 4, "step": 1},
            "tp_pct_1": {"gene": "tp_pct_1", "low": 0.01, "high": 0.5, "step": 0.005},
            "tp_pct_2": {"gene": "tp_pct_2", "low": 0.01, "high": 0.5, "step": 0.005},
            "tp_pct_3": {"gene": "tp_pct_3", "low": 0.01, "high": 0.5, "step": 0.005},
            "tp_pct_4": {"gene": "tp_pct_4", "low": 0.01, "high": 0.5, "step": 0.005},
        },
    }
    strategy = {
        "entry_rules": {"combination_logic": "AND", "conditions": []},
        "exit_rules": exit_rules,
    }
    monkeypatch.setattr(main, "STRATEGY_RULES", strategy, raising=False)
    monkeypatch.setattr(main.config, "STRATEGY_RULES", strategy, raising=False)

    gene_space = [
        {"low": 1, "high": 4, "step": 1},
        {"low": 0.01, "high": 0.5, "step": 0.005},
        {"low": 0.01, "high": 0.5, "step": 0.005},
        {"low": 0.01, "high": 0.5, "step": 0.005},
        {"low": 0.01, "high": 0.5, "step": 0.005},
    ]
    gene_map = {
        0: {
            "name": "num_tp_levels",
            "path": ["exit_rules", "trade_management", "num_tp_levels"],
            "type": int,
        },
        1: {
            "name": "tp_pct_1",
            "path": ["exit_rules", "trade_management", "tp_pct_1"],
            "type": float,
        },
        2: {
            "name": "tp_pct_2",
            "path": ["exit_rules", "trade_management", "tp_pct_2"],
            "type": float,
        },
        3: {
            "name": "tp_pct_3",
            "path": ["exit_rules", "trade_management", "tp_pct_3"],
            "type": float,
        },
        4: {
            "name": "tp_pct_4",
            "path": ["exit_rules", "trade_management", "tp_pct_4"],
            "type": float,
        },
    }
    gene_types = [int, float, float, float, float]

    def parse_stub(*_args, **_kwargs):
        return gene_space, gene_map, gene_types

    monkeypatch.setattr(main, "parse_genes_from_config", parse_stub)

    solution = [4, 0.02, 0.02, 0.02, 0.02]

    class BannerGA:
        def __init__(self, *args, **kwargs):
            self.num_generations = 1
            self.generations_completed = 1
            self.last_generation_fitness = [1.0]
            self.best_solutions_fitness = [1.0]

        def run(self):
            return None

        def best_solution(self, **kwargs):
            return solution, 1.23, None

    monkeypatch.setattr(main.pygad, "GA", BannerGA)

    main.main([])
    output = capsys.readouterr().out
    assert "tp_pct_2: 0.025" in output
    assert "tp_pct_3: 0.03" in output
    assert "tp_pct_4: 0.035" in output
