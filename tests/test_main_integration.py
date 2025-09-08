import sys
import types
from pathlib import Path

import pandas as pd

import deps
import vbt_stub

# Ensure repository root is on the import path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Stub heavy optional dependencies
sys.modules.setdefault("pandas_ta", types.ModuleType("pandas_ta"))
sys.modules.setdefault("vectorbt", vbt_stub)
sys.modules.setdefault("yfinance", types.ModuleType("yfinance"))

import main  # noqa: E402


def test_main_runs(monkeypatch):
    # Provide minimal OHLC data
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

    # Patch data loader to avoid network requests
    monkeypatch.setattr(main.data_loader, "get_data", lambda *a, **k: (df, "cache"))
    monkeypatch.setitem(main.config.MULTI_ASSET, "enabled", False)

    # Patch gene parser to return a single gene definition
    gene_space = [{"low": 0, "high": 1}]
    gene_map = {0: {"name": "x", "path": [], "type": float}}
    gene_types = [float]

    def parser_stub(*_args, **_kwargs):
        return gene_space, gene_map, gene_types

    monkeypatch.setattr(main, "parse_genes_from_config", parser_stub)

    # Dummy GA class to bypass heavy optimisation
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

    # Patch analysis and fitness evaluator
    monkeypatch.setattr(main.analysis, "run_champion_analysis", lambda *a, **k: None)
    monkeypatch.setattr(main, "ensure_real_vectorbt", lambda *a, **k: None)
    monkeypatch.setattr(main, "ensure_pandas_ta", lambda: None)
    monkeypatch.setattr(deps, "ensure_pandas_ta", lambda: None)
    monkeypatch.setattr(main.run_metadata, "_write_run_metadata", lambda *a, **k: None)
    monkeypatch.setattr(
        main.analysis, "_write_run_metadata", lambda *a, **k: None, raising=False
    )

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

    # Simplify config to avoid walk forward validation and reduce parameters
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
    # Execute main and ensure no exception is raised
    main.main()


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
    monkeypatch.setattr(main, "ensure_pandas_ta", lambda: None)
    monkeypatch.setattr(deps, "ensure_pandas_ta", lambda: None)
    monkeypatch.setattr(main.run_metadata, "_write_run_metadata", lambda *a, **k: None)
    monkeypatch.setattr(
        main.analysis, "_write_run_metadata", lambda *a, **k: None, raising=False
    )
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
    monkeypatch.setattr(main, "ensure_pandas_ta", lambda: None)
    monkeypatch.setattr(deps, "ensure_pandas_ta", lambda: None)
    monkeypatch.setattr(main.run_metadata, "_write_run_metadata", lambda *a, **k: None)
    monkeypatch.setattr(
        main.analysis, "_write_run_metadata", lambda *a, **k: None, raising=False
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
    monkeypatch.setattr(
        main.config,
        "STRATEGY_RULES",
        {"entry_rules": {"combination_logic": "AND", "conditions": []}},
        raising=False,
    )

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

    assert events["ion"]
    assert events["plot_called"]
