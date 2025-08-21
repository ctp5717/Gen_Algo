import os
from datetime import datetime
from pathlib import Path
import numpy as np
import pygad
import vectorbt as vbt
import pandas as pd

import config
import data_loader
import fitness
import strategy_engine as engine
import analysis
from utils import _norm_freq, set_global_seed


def _evaluate_on_validation(solution, gene_map):
    """Evaluate solution on validation data and return the objective score."""
    # If heavy optional dependencies are missing, skip evaluation to keep tests
    # lightweight. We check for the pandas_ta accessor and vectorbt's Portfolio
    # class. When absent, return -inf so the tuner can continue without errors.
    if not hasattr(pd.DataFrame(), "ta") or not hasattr(vbt, "Portfolio"):
        return -np.inf

    if getattr(config, "MULTI_ASSET", {}).get("enabled"):
        val_data = data_loader.get_group_data(
            asset_group=config.ASSET_GROUP,
            start_date=config.VALIDATION_PERIOD["start"],
            end_date=config.VALIDATION_PERIOD["end"],
            interval=config.TIMEFRAME,
            coverage_threshold=config.COVERAGE_THRESHOLD,
        )
        if not val_data:
            return -np.inf
        evaluator = fitness.MultiAssetFitnessEvaluator(
            val_data,
            config.STRATEGY_RULES,
            gene_map,
            config.MULTI_ASSET,
        )
        return evaluator(None, solution, 0)

    val_data = data_loader.get_data(
        ticker=config.TICKER,
        start_date=config.VALIDATION_PERIOD["start"],
        end_date=config.VALIDATION_PERIOD["end"],
        interval=config.TIMEFRAME,
    )
    if val_data.empty:
        return -np.inf

    rules = fitness._inject_genes_into_rules(config.STRATEGY_RULES, gene_map, solution)
    entries = engine.process_strategy_rules(val_data, rules)
    if entries.sum() < 1:
        return -np.inf

    exit_rules = rules.get("exit_rules", {})
    sl_rule = exit_rules.get("stop_loss", {})
    tsl_rule = exit_rules.get("trailing_stop", {})
    tp_rule = exit_rules.get("take_profit", {})

    sl_stop = sl_rule.get("params", {}).get("value") if sl_rule.get("is_active", False) else None
    sl_trail = tsl_rule.get("params", {}).get("value") if tsl_rule.get("is_active", False) else None
    tp_stop = tp_rule.get("params", {}).get("value") if tp_rule.get("is_active", False) else None

    time_exit = entries.shift(config.MAX_HOLD_PERIOD, fill_value=False)
    time_exit = time_exit.reindex(entries.index, fill_value=False)

    portfolio = vbt.Portfolio.from_signals(
        close=val_data["Close"],
        entries=entries,
        exits=time_exit,
        sl_stop=sl_stop,
        tp_stop=tp_stop,
        sl_trail=sl_trail,
        fees=0.001,
        freq=_norm_freq(config.TIMEFRAME),
    )
    stats = portfolio.stats()
    score = stats.get("Sortino Ratio")
    return -np.inf if np.isnan(score) else score


_on_gen_best: dict | None = None
_on_gen_eval = None
_on_gen_func = None


def _on_generation_cb(ga_instance):
    """Module-level callback used by ``_make_on_generation``.

    Defining the callback at the module level keeps it picklable when the
    :class:`pygad.GA` instance is sent to worker processes during parallel
    fitness evaluation. Any state required by the callback is stored in module
    globals which are initialised by ``_make_on_generation``.
    """

    if _on_gen_best is None:
        return

    best_sol, fit, _ = ga_instance.best_solution(
        pop_fitness=ga_instance.last_generation_fitness
    )
    if fit > _on_gen_best["fitness"]:
        _on_gen_best["fitness"] = fit
        try:
            _on_gen_func(None, best_sol, 0)
            details = getattr(_on_gen_eval, "last_details", {})
            if details:
                charts_cfg = getattr(config, "CHARTS", {})
                run_ts = charts_cfg.get("run_ts")
                if not run_ts:
                    run_ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                    charts_cfg["run_ts"] = run_ts
                out_path = Path("reports") / run_ts / "tune_extremes.json"
                analysis.log_asset_extremes(
                    details,
                    save_path=out_path,
                    quiet=True,
                )
        except Exception:
            pass


def _make_on_generation(fitness_eval, fitness_func):
    """Create a picklable ``on_generation`` callback that logs asset extremes."""

    global _on_gen_best, _on_gen_eval, _on_gen_func
    _on_gen_best = {"fitness": -float("inf")}
    _on_gen_eval = fitness_eval
    _on_gen_func = fitness_func
    return _on_generation_cb


def find_best_hyperparameters(ohlc_data, gene_space, gene_map, gene_types):
    """Run short GA optimisations to find the best hyperparameter set."""
    print("\n--- Express Hyperparameter Tuning ---")
    seed = None
    if getattr(config, "DETERMINISTIC", False):
        seed = getattr(config, "RANDOM_SEED", 42)
        set_global_seed(seed)
        print(f"Deterministic mode enabled. Seed={seed}")
    fitness_evaluator = fitness.get_fitness_evaluator(
        ohlc_data, config.STRATEGY_RULES, gene_map
    )
    evaluator_name = type(fitness_evaluator).__name__
    objective = getattr(fitness_evaluator, "settings", {}).get("metric", "composite")
    print(f"Active evaluator: {evaluator_name} | Objective: {objective}")
    assert objective, "Objective must be defined"
    fitness_func = fitness_evaluator.__call__
    num_cores = os.cpu_count()

    results = []

    original_lambda = getattr(config, "MULTI_ASSET", {}).get("lambda_dispersion")

    for idx, params in enumerate(config.HYPERPARAMETER_SEARCH_SPACE, 1):
        print(f"Tuning with config {idx} of {len(config.HYPERPARAMETER_SEARCH_SPACE)}: {params}")
        lam = params.get("lambda_dispersion")
        if lam is not None and hasattr(config, "MULTI_ASSET"):
            config.MULTI_ASSET["lambda_dispersion"] = lam
        ga = pygad.GA(
            num_generations=config.GENERATIONS_PER_TUNE,
            num_parents_mating=params["num_parents_mating"],
            sol_per_pop=params["sol_per_pop"],
            num_genes=len(gene_space),
            gene_space=gene_space,
            gene_type=list(gene_types),
            mutation_num_genes=params["mutation_num_genes"],
            fitness_func=fitness_func,
            parallel_processing=["process", num_cores],
            random_seed=seed,
            on_generation=_make_on_generation(fitness_evaluator, fitness_func),
        )
        ga.run()
        analysis.log_asset_extremes(getattr(fitness_evaluator, "last_details", None))
        analysis.persist_details(fitness_evaluator)
        best_solution, _, _ = ga.best_solution()
        score = _evaluate_on_validation(best_solution, gene_map)
        results.append({"params": params, "score": score})
        print(f"Validation score: {score}")

    if original_lambda is not None and hasattr(config, "MULTI_ASSET"):
        config.MULTI_ASSET["lambda_dispersion"] = original_lambda

    print("\n-- Tuning Summary --")
    for r in results:
        print(f"{r['params']} => {r['score']}")

    best = max(results, key=lambda x: x["score"]) if results else {"params": None}
    print(f"Best hyperparameters found: {best['params']}")
    return best["params"]
