import os
import numpy as np
import pygad
import vectorbt as vbt
import pandas as pd

import config
import data_loader
import fitness
import strategy_engine as engine


def _evaluate_on_validation(solution, gene_map):
    """Evaluate solution on validation data and return Sortino Ratio."""
    # Skip evaluation if heavy optional dependencies are missing
    if not hasattr(pd.DataFrame(), "ta") or not hasattr(vbt, "Portfolio"):
        return -np.inf

    ticker = (
        config.TUNING_ASSET
        if getattr(config, "PORTFOLIO_OPTIMIZATION_ENABLED", False)
        else config.TICKER
    )

    val_data = data_loader.get_data(
        ticker=ticker,
        start_date=config.VALIDATION_PERIOD["start"],
        end_date=config.VALIDATION_PERIOD["end"],
        interval=config.TIMEFRAME,
    )
    if val_data.empty:
        return -np.inf

    rules = fitness._inject_genes_into_rules(config.STRATEGY_RULES, gene_map, solution)
    entries = engine.process_strategy_rules(val_data, rules)
    if fitness._count_trades(entries) < 1:
        return -np.inf

    exit_rules = rules.get("exit_rules", {})
    sl_rule = exit_rules.get("stop_loss", {})
    tsl_rule = exit_rules.get("trailing_stop", {})
    tp_rule = exit_rules.get("take_profit", {})

    sl_stop = sl_rule.get("params", {}).get("value") if sl_rule.get("is_active", False) else None
    sl_trail = tsl_rule.get("params", {}).get("value") if tsl_rule.get("is_active", False) else None
    tp_stop = tp_rule.get("params", {}).get("value") if tp_rule.get("is_active", False) else None

    _, _, agg_stats, _ = fitness.run_portfolio_backtest(
        val_data,
        entries,
        sl_stop=sl_stop,
        sl_trail=sl_trail,
        tp_stop=tp_stop,
        weights=getattr(config, "PORTFOLIO_WEIGHTS", None),
    )
    score = agg_stats.get("Sortino Ratio") if not isinstance(agg_stats, pd.DataFrame) else agg_stats.loc["Sortino Ratio"].iloc[0]
    return -np.inf if np.isnan(score) else score


def find_best_hyperparameters(gene_space, gene_map, gene_types):
    """Run short GA optimisations to find the best hyperparameter set.

    When portfolio optimisation is disabled the function loads ``config.TICKER``.
    If portfolio mode is enabled only ``config.TUNING_ASSET`` is used so the
    tuning phase stays lightweight.
    """
    print("\n--- Express Hyperparameter Tuning ---")

    tuning_ticker = (
        config.TUNING_ASSET
        if getattr(config, "PORTFOLIO_OPTIMIZATION_ENABLED", False)
        else config.TICKER
    )
    ohlc_data = data_loader.get_data(
        ticker=tuning_ticker,
        start_date=config.TRAINING_PERIOD["start"],
        end_date=config.TRAINING_PERIOD["end"],
        interval=config.TIMEFRAME,
    )
    fitness_evaluator = fitness.FitnessEvaluator(ohlc_data, config.STRATEGY_RULES, gene_map)
    fitness_func = fitness_evaluator.__call__
    num_cores = os.cpu_count()

    results = []

    for idx, params in enumerate(config.HYPERPARAMETER_SEARCH_SPACE, 1):
        print(f"Tuning with config {idx} of {len(config.HYPERPARAMETER_SEARCH_SPACE)}: {params}")
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
        )
        ga.run()
        best_solution, _, _ = ga.best_solution()
        score = _evaluate_on_validation(best_solution, gene_map)
        results.append({"params": params, "score": score})
        print(f"Validation score: {score}")

    print("\n-- Tuning Summary --")
    for r in results:
        print(f"{r['params']} => {r['score']}")

    best = max(results, key=lambda x: x["score"]) if results else {"params": None}
    print(f"Best hyperparameters found: {best['params']}")
    return best["params"]
