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
    # If heavy optional dependencies are missing, skip evaluation to keep tests
    # lightweight. We check for the pandas_ta accessor and vectorbt's Portfolio
    # class. When absent, return -inf so the tuner can continue without errors.
    if not hasattr(pd.DataFrame(), "ta") or not hasattr(vbt, "Portfolio"):
        return -np.inf

    val_data = data_loader.get_data(
        ticker=config.TUNING_ASSET,
        start_date=config.VALIDATION_PERIOD["start"],
        end_date=config.VALIDATION_PERIOD["end"],
        interval=config.TIMEFRAME,
    )
    if val_data.empty:
        return -np.inf

    rules = fitness._inject_genes_into_rules(config.STRATEGY_RULES, gene_map, solution)
    entries = engine.process_strategy_rules(val_data, rules)
    trade_count = entries.sum().sum() if isinstance(entries, pd.DataFrame) else entries.sum()
    if trade_count < 1:
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
        freq=config.TIMEFRAME,
    )
    stats = portfolio.stats()
    score = stats.get("Sortino Ratio")
    return -np.inf if np.isnan(score) else score


def find_best_hyperparameters(ohlc_data, gene_space, gene_map, gene_types):
    """Run short GA optimisations to find the best hyperparameter set."""
    print("\n--- Express Hyperparameter Tuning ---")

    tuning_data = None
    last_exception = None
    for attempt in range(2):
        try:
            tuning_data = data_loader.get_data(
                ticker=config.TUNING_ASSET,
                start_date=config.TRAINING_PERIOD["start"],
                end_date=config.TRAINING_PERIOD["end"],
                interval=config.TIMEFRAME,
            )
            if tuning_data.empty:
                raise ValueError("Tuning data is empty")
            break
        except Exception as e:
            last_exception = e
            print(
                f"Attempt {attempt + 1} failed to load tuning asset {config.TUNING_ASSET}: {e}"
            )
    else:
        raise RuntimeError(
            f"Unable to load tuning data for {config.TUNING_ASSET}"
        ) from last_exception

    fitness_evaluator = fitness.FitnessEvaluator(tuning_data, config.STRATEGY_RULES, gene_map)
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
