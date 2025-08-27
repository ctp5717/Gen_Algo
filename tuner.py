import os
import numpy as np
import pygad
import vectorbt as vbt
import pandas as pd
import trade_floor

import config
import fitness
import strategy_engine as engine


def _evaluate_on_validation(solution, gene_map, val_data):
    """Evaluate solution on preloaded validation data and return the score."""
    # Skip evaluation gracefully if optional heavy dependencies are missing.
    if not hasattr(pd.DataFrame(), "ta") or not hasattr(vbt, "Portfolio"):
        return -np.inf

    if getattr(config, "MULTI_ASSET", {}).get("enabled"):
        if not val_data:
            return -np.inf
        settings = dict(config.MULTI_ASSET)
        rate = settings.get("min_total_trades_per_year")
        if rate:
            start = pd.to_datetime(config.VALIDATION_PERIOD["start"])
            end = pd.to_datetime(config.VALIDATION_PERIOD["end"])
            floor, info = trade_floor.scale_floor(rate, start, end)
            settings["min_total_trades"] = floor
            print(
                f"Scaled min_total_trades (validation): {floor} | info={info}"
            )
        settings["trade_floor_policy"] = "soft_penalty"
        settings["soft_penalty_mode"] = "multiplicative"
        print(
            "Tuner: using trade_floor_policy=soft_penalty (multiplicative) for validation."
        )
        evaluator = fitness.MultiAssetFitnessEvaluator(
            val_data, config.STRATEGY_RULES, gene_map, settings
        )
        return evaluator(None, solution, 0)

    if val_data is None or val_data.empty:
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
        fees=config.FEES,
        freq=config.to_pandas_freq(config.TIMEFRAME),
    )
    stats = portfolio.stats()
    score = stats.get("Sortino Ratio")
    return -np.inf if np.isnan(score) else score


def find_best_hyperparameters(train_data, gene_space, gene_map, gene_types, val_data):
    """Run short GA optimisations using preloaded data."""
    print("\n--- Express Hyperparameter Tuning ---")
    np.random.seed(config.SEED)

    # Optional coarse tuning of lambda dispersion
    if getattr(config, "MULTI_ASSET", {}).get("enabled"):
        lam_grid = config.MULTI_ASSET.get("lambda_grid")
        if lam_grid:
            print("\n-- Lambda Dispersion Grid --")
            best_lam = None
            best_score = -np.inf
            for lam in lam_grid:
                settings = dict(config.MULTI_ASSET)
                settings["lambda_dispersion"] = lam
                settings["trade_floor_policy"] = "soft_penalty"
                settings["soft_penalty_mode"] = "multiplicative"
                evaluator = fitness.MultiAssetFitnessEvaluator(
                    train_data, config.STRATEGY_RULES, gene_map, settings
                )
                probe = pygad.GA(
                    num_generations=1,
                    num_parents_mating=2,
                    sol_per_pop=4,
                    num_genes=len(gene_space),
                    gene_space=gene_space,
                    gene_type=list(gene_types),
                    mutation_num_genes=1,
                    fitness_func=evaluator.__call__,
                    random_seed=config.SEED,
                )
                probe.run()
                _, score, _ = probe.best_solution()
                print(f"λ={lam}: {score}")
                if score > best_score:
                    best_score = score
                    best_lam = lam
            if best_lam is not None:
                config.MULTI_ASSET["lambda_dispersion"] = best_lam
                print(f"Selected λ={best_lam}")

    fitness_evaluator = fitness.get_fitness_evaluator(
        train_data, config.STRATEGY_RULES, gene_map
    )
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
            random_seed=config.SEED,
        )
        ga.run()
        best_solution, _, _ = ga.best_solution()
        score = _evaluate_on_validation(best_solution, gene_map, val_data)
        results.append({"params": params, "score": score})
        print(f"Validation score: {score}")

    print("\n-- Tuning Summary --")
    for r in results:
        print(f"{r['params']} => {r['score']}")

    best = max(results, key=lambda x: x["score"]) if results else {"params": None}
    print(f"Best hyperparameters found: {best['params']}")
    return best["params"]
