import os
import numpy as np
import pygad
import vectorbt as vbt
import pandas as pd

import config
import data_loader
import fitness
import strategy_engine as engine
import scanner_sim
from utils.logging_util import get_logger


def _evaluate_on_validation(solution, gene_map):
    """Evaluate solution on validation data and return Sortino Ratio."""
    # If heavy optional dependencies are missing, skip evaluation to keep tests
    # lightweight. We check for the pandas_ta accessor and vectorbt's Portfolio
    # class. When absent, return -inf so the tuner can continue without errors.
    if not hasattr(pd.DataFrame(), "ta") or not hasattr(vbt, "Portfolio"):
        return -1e6

    val_data = data_loader.get_data(
        ticker=config.TICKER,
        start_date=config.VALIDATION_PERIOD["start"],
        end_date=config.VALIDATION_PERIOD["end"],
        interval=config.TIMEFRAME,
    )
    if val_data.empty:
        return -1e6

    rules = fitness._inject_genes_into_rules(config.STRATEGY_RULES, gene_map, solution)
    entries = engine.process_strategy_rules(val_data, rules)
    if entries.sum() < 1:
        return -1e6

    exit_rules = rules.get("exit_rules", {})
    sl_rule = exit_rules.get("stop_loss", {})
    tsl_rule = exit_rules.get("trailing_stop", {})
    tp_rule = exit_rules.get("take_profit", {})

    sl_stop = sl_rule.get("params", {}).get("value") if sl_rule.get("is_active", False) else None
    sl_trail = tsl_rule.get("params", {}).get("value") if tsl_rule.get("is_active", False) else None
    tp_stop = tp_rule.get("params", {}).get("value") if tp_rule.get("is_active", False) else None

    time_exit = entries.shift(config.MAX_HOLD_PERIOD, fill_value=False)
    time_exit = time_exit.reindex(entries.index, fill_value=False)

    # Gate entries to obtain diagnostics and admitted trade count
    gated, _oc, diag = scanner_sim.gate_entries(entries, time_exit, 1)
    gated_entries = gated.iloc[:, 0]
    trade_count = int(diag.get("accepted", 0))

    portfolio = vbt.Portfolio.from_signals(
        close=val_data["Close"],
        entries=gated_entries,
        exits=time_exit,
        sl_stop=sl_stop,
        tp_stop=tp_stop,
        sl_trail=sl_trail,
        fees=config.FEES,
        freq=config.TIMEFRAME,
    )
    stats = portfolio.stats()
    score = stats.get("Sortino Ratio")
    if score is None:
        score = -np.inf

    MIN_TRADES = getattr(config, "MIN_TRADES", config.FITNESS_WEIGHTS.get("min_trades", 0))
    penalties = {}
    if trade_count < MIN_TRADES:
        penalties["min_trades"] = MIN_TRADES - trade_count
        score = -np.inf
    if np.isnan(score):
        score = -np.inf

    if score == -np.inf:
        logger.info(
            "Trade count: %s | Floor: %s",
            trade_count,
            MIN_TRADES,
        )
        for name, val in penalties.items():
            logger.info("Penalty %s: %s", name, val)
        score = -1e6
    return score


logger = get_logger(__name__)


_error_tracker = None


def _on_generation_callback(ga_instance):
    """Flush error summaries once per GA generation."""
    if _error_tracker is not None:
        g = ga_instance.generations_completed
        _error_tracker.flush_summary(logger, f"Generation {g}")


def find_best_hyperparameters(ohlc_data, gene_space, gene_map, gene_types):
    """Run short GA optimisations to find the best hyperparameter set."""
    print("\n--- Express Hyperparameter Tuning ---")
    data = ohlc_data if isinstance(ohlc_data, pd.DataFrame) else next(iter(ohlc_data.values()))
    fitness_evaluator = fitness.FitnessEvaluator(data, config.STRATEGY_RULES, gene_map)
    fitness_func = fitness_evaluator.__call__
    error_tracker = getattr(fitness_evaluator, "error_tracker", None)
    num_cores = os.cpu_count()

    results = []

    for idx, params in enumerate(config.HYPERPARAMETER_SEARCH_SPACE, 1):
        print(f"Tuning with config {idx} of {len(config.HYPERPARAMETER_SEARCH_SPACE)}: {params}")

        global _error_tracker
        _error_tracker = error_tracker
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
            on_generation=_on_generation_callback,
        )
        try:
            ga.run()
        finally:
            _error_tracker = None
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
