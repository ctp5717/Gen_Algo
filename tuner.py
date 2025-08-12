import os
from typing import Dict, List
import numpy as np
import pandas as pd
import pygad
import config
import data_loader
import fitness

def _eval_on_validation(best_solution: List[float], val_data: pd.DataFrame, base_rules: dict, gene_map: Dict[int, dict]) -> float:
    evaluator = fitness.FitnessEvaluator(val_data, base_rules, gene_map)
    return evaluator(None, best_solution, 0)

def find_best_hyperparameters(gene_space, gene_types, gene_map) -> Dict[str, int]:
    # If portfolio mode enabled, tune on single TUNING_ASSET for speed
    if getattr(config, 'PORTFOLIO_OPTIMIZATION_ENABLED', False):
        tune_ticker = getattr(config, 'TUNING_ASSET', config.TICKER)
        train_data = data_loader.get_data(tune_ticker, config.TRAINING_PERIOD['start'], config.TRAINING_PERIOD['end'], config.TIMEFRAME)
        val_data = data_loader.get_data(tune_ticker, config.VALIDATION_PERIOD['start'], config.VALIDATION_PERIOD['end'], config.TIMEFRAME)
    else:
        train_data = data_loader.get_data(config.TICKER, config.TRAINING_PERIOD['start'], config.TRAINING_PERIOD['end'], config.TIMEFRAME)
        val_data = data_loader.get_data(config.TICKER, config.VALIDATION_PERIOD['start'], config.VALIDATION_PERIOD['end'], config.TIMEFRAME)

    best = None
    best_score = -np.inf
    for opt in config.HYPERPARAMETER_SEARCH_SPACE[:]:
        ga = pygad.GA(
            num_generations=min(config.GENERATIONS_PER_TUNE, 10),
            num_parents_mating=opt.get('num_parents_mating', 20),
            sol_per_pop=opt.get('sol_per_pop', 50),
            mutation_num_genes=opt.get('mutation_num_genes', 1),
            num_genes=len(gene_space),
            gene_space=gene_space,
            gene_type=gene_types,
            fitness_func=fitness.FitnessEvaluator(train_data, config.STRATEGY_RULES, gene_map).__call__,
            parallel_processing=['process', max(1, os.cpu_count() or 1)],
        )
        ga.run()
        sol, train_score, _ = ga.best_solution()
        val_score = _eval_on_validation(sol, val_data, config.STRATEGY_RULES, gene_map)
        if val_score > best_score:
            best_score = val_score
            best = opt
    return best or {"sol_per_pop": 50, "num_parents_mating": 20, "mutation_num_genes": 1}
