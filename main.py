import os
import pprint
import time
import traceback
from typing import List
import matplotlib.pyplot as plt
import pygad

import config
import data_loader
import fitness
import analysis
from gene_parser import parse_genes_from_config
import tuner

start_time: float = 0.0

def on_generation(ga_instance: pygad.GA) -> None:
    generation = ga_instance.generations_completed
    total_generations = ga_instance.num_generations
    fitness_score = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]
    elapsed = time.time() - start_time
    remaining = (elapsed / generation) * (total_generations - generation) if generation > 0 else 0
    print(f"Generation {generation}/{total_generations} | Best Fitness: {fitness_score:.4f} | Est. Time Left: {int(remaining)}s", end="\\r")

def main() -> None:
    print("--- GA Trading Strategy Framework ---")
    if getattr(config, 'PORTFOLIO_OPTIMIZATION_ENABLED', False):
        asset_list = getattr(config, 'ASSET_BASKET', [config.TICKER])
        print(f"Starting optimisation for portfolio: {asset_list}")
    else:
        print(f"Starting optimisation for: {config.SELECTED_ASSET_NAME} ({config.TICKER})")
    num_cores = os.cpu_count()
    print(f"Detected {num_cores} CPU cores available for parallel processing.")
    print("-" * 35)

    train_tickers = getattr(config, 'ASSET_BASKET', [config.TICKER]) if getattr(config, 'PORTFOLIO_OPTIMIZATION_ENABLED', False) else config.TICKER
    print(f"Loading TRAINING data from {config.TRAINING_PERIOD['start']} to {config.TRAINING_PERIOD['end']}...")
    ohlc_data = data_loader.get_data(train_tickers, config.TRAINING_PERIOD['start'], config.TRAINING_PERIOD['end'], config.TIMEFRAME)
    if ohlc_data.empty:
        print("No training data.")
        return

    print("Parsing strategy rules to identify genes for optimisation...")
    gene_space, gene_map, gene_types = parse_genes_from_config(config.STRATEGY_RULES)
    if not gene_space:
        print("No genes found. Exiting.")
        return
    print(f"Found {len(gene_space)} genes to optimise:")
    pprint.pprint(gene_map)
    print("-" * 35)

    fitness_evaluator = fitness.FitnessEvaluator(ohlc_data=ohlc_data, base_rules=config.STRATEGY_RULES, gene_map=gene_map)
    fitness_function = fitness_evaluator.__call__

    if getattr(config, 'AUTO_TUNE_ENABLED', False):
        if getattr(config, 'PORTFOLIO_OPTIMIZATION_ENABLED', False):
            tuning_ticker = getattr(config, 'TUNING_ASSET', config.TICKER)
            tune_data = data_loader.get_data(tuning_ticker, config.TRAINING_PERIOD['start'], config.TRAINING_PERIOD['end'], config.TIMEFRAME)
        best_hparams = tuner.find_best_hyperparameters(gene_space, gene_types, gene_map)
        print("Auto-tuner selected hyperparameters:", best_hparams)

    global start_time
    start_time = time.time()
    ga = pygad.GA(
        num_generations=config.GA_NUM_GENERATIONS,
        num_parents_mating=config.GA_PARENTS_MATING,
        sol_per_pop=config.GA_POPULATION_SIZE,
        num_genes=len(gene_space),
        gene_space=gene_space,
        gene_type=gene_types,
        mutation_num_genes=config.GA_MUTATION_NUM_GENES,
        fitness_func=fitness_function,
        on_generation=on_generation,
        parallel_processing=['process', num_cores],
    )
    try:
        ga.run()
        best_solution, best_fitness, _ = ga.best_solution()
        print(f"\\nBest training fitness: {best_fitness:.4f}")
        print("Winning gene values:")
        for i, v in enumerate(best_solution):
            print(f"  {gene_map[i]['name']}: {v}")
    except Exception as e:
        print(f"\\nError during GA run: {e}")
        traceback.print_exc()
        return

    print("\\n--- Running analysis on validation data ---")
    analysis.run_champion_analysis(best_solution, gene_map)

    if getattr(config, 'WALK_FORWARD_SETTINGS', {}).get('enabled', False):
        from walk_forward import run_walk_forward_validation
        run_walk_forward_validation(initial_champions=[list(best_solution)])

if __name__ == "__main__":
    main()
