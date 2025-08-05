# main.py

"""
Main Application Orchestrator for the GA Trading Framework
(This version includes a progress indicator for the GA run)
"""
import os
import multiprocessing as mp
import logging
import pprint
import time  # <-- NEW: Import the time module

import matplotlib.pyplot as plt  # For non-blocking plot display
import pygad
import tuner
import traceback

# Import our custom modules
import config
import data_loader
import fitness
import analysis
from gene_parser import parse_genes_from_config  # now defined in its own module


logging.basicConfig(level=logging.DEBUG)


# --- NEW: Callback function for progress tracking ---
start_time = 0.0
def on_generation(ga_instance):
    """
    This function is called by PyGAD after each generation completes.
    It prints a progress update to the console.
    """
    generation = ga_instance.generations_completed
    total_generations = ga_instance.num_generations
    fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]
    
    elapsed_time = time.time() - start_time
    est_time_remaining = (elapsed_time / generation) * (total_generations - generation) if generation > 0 else 0
    
    # Use carriage return `\r` and `end=''` to keep the output on a single, updating line.
    print(
        "Generation "
        f"{generation}/{total_generations} | Best Fitness: {fitness:.4f} | Est. Time Left: {int(est_time_remaining)}s",
        end="\r",
    )

def main():
    """ The main execution function. """
    # Vectorbt and other numeric libraries can misbehave when forked; ensure
    # the safer "spawn" start method is used for multiprocessing to avoid
    # silent worker crashes that manifest as BrokenProcessPool errors.
    mp.set_start_method("spawn", force=True)

    print("--- GA Trading Strategy Framework ---")
    if getattr(config, "PORTFOLIO_OPTIMIZATION_ENABLED", False):
        print(
            "Starting portfolio optimization for basket: "
            f"{config.ASSET_BASKET} (tuning asset: {config.TUNING_ASSET})"
        )
        tickers = config.ASSET_BASKET
    else:
        print(
            f"Starting optimization for: {config.SELECTED_ASSET_NAME} ({config.TICKER})"
        )
        tickers = config.TICKER
    num_cores = os.cpu_count()
    print(f"Detected {num_cores} CPU cores available for parallel processing.")
    print("-" * 35)

    print(
        "Loading TRAINING data from "
        f"{config.TRAINING_PERIOD['start']} to {config.TRAINING_PERIOD['end']}..."
    )
    ohlc_data = data_loader.get_data(
        ticker=tickers,
        start_date=config.TRAINING_PERIOD['start'],
        end_date=config.TRAINING_PERIOD['end'],
        interval=config.TIMEFRAME
    )
    if ohlc_data.empty: return

    print("Parsing strategy rules to identify genes for optimization...")
    gene_space, gene_map, gene_types = parse_genes_from_config(config.STRATEGY_RULES)
    if not gene_space: print("No genes found. Exiting."); return
    print(f"Found {len(gene_space)} genes to optimize:"); pprint.pprint(gene_map); print("-" * 35)

    fitness_evaluator = fitness.FitnessEvaluator(
        ohlc_data=ohlc_data, base_rules=config.STRATEGY_RULES, gene_map=gene_map
    )
    fitness_function = fitness_evaluator.__call__

    if getattr(config, "AUTO_TUNE_ENABLED", False):
        tuned = tuner.find_best_hyperparameters(ohlc_data, gene_space, gene_map, gene_types)
        sol_per_pop = tuned.get("sol_per_pop", config.GA_POPULATION_SIZE) if tuned else config.GA_POPULATION_SIZE
        num_parents_mating = tuned.get("num_parents_mating", config.GA_PARENTS_MATING) if tuned else config.GA_PARENTS_MATING
        mutation_num_genes = tuned.get("mutation_num_genes", config.GA_MUTATION_NUM_GENES) if tuned else config.GA_MUTATION_NUM_GENES
    else:
        sol_per_pop = config.GA_POPULATION_SIZE
        num_parents_mating = config.GA_PARENTS_MATING
        mutation_num_genes = config.GA_MUTATION_NUM_GENES

    print("Initializing and running the Genetic Algorithm in parallel...")
    global start_time; start_time = time.time() # Start the timer right before the GA run

    ga_instance = pygad.GA(
        num_generations=config.GA_NUM_GENERATIONS,
        num_parents_mating=num_parents_mating,
        sol_per_pop=sol_per_pop,
        num_genes=len(gene_space),
        gene_space=gene_space,
        gene_type=list(gene_types),
        mutation_num_genes=mutation_num_genes,
        fitness_func=fitness_function,
        parallel_processing=['process', num_cores],
        # --- NEW: Pass the callback function to the GA instance ---
        on_generation=on_generation
    )
    
    ga_instance.run()
    
    # Print a newline character to move off the progress line.
    print("\n" + "-" * 35)
    print("Optimization finished.")

    best_solution, best_solution_fitness, _ = ga_instance.best_solution()
    print(f"\nBest Solution's Fitness (Training Period): {best_solution_fitness:.4f}")
    print("Optimal Parameters Found:")
    for i, gene_value in enumerate(best_solution):
        gene_name = gene_map[i]['name']
        gene_type = gene_map[i]['type']
        if gene_type == int: print(f"  - {gene_name}: {int(gene_value)}")
        else: print(f"  - {gene_name}: {gene_value:.4f}")
    print("\nDisplaying GA fitness evolution plot...")
    # Enable interactive mode so the plot window does not block execution.
    plt.ion()
    ga_instance.plot_fitness(legend=False)

    try:
        analysis.run_champion_analysis(best_solution, gene_map)
    except Exception as e:
        print(f"\nAn error occurred during the analysis phase: {e}")
        traceback.print_exc()

    wf_settings = getattr(config, "WALK_FORWARD_SETTINGS", {})
    wf_enabled = wf_settings.get(
        "enabled",
        getattr(config, "ENABLE_WALK_FORWARD_VALIDATION", False),
    )
    if wf_enabled:
        try:
            import walk_forward
            walk_forward.run_walk_forward_validation(initial_champions=[best_solution])
        except Exception as e:
            print(f"An error occurred during walk-forward validation: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    main()
