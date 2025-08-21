# main.py

"""
Main Application Orchestrator for the GA Trading Framework
(This version includes a progress indicator for the GA run)
"""
import os
import pygad
import pprint
import tuner
import traceback
import time  # <-- NEW: Import the time module
import matplotlib.pyplot as plt  # For non-blocking plot display
from utils import set_global_seed

# Import our custom modules
import config
import data_loader
import fitness
import analysis
from gene_parser import parse_genes_from_config  # now defined in its own module
from datetime import datetime
from pathlib import Path


# --- NEW: Callback function for progress tracking and logging ---
start_time = 0.0
_best_fitness_seen = float("-inf")
_fitness_func_ref = None
_fitness_eval_ref = None


def on_generation(ga_instance):
    """Progress callback passed to ``pygad.GA``.

    When a new global best fitness is observed it re-evaluates the
    champion to obtain per-asset diagnostics and logs the top and bottom
    performers.
    """

    global _best_fitness_seen

    generation = ga_instance.generations_completed
    total_generations = ga_instance.num_generations
    best_solution, fitness, _ = ga_instance.best_solution(
        pop_fitness=ga_instance.last_generation_fitness
    )

    elapsed_time = time.time() - start_time
    est_time_remaining = (
        (elapsed_time / generation) * (total_generations - generation)
        if generation > 0
        else 0
    )
    remaining_seconds = int(est_time_remaining)
    if remaining_seconds == 0:
        time_left_str = "0s"
    else:
        unit = "sec" if remaining_seconds == 1 else "secs"
        time_left_str = f"{remaining_seconds} {unit}"

    # Use carriage return to update progress on a single line.
    print(
        "Generation "
        f"{generation}/{total_generations} | Best Fitness: {fitness:.4f} | Est. Time Left: {time_left_str}",
        end="\r",
    )

    if fitness > _best_fitness_seen and _fitness_func_ref and _fitness_eval_ref:
        _best_fitness_seen = fitness
        try:
            _fitness_func_ref(None, best_solution, 0)
            details = getattr(_fitness_eval_ref, "last_details", {})
            if details:
                charts_cfg = getattr(config, "CHARTS", {})
                run_ts = charts_cfg.get("run_ts")
                if not run_ts:
                    run_ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                    charts_cfg["run_ts"] = run_ts
                out_path = Path("reports") / run_ts / "opt_extremes.json"
                analysis.log_asset_extremes(details, save_path=out_path, quiet=True)
        except Exception:
            pass

def main():
    """ The main execution function. """
    print("--- GA Trading Strategy Framework ---")
    group_names = ", ".join(name for name, _ in getattr(config, "ASSET_GROUP", []))
    print(f"Starting multi-asset optimization for {group_names}")
    num_cores = os.cpu_count()
    print(f"Detected {num_cores} CPU cores available for parallel processing.")
    print("-" * 35)

    seed = None
    if getattr(config, "DETERMINISTIC", False):
        seed = getattr(config, "RANDOM_SEED", 42)
        set_global_seed(seed)
        print(f"Deterministic mode enabled. Seed={seed}")

    # Load price data.  When multi-asset mode is enabled we fetch and align data
    # for each asset in the configured group; otherwise fall back to the single
    # selected asset as before.
    if getattr(config, "MULTI_ASSET", {}).get("enabled"):
        print("Loading TRAINING data for asset group...")
        ohlc_data = data_loader.get_group_data(
            asset_group=config.ASSET_GROUP,
            start_date=config.TRAINING_PERIOD["start"],
            end_date=config.TRAINING_PERIOD["end"],
            interval=config.TIMEFRAME,
            coverage_threshold=config.COVERAGE_THRESHOLD,
        )
        if not ohlc_data:
            return
    else:
        print(
            "Loading TRAINING data from "
            f"{config.TRAINING_PERIOD['start']} to {config.TRAINING_PERIOD['end']}..."
        )
        ohlc_data = data_loader.get_data(
            ticker=config.TICKER,
            start_date=config.TRAINING_PERIOD['start'],
            end_date=config.TRAINING_PERIOD['end'],
            interval=config.TIMEFRAME
        )
        if ohlc_data.empty:
            return

    print("Parsing strategy rules to identify genes for optimization...")
    gene_space, gene_map, gene_types = parse_genes_from_config(config.STRATEGY_RULES)
    if not gene_space: print("No genes found. Exiting."); return
    print(f"Found {len(gene_space)} genes to optimize:"); pprint.pprint(gene_map); print("-" * 35)

    # Build the appropriate fitness evaluator (single- or multi-asset)
    fitness_evaluator = fitness.get_fitness_evaluator(
        ohlc_data=ohlc_data, base_rules=config.STRATEGY_RULES, gene_map=gene_map
    )
    evaluator_name = type(fitness_evaluator).__name__
    objective = getattr(fitness_evaluator, "settings", {}).get("metric", "composite")
    print(f"Active evaluator: {evaluator_name} | Objective: {objective}")
    assert objective, "Objective must be defined"
    fitness_function = fitness_evaluator.__call__

    global _fitness_func_ref, _fitness_eval_ref, _best_fitness_seen
    _fitness_func_ref = fitness_function
    _fitness_eval_ref = fitness_evaluator
    _best_fitness_seen = float("-inf")

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
        on_generation=on_generation,
        random_seed=seed
    )

    ga_instance.run()
    analysis.log_asset_extremes(fitness_evaluator.last_details)
    analysis.persist_details(fitness_evaluator)

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
    ga_instance.plot_fitness()

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
