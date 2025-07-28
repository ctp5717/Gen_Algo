# main.py

"""
Main Application Orchestrator for the GA Trading Framework
(This version includes a progress indicator for the GA run)
"""
import os
import pygad
import pprint
import traceback
import time # <-- NEW: Import the time module

# Import our custom modules
import config
import data_loader
import fitness
import analysis

# The parse_genes_from_config function remains the same.
def parse_genes_from_config(rules: dict):
    """
    This version is corrected to respect the 'is_active' flag for all rules.
    """
    gene_space = []
    gene_map = {}
    gene_types = []
    gene_index = 0

    def find_genes(sub_config, path):
        nonlocal gene_index
        # --- NEW: Check if the current dictionary represents a rule with an 'is_active' flag ---
        # If the rule is inactive, we stop searching deeper in this branch.
        if isinstance(sub_config, dict) and sub_config.get('is_active') is False:
            return

        if isinstance(sub_config, dict):
            for key, value in sub_config.items():
                current_path = path + [key]
                if isinstance(value, dict) and 'gene' in value:
                    gene_info = value
                    gene_name = gene_info['gene']
                    gene_type = int if isinstance(gene_info.get('step', 1.0), int) else float
                    
                    space_item = {'low': gene_info['low'], 'high': gene_info['high']}
                    if 'step' in gene_info:
                        space_item['step'] = gene_info['step']

                    gene_space.append(space_item)
                    gene_types.append(gene_type)
                    gene_map[gene_index] = {
                        'name': gene_name,
                        'path': current_path,
                        'type': gene_type
                    }
                    gene_index += 1
                elif isinstance(value, dict):
                    find_genes(value, current_path)
                elif isinstance(value, list):
                    find_genes(value, current_path)
        elif isinstance(sub_config, list):
            for i, item in enumerate(sub_config):
                current_path = path + [i]
                find_genes(item, current_path)

    find_genes(rules, [])
    return gene_space, gene_map, gene_types


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
    print(f"Generation {generation}/{total_generations} | Best Fitness: {fitness:.4f} | Est. Time Left: {int(est_time_remaining)}s", end='\r')

def main():
    """ The main execution function. """
    print("--- GA Trading Strategy Framework ---")
    print(f"Starting optimization for: {config.SELECTED_ASSET_NAME} ({config.TICKER})")
    num_cores = os.cpu_count()
    print(f"Detected {num_cores} CPU cores available for parallel processing.")
    print("-" * 35)

    print(f"Loading TRAINING data from {config.TRAINING_PERIOD['start']} to {config.TRAINING_PERIOD['end']}...")
    ohlc_data = data_loader.get_data(
        ticker=config.TICKER,
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

    print("Initializing and running the Genetic Algorithm in parallel...")
    global start_time; start_time = time.time() # Start the timer right before the GA run
    
    ga_instance = pygad.GA(
        num_generations=config.GA_NUM_GENERATIONS,
        num_parents_mating=config.GA_PARENTS_MATING,
        sol_per_pop=config.GA_POPULATION_SIZE,
        num_genes=len(gene_space),
        gene_space=gene_space,
        gene_type=gene_types,
        mutation_num_genes=config.GA_MUTATION_NUM_GENES,
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
    ga_instance.plot_fitness()

    try:
        analysis.run_champion_analysis(best_solution, gene_map)
    except Exception as e:
        print(f"\nAn error occurred during the analysis phase: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
