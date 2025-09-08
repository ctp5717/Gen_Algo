# main.py

"""
Main Application Orchestrator for the GA Trading Framework
(This version includes a progress indicator for the GA run)
"""
import copy
import os
import pprint
import time
import traceback
import types
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt  # For non-blocking plot display
import pandas as pd
import pygad

import config
import data_loader
import run_metadata
import strategy_engine
from deps import ensure_pandas_ta, ensure_real_vectorbt, warn_if_deps_diverge
from gene_parser import parse_genes_from_config  # now defined in its own module

# --- NEW: Callback function for progress tracking ---
start_time = 0.0


# Placeholders for delayed imports (useful for tests to monkeypatch)
def _default_run_champion(*a, **k):
    return None


analysis = types.SimpleNamespace(run_champion_analysis=_default_run_champion)
fitness = types.SimpleNamespace(FitnessEvaluator=None)
tuner = types.SimpleNamespace(find_best_hyperparameters=None)


def on_generation(ga_instance):
    """
    This function is called by PyGAD after each generation completes.
    It prints a progress update to the console.
    """
    generation = ga_instance.generations_completed
    total_generations = ga_instance.num_generations
    fitness = ga_instance.best_solution(
        pop_fitness=ga_instance.last_generation_fitness
    )[1]

    elapsed_time = time.time() - start_time
    est_time_remaining = (
        (elapsed_time / generation) * (total_generations - generation)
        if generation > 0
        else 0
    )

    # Use carriage return `\r` and `end=''` to keep the output on a single, updating line.
    print(
        f"Generation {generation}/{total_generations} | "
        f"Best Fitness: {fitness:.4f} | "
        f"Est. Time Left: {int(est_time_remaining):>4}s   ",
        end="\r",
        flush=True,
    )


def indicator_preflight(sample: pd.DataFrame, rules: dict) -> None:
    """Compute indicators once to ensure required components exist."""
    print("Performing indicator preflight...")
    start = datetime.now(timezone.utc)
    rules_pf = copy.deepcopy(rules)
    entry = rules_pf.setdefault("entry_rules", {})
    logic = entry.get("combination_logic", "AND")
    if isinstance(logic, dict):
        entry["combination_logic"] = "AND"
    vt = entry.get("vote_threshold")
    if isinstance(vt, dict):
        entry["vote_threshold"] = vt.get("low", vt.get("high"))

    indicator_columns = {}
    indicator_errors: dict[str, list[dict]] = {}
    indicator_resolutions: list[dict[str, str | None]] = []
    for cond in entry.get("conditions", []):
        ind = cond.get("indicator")
        func = strategy_engine.INDICATOR_MAPPING.get(ind)
        if func is None:
            continue
        params_pf = {}
        for p, val in cond.get("params", {}).items():
            if isinstance(val, dict) and "gene" in val:
                if "options" in val:
                    params_pf[p] = val.get("options", [None])[0]
                else:
                    params_pf[p] = val.get("low", val.get("high"))
            else:
                params_pf[p] = val
        cond["params"] = params_pf
        try:
            out = func(sample, **params_pf)
        except Exception as e:
            indicator_errors.setdefault(ind, []).append(
                {
                    "params": params_pf,
                    "error": "".join(
                        traceback.format_exception_only(type(e), e)
                    ).strip(),
                }
            )
            continue
        resolved = cond.get("column") or cond.get("band")
        if isinstance(out, pd.DataFrame):
            cols = list(out.columns)
            print(f"{ind}: {cols}")
            indicator_columns[ind] = {"type": "DataFrame", "columns": cols}
            if resolved is None and cols:
                resolved = cols[0]
        else:
            print(f"{ind}: (Series)")
            indicator_columns[ind] = {"type": "Series", "columns": []}
        indicator_resolutions.append({"indicator": ind, "resolved_column": resolved})
    try:
        strategy_engine.process_strategy_rules(sample, rules_pf)
    except Exception as e:
        indicator_errors.setdefault("process_strategy_rules", []).append(
            {
                "params": {},
                "error": "".join(traceback.format_exception_only(type(e), e)).strip(),
            }
        )
    extra = {
        "indicator_columns": indicator_columns,
        "indicator_errors": indicator_errors,
        "indicator_resolutions": indicator_resolutions,
    }
    analysis._write_run_metadata(start, [], extra)
    if indicator_errors:
        summary = ", ".join(sorted(indicator_errors.keys()))
        exc = SystemExit(
            f"Indicator preflight failed for {len(indicator_errors)} indicator(s): {summary}"
        )
        exc.code = 2
        raise exc


def main():
    """The main execution function."""
    ensure_real_vectorbt(Path(__file__).resolve().parent)

    ta_ok = False
    try:
        ensure_pandas_ta()
        ta_ok = True
    except Exception:
        ta_ok = False
        raise
    finally:
        try:
            run_metadata._write_run_metadata({"ta_accessor_ok": ta_ok})
        except Exception:
            pass
    warn_if_deps_diverge()

    # Delay heavy imports until after vectorbt is validated
    global analysis, fitness, tuner
    patched_analysis = analysis
    patched_fitness = fitness
    patched_tuner = tuner

    import analysis as _analysis
    import fitness as _fitness

    if patched_analysis.run_champion_analysis is not _default_run_champion:
        _analysis.run_champion_analysis = patched_analysis.run_champion_analysis
    if patched_fitness.FitnessEvaluator is not None:
        _fitness.FitnessEvaluator = patched_fitness.FitnessEvaluator
    analysis, fitness = _analysis, _fitness

    if getattr(config, "AUTO_TUNE_ENABLED", False):
        if patched_tuner.find_best_hyperparameters is not None:
            tuner = patched_tuner
        else:
            import tuner as _tuner

            tuner = _tuner
    else:
        tuner = patched_tuner

    print("--- GA Trading Strategy Framework ---")
    if getattr(config, "MULTI_ASSET", {}).get("enabled"):
        assets = [name for name, _ in getattr(config, "ASSET_GROUP", [])]
        preview = ", ".join(assets[:5])
        more = "" if len(assets) <= 5 else ", ..."
        print(
            f"Starting multi-asset optimization for {len(assets)} assets ({preview}{more})"
        )
    else:
        print(
            f"Starting optimization for: {config.SELECTED_ASSET_NAME} ({config.TICKER})"
        )
    num_cores = os.cpu_count()
    print(f"Detected {num_cores} CPU cores available for parallel processing.")
    print("-" * 35)

    # Determine the full date range needed across training, validation, and walk-forward
    train_start = pd.to_datetime(config.TRAINING_PERIOD["start"])
    train_end = pd.to_datetime(config.TRAINING_PERIOD["end"])
    val_start = pd.to_datetime(config.VALIDATION_PERIOD["start"])
    val_end = pd.to_datetime(config.VALIDATION_PERIOD["end"])
    wf_settings = getattr(config, "WALK_FORWARD_SETTINGS", {})
    wf_enabled = wf_settings.get(
        "enabled", getattr(config, "ENABLE_WALK_FORWARD_VALIDATION", False)
    )
    if wf_enabled:
        wf_range = wf_settings.get("total_data_range", {})
        wf_start = pd.to_datetime(wf_range.get("start", train_start))
        wf_end = pd.to_datetime(wf_range.get("end", val_end))
    else:
        wf_start, wf_end = train_start, val_end
    earliest = min(train_start, val_start, wf_start).strftime("%Y-%m-%d")
    latest = max(train_end, val_end, wf_end).strftime("%Y-%m-%d")

    # Load price data once for the full range and slice for each phase
    if getattr(config, "MULTI_ASSET", {}).get("enabled"):
        print(f"Loading data for asset group from {earliest} to {latest}...")
        all_data = data_loader.get_group_data(
            asset_group=config.ASSET_GROUP,
            start_date=earliest,
            end_date=latest,
            interval=config.TIMEFRAME,
            coverage_threshold=config.COVERAGE_THRESHOLD,
            verbose=False,
        )
        if not all_data:
            return
        training_data = {
            t: df.loc[config.TRAINING_PERIOD["start"] : config.TRAINING_PERIOD["end"]]
            for t, df in all_data.items()
        }
        validation_data = {
            t: df.loc[
                config.VALIDATION_PERIOD["start"] : config.VALIDATION_PERIOD["end"]
            ]
            for t, df in all_data.items()
        }
    else:
        print(f"Loading data from {earliest} to {latest}...")
        all_data, _ = data_loader.get_data(
            ticker=config.TICKER,
            start_date=earliest,
            end_date=latest,
            interval=config.TIMEFRAME,
            verbose=True,
        )
        if all_data.empty:
            return
        training_data = all_data.loc[
            config.TRAINING_PERIOD["start"] : config.TRAINING_PERIOD["end"]
        ]
        validation_data = all_data.loc[
            config.VALIDATION_PERIOD["start"] : config.VALIDATION_PERIOD["end"]
        ]

    sample = (
        next(iter(training_data.values()))
        if isinstance(training_data, dict)
        else training_data
    )
    indicator_preflight(sample, config.STRATEGY_RULES)

    print("Parsing strategy rules to identify genes for optimization...")
    gene_space, gene_map, gene_types = parse_genes_from_config(config.STRATEGY_RULES)
    if not gene_space:
        print("No genes found. Exiting.")
        return
    print(f"Found {len(gene_space)} genes to optimize:")
    pprint.pprint(gene_map)
    print("-" * 35)

    # Build the appropriate fitness evaluator (single- or multi-asset)
    fitness_evaluator = fitness.get_fitness_evaluator(
        ohlc_data=training_data, base_rules=config.STRATEGY_RULES, gene_map=gene_map
    )
    fitness_function = fitness_evaluator.__call__

    if getattr(config, "AUTO_TUNE_ENABLED", False):
        tuned = tuner.find_best_hyperparameters(
            training_data, gene_space, gene_map, gene_types, validation_data
        )
        sol_per_pop = (
            tuned.get("sol_per_pop", config.GA_POPULATION_SIZE)
            if tuned
            else config.GA_POPULATION_SIZE
        )
        num_parents_mating = (
            tuned.get("num_parents_mating", config.GA_PARENTS_MATING)
            if tuned
            else config.GA_PARENTS_MATING
        )
        mutation_num_genes = (
            tuned.get("mutation_num_genes", config.GA_MUTATION_NUM_GENES)
            if tuned
            else config.GA_MUTATION_NUM_GENES
        )
    else:
        sol_per_pop = config.GA_POPULATION_SIZE
        num_parents_mating = config.GA_PARENTS_MATING
        mutation_num_genes = config.GA_MUTATION_NUM_GENES

    print("Initializing and running the Genetic Algorithm in parallel...")
    global start_time
    start_time = time.time()  # Start the timer right before the GA run

    ga_instance = pygad.GA(
        num_generations=config.GA_NUM_GENERATIONS,
        num_parents_mating=num_parents_mating,
        sol_per_pop=sol_per_pop,
        num_genes=len(gene_space),
        gene_space=gene_space,
        gene_type=list(gene_types),
        mutation_num_genes=mutation_num_genes,
        fitness_func=fitness_function,
        parallel_processing=["process", num_cores],
        # --- NEW: Pass the callback function to the GA instance ---
        on_generation=on_generation,
    )

    ga_instance.run()

    # Print a newline character to move off the progress line.
    print("\n" + "-" * 35)
    print("Optimization finished.")

    best_solution, best_solution_fitness, _ = ga_instance.best_solution()
    print(f"\nBest Solution's Fitness (Training Period): {best_solution_fitness:.4f}")
    print("Optimal Parameters Found:")
    for i, gene_value in enumerate(best_solution):
        gene_name = gene_map[i]["name"]
        gene_type = gene_map[i]["type"]
        if gene_type == int:
            print(f"  - {gene_name}: {int(gene_value)}")
        else:
            print(f"  - {gene_name}: {gene_value:.4f}")
    print("\nDisplaying GA fitness evolution plot...")
    plt.ion()
    fig, ax = plt.subplots()
    if getattr(ga_instance, "best_solutions_fitness", None):
        ax.plot(ga_instance.best_solutions_fitness, label="Best Fitness")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels)
        ax.set_title("GA Fitness Evolution")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness")
    fig_path = "ga_fitness_evolution.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    artifacts = [fig_path]

    try:
        analysis.run_champion_analysis(
            best_solution, gene_map, validation_data, artifacts
        )
    except Exception as e:
        print(f"\nAn error occurred during the analysis phase: {e}")
        traceback.print_exc()

    if wf_enabled:
        try:
            import walk_forward

            wf_range = wf_settings.get("total_data_range", {})
            wf_start = wf_range.get("start", config.TRAINING_PERIOD["start"])
            wf_end = wf_range.get("end", config.VALIDATION_PERIOD["end"])
            if getattr(config, "MULTI_ASSET", {}).get("enabled"):
                wf_data = {t: df.loc[wf_start:wf_end] for t, df in all_data.items()}
            else:
                wf_data = all_data.loc[wf_start:wf_end]
            walk_forward.run_walk_forward_validation(
                initial_champions=[best_solution], data=wf_data
            )
        except Exception as e:
            print(f"An error occurred during walk-forward validation: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
