"""Walk-Forward Validation Module."""

from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import os
import json
import numpy as np
import pygad
import vectorbt as vbt

import config
import data_loader
import strategy_engine as engine
from gene_parser import parse_genes_from_config
import fitness
from multi_asset_fitness import MultiAssetFitnessEvaluator
import scanner_sim
from scoring import SCORE_FUNCTIONS
from log_utils import get_run_logger, log_run_parameters


def _generate_periods(start: datetime, end: datetime, train_months: int, test_months: int):
    """Generate rolling training and testing windows."""
    # Ensure plain Python datetimes for relativedelta calculations
    start = pd.to_datetime(start).to_pydatetime()
    end = pd.to_datetime(end).to_pydatetime()

    # Quick check to avoid an infinite loop when the dataset is too short
    if start + relativedelta(months=train_months + test_months) > end:
        return []

    periods = []
    current_start = start
    while True:
        train_end = current_start + relativedelta(months=train_months)
        test_end = train_end + relativedelta(months=test_months)
        if test_end > end:
            break
        periods.append({
            'train_start': current_start,
            'train_end': train_end,
            'test_start': train_end,
            'test_end': test_end,
        })
        current_start += relativedelta(months=test_months)
    return periods


def _update_champion_pool(pool, best_solution, validation_score, gene_space, settings):
    """Update champion pool based on validation fitness."""
    survival = settings.get("survival_threshold", 0.0)
    cloning = settings.get("cloning_threshold", float("inf"))
    num_clones = settings.get("num_clones", 0)
    mutation_rate = settings.get("clone_mutation_rate", 0.0)

    if validation_score < survival:
        print("Champion discarded due to poor performance.")
        return pool

    if validation_score >= cloning:
        print("Elite Champion found. Cloning champion.")
        pool.append(list(best_solution))
        for _ in range(num_clones):
            clone = list(best_solution)
            for idx in range(len(clone)):
                if np.random.rand() < mutation_rate:
                    gs = gene_space[idx]
                    low, high = gs["low"], gs["high"]
                    step = gs.get("step")
                    if step is not None:
                        steps = int(round((high - low) / step))
                        val = low + step * np.random.randint(0, steps + 1)
                    else:
                        val = np.random.uniform(low, high)
                    clone[idx] = type(clone[idx])(val)
            pool.append(clone)
    else:
        print("Viable Champion found and kept for next fold.")
        pool.append(list(best_solution))

    return pool


def run_walk_forward_validation(initial_champions=None):
    """Execute walk-forward validation across the available data.

    Parameters
    ----------
    initial_champions : list[list[float]] or None
        Optional list of solutions to seed the first population. Each solution
        should be an iterable of gene values matching the strategy's genes.
    """
    print("\n=== Running Walk-Forward Validation ===")
    num_cores = os.cpu_count()
    print(f"Using {num_cores} CPU cores for GA optimisation during each window.")
    logger = get_run_logger()
    log_run_parameters(logger)
    wf_settings = getattr(config, "WALK_FORWARD_SETTINGS", {})
    date_range = wf_settings.get("total_data_range", {})
    start_date = date_range.get("start", config.TRAINING_PERIOD["start"])
    end_date = date_range.get("end", config.VALIDATION_PERIOD["end"])

    all_data = data_loader.load_group_data(
        config.ASSET_GROUP,
        start_date,
        end_date,
        config.TIMEFRAME,
    )
    if not all_data:
        print("No data available for walk-forward validation.")
        return

    index = next(iter(all_data.values())).index
    start = index[0]
    end = index[-1]
    train_months = wf_settings.get(
        "training_period_length",
        getattr(config, "WALK_FORWARD_TRAINING_MONTHS", 12),
    )
    test_months = wf_settings.get(
        "validation_period_length",
        getattr(config, "WALK_FORWARD_TEST_MONTHS", 3),
    )

    periods = _generate_periods(start, end, train_months, test_months)
    if not periods:
        print("Insufficient data for the requested walk-forward windows.")
        return

    results = []
    champion_pool = list(initial_champions or [])
    results_dir = os.path.join(os.path.dirname(__file__), "wf_results")
    os.makedirs(results_dir, exist_ok=True)

    for idx, p in enumerate(periods, start=1):
        print(f"\n--- Window {idx} ---")
        print(f"Train: {p['train_start'].date()} -> {p['train_end'].date()}")
        print(f"Test : {p['test_start'].date()} -> {p['test_end'].date()}")

        train_dict = {
            name: df.loc[p["train_start"] : p["train_end"]]
            for name, df in all_data.items()
        }
        test_dict = {
            name: df.loc[p["test_start"] : p["test_end"]]
            for name, df in all_data.items()
        }

        gene_space, gene_map, gene_types = parse_genes_from_config(
            config.STRATEGY_RULES
        )
        evaluator = MultiAssetFitnessEvaluator(
            train_dict, config.STRATEGY_RULES, gene_map
        )
        ga_instance = pygad.GA(
            num_generations=config.GA_NUM_GENERATIONS,
            num_parents_mating=config.GA_PARENTS_MATING,
            sol_per_pop=config.GA_POPULATION_SIZE,
            num_genes=len(gene_space),
            gene_space=gene_space,
            gene_type=gene_types,
            mutation_num_genes=config.GA_MUTATION_NUM_GENES,
            fitness_func=evaluator.__call__,
            parallel_processing=["process", num_cores],
        )
        if champion_pool and hasattr(ga_instance, "population"):
            champs = np.array(champion_pool, dtype=float)
            if champs.ndim == 1:
                champs = champs.reshape(1, -1)
            if champs.shape[1] == ga_instance.population.shape[1]:
                champs = champs[-config.GA_POPULATION_SIZE :]
                num_champs = min(len(champs), ga_instance.population.shape[0])
                ga_instance.population[:num_champs] = champs[:num_champs]
                if hasattr(ga_instance, "initial_population"):
                    ga_instance.initial_population[:num_champs] = champs[:num_champs]
        ga_instance.run()
        best_solution, best_fitness, _ = ga_instance.best_solution()
        print(f"Best training fitness: {best_fitness:.4f}")

        winning_params = {
            gene_map[i]["name"]: best_solution[i] for i in range(len(best_solution))
        }

        # Build signals for test set
        entries = {}
        exits = {}
        scores = {}
        score_func_name = config.SCANNER.get("score_func", "pct_change")
        score_func = SCORE_FUNCTIONS.get(
            score_func_name, SCORE_FUNCTIONS["pct_change"]
        )
        rules = fitness._inject_genes_into_rules(
            config.STRATEGY_RULES, gene_map, best_solution
        )
        for name, data in test_dict.items():
            asset_entries = engine.process_strategy_rules(data, rules)
            asset_exits = asset_entries.shift(
                config.MAX_HOLD_PERIOD, fill_value=False
            )
            asset_exits = asset_exits.reindex(asset_entries.index, fill_value=False)
            entries[name] = asset_entries
            exits[name] = asset_exits
            if config.SCANNER.get("tie_break_policy") == "score":
                scores[name] = (
                    score_func(data).reindex(asset_entries.index).fillna(0.0)
                )
        entries_df = pd.concat(entries, axis=1)
        exits_df = pd.concat(exits, axis=1)
        scores_df = pd.concat(scores, axis=1) if scores else None

        gated, open_count, diag = scanner_sim.gate_entries(
            entries_df,
            exits_df,
            config.SCANNER.get("max_concurrent_trades", 1),
            config.SCANNER.get("tie_break_policy", "fifo"),
            seed=config.SCANNER.get("seed", 0),
            scores=scores_df,
        )

        returns_df = pd.DataFrame(0.0, index=gated.index, columns=gated.columns)
        total_wins = 0
        total_trades = 0
        for name in gated.columns:
            data = test_dict[name]
            asset_entries = gated[name].reindex(data.index, fill_value=False)
            asset_exits = exits_df[name].reindex(data.index, fill_value=False)
            if asset_entries.any():
                pf = vbt.Portfolio.from_signals(
                    close=data["Close"],
                    entries=asset_entries,
                    exits=asset_exits,
                    fees=config.FEES,
                    freq=config.TIMEFRAME,
                )
                returns_df[name] = pf.returns()
                t_stats = pf.trades.stats()
                wins = t_stats.get("Win Rate [%]", 0.0) / 100 * t_stats.get(
                    "Count", 0
                )
                total_wins += wins
                total_trades += t_stats.get("Count", 0)
            else:
                returns_df[name] = 0.0

        open_count_safe = open_count.reindex(returns_df.index).replace(0, np.nan)
        portfolio_returns = (
            returns_df.sum(axis=1) / open_count_safe
        ).fillna(0.0)
        sortino, _pf, max_dd = MultiAssetFitnessEvaluator._calc_stats(
            portfolio_returns
        )
        total_return = (portfolio_returns + 1).prod() - 1
        sharpe = (
            portfolio_returns.mean() / portfolio_returns.std(ddof=0)
            if portfolio_returns.std(ddof=0) != 0
            else 0
        )
        win_rate = (
            total_wins / total_trades * 100 if total_trades > 0 else 0
        )

        print(f"Test Return: {total_return * 100:.2f}% | Max DD: {max_dd:.2f}%")
        print("Winning Parameters:")
        for param_name, param_value in winning_params.items():
            print(f"  {param_name}: {param_value}")

        # Evaluate champion on validation data using composite fitness
        val_evaluator = MultiAssetFitnessEvaluator(
            test_dict, config.STRATEGY_RULES, gene_map
        )
        validation_score, *_ = val_evaluator._evaluate_once(
            best_solution, config.SCANNER.get("seed", 0), val_evaluator.assets
        )
        champion_settings = getattr(config, "CHAMPION_SELECTION_SETTINGS", {})
        champion_pool = _update_champion_pool(
            champion_pool, best_solution, validation_score, gene_space, champion_settings
        )

        with open(
            os.path.join(results_dir, f"fold_{idx}_champion.json"), "w"
        ) as f:
            json.dump(
                {"solution": [float(x) for x in best_solution], "params": winning_params},
                f,
                indent=2,
            )
        with open(
            os.path.join(results_dir, f"fold_{idx}_diagnostics.json"), "w"
        ) as f:
            json.dump(diag, f, indent=2)

        results.append(
            {
                "Window": idx,
                "Total Return [%]": total_return * 100,
                "Max Drawdown [%]": max_dd,
                "Sharpe Ratio": sharpe,
                "Sortino Ratio": sortino,
                "Win Rate [%]": win_rate,
                "Params": winning_params,
                "Diagnostics": diag,
            }
        )

    if not results:
        print("\nNo walk-forward runs produced trades.")
        return None

    results_df = pd.DataFrame(results)
    avg_return = results_df['Total Return [%]'].mean()
    std_return = results_df['Total Return [%]'].std()
    avg_sharpe = results_df['Sharpe Ratio'].mean()
    avg_sortino = results_df['Sortino Ratio'].mean()
    avg_win = results_df['Win Rate [%]'].mean()
    total_compounded_return = (results_df['Total Return [%]'] / 100 + 1).prod() - 1

    print("\n=== Walk-Forward Summary ===")
    with pd.option_context('display.max_colwidth', None, 'display.width', None):
        print(results_df.to_string(index=False))
    print("\nAggregate Metrics:")
    print(f"Average Return: {avg_return:.2f}% (+/- {std_return:.2f}%)")
    print(f"Average Sharpe: {avg_sharpe:.2f}")
    print(f"Average Sortino: {avg_sortino:.2f}")
    print(f"Average Win Rate: {avg_win:.2f}%")
    print(f"Total Compounded Return: {total_compounded_return * 100:.2f}%")

    return {
        'folds': results_df,
        'average_return': avg_return,
        'std_return': std_return,
        'average_sharpe': avg_sharpe,
        'average_sortino': avg_sortino,
        'average_win_rate': avg_win,
        'total_compounded_return': total_compounded_return,
    }
