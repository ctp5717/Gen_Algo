"""Walk-Forward Validation Module."""

from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import os
import numpy as np
import pygad

import config
import data_loader
import strategy_engine as engine
from gene_parser import parse_genes_from_config
import fitness
import ga_utils


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
    wf_settings = getattr(config, "WALK_FORWARD_SETTINGS", {})
    date_range = wf_settings.get("total_data_range", {})
    start_date = date_range.get("start", config.TRAINING_PERIOD["start"])
    end_date = date_range.get("end", config.VALIDATION_PERIOD["end"])

    tickers = (
        config.ASSET_BASKET
        if getattr(config, "PORTFOLIO_OPTIMIZATION_ENABLED", False)
        else config.TICKER
    )
    all_data = data_loader.get_data(
        ticker=tickers,
        start_date=start_date,
        end_date=end_date,
        interval=config.TIMEFRAME,
    )
    if all_data.empty:
        print("No data available for walk-forward validation.")
        return

    start = all_data.index[0]
    end = all_data.index[-1]
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

    for idx, p in enumerate(periods, start=1):
        print(f"\n--- Window {idx} ---")
        print(f"Train: {p['train_start'].date()} -> {p['train_end'].date()}")
        print(f"Test : {p['test_start'].date()} -> {p['test_end'].date()}")
        # fmt: off
        train_data = all_data.loc[p['train_start']:p['train_end']]
        test_data = all_data.loc[p['test_start']:p['test_end']]
        # fmt: on

        gene_space, gene_map, gene_types = parse_genes_from_config(config.STRATEGY_RULES)
        evaluator = fitness.FitnessEvaluator(train_data, config.STRATEGY_RULES, gene_map)
        ga_instance = pygad.GA(
            num_generations=config.GA_NUM_GENERATIONS,
            num_parents_mating=config.GA_PARENTS_MATING,
            sol_per_pop=config.GA_POPULATION_SIZE,
            num_genes=len(gene_space),
            gene_space=gene_space,
            gene_type=gene_types,
            mutation_num_genes=config.GA_MUTATION_NUM_GENES,
            fitness_func=evaluator.__call__,
            parallel_processing=['process', num_cores],
            on_generation=ga_utils.make_stagnation_callback(),
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

        rules = fitness._inject_genes_into_rules(config.STRATEGY_RULES, gene_map, best_solution)
        entries = engine.process_strategy_rules(test_data, rules)
        if fitness._count_trades(entries) < config.FITNESS_WEIGHTS['min_trades']:
            print("No trades in test period.")
            results.append({
                'Window': idx,
                'Total Return [%]': np.nan,
                'Max Drawdown [%]': np.nan,
                'Sharpe Ratio': np.nan,
                'Sortino Ratio': np.nan,
                'Win Rate [%]': np.nan,
                'Params': None,
            })
            continue
        exit_rules = rules.get('exit_rules', {})
        sl_rule = exit_rules.get('stop_loss', {})
        tsl_rule = exit_rules.get('trailing_stop', {})
        tp_rule = exit_rules.get('take_profit', {})

        sl_stop = (
            sl_rule.get("params", {}).get("value")
            if sl_rule.get("is_active", False)
            else None
        )
        sl_trail = (
            tsl_rule.get("params", {}).get("value")
            if tsl_rule.get("is_active", False)
            else None
        )
        tp_stop = (
            tp_rule.get("params", {}).get("value")
            if tp_rule.get("is_active", False)
            else None
        )

        try:
            _, _, agg_stats, _ = fitness.run_portfolio_backtest(
                test_data,
                entries,
                sl_stop=sl_stop,
                sl_trail=sl_trail,
                tp_stop=tp_stop,
                weights=getattr(config, "PORTFOLIO_WEIGHTS", None),
            )
        except RuntimeError as e:
            print(e)
            continue

        tr = agg_stats.get('Total Return [%]') if not isinstance(agg_stats, pd.DataFrame) else agg_stats.loc['Total Return [%]'].iloc[0]
        dd = agg_stats.get('Max Drawdown [%]') if not isinstance(agg_stats, pd.DataFrame) else agg_stats.loc['Max Drawdown [%]'].iloc[0]
        sharpe = agg_stats.get('Sharpe Ratio') if not isinstance(agg_stats, pd.DataFrame) else agg_stats.loc['Sharpe Ratio'].iloc[0]
        sortino = agg_stats.get('Sortino Ratio') if not isinstance(agg_stats, pd.DataFrame) else agg_stats.loc['Sortino Ratio'].iloc[0]
        win_rate = agg_stats.get('Win Rate [%]') if not isinstance(agg_stats, pd.DataFrame) else agg_stats.loc['Win Rate [%]'].iloc[0]
        print(f"Test Return: {tr:.2f}% | Max DD: {dd:.2f}%")
        print("Winning Parameters:")
        for param_name, param_value in winning_params.items():
            print(f"  {param_name}: {param_value}")

        # Evaluate champion on validation data using composite fitness
        val_evaluator = fitness.FitnessEvaluator(test_data, config.STRATEGY_RULES, gene_map)
        validation_score = val_evaluator(None, best_solution, 0)
        champion_settings = getattr(config, "CHAMPION_SELECTION_SETTINGS", {})
        champion_pool = _update_champion_pool(
            champion_pool, best_solution, validation_score, gene_space, champion_settings
        )

        results.append({
            'Window': idx,
            'Total Return [%]': tr,
            'Max Drawdown [%]': dd,
            'Sharpe Ratio': sharpe,
            'Sortino Ratio': sortino,
            'Win Rate [%]': win_rate,
            'Params': winning_params,
        })

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
