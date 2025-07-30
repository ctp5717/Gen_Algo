"""Walk-Forward Validation Module."""

from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import os
import pygad
import vectorbt as vbt

import config
import data_loader
import strategy_engine as engine
from gene_parser import parse_genes_from_config
import fitness


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


def run_walk_forward_validation():
    """Execute walk-forward validation across the available data."""
    print("\n=== Running Walk-Forward Validation ===")
    num_cores = os.cpu_count()
    print(f"Using {num_cores} CPU cores for GA optimisation during each window.")
    all_data = data_loader.get_data(
        ticker=config.TICKER,
        start_date=config.TRAINING_PERIOD['start'],
        end_date=config.VALIDATION_PERIOD['end'],
        interval=config.TIMEFRAME,
    )
    if all_data.empty:
        print("No data available for walk-forward validation.")
        return

    start = all_data.index[0]
    end = all_data.index[-1]
    periods = _generate_periods(
        start,
        end,
        config.WALK_FORWARD_TRAINING_MONTHS,
        config.WALK_FORWARD_TEST_MONTHS,
    )
    if not periods:
        print("Insufficient data for the requested walk-forward windows.")
        return

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
        )
        ga_instance.run()
        best_solution, best_fitness, _ = ga_instance.best_solution()
        print(f"Best training fitness: {best_fitness:.4f}")

        rules = fitness._inject_genes_into_rules(config.STRATEGY_RULES, gene_map, best_solution)
        entries = engine.process_strategy_rules(test_data, rules)
        if entries.sum() < config.FITNESS_WEIGHTS['min_trades']:
            print("No trades in test period.")
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

        time_exit = entries.shift(config.MAX_HOLD_PERIOD, fill_value=False)
        time_exit = time_exit.reindex(entries.index, fill_value=False)

        portfolio = vbt.Portfolio.from_signals(
            close=test_data['Close'],
            entries=entries,
            exits=time_exit,
            sl_stop=sl_stop,
            tp_stop=tp_stop,
            sl_trail=sl_trail,
            fees=0.001,
            freq=config.TIMEFRAME,
        )
        stats = portfolio.stats()
        tr = stats['Total Return [%]']
        dd = stats['Max Drawdown [%]']
        print(f"Test Return: {tr:.2f}% | Max DD: {dd:.2f}%")
