from datetime import datetime
from dateutil.relativedelta import relativedelta
import os
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import pygad
import vectorbt as vbt

import config
import data_loader
import strategy_engine as engine
from gene_parser import parse_genes_from_config
import fitness

def _generate_periods(start: datetime, end: datetime, train_months: int, test_months: int) -> List[Dict[str, datetime]]:
    start_dt = pd.to_datetime(start).to_pydatetime()
    end_dt = pd.to_datetime(end).to_pydatetime()
    if start_dt + relativedelta(months=train_months + test_months) > end_dt:
        return []
    periods: List[Dict[str, datetime]] = []
    current_start = start_dt
    while True:
        train_end = current_start + relativedelta(months=train_months)
        test_end = train_end + relativedelta(months=test_months)
        if test_end > end_dt:
            break
        periods.append({'train_start': current_start,'train_end': train_end,'test_start': train_end,'test_end': test_end})
        current_start += relativedelta(months=test_months)
    return periods

def _update_champion_pool(pool: List[List[float]], best_solution: List[float], validation_score: float, gene_space: List[Dict[str, any]], settings: Dict[str, any]) -> List[List[float]]:
    survival = settings.get('survival_threshold', 0.0)
    cloning = settings.get('cloning_threshold', float('inf'))
    num_clones = settings.get('num_clones', 0)
    mutation_rate = settings.get('clone_mutation_rate', 0.0)
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
                    low, high = gs['low'], gs['high']
                    step = gs.get('step')
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

def run_walk_forward_validation(initial_champions: Optional[List[List[float]]] = None):
    print("\n=== Running Walk-Forward Validation ===")
    num_cores = os.cpu_count()
    wf_settings = getattr(config, 'WALK_FORWARD_SETTINGS', {})
    date_range = wf_settings.get('total_data_range', {})
    start_date = date_range.get('start', config.TRAINING_PERIOD['start'])
    end_date = date_range.get('end', config.VALIDATION_PERIOD['end'])

    if getattr(config, 'PORTFOLIO_OPTIMIZATION_ENABLED', False):
        tickers = getattr(config, 'ASSET_BASKET', [config.TICKER])
    else:
        tickers = config.TICKER

    all_data = data_loader.get_data(tickers, start_date, end_date, config.TIMEFRAME)
    if all_data.empty:
        print("No data available for walk-forward validation.")
        return None

    start = all_data.index[0]
    end = all_data.index[-1]
    train_months = wf_settings.get('training_period_length', 12)
    test_months = wf_settings.get('validation_period_length', 3)
    periods = _generate_periods(start, end, train_months, test_months)
    if not periods:
        print("Insufficient data for the requested walk-forward windows.")
        return None

    results: List[Dict[str, any]] = []
    champion_pool: List[List[float]] = list(initial_champions or [])
    gene_space, gene_map, gene_types = parse_genes_from_config(config.STRATEGY_RULES)

    for idx, p in enumerate(periods, start=1):
        print(f"\n--- Window {idx} ---")
        print(f"Train: {p['train_start'].date()} -> {p['train_end'].date()}")
        print(f"Test : {p['test_start'].date()} -> {p['test_end'].date()}")

        train_data = all_data.loc[p['train_start']:p['train_end']]
        test_data = all_data.loc[p['test_start']:p['test_end']]

        ga = pygad.GA(
            num_generations=config.GA_NUM_GENERATIONS,
            num_parents_mating=config.GA_PARENTS_MATING,
            sol_per_pop=config.GA_POPULATION_SIZE,
            num_genes=len(gene_space),
            gene_space=gene_space,
            gene_type=gene_types,
            mutation_num_genes=config.GA_MUTATION_NUM_GENES,
            fitness_func=fitness.FitnessEvaluator(train_data, config.STRATEGY_RULES, gene_map).__call__,
            parallel_processing=['process', num_cores],
        )

        if champion_pool and hasattr(ga, 'population'):
            import numpy as np
            champs = np.array(champion_pool, dtype=float)
            if champs.ndim == 1:
                champs = champs.reshape(1, -1)
            if champs.shape[1] == ga.population.shape[1]:
                champs = champs[-config.GA_POPULATION_SIZE:]
                num_champs = min(len(champs), ga.population.shape[0])
                ga.population[:num_champs] = champs[:num_champs]
                if hasattr(ga, 'initial_population'):
                    ga.initial_population[:num_champs] = champs[:num_champs]

        ga.run()
        best_solution, best_fitness, _ = ga.best_solution()
        print(f"Best training fitness: {best_fitness:.4f}")
        winning_params = {gene_map[i]['name']: best_solution[i] for i in range(len(best_solution))}

        entries = engine.process_strategy_rules(test_data, fitness._inject_genes_into_rules(config.STRATEGY_RULES, gene_map, best_solution))
        total_trades = entries.astype(bool).values.sum() if isinstance(entries, pd.DataFrame) else int(entries.sum())
        if total_trades < config.FITNESS_WEIGHTS['min_trades']:
            print("No trades in test period.")
            results.append({'Window': idx,'Total Return [%]': np.nan,'Max Drawdown [%]': np.nan,'Sharpe Ratio': np.nan,'Sortino Ratio': np.nan,'Win Rate [%]': np.nan,'Params': None})
            continue

        exit_rules = config.STRATEGY_RULES.get('exit_rules', {}) or {}
        def getp(name):
            r = exit_rules.get(name, {}) or {}
            return r.get('params', {}).get('value') if r.get('is_active', False) else None
        sl_stop, sl_trail, tp_stop = getp('stop_loss'), getp('trailing_stop'), getp('take_profit')

        time_exit = entries.shift(config.MAX_HOLD_PERIOD, fill_value=False).reindex(entries.index, fill_value=False)
        if isinstance(test_data.columns, pd.MultiIndex):
            close_prices = test_data.xs('Close', level=1, axis=1)
        else:
            close_prices = test_data['Close']

        portfolio = vbt.Portfolio.from_signals(
            close=close_prices, entries=entries, exits=time_exit,
            sl_stop=sl_stop, tp_stop=tp_stop, sl_trail=sl_trail,
            fees=0.001, freq=config.TIMEFRAME,
        )
        stats = portfolio.stats()
        def _get(obj, key): return obj[key] if isinstance(obj, dict) else obj.get(key)
        tr, dd = _get(stats, 'Total Return [%]'), _get(stats, 'Max Drawdown [%]')
        sharpe, sortino, winr = _get(stats, 'Sharpe Ratio'), _get(stats, 'Sortino Ratio'), _get(stats, 'Win Rate [%]')

        print(f"Test Return: {tr:.2f}% | Max DD: {dd:.2f}%")
        print("Winning Parameters:")
        for k, v in winning_params.items():
            print(f"  {k}: {v}")

        validation_score = fitness.FitnessEvaluator(test_data, config.STRATEGY_RULES, gene_map)(None, best_solution, 0)
        champion_pool = _update_champion_pool(champion_pool, best_solution, validation_score, gene_space, getattr(config, 'CHAMPION_SELECTION_SETTINGS', {}))

        results.append({'Window': idx,'Total Return [%]': tr,'Max Drawdown [%]': dd,'Sharpe Ratio': sharpe,'Sortino Ratio': sortino,'Win Rate [%]': winr,'Params': winning_params})

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

    return {'folds': results_df,'average_return': avg_return,'std_return': std_return,'average_sharpe': avg_sharpe,'average_sortino': avg_sortino,'average_win_rate': avg_win,'total_compounded_return': total_compounded_return}
