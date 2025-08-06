# fitness.py

"""
Fitness Function for Genetic Algorithm
(This version uses the correct pandas .shift() method for time-based exits)
"""
import copy
import logging

import numpy as np
import pandas as pd
import vectorbt as vbt

import strategy_engine as engine
import config

def _inject_genes_into_rules(base_rules: dict, gene_map: dict, solution: list) -> dict:
    """
    Injects the gene values from a GA solution into a copy of the strategy rules.
    """
    injected_rules = copy.deepcopy(base_rules)
    for i, gene_value in enumerate(solution):
        gene_info = gene_map.get(i)
        if not gene_info:
            continue

        path = gene_info.get("path", [])
        if not path:
            # When path is empty, there's nowhere to inject the gene.
            # This can occur in tests that mock gene parsing; skip.
            continue

        current_level = injected_rules

        for key in path[:-1]:
            current_level = current_level[key]

        param_key = path[-1]

        expected_type = gene_info.get("type")
        if expected_type is int:
            gene_value = int(gene_value)
        elif expected_type is float:
            gene_value = float(gene_value)

        current_level[param_key] = gene_value

    return injected_rules


class FitnessEvaluator:
    def __init__(self, ohlc_data: pd.DataFrame, base_rules: dict, gene_map: dict):
        self.ohlc_data = ohlc_data
        self.base_rules = base_rules
        self.gene_map = gene_map

    def __call__(self, ga_instance, solution, sol_idx):
        try:
            rules = _inject_genes_into_rules(self.base_rules, self.gene_map, solution)
            entries = engine.process_strategy_rules(self.ohlc_data, rules)
            
            trade_count = entries.sum().sum() if isinstance(entries, pd.DataFrame) else entries.sum()
            if trade_count < config.FITNESS_WEIGHTS['min_trades']:
                return -1.0

            # --- NEW: Logic to handle multiple, selectable exit types ---
            exit_rules = rules.get('exit_rules', {})
            sl_rule = exit_rules.get('stop_loss', {})
            tsl_rule = exit_rules.get('trailing_stop', {})
            tp_rule = exit_rules.get('take_profit', {})

            sl_stop = sl_rule.get('params', {}).get('value') if sl_rule.get('is_active', False) else None
            sl_trail = tsl_rule.get('params', {}).get('value') if tsl_rule.get('is_active', False) else None
            tp_stop = tp_rule.get('params', {}).get('value') if tp_rule.get('is_active', False) else None
            
            time_based_exit = entries.shift(config.MAX_HOLD_PERIOD, fill_value=False)
            time_based_exit = time_based_exit.reindex(entries.index, fill_value=False)

            close_prices = (
                self.ohlc_data['Close']
                if 'Close' in self.ohlc_data
                else self.ohlc_data.xs('Close', level=-1, axis=1)
            )

            portfolio = vbt.Portfolio.from_signals(
                close=close_prices,
                entries=entries,
                exits=time_based_exit,
                sl_stop=sl_stop,
                tp_stop=tp_stop,
                sl_trail=sl_trail,  # Pass the trailing stop value to the backtester
                fees=0.001,
                freq=config.TIMEFRAME,
            )

            stats = portfolio.stats(agg_func=None)

            # ``portfolio.stats`` returns a Series for single-column inputs and a
            # DataFrame for multi-column inputs.  The latter caused the fitness
            # function to work with DataFrames, eventually raising an exception
            # when applying scalar operations (e.g. ``profit_factor > 5``) and
            # terminating the worker process.  Reduce multi-column stats by
            # averaging across columns so we always operate on scalars.
            if isinstance(stats, pd.DataFrame):
                stats = stats.mean(axis=1, numeric_only=True)

            sortino = stats.get('Sortino Ratio', np.nan)
            profit_factor = stats.get('Profit Factor', np.nan)
            max_drawdown = stats.get('Max Drawdown [%]', np.nan)

            if np.isinf(profit_factor) or profit_factor > 5:
                profit_factor = 5

            drawdown_score = 1 - (max_drawdown / 100.0)

            if np.isnan(sortino) or sortino == 0:
                logging.warning(
                    "Penalizing fitness due to invalid Sortino Ratio: %s", sortino
                )
                return -999.0
            if np.isnan(profit_factor) or profit_factor == 0:
                logging.warning(
                    "Penalizing fitness due to invalid Profit Factor: %s", profit_factor
                )
                return -999.0
            if np.isnan(drawdown_score) or drawdown_score == 0:
                logging.warning(
                    "Penalizing fitness due to invalid Drawdown Score: %s", drawdown_score
                )
                return -999.0

            weights = config.FITNESS_WEIGHTS

            fitness_score = (
                (sortino * weights['sortino_ratio']) +
                (profit_factor * weights['profit_factor']) +
                (drawdown_score * weights['max_drawdown'])
            )

            return fitness_score if not np.isnan(fitness_score) else -1.0

        except Exception as e:
            print(f"Error in fitness evaluation: {e}")
            return -999.0
