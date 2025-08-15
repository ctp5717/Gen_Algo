# fitness.py

"""
Fitness Function for Genetic Algorithm
(This version uses the correct pandas .shift() method for time-based exits)
"""
import copy
import pandas as pd
import numpy as np
import vectorbt as vbt
import strategy_engine as engine
import config
from utils.warnings_util import suppress_third_party_warnings
from utils.logging_util import get_logger, OncePerGenerationErrors

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

        current_level[param_key] = gene_value

    return injected_rules


get_logger(__name__)
error_tracker = OncePerGenerationErrors()
VERY_LOW_FITNESS = -999.0


class FitnessEvaluator:
    def __init__(self, ohlc_data: pd.DataFrame, base_rules: dict, gene_map: dict):
        self.ohlc_data = ohlc_data
        self.base_rules = base_rules
        self.gene_map = gene_map
        self.error_tracker = error_tracker

    def __call__(self, ga_instance, solution, sol_idx):
        logger = get_logger(__name__)
        suppress_third_party_warnings()
        try:
            rules = _inject_genes_into_rules(self.base_rules, self.gene_map, solution)
            entries = engine.process_strategy_rules(self.ohlc_data, rules)

            if entries.sum() < config.FITNESS_WEIGHTS['min_trades']:
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

            portfolio = vbt.Portfolio.from_signals(
                close=self.ohlc_data['Close'],
                entries=entries,
                exits=time_based_exit,
                sl_stop=sl_stop,
                tp_stop=tp_stop,
                sl_trail=sl_trail,  # Pass the trailing stop value to the backtester
                fees=config.FEES,
                freq=config.TIMEFRAME
            )

            stats = portfolio.stats()
            sortino = stats['Sortino Ratio']
            profit_factor = stats['Profit Factor']
            max_drawdown = stats['Max Drawdown [%]']

            if np.isinf(profit_factor) or profit_factor > 5:
                profit_factor = 5
            if np.isnan(sortino):
                sortino = 0
            if np.isnan(profit_factor):
                profit_factor = 0
            if np.isnan(max_drawdown):
                max_drawdown = 100.0

            drawdown_score = 1 - (max_drawdown / 100.0)
            weights = config.FITNESS_WEIGHTS

            fitness_score = (
                (sortino * weights['sortino_ratio']) +
                (profit_factor * weights['profit_factor']) +
                (drawdown_score * weights['max_drawdown'])
            )

            return fitness_score if not np.isnan(fitness_score) else -1.0

        except Exception as e:
            error_tracker.log_exception(logger, "Fitness evaluation failed", e)
            return VERY_LOW_FITNESS
