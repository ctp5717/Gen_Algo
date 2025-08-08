import copy
from typing import Dict
import numpy as np
import pandas as pd
import vectorbt as vbt
import strategy_engine as engine
import config

def _inject_genes_into_rules(base_rules: dict, gene_map: Dict[int, dict], solution: list) -> dict:
    injected_rules = copy.deepcopy(base_rules)
    for i, gene_value in enumerate(solution):
        gene_info = gene_map.get(i)
        if not gene_info:
            continue
        path = gene_info.get('path', [])
        current_level = injected_rules
        for key in path[:-1]:
            current_level = current_level[key]
        current_level[path[-1]] = gene_value
    return injected_rules

class FitnessEvaluator:
    def __init__(self, ohlc_data: pd.DataFrame, base_rules: dict, gene_map: Dict[int, dict]):
        self.ohlc_data = ohlc_data
        self.base_rules = base_rules
        self.gene_map = gene_map

    def __call__(self, ga_instance, solution, sol_idx):
        try:
            rules = _inject_genes_into_rules(self.base_rules, self.gene_map, solution)
            entries = engine.process_strategy_rules(self.ohlc_data, rules)

            total_trades = entries.astype(bool).values.sum() if isinstance(entries, pd.DataFrame) else int(entries.sum())
            if total_trades < config.FITNESS_WEIGHTS['min_trades']:
                return -1.0

            exit_rules = rules.get('exit_rules', {}) or {}
            sl_rule = exit_rules.get('stop_loss', {}) or {}
            tsl_rule = exit_rules.get('trailing_stop', {}) or {}
            tp_rule = exit_rules.get('take_profit', {}) or {}

            sl_stop = sl_rule.get('params', {}).get('value') if sl_rule.get('is_active', False) else None
            sl_trail = tsl_rule.get('params', {}).get('value') if tsl_rule.get('is_active', False) else None
            tp_stop = tp_rule.get('params', {}).get('value') if tp_rule.get('is_active', False) else None

            time_based_exit = entries.shift(config.MAX_HOLD_PERIOD, fill_value=False).reindex(entries.index, fill_value=False)

            if isinstance(self.ohlc_data.columns, pd.MultiIndex):
                close_prices = self.ohlc_data.xs('Close', level=1, axis=1)
            else:
                close_prices = self.ohlc_data['Close']

            portfolio = vbt.Portfolio.from_signals(
                close=close_prices,
                entries=entries,
                exits=time_based_exit,
                sl_stop=sl_stop,
                tp_stop=tp_stop,
                sl_trail=sl_trail,
                fees=0.001,
                freq=config.TIMEFRAME,
            )
            stats = portfolio.stats()
            def _get(obj, key, default=np.nan):
                try:
                    return obj[key] if isinstance(obj, dict) else obj.get(key, default)
                except Exception:
                    return default
            sortino = _get(stats, 'Sortino Ratio', 0.0)
            profit_factor = _get(stats, 'Profit Factor', 0.0)
            max_drawdown = _get(stats, 'Max Drawdown [%]', 100.0)

            if np.isinf(profit_factor) or profit_factor > 5:
                profit_factor = 5
            sortino = 0 if np.isnan(sortino) else sortino
            profit_factor = 0 if np.isnan(profit_factor) else profit_factor
            max_drawdown = 100.0 if np.isnan(max_drawdown) else max_drawdown

            drawdown_score = 1 - (max_drawdown / 100.0)
            w = config.FITNESS_WEIGHTS
            fitness_score = (sortino * w['sortino_ratio']) + (profit_factor * w['profit_factor']) + (drawdown_score * w['max_drawdown'])
            return fitness_score if not np.isnan(fitness_score) else -1.0
        except Exception as err:
            print(f"Error in fitness evaluation: {err}")
            return -999.0
