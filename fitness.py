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


def _get_exit_param(rule: dict) -> float | None:
    """Return a numeric exit parameter or ``None`` if not available.

    The configuration loader expresses optimisable parameters as nested
    dictionaries (``{"gene": ..., "low": ...}``).  If such a dictionary is
    passed through without gene injection, ``vectorbt`` later attempts to treat
    it like a pandas object and fails with ``'dict' object has no attribute
    'index'``.  To guard against this, gracefully fall back to ``None`` whenever
    the value is not already numeric.
    """

    value = rule.get("params", {}).get("value")
    return float(value) if isinstance(value, (int, float, np.number)) else None


def _build_exit_kwargs(exit_rules: dict) -> dict:
    """Extract numeric exit parameters for :func:`vectorbt.Portfolio.from_signals`.

    The backtester expects plain numbers for stop-loss, trailing stop and take-profit
    thresholds.  When gene injection fails the configuration may still contain the
    original gene dictionaries which would otherwise trigger errors like
    ``'dict' object has no attribute 'index'`` inside vectorbt.  This helper filters
    out any non-numeric values and only returns kwargs for the parameters that are
    both active and numeric.
    """

    sl_rule = exit_rules.get("stop_loss", {})
    tsl_rule = exit_rules.get("trailing_stop", {})
    tp_rule = exit_rules.get("take_profit", {})

    sl_stop = _get_exit_param(sl_rule) if sl_rule.get("is_active", False) else None
    sl_trail = _get_exit_param(tsl_rule) if tsl_rule.get("is_active", False) else None
    tp_stop = _get_exit_param(tp_rule) if tp_rule.get("is_active", False) else None

    kwargs = {}
    if sl_stop is not None:
        kwargs["sl_stop"] = sl_stop
    if sl_trail is not None:
        kwargs["sl_trail"] = sl_trail
    if tp_stop is not None:
        kwargs["tp_stop"] = tp_stop
    return kwargs

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


class FitnessEvaluator:
    def __init__(self, ohlc_data: pd.DataFrame, base_rules: dict, gene_map: dict):
        self.ohlc_data = ohlc_data
        self.base_rules = base_rules
        self.gene_map = gene_map

    def __call__(self, ga_instance, solution, sol_idx):
        try:
            rules = _inject_genes_into_rules(self.base_rules, self.gene_map, solution)
            entries = engine.process_strategy_rules(self.ohlc_data, rules)
            
            if entries.sum() < config.FITNESS_WEIGHTS['min_trades']:
                return -1.0

            # Build exit-rule kwargs, skipping any unresolved gene dictionaries
            exit_kwargs = _build_exit_kwargs(rules.get("exit_rules", {}))

            time_based_exit = entries.shift(config.MAX_HOLD_PERIOD, fill_value=False)
            time_based_exit = time_based_exit.reindex(entries.index, fill_value=False)

            portfolio = vbt.Portfolio.from_signals(
                close=self.ohlc_data['Close'],
                entries=entries,
                exits=time_based_exit,
                fees=config.FEES,
                freq=config.TIMEFRAME,
                **exit_kwargs,
            )
            
            stats = portfolio.stats()
            sortino = stats['Sortino Ratio']
            profit_factor = stats['Profit Factor']
            max_drawdown = stats['Max Drawdown [%]']
            
            if np.isinf(profit_factor) or profit_factor > 5: profit_factor = 5
            if np.isnan(sortino): sortino = 0
            if np.isnan(profit_factor): profit_factor = 0
            if np.isnan(max_drawdown): max_drawdown = 100.0

            drawdown_score = 1 - (max_drawdown / 100.0)
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
