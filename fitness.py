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
from utils.dataframe_util import assert_monotonic_datetime_index

EPSILON = 1e-09


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


def _calc_stats(returns: pd.Series) -> tuple[float, float, float]:
    logger = get_logger(__name__)
    equity = (1 + returns).cumprod()
    running_max = equity.cummax()
    drawdown = (equity / running_max) - 1.0
    max_dd = -drawdown.min() * 100
    pos = returns[returns > 0].sum()
    neg = returns[returns < 0].sum()
    denom_pf = abs(neg)
    if denom_pf == 0.0:
        logger.debug("Profit factor denominator was zero; using EPSILON fallback")
    denom_pf = denom_pf if denom_pf != 0.0 else EPSILON
    profit_factor = pos / denom_pf
    downside = returns[returns < 0]
    downside_std = downside.std(ddof=0)
    if downside_std == 0.0:
        logger.debug("Sortino denominator was zero; using EPSILON fallback")
    downside_std = downside_std if downside_std != 0.0 else EPSILON
    sortino = returns.mean() / downside_std
    return sortino, profit_factor, max_dd


def _clamp_metrics(
    sortino: float, profit_factor: float, max_dd: float
) -> tuple[float, float, float]:
    sortino = (
        float(np.clip(sortino, -5.0, 5.0)) if np.isfinite(sortino) else 0.0
    )
    profit_factor = (
        float(np.clip(profit_factor, 0.0, 5.0)) if np.isfinite(profit_factor) else 0.0
    )
    max_dd = float(np.clip(max_dd, 0.0, 100.0)) if np.isfinite(max_dd) else 100.0
    return sortino, profit_factor, max_dd


class FitnessEvaluator:
    def __init__(self, ohlc_data: pd.DataFrame, base_rules: dict, gene_map: dict):
        assert_monotonic_datetime_index(ohlc_data, "ohlc_data")
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
            shifted_entries = entries.shift(config.ENTRY_LAG_BARS, fill_value=False)

            if shifted_entries.sum() < config.FITNESS_WEIGHTS['min_trades']:
                return -1.0

            # --- NEW: Logic to handle multiple, selectable exit types ---
            exit_rules = rules.get('exit_rules', {})
            sl_rule = exit_rules.get('stop_loss', {})
            tsl_rule = exit_rules.get('trailing_stop', {})
            tp_rule = exit_rules.get('take_profit', {})

            sl_stop = sl_rule.get('params', {}).get('value') if sl_rule.get('is_active', False) else None
            sl_trail = tsl_rule.get('params', {}).get('value') if tsl_rule.get('is_active', False) else None
            tp_stop = tp_rule.get('params', {}).get('value') if tp_rule.get('is_active', False) else None

            time_based_exit = shifted_entries.shift(config.MAX_HOLD_PERIOD, fill_value=False)
            time_based_exit = time_based_exit.reindex(entries.index, fill_value=False)

            portfolio = vbt.Portfolio.from_signals(
                close=self.ohlc_data['Close'],
                entries=shifted_entries,
                exits=time_based_exit,
                sl_stop=sl_stop,
                tp_stop=tp_stop,
                sl_trail=sl_trail,  # Pass the trailing stop value to the backtester
                fees=config.FEES,
                freq=config.TIMEFRAME
            )

            returns = portfolio.returns()
            sortino, profit_factor, max_drawdown = _calc_stats(returns)
            sortino, profit_factor, max_drawdown = _clamp_metrics(
                sortino, profit_factor, max_drawdown
            )

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
