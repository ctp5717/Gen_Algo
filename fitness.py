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


def weighted_mean_std(values, weights):
    """Compute weighted mean and standard deviation.

    Parameters
    ----------
    values : array-like
        Sequence of values :math:`m_i`.
    weights : array-like
        Corresponding weights :math:`u_i`. They do not need to be normalised.

    Returns
    -------
    tuple of float
        ``(mu, sigma)`` where ``mu`` is the weighted mean and ``sigma`` is the
        weighted *population* standard deviation :math:`\sqrt{\sum u_i(m_i-\mu)^2}`.
        Weights are normalised internally so that ``sum(u_i)=1``. This helper
        is shared by evaluators and tests to guarantee consistent dispersion
        calculations across the project.
    """

    w = np.asarray(weights, dtype=float)
    if w.ndim == 0:
        w = np.array([float(w)])
    total = w.sum()
    if total == 0:
        w = np.ones_like(w) / len(w)
    else:
        w = w / total
    x = np.asarray(values, dtype=float)
    mu = float(np.sum(w * x))
    variance = float(np.sum(w * (x - mu) ** 2))
    sigma = float(np.sqrt(variance))
    return mu, sigma

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
                sl_trail=sl_trail, # Pass the trailing stop value to the backtester
                fees=0.001,
                freq=config.TIMEFRAME
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


class MultiAssetFitnessEvaluator:
    """Evaluate a candidate solution across multiple assets.

    The evaluator computes the per-asset composite metric using the same
    recipe as :class:`FitnessEvaluator` and then aggregates the results using a
    dispersion penalty.  The behaviour is governed by ``config.MULTI_ASSET``
    but can be overridden by passing a custom ``settings`` dictionary.
    """

    def __init__(self, group_data: dict, base_rules: dict, gene_map: dict, settings: dict | None = None):
        self.group_data = group_data  # dict[ticker -> OHLCV DataFrame]
        self.base_rules = base_rules
        self.gene_map = gene_map
        defaults = getattr(config, "MULTI_ASSET", {})
        self.settings = copy.deepcopy(defaults)
        if settings:
            self.settings.update(settings)
        self.last_details = {}

    # ------------------------------------------------------------------
    def _evaluate_single_asset(self, ohlc: pd.DataFrame, rules: dict) -> dict:
        """Run the strategy on a single asset and return raw statistics."""
        entries = engine.process_strategy_rules(ohlc, rules)

        # Record the actual executed trades using vectorbt.
        exit_rules = rules.get("exit_rules", {})
        sl_rule = exit_rules.get("stop_loss", {})
        tsl_rule = exit_rules.get("trailing_stop", {})
        tp_rule = exit_rules.get("take_profit", {})

        sl_stop = sl_rule.get("params", {}).get("value") if sl_rule.get("is_active", False) else None
        sl_trail = tsl_rule.get("params", {}).get("value") if tsl_rule.get("is_active", False) else None
        tp_stop = tp_rule.get("params", {}).get("value") if tp_rule.get("is_active", False) else None

        time_exit = entries.shift(config.MAX_HOLD_PERIOD, fill_value=False)
        time_exit = time_exit.reindex(entries.index, fill_value=False)

        portfolio = vbt.Portfolio.from_signals(
            close=ohlc["Close"],
            entries=entries,
            exits=time_exit,
            sl_stop=sl_stop,
            tp_stop=tp_stop,
            sl_trail=sl_trail,
            fees=0.001,
            freq=config.TIMEFRAME,
        )

        trades = int(portfolio.trades.count())
        if trades == 0:
            return {
                "sortino": None,
                "profit_factor": None,
                "max_drawdown": None,
                "trades": 0,
                "total_return": None,
                "equity_curve": portfolio.value(),
            }

        stats = portfolio.stats()
        return {
            "sortino": stats.get("Sortino Ratio"),
            "profit_factor": stats.get("Profit Factor"),
            "max_drawdown": stats.get("Max Drawdown [%]"),
            "trades": trades,
            "total_return": stats.get("Total Return [%]"),
            "equity_curve": portfolio.value(),
        }

    # ------------------------------------------------------------------
    def __call__(self, ga_instance, solution, sol_idx):
        try:
            rules = _inject_genes_into_rules(self.base_rules, self.gene_map, solution)

            per_asset_metrics = []
            included_assets = []
            per_asset_details = {}
            total_trades = 0

            for ticker, ohlc in self.group_data.items():
                stats = self._evaluate_single_asset(ohlc, rules)
                trades = stats.get("trades", 0)
                total_trades += trades

                if trades < self.settings.get("per_asset_min_trades", 1):
                    if self.settings.get("zero_trade_policy") == "penalize":
                        val = self.settings.get("zero_trade_penalty", -1.0)
                        per_asset_metrics.append(val)
                        included_assets.append(ticker)
                        per_asset_details[ticker] = {
                            **stats,
                            "score": val,
                            "included": True,
                        }
                    else:
                        per_asset_details[ticker] = {
                            **stats,
                            "score": None,
                            "included": False,
                        }
                        continue
                else:
                    metric_type = self.settings.get("metric", "composite")
                    if metric_type == "sortino":
                        val = stats.get("sortino", self.settings.get("nan_fallback", 0.0))
                    elif metric_type == "profit_factor":
                        val = stats.get("profit_factor", self.settings.get("nan_fallback", 0.0))
                    elif metric_type == "return":
                        val = stats.get("total_return", self.settings.get("nan_fallback", 0.0))
                    else:  # composite metric
                        sortino = stats.get("sortino")
                        pf = stats.get("profit_factor")
                        dd = stats.get("max_drawdown")

                        cap = self.settings.get("winsorize_pf_cap", 5.0)
                        if pf is None or np.isnan(pf):
                            pf = self.settings.get("nan_fallback", 0.0)
                        pf = min(cap, pf) if not np.isinf(pf) else cap

                        if sortino is None or np.isnan(sortino):
                            sortino = self.settings.get("nan_fallback", 0.0)

                        if dd is None or np.isnan(dd):
                            dd = 100.0
                        drawdown_score = 1 - (dd / 100.0)

                        w = config.FITNESS_WEIGHTS
                        val = (
                            sortino * w["sortino_ratio"]
                            + pf * w["profit_factor"]
                            + drawdown_score * w["max_drawdown"]
                        )

                    per_asset_metrics.append(val)
                    included_assets.append(ticker)
                    per_asset_details[ticker] = {
                        **stats,
                        "score": val,
                        "included": True,
                    }

            if not per_asset_metrics:
                poor = self.settings.get("poor_score", -999.0)
                min_trades = self.settings.get("min_total_trades", 0)
                policy = self.settings.get("trade_floor_policy", "hard_floor")
                trade_penalty = None
                F = poor
                if total_trades < min_trades:
                    if policy == "hard_floor":
                        trade_penalty = "hard_floor"
                    elif policy == "soft_penalty":
                        mode = self.settings.get("soft_penalty_mode", "multiplicative")
                        strength = self.settings.get("soft_penalty_strength", 1.0)
                        if mode == "additive":
                            penalty = strength * (1 - total_trades / max(1, min_trades))
                            F -= penalty
                            trade_penalty = {"mode": "additive", "penalty": penalty}
                        else:
                            scale = (total_trades / max(1, min_trades)) ** strength
                            F *= scale
                            trade_penalty = {"mode": "multiplicative", "scale": scale}
                self.last_details = {
                    "per_asset": per_asset_details,
                    "mu": None,
                    "sigma": None,
                    "lambda_sigma": None,
                    "total_trades": total_trades,
                    "assets_included": 0,
                    "assets_ignored": len(self.group_data),
                    "penalties": {"trade_floor": trade_penalty, "coverage": None},
                    "fitness": F,
                }
                return F

            # Determine weights for included assets and renormalise
            asset_weights = self.settings.get("asset_weights") or {}
            weights = [asset_weights.get(t, 1.0) for t in included_assets]
            weight_sum = sum(weights)
            if weight_sum == 0:
                weights = [1.0 / len(per_asset_metrics)] * len(per_asset_metrics)
            else:
                weights = [w / weight_sum for w in weights]

            m_arr = np.array(per_asset_metrics, dtype=float)
            w_arr = np.array(weights, dtype=float)
            mu, sigma = weighted_mean_std(m_arr, w_arr)

            lam = self.settings.get("lambda_dispersion", 0.0)
            F = mu - lam * sigma

            min_trades = self.settings.get("min_total_trades", 0)
            policy = self.settings.get("trade_floor_policy", "hard_floor")
            poor_score = self.settings.get("poor_score", -999.0)
            trade_penalty = None
            coverage_penalty = 0.0

            if policy == "hard_floor" and total_trades < min_trades:
                F = poor_score
                trade_penalty = "hard_floor"
            else:
                if policy == "soft_penalty" and total_trades < min_trades:
                    mode = self.settings.get("soft_penalty_mode", "multiplicative")
                    strength = self.settings.get("soft_penalty_strength", 1.0)
                    if mode == "additive":
                        penalty = strength * (
                            1 - total_trades / max(1, min_trades)
                        )
                        F -= penalty
                        trade_penalty = {"mode": "additive", "penalty": penalty}
                    else:
                        scale = (total_trades / max(1, min_trades)) ** strength
                        F *= scale
                        trade_penalty = {"mode": "multiplicative", "scale": scale}

                if (
                    self.settings.get("zero_trade_policy") == "ignore"
                    and self.settings.get("coverage_penalty_weight") is not None
                ):
                    weight = self.settings.get("coverage_penalty_weight")
                    coverage = len(included_assets) / max(1, len(self.group_data))
                    coverage_penalty = weight * (1 - coverage)
                    F -= coverage_penalty

            # store diagnostics for optional inspection
            self.last_details = {
                "per_asset": per_asset_details,
                "mu": mu,
                "sigma": sigma,
                "lambda_sigma": lam * sigma,
                "total_trades": total_trades,
                "assets_included": len(included_assets),
                "assets_ignored": len(self.group_data) - len(included_assets),
                "penalties": {
                    "trade_floor": trade_penalty,
                    "coverage": coverage_penalty,
                },
                "fitness": F,
            }

            return F

        except Exception as e:
            print(f"Error in multi-asset fitness evaluation: {e}")
            poor = self.settings.get("poor_score", -999.0)
            self.last_details = {
                "per_asset": {},
                "mu": None,
                "sigma": None,
                "lambda_sigma": None,
                "total_trades": 0,
                "assets_included": 0,
                "assets_ignored": len(self.group_data),
                "penalties": {"trade_floor": "error", "coverage": None},
                "fitness": poor,
            }
            return poor


def get_fitness_evaluator(ohlc_data, base_rules, gene_map):
    """Factory returning the appropriate fitness evaluator.

    Parameters
    ----------
    ohlc_data : pd.DataFrame or dict
        If ``config.MULTI_ASSET['enabled']`` is True, ``ohlc_data`` should be a
        mapping of ticker -> DataFrame.  Otherwise it is a single DataFrame.
    """

    settings = getattr(config, "MULTI_ASSET", {})
    if settings.get("enabled") and isinstance(ohlc_data, dict):
        return MultiAssetFitnessEvaluator(ohlc_data, base_rules, gene_map, settings)
    return FitnessEvaluator(ohlc_data, base_rules, gene_map)
