# fitness.py

"""
Fitness Function for Genetic Algorithm
(This version uses the correct pandas .shift() method for time-based exits)
"""
import copy
import math
from collections import deque
import hashlib
import json

import pandas as pd
import numpy as np
import vectorbt as vbt
import strategy_engine as engine
import config
import data_loader
from utils import _norm_freq


_EVAL_CACHE: dict[tuple[str, str], dict] = {}


def format_settings_summary(settings: dict, coverage_threshold: float | None = None) -> str:
    """Return a human-friendly summary of multi-asset settings."""

    lam = settings.get("lambda_dispersion")
    pf_cap = settings.get("pf_cap")
    sortino_cap = settings.get("sortino_cap")
    squash = settings.get("squash")
    if squash:
        params = settings.get("squash_params", {})
        squash_repr = f"on {params}"
    else:
        squash_repr = "off"
    floor = settings.get("min_total_trades")
    cov = coverage_threshold
    cov_str = f"{cov * 100:.0f}%" if cov is not None else "n/a"
    return (
        "objective=μ−λσ | "
        f"λ={lam} | PF_CAP={pf_cap} | SORTINO_CAP={sortino_cap} | "
        f"squash={squash_repr} | effective_floor={floor} | coverage_threshold={cov_str}"
    )


def print_settings_summary(settings: dict, coverage_threshold: float | None = None) -> None:
    """Print a one-line summary of evaluator settings."""

    print(format_settings_summary(settings, coverage_threshold))


def _hash_rules(rules: dict) -> str:
    """Create a stable hash for a nested rules dictionary."""
    dumped = json.dumps(rules, sort_keys=True, default=str)
    return hashlib.sha256(dumped.encode()).hexdigest()


def weighted_mean_std(values, weights):
    r"""Compute weighted mean and standard deviation.

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
                sl_trail=sl_trail,  # Pass the trailing stop value to the backtester
                fees=0.001,
                freq=_norm_freq(config.TIMEFRAME)
            )
            
            stats = portfolio.stats()
            sortino = stats['Sortino Ratio']
            profit_factor = stats['Profit Factor']
            max_drawdown = stats['Max Drawdown [%]']
            
            if np.isinf(profit_factor) or profit_factor > config.PF_CAP:
                profit_factor = config.PF_CAP
            if np.isnan(sortino):
                sortino = 0
            elif np.isinf(sortino) or sortino > config.SORTINO_CAP:
                sortino = config.SORTINO_CAP
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
        user_settings = settings or {}
        if user_settings:
            self.settings.update(user_settings)
        legacy = {
            "coverage_penalty_lambda": "coverage_penalty_kappa",
            "min_trades_per_year": "min_total_trades_per_year",
        }
        for old, new in legacy.items():
            if old in user_settings and new not in user_settings:
                self.settings[new] = user_settings[old]
        # Store original floor and trade history settings for dynamic scaling
        self._base_min_total_trades = self.settings.get("min_total_trades", 0)
        self._max_total_trades = self.settings.get("max_total_trades")
        window = self.settings.get("trade_floor_window", 5)
        try:
            self._recent_totals: deque[int] = deque(maxlen=int(window) if window else 0)
        except Exception:
            self._recent_totals = deque(maxlen=0)
        self._current_generation = None
        self._current_gen_scores: list[tuple[float, int]] = []
        rate = self.settings.get("min_total_trades_per_year")
        if rate:
            try:
                starts: list[pd.Timestamp] = []
                ends: list[pd.Timestamp] = []
                for df in self.group_data.values():
                    if not df.empty:
                        starts.append(df.index[0])
                        ends.append(df.index[-1])
                if starts and ends:
                    years = (max(ends) - min(starts)).days / 365.25
                    floor = math.ceil(rate * max(years, 0))
                    self.settings["min_total_trades"] = floor
                    print(f"[MultiAssetFitnessEvaluator] trade floor={floor}")
            except Exception:
                pass
        self.last_details = {}
        # Store exclusions produced during data loading so they can be
        # surfaced later in diagnostic reports.
        try:
            self.coverage_exclusions = data_loader.get_last_excluded_assets()
        except Exception:
            self.coverage_exclusions = []
        print_settings_summary(self.settings, getattr(config, "COVERAGE_THRESHOLD", None))

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
            freq=_norm_freq(config.TIMEFRAME),
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
            # Detect generation boundaries and update the dynamic trade floor
            gen = getattr(ga_instance, "generations_completed", self._current_generation)
            if self._current_generation is None:
                self._current_generation = gen
            elif gen != self._current_generation:
                if self._current_gen_scores:
                    best_trades = max(self._current_gen_scores, key=lambda x: x[0])[1]
                    if self._recent_totals.maxlen:
                        self._recent_totals.append(best_trades)
                adaptive = self.settings.get("adaptive_floor", {})
                if adaptive.get("enabled") and self._recent_totals:
                    arr = np.array(self._recent_totals, dtype=float)
                    pct = adaptive.get("percentile", 50)
                    candidate = np.percentile(arr, pct)
                    bounds = adaptive.get("bounds", {})
                    b_min = bounds.get("min")
                    b_max = bounds.get("max")
                    if b_min is not None:
                        candidate = max(candidate, b_min)
                    candidate = max(self._base_min_total_trades, candidate)
                    if b_max is not None:
                        candidate = min(candidate, b_max)
                    if self._max_total_trades is not None:
                        candidate = min(candidate, self._max_total_trades)
                    effective_floor = int(round(candidate))
                    self.settings["min_total_trades"] = effective_floor
                    print(f"[MultiAssetFitnessEvaluator] effective_floor={effective_floor}")
                self._current_gen_scores.clear()
                self._current_generation = gen

            rules = _inject_genes_into_rules(self.base_rules, self.gene_map, solution)
            rules_hash = _hash_rules(rules)

            per_asset_metrics = []
            included_assets = []
            per_asset_details = {}
            total_trades = 0
            clip_abs = self.settings.get("clip_composite_abs")
            zero_trade_exclusions: list[dict] = []

            shrink_cfg = self.settings.get("low_trade_shrink") or {}
            shrink_enabled = shrink_cfg.get("enabled", False)
            base_k = shrink_cfg.get("k")
            s = shrink_cfg.get("s", 1.0)
            override_k = self.settings.get("partial_trades_threshold")
            if override_k is not None:
                base_k = override_k
                shrink_enabled = True
                s = self.settings.get("partial_trades_exponent", s)
            elif self.settings.get("partial_trades_exponent") is not None and base_k is None:
                s = self.settings.get("partial_trades_exponent", s)

            for ticker, ohlc in self.group_data.items():
                cache_key = (ticker, rules_hash)
                stats = _EVAL_CACHE.get(cache_key)
                if stats is None:
                    stats = self._evaluate_single_asset(ohlc, rules)
                    _EVAL_CACHE[cache_key] = stats
                trades = stats.get("trades", 0)
                total_trades += trades

                per_asset_min_trades = self.settings.get("per_asset_min_trades", 1)
                k = base_k
                if k is not None:
                    k = max(k, per_asset_min_trades)
                insufficient = trades < per_asset_min_trades

                # Pre-compute capped metrics and drawdown score for storage
                pf_raw = stats.get("profit_factor")
                pf_cap = self.settings.get(
                    "pf_cap", self.settings.get("winsorize_pf_cap")
                )
                if pf_raw is None or np.isnan(pf_raw):
                    pf_capped = self.settings.get("nan_fallback", 0.0)
                else:
                    if pf_cap is not None:
                        pf_capped = (
                            min(pf_cap, pf_raw) if not np.isinf(pf_raw) else pf_cap
                        )
                    else:
                        pf_capped = pf_raw

                sortino_raw = stats.get("sortino")
                sortino_cap = self.settings.get("sortino_cap")
                if sortino_raw is None or np.isnan(sortino_raw):
                    sortino_capped = self.settings.get("nan_fallback", 0.0)
                else:
                    if sortino_cap is not None:
                        sortino_capped = (
                            min(sortino_cap, sortino_raw)
                            if not np.isinf(sortino_raw)
                            else sortino_cap
                        )
                    else:
                        sortino_capped = sortino_raw

                dd_raw = stats.get("max_drawdown")
                if dd_raw is None or np.isnan(dd_raw):
                    dd_raw = 100.0
                drawdown_score = 1 - (dd_raw / 100.0)

                penalties = {}

                shrinkage_info = {
                    "enabled": shrink_enabled,
                    "k": k,
                    "s": s,
                    "applied": False,
                    "multiplier": None,
                }

                # Skip aggregation if zero trades and policy is to ignore
                if (
                    trades == 0
                    and self.settings.get("zero_trade_policy") == "ignore"
                ):
                    per_asset_details[ticker] = {
                        **stats,
                        "score": None,
                        "included": False,
                        "insufficient": True,
                        "sortino_capped": sortino_capped,
                        "profit_factor_capped": pf_capped,
                        "drawdown_score": drawdown_score,
                        "shrinkage_multiplier": None,
                        "shrinkage": shrinkage_info,
                        "penalties": None,
                        "caps": {
                            "profit_factor": {
                                "raw": pf_raw,
                                "cap": pf_cap,
                                "capped": pf_capped,
                            },
                            "sortino": {
                                "raw": sortino_raw,
                                "cap": sortino_cap,
                                "capped": sortino_capped,
                            },
                        },
                    }
                    zero_trade_exclusions.append({"ticker": ticker, "reason": "zero_trades"})
                    continue

                if self.settings.get("squash"):
                    params = self.settings.get("squash_params", {})
                    sortino_c = params.get("sortino_c", 1.0)
                    pf_c = params.get("pf_c", 1.0)
                    sortino_capped = float(np.tanh(sortino_capped / sortino_c))
                    pf_capped = float(np.tanh(pf_capped / pf_c))

                metric_type = self.settings.get("metric", "composite")
                if metric_type == "sortino":
                    val = sortino_capped
                elif metric_type == "profit_factor":
                    val = pf_capped
                elif metric_type == "return":
                    val = stats.get("total_return", self.settings.get("nan_fallback", 0.0))
                else:  # composite metric
                    w = config.FITNESS_WEIGHTS
                    val = (
                        sortino_capped * w["sortino_ratio"]
                        + pf_capped * w["profit_factor"]
                        + drawdown_score * w["max_drawdown"]
                    )

                shrinkage_multiplier = None
                if shrink_enabled and k is not None and 1 <= trades < k:
                    shrinkage_multiplier = (trades / k) ** s
                    val *= shrinkage_multiplier
                    shrinkage_info["applied"] = True
                    shrinkage_info["multiplier"] = shrinkage_multiplier

                if clip_abs is not None:
                    val = float(np.clip(val, -abs(clip_abs), abs(clip_abs)))

                per_asset_metrics.append(val)
                included_assets.append(ticker)
                per_asset_details[ticker] = {
                    **stats,
                    "score": val,
                    "included": True,
                    "insufficient": insufficient,
                    "sortino_capped": sortino_capped,
                    "profit_factor_capped": pf_capped,
                    "drawdown_score": drawdown_score,
                    "shrinkage_multiplier": shrinkage_multiplier,
                    "shrinkage": shrinkage_info,
                    "penalties": penalties or None,
                    "caps": {
                        "profit_factor": {
                            "raw": pf_raw,
                            "cap": pf_cap,
                            "capped": pf_capped,
                        },
                        "sortino": {
                            "raw": sortino_raw,
                            "cap": sortino_cap,
                            "capped": sortino_capped,
                        },
                    },
                }

            if not per_asset_metrics:
                poor = self.settings.get("poor_score", -999.0)
                trade_floor = self.settings.get("min_total_trades", 0)
                floor_ratio = total_trades / max(1, trade_floor)
                mode = self.settings.get("mode")
                policy_map = self.settings.get("trade_floor_policy_by_phase", {})
                floor_policy = policy_map.get(
                    mode, self.settings.get("trade_floor_policy", "hard_floor")
                )
                trade_penalty = None
                F = poor
                if mode == "walk_forward" and total_trades < trade_floor:
                    trade_penalty = "hard_floor"

                coverage_penalty = None
                if (
                    self.group_data
                    and self.settings.get("zero_trade_policy") == "ignore"
                    and self.settings.get("coverage_penalty_kappa") is not None
                ):
                    kappa = self.settings.get("coverage_penalty_kappa")
                    coverage_penalty = kappa * 1.0
                    F -= coverage_penalty

                self.last_details = {
                    "per_asset": per_asset_details,
                    "mu": None,
                    "sigma": None,
                    "lambda_sigma": None,
                    "total_trades": total_trades,
                    "assets_included": 0,
                    "assets_ignored": len(self.group_data),
                    "excluded_assets": self.coverage_exclusions + zero_trade_exclusions,
                    "penalties": {"trade_floor": trade_penalty, "floor_ratio": floor_ratio, "coverage": coverage_penalty},
                    "fitness": F,
                    "effective_floor": trade_floor,
                    "floor_ratio": floor_ratio,
                    "floor_policy": floor_policy,
                    "pf_cap": self.settings.get("pf_cap"),
                    "sortino_cap": self.settings.get("sortino_cap"),
                    "squash": self.settings.get("squash"),
                    "squash_params": self.settings.get("squash_params"),
                    "coverage_threshold": getattr(config, "COVERAGE_THRESHOLD", None),
                }
                self._current_gen_scores.append((F, total_trades))
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

            # Coverage penalty is computed regardless of trade floor policy
            coverage_penalty = 0.0
            if (
                self.settings.get("zero_trade_policy") == "ignore"
                and self.settings.get("coverage_penalty_kappa") is not None
            ):
                kappa = self.settings.get("coverage_penalty_kappa")
                coverage = len(included_assets) / max(1, len(self.group_data))
                coverage_penalty = kappa * (1 - coverage)
                F -= coverage_penalty

            trade_floor = self.settings.get("min_total_trades", 0)
            floor_ratio = total_trades / max(1, trade_floor)
            poor_score = self.settings.get("poor_score", -999.0)
            mode = self.settings.get("mode")
            policy_map = self.settings.get("trade_floor_policy_by_phase", {})
            floor_policy = policy_map.get(
                mode, self.settings.get("trade_floor_policy", "hard_floor")
            )
            strength = self.settings.get("soft_penalty_strength", 1.0)
            trade_penalty = None
            if mode == "walk_forward":
                if total_trades < trade_floor:
                    F = poor_score
                    trade_penalty = "hard_floor"
            elif mode in ("tuning", "ga") and total_trades < trade_floor:
                F *= floor_ratio ** strength
                trade_penalty = "soft_penalty"
            elif total_trades < trade_floor:
                if floor_policy == "soft_penalty":
                    F *= floor_ratio ** strength
                    trade_penalty = "soft_penalty"
                elif floor_policy == "hard_floor":
                    F = poor_score
                    trade_penalty = "hard_floor"

            # store diagnostics for optional inspection
            self.last_details = {
                "per_asset": per_asset_details,
                "mu": mu,
                "sigma": sigma,
                "lambda": lam,
                "lambda_sigma": lam * sigma,
                "total_trades": total_trades,
                "assets_included": len(included_assets),
                "assets_ignored": len(self.group_data) - len(included_assets),
                "excluded_assets": self.coverage_exclusions + zero_trade_exclusions,
                "penalties": {
                    "trade_floor": trade_penalty,
                    "floor_ratio": floor_ratio,
                    "coverage": coverage_penalty,
                },
                "fitness": F,
                "effective_floor": trade_floor,
                "floor_ratio": floor_ratio,
                "floor_policy": floor_policy,
                "pf_cap": self.settings.get("pf_cap"),
                "sortino_cap": self.settings.get("sortino_cap"),
                "squash": self.settings.get("squash"),
                "squash_params": self.settings.get("squash_params"),
                "coverage_threshold": getattr(config, "COVERAGE_THRESHOLD", None),
            }
            self._current_gen_scores.append((F, total_trades))
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
                "excluded_assets": self.coverage_exclusions,
                "penalties": {"trade_floor": "error", "coverage": None},
                "fitness": poor,
                "effective_floor": self.settings.get("min_total_trades", 0),
                "floor_ratio": 0.0,
                "floor_policy": self.settings.get("trade_floor_policy_by_phase", {}).get(
                    self.settings.get("mode"),
                    self.settings.get("trade_floor_policy", "hard_floor"),
                ),
                "pf_cap": self.settings.get("pf_cap"),
                "sortino_cap": self.settings.get("sortino_cap"),
                "squash": self.settings.get("squash"),
                "squash_params": self.settings.get("squash_params"),
                "coverage_threshold": getattr(config, "COVERAGE_THRESHOLD", None),
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
