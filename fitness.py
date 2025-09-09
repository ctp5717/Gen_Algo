# fitness.py

"""
Fitness Function for Genetic Algorithm
(This version uses the correct pandas .shift() method for time-based exits)
"""
import copy
import logging
import traceback
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import vectorbt as vbt

import config
import strategy_engine as engine
import trade_floor

logger = logging.getLogger(__name__)
MACD_REPAIR_COUNT = 0


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

    Examples
    --------
    >>> vals = [1.6, 1.0, 0.4]
    >>> weights = [1/3, 1/3, 1/3]
    >>> mu, sigma = weighted_mean_std(vals, weights)
    >>> round(mu, 1), round(sigma, 4)
    (1.0, 0.4899)
    """

    w = np.asarray(weights, dtype=float)
    x = np.asarray(values, dtype=float)
    if w.ndim == 0:
        w = np.array([float(w)])
    if x.ndim == 0:
        x = np.array([float(x)])
    if len(w) != len(x) or len(w) == 0:
        raise ValueError("weighted_mean_std: values/weights length mismatch")
    if (w < 0).any():
        raise ValueError("weighted_mean_std: weights must be non-negative")
    total = w.sum()
    if total == 0:
        w = np.ones_like(w) / len(w)
    else:
        w = w / total
    mu = float(np.sum(w * x))
    variance = float(np.sum(w * (x - mu) ** 2))
    sigma_pop = float(np.sqrt(variance))  # population stdev (ddof=0)
    return mu, sigma_pop


def print_floor_failures(counter: Counter):
    """Utility to print a consistent hard-floor failure summary."""
    if not counter or sum(counter.values()) == 0:
        print("Hard-floor failures: none")
    else:
        print(f"Hard-floor failures: {dict(counter)}")


def _normalize_macd_params(params: dict) -> dict:
    """Repair MACD params so they satisfy fast < slow and 1 <= signal < slow."""

    fast, slow, signal = (
        params.get("fast"),
        params.get("slow"),
        params.get("signal"),
    )
    if fast is None or slow is None or signal is None:
        raise ValueError("MACD params must be non-null: fast, slow, signal")
    original = (fast, slow, signal)
    if slow <= fast:
        slow = fast + 1
    if signal < 1:
        signal = 1
    if signal >= slow:
        signal = slow - 1
    fast, slow, signal = int(fast), int(slow), int(signal)
    params.update({"fast": fast, "slow": slow, "signal": signal})
    repaired = (fast, slow, signal)
    if repaired != original:
        logger.debug("Repaired MACD params %s -> %s", original, repaired)
        global MACD_REPAIR_COUNT
        MACD_REPAIR_COUNT += 1
    return params


def _inject_genes_into_rules(base_rules: dict, gene_map: dict, solution: list) -> dict:
    """Inject gene values into a copy of strategy rules, resolving defaults."""

    def _resolve_defaults(obj):
        if isinstance(obj, dict):
            if "gene" in obj:
                if "options" in obj:
                    return obj.get("options", [None])[0]
                return obj.get("low", obj.get("high"))
            return {k: _resolve_defaults(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_resolve_defaults(v) for v in obj]
        return obj

    injected_rules = _resolve_defaults(copy.deepcopy(base_rules))
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

    def _apply_macd_repair(obj):
        if isinstance(obj, dict):
            if obj.get("indicator") == "macd":
                params = obj.get("params", {})
                if {"fast", "slow", "signal"} <= params.keys():
                    _normalize_macd_params(params)
            for val in obj.values():
                _apply_macd_repair(val)
        elif isinstance(obj, list):
            for item in obj:
                _apply_macd_repair(item)

    _apply_macd_repair(injected_rules)
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

            if entries.sum() < config.FITNESS_WEIGHTS["min_trades"]:
                return -1.0

            # --- NEW: Logic to handle multiple, selectable exit types ---
            exit_rules = rules.get("exit_rules", {})
            sl_rule = exit_rules.get("stop_loss", {})
            tsl_rule = exit_rules.get("trailing_stop", {})
            tp_rule = exit_rules.get("take_profit", {})

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

            time_based_exit = entries.shift(config.MAX_HOLD_PERIOD, fill_value=False)
            time_based_exit = time_based_exit.reindex(entries.index, fill_value=False)

            portfolio = vbt.Portfolio.from_signals(
                close=self.ohlc_data["Close"],
                entries=entries,
                exits=time_based_exit,
                sl_stop=sl_stop,
                tp_stop=tp_stop,
                sl_trail=sl_trail,  # Pass the trailing stop value to the backtester
                fees=config.FEES,
                freq=config.to_pandas_freq(config.TIMEFRAME),
            )

            stats = portfolio.stats()
            sortino = stats.get("Sortino Ratio")
            profit_factor = stats.get("Profit Factor")
            max_drawdown = stats.get("Max Drawdown [%]")

            cap = getattr(config, "MULTI_ASSET", {}).get("winsorize_pf_cap", 5.0)
            if np.isinf(profit_factor) or profit_factor > cap:
                profit_factor = cap
            if np.isnan(sortino):
                sortino = 0
            if np.isnan(profit_factor):
                profit_factor = 0
            if np.isnan(max_drawdown):
                max_drawdown = 100.0

            drawdown_score = 1 - (max_drawdown / 100.0)
            weights = config.FITNESS_WEIGHTS

            fitness_score = (
                (sortino * weights["sortino_ratio"])
                + (profit_factor * weights["profit_factor"])
                + (drawdown_score * weights["max_drawdown"])
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

    def __init__(
        self,
        group_data: dict,
        base_rules: dict,
        gene_map: dict,
        settings: dict | None = None,
    ):
        self.group_data = group_data  # dict[ticker -> OHLCV DataFrame]
        self.base_rules = base_rules
        self.gene_map = gene_map
        defaults = getattr(config, "MULTI_ASSET", {})
        self.settings = copy.deepcopy(defaults)
        if settings:
            self.settings.update(settings)
        # Clamp min_included_assets to available data after alignment
        mia = self.settings.get("min_included_assets", 1)
        self.settings["min_included_assets"] = min(mia, len(group_data))
        self.last_details = {}
        self.floor_failures = Counter()

        # Validate key configuration values to catch misconfiguration early.
        assert (
            self.settings.get("lambda_dispersion", 0.0) >= 0
        ), "lambda_dispersion must be >= 0"
        assert (
            self.settings.get("winsorize_pf_cap", 1.0) >= 1
        ), "winsorize_pf_cap must be >= 1"
        assert (
            self.settings.get("soft_penalty_strength", 0.0) >= 0
        ), "soft_penalty_strength must be >= 0"
        assert (
            self.settings.get("min_total_trades", 0) >= 0
        ), "min_total_trades must be >= 0"

        # Warn if the configured floors are unreachable
        min_group = self.settings.get("min_total_trades", 0)
        need = self.settings.get("min_included_assets", 0) * self.settings.get(
            "per_asset_min_trades", 0
        )
        if min_group and need > min_group:
            if self.settings.get("trade_floor_policy") == "soft_penalty":
                self.settings["min_total_trades"] = need
            else:
                warnings.warn(
                    "min_total_trades < min_included_assets * per_asset_min_trades; run may be infeasible.",
                    stacklevel=2,
                )

    # ------------------------------------------------------------------
    def _evaluate_single_asset(self, ohlc: pd.DataFrame, rules: dict) -> dict:
        """Run the strategy on a single asset and return raw statistics."""
        # Empty or very short dataframes can cause downstream libraries to
        # raise ``IndexError`` when statistics are requested.  In walk forward
        # validation some assets may have no data for a given window.  Handle
        # this case early and return a stub result that indicates zero trades
        # so that the caller can decide whether to ignore or penalise it.
        if ohlc is None or ohlc.empty:
            return {
                "sortino": None,
                "profit_factor": None,
                "max_drawdown": None,
                "trades": 0,
                "total_return": None,
                "equity_curve": pd.Series(dtype=float),
                "signal_counts": {},
            }

        entries, signal_counts = engine.process_strategy_rules(
            ohlc, rules, collect_counts=True
        )

        # Record the actual executed trades using vectorbt.
        exit_rules = rules.get("exit_rules", {})
        sl_rule = exit_rules.get("stop_loss", {})
        tsl_rule = exit_rules.get("trailing_stop", {})
        tp_rule = exit_rules.get("take_profit", {})

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
            close=ohlc["Close"],
            entries=entries,
            exits=time_exit,
            sl_stop=sl_stop,
            tp_stop=tp_stop,
            sl_trail=sl_trail,
            fees=config.FEES,
            freq=config.to_pandas_freq(config.TIMEFRAME),
        )

        stats = portfolio.stats()
        trades = int(portfolio.trades.count())
        return {
            "sortino": stats.get("Sortino Ratio"),
            "profit_factor": stats.get("Profit Factor"),
            "max_drawdown": stats.get("Max Drawdown [%]"),
            "trades": trades,
            "total_return": stats.get("Total Return [%]"),
            "equity_curve": portfolio.value(),
            "signal_counts": signal_counts,
        }

    # ------------------------------------------------------------------
    def __call__(self, ga_instance, solution, sol_idx):
        try:
            rules = _inject_genes_into_rules(self.base_rules, self.gene_map, solution)

            per_asset_metrics = []
            included_assets = []
            per_asset_details = {}
            total_trades = 0
            assets_traded = 0
            asset_weights_cfg = self.settings.get("asset_weights") or {}
            verbose = bool(self.settings.get("verbose_asset_errors"))

            for ticker in sorted(self.group_data):
                ohlc = self.group_data[ticker]
                eval_reason = None
                reason_detail = None
                reason_trace = None
                if ohlc is None or ohlc.empty:
                    eval_reason = "insufficient_coverage"
                    stats = {
                        "sortino": None,
                        "profit_factor": None,
                        "max_drawdown": None,
                        "trades": 0,
                        "total_return": None,
                        "equity_curve": pd.Series(dtype=float),
                        "signal_counts": {},
                    }
                else:
                    try:
                        stats = self._evaluate_single_asset(ohlc, rules)
                    except Exception as e:
                        if verbose:
                            print(f"Error evaluating asset {ticker}: {e}")
                            tb = traceback.format_exception(
                                e.__class__, e, e.__traceback__
                            )
                            reason_trace = (tb[0].strip(), tb[-1].strip())
                        eval_reason = "evaluation_error"
                        reason_detail = repr(e)
                        stats = {
                            "sortino": None,
                            "profit_factor": None,
                            "max_drawdown": None,
                            "trades": 0,
                            "total_return": None,
                            "equity_curve": pd.Series(dtype=float),
                            "signal_counts": {},
                        }

                trades = stats.get("trades", 0)
                total_trades += trades

                weight = asset_weights_cfg.get(ticker, 1.0)
                pf_raw = stats.get("profit_factor")
                cap = self.settings.get("winsorize_pf_cap", 5.0)
                if pf_raw is None or np.isnan(pf_raw):
                    pf_capped = self.settings.get("nan_fallback", 0.0)
                else:
                    pf_capped = min(cap, pf_raw) if not np.isinf(pf_raw) else cap

                if trades < self.settings.get("per_asset_min_trades", 1):
                    if self.settings.get("zero_trade_policy") == "penalize":
                        val = self.settings.get("zero_trade_penalty", -1.0)
                        per_asset_metrics.append(val)
                        included_assets.append(ticker)
                        if trades > 0:
                            assets_traded += 1
                        details = {
                            **stats,
                            "score": val,
                            "included": True,
                            "asset_weight": weight,
                            "profit_factor_capped": pf_capped,
                        }
                        if reason_detail:
                            details["reason_detail"] = reason_detail
                        if reason_trace:
                            details["reason_trace"] = " | ".join(
                                str(x) for x in reason_trace
                            )
                        per_asset_details[ticker] = details
                    else:
                        reason = eval_reason or (
                            "ignored_zero_trades"
                            if trades == 0
                            else "below_per_asset_min_trades"
                        )
                        info = self.settings.get("per_asset_floor_info")
                        if info:
                            reason += (
                                "; Per-asset floor: base="
                                f"{info['base_floor']} → scaled={info['ceil']} "
                                f"(window={info['window_days']}d, base={info['trading_days_per_year']}d)"
                            )
                        details = {
                            **stats,
                            "score": None,
                            "included": False,
                            "asset_weight": weight,
                            "profit_factor_capped": pf_capped,
                            "reason": reason,
                        }
                        if reason_detail:
                            details["reason_detail"] = reason_detail
                        if reason_trace:
                            details["reason_trace"] = " | ".join(
                                str(x) for x in reason_trace
                            )
                        per_asset_details[ticker] = details
                        continue
                else:
                    metric_type = self.settings.get("metric", "composite")
                    if metric_type == "sortino":
                        val = stats.get(
                            "sortino", self.settings.get("nan_fallback", 0.0)
                        )
                    elif metric_type == "profit_factor":
                        val = pf_capped
                    elif metric_type == "return":
                        val = stats.get(
                            "total_return", self.settings.get("nan_fallback", 0.0)
                        )
                    else:  # composite metric
                        sortino = stats.get("sortino")
                        pf = pf_capped
                        dd = stats.get("max_drawdown")

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
                    if trades > 0:
                        assets_traded += 1
                    details = {
                        **stats,
                        "score": val,
                        "included": True,
                        "asset_weight": weight,
                        "profit_factor_capped": pf_capped,
                    }
                    if reason_detail:
                        details["reason_detail"] = reason_detail
                    if reason_trace:
                        details["reason_trace"] = " | ".join(
                            str(x) for x in reason_trace
                        )
                    per_asset_details[ticker] = details

            if not per_asset_metrics:
                poor_score = self.settings.get("poor_score", -999.0)
                reason = "no_assets"
                self.floor_failures[reason] += 1
                self.last_details = {
                    "per_asset": per_asset_details,
                    "mu": 0.0,
                    "sigma": 0.0,
                    "lambda_sigma": 0.0,
                    "total_trades": total_trades,
                    "assets_included": 0,
                    "assets_traded": 0,
                    "assets_ignored": len(self.group_data),
                    "penalties": {
                        "trade_floor": reason,
                        "coverage": 0.0,
                        "min_assets": reason,
                    },
                    "min_total_trades": self.settings.get("min_total_trades", 0),
                    "fitness": poor_score,
                    "asset_weights": {},
                }
                return poor_score

            # Determine weights for included assets and renormalise
            asset_weights = self.settings.get("asset_weights") or {}
            raw_weights = []
            neg_seen = False
            for t in included_assets:
                w = asset_weights.get(t, 1.0)
                if w < 0:
                    neg_seen = True
                    w = 0.0
                raw_weights.append(w)
            if neg_seen:
                print("Warning: negative asset weights clipped to zero")
            weight_sum = sum(raw_weights)
            if weight_sum == 0:
                if raw_weights:
                    print(
                        "Warning: all asset weights were zero; reverting to equal weights"
                    )
                weights = [1.0 / len(per_asset_metrics)] * len(per_asset_metrics)
            else:
                weights = [w / weight_sum for w in raw_weights]

            w_map = {}
            for t, w in zip(included_assets, weights):
                per_asset_details[t]["asset_weight"] = w
                w_map[t] = w

            m_arr = np.array(per_asset_metrics, dtype=float)
            w_arr = np.array(weights, dtype=float)
            mu, sigma = weighted_mean_std(m_arr, w_arr)

            lam = self.settings.get("lambda_dispersion", 0.0)
            F = mu - lam * sigma

            policy = self.settings.get("trade_floor_policy", "hard_floor")
            poor_score = self.settings.get("poor_score", -999.0)
            min_trades = self.settings.get("min_total_trades", 0)
            min_assets = self.settings.get("min_included_assets", 1)
            trade_penalty = None
            min_assets_penalty = None

            assets_count = len(included_assets)
            if assets_count < min_assets:
                if policy == "hard_floor":
                    F = poor_score
                    reason = "below_min_included_assets"
                    trade_penalty = reason
                    min_assets_penalty = reason
                    self.floor_failures[reason] += 1
                    self.last_details = {
                        "per_asset": per_asset_details,
                        "mu": mu,
                        "sigma": sigma,
                        "lambda_sigma": lam * sigma,
                        "total_trades": total_trades,
                        "assets_included": assets_count,
                        "assets_traded": assets_traded,
                        "assets_ignored": len(self.group_data) - assets_count,
                        "penalties": {
                            "trade_floor": trade_penalty,
                            "coverage": 0.0,
                            "min_assets": min_assets_penalty,
                        },
                        "min_total_trades": min_trades,
                        "fitness": F,
                        "asset_weights": w_map,
                    }
                    return F
                else:
                    strength = self.settings.get("soft_penalty_strength", 1.0)
                    scale = (assets_count / max(1, min_assets)) ** strength
                    F *= scale
                    min_assets_penalty = {"scale": scale}

            if policy == "hard_floor" and total_trades < min_trades:
                F = poor_score
                reason = "below_group_floor"
                trade_penalty = reason
                self.floor_failures[reason] += 1
                self.last_details = {
                    "per_asset": per_asset_details,
                    "mu": mu,
                    "sigma": sigma,
                    "lambda_sigma": lam * sigma,
                    "total_trades": total_trades,
                    "assets_included": assets_count,
                    "assets_traded": assets_traded,
                    "assets_ignored": len(self.group_data) - assets_count,
                    "penalties": {
                        "trade_floor": trade_penalty,
                        "coverage": 0.0,
                        "min_assets": min_assets_penalty,
                    },
                    "min_total_trades": min_trades,
                    "fitness": F,
                    "asset_weights": w_map,
                }
                return F
            elif policy == "soft_penalty" and total_trades < min_trades:
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

            coverage_penalty = 0.0
            if self.settings.get("zero_trade_policy") == "ignore":
                kappa = self.settings.get("coverage_penalty", 0.0)
                coverage = assets_count / max(1, len(self.group_data))
                coverage_penalty = kappa * (1 - coverage)
                F -= coverage_penalty

            # store diagnostics for optional inspection
            self.last_details = {
                "per_asset": per_asset_details,
                "mu": mu,
                "sigma": sigma,
                "lambda_sigma": lam * sigma,
                "total_trades": total_trades,
                "assets_included": assets_count,
                "assets_traded": assets_traded,
                "assets_ignored": len(self.group_data) - assets_count,
                "penalties": {
                    "trade_floor": trade_penalty,
                    "coverage": coverage_penalty,
                    "min_assets": min_assets_penalty,
                },
                "min_total_trades": min_trades,
                "fitness": F,
                "asset_weights": w_map,
            }

            return F

        except Exception as e:
            print(f"Error in multi-asset fitness evaluation: {e}")
            poor = self.settings.get("poor_score", -999.0)
            self.last_details = {
                "per_asset": {},
                "mu": 0.0,
                "sigma": 0.0,
                "lambda_sigma": 0.0,
                "total_trades": 0,
                "assets_included": 0,
                "assets_ignored": len(self.group_data),
                "penalties": {"trade_floor": None, "coverage": 0.0, "min_assets": None},
                "min_total_trades": self.settings.get("min_total_trades", 0),
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

    settings = copy.deepcopy(getattr(config, "MULTI_ASSET", {}))
    if settings.get("enabled"):
        start = pd.to_datetime(config.TRAINING_PERIOD["start"])
        end = pd.to_datetime(config.TRAINING_PERIOD["end"])

        per_asset_base = settings.get("per_asset_min_trades")
        if per_asset_base:
            floor_pa, info_pa = trade_floor.scale_floor(
                per_asset_base,
                start,
                end,
                settings.get("trading_days_per_year", 252),
            )
            settings["per_asset_min_trades"] = floor_pa
            settings["per_asset_floor_info"] = info_pa
            print(
                "Per-asset floor: base="
                f"{per_asset_base} → scaled={floor_pa} "
                f"(window={info_pa['window_days']}d, base={info_pa['trading_days_per_year']}d)"
            )

        rate = settings.get("min_total_trades_per_year")
        if rate:
            floor, info = trade_floor.scale_floor(
                rate, start, end, settings.get("trading_days_per_year", 252)
            )
            settings["min_total_trades"] = floor
            settings["group_floor_info"] = info
        return MultiAssetFitnessEvaluator(ohlc_data, base_rules, gene_map, settings)
    return FitnessEvaluator(ohlc_data, base_rules, gene_map)
