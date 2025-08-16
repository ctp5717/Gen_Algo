"""Multi-asset fitness evaluation with capacity-constrained scanner simulation."""
from __future__ import annotations

from typing import Dict, List
import warnings
import multiprocessing as mp

import numpy as np
import pandas as pd
import vectorbt as vbt

import config
import strategy_engine as engine
from fitness import _inject_genes_into_rules
import scanner_sim
from scoring import SCORE_FUNCTIONS, apply_score_scaling
from utils.warnings_util import suppress_third_party_warnings
from utils.dataframe_util import to_frame, assert_monotonic_datetime_index
from utils.logging_util import get_logger, OncePerGenerationErrors

try:  # Optional dependency for JIT acceleration
    import numba as nb
except Exception:  # pragma: no cover - numba not installed
    nb = None

EPSILON = 1e-09


if nb is not None:
    @nb.njit(cache=True, parallel=True)
    def _calc_stats_numba(
        returns: np.ndarray,
    ) -> tuple[float, float, float, bool, bool]:
        n = returns.shape[0]
        equity = np.empty(n)
        equity[0] = 1.0 + returns[0]
        for i in range(1, n):
            equity[i] = equity[i - 1] * (1.0 + returns[i])

        running_max = np.empty(n)
        running_max[0] = equity[0]
        for i in range(1, n):
            running_max[i] = equity[i] if equity[i] > running_max[i - 1] else running_max[i - 1]

        drawdown = equity / running_max - 1.0
        max_dd = -np.min(drawdown) * 100.0

        pos = 0.0
        neg = 0.0
        for v in returns:
            if v > 0:
                pos += v
            elif v < 0:
                neg += v
        pf_denom_zero = neg == 0.0
        denom_pf = abs(neg) if not pf_denom_zero else EPSILON
        profit_factor = pos / denom_pf

        downside_sum = 0.0
        count = 0
        for v in returns:
            if v < 0:
                downside_sum += v * v
                count += 1
        downside_std = np.sqrt(downside_sum / count) if count else 0.0

        mean = np.mean(returns)
        sortino_denom_zero = downside_std == 0.0
        denom_sortino = downside_std if not sortino_denom_zero else EPSILON
        sortino = mean / denom_sortino
        return sortino, profit_factor, max_dd, pf_denom_zero, sortino_denom_zero


get_logger(__name__)
error_tracker = OncePerGenerationErrors()
VERY_LOW_FITNESS = -999.0


class MultiAssetFitnessEvaluator:
    """Evaluate a single strategy across multiple assets.

    The evaluator builds entry signals for each asset using the provided genes,
    gates them through :func:`scanner_sim.gate_entries` to respect the
    configured maximum number of concurrent trades, and finally computes a
    portfolio-level fitness score using the same metric blend as the
    single-asset evaluator.

    Notes
    -----
    Signals are generated using data available at time ``t`` and actual
    positions are entered on bar ``t+1``.  All return calculations therefore
    start from the next bar to avoid look-ahead bias.
    """

    def __init__(self, ohlc_dict: Dict[str, pd.DataFrame], base_rules: dict, gene_map: dict):
        tz = None
        for name, df in ohlc_dict.items():
            tz_cur = assert_monotonic_datetime_index(df, name)
            if tz is None:
                tz = tz_cur
            elif tz_cur != tz:
                raise ValueError("All DataFrames must share the same timezone")
        self.ohlc_dict = ohlc_dict
        self.base_rules = base_rules
        self.gene_map = gene_map
        self.assets: List[str] = list(ohlc_dict.keys())
        self.last_assets: List[str] = []  # exposed for testing
        # Diagnostics from the most recent full evaluation
        self.last_open_count: pd.Series | None = None
        self.last_trade_counts: pd.Series | None = None
        self.last_diagnostics: Dict[str, float] | None = None
        self.error_tracker = error_tracker

    def _build_signals(
        self, solution, assets: List[str]
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame | None,
        float | None,
        float | None,
        float | None,
    ]:
        entries: Dict[str, pd.Series] = {}
        exits: Dict[str, pd.Series] = {}
        scores: Dict[str, pd.Series] = {}
        close_dict: Dict[str, pd.Series] = {}

        # Extract exit rule parameters once since the strategy is shared across assets
        rules = _inject_genes_into_rules(self.base_rules, self.gene_map, solution)
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

        score_func_name = config.SCANNER.get("score_func", "pct_change")
        score_func = SCORE_FUNCTIONS.get(score_func_name, SCORE_FUNCTIONS["pct_change"])

        for name in assets:
            data = self.ohlc_dict[name]
            close_dict[name] = data["Close"]
            asset_entries = engine.process_strategy_rules(data, rules)

            # Shift signals forward so that trades occur on the next bar.
            # ``asset_entries`` remains unshifted for gating (decisions at ``t``),
            # while ``shifted_entries`` is used for exit simulation.
            shifted_entries = asset_entries.shift(
                getattr(config, "ENTRY_LAG_BARS", 1), fill_value=False
            )

            # Initial time-based exit for max-hold measured from execution time
            time_exit = shifted_entries.shift(config.MAX_HOLD_PERIOD, fill_value=False)

            # Use vectorbt to simulate exits from all exit rules
            pf = vbt.Portfolio.from_signals(
                close=data["Close"],
                entries=shifted_entries,
                exits=time_exit,
                sl_stop=sl_stop,
                tp_stop=tp_stop,
                sl_trail=sl_trail,
                fees=config.FEES,
                slippage=getattr(config, "SLIPPAGE", 0.0),
                freq=config.TIMEFRAME,
            )
            sell_orders = pf.orders.records_readable
            sell_orders = sell_orders[sell_orders["Side"] == "Sell"]
            asset_exits = pd.Series(False, index=data.index)
            if not sell_orders.empty:
                asset_exits.loc[sell_orders["Timestamp"]] = True

            entries[name] = asset_entries
            exits[name] = asset_exits.reindex(asset_entries.index, fill_value=False)
            if config.SCANNER.get("tie_break_policy") == "score":
                score_series = score_func(data).reindex(asset_entries.index).fillna(0.0)
                scale_method = config.SCANNER.get("score_scaling")
                if scale_method:
                    score_series = apply_score_scaling(score_series, data, scale_method)
                scores[name] = score_series.fillna(0.0)

        close_df = pd.DataFrame(close_dict).sort_index()
        common_index = close_df.index
        entries_df = (
            to_frame(entries, "entries", common_index)
            .fillna(False)
            .astype(bool)
            .reindex(columns=assets)
        )
        exits_df = (
            to_frame(exits, "exits", common_index)
            .fillna(False)
            .astype(bool)
            .reindex(columns=assets)
        )
        scores_df = (
            to_frame(scores, "scores", common_index).fillna(0.0).reindex(columns=assets)
            if scores
            else None
        )
        close_df = close_df.reindex(columns=assets)
        self.last_assets = list(assets)
        return close_df, entries_df, exits_df, scores_df, sl_stop, tp_stop, sl_trail

    def _evaluate_once(
        self, solution, seed: int | None, assets: List[str]
    ) -> tuple[
        float,
        Dict[str, float],
        pd.Series,
        pd.Series,
        Dict[str, float],
        pd.Series,
    ]:
        suppress_third_party_warnings()
        (
            close_df,
            entries_df,
            exits_df,
            scores_df,
            sl_stop,
            tp_stop,
            sl_trail,
        ) = self._build_signals(solution, assets)
        gated, open_count, diag = scanner_sim.gate_entries(
            entries_df,
            exits_df,
            config.SCANNER.get("max_concurrent_trades", 1),
            config.SCANNER.get("tie_break_policy", "fifo"),
            seed=seed,
            scores=scores_df,
            price_index=close_df.index,
            collect_collision_histogram=True,
        )

        returns_df = pd.DataFrame(0.0, index=gated.index, columns=gated.columns)
        per_asset_sortino: Dict[str, float] = {}
        trade_counts_dict: Dict[str, float] = {}
        for name in assets:
            data = self.ohlc_dict[name]
            asset_entries = gated[name].reindex(data.index, fill_value=False)
            if asset_entries.any():
                # Execute trades on the next bar
                shifted_entries = asset_entries.shift(
                    getattr(config, "ENTRY_LAG_BARS", 1), fill_value=False
                )
                time_exit = shifted_entries.shift(
                    config.MAX_HOLD_PERIOD, fill_value=False
                )
                pf = vbt.Portfolio.from_signals(
                    close=data["Close"],
                    entries=shifted_entries,
                    exits=time_exit,
                    sl_stop=sl_stop,
                    tp_stop=tp_stop,
                    sl_trail=sl_trail,
                    fees=config.FEES,
                    slippage=getattr(config, "SLIPPAGE", 0.0),
                    freq=config.TIMEFRAME,
                )
                returns_df[name] = pf.returns()
                per_asset_sortino[name] = pf.stats()["Sortino Ratio"]
                trade_counts_dict[name] = pf.trades.count()
            else:
                returns_df[name] = 0.0
                per_asset_sortino[name] = 0.0
                trade_counts_dict[name] = 0.0

        # Returns are realized one bar after the position count used for gating
        open_count_safe = (
            open_count.shift(getattr(config, "ENTRY_LAG_BARS", 1))
            .reindex(returns_df.index)
            .replace(0, np.nan)
        )
        portfolio_returns = (returns_df.sum(axis=1) / open_count_safe).fillna(0.0)
        trade_counts = pd.Series(trade_counts_dict)
        total_trades = trade_counts.sum()
        concentration_ratio = float(trade_counts.max() / total_trades) if total_trades > 0 else 0.0

        if len(assets) == 1:
            name = assets[0]
            data_sa = self.ohlc_dict[name]
            sa_entries = entries_df[name]
            shifted_sa = sa_entries.shift(
                getattr(config, "ENTRY_LAG_BARS", 1), fill_value=False
            )
            time_exit_sa = shifted_sa.shift(config.MAX_HOLD_PERIOD, fill_value=False)
            pf_sa = vbt.Portfolio.from_signals(
                close=data_sa["Close"],
                entries=shifted_sa,
                exits=time_exit_sa,
                sl_stop=sl_stop,
                tp_stop=tp_stop,
                sl_trail=sl_trail,
                fees=config.FEES,
                slippage=getattr(config, "SLIPPAGE", 0.0),
                freq=config.TIMEFRAME,
            )
            sa_returns = pf_sa.returns()
            sa_trade_count = pf_sa.trades.count()
            np.testing.assert_allclose(
                portfolio_returns.values,
                sa_returns.values,
                rtol=1e-6,
                atol=1e-8,
            )
            assert int(trade_counts[name]) == int(sa_trade_count)

        sortino, profit_factor, max_dd = self._calc_stats(portfolio_returns)
        sortino, profit_factor, max_dd = self._clamp_metrics(
            sortino, profit_factor, max_dd
        )

        drawdown_score = 1 - (max_dd / 100.0)
        weights = config.FITNESS_WEIGHTS
        fitness_score = (
            sortino * weights["sortino_ratio"]
            + profit_factor * weights["profit_factor"]
            + drawdown_score * weights["max_drawdown"]
        )
        if np.isnan(fitness_score):
            fitness_score = -1.0
        return (
            float(fitness_score),
            per_asset_sortino,
            portfolio_returns,
            open_count,
            diag,
            trade_counts,
            concentration_ratio,
        )

    @staticmethod
    def _calc_stats(returns: pd.Series) -> tuple[float, float, float]:
        """Compute Sortino, Profit Factor and Max Drawdown from returns.

        Falls back to the standard pandas implementation unless the config
        requests a Numba backend and the library is available.
        """
        logger = get_logger(__name__)
        if config.PARALLEL.get("backend") == "numba" and nb is not None:
            sortino, profit_factor, max_dd, pf_zero, sort_zero = _calc_stats_numba(
                returns.to_numpy()
            )
            if pf_zero:
                logger.debug(
                    "Profit factor denominator was zero; using EPSILON fallback"
                )
            if sort_zero:
                logger.debug(
                    "Sortino denominator was zero; using EPSILON fallback"
                )
            return float(sortino), float(profit_factor), float(max_dd)

        equity = (1 + returns).cumprod()
        running_max = equity.cummax()
        drawdown = (equity / running_max) - 1.0
        max_dd = -drawdown.min() * 100
        pos = returns[returns > 0].sum()
        neg = returns[returns < 0].sum()
        denom_pf = abs(neg)
        if denom_pf == 0.0:
            logger.debug(
                "Profit factor denominator was zero; using EPSILON fallback"
            )
        denom_pf = denom_pf if denom_pf != 0.0 else EPSILON
        profit_factor = pos / denom_pf
        downside = returns[returns < 0]
        downside_std = downside.std(ddof=0)
        if downside_std == 0.0:
            logger.debug(
                "Sortino denominator was zero; using EPSILON fallback"
            )
        downside_std = downside_std if downside_std != 0.0 else EPSILON
        sortino = returns.mean() / downside_std
        return sortino, profit_factor, max_dd

    @staticmethod
    def _clamp_metrics(
        sortino: float, profit_factor: float, max_dd: float
    ) -> tuple[float, float, float]:
        sortino = (
            float(np.clip(sortino, -5.0, 5.0)) if np.isfinite(sortino) else 0.0
        )
        profit_factor = (
            float(np.clip(profit_factor, 0.0, 5.0))
            if np.isfinite(profit_factor)
            else 0.0
        )
        max_dd = (
            float(np.clip(max_dd, 0.0, 100.0)) if np.isfinite(max_dd) else 100.0
        )
        return sortino, profit_factor, max_dd

    def __call__(self, ga_instance, solution, sol_idx):
        logger = get_logger(__name__)
        generation = getattr(ga_instance, "generations_completed", 0)
        try:
            # Determine asset subset for mini-batching
            assets = self.assets
            is_elite_eval = False
            if config.MINIBATCH.get("enabled"):
                size = config.MINIBATCH.get("size", len(self.assets)) or len(self.assets)
                size = min(size, len(self.assets))
                if config.MINIBATCH.get("elite_eval_period", 0) > 0 and (
                    generation % config.MINIBATCH["elite_eval_period"] == 0
                    and sol_idx < config.MINIBATCH.get("elite_count", 0)
                ):
                    assets = self.assets
                    is_elite_eval = True
                else:
                    rng = np.random.default_rng(
                        config.SCANNER.get("seed", 0) + generation + sol_idx
                    )
                    assets = list(rng.choice(self.assets, size=size, replace=False))

            if config.SCANNER.get("verbose"):
                logger.info(
                    "Generation %d, solution %d, assets: %s",
                    generation,
                    sol_idx,
                    ",".join(map(str, assets)),
                )

            runs = config.SCANNER.get("monte_carlo_runs", 1)
            # Ensure a minimum number of runs for stochastic tie-breaks
            if (
                config.SCANNER.get("tie_break_policy") == "random"
                and runs < 3
            ):
                warnings.warn(
                    "monte_carlo_runs increased to 3 for random tie-break policy",
                    RuntimeWarning,
                )
                runs = 3
            base_seed = config.SCANNER.get("seed", 0)
            max_solutions = getattr(ga_instance, "sol_per_pop", 0)
            seed_base = base_seed + generation * max_solutions + sol_idx
            run_scores: List[float] = []
            concentration_ratios: List[float] = []
            per_asset_runs: Dict[str, List[float]] = {}
            diag_saved = False

            if runs > 1 and config.PARALLEL.get("backend") == "multiprocessing":
                args = []
                for i in range(runs):
                    seed = seed_base + i
                    logger.debug(
                        "MC run %d using seed %d (base=%d gen=%d sol_idx=%d max_solutions=%d)",
                        i,
                        seed,
                        base_seed,
                        generation,
                        sol_idx,
                        max_solutions,
                    )
                    args.append((solution, seed, assets))
                with mp.Pool(processes=config.PARALLEL.get("workers") or None) as pool:
                    results = pool.starmap(self._evaluate_once, args)
                for score, metrics, _pr, oc, diag, trade_counts, conc_ratio in results:
                    run_scores.append(score)
                    concentration_ratios.append(conc_ratio)
                    for a, m in metrics.items():
                        per_asset_runs.setdefault(a, []).append(m)
                    if not diag_saved and assets == self.assets:
                        self.last_open_count = oc
                        self.last_trade_counts = trade_counts
                        self.last_diagnostics = diag
                        diag_saved = True
            else:
                for i in range(runs):
                    seed = seed_base + i
                    logger.debug(
                        "MC run %d using seed %d (base=%d gen=%d sol_idx=%d max_solutions=%d)",
                        i,
                        seed,
                        base_seed,
                        generation,
                        sol_idx,
                        max_solutions,
                    )
                    (
                        score,
                        metrics,
                        _pr,
                        oc,
                        diag,
                        trade_counts,
                        conc_ratio,
                    ) = self._evaluate_once(
                        solution, seed=seed, assets=assets
                    )
                    run_scores.append(score)
                    concentration_ratios.append(conc_ratio)
                    for a, m in metrics.items():
                        per_asset_runs.setdefault(a, []).append(m)
                    if not diag_saved and assets == self.assets:
                        self.last_open_count = oc
                        self.last_trade_counts = trade_counts
                        self.last_diagnostics = diag
                        diag_saved = True

            aggregated = float(np.median(run_scores))
            dispersion = float(np.std(run_scores))
            penalty_asset = 0.0
            penalty_mc = 0.0
            penalty_conc = 0.0
            if per_asset_runs and config.ROBUSTNESS.get("lambda_asset_dispersion", 0.0) > 0:
                per_asset_avg = {a: np.mean(v) for a, v in per_asset_runs.items()}
                penalty_asset = config.ROBUSTNESS["lambda_asset_dispersion"] * np.std(
                    list(per_asset_avg.values())
                )
            if config.ROBUSTNESS.get("lambda_mc_dispersion", 0.0) > 0:
                penalty_mc = config.ROBUSTNESS["lambda_mc_dispersion"] * dispersion
            if config.ROBUSTNESS.get("lambda_concentration", 0.0) > 0 and concentration_ratios:
                concentration_ratio = float(np.median(concentration_ratios))
                penalty_conc = (
                    config.ROBUSTNESS["lambda_concentration"] * concentration_ratio
                )
            else:
                if concentration_ratios:
                    concentration_ratio = float(np.median(concentration_ratios))
                else:
                    concentration_ratio = 0.0

            logger.debug(
                "run_scores=%s median=%.4f dispersion=%.4f asset_dispersion=%.4f "
                "mc_dispersion=%.4f concentration_ratio=%.4f concentration_penalty=%.4f",
                run_scores,
                aggregated,
                dispersion,
                penalty_asset,
                penalty_mc,
                concentration_ratio,
                penalty_conc,
            )

            result = float(aggregated - penalty_asset - penalty_mc - penalty_conc)

            if is_elite_eval:
                logger.info(
                    "Generation %d, elite solution %d rescored on full asset set with fitness %.4f",
                    generation,
                    sol_idx,
                    result,
                )

            self.last_diagnostics = self.last_diagnostics or {}
            self.last_diagnostics.update(
                {
                    "mc_runs": runs,
                    "run_scores": run_scores,
                    "mc_median": aggregated,
                    "dispersion": dispersion,
                    "asset_dispersion": penalty_asset,
                    "mc_dispersion": penalty_mc,
                    "concentration_ratio": concentration_ratio,
                    "concentration_penalty": penalty_conc,
                }
            )

            if config.SCANNER.get("verbose") and self.last_diagnostics:
                diag = self.last_diagnostics
                print(
                    f"Candidates: {diag.get('total_candidates', 0)} | "
                    f"Accepted: {diag.get('accepted', 0)} | "
                    f"Collisions: {diag.get('collisions', 0)} | "
                    f"Rejected: {diag.get('rejected', 0)} | "
                    f"Acceptance Rate: {diag.get('acceptance_rate', 0.0):.2f} | "
                    f"Avg Open: {diag.get('avg_n_open', 0.0):.2f} | "
                    f"Max Open: {diag.get('max_n_open', 0)} | "
                    f"MC Dispersion: {diag.get('mc_dispersion', 0.0):.4f}"
                )
                per_asset = diag.get("per_asset", {})
                top_n = config.SCANNER.get("verbose_top_n", 5)
                if per_asset:
                    top_assets = sorted(
                        per_asset.items(),
                        key=lambda x: x[1].get("candidates", 0),
                        reverse=True,
                    )[:top_n]
                    top_str = " | ".join(
                        f"{asset}: {stats.get('accepted', 0)}/{stats.get('candidates', 0)}"
                        for asset, stats in top_assets
                    )
                    print(f"Top assets (accepted/candidates): {top_str}")
            return result
        except Exception as e:
            error_tracker.log_exception(logger, "Fitness evaluation failed", e)
            return VERY_LOW_FITNESS
