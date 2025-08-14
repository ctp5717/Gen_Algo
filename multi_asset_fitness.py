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

try:  # Optional dependency for JIT acceleration
    import numba as nb
except Exception:  # pragma: no cover - numba not installed
    nb = None


if nb is not None:
    @nb.njit(cache=True, parallel=True)
    def _calc_stats_numba(returns: np.ndarray) -> tuple[float, float, float]:
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
        profit_factor = pos / abs(neg) if neg != 0 else np.inf

        downside_sum = 0.0
        count = 0
        for v in returns:
            if v < 0:
                downside_sum += v * v
                count += 1
        downside_std = np.sqrt(downside_sum / count) if count else 0.0

        mean = np.mean(returns)
        sortino = mean / downside_std if downside_std != 0 else np.nan
        return sortino, profit_factor, max_dd


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
        self.ohlc_dict = ohlc_dict
        self.base_rules = base_rules
        self.gene_map = gene_map
        self.assets: List[str] = list(ohlc_dict.keys())
        self.last_assets: List[str] = []  # exposed for testing
        # Diagnostics from the most recent full evaluation
        self.last_open_count: pd.Series | None = None
        self.last_trade_counts: pd.Series | None = None
        self.last_diagnostics: Dict[str, float] | None = None

    def _build_signals(
        self, solution, assets: List[str]
    ) -> tuple[
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
            asset_entries = engine.process_strategy_rules(data, rules)

            # Shift signals forward so that trades occur on the next bar.
            # ``asset_entries`` remains unshifted for gating (decisions at ``t``),
            # while ``shifted_entries`` is used for exit simulation.
            shifted_entries = asset_entries.shift(1, fill_value=False)

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

        self.last_assets = list(assets)
        entries_df = pd.concat(entries, axis=1)
        exits_df = pd.concat(exits, axis=1)
        scores_df = pd.concat(scores, axis=1) if scores else None
        return entries_df, exits_df, scores_df, sl_stop, tp_stop, sl_trail

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
        entries_df, exits_df, scores_df, sl_stop, tp_stop, sl_trail = self._build_signals(
            solution, assets
        )
        gated, open_count, diag = scanner_sim.gate_entries(
            entries_df,
            exits_df,
            config.SCANNER.get("max_concurrent_trades", 1),
            config.SCANNER.get("tie_break_policy", "fifo"),
            seed=seed,
            scores=scores_df,
        )

        returns_df = pd.DataFrame(0.0, index=gated.index, columns=gated.columns)
        per_asset_sortino: Dict[str, float] = {}
        for name in assets:
            data = self.ohlc_dict[name]
            asset_entries = gated[name].reindex(data.index, fill_value=False)
            if asset_entries.any():
                # Execute trades on the next bar
                shifted_entries = asset_entries.shift(1, fill_value=False)
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
            else:
                returns_df[name] = 0.0
                per_asset_sortino[name] = 0.0

        # Returns are realized one bar after the position count used for gating
        open_count_safe = open_count.shift(1).reindex(returns_df.index).replace(0, np.nan)
        portfolio_returns = (returns_df.sum(axis=1) / open_count_safe).fillna(0.0)
        trade_counts = gated.sum()
        sortino, profit_factor, max_dd = self._calc_stats(portfolio_returns)

        if np.isinf(profit_factor) or profit_factor > 5:
            profit_factor = 5
        if np.isnan(sortino):
            sortino = 0
        if np.isnan(profit_factor):
            profit_factor = 0
        if np.isnan(max_dd):
            max_dd = 100.0

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
        )

    @staticmethod
    def _calc_stats(returns: pd.Series) -> tuple[float, float, float]:
        """Compute Sortino, Profit Factor and Max Drawdown from returns.

        Falls back to the standard pandas implementation unless the config
        requests a Numba backend and the library is available.
        """
        if config.PARALLEL.get("backend") == "numba" and nb is not None:
            sortino, profit_factor, max_dd = _calc_stats_numba(returns.to_numpy())
            return float(sortino), float(profit_factor), float(max_dd)

        equity = (1 + returns).cumprod()
        running_max = equity.cummax()
        drawdown = (equity / running_max) - 1.0
        max_dd = -drawdown.min() * 100
        pos = returns[returns > 0].sum()
        neg = returns[returns < 0].sum()
        profit_factor = pos / abs(neg) if neg != 0 else np.inf
        downside = returns[returns < 0]
        downside_std = downside.std(ddof=0)
        sortino = returns.mean() / downside_std if downside_std != 0 else np.nan
        return sortino, profit_factor, max_dd

    def __call__(self, ga_instance, solution, sol_idx):
        # Determine asset subset for mini-batching
        assets = self.assets
        if config.MINIBATCH.get("enabled"):
            size = config.MINIBATCH.get("size", len(self.assets)) or len(self.assets)
            size = min(size, len(self.assets))
            generation = getattr(ga_instance, "generations_completed", 0)
            if config.MINIBATCH.get("elite_eval_period", 0) > 0 and (
                generation % config.MINIBATCH["elite_eval_period"] == 0
                and sol_idx < config.MINIBATCH.get("elite_count", 0)
            ):
                assets = self.assets
            else:
                rng = np.random.default_rng(
                    config.SCANNER.get("seed", 0) + generation + sol_idx
                )
                assets = list(rng.choice(self.assets, size=size, replace=False))

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
        run_scores: List[float] = []
        per_asset_runs: Dict[str, List[float]] = {}
        diag_saved = False

        if runs > 1 and config.PARALLEL.get("backend") == "multiprocessing":
            args = [
                (solution, base_seed + i, assets) for i in range(runs)
            ]
            with mp.Pool(processes=config.PARALLEL.get("workers") or None) as pool:
                results = pool.starmap(self._evaluate_once, args)
            for score, metrics, _pr, oc, diag, trade_counts in results:
                run_scores.append(score)
                for a, m in metrics.items():
                    per_asset_runs.setdefault(a, []).append(m)
                if not diag_saved and assets == self.assets:
                    self.last_open_count = oc
                    self.last_trade_counts = trade_counts
                    self.last_diagnostics = diag
                    diag_saved = True
        else:
            for i in range(runs):
                score, metrics, _pr, oc, diag, trade_counts = self._evaluate_once(
                    solution, seed=base_seed + i, assets=assets
                )
                run_scores.append(score)
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
        if per_asset_runs and config.ROBUSTNESS.get("lambda_asset_dispersion", 0.0) > 0:
            per_asset_avg = {a: np.mean(v) for a, v in per_asset_runs.items()}
            penalty_asset = config.ROBUSTNESS["lambda_asset_dispersion"] * np.std(
                list(per_asset_avg.values())
            )
        if config.ROBUSTNESS.get("lambda_mc_dispersion", 0.0) > 0:
            penalty_mc = config.ROBUSTNESS["lambda_mc_dispersion"] * dispersion

        result = float(aggregated - penalty_asset - penalty_mc)
        if self.last_diagnostics is not None:
            self.last_diagnostics.update(
                {"mc_runs": runs, "mc_median": aggregated, "mc_dispersion": dispersion}
            )

        if config.SCANNER.get("verbose") and self.last_diagnostics:
            diag = self.last_diagnostics
            print(
                f"Collisions: {diag['collisions']} | Rejected: {diag['rejected']} | "
                f"Acceptance Rate: {diag['acceptance_rate']:.2f} | "
                f"MC Dispersion: {diag['mc_dispersion']:.4f}"
            )
        return result
