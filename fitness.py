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


def _count_trades(entries: pd.DataFrame) -> int:
    """Return the total number of entry signals.

    Works for both Series and DataFrame inputs and ensures consistent
    trade-counting logic across the codebase.
    """
    if isinstance(entries, pd.DataFrame):
        return int(entries.to_numpy().sum())
    return int(entries.sum())


def run_portfolio_backtest(
    ohlc_data: pd.DataFrame,
    entries: pd.DataFrame,
    sl_stop: float | None = None,
    sl_trail: float | None = None,
    tp_stop: float | None = None,
    weights: list[float] | np.ndarray | None = None,
):
    """Backtest entries on single or multi-asset data.

    Parameters
    ----------
    ohlc_data : pd.DataFrame
        OHLCV data for one or many assets.  Multi-asset data must use a
        ``MultiIndex`` column layout where the first level is the ticker and
        the second level is the OHLCV field.
    entries : pd.DataFrame
        Entry signals produced by the strategy engine.  Should have the same
        column layout as ``ohlc_data``'s close prices.
    sl_stop, sl_trail, tp_stop : float, optional
        Stop-loss, trailing-stop and take-profit parameters passed directly to
        ``vectorbt.Portfolio.from_signals``.
    weights : list[float] or np.ndarray, optional
        Custom portfolio weights applied across assets.  If ``None`` each asset
        is equally weighted.  The weights are normalised to sum to one and are
        used when aggregating per-asset equity curves after the backtest.  They
        are not passed to ``vectorbt.Portfolio.from_signals`` due to API
        changes in recent versions of *vectorbt*.

    Returns
    -------
    tuple
        ``(portfolio, agg_portfolio, agg_stats, per_asset_stats)`` where
        ``agg_portfolio`` represents the combined equity curve when multiple
        assets are present.
    """
    if not hasattr(vbt, "Portfolio"):
        raise RuntimeError("vectorbt is required for backtesting but is not installed")

    time_exit = entries.shift(config.MAX_HOLD_PERIOD, fill_value=False)
    time_exit = time_exit.reindex(entries.index, fill_value=False)

    close_prices = (
        ohlc_data.xs("Close", level=1, axis=1)
        if isinstance(ohlc_data.columns, pd.MultiIndex)
        else ohlc_data["Close"]
    )

    # Equal-weight all assets unless custom weights are provided
    group_by = None
    weights_arr = None
    if isinstance(close_prices, pd.DataFrame) and close_prices.shape[1] > 1:
        group_by = close_prices.columns
        if weights is None:
            weights_arr = np.full(close_prices.shape[1], 1 / close_prices.shape[1])
        else:
            weights_arr = np.asarray(weights, dtype=float)
            if weights_arr.size != close_prices.shape[1]:
                raise ValueError("weights length must match number of assets")
            weights_arr = weights_arr / weights_arr.sum()

    portfolio = vbt.Portfolio.from_signals(
        close=close_prices,
        entries=entries,
        exits=time_exit,
        sl_stop=sl_stop,
        tp_stop=tp_stop,
        sl_trail=sl_trail,
        fees=0.001,
        freq=config.TIMEFRAME,
        group_by=group_by,
    )

    agg_portfolio = portfolio.value()
    if isinstance(close_prices, pd.DataFrame):
        if weights_arr is None:
            weights_arr = np.full(close_prices.shape[1], 1 / close_prices.shape[1])

        try:
            per_asset_stats = pd.concat(
                {col: portfolio.stats(column=col) for col in close_prices.columns},
                axis=1,
            )
        except TypeError:
            stats_df = portfolio.stats()
            per_asset_stats = (
                stats_df if isinstance(stats_df, pd.DataFrame) else pd.DataFrame(stats_df)
            )
        per_asset_stats = per_asset_stats.applymap(
            lambda x: (
                int(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0))
                if isinstance(x, (int, np.integer))
                else float(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0))
                if isinstance(x, (float, np.floating))
                else x
            )
        )

        try:
            agg_stats = portfolio.stats(silence_warnings=True)
        except TypeError:
            stats_res = portfolio.stats()
            agg_stats = stats_res if isinstance(stats_res, pd.Series) else stats_res.iloc[:, 0]
        agg_stats = agg_stats.astype(object)

        weighted_value = (portfolio.value() * weights_arr).sum(axis=1)
        agg_portfolio = weighted_value

        if hasattr(portfolio, "returns"):
            weighted_returns = (portfolio.returns() * weights_arr).sum(axis=1)
            agg_stats["Volatility"] = float(weighted_returns.std())
        else:
            agg_stats["Volatility"] = np.nan

        if hasattr(portfolio, "trades"):
            trades_df = portfolio.trades.records_readable.copy()
            if not trades_df.empty:
                weight_map = {col: w for col, w in zip(close_prices.columns, weights_arr)}
                trades_df["weighted_pnl"] = trades_df["PnL"] * trades_df["Column"].map(weight_map)
                losses = trades_df["weighted_pnl"] < 0
                max_consec = (
                    losses.groupby((losses != losses.shift()).cumsum()).cumsum().max()
                )
                agg_stats["Max Consecutive Losses"] = int(max_consec) if pd.notna(max_consec) else 0
            else:
                agg_stats["Max Consecutive Losses"] = 0
        else:
            agg_stats["Max Consecutive Losses"] = 0

        if "Total Trades" in per_asset_stats.index:
            agg_stats["Total Trades"] = int(
                np.nan_to_num(
                    per_asset_stats.loc["Total Trades"], nan=0.0, posinf=0.0, neginf=0.0
                ).sum()
            )

        for key, val in agg_stats.items():
            if isinstance(val, (int, np.integer)):
                agg_stats[key] = int(val)
            elif isinstance(val, (float, np.floating)):
                agg_stats[key] = float(
                    np.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)
                )
    else:
        per_asset_stats = portfolio.stats()
        per_asset_stats = per_asset_stats.apply(
            lambda x: (
                int(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0))
                if isinstance(x, (int, np.integer))
                else float(
                    np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                )
                if isinstance(x, (float, np.floating))
                else x
            )
        )
        agg_stats = per_asset_stats.copy().astype(object)
        if "Total Trades" in per_asset_stats.index:
            agg_stats["Total Trades"] = int(per_asset_stats.loc["Total Trades"])

        for key, val in agg_stats.items():
            if isinstance(val, (int, np.integer)):
                agg_stats[key] = int(val)
            elif isinstance(val, (float, np.floating)):
                agg_stats[key] = float(
                    np.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)
                )

    return portfolio, agg_portfolio, agg_stats, per_asset_stats

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

        current_level[param_key] = gene_info["type"](gene_value)

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

            if _count_trades(entries) < config.FITNESS_WEIGHTS['min_trades']:
                return -1.0

            # --- NEW: Logic to handle multiple, selectable exit types ---
            exit_rules = rules.get('exit_rules', {})
            sl_rule = exit_rules.get('stop_loss', {})
            tsl_rule = exit_rules.get('trailing_stop', {})
            tp_rule = exit_rules.get('take_profit', {})

            sl_stop = (
                sl_rule.get('params', {}).get('value')
                if sl_rule.get('is_active', False)
                else None
            )
            sl_trail = (
                tsl_rule.get('params', {}).get('value')
                if tsl_rule.get('is_active', False)
                else None
            )
            tp_stop = (
                tp_rule.get('params', {}).get('value')
                if tp_rule.get('is_active', False)
                else None
            )

            _, _, agg_stats, _ = run_portfolio_backtest(
                self.ohlc_data,
                entries,
                sl_stop=sl_stop,
                sl_trail=sl_trail,
                tp_stop=tp_stop,
                weights=getattr(config, "PORTFOLIO_WEIGHTS", None),
            )

            if isinstance(agg_stats, pd.DataFrame):
                # Should not happen but keep fallback
                sortino = agg_stats.loc.get('Sortino Ratio', [np.nan])[0]
                profit_factor = agg_stats.loc.get('Profit Factor', [np.nan])[0]
                max_drawdown = agg_stats.loc.get('Max Drawdown [%]', [np.nan])[0]
                volatility = agg_stats.loc.get('Volatility', [np.nan])[0]
            else:
                sortino = agg_stats.get('Sortino Ratio', np.nan)
                profit_factor = agg_stats.get('Profit Factor', np.nan)
                max_drawdown = agg_stats.get('Max Drawdown [%]', np.nan)
                volatility = agg_stats.get('Volatility', np.nan)

            if np.isinf(sortino):
                sortino = 0.0
            if np.isinf(profit_factor) or profit_factor > 5:
                profit_factor = 5.0
            if np.isinf(max_drawdown):
                max_drawdown = 100.0

            metrics = [sortino, profit_factor, max_drawdown, volatility]
            if any(not np.isfinite(m) for m in metrics) or volatility == 0:
                return -999.0

            drawdown_score = 1 - (max_drawdown / 100.0)
            weights = config.FITNESS_WEIGHTS

            fitness_score = (
                (sortino * weights['sortino_ratio']) +
                (profit_factor * weights['profit_factor']) +
                (drawdown_score * weights['max_drawdown'])
            )

            return fitness_score if np.isfinite(fitness_score) else -999.0

        except Exception as e:
            print(f"Error in fitness evaluation: {e}")
            return -999.0
