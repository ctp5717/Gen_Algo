# analysis.py

"""
Analysis & Reporting Module
(This version uses the correct pandas .shift() method for time-based exits)
"""

import pandas as pd
import vectorbt as vbt
import numpy as np
import config
import data_loader
import fitness
import strategy_engine as engine
import traceback
import matplotlib.pyplot as plt  # To display plots without blocking

def run_champion_analysis(best_solution: list, gene_map: dict):
    """
    Runs a full backtest and analysis on the champion solution using validation data.
    """
    print("\n\n--- Champion Strategy Analysis on Unseen Data ---")
    
    print(
        "Loading validation data from "
        f"{config.VALIDATION_PERIOD['start']} to {config.VALIDATION_PERIOD['end']}..."
    )
    tickers = (
        config.ASSET_BASKET
        if getattr(config, "PORTFOLIO_OPTIMIZATION_ENABLED", False)
        else config.TICKER
    )
    validation_data = data_loader.get_data(
        ticker=tickers,
        start_date=config.VALIDATION_PERIOD['start'],
        end_date=config.VALIDATION_PERIOD['end'],
        interval=config.TIMEFRAME
    )
    if validation_data.empty: return

    try:
        rules = fitness._inject_genes_into_rules(config.STRATEGY_RULES, gene_map, best_solution)
        entries = engine.process_strategy_rules(validation_data, rules)

        if fitness._count_trades(entries) < 1:
            print("\nChampion strategy produced no trades in the validation period.")
            return
        # --- NEW: Logic to handle multiple, selectable exit types ---
        exit_rules = rules.get('exit_rules', {})
        sl_rule = exit_rules.get('stop_loss', {})
        tsl_rule = exit_rules.get('trailing_stop', {})
        tp_rule = exit_rules.get('take_profit', {})

        sl_stop = sl_rule.get('params', {}).get('value') if sl_rule.get('is_active', False) else None
        sl_trail = tsl_rule.get('params', {}).get('value') if tsl_rule.get('is_active', False) else None
        tp_stop = tp_rule.get('params', {}).get('value') if tp_rule.get('is_active', False) else None

        portfolio, agg_portfolio, agg_stats, per_asset_stats = fitness.run_portfolio_backtest(
            validation_data,
            entries,
            sl_stop=sl_stop,
            sl_trail=sl_trail,
            tp_stop=tp_stop,
            weights=getattr(config, "PORTFOLIO_WEIGHTS", None),
        )

    except Exception as e:
        print(f"An error occurred during analysis backtest: {e}")
        traceback.print_exc() # Use traceback for more detailed errors
        return

    print("\n--- Validation Period Performance Stats ---")
    metrics_to_show = [
        'Start', 'End', 'Period', 'Total Return [%]', 'Benchmark Return [%]',
        'Max Drawdown [%]', 'Sortino Ratio', 'Sharpe Ratio', 'Profit Factor',
        'Win Rate [%]', 'Total Trades', 'Avg Winning Trade [%]', 'Avg Losing Trade [%]',
        'Volatility', 'Calmar Ratio', 'Max Consecutive Losses'
    ]

    agg_to_print = (
        agg_stats.reindex(metrics_to_show)
        if not isinstance(agg_stats, pd.DataFrame)
        else agg_stats.reindex(metrics_to_show)
    )
    print(agg_to_print.to_string())

    if isinstance(per_asset_stats, pd.DataFrame):
        print("\nPer-Asset Breakdown:")
        print(per_asset_stats.reindex(metrics_to_show).to_string())

    print("\nDisplaying equity curve plot for the validation period...")
    # Enable interactive mode so the plot window does not block execution.
    plt.ion()
    title_asset = (
        "Portfolio" if getattr(config, "PORTFOLIO_OPTIMIZATION_ENABLED", False)
        else config.SELECTED_ASSET_NAME
    )
    wrapper = getattr(portfolio, "wrapper", None)
    is_grouped = getattr(wrapper, "grouper", None) is not None
    is_multi = getattr(wrapper, "ndim", 1) > 1

    ax = agg_portfolio.plot(
        title=f"Champion Strategy Performance on {title_asset} (Validation)"
    )
    if hasattr(portfolio, "trades"):
        trades_df = portfolio.trades.records_readable
        if not trades_df.empty:
            cols = list(getattr(portfolio.value(), "columns", []))
            weights = getattr(config, "PORTFOLIO_WEIGHTS", None)
            if weights is None:
                if len(cols) > 0:
                    weights_arr = np.full(len(cols), 1 / len(cols))
                else:
                    weights_arr = np.array([1.0])
                    cols = list(trades_df["Column"].unique())
            else:
                weights_arr = np.asarray(weights, dtype=float)
                weights_arr = weights_arr / weights_arr.sum()
                if len(cols) == 0:
                    cols = list(trades_df["Column"].unique())
            weight_map = {col: w for col, w in zip(cols, weights_arr)}
            trades_df = trades_df.copy()
            trades_df["weighted_pnl"] = trades_df["PnL"] * trades_df["Column"].map(weight_map)
            exit_candidates = ["Exit", "Exit Time", "Exit Price", "exit_time"]
            exit_col = None
            for candidate in exit_candidates:
                if candidate in trades_df.columns:
                    if candidate != "Exit":
                        trades_df = trades_df.rename(columns={candidate: "Exit"})
                    exit_col = "Exit"
                    break
            if exit_col is None:
                raise ValueError(
                    f"Exit column not found. Available columns: {list(trades_df.columns)}"
                )

            ax.scatter(
                trades_df[exit_col],
                trades_df["weighted_pnl"],
                color="red",
                marker="x",
                label="Trade PnL",
            )
            ax.legend()
    ax.figure.show()

    if is_grouped or is_multi:
        if hasattr(portfolio, "columns"):
            for col in portfolio.columns:
                col_fig = portfolio.plot(column=col, title=f"{col} Equity Curve (Validation)")
                col_fig.show()
