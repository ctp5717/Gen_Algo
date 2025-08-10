# analysis.py

"""
Analysis & Reporting Module
(This version uses the correct pandas .shift() method for time-based exits)
"""

import pandas as pd
import vectorbt as vbt
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
    fig = portfolio.plot(
        title=f"Champion Strategy Performance on {title_asset} (Validation)"
    )
    fig.show()
    if agg_portfolio is not portfolio:
        agg_fig = agg_portfolio.plot(title="Aggregated Portfolio Equity (Validation)")
        agg_fig.show()
