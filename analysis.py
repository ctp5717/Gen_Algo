# analysis.py

"""
Analysis & Reporting Module
(This version uses the correct pandas .shift() method for time-based exits)
"""

import vectorbt as vbt
import pandas as pd
import config
import data_loader
import fitness
import strategy_engine as engine
import traceback
import matplotlib.pyplot as plt  # To display plots without blocking


def _reduce_stats_df(stats: pd.DataFrame) -> pd.Series:
    """Reduce multi-column stats to a single series.

    Numerical ratio metrics are averaged while count metrics are summed.
    Non-numeric metrics take the first column's value.
    """

    ratio_metrics = {
        'Total Return [%]',
        'Benchmark Return [%]',
        'Max Drawdown [%]',
        'Sortino Ratio',
        'Sharpe Ratio',
        'Profit Factor',
        'Win Rate [%]',
        'Avg Winning Trade [%]',
        'Avg Losing Trade [%]'
    }
    count_metrics = {'Total Trades'}

    reduced = {}
    for metric in stats.index:
        values = stats.loc[metric]
        numeric_values = pd.to_numeric(values, errors='coerce')
        if metric in count_metrics:
            reduced[metric] = numeric_values.sum()
        elif metric in ratio_metrics:
            reduced[metric] = numeric_values.mean()
        else:
            # Use the first non-null value for non-numeric metrics like dates
            reduced[metric] = values.dropna().iloc[0]

    return pd.Series(reduced)

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
    if validation_data.empty:
        return

    requested = tickers if isinstance(tickers, (list, tuple)) else [tickers]
    if isinstance(validation_data.columns, pd.MultiIndex):
        col_level0 = validation_data.columns.get_level_values(0)
        available = col_level0.unique().tolist()
        missing = [tk for tk in requested if tk not in available]
        if missing:
            print(f"Warning: Missing data for tickers: {', '.join(missing)}")
            if len(missing) / len(requested) > 0.5:
                print("Too many requested tickers missing. Aborting analysis.")
                return
        keep = [tk for tk in requested if tk in available]
        validation_data = validation_data.loc[:, col_level0.isin(keep)]
    else:
        missing = []

    try:
        rules = fitness._inject_genes_into_rules(config.STRATEGY_RULES, gene_map, best_solution)
        entries = engine.process_strategy_rules(validation_data, rules)
        
        trade_count = entries.sum().sum() if isinstance(entries, pd.DataFrame) else entries.sum()
        if trade_count < 1:
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
            
        time_based_exit = entries.shift(config.MAX_HOLD_PERIOD, fill_value=False)
        time_based_exit = time_based_exit.reindex(entries.index, fill_value=False)

        close_prices = (
            validation_data['Close']
            if 'Close' in validation_data
            else validation_data.xs('Close', level=-1, axis=1)
        )

        portfolio = vbt.Portfolio.from_signals(
            close=close_prices,
            entries=entries,
            exits=time_based_exit,
            sl_stop=sl_stop,
            tp_stop=tp_stop,
            sl_trail=sl_trail, # Pass the trailing stop value to the backtester
            fees=0.001,
            freq=config.TIMEFRAME
        )

    except Exception as e:
        print(f"An error occurred during analysis backtest: {e}")
        traceback.print_exc() # Use traceback for more detailed errors
        return

    print("\n--- Validation Period Performance Stats ---")
    stats = portfolio.stats(agg_func=None)
    if isinstance(stats, pd.DataFrame):
        stats = _reduce_stats_df(stats)
    metrics_to_show = [
        'Start',
        'End',
        'Period',
        'Total Return [%]',
        'Benchmark Return [%]',
        'Max Drawdown [%]',
        'Sortino Ratio',
        'Sharpe Ratio',
        'Profit Factor',
        'Win Rate [%]',
        'Total Trades',
        'Avg Winning Trade [%]',
        'Avg Losing Trade [%]'
    ]
    available = [m for m in metrics_to_show if m in stats.index]
    missing = set(metrics_to_show) - set(available)
    if missing:
        print(f"Warning: Missing metrics: {', '.join(sorted(missing))}")
    print(stats.reindex(available).to_string())

    if isinstance(entries, pd.DataFrame) and len(entries.columns) > 1:
        print("\n--- Per-Asset Performance Breakdown ---")
        for asset in entries.columns:
            asset_stats = portfolio[asset].stats(agg_func=None)
            if isinstance(asset_stats, pd.DataFrame):
                asset_stats = _reduce_stats_df(asset_stats)
            print(f"\nAsset: {asset}")
            available = [m for m in metrics_to_show if m in asset_stats.index]
            missing = set(metrics_to_show) - set(available)
            if missing:
                print(
                    f"Warning: Missing metrics for {asset}: {', '.join(sorted(missing))}"
                )
            print(asset_stats.reindex(available).to_string())

    print("\nDisplaying equity curve plot for the validation period...")
    # Enable interactive mode so the plot window does not block execution.
    plt.ion()
    if isinstance(entries, pd.DataFrame) and len(entries.columns) > 1:
        # Plot aggregated equity when multiple assets are present
        fig = portfolio.total().plot(
            title="Champion Strategy Portfolio Performance (Validation)",
        )
    else:
        # Plot the single asset or aggregated column directly
        column = entries.columns[0] if isinstance(entries, pd.DataFrame) else 0
        fig = portfolio.plot(
            column=column,
            title=f"Champion Strategy Performance on {config.SELECTED_ASSET_NAME} (Validation)",
        )
    fig.show()
