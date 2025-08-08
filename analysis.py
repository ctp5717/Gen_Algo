import traceback
from typing import Dict, List
import matplotlib.pyplot as plt
import pandas as pd
import vectorbt as vbt
import config
import data_loader
import fitness
import strategy_engine as engine

def run_champion_analysis(best_solution: List[float], gene_map: Dict[int, dict]) -> None:
    print("\n\n--- Champion Strategy Analysis on Unseen Data ---")
    start = config.VALIDATION_PERIOD['start']
    end = config.VALIDATION_PERIOD['end']

    tickers = getattr(config, 'ASSET_BASKET', [config.TICKER]) if getattr(config, 'PORTFOLIO_OPTIMIZATION_ENABLED', False) else config.TICKER
    validation_data = data_loader.get_data(tickers, start, end, config.TIMEFRAME)
    if validation_data.empty:
        print("No validation data available.")
        return
    try:
        rules = fitness._inject_genes_into_rules(config.STRATEGY_RULES, gene_map, best_solution)
        entries = engine.process_strategy_rules(validation_data, rules)
        total_trades = entries.astype(bool).values.sum() if isinstance(entries, pd.DataFrame) else int(entries.sum())
        if total_trades < 1:
            print("\nChampion strategy produced no trades in the validation period.")
            return

        exit_rules = rules.get('exit_rules', {}) or {}
        def getp(name):
            r = exit_rules.get(name, {}) or {}
            return r.get('params', {}).get('value') if r.get('is_active', False) else None
        sl_stop, sl_trail, tp_stop = getp('stop_loss'), getp('trailing_stop'), getp('take_profit')

        time_based_exit = entries.shift(config.MAX_HOLD_PERIOD, fill_value=False).reindex(entries.index, fill_value=False)
        if isinstance(validation_data.columns, pd.MultiIndex):
            close_prices = validation_data.xs('Close', level=1, axis=1)
        else:
            close_prices = validation_data['Close']

        portfolio = vbt.Portfolio.from_signals(
            close=close_prices, entries=entries, exits=time_based_exit,
            sl_stop=sl_stop, tp_stop=tp_stop, sl_trail=sl_trail,
            fees=0.001, freq=config.TIMEFRAME,
        )
    except Exception as err:
        print(f"An error occurred during analysis backtest: {err}")
        traceback.print_exc()
        return

    print("\n--- Validation Period Performance Stats ---")
    stats = portfolio.stats()
    metrics = ['Total Return [%]','Benchmark Return [%]','Max Drawdown [%]','Sortino Ratio','Sharpe Ratio','Profit Factor','Win Rate [%]','Total Trades']
    if isinstance(stats, dict):
        stats_df = pd.DataFrame([stats])[metrics]
        print(stats_df.to_string(index=False))
    else:
        print(stats[metrics].to_string(index=False))

    if isinstance(close_prices, pd.DataFrame) and close_prices.shape[1] > 1:
        try:
            per_asset_results = []
            for asset in close_prices.columns:
                asset_port = vbt.Portfolio.from_signals(
                    close=close_prices[asset],
                    entries=entries[asset] if isinstance(entries, pd.DataFrame) else entries,
                    exits=time_based_exit[asset] if isinstance(time_based_exit, pd.DataFrame) else time_based_exit,
                    sl_stop=sl_stop, tp_stop=tp_stop, sl_trail=sl_trail, fees=0.001, freq=config.TIMEFRAME,
                )
                asset_stats = asset_port.stats()
                per_asset_results.append({
                    'Asset': asset,
                    'Total Return [%]': asset_stats.get('Total Return [%]', float('nan')) if isinstance(asset_stats, dict) else asset_stats.get('Total Return [%]'),
                    'Max Drawdown [%]': asset_stats.get('Max Drawdown [%]', float('nan')) if isinstance(asset_stats, dict) else asset_stats.get('Max Drawdown [%]'),
                })
            per_df = pd.DataFrame(per_asset_results)
            print("\n--- Per-Asset Breakdown ---")
            print(per_df.to_string(index=False))
        except Exception:
            pass

    print("\nDisplaying equity curve plot for the validation period...")
    plt.ion()
    fig = portfolio.plot(title="Champion Strategy Performance on Validation Portfolio")
    fig.show()
