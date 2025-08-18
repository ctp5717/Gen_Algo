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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # Use non-interactive backend
import artifact_utils
import charts
from multi_asset_fitness import MultiAssetFitnessEvaluator
from log_utils import get_run_logger, log_run_parameters

def run_champion_analysis(best_solution: list, gene_map: dict):
    """
    Runs a full backtest and analysis on the champion solution using validation data.
    """
    print("\n\n--- Champion Strategy Analysis on Unseen Data ---")
    
    print(
        "Loading validation data from "
        f"{config.VALIDATION_PERIOD['start']} to {config.VALIDATION_PERIOD['end']}..."
    )
    validation_data = data_loader.get_data(
        ticker=config.TICKER,
        start_date=config.VALIDATION_PERIOD['start'],
        end_date=config.VALIDATION_PERIOD['end'],
        interval=config.TIMEFRAME
    )
    if validation_data.empty: return

    try:
        rules = fitness._inject_genes_into_rules(config.STRATEGY_RULES, gene_map, best_solution)
        entries = engine.process_strategy_rules(validation_data, rules)
        
        if entries.sum() < 1:
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

        portfolio = vbt.Portfolio.from_signals(
            close=validation_data['Close'],
            entries=entries,
            exits=time_based_exit,
            sl_stop=sl_stop,
            tp_stop=tp_stop,
            sl_trail=sl_trail, # Pass the trailing stop value to the backtester
            fees=config.FEES,
            freq=config.TIMEFRAME
        )

    except Exception as e:
        print(f"An error occurred during analysis backtest: {e}")
        traceback.print_exc() # Use traceback for more detailed errors
        return

    print("\n--- Validation Period Performance Stats ---")
    stats = portfolio.stats()
    metrics_to_show = [
        'Start', 'End', 'Period', 'Total Return [%]', 'Benchmark Return [%]',
        'Max Drawdown [%]', 'Sortino Ratio', 'Sharpe Ratio', 'Profit Factor',
        'Win Rate [%]', 'Total Trades', 'Avg Winning Trade [%]', 'Avg Losing Trade [%]'
    ]
    print(stats[metrics_to_show].to_string())

    print("\nSaving equity curve plot for the validation period...")
    fig = portfolio.plot(
        title=f"Champion Strategy Performance on {config.SELECTED_ASSET_NAME} (Validation)"
    )
    artifact_utils.ARTIFACTS_DIR.mkdir(exist_ok=True)
    path = artifact_utils.ARTIFACTS_DIR / "validation_equity.png"
    fig.savefig(path)
    plt.close(fig)
    artifact_utils.append_to_manifest(path)


def run_champion_analysis_multi(best_solution: list, gene_map: dict):
    """Run multi-asset analysis on the champion using validation data."""
    print("\n\n--- Multi-Asset Champion Analysis on Unseen Data ---")
    logger = get_run_logger()
    log_run_parameters(logger)
    print(
        "Loading validation data from "
        f"{config.VALIDATION_PERIOD['start']} to {config.VALIDATION_PERIOD['end']}..."
    )
    ohlc_dict = data_loader.load_group_data(
        config.ASSET_GROUP,
        config.VALIDATION_PERIOD["start"],
        config.VALIDATION_PERIOD["end"],
        config.TIMEFRAME,
    )
    if not ohlc_dict:
        return

    evaluator = MultiAssetFitnessEvaluator(ohlc_dict, config.STRATEGY_RULES, gene_map)
    res = evaluator._evaluate_once(
        best_solution, seed=config.SCANNER.get("seed", 0), assets=evaluator.assets
    )

    per_asset_sortino = res.per_asset_sortino
    portfolio_returns = res.portfolio_returns
    open_count = res.open_count
    diag = res.diagnostics
    trade_counts = res.trade_counts
    concentration_ratio = res.concentration_ratio

    sortino, profit_factor, max_dd = evaluator._calc_stats(portfolio_returns)
    print("\n--- Validation Period Portfolio Stats ---")
    print(f"Sortino Ratio: {sortino:.3f}")
    print(f"Profit Factor: {profit_factor:.3f}")
    print(f"Max Drawdown [%]: {max_dd:.2f}")
    print(
        f"Collisions: {diag['collisions']} | Rejected: {diag['rejected']} | "
        f"Acceptance Rate: {diag['acceptance_rate']:.2f}"
    )
    conc_label = (
        "diversified" if concentration_ratio <= 0.25 else
        "moderate" if concentration_ratio <= 0.5 else
        "concentrated"
    )
    print(
        f"Portfolio concentration (Herfindahl): {concentration_ratio:.3f} "
        f"({conc_label})"
    )
    logger.info(
        "Portfolio concentration (Herfindahl): %.3f (%s)",
        concentration_ratio,
        conc_label,
    )

    per_asset = diag.get("per_asset", {})
    rows = []
    for asset, stats in per_asset.items():
        candidates = stats.get("candidates", 0)
        accepted = stats.get("accepted", 0)
        acceptance_pct = (accepted / candidates * 100) if candidates else 0.0
        rows.append({
            "Asset": asset,
            "Candidates": candidates,
            "Accepted": accepted,
            "Acceptance%": acceptance_pct,
            "Sortino": per_asset_sortino.get(asset, float("nan")),
        })
    per_asset_df = pd.DataFrame(rows).set_index("Asset")
    print("Per-asset stats:")
    print(
        per_asset_df.to_string(
            formatters={
                "Acceptance%": "{:.2f}".format,
                "Sortino": "{:.3f}".format,
            }
        )
    )

    equity = (1 + portfolio_returns).cumprod()
    fig1, ax1 = plt.subplots()
    equity.plot(
        ax=ax1,
        title=(
            "Portfolio Equity Curve (Validation)\n"
            f"Herfindahl Concentration: {concentration_ratio:.3f}"
        ),
    )
    fig2, ax2 = plt.subplots()
    open_count.plot(ax=ax2, title="Open Positions Over Time")
    fig3, ax3 = plt.subplots()
    trade_counts.plot(kind="bar", ax=ax3, title="Per-Asset Admitted Trades")
    fig4, ax4 = plt.subplots()
    pd.Series(per_asset_sortino).plot(
        kind="bar", ax=ax4, title="Per-Asset Sortino Ratio"
    )
    artifact_utils.ARTIFACTS_DIR.mkdir(exist_ok=True)
    paths = [
        artifact_utils.ARTIFACTS_DIR / "multi_equity.png",
        artifact_utils.ARTIFACTS_DIR / "multi_open_positions.png",
        artifact_utils.ARTIFACTS_DIR / "multi_trade_counts.png",
        artifact_utils.ARTIFACTS_DIR / "multi_sortino.png",
    ]
    for fig, path in zip([fig1, fig2, fig3, fig4], paths):
        fig.savefig(path)
        plt.close(fig)
        artifact_utils.append_to_manifest(path)

    collisions = diag.get("collisions_by_asset")
    if collisions:
        fig = charts.plot_collisions_histogram(
            collisions,
            config.SCANNER.get("max_concurrent_trades"),
            config.SCANNER.get("tie_break_policy"),
        )
        path = artifact_utils.ARTIFACTS_DIR / "multi_collisions_histogram.png"
        fig.savefig(path)
        plt.close(fig)
        artifact_utils.append_to_manifest(path)
    per_asset = diag.get("per_asset")
    if per_asset:
        fig = charts.plot_per_asset_acceptance_rate(
            per_asset,
            config.SCANNER.get("max_concurrent_trades"),
            config.SCANNER.get("tie_break_policy"),
        )
        path = artifact_utils.ARTIFACTS_DIR / "multi_per_asset_acceptance.png"
        fig.savefig(path)
        plt.close(fig)
        artifact_utils.append_to_manifest(path)
    run_scores = diag.get("run_scores")
    if run_scores:
        fig = charts.plot_mc_dispersion(
            run_scores,
            diag.get("mc_median"),
            config.SCANNER.get("max_concurrent_trades"),
            config.SCANNER.get("tie_break_policy"),
        )
        path = artifact_utils.ARTIFACTS_DIR / "multi_mc_dispersion.png"
        fig.savefig(path)
        plt.close(fig)
        artifact_utils.append_to_manifest(path)
