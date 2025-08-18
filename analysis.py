# analysis.py

"""
Analysis & Reporting Module
(This version uses the correct pandas .shift() method for time-based exits)
"""

import vectorbt as vbt
import config
import data_loader
import fitness
import strategy_engine as engine
import traceback
import matplotlib.pyplot as plt  # To display plots without blocking
import numpy as np

def run_champion_analysis(best_solution: list, gene_map: dict):
    """Run analysis on the champion solution."""
    if getattr(config, "MULTI_ASSET", {}).get("enabled"):
        _run_multi_asset_analysis(best_solution, gene_map)
        return

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
    if validation_data.empty:
        return

    try:
        rules = fitness._inject_genes_into_rules(config.STRATEGY_RULES, gene_map, best_solution)
        entries = engine.process_strategy_rules(validation_data, rules)

        if entries.sum() < 1:
            print("\nChampion strategy produced no trades in the validation period.")
            return

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
            sl_trail=sl_trail,
            fees=0.001,
            freq=config.TIMEFRAME
        )

    except Exception as e:
        print(f"An error occurred during analysis backtest: {e}")
        traceback.print_exc()
        return

    print("\n--- Validation Period Performance Stats ---")
    stats = portfolio.stats()
    metrics_to_show = [
        'Start', 'End', 'Period', 'Total Return [%]', 'Benchmark Return [%]',
        'Max Drawdown [%]', 'Sortino Ratio', 'Sharpe Ratio', 'Profit Factor',
        'Win Rate [%]', 'Total Trades', 'Avg Winning Trade [%]', 'Avg Losing Trade [%]'
    ]
    print(stats[metrics_to_show].to_string())

    print("\nDisplaying equity curve plot for the validation period...")
    plt.ion()
    fig = portfolio.plot(
        title=f"Champion Strategy Performance on {config.SELECTED_ASSET_NAME} (Validation)"
    )
    fig.show()


def _run_multi_asset_analysis(best_solution: list, gene_map: dict):
    """Generate overview charts for multi-asset validation."""
    print("\n\n--- Multi-Asset Champion Analysis ---")
    group_data = data_loader.get_group_data(
        asset_group=config.ASSET_GROUP,
        start_date=config.VALIDATION_PERIOD['start'],
        end_date=config.VALIDATION_PERIOD['end'],
        interval=config.TIMEFRAME,
        coverage_threshold=config.COVERAGE_THRESHOLD,
    )
    if not group_data:
        print("No validation data available for asset group.")
        return

    settings = config.MULTI_ASSET
    evaluator = fitness.MultiAssetFitnessEvaluator(group_data, config.STRATEGY_RULES, gene_map, settings)
    F = evaluator(None, best_solution, 0)
    details = evaluator.last_details

    per_asset_scores = {
        t: d["score"] for t, d in details["per_asset"].items() if d["score"] is not None
    }
    per_asset_trades = {
        t: d.get("trades", 0) for t, d in details["per_asset"].items() if d["score"] is not None
    }
    equity_curves = {
        t: d.get("equity_curve") for t, d in details["per_asset"].items() if d["score"] is not None
    }
    mu = details.get("mu")
    sigma = details.get("sigma")
    lam_sigma = details.get("lambda_sigma")
    total_trades = details.get("total_trades", 0)
    cov_pen = details.get("penalties", {}).get("coverage", 0.0)
    assets_incl = details.get("assets_included")
    total_assets = len(group_data)
    print(
        f"Fitness: {F:.3f} | Mu: {mu:.3f} | Lambda*Sigma: {lam_sigma:.3f} | Coverage Penalty: {cov_pen:.3f} | Total Trades: {total_trades} | Assets: {assets_incl}/{total_assets}"
    )

    scored = [
        (t, d["score"], d.get("trades", 0))
        for t, d in details["per_asset"].items()
        if d["score"] is not None
    ]
    if scored:
        scored.sort(key=lambda x: x[1])
        bottom = scored[:3]
        top = scored[-3:][::-1]
        print("Top assets:")
        for t, s, tr in top:
            print(f"  {t}: score={s:.3f}, trades={tr}")
        print("Bottom assets:")
        for t, s, tr in bottom:
            print(f"  {t}: score={s:.3f}, trades={tr}")

    charts_cfg = getattr(config, "CHARTS", {})
    _plot_multi_asset_overview(
        per_asset_scores,
        per_asset_trades,
        equity_curves,
        mu,
        sigma,
        lam_sigma,
        F,
        total_trades,
        assets_incl,
        total_assets,
        settings,
        charts_cfg,
    )


def _plot_multi_asset_overview(
    scores,
    trades,
    equities,
    mu,
    sigma,
    lam_sigma,
    F,
    total_trades,
    assets_included,
    total_assets,
    settings,
    charts_cfg,
):
    """Render multi-asset overview charts with KPI strip."""

    plt.ion()
    tickers = list(scores.keys())
    asset_weights = settings.get("asset_weights") or {}
    weights = np.array([asset_weights.get(t, 1.0) for t in tickers], dtype=float)
    weight_sum = weights.sum()
    weights = weights / weight_sum if weight_sum else np.ones_like(weights) / len(weights)

    combined_equity = None
    for w, ticker in zip(weights, tickers):
        eq = equities[ticker]
        eq_norm = eq / eq.iloc[0]
        combined_equity = (
            eq_norm * w if combined_equity is None else combined_equity + eq_norm * w
        )

    max_assets = charts_cfg.get("max_assets_in_overview")
    sorted_items = sorted(scores.items(), key=lambda kv: kv[1])
    if max_assets and len(sorted_items) > max_assets:
        half = max_assets // 2
        bottom_items = sorted_items[:half]
        top_items = sorted_items[-half:][::-1]
        sorted_items = top_items + bottom_items
    else:
        sorted_items = sorted_items[::-1]  # descending

    tick_sorted = [k for k, _ in sorted_items]
    vals_sorted = [v for _, v in sorted_items]

    fig = plt.figure(figsize=(10, 12))
    gs = fig.add_gridspec(4, 1, height_ratios=[0.2, 1, 1, 1])
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])
    ax3 = fig.add_subplot(gs[3])
    ax0.axis("off")
    kpi = (
        f"F={F:.2f} | μ={mu:.2f} | λσ={lam_sigma:.2f} | Trades={total_trades} | "
        f"Assets {assets_included}/{total_assets}"
    )
    ax0.text(0.5, 0.5, kpi, ha="center", va="center", fontsize=10)

    combined_equity.plot(ax=ax1, title="Combined Equity Curve (visualisation only)")
    ax1.set_ylabel("Equity")

    ax2.bar(tick_sorted, vals_sorted)
    ax2.axhline(mu, color="red", linestyle="--", label="μ")
    ax2.axhspan(mu - sigma, mu + sigma, color="red", alpha=0.1, label="μ ± σ")
    ax2.set_title("Per-asset scores")
    ax2.legend()

    trades_sorted = [trades[t] for t in tick_sorted]
    ax3.bar(tick_sorted, trades_sorted)
    if settings.get("min_total_trades"):
        ax3.axhline(
            settings["min_total_trades"] / max(1, total_assets),
            color="gray",
            linestyle="--",
            label="floor/N",
        )
    ax3.set_title("Trades per asset")
    ax3.legend()

    fig.tight_layout()
    fig.show()
