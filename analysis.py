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
import os
import subprocess
from datetime import datetime
from pathlib import Path
from utils import _norm_freq

# Expose last analysis details for external inspection
last_details = {}

def run_champion_analysis(best_solution: list, gene_map: dict):
    """Run analysis on the champion solution."""
    if getattr(config, "MULTI_ASSET", {}).get("enabled") and any(
        g.get("path") for g in gene_map.values()
    ):
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
            freq=_norm_freq(config.TIMEFRAME)
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
    global last_details
    last_details = details

    per_asset_scores = {
        t: d["score"] for t, d in details["per_asset"].items() if d.get("included")
    }
    per_asset_trades = {
        t: d.get("trades", 0) for t, d in details["per_asset"].items() if d.get("included")
    }
    equity_curves = {
        t: d.get("equity_curve") for t, d in details["per_asset"].items() if d.get("included")
    }
    ignored_assets = {
        t: d.get("ignored_reason", "unknown")
        for t, d in details["per_asset"].items()
        if not d.get("included")
    }
    mu = details.get("mu")
    sigma = details.get("sigma")
    lam_sigma = details.get("lambda_sigma")
    total_trades = details.get("total_trades", 0)
    cov_pen = details.get("penalties", {}).get("coverage")
    cov_pen = cov_pen if isinstance(cov_pen, (int, float)) else 0.0
    assets_incl = details.get("assets_included")
    total_assets = len(group_data)
    mu_str = f"{mu:.3f}" if isinstance(mu, (int, float)) else "nan"
    lam_str = f"{lam_sigma:.3f}" if isinstance(lam_sigma, (int, float)) else "nan"
    sigma_str = (
        f"{sigma:.3f}"
        if isinstance(sigma, (int, float))
        else "nan (no scored assets)"
    )
    assets_str = f"{assets_incl}/{total_assets}"
    print(
        "Fitness: {f:.3f} | mu={mu} | sigma={sigma} | lambda*sigma={lam} | "
        "coverage_penalty={cov:.3f} | total_trades={trades} | assets={assets}".format(
            f=F,
            mu=mu_str,
            sigma=sigma_str,
            lam=lam_str,
            cov=cov_pen,
            trades=total_trades,
            assets=assets_str,
        )
    )

    plt.ion()

    scored = [
        (
            t,
            d["score"],
            d.get("trades", 0),
            d.get("profit_factor_capped"),
            d.get("drawdown_score"),
            d.get("penalties"),
        )
        for t, d in details["per_asset"].items()
        if d["score"] is not None
    ]
    export_tickers = {t for t, tr in per_asset_trades.items() if tr > 0}
    if scored:
        scored.sort(key=lambda x: x[1])
        bottom = scored[:3]
        top = scored[-3:][::-1]
        print("Top assets:")
        for t, s, tr, pf, dd, pen in top:
            pf_str = f"{pf:.3f}" if isinstance(pf, (int, float)) else "nan"
            dd_str = f"{dd:.3f}" if isinstance(dd, (int, float)) else "nan"
            pen_str = pen if pen else "None"
            print(
                f"  {t}: score={s:.3f}, trades={tr}, pf={pf_str}, dd={dd_str}, penalties={pen_str}"
            )
        print("Bottom assets:")
        for t, s, tr, pf, dd, pen in bottom:
            pf_str = f"{pf:.3f}" if isinstance(pf, (int, float)) else "nan"
            dd_str = f"{dd:.3f}" if isinstance(dd, (int, float)) else "nan"
            pen_str = pen if pen else "None"
            print(
                f"  {t}: score={s:.3f}, trades={tr}, pf={pf_str}, dd={dd_str}, penalties={pen_str}"
            )
        export_tickers.update(t for t, *_ in top)
        export_tickers.update(t for t, *_ in bottom)

    if ignored_assets:
        print("Ignored assets:")
        for t, reason in ignored_assets.items():
            print(f"  {t}: {reason}")

    if per_asset_scores:
        charts_cfg = getattr(config, "CHARTS", {}).copy()
        if os.getenv("DISABLE_PNG_REPORTS") == "1":
            charts_cfg["save_pngs"] = False
        else:
            charts_cfg["save_pngs"] = True
        run_ts = charts_cfg.get("run_ts") or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        charts_cfg["run_ts"] = run_ts
        try:
            sha = charts_cfg.get("sha") or subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                text=True,
            ).strip()
        except Exception:
            sha = "unknown"
        charts_cfg["sha"] = sha
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
            evaluator.settings,
            charts_cfg,
        )
        rules = fitness._inject_genes_into_rules(config.STRATEGY_RULES, gene_map, best_solution)
        ranked_items = sorted(
            (
                t,
                details["per_asset"][t].get("score"),
            )
            for t in per_asset_scores.keys()
        )
        ranked_items.sort(key=lambda kv: float("-inf") if kv[1] is None else kv[1], reverse=True)
        rank_map = {t: i + 1 for i, (t, _) in enumerate(ranked_items)}
        out_dir = Path("reports") / run_ts
        for t in export_tickers:
            ohlc = group_data.get(t)
            if ohlc is None:
                continue
            d = details["per_asset"][t]
            rank = rank_map.get(t, len(rank_map) + 1)
            score = d.get("score")
            tr = d.get("trades", 0)
            _plot_asset_panels(
                t,
                ohlc,
                rules,
                rank,
                tr,
                score,
                charts_cfg.get("save_pngs"),
                out_dir,
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
    sigma_str = f"{sigma:.2f}" if isinstance(sigma, (int, float)) else "nan"
    floor = settings.get("min_total_trades")
    kpi = (
        f"F={F:.2f} | μ={mu:.2f} | σ={sigma_str} | λσ={lam_sigma:.2f} | "
        f"Trades={total_trades} | Floor={floor} | Assets={assets_included}/{total_assets}"
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
    if floor:
        ax3.axhline(
            floor / max(1, total_assets),
            color="gray",
            linestyle="--",
            label=f"floor/N ({floor}/{total_assets})",
        )
    ax3.set_title("Trades per asset")
    ax3.legend()

    fig.tight_layout()
    fig.show()

    if charts_cfg.get("save_pngs"):
        run_ts = charts_cfg.get("run_ts") or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        try:
            sha = charts_cfg.get("sha") or subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                text=True,
            ).strip()
        except Exception:
            sha = "unknown"
        out_dir = Path("reports") / run_ts
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"overview_{sha}.png"
        fig.savefig(out_path, dpi=150)


def _plot_asset_panels(
    ticker,
    ohlc,
    rules,
    rank,
    trades,
    score,
    save_png,
    out_dir,
):
    """Render a three-panel chart for a single asset."""

    plt.ion()

    entries = engine.process_strategy_rules(ohlc, rules)
    exit_rules = rules.get("exit_rules", {})
    sl_rule = exit_rules.get("stop_loss", {})
    tsl_rule = exit_rules.get("trailing_stop", {})
    tp_rule = exit_rules.get("take_profit", {})
    sl_stop = sl_rule.get("params", {}).get("value") if sl_rule.get("is_active", False) else None
    sl_trail = tsl_rule.get("params", {}).get("value") if tsl_rule.get("is_active", False) else None
    tp_stop = tp_rule.get("params", {}).get("value") if tp_rule.get("is_active", False) else None
    time_exit = entries.shift(config.MAX_HOLD_PERIOD, fill_value=False)
    time_exit = time_exit.reindex(entries.index, fill_value=False)

    portfolio = vbt.Portfolio.from_signals(
        close=ohlc["Close"],
        entries=entries,
        exits=time_exit,
        sl_stop=sl_stop,
        tp_stop=tp_stop,
        sl_trail=sl_trail,
        fees=0.001,
        freq=_norm_freq(config.TIMEFRAME),
    )

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

    ohlc["Close"].plot(ax=ax1, title=f"{ticker} Price & Signals")
    records = portfolio.trades.records
    if not records.empty:
        entry_dates = ohlc.index[records["entry_idx"]]
        entry_prices = ohlc["Close"].iloc[records["entry_idx"]]
        exit_dates = ohlc.index[records["exit_idx"]]
        exit_prices = ohlc["Close"].iloc[records["exit_idx"]]
        ax1.scatter(entry_dates, entry_prices, marker="^", color="green", label="Entry")
        ax1.scatter(exit_dates, exit_prices, marker="v", color="red", label="Exit")
        ax1.legend()

        pnl_pct = records["pnl"] / records["entry_price"]
        colors = np.where(pnl_pct >= 0, "green", "red")
        ax2.scatter(exit_dates, pnl_pct, c=colors)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_title("Trade PnL (%)")
    ax2.set_ylabel("PnL %")

    equity = portfolio.value()
    benchmark = ohlc["Close"] / ohlc["Close"].iloc[0] * equity.iloc[0]
    ax3.plot(equity.index, equity, label="Strategy")
    ax3.plot(benchmark.index, benchmark, label="Buy & Hold")
    ax3.set_title("Equity vs Benchmark")
    ax3.legend()

    fig.tight_layout()
    fig.show()

    if save_png:
        out_dir.mkdir(parents=True, exist_ok=True)
        score_val = score if isinstance(score, (int, float)) else float("nan")
        out_path = out_dir / f"{rank}_{ticker}_{trades}_{score_val:.2f}.png"
        fig.savefig(out_path, dpi=150)
