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
import pandas as pd
import math
import json

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
            fees=config.FEES,
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

    settings = dict(config.MULTI_ASSET)
    start = pd.to_datetime(config.VALIDATION_PERIOD["start"])
    end = pd.to_datetime(config.VALIDATION_PERIOD["end"])
    rate = settings.get("min_total_trades_per_year")
    if rate:
        span_years = (end - start).days / 365.25
        floor = math.ceil(rate * span_years)
        settings["min_total_trades"] = floor
        months = (end - start).days / 30.4375
        print(
            f"Scaled min_total_trades (validation): {floor} (rate={rate}/yr, span={int(round(months))}mo)"
        )
    end_str = end.strftime('%Y-%m-%d')
    evaluator = fitness.MultiAssetFitnessEvaluator(group_data, config.STRATEGY_RULES, gene_map, settings)
    F = evaluator(None, best_solution, 0)
    details = evaluator.last_details

    tickers = sorted(
        t for t, d in details["per_asset"].items() if d["score"] is not None
    )
    per_asset_scores = {t: details["per_asset"][t]["score"] for t in tickers}
    per_asset_trades = {
        t: details["per_asset"][t].get("trades", 0) for t in tickers
    }
    equity_curves = {
        t: details["per_asset"][t].get("equity_curve") for t in tickers
    }
    mu = details.get("mu")
    sigma = details.get("sigma")
    lam_sigma = details.get("lambda_sigma")
    lam = settings.get("lambda_dispersion")
    total_trades = details.get("total_trades", 0)
    cov_pen = details.get("penalties", {}).get("coverage", 0.0)
    assets_incl = details.get("assets_included")
    assets_traded = details.get("assets_traded")
    total_assets = len(group_data)
    floor_reason = details.get("penalties", {}).get("trade_floor")
    print(
        f"Fitness: {F:.3f} | Mu: {mu:.3f} | Lambda={lam:.3f} | Lambda*Sigma: {lam_sigma:.3f} | "
        f"Coverage Penalty: {cov_pen:.3f} | Total Trades: {total_trades} | "
        f"Assets included={assets_incl}, traded={assets_traded} / total={total_assets}"
    )
    if isinstance(floor_reason, str):
        print(f"Hard floor triggered ({floor_reason})")
    elif isinstance(floor_reason, dict):
        mode = floor_reason.get("mode")
        if mode == "multiplicative":
            print(f"Soft penalty applied: scale={floor_reason.get('scale'):.3f}")
        elif mode == "additive":
            print(f"Soft penalty applied: penalty={floor_reason.get('penalty'):.3f}")
    if getattr(evaluator, "floor_failures", None):
        ff = dict(evaluator.floor_failures)
        if ff:
            print(f"Hard floor failure counts: {ff}")

    scored = [
        (t, d["score"], d.get("trades", 0))
        for t, d in details["per_asset"].items()
        if d["score"] is not None
    ]
    if scored:
        scored.sort(key=lambda x: x[1])
        n = min(3, len(scored) // 2)
        bottom = scored[:n]
        top = scored[-n:][::-1]
        print("Top assets:")
        for t, s, tr in top:
            print(f"  {t}: score={s:.3f}, trades={tr}")
        print("Bottom assets:")
        for t, s, tr in bottom:
            print(f"  {t}: score={s:.3f}, trades={tr}")

    charts_cfg = getattr(config, "CHARTS", {})
    rows = []
    for t in tickers:
        d = details["per_asset"][t]
        rows.append(
            {
                "ticker": t,
                "included": d.get("included", False),
                "asset_weight": d.get("asset_weight"),
                "score": d.get("score"),
                "trades": d.get("trades", 0),
                "sortino": d.get("sortino"),
                "profit_factor_capped": d.get("profit_factor_capped"),
                "max_drawdown": d.get("max_drawdown"),
                "per_asset_min_trades": settings.get("per_asset_min_trades", 1),
                "reason": "" if d.get("included", False) else "below_per_asset_min_trades",
            }
        )
    if rows:
        df = pd.DataFrame(rows).sort_values("score", ascending=False)
        fname = f"multi_asset_stats_{config.TIMEFRAME}_{end_str}.csv"
        df.to_csv(fname, index=False)
        print(f"Saved per-asset stats: {fname}")
        summary = {
            "F": F,
            "mu": mu,
            "sigma": sigma,
            "lambda_sigma": lam_sigma,
            "lambda_dispersion": lam,
            "total_trades": total_trades,
            "assets_included": assets_incl,
            "assets_traded": assets_traded,
            "total_assets": total_assets,
            "coverage_penalty": cov_pen,
            "min_total_trades": settings.get("min_total_trades"),
            "start_date": config.VALIDATION_PERIOD["start"],
            "end_date": config.VALIDATION_PERIOD["end"],
        }
        jf = f"multi_asset_summary_{config.TIMEFRAME}_{end_str}.json"
        with open(jf, "w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"Saved run summary: {jf}")
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
        assets_traded,
        total_assets,
        settings,
        charts_cfg,
        config.VALIDATION_PERIOD["start"],
        config.VALIDATION_PERIOD["end"],
        settings.get("min_total_trades"),
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
    assets_traded,
    total_assets,
    settings,
    charts_cfg,
    start_date,
    end_date,
    floor,
):
    """Render multi-asset overview charts with KPI strip."""

    plt.ion()
    tickers = sorted(scores.keys())
    asset_weights = settings.get("asset_weights") or {}
    weights = np.array([asset_weights.get(t, 1.0) for t in tickers], dtype=float)
    weight_sum = weights.sum()
    weights = weights / weight_sum if weight_sum else np.ones_like(weights) / len(weights)

    combined_equity = None
    for w, ticker in zip(weights, tickers):
        eq = equities.get(ticker)
        if eq is None or len(eq) == 0:
            continue
        eq_norm = eq / eq.iloc[0]
        combined_equity = (
            eq_norm * w if combined_equity is None else combined_equity + eq_norm * w
        )
    if combined_equity is None:
        combined_equity = pd.Series(dtype=float)

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

    show_dist = charts_cfg.get("show_distribution")
    nrows = 5 if show_dist else 4
    ratios = [0.2, 1, 1, 1, 1] if show_dist else [0.2, 1, 1, 1]
    fig = plt.figure(figsize=(10, 12))
    gs = fig.add_gridspec(nrows, 1, height_ratios=ratios)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])
    if show_dist:
        ax_hist = fig.add_subplot(gs[3])
        ax3 = fig.add_subplot(gs[4])
    else:
        ax3 = fig.add_subplot(gs[3])
        ax_hist = None
    ax0.axis("off")
    lam = settings.get("lambda_dispersion", 0.0)
    kpi = (
        f"F={F:.2f} | μ={mu:.2f} | λ={lam:.2f} | λσ={lam_sigma:.2f} | Trades={total_trades} | "
        f"Assets included={assets_included}, traded={assets_traded} / total={total_assets} | "
        f"Floor={floor} | {start_date} → {end_date}"
    )
    ax0.text(0.5, 0.5, kpi, ha="center", va="center", fontsize=10)

    if combined_equity.empty:
        ax1.set_title("Combined Equity Curve (no tradable assets in selection)")
    else:
        combined_equity.plot(ax=ax1, title="Combined Equity Curve (visualisation only)")
        ax1.set_ylabel("Equity")

    ax2.bar(tick_sorted, vals_sorted)
    ax2.axhline(mu, color="red", linestyle="--", label="μ")
    ax2.axhspan(mu - sigma, mu + sigma, color="red", alpha=0.1, label="μ ± σ")
    ax2.set_title("Per-asset scores")
    ax2.legend()

    if show_dist and ax_hist is not None:
        ax_hist.hist(vals_sorted, bins=max(5, len(vals_sorted) // 2))
        ax_hist.set_title("Score distribution")

    trades_sorted = [trades[t] for t in tick_sorted]
    ax3.bar(tick_sorted, trades_sorted)
    if floor:
        scaled_floor = floor / max(1, total_assets)
        ax3.axhline(
            scaled_floor,
            color="gray",
            linestyle="--",
            label=f"floor/N={scaled_floor:.1f} trades",
        )
    ax3.set_title("Trades per asset")
    ax3.legend()

    fig.tight_layout()
    end_dt = pd.to_datetime(end_date)
    end_str = end_dt.strftime('%Y-%m-%d')
    if charts_cfg.get("save_pngs"):
        fname = f"multi_asset_overview_{config.TIMEFRAME}_{end_str}.png"
        fig.savefig(fname, dpi=144, bbox_inches="tight")
    else:
        fig.show()
