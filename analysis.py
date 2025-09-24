# analysis.py

"""
Analysis & Reporting Module
(This version uses the correct pandas .shift() method for time-based exits)
"""

import hashlib
import json
import os
import re
import subprocess
import traceback
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt  # To display plots without blocking
import numpy as np
import pandas as pd
import vectorbt as vbt

import config
import data_loader
import fitness
import strategy_engine as engine
import trade_floor
from deps import ensure_real_vectorbt
from exits_nb import (
    ExitReason,
    coerce_exit_params,
    compute_exit_metrics,
    generate_dynamic_exit_signals_nb,
    summarise_exit_reasons,
)
from params_resolver import inject_genes_into_rules
from portfolio_utils import extract_exit_params
from run_metadata import merge_run_metadata
from strategy_rules import STRATEGY_RULES

RUN_DIR = Path(".")


ConfigError = getattr(config, "ConfigurationError", RuntimeError)


def set_run_dir(path: Path) -> None:
    """Set the directory where analysis artifacts and metadata are stored."""
    global RUN_DIR
    RUN_DIR = Path(path)


def _write_exit_reason_breakdown_csv(
    summary: Mapping[str, Mapping[str, Mapping[str, float]]],
    artifacts: list[str],
) -> None:
    """Persist aggregated exit fractions per reason for later inspection."""

    settings = getattr(config, "EXIT_TELEMETRY", {})
    if not settings:
        return
    if not settings.get("write_reason_breakdown_csv", True):
        return
    rows: list[dict[str, object]] = []
    for asset, reasons in summary.items():
        for reason_name_key, stats in reasons.items():
            try:
                code_int = int(stats.get("code", 0))
            except (TypeError, ValueError):
                continue
            try:
                reason_enum = ExitReason(code_int)
                reason_name = reason_enum.name
            except ValueError:
                reason_name = str(reason_name_key)
            rows.append(
                {
                    "asset": asset,
                    "reason_code": code_int,
                    "reason_name": reason_name,
                    "count": float(stats.get("count", 0.0)),
                    "fraction": float(stats.get("fraction", 0.0)),
                }
            )
    if not rows:
        return
    df = pd.DataFrame(rows)
    df = df.sort_values(["asset", "reason_code"]).reset_index(drop=True)
    csv_path = RUN_DIR / "exit_reason_breakdown.csv"
    df.to_csv(csv_path, index=False)
    artifacts.append(str(csv_path))


def _render_exit_kpi_strip(
    metrics: Mapping[str, float], artifacts: list[str]
) -> dict[str, float]:
    """Persist a small exit KPI table and return the values recorded."""

    if not metrics:
        return {}

    avg_tp_level = float(metrics.get("avg_tp_level_reached", 0.0))
    be_rate = float(metrics.get("breakeven_touch_rate", 0.0)) * 100.0
    timeout_rate = float(metrics.get("sl_timeout_usage_rate", 0.0)) * 100.0
    trailing_rate = float(metrics.get("trailing_tp_hit_rate", 0.0)) * 100.0

    kpi_values = {
        "avg_tp_level_reached": avg_tp_level,
        "breakeven_touch_pct": be_rate,
        "sl_timeout_usage_pct": timeout_rate,
        "trailing_tp_hit_pct": trailing_rate,
    }

    df = pd.DataFrame.from_dict(
        kpi_values, orient="index", columns=["value"]
    ).rename_axis("metric")
    print("\nExit KPI strip:\n" + df.to_string(float_format=lambda v: f"{v:.3f}"))
    kpi_path = RUN_DIR / "exit_kpi_strip.csv"
    df.to_csv(kpi_path)
    artifacts.append(str(kpi_path))
    return kpi_values


def _get_commit_hash() -> str:
    """Return the current git commit hash or "unknown" if unavailable."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _file_sha256(path: Path) -> str | None:
    """Return the SHA256 hash of a file or ``None`` if missing."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except FileNotFoundError:
        return None


def _get_cache_hashes() -> dict:
    """Compute hashes for any cache files backing the current run."""
    config.initialize_config()
    train_start = pd.to_datetime(config.TRAINING_PERIOD["start"])
    train_end = pd.to_datetime(config.TRAINING_PERIOD["end"])
    val_start = pd.to_datetime(config.VALIDATION_PERIOD["start"])
    val_end = pd.to_datetime(config.VALIDATION_PERIOD["end"])
    wf_settings = getattr(config, "WALK_FORWARD_SETTINGS", {})
    wf_enabled = wf_settings.get(
        "enabled", getattr(config, "ENABLE_WALK_FORWARD_VALIDATION", False)
    )
    if wf_enabled:
        wf_range = wf_settings.get("total_data_range", {})
        wf_start = pd.to_datetime(wf_range.get("start", train_start))
        wf_end = pd.to_datetime(wf_range.get("end", val_end))
    else:
        wf_start, wf_end = train_start, val_end
    earliest = min(train_start, val_start, wf_start).strftime("%Y-%m-%d")
    latest = max(train_end, val_end, wf_end).strftime("%Y-%m-%d")
    source = getattr(config, "DATA_SOURCE", "")
    interval = getattr(config, "TIMEFRAME", "")

    tickers = (
        [t for _, t in getattr(config, "ASSET_GROUP", [])]
        if getattr(config, "MULTI_ASSET", {}).get("enabled")
        else [config.TICKER]
    )
    hashes = {}
    for t in tickers:
        cache_stem = data_loader.build_cache_stem(
            t,
            earliest,
            latest,
            interval,
            source=source,
        )
        for ext in (
            data_loader.CACHE_EXTENSION,
            data_loader.LEGACY_CACHE_EXTENSION,
        ):
            fname = f"{cache_stem}{ext}"
            fpath = Path(data_loader.CACHE_DIR) / fname
            hashes[fname] = _file_sha256(fpath)
    return hashes


def _write_run_metadata(
    start_time: datetime, artifacts: list[str], extra: dict | None = None
) -> None:
    """Persist run metadata for reproducibility."""
    end_time = datetime.now(timezone.utc)

    # Only include artifact paths that actually exist
    existing_artifacts = []
    run_dir = RUN_DIR.resolve()
    for a in artifacts:
        p = Path(a)
        if not p.is_absolute():
            p = run_dir / p
        if not p.exists():
            continue
        resolved = p.resolve()
        try:
            rel = resolved.relative_to(run_dir)
            existing_artifacts.append(str(rel))
        except ValueError:
            existing_artifacts.append(str(resolved))

    # Deduplicate while preserving order
    existing_artifacts = list(dict.fromkeys(existing_artifacts))

    metadata = {
        "artifact_version": "1.0.0",
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "wall_time": (end_time - start_time).total_seconds(),
        "data_source": config.DATA_SOURCE,
        "cpu_count": os.cpu_count(),
        "cache_files": _get_cache_hashes(),
        "library_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "vectorbt": {
                "version": getattr(vbt, "__version__", "unknown"),
                "path": str(Path(getattr(vbt, "__file__", "")).resolve()),
            },
        },
        "artifacts": existing_artifacts,
    }
    if extra:
        metadata.update(extra)
    merge_run_metadata(RUN_DIR / "run_metadata.json", metadata)


def _canonical_rule_slugs(rules: dict):
    conds = rules.get("entry_rules", {}).get("conditions", [])
    slugs = []
    mapping = {}
    used = set()
    for i, rule in enumerate(conds):
        name = engine.canonical_rule_label(rule)
        base = re.sub(r"[^0-9a-z]+", "_", name.lower()).strip("_")
        slug = base
        while slug in used:
            slug = f"{base}__{i}"
        used.add(slug)
        slugs.append(slug)
        mapping[slug] = name
    return slugs, mapping


def _build_per_asset_counts(per_asset_signal_counts, rules):
    slugs, mapping = _canonical_rule_slugs(rules)
    entry = rules.get("entry_rules", {})
    combo = entry.get("combination_logic")
    vt = entry.get("vote_threshold")
    nan_policy = entry.get("nan_policy", config.NAN_POLICY)
    rows = []
    for asset in sorted(per_asset_signal_counts):
        counts = per_asset_signal_counts[asset]
        row = {
            "asset": asset,
            "combination_logic": combo,
            "vote_threshold": vt,
            "nan_policy": nan_policy,
        }
        for slug in slugs:
            row[f"count_{slug}"] = counts.get(mapping[slug], 0)
        rows.append(row)
    columns = [
        "asset",
        "combination_logic",
        "vote_threshold",
        "nan_policy",
    ] + [f"count_{s}" for s in slugs]
    return pd.DataFrame(rows, columns=columns)


def _coerce_numeric_cols(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def run_champion_analysis(
    best_solution: list,
    gene_map: dict,
    validation_data,
    artifacts: list[str] | None = None,
):
    """Run analysis on the champion solution using preloaded data."""
    config.initialize_config()
    ensure_real_vectorbt(Path(__file__).resolve().parent)
    start_time = datetime.now(timezone.utc)
    artifacts = [] if artifacts is None else list(artifacts)
    multi_asset_cfg = getattr(config, "MULTI_ASSET", {})
    if multi_asset_cfg.get("enabled") and isinstance(validation_data, Mapping):
        _run_multi_asset_analysis(best_solution, gene_map, validation_data, artifacts)
        return

    print("\n\n--- Champion Strategy Analysis on Unseen Data ---")
    if validation_data is None or validation_data.empty:
        _write_run_metadata(start_time, artifacts)
        return

    try:
        rules = inject_genes_into_rules(STRATEGY_RULES, gene_map, best_solution)
        outputs = engine.process_strategy_rules(
            validation_data, rules, collect_counts=True
        )
        if isinstance(outputs, tuple):
            entries, signal_counts = outputs
        else:  # pragma: no cover - backward compatibility
            entries, signal_counts = outputs, {}

        if entries.sum() < 1:
            print("\nChampion strategy produced no trades in the validation period.")
            return

        exit_rules = rules.get("exit_rules", {})
        exit_params_snapshot: dict | None = None
        exit_reason_summary: dict[str, dict[str, dict[str, float]]] = {}
        exit_metrics_summary: dict[str, dict[str, float]] = {}
        exit_kpi_values: dict[str, float] = {}
        if config.USE_DYNAMIC_EXIT_SIMULATOR:
            base_entry_size = float(getattr(config, "BASE_ENTRY_SIZE", 1.0))
            size_mode = getattr(config, "DYNAMIC_EXIT_SIZE_MODE", "fraction_base")
            try:
                exit_params = coerce_exit_params(
                    exit_rules,
                    config.MAX_HOLD_PERIOD,
                    getattr(config, "TIMEFRAME", None),
                )
                price_map = {
                    "close": validation_data["Close"].to_numpy(dtype=float),
                    "high": validation_data.get(
                        "High", validation_data["Close"]
                    ).to_numpy(dtype=float),
                    "low": validation_data.get(
                        "Low", validation_data["Close"]
                    ).to_numpy(dtype=float),
                }
                telemetry_cfg = getattr(config, "EXIT_TELEMETRY", {})
                collect_traces = telemetry_cfg.get(
                    "collect_traces", telemetry_cfg.get("enabled", True)
                )
                exit_result = generate_dynamic_exit_signals_nb(
                    entries.to_numpy(dtype=bool),
                    price_map,
                    exit_params,
                    seed=getattr(config, "SEED", None),
                    collect_traces=collect_traces,
                )
            except ValueError as exc:
                print(f"Invalid exit configuration for analysis: {exc}")
                error_note = {"exit": {"error": str(exc)}}
                _write_run_metadata(start_time, artifacts, error_note)
                return
            exits_series = pd.Series(exit_result.exits, index=entries.index)
            exit_size_series = pd.Series(
                exit_result.exit_size, index=entries.index, dtype=float
            )
            exit_reason_summary = summarise_exit_reasons(exit_result, [config.TICKER])
            exit_metrics_summary = compute_exit_metrics(exit_result, [config.TICKER])
            _write_exit_reason_breakdown_csv(exit_reason_summary, artifacts)
            ticker_metrics = exit_metrics_summary.get(config.TICKER, {})
            exit_kpi_values = _render_exit_kpi_strip(ticker_metrics, artifacts)
            exit_params_snapshot = exit_params.as_dict()
            timeframe_snapshot = getattr(config, "TIMEFRAME", None)
            exit_params_snapshot.setdefault("timeframe", timeframe_snapshot)
            if hasattr(config, "get_tp_cap_for_timeframe"):
                exit_params_snapshot.setdefault(
                    "tp_cap",
                    config.get_tp_cap_for_timeframe(timeframe_snapshot),
                )
            entries_active, exits_series, size_series = (
                fitness.build_dynamic_exit_orders(
                    entries=entries,
                    exits_series=exits_series,
                    exit_size_series=exit_size_series,
                    base_entry_size=base_entry_size,
                    mode=size_mode,
                    asset_label=str(config.TICKER or "asset"),
                )
            )
            accumulate_flag = bool(getattr(config, "DYNAMIC_EXIT_ACCUMULATE", False))
            if not accumulate_flag:
                raise ConfigError(
                    "Dynamic exit simulator requires"
                    " config.DYNAMIC_EXIT_ACCUMULATE=True for staged exits"
                )
            portfolio = vbt.Portfolio.from_signals(
                close=validation_data["Close"],
                entries=entries_active,
                exits=exits_series,
                size=size_series,
                accumulate=accumulate_flag,
                size_type="amount",
                fees=config.FEES,
                freq=config.to_pandas_freq(config.TIMEFRAME),
            )
        else:
            time_based_exit, sl_stop, sl_trail, tp_stop = extract_exit_params(
                entries, exit_rules, config.MAX_HOLD_PERIOD
            )
            portfolio = vbt.Portfolio.from_signals(
                close=validation_data["Close"],
                entries=entries,
                exits=time_based_exit,
                sl_stop=sl_stop,
                tp_stop=tp_stop,
                sl_trail=sl_trail,
                fees=config.FEES,
                freq=config.to_pandas_freq(config.TIMEFRAME),
            )

    except Exception as e:
        print(f"An error occurred during analysis backtest: {e}")
        traceback.print_exc()
        _write_run_metadata(start_time, artifacts)
        return

    print("\n--- Validation Period Performance Stats ---")
    stats = portfolio.stats()
    metrics_to_show = [
        "Start",
        "End",
        "Period",
        "Total Return [%]",
        "Benchmark Return [%]",
        "Max Drawdown [%]",
        "Sortino Ratio",
        "Sharpe Ratio",
        "Profit Factor",
        "Win Rate [%]",
        "Total Trades",
        "Avg Winning Trade [%]",
        "Avg Losing Trade [%]",
    ]
    print(stats[metrics_to_show].to_string())

    print("\nDisplaying equity curve plot for the validation period...")
    plt.ion()
    fig = portfolio.plot(
        title=f"Champion Strategy Performance on {config.SELECTED_ASSET_NAME} (Validation)"
    )
    fig.show()
    fig_path = RUN_DIR / "champion_equity.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    artifacts.append(str(fig_path))
    combo = rules.get("entry_rules", {}).get("combination_logic")
    vt = rules.get("entry_rules", {}).get("vote_threshold")
    extra = {
        "combination_logic": combo,
        "vote_threshold": vt,
        "per_asset_signal_counts": {config.TICKER: signal_counts},
    }
    if exit_params_snapshot:
        extra["exit"] = {
            "params": exit_params_snapshot,
            "reason_breakdown": exit_reason_summary.get(config.TICKER, {}),
            "metrics": exit_metrics_summary.get(config.TICKER, {}),
        }
        if exit_kpi_values:
            extra["exit"]["kpi_strip"] = exit_kpi_values
    _write_run_metadata(start_time, artifacts, extra)


def _run_multi_asset_analysis(
    best_solution: list, gene_map: dict, group_data: dict, artifacts: list[str]
):
    """Generate overview charts for multi-asset validation."""
    config.initialize_config()
    start_time = datetime.now(timezone.utc)
    artifacts = list(artifacts)
    print("\n\n--- Multi-Asset Champion Analysis ---")
    if not group_data:
        print("No validation data available for asset group.")
        _write_run_metadata(start_time, artifacts)
        return

    settings = dict(config.MULTI_ASSET)
    settings["collect_equity_curve"] = True
    start = pd.to_datetime(config.VALIDATION_PERIOD["start"])
    end = pd.to_datetime(config.VALIDATION_PERIOD["end"])
    per_asset_base = settings.get("per_asset_min_trades")
    if per_asset_base:
        floor_pa, info_pa = trade_floor.scale_floor(
            per_asset_base, start, end, settings.get("trading_days_per_year", 252)
        )
        settings["per_asset_min_trades"] = floor_pa
        settings["per_asset_floor_info"] = info_pa
        print(
            f"Per-asset floor: base={per_asset_base} → scaled={floor_pa} "
            f"(window={info_pa['window_days']}d, base={info_pa['trading_days_per_year']}d)"
        )
    rate = settings.get("min_total_trades_per_year")
    if rate:
        floor, info = trade_floor.scale_floor(
            rate, start, end, settings.get("trading_days_per_year", 252)
        )
        settings["min_total_trades"] = floor
        print(f"Scaled min_total_trades (validation): {floor} | info={info}")
    end_str = end.strftime("%Y-%m-%d")
    evaluator = fitness.MultiAssetFitnessEvaluator(
        group_data, STRATEGY_RULES, gene_map, settings
    )
    F = evaluator(None, best_solution, 0)
    details = evaluator.last_details
    rules = inject_genes_into_rules(STRATEGY_RULES, gene_map, best_solution)
    combo = rules.get("entry_rules", {}).get("combination_logic")
    vt = rules.get("entry_rules", {}).get("vote_threshold")
    fitness.print_floor_failures(getattr(evaluator, "floor_failures", {}))

    tickers = sorted(
        t for t, d in details["per_asset"].items() if d["score"] is not None
    )
    per_asset_scores = {t: details["per_asset"][t]["score"] for t in tickers}
    per_asset_trades = {t: details["per_asset"][t].get("trades", 0) for t in tickers}
    equity_curves = {t: details["per_asset"][t].get("equity_curve") for t in tickers}
    weights_map = dict(settings.get("asset_weights") or {})
    frames = []
    for t in tickers:
        eq = equity_curves.get(t)
        if eq is None or len(eq) == 0:
            continue
        base = float(eq.iloc[0])
        if base == 0:
            continue
        frames.append(eq.rename(t) / base)

    combined = None
    if frames:
        df = pd.concat(frames, axis=1).sort_index()
        # Cast to float before calling ``fillna`` to avoid pandas object-dtype
        # ``FutureWarning`` when filling missing values on mixed inputs.
        w = (
            pd.Series(weights_map, dtype=float)
            .reindex(df.columns)
            .astype(float)
            .fillna(1.0)
        )
        w = w / w.sum() if w.sum() else w

        mask = df.notna()
        active_w = mask.mul(w, axis=1)
        row_w = active_w.sum(axis=1).replace(0.0, np.nan)
        combined = (df.fillna(0.0) * active_w).sum(axis=1) / row_w

    if combined is not None:
        combined = combined.where(np.isfinite(combined))
        if not combined.dropna().empty:
            plt.ion()
            fig, ax = plt.subplots()
            combined.plot(ax=ax, title="Champion Equity Curve")
            ax.set_xlabel("Date")
            ax.set_ylabel("Equity")
            champ_path = RUN_DIR / "champion_equity.png"
            fig.savefig(champ_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            artifacts.append(str(champ_path))
            _write_run_metadata(start_time, [str(champ_path)])
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
    w_map = details.get("asset_weights", {})
    if w_map:
        assert abs(sum(w_map.values()) - 1.0) < 1e-9
    per_asset_signal_counts = {}
    for t in sorted(details["per_asset"]):
        d = details["per_asset"][t]
        included = d.get("included", False)
        per_asset_signal_counts[t] = d.get("signal_counts", {})
        rows.append(
            {
                "ticker": t,
                "included": included,
                "asset_weight": w_map.get(t),
                "score": d.get("score"),
                "trades": d.get("trades", 0),
                "sortino": d.get("sortino"),
                "profit_factor_capped": d.get("profit_factor_capped"),
                "max_drawdown": d.get("max_drawdown"),
                "per_asset_min_trades": settings.get("per_asset_min_trades", 1),
                "reason": d.get("reason", ""),
                "reason_detail": d.get("reason_detail", ""),
                "reason_trace": d.get("reason_trace", ""),
            }
        )
    counts_df = _build_per_asset_counts(per_asset_signal_counts, rules)
    if rows:
        df = pd.DataFrame(rows)
        num_cols = [
            "asset_weight",
            "score",
            "trades",
            "sortino",
            "profit_factor_capped",
            "max_drawdown",
            "per_asset_min_trades",
        ]
        df = _coerce_numeric_cols(df, num_cols)
        df[num_cols] = df[num_cols].fillna(0.0)
        df = df.sort_values(
            by=["score"],
            key=lambda s: pd.to_numeric(s, errors="coerce").fillna(-np.inf),
            ascending=False,
        )
        fname = RUN_DIR / f"multi_asset_stats_{config.TIMEFRAME}_{end_str}.csv"
        save_csv = charts_cfg.get("save_csv", True)
        if save_csv:
            df.to_csv(fname, index=False)
            print(f"Saved per-asset stats: {fname}")
        counts_fname = RUN_DIR / f"per_asset_counts_{config.TIMEFRAME}_{end_str}.csv"
        if save_csv:
            counts_df.to_csv(counts_fname, index=False)
            print(f"Saved per-asset counts: {counts_fname}")
        exit_params_meta = (
            details.get("exit_params") if isinstance(details, dict) else None
        )
        exit_breakdown_meta = (
            details.get("exit_reason_breakdown") if isinstance(details, dict) else None
        )
        if isinstance(exit_breakdown_meta, Mapping):
            _write_exit_reason_breakdown_csv(exit_breakdown_meta, artifacts)

        summary = {
            "F": F,
            "mu": mu,
            "sigma": sigma,
            "lambda_sigma": lam_sigma,
            "lambda_dispersion": lam,
            "total_trades": total_trades,
            "assets_included": assets_incl,
            "assets_traded": assets_traded,
            "assets_ignored": details.get("assets_ignored"),
            "total_assets": total_assets,
            "coverage_penalty": cov_pen,
            "min_total_trades": settings.get("min_total_trades"),
            "start_date": config.VALIDATION_PERIOD["start"],
            "end_date": config.VALIDATION_PERIOD["end"],
            "asset_weights": w_map,
            "seed": config.SEED,
            "ga_seed": os.environ.get("GA_SEED"),
            "commit_hash": _get_commit_hash(),
            "combination_logic": combo,
            "vote_threshold": vt,
            "per_asset_signal_counts": per_asset_signal_counts,
            "metric_sources": details.get("metric_sources"),
        }
        if exit_params_meta:
            summary["exit_params"] = exit_params_meta
        if exit_breakdown_meta:
            summary["exit_reason_breakdown"] = exit_breakdown_meta
        jf = RUN_DIR / f"multi_asset_summary_{config.TIMEFRAME}_{end_str}.json"
        summary["run_metadata_file"] = "run_metadata.json"
        with open(jf, "w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"Saved run summary: {jf}")
        written = [str(jf)]
        if save_csv:
            written.extend([str(fname), str(counts_fname)])
        artifacts.extend(written)
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
    if charts_cfg.get("save_pngs"):
        png = RUN_DIR / f"multi_asset_overview_{config.TIMEFRAME}_{end_str}.png"
        artifacts.append(str(png))

    extra = {
        "combination_logic": combo,
        "vote_threshold": vt,
        "per_asset_signal_counts": per_asset_signal_counts,
    }
    if exit_params_meta or exit_breakdown_meta:
        extra["exit"] = {
            "params": exit_params_meta or {},
            "reason_breakdown": exit_breakdown_meta or {},
        }
    _write_run_metadata(start_time, artifacts, extra)


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
) -> None:
    """Render multi-asset overview charts with KPI strip."""

    plt.ion()
    tickers = sorted(scores.keys())
    asset_weights = settings.get("asset_weights") or {}
    weights = np.array([asset_weights.get(t, 1.0) for t in tickers], dtype=float)
    weight_sum = weights.sum()
    weights = (
        weights / weight_sum if weight_sum else np.ones_like(weights) / len(weights)
    )

    combined_equity = pd.Series(dtype=float)
    for w, ticker in zip(weights, tickers):
        eq = equities.get(ticker)
        if eq is None or len(eq) == 0:
            continue
        eq_norm = eq / eq.iloc[0]
        combined_equity = (
            eq_norm * w if combined_equity.empty else combined_equity + eq_norm * w
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
        f"F={F:.2f} | μ={mu:.2f} | σ={sigma:.2f} | λ={lam:.2f} | λσ={lam_sigma:.2f} | "
        f"Trades={total_trades} | Assets included={assets_included}, traded={assets_traded} / total={total_assets} | "
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
    ax3.bar(tick_sorted, trades_sorted, label="Trades")
    if floor:
        scaled_floor = floor / max(1, total_assets)
        if scaled_floor >= 1:
            ax3.axhline(
                scaled_floor,
                color="gray",
                linestyle="--",
                label=f"floor/N={scaled_floor:.1f} trades",
            )
        else:
            ax3.plot([], [], color="none", label=f"Group floor: {floor} total trades")
    ax3.set_title("Trades per asset")
    handles, labels = ax3.get_legend_handles_labels()
    if handles:
        ax3.legend(handles, labels)

    fig.tight_layout()
    end_dt = pd.to_datetime(end_date)
    end_str = end_dt.strftime("%Y-%m-%d")
    if charts_cfg.get("save_pngs"):
        fname = RUN_DIR / f"multi_asset_overview_{config.TIMEFRAME}_{end_str}.png"
        fig.savefig(fname, dpi=144, bbox_inches="tight")
    else:
        fig.show()
    plt.close(fig)
