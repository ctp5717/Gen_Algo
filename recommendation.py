from __future__ import annotations

"""Strategy Recommendation Engine."""

import statistics
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

import config
from run_metadata import merge_run_metadata
from schemas import (
    PerAssetRow,
    WalkForwardPerAssetV1,
    load_wf_per_asset,
    load_wf_summary,
)

# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def _compute_confidence(fitness_vals: Iterable[float]) -> Dict[str, object]:
    vals = list(fitness_vals)
    median_fitness = statistics.median(vals)
    positive_fold_pct = 100 * np.mean([v > 0 for v in vals])
    worst_fold_fitness = min(vals)
    downside_vals = [v for v in vals if v < 0]
    downside_deviation = float(np.std(downside_vals)) if downside_vals else 0.0

    cfg = config.RECOMMENDATION
    score_median = max(0.0, min(100.0, 100 * (median_fitness / cfg["MEDIAN_TARGET"])))
    score_consistency = positive_fold_pct
    score_tail = 100.0 - max(
        0.0, min(100.0, 100 * (abs(worst_fold_fitness) / cfg["TAIL_PENALTY_REF"]))
    )
    score_downside = 100.0 - max(
        0.0, min(100.0, 100 * (downside_deviation / cfg["DOWNSIDE_REF"]))
    )

    w = cfg["WEIGHTS"]
    final_score = (
        w["median"] * score_median
        + w["consistency"] * score_consistency
        + w["tail"] * score_tail
        + w["downside"] * score_downside
    )

    cuts = cfg["CATEGORY_CUTOFFS"]
    if final_score >= cuts["high"]:
        category = "High"
    elif final_score >= cuts["medium"]:
        category = "Medium"
    else:
        category = "Low"

    return {
        "score": int(round(final_score)),
        "category": category,
        "factors": {
            "median_fitness": median_fitness,
            "positive_fold_pct": positive_fold_pct,
            "worst_fold_fitness": worst_fold_fitness,
            "downside_deviation": downside_deviation,
        },
        "scores": {
            "median": score_median,
            "consistency": score_consistency,
            "tail": score_tail,
            "downside": score_downside,
        },
    }


def _build_asset_matrix(
    per_asset: WalkForwardPerAssetV1,
) -> Dict[str, Dict[str, object]]:
    cfg = config.RECOMMENDATION
    min_trades = cfg["MIN_TRADES_FOR_SAMPLE"]
    min_samples = cfg["MIN_SAMPLES_FOR_ASSET"]
    thresholds = cfg["ASSET_CLASS_THRESHOLDS"]
    all_tickers = {r.ticker for r in per_asset.rows if r.included}
    groups: Dict[str, List[PerAssetRow]] = {}
    for r in per_asset.rows:
        if not r.included or r.trades < min_trades or r.score is None:
            continue
        groups.setdefault(r.ticker, []).append(r)

    out: Dict[str, Dict[str, object]] = {}
    for ticker in sorted(all_tickers):
        samples = groups.get(ticker, [])
        scores = [r.score for r in samples]
        if len(scores) < min_samples:
            performance = np.median(scores) if scores else 0.0
            consistency = np.mean([s > 0 for s in scores]) * 100 if scores else 0.0
            out[ticker] = {
                "performance": float(performance),
                "consistency": float(consistency),
                "class": "Insufficient Data",
                "samples": len(samples),
            }
            continue
        performance = float(np.median(scores))
        consistency = float(np.mean([s > 0 for s in scores]) * 100)
        if (
            performance >= thresholds["star"]["performance"]
            and consistency >= thresholds["star"]["consistency"]
        ):
            cls = "Stars"
        elif (
            thresholds["stalwart"]["performance_low"]
            <= performance
            < thresholds["stalwart"]["performance_high"]
            and consistency >= thresholds["stalwart"]["consistency"]
        ):
            cls = "Stalwarts"
        elif (
            performance >= thresholds["gamble"]["performance"]
            and consistency < thresholds["gamble"]["consistency"]
        ):
            cls = "Gambles"
        elif (
            performance < thresholds["drag"]["performance"]
            and consistency < thresholds["drag"]["consistency"]
        ):
            cls = "Drags"
        else:
            cls = "Borderline"
        out[ticker] = {
            "performance": performance,
            "consistency": consistency,
            "class": cls,
            "samples": len(samples),
        }
    return out


def _param_stability(
    folds: List,
) -> tuple[Dict[str, float], List[str], List[str]]:
    folds_in = [
        f for f in folds if getattr(f, "champion_status", None) in {"Elite", "Viable"}
    ] or list(folds)
    values: Dict[str, List[float]] = {}
    for f in folds_in:
        for k, v in f.params.items():
            if isinstance(v, (int, float)) and not np.isnan(v):
                values.setdefault(k, []).append(float(v))
    cov: Dict[str, float] = {}
    for gene, vals in values.items():
        mean = float(np.mean(vals))
        std = float(np.std(vals))
        cov[gene] = float(std / abs(mean)) if mean else float("inf")
    threshold = config.RECOMMENDATION["PARAM_COV_UNSTABLE"]
    watch_low = config.RECOMMENDATION["PARAM_COV_WATCHLIST"]
    unstable = sorted(
        [g for g, c in cov.items() if c >= threshold],
        key=lambda g: cov[g],
        reverse=True,
    )
    watchlist = sorted(
        [g for g, c in cov.items() if watch_low <= c < threshold],
        key=lambda g: cov[g],
        reverse=True,
    )
    return cov, unstable, watchlist


def _build_narrative(
    conf: Dict[str, object],
    assets: Dict[str, Dict[str, object]],
    unstable: List[str],
    watchlist: List[str],
) -> Dict[str, str]:
    f = conf["factors"]
    overall = (
        f"Confidence {conf['category']} ({conf['score']}). "
        f"Median fitness {f['median_fitness']:.2f}, "
        f"{f['positive_fold_pct']:.1f}% positive folds; "
        f"worst fold {f['worst_fold_fitness']:.2f} and "
        f"downside deviation {f['downside_deviation']:.2f}."
    )
    stars = [t for t, a in assets.items() if a["class"] == "Stars"]
    stalwarts = [t for t, a in assets.items() if a["class"] == "Stalwarts"]
    drags = [t for t, a in assets.items() if a["class"] == "Drags"]
    insuff = [t for t, a in assets.items() if a["class"] == "Insufficient Data"]
    parts: List[str] = []
    if stars:
        parts.append("Stars: " + ", ".join(stars))
    if stalwarts:
        parts.append("Stalwarts: " + ", ".join(stalwarts))
    if drags:
        parts.append("Drags: " + ", ".join(drags))
    if insuff:
        parts.append("Insufficient Data: " + ", ".join(insuff))
    assets_text = "; ".join(parts) if parts else "No standout assets."
    param_parts: List[str] = []
    if unstable:
        param_parts.append("Unstable parameters: " + ", ".join(unstable))
    if watchlist:
        param_parts.append("Watchlist: " + ", ".join(watchlist))
    params_text = "; ".join(param_parts) if param_parts else "Parameters appear stable."
    return {"overall": overall, "assets": assets_text, "params": params_text}


def _write_markdown(path: Path, payload: Dict[str, object]) -> None:
    conf = payload["confidence"]
    lines = ["# Strategy Recommendation Report", ""]
    lines.append(f"## Overall Confidence\n{conf['category']} ({conf['score']})\n")
    lines.append(payload["narrative"]["overall"])
    lines.append("")
    lines.append("## Asset Performance Matrix")
    lines.append("| Ticker | Performance | Consistency | Class | Samples |")
    lines.append("|---|---|---|---|---|")
    for t, a in sorted(payload["assets"].items()):
        lines.append(
            f"| {t} | {a['performance']:.2f} | {a['consistency']:.1f}% | "
            f"{a['class']} | {a['samples']} |"
        )
    lines.append("")
    th = config.RECOMMENDATION["ASSET_CLASS_THRESHOLDS"]
    lines.append(
        (
            "Legend: Stars ≥{star_p} perf & ≥{star_c}% consistency; "
            "Stalwarts {stal_low}–{stal_high} perf & ≥{stal_c}% consistency; "
            "Gambles ≥{gamble_p} perf & <{gamble_c}% consistency; "
            "Drags <{drag_p} perf & <{drag_c}% consistency"
        ).format(
            star_p=th["star"]["performance"],
            star_c=th["star"]["consistency"],
            stal_low=th["stalwart"]["performance_low"],
            stal_high=th["stalwart"]["performance_high"],
            stal_c=th["stalwart"]["consistency"],
            gamble_p=th["gamble"]["performance"],
            gamble_c=th["gamble"]["consistency"],
            drag_p=th["drag"]["performance"],
            drag_c=th["drag"]["consistency"],
        )
    )
    lines.append("")
    lines.append("## Parameter Stability")
    unstable = payload["param_stability"]["unstable_genes"]
    if unstable:
        for g in unstable:
            c = payload["param_stability"]["cov_by_gene"][g]
            lines.append(f"- {g}: CoV {c:.2f}")
    else:
        lines.append("No unstable parameters detected.")
    watchlist = payload["param_stability"].get("watchlist_genes", [])
    if watchlist:
        lines.append("### Watchlist")
        for g in watchlist:
            c = payload["param_stability"]["cov_by_gene"][g]
            lines.append(f"- {g}: CoV {c:.2f}")
    path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_recommendation(run_context: Dict[str, object]) -> Dict[str, object]:
    """Create recommendation artifacts from walk-forward outputs."""
    run_dir = Path(run_context.get("run_dir", "."))
    wf_dir = run_dir / "walk_forward"
    summary_path = wf_dir / "walk_forward_summary.json"
    per_asset_path = wf_dir / "walk_forward_per_asset.csv"
    try:
        summary = load_wf_summary(summary_path)
        per_asset = load_wf_per_asset(per_asset_path)
    except Exception as e:  # pragma: no cover - exercised in integration
        msg = f"schema validation failed: {e}"
        error_payload: Dict[str, object] = {
            "error": "schema_validation_failed",
            "message": msg,
            "schema_version": "1.0",
        }
        merge_run_metadata(
            run_dir / "run_metadata.json", {"recommendation": error_payload}
        )
        print(f"Recommendation generation failed: {msg}")
        return error_payload

    conf = _compute_confidence([f.validation_fitness for f in summary.folds])
    assets = _build_asset_matrix(per_asset)
    cov, unstable, watchlist = _param_stability(summary.folds)
    narrative = _build_narrative(conf, assets, unstable, watchlist)
    payload: Dict[str, object] = {
        "confidence": conf,
        "assets": assets,
        "param_stability": {
            "cov_by_gene": cov,
            "unstable_genes": unstable,
            "watchlist_genes": watchlist,
        },
        "narrative": narrative,
        "schema_version": "1.0",
    }

    merge_run_metadata(run_dir / "run_metadata.json", {"recommendation": payload})
    _write_markdown(run_dir / "strategy_recommendation.md", payload)
    print(
        f"Recommendation: {conf['category']} ({conf['score']}) - "
        "see strategy_recommendation.md for details."
    )
    return payload
