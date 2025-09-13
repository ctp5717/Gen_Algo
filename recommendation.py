from __future__ import annotations

"""Strategy Recommendation Engine."""

import math
import statistics
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
from pydantic import ValidationError

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
    valid_included = [r for r in per_asset.rows if r.included]
    all_tickers = {r.ticker for r in valid_included}
    groups: Dict[str, List[PerAssetRow]] = {}
    for r in valid_included:
        if (
            r.trades < min_trades
            or r.score is None
            or (isinstance(r.score, float) and np.isnan(r.score))
        ):
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
    cov_raw: Dict[str, float] = {}
    for gene, vals in values.items():
        if len(set(vals)) <= 1:
            continue
        mean = float(np.mean(vals))
        std = float(np.std(vals))
        if std == 0:
            continue
        raw = float(std / abs(mean)) if mean else float("inf")
        cov_raw[gene] = raw
        cov[gene] = round(raw, 2)
    threshold = config.RECOMMENDATION["PARAM_COV_UNSTABLE"]
    watch_low = config.RECOMMENDATION["PARAM_COV_WATCHLIST"]

    def sort_key(g: str) -> tuple[float, str]:
        return (-cov_raw[g], g)

    unstable = sorted([g for g, c in cov_raw.items() if c >= threshold], key=sort_key)
    watchlist = sorted(
        [g for g, c in cov_raw.items() if watch_low <= c < threshold],
        key=sort_key,
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
    def _fmt_cov(x: float) -> str:
        return "∞" if not math.isfinite(x) else f"{x:.2f}"

    conf = payload["confidence"]
    lines = ["# Strategy Recommendation Report", ""]
    lines.append(f"## Overall Confidence\n{conf['category']} ({conf['score']})\n")
    lines.append(payload["narrative"]["overall"])
    lines.append("")
    scores = conf["scores"]
    lines.append("### Confidence Factors")
    lines.append("| Factor | Score |")
    lines.append("|---|---|")
    lines.append(f"| Median Fitness | {scores['median']:.1f} |")
    lines.append(f"| Consistency | {scores['consistency']:.1f} |")
    lines.append(f"| Tail (worst fold) | {scores['tail']:.1f} |")
    lines.append(f"| Downside Deviation | {scores['downside']:.1f} |")
    lines.append("")
    lines.append("## Asset Summary")
    lines.append(payload["narrative"]["assets"])
    lines.append("")
    lines.append("## Parameter Summary")
    lines.append(payload["narrative"]["params"])
    lines.append("")
    lines.append("## Asset Performance Matrix")
    lines.append("| Ticker | Performance | Consistency | Class | Samples |")
    lines.append("|---|---|---|---|---|")
    order = {
        "Stars": 0,
        "Stalwarts": 1,
        "Gambles": 2,
        "Borderline": 3,
        "Drags": 4,
        "Insufficient Data": 5,
    }
    for t, a in sorted(
        payload["assets"].items(),
        key=lambda kv: (
            order.get(kv[1]["class"], 99),
            -kv[1]["performance"],
            -kv[1]["consistency"],
            kv[0],
        ),
    ):
        lines.append(
            f"| {t} | {a['performance']:.2f} | {a['consistency']:.1f}% | "
            f"{a['class']} | {a['samples']} |"
        )
    lines.append("")
    th = config.RECOMMENDATION["ASSET_CLASS_THRESHOLDS"]
    legend = (
        "Legend: "
        f"Stars ≥{th['star']['performance']} perf & ≥{th['star']['consistency']}% consistency; "
        f"Stalwarts {th['stalwart']['performance_low']}–{th['stalwart']['performance_high']} perf "
        f"& ≥{th['stalwart']['consistency']}% consistency; "
        f"Gambles ≥{th['gamble']['performance']} perf & "
        f"<{th['gamble']['consistency']}% consistency; "
        f"Drags <{th['drag']['performance']} perf & <{th['drag']['consistency']}% consistency"
    )
    lines.append(legend)
    lines.append("")
    lines.append("## Parameter Stability")
    unstable = payload["param_stability"]["unstable_genes"]
    if unstable:
        for g in unstable:
            c = payload["param_stability"]["cov_by_gene"][g]
            lines.append(f"- {g}: CoV {_fmt_cov(c)}")
    else:
        lines.append("No unstable parameters detected.")
    watchlist = payload["param_stability"].get("watchlist_genes", [])
    if watchlist:
        lines.append("### Watchlist")
        for g in watchlist:
            c = payload["param_stability"]["cov_by_gene"][g]
            lines.append(f"- {g}: CoV {_fmt_cov(c)}")
    cfg = config.RECOMMENDATION
    cuts = cfg["CATEGORY_CUTOFFS"]
    weights = cfg["WEIGHTS"]
    lines.append("")
    lines.append("## SRE Config")
    lines.append("### Category Cutoffs")
    lines.append(f"- High: ≥{cuts['high']}")
    lines.append(f"- Medium: ≥{cuts['medium']}")
    lines.append("### Weights")
    for k in ["median", "consistency", "tail", "downside"]:
        lines.append(f"- {k}: {weights[k]}")
    lines.append("### Asset Class Thresholds")
    lines.append(
        f"- Stars: ≥{th['star']['performance']} perf & ≥{th['star']['consistency']}% consistency"
    )
    lines.append(
        f"- Stalwarts: {th['stalwart']['performance_low']}–"
        f"{th['stalwart']['performance_high']} perf & ≥"
        f"{th['stalwart']['consistency']}% consistency"
    )
    lines.append(
        f"- Gambles: ≥{th['gamble']['performance']} perf & <"
        f"{th['gamble']['consistency']}% consistency"
    )
    lines.append(
        f"- Drags: <{th['drag']['performance']} perf & <"
        f"{th['drag']['consistency']}% consistency"
    )
    path.write_text("\n".join(lines))


def _schema_error(run_dir: Path, msg: str, details: str) -> Dict[str, object]:
    error_payload: Dict[str, object] = {
        "error": "schema_validation_failed",
        "message": msg,
        "schema_version": "1.0",
    }
    merge_run_metadata(
        run_dir / "run_metadata.json",
        {"recommendation": error_payload, "artifacts": ["strategy_recommendation.md"]},
    )
    error_md = run_dir / "strategy_recommendation.md"
    error_md.write_text(
        "# Strategy Recommendation Report\n\n"
        + "Schema validation failed.\n\n"
        + details
        + "\n"
    )
    print(f"Recommendation generation failed: {msg}")
    return error_payload


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
    except ValidationError as e:  # pragma: no cover - exercised in integration
        details = "\n".join(
            f"- summary.{'.'.join(str(x) for x in err['loc'])}: {err['msg']}"
            for err in e.errors()
        )
        return _schema_error(run_dir, "schema validation failed (summary)", details)
    except Exception as e:  # pragma: no cover - exercised in integration
        msg = f"schema validation failed (summary): {e}"
        return _schema_error(run_dir, msg, msg)
    try:
        per_asset = load_wf_per_asset(per_asset_path)
    except ValidationError as e:  # pragma: no cover - exercised in integration
        details = "\n".join(
            f"- per_asset.{'.'.join(str(x) for x in err['loc'])}: {err['msg']}"
            for err in e.errors()
        )
        return _schema_error(run_dir, "schema validation failed (per_asset)", details)
    except Exception as e:  # pragma: no cover - exercised in integration
        msg = f"schema validation failed (per_asset): {e}"
        return _schema_error(run_dir, msg, msg)

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

    merge_run_metadata(
        run_dir / "run_metadata.json",
        {"recommendation": payload, "artifacts": ["strategy_recommendation.md"]},
    )
    _write_markdown(run_dir / "strategy_recommendation.md", payload)
    print(
        f"Recommendation: {conf['category']} ({conf['score']}) - "
        "see strategy_recommendation.md for details."
    )
    return payload
