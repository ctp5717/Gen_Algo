from __future__ import annotations

"""Strategy Recommendation Engine."""

import json
import math
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from pydantic import ValidationError

import config
from run_metadata import merge_run_metadata
from schemas import (
    PerAssetRow,
    SchemaCsvError,
    WalkForwardPerAssetV1,
    load_wf_per_asset,
    load_wf_summary,
)
from strings import DRAG_STANCE, PARAM_STABILITY_IMPLICATION
from utils.format import fmt_num, fmt_pct

# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


ASSET_CLASS_ORDER = {
    "Stars": 0,
    "Stalwarts": 1,
    "Gambles": 2,
    "Borderline": 3,
    "Drags": 4,
    "Insufficient Data": 5,
}


def _asset_sort_key(kv: Tuple[str, Dict[str, object]]) -> Tuple:
    ticker, a = kv
    cls = str(a.get("class", ""))
    return (
        ASSET_CLASS_ORDER.get(cls, 99),
        -float(a.get("performance") or 0.0),
        -float(a.get("consistency") or 0.0),
        ticker,
    )


def _compute_confidence(fitness_vals: Iterable[float]) -> Dict[str, object]:
    vals = list(fitness_vals)
    median_fitness = statistics.median(vals)
    positive_fold_pct = 100 * np.mean([v > 0 for v in vals])
    worst_fold_fitness = min(vals)
    downside_vals = [v for v in vals if v < 0]
    downside_deviation = (
        float(np.std(downside_vals, ddof=1)) if len(downside_vals) > 1 else 0.0
    )  # sample std (ddof=1) to reduce small-n bias; CoV uses ddof=0 elsewhere

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
        f"Median fitness {fmt_num(f['median_fitness'])}, "
        f"{fmt_pct(f['positive_fold_pct'])} positive folds; "
        f"worst fold {fmt_num(f['worst_fold_fitness'])} and "
        f"downside deviation {fmt_num(f['downside_deviation'])}."
    )

    def _sorted(cls: str) -> List[str]:
        return [
            t
            for t, _ in sorted(
                ((t, a) for t, a in assets.items() if a["class"] == cls),
                key=_asset_sort_key,
            )
        ]

    stars = _sorted("Stars")
    stalwarts = _sorted("Stalwarts")
    drags = _sorted("Drags")
    border = _sorted("Borderline")
    insuff = _sorted("Insufficient Data")

    parts: List[str] = []
    if stars:
        parts.append("Stars: " + ", ".join(stars))
    if stalwarts:
        parts.append("Stalwarts: " + ", ".join(stalwarts))
    if border:
        parts.append("Borderline: " + ", ".join(border))
    if drags:
        parts.append("Drags: " + ", ".join(drags))
    if insuff:
        parts.append("Insufficient Data: " + ", ".join(insuff))
    assets_text = "; ".join(parts) if parts else "No standout assets."
    min_samples = config.RECOMMENDATION["MIN_SAMPLES_FOR_ASSET"]
    if assets and all((a.get("samples", 0) >= min_samples) for a in assets.values()):
        assets_text += f"; All assets have ≥{min_samples} qualifying fold(s)."
    if drags:
        examples = ", ".join(drags[:3])
        assets_text += " " + DRAG_STANCE.format(examples=examples)

    param_parts: List[str] = []
    if unstable:
        param_parts.append("Unstable parameters: " + ", ".join(unstable))
    if watchlist:
        param_parts.append("Watchlist: " + ", ".join(watchlist))
    params_text = "; ".join(param_parts) if param_parts else "Parameters appear stable."
    return {"overall": overall, "assets": assets_text, "params": params_text}


def _write_markdown(path: Path, payload: Dict[str, object]) -> None:
    def _fmt_cov(x: float) -> str:
        return "∞" if not math.isfinite(x) else fmt_num(x)

    conf = payload["confidence"]
    lines = ["# Strategy Recommendation Report", ""]
    lines.append(f"## Overall Confidence\n{conf['category']} ({conf['score']})\n")
    lines.append(payload["narrative"]["overall"])
    lines.append("")
    scores = conf["scores"]
    fcts = conf["factors"]
    lines.append("### Confidence Factors")
    lines.append(
        "Folds: median "
        f"{fmt_num(fcts['median_fitness'])}, "
        f"worst {fmt_num(fcts['worst_fold_fitness'])}, "
        f"positive {fmt_pct(fcts['positive_fold_pct'])}."
    )
    lines.append("| Factor | Score |")
    lines.append("|---|---|")
    lines.append(f"| Median Fitness | {fmt_num(scores['median'], 1)} |")
    lines.append(f"| Consistency | {fmt_num(scores['consistency'], 1)} |")
    lines.append(f"| Tail (worst fold) | {fmt_num(scores['tail'], 1)} |")
    lines.append(f"| Downside Deviation | {fmt_num(scores['downside'], 1)} |")
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
    for t, a in sorted(payload["assets"].items(), key=_asset_sort_key):
        lines.append(
            f"| {t} | {fmt_num(a['performance'])} | {fmt_pct(a['consistency'])} | "
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
    if unstable or watchlist:
        lines.append("")
        lines.append(PARAM_STABILITY_IMPLICATION)
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
    lines.append("")
    lines.append("### Stability Regularizer")
    lines.append(f"- Enabled: {getattr(config, 'ENABLE_STABILITY_REG', False)}")
    lines.append(f"- Alpha: {getattr(config, 'STABILITY_ALPHA', 0.0)}")
    genes = ", ".join(getattr(config, "STABILITY_GENES", [])) or "—"
    lines.append(f"- Genes: {genes}")
    path.write_text("\n".join(lines), encoding="utf-8")


def _schema_error(
    run_dir: Path, msg: str, details: str, diagnostics: str | None = None
) -> Dict[str, object]:
    error_payload: Dict[str, object] = {
        "error": "schema_validation_failed",
        "message": msg,
        "schema_version": "1.0",
    }
    if diagnostics:
        error_payload["diagnostics"] = diagnostics
    merge_run_metadata(
        run_dir / "run_metadata.json",
        {"recommendation": error_payload, "artifacts": ["strategy_recommendation.md"]},
    )
    error_md = run_dir / "strategy_recommendation.md"
    body = (
        "# Strategy Recommendation Report\n\n"
        + "Schema validation failed.\n\n"
        + details
    )
    if diagnostics:
        body += f"\n\n## Diagnostics\n{diagnostics}"
    body += "\n"
    error_md.write_text(body, encoding="utf-8")
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
        per_asset, unknown_cols = load_wf_per_asset(per_asset_path)
    except ValidationError as e:  # pragma: no cover - exercised in integration
        details = "\n".join(
            f"- per_asset.{'.'.join(str(x) for x in err['loc'])}: {err['msg']}"
            for err in e.errors()
        )
        return _schema_error(run_dir, "schema validation failed (per_asset)", details)
    except SchemaCsvError as e:  # pragma: no cover - exercised in integration
        diag = None
        if e.unknown_columns:
            limit = 10
            cols = e.unknown_columns[:limit]
            extra = len(e.unknown_columns) - limit
            diag = ", ".join(cols)
            if extra > 0:
                diag += f" (+{extra} more)"
            diag = f"Unknown columns: {diag}"
        msg = f"schema validation failed (per_asset): {e}"
        return _schema_error(run_dir, msg, msg, diag)
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

    meta_update: Dict[str, object] = {
        "recommendation": payload,
        "artifacts": ["strategy_recommendation.md"],
    }
    if config.RECOMMENDATION.get("LOG_UNKNOWN_COLUMNS_ON_SUCCESS") and unknown_cols:
        diag_entry = {
            "source": "walk_forward_per_asset.csv",
            "unknown_columns": unknown_cols,
        }
        meta_path = run_dir / "run_metadata.json"
        try:
            existing_meta = json.loads(meta_path.read_text())
            diag_list = list(existing_meta.get("diagnostics", []))
        except Exception:
            diag_list = []
        diag_list.append(diag_entry)
        meta_update["diagnostics"] = diag_list

    merge_run_metadata(run_dir / "run_metadata.json", meta_update)
    _write_markdown(run_dir / "strategy_recommendation.md", payload)
    print(
        f"Recommendation: {conf['category']} ({conf['score']}) - "
        "see strategy_recommendation.md for details."
    )
    return payload
