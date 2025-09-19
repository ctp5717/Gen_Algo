from __future__ import annotations

"""Final Strategy Synthesizer (FSS) implementation."""

import hashlib
import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Union, cast

import config
from run_metadata import merge_run_metadata
from schemas import (
    Fold,
    WalkForwardPerAssetV1,
    WalkForwardSummaryV1,
    load_wf_per_asset,
    load_wf_summary,
)

LOGGER = logging.getLogger(__name__)


_STRICT_ENV_VALUES = {"1", "true", "yes", "on"}


AssetPayload = Dict[str, Dict[str, Union[str, float]]]


def _strict_missing_weight_mode() -> bool:
    """Return True when missing asset weights should raise."""

    value = os.getenv("FSS_STRICT", "")
    return value.strip().lower() in _STRICT_ENV_VALUES


def _coerce_int(value: Any, default: int = 0) -> int:
    """Best-effort conversion of inputs to an integer for configuration values."""

    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return default
        try:
            return int(stripped)
        except ValueError:
            try:
                return int(float(stripped))
            except ValueError:
                return default
    return default


@dataclass
class WeightedFold:
    """Fold with the computed synthesis weight and diagnostics."""

    fold: Fold
    base_weight: float
    decay_factor: float
    weight: float


@dataclass
class ParameterSummary:
    """Aggregated parameter diagnostics for reporting."""

    name: str
    value: object
    stability: str
    rcv: float
    iqr: float
    median: float
    ascii_box: str
    is_numeric: bool
    multi_modal: bool
    median_near_zero: bool
    precision: int


@dataclass
class AssetDerivation:
    """Intermediate values used to derive final portfolio weights."""

    raw_weight: float
    performance: float
    consistency: float
    volatility: float


class FinalStrategyError(RuntimeError):
    """Raised when final strategy synthesis cannot proceed."""


def _compute_file_sha256(path: Path) -> str:
    """Return the SHA-256 digest for a file."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _compose_notes(notes: Iterable[str]) -> str:
    """Normalise a list of note strings for serialization."""

    filtered = [note.strip() for note in notes if note]
    return "\n".join(filtered)


def _describe_weighting_scheme(cfg: Dict[str, object]) -> str:
    """Human-readable explanation of the asset weighting configuration."""

    scheme = str(cfg.get("WEIGHTING_SCHEME", "risk_adjusted"))
    scheme = scheme.strip()
    descriptions = {
        "equal": "equal weights across included assets",
        "proportional": "weights ∝ performance × consistency",
        "risk_adjusted": "weights ∝ (performance / volatility) × consistency",
        "override": "uses ASSET_WEIGHTS_OVERRIDE as provided",
    }
    detail = descriptions.get(scheme, "custom weighting configuration")
    cap = float(cfg.get("MAX_WEIGHT_CAP", 1.0))
    floor = float(cfg.get("MIN_WEIGHT_FLOOR", 0.0))
    shrink = float(cfg.get("SHRINK_TO_EQUAL", 0.0))
    extras: List[str] = []
    if scheme != "override":
        extras.append(f"cap {cap:.2f}")
        extras.append(f"floor {floor:.2f}")
        if shrink > 0:
            extras.append(f"shrink {shrink:.2f}")
    else:
        overrides = cfg.get("ASSET_WEIGHTS_OVERRIDE", {}) or {}
        extras.append(
            f"{len(overrides)} override weight{'s' if len(overrides) != 1 else ''}"
        )
    if extras:
        detail = f"{detail} ({', '.join(extras)})"
    return f"{scheme} — {detail}"


def _ensure_schema_version(current: str, expected: str, source: str) -> None:
    if current != expected:
        raise FinalStrategyError(
            f"Unsupported {source} schema_version {current!r}; expected {expected!r}. "
            "Update final_strategy.py to handle the new schema or downgrade the producer."
        )


def _weighted_pairs(values: Iterable[Tuple[float, float]]) -> List[Tuple[float, float]]:
    pairs = [(float(v), float(w)) for v, w in values if w > 0]
    pairs = [p for p in pairs if math.isfinite(p[0]) and math.isfinite(p[1])]
    return sorted(pairs, key=lambda item: item[0])


def _weighted_quantile(values: List[Tuple[float, float]], q: float) -> float:
    if not values:
        raise ValueError("cannot compute quantile with no data")
    total = sum(w for _, w in values)
    if total <= 0:
        raise ValueError("sum of weights must be positive")
    threshold = total * q
    cumulative = 0.0
    for value, weight in values:
        cumulative += weight
        if cumulative >= threshold:
            return value
    return values[-1][0]


def _weighted_median(values: List[Tuple[float, float]]) -> float:
    return _weighted_quantile(values, 0.5)


def _weighted_iqr(values: List[Tuple[float, float]]) -> float:
    if not values:
        return 0.0
    q1 = _weighted_quantile(values, 0.25)
    q3 = _weighted_quantile(values, 0.75)
    return float(q3 - q1)


def _weighted_std(values: List[Tuple[float, float]]) -> float:
    if not values:
        return 0.0
    total_weight = sum(w for _, w in values)
    if total_weight <= 0:
        return 0.0
    mean = sum(v * w for v, w in values) / total_weight
    variance = sum(((v - mean) ** 2) * w for v, w in values) / total_weight
    return math.sqrt(max(variance, 0.0))


def _ascii_box(values: List[Tuple[float, float]]) -> str:
    if not values:
        return ""
    if len(values) == 1:
        return f"[{values[0][0]:.3f}]"
    q1 = _weighted_quantile(values, 0.25)
    q2 = _weighted_quantile(values, 0.5)
    q3 = _weighted_quantile(values, 0.75)
    vmin = min(v for v, _ in values)
    vmax = max(v for v, _ in values)
    return f"{vmin:>7.3f} ┤{q1:>7.3f} ┼{q2:>7.3f} ┼{q3:>7.3f} ┤ {vmax:>7.3f}"


def _kmeans_two_clusters(
    values: List[Tuple[float, float]], max_iter: int = 32
) -> List[List[int]]:
    raw = [v for v, _ in values]
    if len(set(raw)) < 2:
        return [list(range(len(values))), []]
    centers = [min(raw), max(raw)]
    assignments = [0] * len(values)
    for _ in range(max_iter):
        changed = False
        for idx, (value, _) in enumerate(values):
            distances = [abs(value - centers[0]), abs(value - centers[1])]
            assignment = 0 if distances[0] <= distances[1] else 1
            if assignments[idx] != assignment:
                changed = True
                assignments[idx] = assignment
        if not changed:
            break
        weights = [0.0, 0.0]
        totals = [0.0, 0.0]
        for (value, weight), assignment in zip(values, assignments):
            weights[assignment] += value * weight
            totals[assignment] += weight
        for i in (0, 1):
            if totals[i] > 0:
                centers[i] = weights[i] / totals[i]
    clusters = [[], []]
    for idx, assignment in enumerate(assignments):
        clusters[assignment].append(idx)
    return clusters


def _detect_multimodal(
    values: List[Tuple[float, float]],
    cfg: Dict[str, float],
) -> bool:
    if len(values) < 3:
        return False
    total_weight = sum(w for _, w in values)
    if total_weight <= 0:
        return False
    clusters = _kmeans_two_clusters(values)
    if not clusters[0] or not clusters[1]:
        return False
    weights = []
    medians = []
    for idxs in clusters:
        subset = [values[i] for i in idxs]
        weight = sum(w for _, w in subset)
        weights.append(weight / total_weight)
        medians.append(_weighted_median(_weighted_pairs(subset)))
    min_cluster_weight = cfg["MULTIMODAL_MIN_CLUSTER_WEIGHT"]
    if weights[0] < min_cluster_weight or weights[1] < min_cluster_weight:
        return False
    iqr = _weighted_iqr(values)
    if iqr <= 0:
        return False
    separation = abs(medians[0] - medians[1])
    threshold = cfg["MULTIMODAL_MIN_SEPARATION"] * iqr
    return separation >= threshold


def _candidate_folds(summary: WalkForwardSummaryV1) -> Tuple[List[Fold], bool]:
    ordered = sorted(summary.folds, key=lambda fold: fold.fold_id)
    elite = [f for f in ordered if f.champion_status in {"Elite", "Viable"}]
    if elite:
        return elite, False
    if not ordered:
        raise FinalStrategyError(
            "No folds found in walk_forward_summary.json; rerun walk_forward.py before FSS."
        )
    return ordered, True


def _compute_fold_weights(
    folds: List[Fold], cfg: Dict[str, object]
) -> Tuple[List[WeightedFold], Dict[int, float]]:
    if not folds:
        raise FinalStrategyError("No folds available for synthesis")
    folds = sorted(folds, key=lambda fold: fold.fold_id)
    base = [max(f.validation_fitness, 0.0) for f in folds]
    use_recency = bool(cfg.get("USE_RECENCY_WEIGHTING"))
    gamma = float(cfg.get("FOLD_DECAY_RATE", 0.0))
    latest = max(f.fold_id for f in folds)
    weights: List[WeightedFold] = []
    weight_values: List[float] = []
    for fold, base_weight in zip(folds, base):
        t = float(latest - fold.fold_id)
        decay = math.exp(-gamma * t) if use_recency and gamma > 0 else 1.0
        final = base_weight * decay
        weight_values.append(final)
        weights.append(
            WeightedFold(
                fold=fold,
                base_weight=base_weight,
                decay_factor=decay,
                weight=final,
            )
        )
    total = sum(weight_values)
    if total <= 0:
        uniform = 1.0 / len(weights)
        for wf in weights:
            wf.weight = uniform
        LOGGER.info(
            "All fold fitness values were non-positive; defaulting to equal weights."
        )
    else:
        for wf in weights:
            wf.weight = wf.weight / total
    mapping = {wf.fold.fold_id: wf.weight for wf in weights}
    if cfg.get("SHOW_RECENCY_HALFLIFE") and use_recency and gamma > 0:
        half_life = math.log(2) / gamma
        LOGGER.info("Recency weighting enabled; half-life ≈ %.2f folds", half_life)
    LOGGER.info("Fold weights (fold_id | base | decay | final_weight)")
    for wf in weights:
        LOGGER.info(
            "%7d | %6.3f | %5.3f | %12.6f",
            wf.fold.fold_id,
            wf.base_weight,
            wf.decay_factor,
            wf.weight,
        )
    return weights, mapping


def _aggregate_parameters(
    weighted_folds: List[WeightedFold],
    cfg: Dict[str, object],
) -> Tuple[Dict[str, object], Dict[str, ParameterSummary]]:
    values: Dict[str, List[Tuple[float, float]]] = {}
    enums: Dict[str, Dict[object, float]] = {}
    decimals_cfg = cfg.get("PARAM_VALUE_DECIMALS", {})
    if isinstance(decimals_cfg, dict):
        default_precision = int(decimals_cfg.get("default", 3))
    else:
        decimals_cfg = {}
        default_precision = 3
    for wf in weighted_folds:
        weight = wf.weight
        if weight <= 0:
            continue
        for name, raw_value in wf.fold.params.items():
            if raw_value is None:
                continue
            if isinstance(raw_value, (int, float)) and not isinstance(raw_value, bool):
                pair_list = values.setdefault(name, [])
                pair_list.append((float(raw_value), weight))
            else:
                enum_map = enums.setdefault(name, {})
                enum_map[raw_value] = enum_map.get(raw_value, 0.0) + weight
    summaries: Dict[str, ParameterSummary] = {}
    output: Dict[str, object] = {}
    for name, pairs in sorted(values.items()):
        ordered = _weighted_pairs(pairs)
        median = _weighted_median(ordered)
        iqr = _weighted_iqr(ordered)
        abs_median = abs(median)
        if abs_median < 1e-12:
            rcv = float("inf")
            median_near_zero = True
        else:
            rcv = iqr / abs_median
            median_near_zero = False
        status = "Stable"
        watch = cfg["PARAM_RCV_WATCHLIST"]
        unstable = cfg["PARAM_RCV_UNSTABLE"]
        if rcv >= unstable:
            status = "Unstable"
        elif rcv >= watch:
            status = "Watchlist"
        multi_modal = _detect_multimodal(ordered, cfg)
        if multi_modal:
            status = f"{status} | Multi-modal"
        ascii_box = _ascii_box(ordered) if cfg.get("SHOW_PARAM_DISTS", True) else ""
        summaries[name] = ParameterSummary(
            name=name,
            value=None,
            stability=status,
            rcv=rcv,
            iqr=iqr,
            median=median,
            ascii_box=ascii_box,
            is_numeric=True,
            multi_modal=multi_modal,
            median_near_zero=median_near_zero,
            precision=default_precision,
        )
        raw_values = [pair[0] for pair in ordered]
        is_int = all(float(v).is_integer() for v in raw_values)
        value = median
        precision = default_precision
        if isinstance(decimals_cfg, dict):
            precision = int(decimals_cfg.get(name, default_precision))
        if is_int:
            value = int(round(value))
            precision = 0
        else:
            value = round(value, precision)
        output[name] = value
        summaries[name].value = value
        summaries[name].precision = precision
    for name, enum_weights in enums.items():
        total = sum(enum_weights.values())
        if total <= 0:
            continue
        best_value = max(
            enum_weights.items(), key=lambda item: (item[1], str(item[0]))
        )[0]
        output[name] = best_value
        summaries[name] = ParameterSummary(
            name=name,
            value=best_value,
            stability="Consensus",
            rcv=0.0,
            iqr=0.0,
            median=0.0,
            ascii_box="",
            is_numeric=False,
            multi_modal=False,
            median_near_zero=False,
            precision=0,
        )
    return output, summaries


def _collect_asset_scores(
    per_asset: WalkForwardPerAssetV1,
    candidate_ids: Iterable[int],
) -> Dict[str, List[Tuple[int, float]]]:
    candidate_set = set(candidate_ids)
    scores: Dict[str, List[Tuple[int, float]]] = {}
    for row in per_asset.rows:
        if not row.included or row.score is None:
            continue
        if row.fold not in candidate_set:
            continue
        if isinstance(row.score, float) and not math.isfinite(row.score):
            continue
        scores.setdefault(row.ticker, []).append((row.fold, float(row.score)))
    return scores


def _bounded_simplex_projection(
    weights: List[float],
    lower: float,
    upper: float,
) -> List[float]:
    n = len(weights)
    if n == 0:
        return []
    upper = min(upper, 1.0)
    lower = max(lower, 0.0)
    if upper <= lower:
        return [1.0 / n for _ in range(n)]
    base = [w - lower for w in weights]
    target = 1.0 - n * lower
    max_sum = n * (upper - lower)
    if target < 0:
        return [1.0 / n for _ in range(n)]
    if target > max_sum:
        target = max_sum
    tau_low = min(base) - (upper - lower)
    tau_high = max(base)
    shifted = base[:]
    for _ in range(60):
        tau = (tau_low + tau_high) / 2.0
        clipped = [min(max(v - tau, 0.0), upper - lower) for v in base]
        total = sum(clipped)
        if abs(total - target) <= 1e-12:
            shifted = clipped
            break
        if total > target:
            tau_low = tau
        else:
            tau_high = tau
        shifted = clipped
    res = [x + lower for x in shifted]
    total = sum(res)
    if total <= 0:
        return [1.0 / n for _ in range(n)]
    res = [x / total for x in res]
    return res


def _compute_asset_allocation(
    sre_assets: Dict[str, Dict[str, Any]],
    per_asset: WalkForwardPerAssetV1,
    fold_weights: Dict[int, float],
    cfg: Dict[str, object],
) -> Tuple[AssetPayload, Dict[str, AssetDerivation], List[str], List[str]]:
    include_raw = cfg.get("INCLUDE_CLASSES", [])
    include_classes = {
        str(cls).strip().lower()
        for cls in include_raw
        if isinstance(cls, str) and cls.strip()
    }
    min_consistency = float(cfg.get("MIN_ASSET_CONSISTENCY", 0.0))
    selected: AssetPayload = {}
    derivation: Dict[str, AssetDerivation] = {}
    exclusions: List[str] = []
    allocation_notes: List[str] = []
    asset_scores = _collect_asset_scores(per_asset, fold_weights.keys())
    for ticker, data in sorted(sre_assets.items()):
        cls_raw = data.get("class")
        cls = str(cls_raw).strip() if cls_raw is not None else ""
        consistency = float(data.get("consistency", 0.0))
        if include_classes and cls.lower() not in include_classes:
            exclusions.append(
                f"{ticker}: class={cls or 'Unknown'} not in INCLUDE_CLASSES"
            )
            continue
        if consistency < min_consistency:
            exclusions.append(
                f"{ticker}: consistency {consistency:.1f}% < {min_consistency:.1f}%"
            )
            continue
        selected[ticker] = {
            "class": cls or "Unknown",
            "performance": float(data.get("performance", 0.0)),
            "consistency": consistency,
            "volatility": 0.0,
            "weight": 0.0,
        }
    if not selected:
        return {}, {}, exclusions, allocation_notes
    # Compute volatilities and raw weights
    eps = 1e-9
    raw_weights: Dict[str, float] = {}
    scheme = cfg.get("WEIGHTING_SCHEME", "risk_adjusted")
    for ticker, asset in selected.items():
        pairs = []
        for fold_id, score in asset_scores.get(ticker, []):
            weight = fold_weights.get(fold_id, 0.0)
            if weight > 0:
                pairs.append((score, weight))
        ordered = _weighted_pairs(pairs)
        iqr = _weighted_iqr(ordered)
        if iqr <= 0:
            iqr = _weighted_std(ordered)
        volatility = float(iqr if iqr > 0 else eps)
        selected[ticker]["volatility"] = volatility
        performance = float(asset["performance"])
        consistency = float(asset["consistency"])
        if scheme == "equal":
            raw = 1.0
        elif scheme == "proportional":
            raw = performance * consistency
        elif scheme == "risk_adjusted":
            raw = (performance / max(volatility, eps)) * consistency
        else:
            raw = 0.0
        raw_weights[ticker] = max(raw, 0.0)
        derivation[ticker] = AssetDerivation(
            raw_weight=raw_weights[ticker],
            performance=performance,
            consistency=consistency,
            volatility=volatility,
        )
    weights: Dict[str, float] = {}
    n_assets = len(selected)
    orig_lower = float(cfg.get("MIN_WEIGHT_FLOOR", 0.0))
    orig_upper = float(cfg.get("MAX_WEIGHT_CAP", 1.0))
    effective_lower = min(orig_lower, 1.0 / n_assets)
    if effective_lower < orig_lower:
        allocation_notes.append(
            f"Weight floor relaxed to {effective_lower:.3f} due to portfolio size ({n_assets})."
        )
    effective_upper = max(orig_upper, effective_lower)
    if effective_upper > orig_upper:
        allocation_notes.append(
            f"Weight cap raised to {effective_upper:.3f} to remain feasible with {n_assets} assets."
        )
    if effective_upper * n_assets < 1.0 - 1e-9:
        adjusted_upper = min(1.0, max(effective_upper, 1.0 / n_assets + 1e-6))
        if adjusted_upper > effective_upper:
            allocation_notes.append(
                f"Weight cap further relaxed to {adjusted_upper:.3f} for feasibility."
            )
        effective_upper = adjusted_upper
    if scheme == "override":
        overrides = cfg.get("ASSET_WEIGHTS_OVERRIDE", {})
        keys = set(overrides.keys())
        if keys != set(selected.keys()):
            missing = ", ".join(sorted(set(selected.keys()) - keys))
            extra = ", ".join(sorted(keys - set(selected.keys())))
            msg = "override weights must match included assets"
            if missing:
                msg += f"; missing: {missing}"
            if extra:
                msg += f"; extras: {extra}"
            raise FinalStrategyError(msg)
        weights = {k: float(overrides[k]) for k in selected.keys()}
    else:
        if effective_upper * n_assets < 1.0 - 1e-9:
            allocation_notes.append(
                "Caps and floors infeasible; reverting to equal weights for included assets."
            )
            equal = 1.0 / n_assets
            for ticker in selected:
                weights[ticker] = equal
        else:
            totals = sum(raw_weights.values())
            if totals <= 0:
                base = [1.0 for _ in selected]
            else:
                base = [raw_weights[t] for t in selected]
            projected = _bounded_simplex_projection(
                base, effective_lower, effective_upper
            )
            for ticker, weight in zip(selected.keys(), projected):
                weights[ticker] = weight
        shrink = float(cfg.get("SHRINK_TO_EQUAL", 0.0))
        if shrink:
            n = len(weights)
            equal = 1.0 / n
            for ticker in weights:
                weights[ticker] = (1 - shrink) * weights[ticker] + shrink * equal
            total = sum(weights.values())
            if total > 0:
                for ticker in weights:
                    weights[ticker] /= total
    for ticker, asset in selected.items():
        asset["weight"] = float(weights.get(ticker, 0.0))
    total_weight = sum(float(asset["weight"]) for asset in selected.values())
    if abs(total_weight - 1.0) > 1e-9:
        raise FinalStrategyError("Asset weights must sum to 1.0 within tolerance")
    return selected, derivation, exclusions, allocation_notes


def _format_asset_derivation(derivation: Dict[str, AssetDerivation]) -> str:
    if not derivation:
        return ""
    lines = [
        "| Ticker | Raw Weight | Performance | Consistency | Volatility |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for ticker, detail in derivation.items():
        lines.append(
            f"| {ticker} | {detail.raw_weight:.6f} | {detail.performance:.3f} | "
            f"{detail.consistency:.1f}% | {detail.volatility:.4f} |"
        )
    return "\n".join(lines)


def _notes_from_summaries(
    summaries: Dict[str, ParameterSummary],
) -> List[str]:
    notes: List[str] = []
    unstable = [p.name for p in summaries.values() if "Unstable" in p.stability]
    watch = [p.name for p in summaries.values() if p.stability.startswith("Watchlist")]
    multimodal = [p.name for p in summaries.values() if p.multi_modal]
    zero_median = [p.name for p in summaries.values() if p.median_near_zero]
    if unstable:
        notes.append("Unstable parameters: " + ", ".join(sorted(unstable)))
    if watch:
        notes.append("Watchlist parameters: " + ", ".join(sorted(watch)))
    if multimodal:
        notes.append(
            "Multi-modal parameters detected: "
            + ", ".join(sorted(multimodal))
            + ". Expect regime sensitivity; tighten risk or monitoring."
        )
    if zero_median:
        notes.append(
            "Some parameters have median ≈ 0; "
            "treat infinite RCV as unstable unless domain logic enforces zero."
        )
    return notes


def _render_parameters_table(summaries: Dict[str, ParameterSummary]) -> str:
    lines = ["| Gene | Value | Stability | Distribution |", "| --- | --- | --- | --- |"]
    if not summaries:
        lines.append("| _None_ | | | |")
        return "\n".join(lines)
    for name in sorted(summaries):
        summary = summaries[name]
        dist = summary.ascii_box if summary.is_numeric else ""
        value = summary.value
        if isinstance(value, float):
            decimals = max(getattr(summary, "precision", 3), 0)
            value_str = f"{value:.{decimals}f}"
        else:
            value_str = str(value)
        lines.append(f"| {name} | {value_str} | {summary.stability} | {dist} |")
    return "\n".join(lines)


def _detect_missing_asset_weights(assets: AssetPayload) -> List[str]:
    """Ensure assets include a weight entry and warn when missing."""

    missing: List[str] = []
    for ticker, data in assets.items():
        if "weight" not in data:
            LOGGER.warning("Asset %s missing weight; using 0.0", ticker)
            data["weight"] = 0.0
            missing.append(ticker)
            continue
        value = data["weight"]
        try:
            weight = float(value)
        except (TypeError, ValueError):
            LOGGER.warning(
                "Asset %s has non-numeric weight %r; using 0.0", ticker, value
            )
            data["weight"] = 0.0
            missing.append(ticker)
            continue
        if not math.isfinite(weight):
            LOGGER.warning(
                "Asset %s has non-finite weight %r; using 0.0", ticker, value
            )
            data["weight"] = 0.0
            missing.append(ticker)
            continue
        data["weight"] = weight
    if missing and _strict_missing_weight_mode():
        joined = ", ".join(missing)
        raise FinalStrategyError(
            "Missing asset weights detected in strict mode: "
            f"{joined}. Set FSS_STRICT=0 to allow defaults."
        )
    return sorted(missing)


def _render_asset_table(assets: AssetPayload) -> str:
    lines = [
        "| Ticker | Class | Performance | Consistency | Volatility | Weight |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    if not assets:
        lines.append("| _None_ | | | | | |")
        return "\n".join(lines)
    ordered = sorted(
        assets.items(),
        key=lambda item: float(item[1].get("weight", 0.0)),
        reverse=True,
    )
    total_weight = 0.0
    for ticker, data in ordered:
        if "weight" not in data:
            LOGGER.warning("Asset %s missing weight; using 0.0", ticker)
            weight = 0.0
        else:
            weight = float(data["weight"])
        total_weight += weight
        lines.append(
            f"| {ticker} | {data['class']} | {data['performance']:.3f} | "
            f"{data['consistency']:.1f}% | {data['volatility']:.4f} | {weight:.4f} |"
        )
    lines.append(f"| **Total** | | | | | {total_weight:.6f} |")
    lines.append("")
    lines.append(
        "Note: displayed weights are rounded for readability; the internal sum remains exactly 1.0."
    )
    return "\n".join(lines)


def _write_markdown(
    path: Path,
    confidence: Dict[str, Any],
    fold_warning: bool,
    use_recency: bool,
    cfg: Dict[str, object],
    summaries: Dict[str, ParameterSummary],
    assets: AssetPayload,
    derivation: Dict[str, AssetDerivation],
    exclusions: List[str],
    notes: List[str],
    weighting_description: str,
    default_to_uniform: bool,
) -> None:
    overview_lines = [
        f"Confidence: {confidence.get('category', 'Unknown')} ({confidence.get('score', 'N/A')})",
        (
            "Fold selection: Elite/Viable"
            if not fold_warning
            else "Fold selection fallback: all folds"
        ),
        "Recency weighting: enabled" if use_recency else "Recency weighting: disabled",
    ]
    overview_lines.append(f"Weighting scheme: {weighting_description}")
    if default_to_uniform:
        overview_lines.append(
            "Fold weighting note: validation fitness was non-positive; equal weights applied."
        )
        if cfg.get("SHOW_RECENCY_HALFLIFE"):
            gamma = float(cfg.get("FOLD_DECAY_RATE", 0.0))
            if gamma > 0:
                half_life = math.log(2) / gamma
                if use_recency:
                    if default_to_uniform:
                        overview_lines.append(
                            f"Recency half-life: {half_life:.2f} folds "
                            "(base fitness ≤ 0; decay inactive)."
                        )
                    else:
                        overview_lines.append(
                            f"Recency half-life: {half_life:.2f} folds"
                        )
                else:
                    overview_lines.append(
                        f"Recency half-life configured at {half_life:.2f} folds "
                        "(recency weighting disabled)."
                    )
    notes_text = _compose_notes(notes)
    if not notes_text:
        notes_text = "No additional notes."
    config_dump = json.dumps(cfg, indent=2, sort_keys=True)
    body = [
        "# Final Strategy\n",
        "## Overview\n",
        "\n".join(overview_lines) + "\n",
        "## Recommended Parameters\n",
        _render_parameters_table(summaries) + "\n",
        "## Asset Allocation\n",
        _render_asset_table(assets) + "\n",
    ]
    derivation_table = _format_asset_derivation(derivation)
    body.append("### Derivation\n")
    body.append((derivation_table or "_No derivation available._") + "\n")
    body.append("## Excluded Assets\n")
    if exclusions:
        body.extend([f"- {reason}\n" for reason in exclusions])
    else:
        body.append("- None\n")
    body.append("\n")
    body.append("## Confidence & SRE Summary\n")
    body.append(
        f"Inherited confidence: {confidence.get('category', 'Unknown')} "
        f"({confidence.get('score', 'N/A')}).\n"
    )
    body.append(
        "FSS stability classifications use relative coefficient of variation (RCV; IQR/median) "
        "while SRE reports coefficient of variation (CoV), so labels may diverge.\n"
    )
    body.append("## Notes\n" + notes_text + "\n")
    body.append("## Configuration\n")
    body.append("```json\n" + config_dump + "\n```\n")
    path.write_text("".join(body), encoding="utf-8")


def _persist_strategy_artifacts(
    run_dir: Path,
    payload: Dict[str, Any],
    confidence: Dict[str, Any],
    cfg: Dict[str, object],
    fold_warning: bool,
    use_recency: bool,
    summaries: Dict[str, ParameterSummary],
    assets: AssetPayload,
    derivation: Dict[str, AssetDerivation],
    exclusions: List[str],
    notes: List[str],
    weighting_description: str,
    default_to_uniform: bool,
) -> None:
    if payload.get("notes") is None:
        payload["notes"] = ""
    md_path = run_dir / "final_strategy.md"
    _write_markdown(
        md_path,
        confidence,
        fold_warning=fold_warning,
        use_recency=use_recency,
        cfg=cfg,
        summaries=summaries,
        assets=assets,
        derivation=derivation,
        exclusions=exclusions,
        notes=notes,
        weighting_description=weighting_description,
        default_to_uniform=default_to_uniform,
    )
    digest = _compute_file_sha256(md_path)
    merge_run_metadata(
        run_dir / "run_metadata.json",
        {
            "final_strategy": payload,
            "artifacts": ["final_strategy.md"],
            "artifacts_meta": {"final_strategy.md": {"sha256": digest}},
        },
    )
    LOGGER.info(
        "Persisted final strategy markdown at %s (sha256=%s)",
        md_path,
        digest,
    )


def _load_recommendation(meta_path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FinalStrategyError(
            "run_metadata.json not found; run SRE before FSS"
        ) from exc
    except json.JSONDecodeError as exc:
        raise FinalStrategyError("run_metadata.json is not valid JSON") from exc
    recommendation = payload.get("recommendation")
    if not recommendation:
        raise FinalStrategyError("SRE recommendation missing from run_metadata.json")
    schema_version = recommendation.get("schema_version")
    _ensure_schema_version(schema_version, "1.0", "run_metadata.recommendation")
    return recommendation


def _jackknife_sensitivity(
    folds: List[Fold],
    cfg: Dict[str, object],
    sre_assets: Dict[str, Dict[str, Any]],
    per_asset: WalkForwardPerAssetV1,
    summaries: Dict[str, ParameterSummary],
    assets: AssetPayload,
) -> List[str]:
    if len(folds) <= 2 or not assets:
        return []
    param_threshold = float(cfg.get("PARAM_SENSITIVITY_THRESHOLD", 0.15))
    weight_threshold = float(cfg.get("WEIGHT_SENSITIVITY_THRESHOLD", 0.05))
    weight_ratio_threshold_raw = cfg.get("WEIGHT_SENSITIVITY_RATIO_THRESHOLD")
    weight_ratio_threshold = (
        float(weight_ratio_threshold_raw)
        if weight_ratio_threshold_raw is not None
        else None
    )
    param_ranges: Dict[str, List[float]] = {
        k: [] for k, v in summaries.items() if v.is_numeric
    }
    weight_ranges: Dict[str, List[float]] = {k: [] for k in assets.keys()}
    for skip in range(len(folds)):
        subset = [f for idx, f in enumerate(folds) if idx != skip]
        if len(subset) < 1:
            continue
        wf, mapping = _compute_fold_weights(subset, cfg)
        params_subset, _ = _aggregate_parameters(wf, cfg)
        assets_subset, _, _, _ = _compute_asset_allocation(
            sre_assets,
            per_asset,
            mapping,
            cfg,
        )
        for name in param_ranges:
            value = params_subset.get(name)
            if isinstance(value, (int, float)):
                param_ranges[name].append(float(value))
        for ticker in weight_ranges:
            weight = assets_subset.get(ticker, {}).get("weight")
            if weight is not None:
                weight_ranges[ticker].append(float(weight))
    notes: List[str] = []
    for name, collected in param_ranges.items():
        if len(collected) < 2:
            continue
        spread = max(collected) - min(collected)
        base_value = abs(float(summaries[name].value))
        if base_value < 1e-12:
            ratio = float("inf")
        else:
            ratio = spread / base_value
        if param_threshold >= 0 and ratio > param_threshold:
            base_display = f"{base_value:.3f}"
            ratio_display = "inf" if not math.isfinite(ratio) else f"{ratio:.3f}"
            notes.append(
                "Parameter "
                f"{name} is sensitive to fold selection: range {spread:.3f} / "
                f"base {base_display} = {ratio_display} > "
                f"{param_threshold:.2f} threshold."
            )
    for ticker, collected in weight_ranges.items():
        if len(collected) < 2:
            continue
        spread = max(collected) - min(collected)
        base_weight = float(assets.get(ticker, {}).get("weight", 0.0))
        if base_weight < 1e-12:
            ratio = float("inf")
        else:
            ratio = spread / base_weight
        ratio_exceeds = True
        if weight_ratio_threshold is not None:
            ratio_exceeds = math.isinf(ratio) or ratio > weight_ratio_threshold
        if weight_threshold >= 0 and spread > weight_threshold and ratio_exceeds:
            ratio_display = "inf" if not math.isfinite(ratio) else f"{ratio:.3f}"
            thresholds = [f"{weight_threshold:.2f} abs threshold"]
            if weight_ratio_threshold is not None:
                thresholds.append(f"{weight_ratio_threshold:.2f} ratio threshold")
            threshold_text = " and ".join(thresholds)
            notes.append(
                "Weight for "
                f"{ticker} varies by range {spread:.3f} "
                f"(base {base_weight:.3f}, ratio {ratio_display}) > "
                f"{threshold_text} under jackknife analysis."
            )
    return notes


def generate_final_strategy(run_context: Dict[str, Any]) -> Dict[str, Any]:
    cfg = config.FINAL_STRATEGY
    config.validate_final_strategy_config(cfg)
    weighting_description = _describe_weighting_scheme(cfg)
    use_recency = bool(cfg.get("USE_RECENCY_WEIGHTING"))
    run_dir = Path(run_context.get("run_dir", "."))
    wf_dir = run_dir / "walk_forward"
    summary_path = wf_dir / "walk_forward_summary.json"
    per_asset_path = wf_dir / "walk_forward_per_asset.csv"
    if not summary_path.exists():
        raise FinalStrategyError(
            "walk_forward_summary.json not found; run walk_forward.py first"
        )
    if not per_asset_path.exists():
        raise FinalStrategyError(
            "walk_forward_per_asset.csv not found; run walk_forward.py first"
        )
    summary = load_wf_summary(summary_path)
    _ensure_schema_version(
        summary.metadata.schema_version, "1.0", "walk_forward_summary.json"
    )
    per_asset, _ = load_wf_per_asset(per_asset_path)
    recommendation = _load_recommendation(run_dir / "run_metadata.json")
    confidence_raw = recommendation.get("confidence", {})
    confidence: Dict[str, Any]
    if isinstance(confidence_raw, dict):
        confidence = cast(Dict[str, Any], confidence_raw)
    else:
        confidence = {}
    score = _coerce_int(confidence.get("score", 0))
    category = str(confidence.get("category", "")).strip()
    min_conf = _coerce_int(cfg.get("MIN_CONFIDENCE_FOR_FINAL", 0))
    gating = score < min_conf or category.lower() == "low"
    if gating:
        note = (
            "Confidence gate blocked strategy publication. "
            f"Score {score} ({category or 'Unknown'}) is below minimum {min_conf}. "
            "Improve the SRE to raise confidence, or lower MIN_CONFIDENCE_FOR_FINAL when a draft "
            "strategy is required for review. Expanding INCLUDE_CLASSES or relaxing "
            "MIN_ASSET_CONSISTENCY can also surface additional evidence before rerunning."
        )
        notes = [note]
        note_text = _compose_notes(notes)
        payload = {
            "parameters": {},
            "assets": {},
            "confidence": confidence,
            "notes": note_text,
            "schema_version": "1.0",
        }
        _persist_strategy_artifacts(
            run_dir,
            payload,
            confidence,
            cfg,
            fold_warning=False,
            use_recency=use_recency,
            summaries={},
            assets={},
            derivation={},
            exclusions=[],
            notes=notes,
            weighting_description=weighting_description,
            default_to_uniform=False,
        )
        return payload
    folds, fallback = _candidate_folds(summary)
    default_to_uniform = all(f.validation_fitness <= 0 for f in folds)
    weighted_folds, fold_weights = _compute_fold_weights(folds, cfg)
    parameters, summaries = _aggregate_parameters(weighted_folds, cfg)
    sre_assets = recommendation.get("assets", {})
    assets, derivation, exclusions, allocation_notes = _compute_asset_allocation(
        sre_assets,
        per_asset,
        fold_weights,
        cfg,
    )
    if not assets:
        note = (
            "No assets satisfied inclusion rules. Expand INCLUDE_CLASSES, "
            "relax MIN_ASSET_CONSISTENCY, or revisit SRE exclusions before "
            "generating the final strategy."
        )
        notes = [note]
        payload = {
            "parameters": parameters,
            "assets": {},
            "confidence": confidence,
            "notes": _compose_notes(notes),
            "schema_version": "1.0",
        }
        _persist_strategy_artifacts(
            run_dir,
            payload,
            confidence,
            cfg,
            fold_warning=fallback,
            use_recency=use_recency,
            summaries=summaries,
            assets={},
            derivation=derivation,
            exclusions=exclusions,
            notes=notes,
            weighting_description=weighting_description,
            default_to_uniform=default_to_uniform,
        )
        return payload
    missing_weights = _detect_missing_asset_weights(assets)
    notes = _notes_from_summaries(summaries)
    notes.extend(allocation_notes)
    if fallback:
        notes.append(
            "No Elite/Viable folds available; synthesised strategy uses all "
            "folds with equal base weights."
        )
    if missing_weights:
        joined = ", ".join(missing_weights)
        notes.append(
            "Missing weights detected for assets: "
            f"{joined}. Defaulted to 0.0 for reporting."
        )
    sensitivity = _jackknife_sensitivity(
        folds, cfg, sre_assets, per_asset, summaries, assets
    )
    notes.extend(sensitivity)
    notes = [n for n in notes if n]
    notes_text = _compose_notes(notes)
    payload = {
        "parameters": parameters,
        "assets": assets,
        "confidence": confidence,
        "notes": notes_text,
        "schema_version": "1.0",
    }
    _persist_strategy_artifacts(
        run_dir,
        payload,
        confidence,
        cfg,
        fold_warning=fallback,
        use_recency=use_recency,
        summaries=summaries,
        assets=assets,
        derivation=derivation,
        exclusions=exclusions,
        notes=notes,
        weighting_description=weighting_description,
        default_to_uniform=default_to_uniform,
    )
    LOGGER.info(
        "Final strategy: %s (%s). See final_strategy.md for details.",
        confidence.get("category", "Unknown"),
        confidence.get("score", "N/A"),
    )
    return payload
