from __future__ import annotations

"""Pydantic v2 schemas for Strategy Recommendation Engine inputs."""

import ast
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, TypeAdapter

import config


class Fold(BaseModel):
    fold_id: int
    validation_fitness: float
    params: Dict[str, float]
    champion_status: Optional[str] = None


class Metadata(BaseModel):
    schema_version: str = "1.0"
    num_folds: int
    asset_universe: List[str]


class WalkForwardSummaryV1(BaseModel):
    metadata: Metadata
    folds: List[Fold]


class PerAssetRow(BaseModel):
    fold: int
    ticker: str
    score: Optional[float]
    trades: int
    included: bool


class WalkForwardPerAssetV1(BaseModel):
    rows: List[PerAssetRow]


# Helper readers -----------------------------------------------------------


def load_wf_summary(path: str | Path) -> WalkForwardSummaryV1:
    """Load and validate ``walk_forward_summary.json``."""
    raw = json.loads(Path(path).read_text())
    if "metadata" not in raw:
        folds = raw.get("folds", [])
        meta = {
            "schema_version": "1.0",
            "num_folds": len(folds),
            "asset_universe": raw.get("asset_universe", []),
        }
        raw = {"metadata": meta, "folds": folds}
    folds_raw = raw.get("folds")
    if not folds_raw:
        raise ValueError("no folds present")
    mapped: List[Dict[str, object]] = []
    for f in folds_raw:
        fold_id = f.get("fold_id") if f.get("fold_id") is not None else f.get("Window")
        validation_fitness = f.get("validation_fitness")
        if validation_fitness is None:
            validation_fitness = f.get("Fitness")
        if validation_fitness is None and config.RECOMMENDATION.get(
            "USE_RETURN_AS_FITNESS"
        ):
            validation_fitness = f.get("Total Return [%]")
        params = f.get("params")
        if params is None:
            p = f.get("Params")
            if isinstance(p, str):
                try:
                    params = ast.literal_eval(p)
                except Exception:
                    params = {}
            else:
                params = p or {}
        mapped.append(
            {
                "fold_id": fold_id,
                "validation_fitness": validation_fitness,
                "params": params,
                "champion_status": f.get("champion_status"),
            }
        )
    raw["folds"] = mapped
    return WalkForwardSummaryV1.model_validate(raw)


def load_wf_per_asset(path: str | Path) -> WalkForwardPerAssetV1:
    """Load and validate ``walk_forward_per_asset.csv``."""
    mapping = {
        "Fold": "fold",
        "Ticker": "ticker",
        "Score": "score",
        "Trades": "trades",
        "Included": "included",
    }
    rows: List[Dict[str, object]] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            norm = {mapping.get(k, k): v for k, v in row.items()}
            try:
                score = float(norm["score"])
            except (TypeError, ValueError):
                score = None
            try:
                trades = int(norm["trades"])
            except (TypeError, ValueError):
                raise ValueError("invalid trades value")
            rows.append(
                {
                    "fold": int(norm["fold"]),
                    "ticker": str(norm["ticker"]),
                    "score": score,
                    "trades": trades,
                    "included": str(norm["included"]).lower() in {"true", "1", "yes"},
                }
            )
    if not rows:
        raise ValueError("no rows present")
    adapter = TypeAdapter(WalkForwardPerAssetV1)
    return adapter.validate_python({"rows": rows})
