from __future__ import annotations

"""Pydantic v2 schemas for Strategy Recommendation Engine inputs."""

import ast
import csv
import json
from pathlib import Path
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, TypeAdapter, field_validator

import config


class SchemaCsvError(ValueError):
    """Raised when CSV schema validation fails with extra diagnostics."""

    def __init__(self, message: str, unknown_columns: List[str] | None = None):
        super().__init__(message)
        self.unknown_columns = unknown_columns or []


class Fold(BaseModel):
    fold_id: int
    validation_fitness: float
    params: Dict[str, float | int | str | bool | None]
    champion_status: Optional[Literal["Elite", "Viable", "Discarded"]] = None

    @field_validator("champion_status", mode="before")
    @classmethod
    def _validate_champion_status(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        allowed = ("Elite", "Viable", "Discarded")
        if value not in allowed:
            allowed_str = " | ".join(allowed)
            raise ValueError(f"Fold.champion_status must be one of: {allowed_str}")
        return value


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


def load_wf_per_asset(path: str | Path) -> tuple[WalkForwardPerAssetV1, List[str]]:
    """Load and validate ``walk_forward_per_asset.csv``.

    Returns
    -------
    tuple
        Parsed :class:`WalkForwardPerAssetV1` object and a sorted list of
        column names from the CSV that were not recognized by the schema.
    """
    mapping = {
        "Fold": "fold",
        "Ticker": "ticker",
        "Score": "score",
        "Trades": "trades",
        "Included": "included",
    }
    mapping_lowers = {k.lower() for k in mapping.keys()}
    rows: List[Dict[str, object]] = []
    unknown_columns: List[str] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames:
            unknown_columns = sorted(
                [
                    c
                    for c in reader.fieldnames
                    if c.strip().lower() not in mapping_lowers
                ]
            )
        for row in reader:
            norm = {
                mapping.get(
                    k.strip(), mapping.get(k.strip().capitalize(), k.strip())
                ): v
                for k, v in row.items()
            }
            try:
                score = float(norm["score"])
            except (TypeError, ValueError):
                score = None
            try:
                trades = int(norm["trades"])
            except (TypeError, ValueError):
                raise SchemaCsvError("invalid trades value", unknown_columns)
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
        raise SchemaCsvError("no rows present", unknown_columns)
    adapter = TypeAdapter(WalkForwardPerAssetV1)
    try:
        obj = adapter.validate_python({"rows": rows})
    except Exception as e:
        raise SchemaCsvError(str(e), unknown_columns) from e
    return obj, unknown_columns
