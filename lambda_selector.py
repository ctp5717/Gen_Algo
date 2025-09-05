"""Lambda dispersion selection via validation-μ with elbow refinement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class LambdaSweepRow:
    """Container for a single evaluation result.

    Each row represents metrics for a specific combination of ``lambda`` value,
    fold and seed. Only the subset of fields required by the selection logic are
    modelled which keeps the interface small and testing friendly.
    """

    lambda_value: float
    mu_val: float
    sigma_val: float
    mu_tr: float
    sigma_tr: float
    F_tr: float
    coverage: float
    fold: int = 0
    seed: int = 0

    def to_dict(self) -> dict:
        """Return a dictionary representation for DataFrame creation."""

        return {
            "lambda": self.lambda_value,
            "mu_val": self.mu_val,
            "sigma_val": self.sigma_val,
            "mu_tr": self.mu_tr,
            "sigma_tr": self.sigma_tr,
            "F_tr": self.F_tr,
            "coverage": self.coverage,
            "fold": self.fold,
            "seed": self.seed,
        }


def select_lambda_with_elbow(
    rows: Iterable[LambdaSweepRow] | Iterable[dict],
    *,
    shortlist_size: int = 3,
    sigma_pct_threshold: float = 0.75,
    coverage_min: float | None = None,
) -> Tuple[float, pd.DataFrame, pd.DataFrame]:
    """Select the dispersion penalty using validation-μ and elbow refinement.

    Parameters
    ----------
    rows:
        Iterable of :class:`LambdaSweepRow` or mapping objects containing the
        required metrics. Each element corresponds to one evaluation for a
        particular ``lambda`` value, fold and seed.
    shortlist_size:
        Number of top candidates (sorted by validation μ) to consider when
        applying the elbow refinement. Defaults to ``3`` as recommended in the
        feature specification.
    sigma_pct_threshold:
        Percentile threshold used by the optional dispersion screen. Lambda
        values with validation σ above this percentile are discarded unless all
        candidates would be removed. Defaults to ``0.75`` (75th percentile).
    coverage_min:
        Optional lower bound for the average coverage metric. Candidates below
        this threshold are dropped during the sanity screen.

    Returns
    -------
    (selected_lambda, summary_df, shortlist_df)
        ``selected_lambda`` is the chosen dispersion penalty. ``summary_df`` is
        a DataFrame containing the aggregated metrics per lambda value.
        ``shortlist_df`` is the subset of ``summary_df`` used for the elbow
        calculation and includes the ``elbow_dist`` column.
    """

    records: List[dict] = []
    for row in rows:
        if isinstance(row, LambdaSweepRow):
            records.append(row.to_dict())
        else:
            records.append(dict(row))

    if not records:
        raise ValueError("No lambda sweep results provided")

    raw_df = pd.DataFrame.from_records(records)

    grouped = raw_df.groupby("lambda")
    summary = grouped.agg(
        mu_val_mean=("mu_val", "mean"),
        mu_val_std=("mu_val", "std"),
        sigma_val_mean=("sigma_val", "mean"),
        sigma_val_std=("sigma_val", "std"),
        mu_train_mean=("mu_tr", "mean"),
        F_train_mean=("F_tr", "mean"),
        coverage_mean=("coverage", "mean"),
    )
    summary["gap"] = summary["mu_train_mean"] - summary["mu_val_mean"]
    summary = summary.reset_index().rename(columns={"lambda": "lambda"})

    sigma_cut = summary["sigma_val_mean"].quantile(sigma_pct_threshold)
    screened = summary[summary["sigma_val_mean"] <= sigma_cut]
    if coverage_min is not None:
        screened = screened[screened["coverage_mean"] >= coverage_min]
    if screened.empty:
        screened = summary.copy()

    screened = screened.sort_values("mu_val_mean", ascending=False)

    if len(screened) >= 3:
        shortlist = screened.head(shortlist_size)
    else:
        shortlist = screened.head(min(2, len(screened)))

    if len(shortlist) == 1:
        chosen = shortlist.iloc[0]
        shortlist = shortlist.assign(elbow_dist=0.0)
        return float(chosen["lambda"]), summary, shortlist

    pts = shortlist[["sigma_val_mean", "mu_val_mean"]].to_numpy(dtype=float)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    span = np.clip(maxs - mins, 1e-12, None)
    norm = (pts - mins) / span

    idx_A = norm[:, 0].argmin()  # lowest σ
    idx_B = norm[:, 1].argmax()  # highest μ
    A = norm[idx_A]
    B = norm[idx_B]

    def perp_dist(P: np.ndarray, A: np.ndarray, B: np.ndarray) -> float:
        AB = B - A
        if np.allclose(AB, 0.0):
            return 0.0
        return float(
            np.linalg.norm(np.cross(np.append(AB, 0.0), np.append(P - A, 0.0)))
            / np.linalg.norm(AB)
        )

    dists = [perp_dist(norm[i], A, B) for i in range(len(norm))]
    shortlist = shortlist.assign(elbow_dist=dists)

    shortlist = shortlist.sort_values(
        by=[
            "elbow_dist",
            "mu_val_mean",
            "sigma_val_mean",
            "gap",
            "coverage_mean",
            "lambda",
        ],
        ascending=[False, False, True, True, False, True],
    )
    chosen = shortlist.iloc[0]
    return float(chosen["lambda"]), summary, shortlist
