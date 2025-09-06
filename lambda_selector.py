"""Lambda dispersion selection via validation-μ with elbow refinement."""

from __future__ import annotations

import logging
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
    note: str = ""
    round: int = 1
    solution_hash: str | None = None

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
            "note": self.note,
            "round": self.round,
            "solution_hash": self.solution_hash,
        }


def _dedupe_shortlist(df: pd.DataFrame, tol: float) -> pd.DataFrame:
    tol = max(tol, 1e-12)  # clamp to avoid log10(0)
    ndigits = max(0, int(abs(np.log10(tol))))
    mask = df[["mu_val_mean", "sigma_val_mean"]].round(ndigits).duplicated(keep="first")
    return df.loc[~mask]


logger = logging.getLogger(__name__)


def select_lambda_with_elbow(
    rows: Iterable[LambdaSweepRow] | Iterable[dict],
    *,
    shortlist_size: int = 3,
    sigma_pct_threshold: float = 0.75,
    coverage_min: float | None = None,
    duplicate_tol: float = 1e-6,
    rank_stat: str = "mean",
    soft_sigma_enabled: bool = False,
    soft_sigma_eps_abs: float = 0.02,
    soft_sigma_eps_rel: float = 0.02,
    soft_sigma_tau: float = 0.02,
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
    duplicate_tol:
        Maximum absolute difference in ``μ`` and ``σ`` when de-duplicating the
        shortlist. Defaults to ``1e-6``.
    rank_stat:
        Summary statistic used to aggregate metrics per λ. ``"mean"`` (default)
        or ``"median"``.
    soft_sigma_enabled:
        When ``True`` the selection logic softens tie-breaking on ``μ`` when the
        shortlisted candidates have nearly equal validation-μ. ``mu_span_abs``
        and ``mu_span_rel`` capture the absolute and relative spread of
        validation-μ across the shortlist. If either metric falls below their
        respective ``soft_sigma_eps_*`` thresholds the adjusted metric
        ``mu_adj = mu_val_mean - soft_sigma_tau * sigma_val_mean`` is used in the
        ranking.
    soft_sigma_eps_abs:
        Maximum absolute spread in validation-μ for ``soft_sigma_enabled`` to
        trigger. Defaults to ``0.02``.
    soft_sigma_eps_rel:
        Maximum relative spread in validation-μ for ``soft_sigma_enabled`` to
        trigger. Defaults to ``0.02``.
    soft_sigma_tau:
        Weight applied to σ when computing ``mu_adj``. Defaults to ``0.02``.

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

    # Drop rows with NaN metrics and log the affected λ values
    nan_mask = raw_df[["mu_val", "sigma_val"]].isna().any(axis=1)
    if nan_mask.any():
        nan_groups = raw_df[nan_mask].groupby("lambda")
        for lam, grp in nan_groups:
            count = len(grp)
            logger.warning("NaN metrics for λ=%s; dropped %d rows", lam, count)
            total = (raw_df["lambda"] == lam).sum()
            if count == total:
                logger.warning("All rows NaN for λ=%s; excluded from selection", lam)
        raw_df = raw_df[~nan_mask]
        if raw_df.empty:
            raise ValueError("No lambda sweep results after dropping NaNs")

    grouped = raw_df.groupby("lambda")
    aggfunc = "median" if rank_stat == "median" else "mean"
    summary = grouped.agg(
        mu_val_mean=("mu_val", aggfunc),
        mu_val_std=("mu_val", "std"),
        sigma_val_mean=("sigma_val", aggfunc),
        sigma_val_std=("sigma_val", "std"),
        mu_train_mean=("mu_tr", aggfunc),
        F_train_mean=("F_tr", aggfunc),
        coverage_mean=("coverage", aggfunc),
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

    before = len(shortlist)
    shortlist = _dedupe_shortlist(shortlist, duplicate_tol)
    after = len(shortlist)
    if after < before:
        logger.info("Shortlist de-duplicated: %s→%s", before, after)

    mu_span_abs = float(shortlist["mu_val_mean"].max() - shortlist["mu_val_mean"].min())
    mu_span_rel = float(mu_span_abs / max(shortlist["mu_val_mean"].abs().max(), 1e-12))
    use_soft_sigma = soft_sigma_enabled and (
        mu_span_abs <= soft_sigma_eps_abs or mu_span_rel <= soft_sigma_eps_rel
    )
    if use_soft_sigma:
        shortlist = shortlist.assign(
            mu_adj=shortlist["mu_val_mean"]
            - soft_sigma_tau * shortlist["sigma_val_mean"]
        )

    if len(shortlist) < 3:
        logger.info(
            "Shortlist size %d; skipping elbow and applying tie-breakers.",
            len(shortlist),
        )
        mu_col = "mu_adj" if use_soft_sigma else "mu_val_mean"
        tie_order = [
            (mu_col, False, "mu_adj" if use_soft_sigma else "mu"),
            ("sigma_val_mean", True, "sigma"),
            ("gap", True, "gap"),
            ("coverage_mean", False, "coverage"),
            ("lambda", True, "lambda"),
        ]
        if len(shortlist) > 1:
            vals = shortlist[[col for col, _, _ in tie_order]].to_numpy()
            for i, (_col, _asc, name) in enumerate(tie_order):
                col_vals = vals[:, i]
                if not np.allclose(col_vals, col_vals[0]):
                    logger.info("Tie-breaker on %s", name)
                    break
        shortlist = shortlist.assign(elbow_dist=0.0)
        shortlist = shortlist.sort_values(
            by=[col for col, _, _ in tie_order],
            ascending=[asc for _, asc, _ in tie_order],
        )
        chosen = shortlist.iloc[0]
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

    A_sig = shortlist.iloc[idx_A]["sigma_val_mean"]
    A_mu = shortlist.iloc[idx_A]["mu_val_mean"]
    B_sig = shortlist.iloc[idx_B]["sigma_val_mean"]
    B_mu = shortlist.iloc[idx_B]["mu_val_mean"]
    logger.info(
        "Elbow endpoints: A(σ=%0.6f, μ=%0.6f) B(σ=%0.6f, μ=%0.6f)",
        A_sig,
        A_mu,
        B_sig,
        B_mu,
    )

    def perp_dist(P: np.ndarray, A: np.ndarray, B: np.ndarray) -> float:
        AB = B - A
        if np.allclose(AB, 0.0):
            return 0.0
        return float(
            np.linalg.norm(np.cross(np.append(AB, 0.0), np.append(P - A, 0.0)))
            / np.linalg.norm(AB)
        )

    dists = [perp_dist(norm[i], A, B) for i in range(len(norm))]
    for lam, dist in zip(shortlist["lambda"], dists):
        logger.info("λ=%s elbow_dist=%0.6e", lam, dist)
    shortlist = shortlist.assign(elbow_dist=dists)

    if len(shortlist) > 1 and np.all(np.abs(shortlist["elbow_dist"]) <= 1e-9):
        logger.warning("Shortlist is degenerate; falling back to tie-breakers.")

    sort_cols = [
        "elbow_dist",
        "mu_adj" if use_soft_sigma else "mu_val_mean",
        "sigma_val_mean",
        "gap",
        "coverage_mean",
        "lambda",
    ]
    shortlist = shortlist.sort_values(
        by=sort_cols,
        ascending=[False, False, True, True, False, True],
    )
    chosen = shortlist.iloc[0]
    return float(chosen["lambda"]), summary, shortlist
