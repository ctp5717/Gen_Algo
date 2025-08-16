import random
from datetime import datetime
from typing import Any, Dict, Tuple, Mapping

import pandas as pd

from utils.logging_util import get_logger
from utils.dataframe_util import to_frame


logger = get_logger(__name__)


def gate_entries(
    entries: pd.DataFrame | pd.Series | Mapping,
    exits: pd.DataFrame | pd.Series | Mapping | None,
    max_concurrent: int,
    tie_break_policy: str = "fifo",
    seed: int | None = None,
    scores: pd.DataFrame | pd.Series | Mapping | None = None,
    price_index: pd.Index | None = None,
    *,
    verbose: bool = False,
    collect_collision_histogram: bool = False,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """Gate entry signals across multiple assets with a position cap.

    Parameters
    ----------
    entries : DataFrame or Series or dict-like
        Entry signals. Columns represent assets when using a DataFrame or
        dict-like mapping.
    exits : DataFrame or Series or dict-like, optional
        Exit signals aligned with ``entries``.
    max_concurrent : int
        Maximum number of open positions allowed at once.
    tie_break_policy : {'fifo', 'random', 'score'}
        Policy used when more signals arrive than available slots.
    seed : int, optional
        Seed for the ``random`` policy to ensure reproducibility.
    scores : DataFrame or Series or dict-like, optional
        Numerical scores used when ``tie_break_policy='score'``. Higher scores
        are preferred.
    price_index : Index, optional
        Expected index of ``entries``.  Raises ``ValueError`` if not aligned.

    Returns
    -------
    gated : DataFrame
        Entries after capacity gating.
    open_count : Series
        Number of open positions after processing each timestamp.
    diagnostics : dict
        Summary statistics of the gating process.
    """
    entries = to_frame(entries, "entries").fillna(False).astype(bool)
    if exits is not None:
        exits = to_frame(exits, "exits", entries.index).fillna(False).astype(bool)
    else:
        exits = pd.DataFrame(False, index=entries.index, columns=entries.columns)
    if scores is not None:
        scores = to_frame(scores, "scores", entries.index, fill_value=0.0).fillna(0.0)

    if set(entries.columns) != set(exits.columns):
        if entries.shape[1] == 1 and exits.shape[1] == 1:
            exits.columns = entries.columns
        else:
            raise ValueError("Entries and exits must have matching columns")
    if scores is not None and set(scores.columns) != set(entries.columns):
        if scores.shape[1] == 1 and entries.shape[1] == 1:
            scores.columns = entries.columns
        else:
            raise ValueError("Scores must have same columns as entries")

    if price_index is not None and not entries.index.equals(price_index):
        raise ValueError("Entries index must match price index")

    idx = entries.index
    asset_order = list(entries.columns)
    rng = random.Random(seed)

    gated = pd.DataFrame(False, index=idx, columns=asset_order)
    open_positions = set()
    open_count_list: list[int] = []
    collisions = 0
    rejected = 0
    total_candidates = 0
    per_asset: Dict[str, Dict[str, int]] = {
        a: {"candidates": 0, "accepted": 0, "rejected": 0} for a in asset_order
    }
    collision_hist = {a: 0 for a in asset_order} if collect_collision_histogram else None

    for ts in idx:
        # Close positions first
        for asset in list(open_positions):
            if exits.at[ts, asset]:
                open_positions.remove(asset)

        # Gather new candidate entries that aren't already open
        candidates = [a for a in asset_order if entries.at[ts, a] and a not in open_positions]
        open_slots = max_concurrent - len(open_positions)

        total_candidates += len(candidates)
        for a in candidates:
            per_asset[a]["candidates"] += 1
        accepted: list[str] = []
        if candidates:
            if open_slots <= 0:
                collisions += 1
                rejected += len(candidates)
                if collision_hist is not None:
                    for a in candidates:
                        collision_hist[a] += 1
                for a in candidates:
                    per_asset[a]["rejected"] += 1
            else:
                if len(candidates) > open_slots:
                    collisions += 1
                    if collision_hist is not None:
                        for a in candidates:
                            collision_hist[a] += 1
                    if tie_break_policy == "fifo":
                        ordered = candidates
                    elif tie_break_policy == "random":
                        ordered = candidates[:]
                        rng.shuffle(ordered)
                    elif tie_break_policy == "score":
                        if scores is None:
                            raise ValueError("scores required for score tie-break policy")
                        ordered = sorted(
                            candidates,
                            key=lambda a: (-scores.at[ts, a], asset_order.index(a)),
                        )
                    else:
                        raise ValueError(f"Unknown tie-break policy: {tie_break_policy}")
                    accepted = ordered[:open_slots]
                    rejected_assets = [a for a in candidates if a not in accepted]
                    rejected += len(rejected_assets)
                    for a in rejected_assets:
                        per_asset[a]["rejected"] += 1
                else:
                    accepted = candidates

                for a in accepted:
                    gated.at[ts, a] = True
                    open_positions.add(a)
                    per_asset[a]["accepted"] += 1

        open_count_list.append(len(open_positions))

    open_count = pd.Series(open_count_list, index=idx)
    accepted_total = sum(v["accepted"] for v in per_asset.values())
    total_rejected = rejected
    total = accepted_total + total_rejected
    avg_n_open = float(open_count.mean())
    max_n_open = int(open_count.max()) if len(open_count) else 0
    diagnostics: Dict[str, Any] = {
        "collisions": collisions,
        "total_candidates": total_candidates,
        "accepted": accepted_total,
        "rejected": total_rejected,
        "acceptance_rate": float(accepted_total / total_candidates)
        if total_candidates
        else 0.0,
        "avg_n_open": avg_n_open,
        "max_n_open": max_n_open,
        "per_asset": per_asset,
    }
    if total != total_candidates:
        logger.warning(
            (
                "gate_entries mismatch at %s: total_candidates=%d, accepted=%d, "
                "rejected=%d, diagnostics=%s"
            ),
            datetime.now().isoformat(),
            total_candidates,
            accepted_total,
            total_rejected,
            diagnostics,
        )
    if collision_hist is not None:
        diagnostics["collisions_by_asset"] = collision_hist
    if verbose:
        print("Gating diagnostics:", diagnostics)
    return gated, open_count, diagnostics


def plot_admitted_trade_skew(diagnostics: Dict[str, Any]) -> None:
    """Visualize acceptance ratio per asset using a bar chart."""
    import matplotlib.pyplot as plt

    per_asset = diagnostics.get("per_asset", {})
    assets = list(per_asset.keys())
    candidates = [v.get("candidates", 0) for v in per_asset.values()]
    accepted = [v.get("accepted", 0) for v in per_asset.values()]
    rates = [a / c if c else 0 for a, c in zip(accepted, candidates)]

    plt.figure(figsize=(8, 4))
    plt.bar(assets, rates)
    plt.ylabel("Acceptance Rate")
    plt.title("Admitted-trade Skew by Asset")
    plt.ylim(0, 1)
    plt.show()
