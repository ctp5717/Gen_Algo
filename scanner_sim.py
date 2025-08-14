import random
from typing import Any, Dict, Tuple

import pandas as pd


def gate_entries(
    entries: pd.DataFrame,
    exits: pd.DataFrame,
    max_concurrent: int,
    tie_break_policy: str = "fifo",
    seed: int | None = None,
    scores: pd.DataFrame | None = None,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """Gate entry signals across multiple assets with a position cap.

    Parameters
    ----------
    entries : DataFrame
        Boolean entry signals. Columns represent assets.
    exits : DataFrame
        Boolean exit signals aligned with ``entries``.
    max_concurrent : int
        Maximum number of open positions allowed at once.
    tie_break_policy : {'fifo', 'random', 'score'}
        Policy used when more signals arrive than available slots.
    seed : int, optional
        Seed for the ``random`` policy to ensure reproducibility.
    scores : DataFrame, optional
        Numerical scores used when ``tie_break_policy='score'``. Higher scores
        are preferred.

    Returns
    -------
    gated : DataFrame
        Entries after capacity gating.
    open_count : Series
        Number of open positions after processing each timestamp.
    diagnostics : dict
        Summary statistics of the gating process.
    """
    if set(entries.columns) != set(exits.columns):
        raise ValueError("Entries and exits must have matching columns")
    if scores is not None and set(scores.columns) != set(entries.columns):
        raise ValueError("Scores must have same columns as entries")

    # Ensure indexes are aligned
    idx = entries.index.union(exits.index)
    entries = entries.reindex(idx, fill_value=False)
    exits = exits.reindex(idx, fill_value=False)
    if scores is not None:
        scores = scores.reindex(idx).fillna(0.0)

    asset_order = list(entries.columns)
    rng = random.Random(seed)

    gated = pd.DataFrame(False, index=idx, columns=asset_order)
    open_positions = set()
    open_count_list = []
    collisions = 0
    rejected = 0
    total_candidates = 0
    per_asset: Dict[str, Dict[str, int]] = {
        a: {"accepted": 0, "rejected": 0} for a in asset_order
    }

    for ts in idx:
        # Close positions first
        for asset in list(open_positions):
            if exits.at[ts, asset]:
                open_positions.remove(asset)

        # Gather new candidate entries that aren't already open
        candidates = [a for a in asset_order if entries.at[ts, a] and a not in open_positions]
        open_slots = max_concurrent - len(open_positions)

        total_candidates += len(candidates)
        accepted: list[str] = []
        if candidates:
            if open_slots <= 0:
                collisions += 1
                rejected += len(candidates)
                for a in candidates:
                    per_asset[a]["rejected"] += 1
            else:
                if len(candidates) > open_slots:
                    collisions += 1
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
    assert accepted_total + total_rejected == total_candidates
    diagnostics = {
        "collisions": collisions,
        "total_candidates": total_candidates,
        "accepted": accepted_total,
        "rejected": total_rejected,
        "acceptance_rate": float(accepted_total / total_candidates)
        if total_candidates
        else 0.0,
        "per_asset": per_asset,
    }
    return gated, open_count, diagnostics
