from typing import Dict, List

import matplotlib.pyplot as plt

plt.switch_backend("Agg")


def plot_collisions_histogram(
    collisions: Dict[str, int],
    max_concurrent_trades: int | None = None,
    tie_break_policy: str | None = None,
):
    """Return a histogram figure of collisions per asset.

    Parameters
    ----------
    collisions : Dict[str, int]
        Mapping of asset name to number of collisions.
    max_concurrent_trades : int | None, optional
        Maximum concurrent trades ``K`` to include in the title.
    tie_break_policy : str | None, optional
        Tie-break policy to include in the title.
    """
    fig, ax = plt.subplots()
    assets = list(collisions.keys())
    counts = list(collisions.values())
    ax.bar(assets, counts)
    ax.set_xlabel("Asset")
    ax.set_ylabel("Collisions")

    title = "Collisions by Asset"
    subtitle_parts: List[str] = []
    if max_concurrent_trades is not None:
        subtitle_parts.append(f"K={max_concurrent_trades}")
    if tie_break_policy:
        subtitle_parts.append(f"tie_break_policy={tie_break_policy}")
    if subtitle_parts:
        title += "\n" + " | ".join(subtitle_parts)
    ax.set_title(title)
    return fig


def plot_per_asset_acceptance_rate(
    per_asset: Dict[str, Dict[str, int]],
    max_concurrent_trades: int | None = None,
    tie_break_policy: str | None = None,
):
    """Return a bar chart of acceptance rate per asset.

    Parameters
    ----------
    per_asset : Dict[str, Dict[str, int]]
        Diagnostics containing candidate and accepted counts per asset.
    max_concurrent_trades : int | None, optional
        Maximum concurrent trades ``K`` to include in the title.
    tie_break_policy : str | None, optional
        Tie-break policy to include in the title.
    """
    assets = list(per_asset.keys())
    rates = []
    for stats in per_asset.values():
        cand = stats.get("candidates", 0)
        acc = stats.get("accepted", 0)
        rates.append(acc / cand if cand else 0)
    fig, ax = plt.subplots()
    ax.bar(assets, rates)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Asset")
    ax.set_ylabel("Acceptance Rate")

    title = "Per-Asset Acceptance Rate"
    subtitle_parts: List[str] = []
    if max_concurrent_trades is not None:
        subtitle_parts.append(f"K={max_concurrent_trades}")
    if tie_break_policy:
        subtitle_parts.append(f"tie_break_policy={tie_break_policy}")
    if subtitle_parts:
        title += "\n" + " | ".join(subtitle_parts)
    ax.set_title(title)
    return fig


def plot_mc_dispersion(
    run_scores: List[float],
    median: float | None = None,
    max_concurrent_trades: int | None = None,
    tie_break_policy: str | None = None,
):
    """Return a histogram showing dispersion of Monte Carlo run scores."""
    if run_scores is None:
        run_scores = []
    fig, ax = plt.subplots()
    if run_scores:
        ax.hist(run_scores, bins=min(10, len(run_scores)))
    ax.set_xlabel("Fitness Score")
    ax.set_ylabel("Frequency")

    title = "Monte Carlo Run Score Dispersion"
    subtitle_parts: List[str] = []
    if max_concurrent_trades is not None:
        subtitle_parts.append(f"K={max_concurrent_trades}")
    if tie_break_policy:
        subtitle_parts.append(f"tie_break_policy={tie_break_policy}")
    if subtitle_parts:
        title += "\n" + " | ".join(subtitle_parts)
    ax.set_title(title)

    if run_scores and median is not None:
        ax.axvline(
            median, color="red", linestyle="--", label=f"Median {median:.4f}"
        )
        handles, labels = ax.get_legend_handles_labels()
        if handles and any(labels):
            ax.legend()
    return fig
