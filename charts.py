import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List


def plot_collisions_histogram(collisions: Dict[str, int]):
    """Return a histogram figure of collisions per asset.

    Parameters
    ----------
    collisions : Dict[str, int]
        Mapping of asset name to number of collisions.
    """
    fig, ax = plt.subplots()
    assets = list(collisions.keys())
    counts = list(collisions.values())
    ax.bar(assets, counts)
    ax.set_xlabel("Asset")
    ax.set_ylabel("Collisions")
    ax.set_title("Collisions by Asset")
    return fig


def plot_per_asset_acceptance_rate(per_asset: Dict[str, Dict[str, int]]):
    """Return a bar chart of acceptance rate per asset.

    Parameters
    ----------
    per_asset : Dict[str, Dict[str, int]]
        Diagnostics containing candidate and accepted counts per asset.
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
    ax.set_title("Per-Asset Acceptance Rate")
    return fig


def plot_mc_dispersion(run_scores: List[float], median: float | None = None):
    """Return a histogram showing dispersion of Monte Carlo run scores."""
    if run_scores is None:
        run_scores = []
    fig, ax = plt.subplots()
    if run_scores:
        ax.hist(run_scores, bins=min(10, len(run_scores)))
    ax.set_xlabel("Fitness Score")
    ax.set_ylabel("Frequency")
    ax.set_title("Monte Carlo Run Score Dispersion")
    if run_scores and median is not None:
        ax.axvline(median, color="red", linestyle="--", label=f"Median {median:.4f}")
        ax.legend()
    return fig
