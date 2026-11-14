"""Headless plots with portable fonts."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def plot_feature_importance(weights, save_path):
    destination = Path(save_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    ordered = weights.assign(magnitude=weights["weight"].abs()).sort_values("magnitude")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh(ordered["feature"], ordered["weight"], color="#28696d")
    ax.set_title("Model feature weights (diagnostic, not causal)")
    ax.set_xlabel(str(weights["kind"].iloc[0]).replace("_", " "))
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)
