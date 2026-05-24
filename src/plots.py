from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_valid_day_histograms(valid_days: pd.DataFrame, output_path: Path, ncols: int = 4) -> None:
    """Plot one coverage histogram per signal and save to disk."""

    signals = list(valid_days.columns)
    nrows = math.ceil(len(signals) / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 2.5 * nrows), constrained_layout=True)
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]

    for ax, signal in zip(axes, signals):
        ax.hist(valid_days[signal].dropna(), bins=20, color="#4c78a8", edgecolor="white")
        ax.set_title(signal, fontsize=9)
        ax.set_xlabel("Valid days")
        ax.set_ylabel("Participants")

    for ax in axes[len(signals):]:
        ax.axis("off")

    fig.suptitle("Wearable signal coverage by participant", fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
