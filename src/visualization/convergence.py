"""Convergence-history plots for optimization runs."""

from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def plot_convergence(
    history: list,
    title: str = "Optimization Convergence",
    save_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """Plot optimization convergence over generations.

    Args:
        history: Optimization history from pymoo (list of snapshots).
        title: Plot title.
        save_path: Optional path to save plot as PNG.

    Returns:
        Matplotlib figure.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    generations = []
    best_utilization = []
    best_speed = []

    for gen, snapshot in enumerate(history):
        F = snapshot.opt.get("F")
        if F is None or len(F) == 0:
            continue

        generations.append(gen)

        # Track best values (both F[0] and F[1] are negated)
        best_utilization.append(-np.min(F[:, 0]))
        best_speed.append(-np.min(F[:, 1]))

    # Plot utilization convergence
    ax1.plot(generations, best_utilization, "b-", linewidth=2)
    ax1.set_ylabel("Best Utilization", fontsize=11)
    ax1.set_title(title, fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Plot speed convergence
    ax2.plot(generations, best_speed, "g-", linewidth=2)
    ax2.set_xlabel("Generation", fontsize=11)
    ax2.set_ylabel("Best Speed (m/s)", fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, format='png', dpi=150, bbox_inches='tight')

    return fig
