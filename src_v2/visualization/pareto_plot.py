"""Pareto-front scatter plots for multi-objective results."""

from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray


def plot_pareto_front(
    F: NDArray[np.float64],
    G: Optional[NDArray[np.float64]] = None,
    title: str = "Pareto Front",
    save_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """Plot 2D Pareto front with objective values.

    Args:
        F: Objective array of shape (n, 2) with columns [utilization, speed].
        G: Optional constraint array of shape (n, 5) for feasibility coloring.
        title: Plot title.
        save_path: Optional path to save plot as PNG.

    Returns:
        Matplotlib figure with 2D scatter plot.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Flip signs for maximization objectives (both F[0] and F[1] are negated)
    utilization = -F[:, 0]
    avg_speed = -F[:, 1]

    # Determine feasibility for coloring
    if G is not None:
        feasible = np.all(G <= 0, axis=1)
        colors = np.where(feasible, "green", "red")
        labels = ["Feasible", "Infeasible"]
    else:
        colors = "blue"
        labels = ["Solutions"]

    # Create scatter plot
    if isinstance(colors, np.ndarray):
        for color, label in zip(["green", "red"], labels):
            mask = colors == color
            if np.any(mask):
                ax.scatter(
                    utilization[mask],
                    avg_speed[mask],
                    c=color,
                    marker="o",
                    s=50,
                    alpha=0.6,
                    label=label,
                )
    else:
        ax.scatter(utilization, avg_speed, c=colors, marker="o", s=50, alpha=0.6, label=labels[0])

    ax.set_xlabel("Utilization (fraction)", fontsize=11)
    ax.set_ylabel("Avg Speed (m/s)", fontsize=11)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, format='png', dpi=150, bbox_inches='tight')

    return fig
