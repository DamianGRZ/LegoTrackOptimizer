"""Pareto-front scatter plots for multi-objective results."""

from pathlib import Path
from typing import Optional, Union

import matplotlib

# Headless pipeline: PNGs only (Tk + Pool result-handler threads crash Tcl).
# Must be forced BEFORE pymoo.visualization pulls in pyplot.
matplotlib.use("Agg", force=True)
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from numpy.typing import NDArray  # noqa: E402
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting  # noqa: E402
from pymoo.visualization.scatter import Scatter  # noqa: E402

from src.normalization import ideal_nadir, normalize  # noqa: E402


def _plot_scale(F: NDArray, archive_F: Optional[NDArray]) -> tuple[NDArray, NDArray]:
    """Ideal/nadir spanning everything drawn, so no point falls off the axes.

    Degenerate individuals carry the +inf objective sentinel; they are left
    out of the scale (and land outside the view) instead of collapsing it.
    """
    pool = [F[np.isfinite(F).all(axis=1)]]
    if archive_F is not None and len(archive_F) > 0:
        pool.append(np.asarray(archive_F, dtype=float))
    stacked = np.vstack(pool)
    return ideal_nadir(stacked if len(stacked) else np.zeros((1, F.shape[1])))


def _goodness(F: NDArray, ideal: NDArray, nadir: NDArray) -> NDArray:
    """Normalized objectives flipped so that 1 is best on every axis."""
    return 1.0 - normalize(F, ideal, nadir)


def _add_with_multiplicity(plot, pts: NDArray, **style) -> list:
    """Add distinct points to a pymoo plot; return [(xy, count)] for labels.

    A converged terminal population holds hundreds of coincident F vectors,
    which overlapping markers would render as a few lonely dots — distinct
    points are drawn once and duplicates reported for annotation.
    """
    uniq, counts = np.unique(pts.round(6), axis=0, return_counts=True)
    plot.add(uniq, **style)
    return [(xy, int(c)) for xy, c in zip(uniq, counts) if c > 1]


def plot_pareto_front(
    F: NDArray[np.float64],
    G: Optional[NDArray[np.float64]] = None,
    title: str = "Pareto Front",
    save_path: Optional[Union[str, Path]] = None,
    archive_F: Optional[NDArray[np.float64]] = None,
    objective_labels: tuple[str, ...] = ("weighted piece score", "traversal time"),
) -> Figure:
    """Objective-space scatter built on pymoo's ``Scatter`` (maximized view).

    ``Scatter`` picks 2D, 3D or a pairwise matrix from the width of the data.
    Every axis is normalized to the range actually spanned by the plotted
    points and flipped so 1 is best, which keeps objectives on different
    scales comparable at a glance. Raw units are reported by the run log,
    not by these axes.

    Follows pymoo's front-vs-solutions convention: all distinct points are
    drawn with their multiplicity, and the Pareto front is ringed. With
    ``archive_F`` (the run-cumulative feasible front from the monitor) the
    ring marks THAT front — the terminal population is a converged
    monoculture, so the archive is what actually shows the trade-off curve.
    Without it, the ring falls back to the final population's non-dominated
    subset.

    Args:
        F: Raw objective array of shape (n, n_obj), minimized sign.
        G: Optional constraint array for feasibility coloring.
        title: Plot title.
        save_path: Optional path to save the plot as PNG.
        archive_F: Optional (m, n_obj) run-cumulative feasible front, same sign
            as ``F``.
        objective_labels: One axis label per objective.

    Returns:
        Matplotlib figure with the scatter plot.
    """
    F = np.asarray(F, dtype=float)
    planar = F.shape[1] == 2
    feasible = (np.all(G <= 0, axis=1) if G is not None
                else np.ones(len(F), dtype=bool))
    ideal, nadir = _plot_scale(F, archive_F)
    view = _goodness(F, ideal, nadir)

    plot = Scatter(
        title=title, legend=True, figsize=(10, 8), tight_layout=True,
        labels=[f"{label[:1].upper()}{label[1:]} (normalized, 1 = best)"
                for label in objective_labels],
    )
    annotations: list = []
    if feasible.any():
        annotations += _add_with_multiplicity(
            plot, view[feasible], color="green", s=50, alpha=0.6,
            label="Feasible",
        )
    if (~feasible).any():
        annotations += _add_with_multiplicity(
            plot, view[~feasible], color="red", s=50, alpha=0.6,
            label="Infeasible",
        )

    if archive_F is not None and len(archive_F) > 0:
        front_view = _goodness(np.asarray(archive_F, dtype=float), ideal, nadir)
        if planar:
            # A trade-off curve exists only with two objectives; beyond that the
            # front is a surface and a polyline through it would invent an order.
            plot.add(front_view[np.argsort(front_view[:, 0])],
                     plot_type="line", color="black", alpha=0.4)
        plot.add(front_view, s=140, facecolors="none", edgecolors="black",
                 label="Run Pareto front (all generations)")
    elif feasible.any():
        front = NonDominatedSorting().do(
            F[feasible], only_non_dominated_front=True,
        )
        front_pts = np.unique(view[feasible][front].round(6), axis=0)
        plot.add(front_pts, s=140, facecolors="none", edgecolors="black",
                 label="Non-dominated")

    plot.do()
    if planar:
        for (x, y), count in annotations:
            plot.ax.annotate(f"×{count}", (x, y), textcoords="offset points",
                             xytext=(8, 5), fontsize=9)
    plot.ax.grid(True, alpha=0.3)

    if save_path is not None:
        plot.save(save_path)
    return plot.fig
