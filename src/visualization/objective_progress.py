"""F[0] trajectory over generations — weighted piece score, or route length
in studs under f0_objective=route_length.

Not a convergence plot: convergence in multi-objective search means approaching
the true Pareto front and is measured by HV/IGD. This is the value of one
objective per generation, rising because F[0] is maximized.

pymoo has no chart for it — its ``RunningMetricAnimation`` plots front movement
between generations and needs ``save_history``. The documented pymoo idiom is a
matplotlib line over data a Callback collected, which
``ConvergenceMonitorCallback`` already writes to ``convergence.csv``.
"""

from collections.abc import Sequence
from pathlib import Path

import matplotlib

# Headless pipeline: PNGs only (Tk + Pool result-handler threads crash Tcl).
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

# One categorical slot: the chart carries a single series and the title names
# it, so no legend box is needed.
SURFACE = "#fcfcfb"
SERIES = "#2a78d6"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS_RULE = "#c3c2b7"


def load_score_progress(csv_path: Path) -> tuple[NDArray, NDArray]:
    """(generations, F[0] score) from a run's ``convergence.csv``.

    The monitor stores ``best_f0``, the minimum F[0] over feasible individuals,
    so the score is its negation. Generations with no feasible individual carry
    NaN and are dropped.
    """
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    generations = np.atleast_1d(data["n_gen"]).astype(float)
    best_f0 = np.atleast_1d(data["best_f0"]).astype(float)
    keep = np.isfinite(best_f0)
    return generations[keep], -best_f0[keep]


def plot_score_progress(
    generations: Sequence[float],
    scores: Sequence[float],
    max_score: float | None = None,
    save_path: Path | None = None,
    title: str | None = None,
    n_gen_planned: int | None = None,
    f0_label: str = "weighted piece score",
) -> Figure:
    """Best feasible F[0] against generation, labelled for the configured variant.

    ``max_score`` draws the inventory ceiling as a dashed reference line — the
    score with the whole kit placed and no terrain limits; pass None when the
    variant has no computable ceiling (route_length) to plot the bare curve.
    ``n_gen_planned`` extends the x-axis to the full budget, so a chart drawn
    mid-run shows how much of the run is still ahead.
    """
    if title is None:
        title = f"F[0] {f0_label} by generation"
    generations = np.asarray(generations, dtype=float)
    scores = np.asarray(scores, dtype=float)

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=140)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK_PRIMARY, fontsize=14, pad=14, loc="left")

    if len(scores) == 0:
        ax.text(0.5, 0.5, "no feasible individual yet", transform=ax.transAxes,
                ha="center", va="center", color=INK_MUTED, fontsize=13)
        _strip_chrome(ax)
        return _finish(fig, save_path)

    ax.plot(generations, scores, color=SERIES, linewidth=2.0,
            solid_capstyle="round", zorder=3)
    _annotate_endpoint(ax, generations, scores)

    x_max = max(float(generations[-1]), float(n_gen_planned or 0))
    ax.set_xlim(float(generations[0]) - 0.5, x_max * 1.06)
    if max_score is not None:
        _draw_ceiling(ax, float(max_score))
    _set_y_range(ax, scores, max_score)

    ax.set_xlabel("Generation", color=INK_SECONDARY, fontsize=11)
    ax.set_ylabel(f"{f0_label[:1].upper()}{f0_label[1:]} (higher is better)",
                  color=INK_SECONDARY, fontsize=11)
    _strip_chrome(ax)
    return _finish(fig, save_path)


def _draw_ceiling(ax, max_score: float) -> None:
    """Dashed reference line at the inventory ceiling.

    Dashes mark a threshold here, which is the one place they belong — grid and
    axis rules stay solid.
    """
    ax.axhline(max_score, color=INK_MUTED, linewidth=1.4, linestyle=(0, (6, 4)),
               zorder=2)
    ax.annotate(f"inventory ceiling {max_score:.3f} — whole kit, no terrain limit",
                xy=(ax.get_xlim()[0], max_score), xytext=(4, 6),
                textcoords="offset points", color=INK_SECONDARY, fontsize=10)


def _annotate_endpoint(ax, generations: NDArray, scores: NDArray) -> None:
    """Direct-label the latest value; the axis and the shape carry the rest."""
    ax.plot(generations[-1], scores[-1], "o", markersize=8, color=SERIES,
            markeredgecolor=SURFACE, markeredgewidth=2.0, zorder=4)
    ax.annotate(f"{scores[-1]:.3f}", xy=(generations[-1], scores[-1]),
                xytext=(10, -4), textcoords="offset points",
                color=INK_PRIMARY, fontsize=12, fontweight="semibold")


def _set_y_range(ax, scores: NDArray, max_score: float | None) -> None:
    """Span the data and the ceiling, so the remaining headroom is visible."""
    low = float(scores.min())
    high = float(scores.max())
    if max_score is not None:
        high = max(high, float(max_score))
    margin = (high - low) * 0.12 or max(abs(high) * 0.02, 0.01)
    ax.set_ylim(low - margin, high + margin)


def _strip_chrome(ax) -> None:
    """Recessive solid hairlines: horizontal grid only, no top/right spines."""
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS_RULE)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_MUTED, labelsize=10, length=0)


def _finish(fig: Figure, save_path: Path | None) -> Figure:
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, facecolor=SURFACE, bbox_inches="tight")
    return fig
