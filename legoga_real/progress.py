"""Przebieg celow w trakcie biegu: najlepsza poprawna wartosc kazdego celu co pokolenie."""
from __future__ import annotations

import csv
import time
from collections.abc import Sequence
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from pymoo.core.callback import Callback

# Bez interfejsu: tylko pliki PNG (Tk i watki Pool wywracaja Tcl).
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402

COLUMNS = ("n_gen", "n_feas", "best_pieces", "best_routes", "hv", "n_eval", "t_wall",
           "mut_calls", "mut_hits", "cx_calls", "cx_hits")

# Jedna seria: tytul ja nazywa, wiec legenda jest zbedna.
SURFACE = "#fcfcfb"
SERIES = "#2a78d6"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS_RULE = "#c3c2b7"


def plot_score_progress(
    generations: Sequence[float],
    scores: Sequence[float],
    max_score: float | None = None,
    save_path: Path | None = None,
    title: str | None = None,
    n_gen_planned: int | None = None,
    objective_label: str = "weighted piece score",
    maximized: bool = True,
) -> Figure:
    """Najlepsza poprawna wartosc celu w funkcji pokolenia.

    `max_score` rysuje sufit inwentarza jako przerywana linie odniesienia; None
    oznacza cel bez sufitu. `n_gen_planned` wydluza os x do calego budzetu.
    """
    if title is None:
        title = f"{objective_label} by generation"
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
    direction = "higher is better" if maximized else "lower is better"
    ax.set_ylabel(f"{objective_label[:1].upper()}{objective_label[1:]} ({direction})",
                  color=INK_SECONDARY, fontsize=11)
    _strip_chrome(ax)
    return _finish(fig, save_path)


def _draw_ceiling(ax, max_score: float) -> None:
    """Przerywana linia na wysokosci sufitu inwentarza.

    Przerywana kreska oznacza tu prog; siatka i osie zostaja ciagle.
    """
    ax.axhline(max_score, color=INK_MUTED, linewidth=1.4, linestyle=(0, (6, 4)),
               zorder=2)
    ax.annotate(f"inventory ceiling {max_score:.3f} — whole kit, no terrain limit",
                xy=(ax.get_xlim()[0], max_score), xytext=(4, 6),
                textcoords="offset points", color=INK_SECONDARY, fontsize=10)


def _annotate_endpoint(ax, generations: NDArray, scores: NDArray) -> None:
    """Podpisana ostatnia wartosc; reszte niosa os i ksztalt krzywej."""
    ax.plot(generations[-1], scores[-1], "o", markersize=8, color=SERIES,
            markeredgecolor=SURFACE, markeredgewidth=2.0, zorder=4)
    ax.annotate(f"{scores[-1]:.3f}", xy=(generations[-1], scores[-1]),
                xytext=(10, -4), textcoords="offset points",
                color=INK_PRIMARY, fontsize=12, fontweight="semibold")


def _set_y_range(ax, scores: NDArray, max_score: float | None) -> None:
    """Zakres obejmuje dane i sufit, zeby widac bylo pozostaly zapas."""
    low = float(scores.min())
    high = float(scores.max())
    if max_score is not None:
        high = max(high, float(max_score))
    margin = (high - low) * 0.12 or max(abs(high) * 0.02, 0.01)
    ax.set_ylim(low - margin, high + margin)


def _strip_chrome(ax) -> None:
    """Ciagle cienkie linie: tylko pozioma siatka, bez gornej i prawej ramki."""
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


class ProgressCallback(Callback):
    """Co pokolenie: liczba poprawnych ukladow, najlepsza wartosc kazdego celu,
    pole frontu (gdy podano licznik `hv`), liczba ocen, sekundy od startu i
    liczniki operatorow. Poprawny uklad ma CV <= 0; bez poprawnych jest NaN.
    """

    def __init__(self, hv=None) -> None:
        super().__init__()
        self.hv = hv
        self.t0 = time.perf_counter()
        self.data.update((key, []) for key in COLUMNS)

    def notify(self, algorithm) -> None:
        F = algorithm.pop.get("F")
        CV = algorithm.pop.get("CV")
        feasible = F[CV.ravel() <= 0.0]
        n_feas = len(feasible)
        best_pieces, best_routes = -feasible.min(axis=0) if n_feas else (np.nan, np.nan)
        hv = self.hv(feasible) if self.hv and n_feas else np.nan
        mut, cx = algorithm.mating.mutation, algorithm.mating.crossover
        row = (int(algorithm.n_gen), n_feas, float(best_pieces), float(best_routes),
               float(hv), algorithm.evaluator.n_eval, time.perf_counter() - self.t0,
               mut.calls, mut.hits, cx.calls, cx.hits)
        for key, value in zip(COLUMNS, row):
            self.data[key].append(value)

    def save_csv(self, path: str | Path) -> Path:
        path = Path(path)
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(COLUMNS)
            writer.writerows(zip(*(self.data[key] for key in COLUMNS)))
        return path

    def save_plots(self, stem: str | Path, n_gen_planned: int,
                   max_pieces: int) -> tuple[Path, Path]:
        """Dwa wykresy: klocki (z sufitem inwentarza) i trasy."""
        gens = np.asarray(self.data["n_gen"], dtype=float)
        pieces = np.asarray(self.data["best_pieces"], dtype=float)
        routes = np.asarray(self.data["best_routes"], dtype=float)
        keep = np.isfinite(pieces)
        series = (
            ("pieces", pieces, max_pieces, "pieces used"),
            ("routes", routes, None, "routes"),
        )
        paths = []
        for name, values, ceiling, label in series:
            path = Path(f"{stem}_{name}.png")
            fig = plot_score_progress(gens[keep], values[keep], max_score=ceiling,
                                      save_path=path, n_gen_planned=n_gen_planned,
                                      objective_label=label)
            plt.close(fig)
            paths.append(path)
        return tuple(paths)
