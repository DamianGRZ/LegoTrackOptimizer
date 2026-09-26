"""Przebieg celow w trakcie biegu: najlepszy poprawny uklad co pokolenie."""
from __future__ import annotations

import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pymoo.core.callback import Callback

from src.visualization.objective_progress import plot_score_progress

COLUMNS = ("n_gen", "n_feas", "best_pieces", "best_routes", "hv", "n_eval", "t_wall",
           "mut_calls", "mut_hits", "cx_calls", "cx_hits")


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
