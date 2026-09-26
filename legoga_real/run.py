"""
Uruchomienie: python -m legoga_real.run

Prototyp NSGA-II dla ukladania torow LEGO.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2, binary_tournament
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from .ga import (TrackCrossover, TrackDuplicates, TrackMutation, TrackProblem,
                 TrackSampling, tournament)
from .progress import ProgressCallback
from .render_with_v1 import save_gallery, save_layouts
from .settings import Settings


def front_spread(rows, n: int = 6):
    """Rownomiernie rozlozone punkty frontu, po jednym na pare (klocki, trasy).

    Front zawiera zwykle po kilka ukladow o tych samych wartosciach obu celow -
    roznych fizycznie, lecz nieodroznialnych na rysunku. Wybor pierwszych `n`
    wierszy oddaje im caly rysunek i gubi drugi koniec frontu.
    """
    uniq, seen = [], set()
    for row in rows:
        _, used, routes, _, _ = row
        if (used, routes) not in seen:
            seen.add((used, routes))
            uniq.append(row)
    if len(uniq) <= n:
        return uniq
    step = (len(uniq) - 1) / max(n - 1, 1)
    return [uniq[round(i * step)] for i in range(n)]


def run(inventory=None, pop=60, gens=40, seed=1, out="outputs/legoga",
        max_size=(500.0, 500.0), verbose=True, cfg: Settings = Settings(),
        table=None):
    """Pelny przebieg; wszystkie wyniki laduja w katalogu `out`."""
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    gallery_path = out_dir / "tory.png"
    problem = TrackProblem(inventory=inventory, seed=seed,
                           max_size=max_size, cfg=cfg, table=table)
    compare = binary_tournament if cfg.pressure == 2 else tournament
    algo = NSGA2(
        pop_size=pop,
        n_offsprings=cfg.n_offsprings,
        sampling=TrackSampling(),
        selection=TournamentSelection(func_comp=compare, pressure=cfg.pressure),
        crossover=TrackCrossover(prob=cfg.cx_prob),
        mutation=TrackMutation(),
        eliminate_duplicates=TrackDuplicates(cfg.sig_with_size) if cfg.dedupe else False,
    )
    t0 = time.time()
    res = minimize(problem, algo, get_termination("n_gen", gens),
                   seed=seed, verbose=verbose, save_history=False,
                   callback=ProgressCallback())
    dt = time.time() - t0

    X = np.atleast_2d(res.X)
    F = np.atleast_2d(res.F)
    G = np.atleast_2d(res.G) if res.G is not None else None

    rows = []
    for k in range(len(X)):
        lay = X[k, 0] if X.dtype == object else X[k]
        _, w, h = lay.fit(max_size)
        rows.append((lay, -F[k, 0], -F[k, 1],
                     G[k, 0] if G is not None else 0,
                     (w, h)))
    rows.sort(key=lambda r: (-r[2], -r[1]))

    print(f"\nCzas: {dt:.1f} s | front Pareto: {len(rows)} rozwiazan\n")
    print(f"{'elem':>5} {'trasy':>6} {'kolizje':>8} {'rozmiar':>13}  sklad")
    for lay, used, routes, ov, sz in rows:
        c = lay.counts()
        s = " ".join(f"{k}:{v}" for k, v in sorted(c.items()))
        print(f"{used:5.0f} {routes:6.0f} {ov:8.0f} "
              f"{sz[0]:6.0f}x{sz[1]:<6.0f}  {s}")

    if rows:
        picked = front_spread(rows)
        gallery = save_gallery(picked, gallery_path, max_size)
        paths = save_layouts(picked, gallery_path, problem.inv, max_size)
        print("\nRysunki: " + ", ".join(map(str, [gallery, *paths])))

    stem = gallery_path.with_suffix("")
    progress = res.algorithm.callback
    csv_path = progress.save_csv(f"{stem}_progress.csv")
    plot_paths = progress.save_plots(stem, n_gen_planned=gens,
                                     max_pieces=sum(problem.inv.values()))
    print(f"Przebieg: {csv_path}, " + ", ".join(map(str, plot_paths)))
    return res, rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=60)
    ap.add_argument("--gens", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="outputs/legoga", help="katalog wynikow")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    run(pop=a.pop, gens=a.gens, seed=a.seed, out=a.out, verbose=not a.quiet)
