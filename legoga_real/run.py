"""
Uruchomienie: python -m legoga_real.run

Prototyp NSGA-II dla ukladania torow LEGO.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import dataclasses
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pymoo
from pymoo.algorithms.moo.nsga2 import NSGA2, binary_tournament
from pymoo.indicators.hv import HV
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from .ga import (TrackCrossover, TrackDuplicates, TrackMutation, TrackProblem,
                 TrackSampling, physical_counts, tournament)
from .progress import ProgressCallback
from .render import save_gallery, save_layouts
from .settings import Settings


def front_spread(rows, n: int = 6):
    """Punkty frontu w rownych odstepach na liscie par (klocki, trasy), po jednym na pare.

    Front zawiera zwykle po kilka ukladow o tych samych wartosciach obu celow -
    moga roznic sie skladem i ksztaltem, ale wzgledem celow sa rownowazne. Wybor
    pierwszych `n` wierszy oddaje im caly rysunek i gubi drugi koniec frontu.
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


def run(inventory=None, pop=60, gens=60, seed=1, out="outputs/legoga",
        max_size=(500.0, 500.0), verbose=True, cfg: Settings = Settings(),
        table=None, hv=None, render=True):
    """Pelny przebieg; wszystkie wyniki laduja w katalogu `out`.

    Zwraca (res, rows, times); `times` to sekundy utworzenia problemu (z budowa
    tabeli, gdy jej nie podano) i optymalizacji.
    `render=False` pomija rysunki (galeria, uklady, wykresy przebiegu); CSV zostaje.
    """
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    gallery_path = out_dir / "tory.png"
    t0 = time.time()
    problem = TrackProblem(inventory=inventory, seed=seed,
                           max_size=max_size, cfg=cfg, table=table)
    t_table = time.time() - t0
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
                   callback=ProgressCallback(hv))
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

    stem = gallery_path.with_suffix("")
    progress = res.algorithm.callback
    csv_path = progress.save_csv(f"{stem}_progress.csv")
    times = {"table": t_table, "optimize": dt}
    if not render:
        print(f"Przebieg: {csv_path}")
        return res, rows, times

    if rows:
        picked = front_spread(rows)
        gallery = save_gallery(picked, gallery_path, max_size)
        paths = save_layouts(picked, gallery_path, problem.inv, max_size)
        print("\nRysunki: " + ", ".join(map(str, [gallery, *paths])))
    plot_paths = progress.save_plots(stem, n_gen_planned=gens,
                                     max_pieces=sum(problem.inv.values()))
    print(f"Przebieg: {csv_path}, " + ", ".join(map(str, plot_paths)))
    return res, rows, times


class Tee:
    """Wypisuje jednoczesnie do kilku strumieni: na ekran i do log.txt."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def save_results(out: Path, res, pop, gens, seed, times, started) -> None:
    """Zapisuje front.csv, layouts.json i run.json biegu uruchomionego z konsoli."""
    problem = res.problem
    layouts = [x[0] for x in np.atleast_2d(res.X)]
    F, G = np.atleast_2d(res.F), np.atleast_2d(res.G)
    feas = (G <= 0).all(axis=1)
    with (out / "front.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["pieces", "routes", "overlap", "inventory_excess", "over_size",
                    "width", "height", "feasible", "counts"])
        for k, lay in enumerate(layouts):
            _, width, height = lay.fit(problem.max_size)
            w.writerow([int(-F[k, 0]), int(-F[k, 1]), *G[k].tolist(), round(width, 1),
                        round(height, 1), bool(feas[k]), json.dumps(physical_counts(lay))])
    dump = [{"types": lay.types, "match": [[*a, *b] for a, b in lay.match.items()]}
            for lay in layouts]
    (out / "layouts.json").write_text(json.dumps(dump), encoding="utf-8")

    def git(*args):
        return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()
    hv = HV(ref_point=np.array([0.0, 0.0]))
    info = {
        "inventory": problem.inv, "table": "x".join(str(int(v)) for v in problem.max_size),
        "max_size": problem.max_size, "seed": seed, "pop": pop, "gens": gens,
        "settings": dataclasses.asdict(problem.cfg),
        "git": {"commit": git("rev-parse", "HEAD"),
                "dirty": bool(git("status", "--porcelain", "--", "legoga_real"))},
        "python": platform.python_version(), "pymoo": pymoo.__version__,
        "numpy": np.__version__, "started": started, "times": times,
        "n_eval": int(res.algorithm.evaluator.n_eval),
        "hv": float(hv(F[feas])) if feas.any() else 0.0,
        "hv_ref": hv.ref_point.tolist(), "n_front": len(layouts),
        "n_feas_final": int(res.algorithm.callback.data["n_feas"][-1]),
        "best_pieces": float(-F[feas, 0].min()) if feas.any() else 0.0,
        "best_routes": float(-F[feas, 1].min()) if feas.any() else 0.0,
    }
    (out / "run.json").write_text(json.dumps(info, indent=1), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=60)
    ap.add_argument("--gens", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="outputs/legoga", help="katalog wynikow")
    ap.add_argument("--inventory", nargs="+", metavar="KLOCEK=SZTUKI",
                    help="np. S16=90 S24=2 R40=140 WL=7 WR=3 XX=3 DC=1")
    ap.add_argument("--workspace", default="500x500", help="stol w studach, np. 250x350")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    inventory = a.inventory and {k: int(n) for k, n in (p.split("=") for p in a.inventory)}
    width, height = map(float, a.workspace.split("x"))
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    with (out_dir / "log.txt").open("w", encoding="utf-8") as log, \
            contextlib.redirect_stdout(Tee(sys.stdout, log)):
        res, _, times = run(inventory=inventory, pop=a.pop, gens=a.gens, seed=a.seed, out=a.out,
                            max_size=(width, height), verbose=not a.quiet)
    save_results(out_dir, res, a.pop, a.gens, a.seed, times, started)
