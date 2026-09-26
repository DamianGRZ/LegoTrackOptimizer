"""
Warstwa pymoo: problem wielokryterialny + operatory na obiektach Layout.

Reprezentacja jest niestandardowa (obiekt, nie wektor liczb), wiec
uzywamy wlasnych Sampling / Crossover / Mutation. pymoo obsluguje to
przez tablice numpy o dtype=object.

Cele (minimalizowane):
    f1 = -liczba uzytych elementow
    f2 = -liczba roznych zamknietych tras
Ograniczenia (<= 0):
    g1 = nakladanie sie torow
    g2 = przekroczenie magazynu
    g3 = przekroczenie rozmiaru stolu
"""
from __future__ import annotations

import random

import numpy as np
from pymoo.core.crossover import Crossover
from pymoo.core.duplicate import DuplicateElimination
from pymoo.core.mutation import Mutation
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling

from . import ops
from .layout import Layout
from .settings import Settings
from .table import TransformTable


# --- magazyn ----------------------------------------------------------------

# Magazyn z configs/all_pieces.yaml. R40 to JEDEN fizyczny klocek -
# kierunek (CL/CR) wybiera sie przy ulozeniu, wiec CL+CR dziela wspolna
# pule "R40".
DEFAULT_INVENTORY = {
    "S16": 120,
    "S24": 8,
    "R40": 80,
    "WL": 3,
    "WR": 3,
    "XX": 2,
    "DC": 2,
}


def physical_counts(lay: Layout) -> dict:
    """Liczby klockow fizycznych: CL i CR to ten sam R40 ulozony w inna strone."""
    c = lay.counts()
    c["R40"] = c.pop("CL", 0) + c.pop("CR", 0)
    return c


def inventory_excess(lay: Layout, inv: dict) -> int:
    return sum(max(0, n - inv.get(t, 0)) for t, n in physical_counts(lay).items())


def used_pieces(lay: Layout) -> int:
    return len(lay.types)


# --- problem ----------------------------------------------------------------

class TrackProblem(ElementwiseProblem):
    def __init__(self, inventory=None, max_size=(400.0, 400.0), seed: int = 0,
                 cfg: Settings = Settings(), table: TransformTable | None = None):
        self.inv = dict(inventory or DEFAULT_INVENTORY)
        self.cfg = cfg
        self.table = table or TransformTable(max_len=cfg.table_len,
                                             per_bucket=cfg.per_bucket)
        self.max_size = max_size
        self.rng = random.Random(seed)
        super().__init__(n_var=1, n_obj=2, n_ieq_constr=3, vtype=object)

    def _evaluate(self, x, out, *args, **kwargs):
        lay: Layout = x[0]
        _, w, h = lay.fit(self.max_size)
        over_size = max(0.0, w - self.max_size[0]) + max(0.0, h - self.max_size[1])
        out["F"] = [-used_pieces(lay), -lay.count_routes()]
        out["G"] = [lay.overlap(),
                    inventory_excess(lay, self.inv),
                    over_size]


class TrackSampling(Sampling):
    """Populacja startowa: kilka zarodkow + losowe mutacje na kazdym.

    Nie szukamy poprawnego ukladu od zera - bierzemy trywialnie poprawny
    owal/kolo i puszczamy na nim operatory zachowujace domkniecie.
    """

    def _do(self, problem, n_samples, **kwargs):
        rng, inv, cfg = problem.rng, problem.inv, problem.cfg
        max_s = inv.get("S16", 0)
        seeds = []
        for n in cfg.oval_sizes:
            if 2 * n <= max_s:
                seeds.append(ops.oval(n))
        seeds.append(ops.circle())
        if cfg.figure_eight and inv.get("XX", 0):
            seeds.append(ops.figure_eight())

        X = np.empty((n_samples, 1), dtype=object)
        for k in range(n_samples):
            lay = rng.choice(seeds)
            for _ in range(rng.randint(*cfg.warmup)):
                cand = ops.mutate(lay, problem.table, rng, cfg)
                if inventory_excess(cand, inv) == 0:
                    lay = cand
            X[k, 0] = lay
        return X


class TrackCrossover(Crossover):
    def __init__(self, prob: float = 0.9):
        super().__init__(2, 1, prob=prob)

    def _do(self, problem, X, **kwargs):
        rng = problem.rng
        _, n_matings, _ = X.shape
        Y = np.empty((1, n_matings, 1), dtype=object)
        for k in range(n_matings):
            a, b = X[0, k, 0], X[1, k, 0]
            c = ops.crossover(a, b, problem.table, rng, problem.cfg)
            if inventory_excess(c, problem.inv) > 0:
                c = a
            Y[0, k, 0] = c
        return Y


class TrackMutation(Mutation):
    def _do(self, problem, X, **kwargs):
        cfg = problem.cfg
        if cfg.forget_each_gen:
            problem.table.forget()
        rng = problem.rng
        for k in range(len(X)):
            if rng.random() > cfg.mut_prob:
                continue
            lay = X[k, 0]
            for _ in range(rng.randint(1, cfg.n_ops)):
                cand = ops.mutate(lay, problem.table, rng, cfg)
                if inventory_excess(cand, problem.inv) == 0:
                    lay = cand
            X[k, 0] = lay
        return X


def tournament(pop, P, **kwargs):
    """Turniej o rodzica dla grup wiekszych niz dwie; dla dwoch pymoo ma wlasna
    funkcje. Z grupy wygrywa uklad z najmniejszym naruszeniem ograniczen.
    Gdy naruszenie jest zerowe, wygrywa nizsza ranga (blizszy front), a przy
    tej samej randze wiekszy odstep od sasiadow. Uklady z naruszeniem nie
    maja rangi ani odstepu, wiec dla nich klucz konczy sie na naruszeniu."""
    def key(i):
        ind = pop[i]
        if ind.CV[0] > 0:
            return (ind.CV[0],)
        return (0.0, ind.get("rank"), -ind.get("crowding"))
    return np.array([[min(row, key=key)] for row in P])


class TrackDuplicates(DuplicateElimination):
    """Duplikat = rowna sygnatura; zbior zamiast porownania kazdej pary."""

    def __init__(self, with_size: bool = True):
        super().__init__()
        self.with_size = with_size

    def _do(self, pop, other, is_duplicate):
        seen = set() if other is None else {self._sig(ind.X[0]) for ind in other}
        for i, ind in enumerate(pop):
            s = self._sig(ind.X[0])
            is_duplicate[i] = s in seen
            seen.add(s)
        return is_duplicate

    def _sig(self, lay: Layout):
        _, w, h = lay.fit()
        size = (round(max(w, h), 1), round(min(w, h), 1)) if self.with_size else ()
        return (tuple(sorted(physical_counts(lay).items())),
                lay.count_routes(), *size, lay.overlap() == 0)
