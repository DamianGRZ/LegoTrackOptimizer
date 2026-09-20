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
from pymoo.core.duplicate import ElementwiseDuplicateElimination
from pymoo.core.mutation import Mutation
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling

from . import ops
from .layout import Layout
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
    def __init__(self, inventory=None, table_len: int = 11,
                 max_size=(400.0, 400.0), seed: int = 0):
        self.inv = dict(inventory or DEFAULT_INVENTORY)
        self.table = TransformTable(max_len=table_len)
        self.max_size = max_size
        self.rng = random.Random(seed)
        super().__init__(n_var=1, n_obj=2, n_ieq_constr=3, vtype=object)

    def _evaluate(self, x, out, *args, **kwargs):
        lay: Layout = x[0]
        x0, y0, x1, y1 = lay.bbox()
        over_size = max(0.0, (x1 - x0) - self.max_size[0]) + \
                    max(0.0, (y1 - y0) - self.max_size[1])
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
        rng = problem.rng
        inv = problem.inv
        max_s = inv.get("S16", 0)
        seeds = []
        for n in (4, 6, 8, 10, 12, 14):
            if 2 * n <= max_s:
                seeds.append(ops.oval(n))
        seeds.append(ops.circle())
        for n in (6, 10):
            if 2 * n <= max_s:
                seeds.append(ops.oval(n, "S16"))

        X = np.empty((n_samples, 1), dtype=object)
        for k in range(n_samples):
            lay = rng.choice(seeds)
            for _ in range(rng.randint(3, 25)):
                cand = ops.mutate(lay, problem.table, rng)
                if inventory_excess(cand, inv) == 0:
                    lay = cand
            X[k, 0] = lay
        return X


class TrackCrossover(Crossover):
    def __init__(self):
        super().__init__(2, 1)

    def _do(self, problem, X, **kwargs):
        rng = problem.rng
        _, n_matings, _ = X.shape
        Y = np.empty((1, n_matings, 1), dtype=object)
        for k in range(n_matings):
            a, b = X[0, k, 0], X[1, k, 0]
            c = ops.crossover(a, b, problem.table, rng)
            if inventory_excess(c, problem.inv) > 0:
                c = a
            Y[0, k, 0] = c
        return Y


class TrackMutation(Mutation):
    def __init__(self, prob: float = 0.9, n_ops: int = 2):
        super().__init__()
        self.prob_mut = prob
        self.n_ops = n_ops

    def _do(self, problem, X, **kwargs):
        rng = problem.rng
        for k in range(len(X)):
            if rng.random() > self.prob_mut:
                continue
            lay = X[k, 0]
            for _ in range(rng.randint(1, self.n_ops)):
                cand = ops.mutate(lay, problem.table, rng)
                if inventory_excess(cand, problem.inv) == 0:
                    lay = cand
            X[k, 0] = lay
        return X


class TrackDuplicates(ElementwiseDuplicateElimination):
    def is_equal(self, a, b):
        return _sig(a.X[0]) == _sig(b.X[0])


def _sig(lay: Layout):
    x0, y0, x1, y1 = lay.bbox()
    return (tuple(sorted(physical_counts(lay).items())),
            lay.count_routes(),
            round(x1 - x0, 1), round(y1 - y0, 1))
