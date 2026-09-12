"""
Tablica klas rownowaznosci przeksztalcen.

Generuje BFS-em wszystkie ciagi elementow 2-portowych do zadanej dlugosci
i grupuje je po przeksztalceniu netto (dx, dy, dtheta).

Dwa ciagi z tego samego kubelka sa WYMIENNE w dowolnym miejscu dowolnej
petli - podmiana nie narusza domkniecia, niezaleznie od otoczenia.

Ta sama struktura sluzy do:
  * mutacji (podmien fragment na rownowazny),
  * domykania galezi (znajdz ciag realizujacy zadane przeksztalcenie),
  * heurystyki dopuszczalnej w przeszukiwaniu (min. liczba elementow do domkniecia).
"""
from __future__ import annotations

import itertools
import math
import pickle
from collections import defaultdict
from pathlib import Path

from .geometry import ANG, CHAIN_ELEMENTS, advance, compose, inverse

GRID = 0.5          # rozmiar kubelka przestrzennego (study)
# Tolerancja uznania dwoch przeksztalcen za identyczne. Zmierzony port
# odgalezienia rozjazdu (32.75, 13.0) nie lezy idealnie na siatce sum lukow
# R40 (pochodna 3-4-5 daje 32.69, 12.96), wiec laczniki mijanek maja resztki
# ~0.1 studa. 0.15 wciaz jest ostrzejsze niz tolerancja domkniecia
# optymalizatora V1 (4.0 studa), a dryf ogranicza POS_TOL=0.30 w layout.py.
SNAP = 0.15


def _key(t):
    x, y, a = t
    return (int(math.floor(x / GRID)), int(math.floor(y / GRID)), a % ANG)


def _neigh_keys(t):
    x, y, a = t
    bx, by = int(math.floor(x / GRID)), int(math.floor(y / GRID))
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            yield (bx + dx, by + dy, a % ANG)


class TransformTable:
    def __init__(self, max_len: int = 7, elements=CHAIN_ELEMENTS,
                 per_bucket: int = 60):
        self.max_len = max_len
        self.elements = tuple(elements)
        self.per_bucket = per_bucket
        self.buckets: dict = defaultdict(list)   # key -> [(seq, transform)]
        self.min_len: dict = {}                  # key -> najkrotsza dlugosc
        self._build()

    def _build(self):
        adv = {e: advance(e) for e in self.elements}
        frontier = [((), (0.0, 0.0, 0))]
        self._insert((), (0.0, 0.0, 0))
        for _ in range(self.max_len):
            nxt = []
            for seq, t in frontier:
                for e in self.elements:
                    t2 = compose(t, adv[e])
                    seq2 = seq + (e,)
                    self._insert(seq2, t2)
                    nxt.append((seq2, t2))
            frontier = nxt

    def _insert(self, seq, t):
        k = _key(t)
        b = self.buckets[k]
        if len(b) < self.per_bucket:
            b.append((seq, t))
        prev = self.min_len.get(k)
        if prev is None or len(seq) < prev:
            self.min_len[k] = len(seq)

    # -- zapytania ----------------------------------------------------------

    def lookup(self, t, max_len: int | None = None, min_len: int = 0):
        """Ciagi o przeksztalceniu netto ~= t."""
        out = []
        for k in _neigh_keys(t):
            for seq, tt in self.buckets.get(k, ()):
                if abs(tt[0] - t[0]) < SNAP and abs(tt[1] - t[1]) < SNAP:
                    if min_len <= len(seq) and (max_len is None or len(seq) <= max_len):
                        out.append(seq)
        return out

    def closure_len(self, t) -> float:
        """Dolne oszacowanie liczby elementow potrzebnych do osiagniecia t.

        Heurystyka DOPUSZCZALNA w tym sensie, ze jesli zwraca inf, to
        domkniecie w <= max_len elementach jest niemozliwe -> mozna przyciac.
        """
        best = math.inf
        for k in _neigh_keys(t):
            v = self.min_len.get(k)
            if v is not None and v < best:
                best = v
        return best

    def stats(self):
        return {
            "max_len": self.max_len,
            "buckets": len(self.buckets),
            "sequences": sum(len(v) for v in self.buckets.values()),
        }

    # -- laczniki dluzsze niz tablica (meet in the middle) ------------------

    def _flat(self):
        if not hasattr(self, "_flatlist"):
            fl = [(s, t) for b in self.buckets.values() for (s, t) in b if s]
            self._flatlist = fl
        return self._flatlist

    def connectors(self, N, limit: int = 8, scan: int = 8000, rng=None):
        """Lancuchy o przeksztalceniu netto ~= N, do 2*max_len elementow.

        Najpierw zwykly lookup. Jesli za malo trafien, rozbijamy N = A o B
        i szukamy B w tablicy dla kolejnych A (meet in the middle). Zasieg
        rosnie z max_len do 2*max_len bez wzrostu pamieci.
        """
        memo = self.__dict__.setdefault("_conn_memo", {})
        mk = (_key(N), limit)
        if mk in memo:
            return memo[mk]
        out = [tuple(s) for s in self.lookup(N, max_len=self.max_len, min_len=1)]
        if len(out) >= limit:
            memo[mk] = out[:limit]
            return memo[mk]
        flat = self._flat()
        n_scan = min(scan, len(flat))
        idx = rng.sample(range(len(flat)), n_scan) if rng is not None else range(n_scan)
        for k in idx:
            s, t = flat[k]
            B = compose(inverse(t), N)
            for s2 in self.lookup(B, max_len=self.max_len, min_len=1):
                out.append(tuple(s) + tuple(s2))
                if len(out) >= limit:
                    memo[mk] = out
                    return out
        memo[mk] = out
        return out

    # -- cache --------------------------------------------------------------

    def save(self, path):
        Path(path).write_bytes(pickle.dumps(self))

    @staticmethod
    def load(path):
        return pickle.loads(Path(path).read_bytes())


_CACHE = {}


def get_table(max_len: int = 7) -> TransformTable:
    if max_len not in _CACHE:
        _CACHE[max_len] = TransformTable(max_len=max_len)
    return _CACHE[max_len]
