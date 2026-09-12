"""
Reprezentacja ukladu: zbior instancji elementow + DOSKONALE SKOJARZENIE portow.

Warunek "brak otwartych torow" jest tu DEFINICYJNY: kazdy port ma pare.
Zadna funkcja kary tego nie pilnuje - operatory po prostu nie potrafia
wyprodukowac ukladu z wolnym portem.

Co pozostaje miekkie:
  * domkniecie GEOMETRYCZNE (czy petla wraca do siebie w plaszczyznie),
  * brak nakladania sie torow.
"""
from __future__ import annotations

import heapq
import math
from collections import defaultdict, deque

from .geometry import (ANG, ELEM_LEN, ELEMENTS, IDENT, TRACK_WIDTH, apply,
                       compose, inverse, port_pose)

POS_TOL = 0.30      # tolerancja domkniecia geometrycznego (study)
# Droga PO TORZE, ponizej ktorej bliskosc przestrzenna jest legalna.
# Ta stala nie jest dowolna - wynika z geometrii czesci:
#  * rozejscie na rozjezdzie (22.5 st.) osiaga 8 studow odstepu po
#    8/sin(22.5) = 20.9 studach, czyli ~42 studach liczonych przez iglice;
#  * przy promieniu 40 tor po 40 studach jazdy jest 38 studow od punktu
#    startu, wiec nie da sie wrocic do siebie ponizej tego progu.
# Stad 40: ponizej bliskosc jest wymuszona geometria, powyzej to konflikt.
NEAR_STUDS = 40.0


class Layout:
    __slots__ = ("types", "match", "_cache")

    def __init__(self, types=None, match=None):
        self.types: list[str] = list(types or [])
        self.match: dict = dict(match or {})
        self._cache = {}

    # -- konstrukcja --------------------------------------------------------

    @staticmethod
    def from_cycle(seq) -> "Layout":
        """Zamknieta petla z cyklicznego ciagu elementow 2-portowych."""
        n = len(seq)
        lay = Layout(list(seq))
        for i in range(n):
            j = (i + 1) % n
            lay.link((i, 1), (j, 0))
        return lay

    def copy(self) -> "Layout":
        return Layout(self.types, self.match)

    def link(self, a, b):
        self.match[a] = b
        self.match[b] = a
        self._cache.clear()

    def unlink(self, a):
        b = self.match.pop(a, None)
        if b is not None:
            self.match.pop(b, None)
        self._cache.clear()
        return b

    def add(self, tname: str) -> int:
        self.types.append(tname)
        self._cache.clear()
        return len(self.types) - 1

    def compact(self) -> "Layout":
        """Usun instancje bez zadnego polaczenia i przenumeruj."""
        used = sorted({i for (i, _) in self.match})
        remap = {old: new for new, old in enumerate(used)}
        lay = Layout([self.types[i] for i in used])
        for (i, p), (j, q) in self.match.items():
            lay.match[(remap[i], p)] = (remap[j], q)
        return lay

    # -- podstawowe wlasnosci ----------------------------------------------

    def __len__(self):
        return len(self.types)

    def open_ports(self):
        out = []
        for i, t in enumerate(self.types):
            for p in range(ELEMENTS[t].n_ports):
                if (i, p) not in self.match:
                    out.append((i, p))
        return out

    def is_matched(self) -> bool:
        return not self.open_ports()

    def counts(self):
        c = defaultdict(int)
        for t in self.types:
            c[t] += 1
        return dict(c)

    # -- geometria ----------------------------------------------------------

    def place(self):
        """BFS od elementu 0. Zwraca (transformacje, max_niezgodnosc)."""
        if "place" in self._cache:
            return self._cache["place"]
        n = len(self.types)
        T = [None] * n
        if n == 0:
            return T, 0.0
        T[0] = IDENT
        mism = 0.0
        dq = deque([0])
        while dq:
            i = dq.popleft()
            ti = T[i]
            for p in range(ELEMENTS[self.types[i]].n_ports):
                nb = self.match.get((i, p))
                if nb is None:
                    continue
                j, q = nb
                # poza portu i.p, odwrocona o 180 st.
                target = compose(compose(ti, port_pose(self.types[i], p)),
                                 (0.0, 0.0, 8))
                tj = compose(target, inverse(port_pose(self.types[j], q)))
                if T[j] is None:
                    T[j] = tj
                    dq.append(j)
                else:
                    e = math.hypot(T[j][0] - tj[0], T[j][1] - tj[1])
                    if T[j][2] != tj[2]:
                        e += 1e3          # niezgodnosc kata = twardy blad
                    mism = max(mism, e)
        self._cache["place"] = (T, mism)
        return T, mism

    def geometry_error(self) -> float:
        return self.place()[1]

    def is_planar_closed(self) -> bool:
        T, m = self.place()
        return all(t is not None for t in T) and m < POS_TOL

    # -- graf zredukowany ---------------------------------------------------

    def nodes(self):
        return [i for i, t in enumerate(self.types) if ELEMENTS[t].is_node]

    def reduce(self):
        """Sciagnij lancuchy elementow 2-portowych do krawedzi.

        Zwraca {(node_i, port): (node_j, port, dlugosc_lancucha)}.
        """
        if "reduce" in self._cache:
            return self._cache["reduce"]
        nodeset = set(self.nodes())
        edges = {}
        for i in nodeset:
            for p in range(ELEMENTS[self.types[i]].n_ports):
                cur = self.match.get((i, p))
                steps = 0
                while cur is not None and cur[0] not in nodeset:
                    j, q = cur
                    steps += 1
                    cur = self.match.get((j, 1 - q))
                if cur is not None:
                    edges[(i, p)] = (cur[0], cur[1], steps)
        self._cache["reduce"] = edges
        return edges

    def components(self) -> int:
        """Liczba spojnych skladowych (osobno lezacych ukladow)."""
        n = len(self.types)
        parent = list(range(n))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for (i, _), (j, _) in self.match.items():
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj
        return len({find(i) for i in range(n)}) if n else 0

    # -- liczenie tras ------------------------------------------------------

    def count_routes(self) -> int:
        """Liczba roznych ZAMKNIETYCH tras przejezdnych przez pociag.

        Uwzglednia, ze rozjazd nie laczy drogi prostej z odgalezieniem,
        a skrzyzowanie to dwie niezalezne sciezki.
        """
        nodes = self.nodes()
        if not nodes:
            return 1 if self.is_matched() and len(self.types) else 0

        edges = self.reduce()
        # stany = (wezel, port_wejsciowy)
        states = []
        for i in nodes:
            for p in range(ELEMENTS[self.types[i]].n_ports):
                states.append((i, p))
        idx = {s: k for k, s in enumerate(states)}

        # dozwolone przejscia wewnatrz wezla
        succ = defaultdict(list)
        for i in nodes:
            rts = ELEMENTS[self.types[i]].routes
            for (a, b) in rts:
                for (frm, to) in ((a, b), (b, a)):
                    e = edges.get((i, to))
                    if e is not None:
                        succ[idx[(i, frm)]].append(idx[(e[0], e[1])])

        # enumeracja cykli prostych; start zawsze najmniejszym indeksem
        count = 0
        n = len(states)

        def dfs(start, cur, visited):
            nonlocal count
            for nx in succ[cur]:
                if nx == start:
                    count += 1
                elif nx > start and nx not in visited:
                    visited.add(nx)
                    dfs(start, nx, visited)
                    visited.discard(nx)

        for s in range(n):
            dfs(s, s, {s})
        return count // 2      # kazda trasa policzona w obie strony

    def cyclomatic(self) -> int:
        """Liczba niezaleznych petli - tani zamiennik count_routes."""
        v = len(self.types)
        e = len(self.match) // 2
        return e - v + self.components()

    # -- kolizje ------------------------------------------------------------

    def near_pairs(self, max_studs: float = NEAR_STUDS):
        """Pary elementow odlegle o <= max_studs DROGI PO TORZE.

        Kryterium liczy study, nie kroki w grafie. Tory rozchodzace sie na
        rozjezdzie sa 3 study od siebie tuz za iglica - to geometria samej
        czesci, nie kolizja. Ale cztery kroki po prostych 24 to juz 96 studow
        toru i tam bliskosc przestrzenna jest realnym konfliktem. Liczenie
        krokow myli te dwa przypadki; liczenie studow nie.
        """
        if "near" in self._cache:
            return self._cache["near"]
        adj = defaultdict(list)
        for (i, _), (j, _) in self.match.items():
            w = 0.5 * (ELEM_LEN.get(self.types[i], 16.0) +
                       ELEM_LEN.get(self.types[j], 16.0))
            adj[i].append((j, w))
            adj[j].append((i, w))
        out = set()
        for s in range(len(self.types)):
            dist = {s: 0.0}
            heap = [(0.0, s)]
            while heap:
                d, u = heapq.heappop(heap)
                if d > dist.get(u, 1e9) or d > max_studs:
                    continue
                for v, w in adj[u]:
                    nd = d + w
                    if nd <= max_studs and nd < dist.get(v, 1e9):
                        dist[v] = nd
                        heapq.heappush(heap, (nd, v))
            for u in dist:
                if u != s:
                    out.add((min(s, u), max(s, u)))
        self._cache["near"] = out
        return out

    def overlap(self) -> int:
        if "ov" in self._cache:
            return self._cache["ov"]
        v = self._overlap()
        self._cache["ov"] = v
        return v

    def _overlap(self) -> int:
        """Liczba par probek nalezacych do roznych, niepolaczonych elementow,
        ktore leza blizej niz szerokosc toru."""
        T, mism = self.place()
        if mism >= POS_TOL:
            return 10 ** 6

        pts = []
        for i, t in enumerate(self.types):
            if T[i] is None:
                continue
            for (_, _, line) in ELEMENTS[t].centerline:
                for p in line:
                    pts.append((apply(T[i], p), i))

        neighbours = self.near_pairs()

        cell = TRACK_WIDTH
        grid = defaultdict(list)
        for k, ((x, y), i) in enumerate(pts):
            grid[(int(x // cell), int(y // cell))].append(k)

        thr = TRACK_WIDTH          # osie blizej niz szerokosc toru = konflikt
        seen = set()
        bad = 0
        for (cx, cy), bucket in grid.items():
            cand = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    cand.extend(grid.get((cx + dx, cy + dy), ()))
            for k in bucket:
                (x1, y1), i1 = pts[k]
                for m in cand:
                    if m <= k:
                        continue
                    (x2, y2), i2 = pts[m]
                    if i1 == i2:
                        continue
                    key = (min(i1, i2), max(i1, i2))
                    if key in neighbours:
                        continue
                    if (x1 - x2) ** 2 + (y1 - y2) ** 2 < thr * thr:
                        if (k, m) not in seen:
                            seen.add((k, m))
                            bad += 1
        return bad

    def bbox(self):
        T, _ = self.place()
        xs, ys = [], []
        for i, t in enumerate(self.types):
            if T[i] is None:
                continue
            for (_, _, line) in ELEMENTS[t].centerline:
                for p in line:
                    x, y = apply(T[i], p)
                    xs.append(x)
                    ys.append(y)
        if not xs:
            return (0, 0, 0, 0)
        return (min(xs), min(ys), max(xs), max(ys))

    def summary(self):
        x0, y0, x1, y1 = self.bbox()
        return {
            "elements": len(self.types),
            "counts": self.counts(),
            "matched": self.is_matched(),
            "geom_error": round(self.geometry_error(), 4),
            "components": self.components(),
            "routes": self.count_routes(),
            "overlap": self.overlap(),
            "size_studs": (round(x1 - x0, 1), round(y1 - y0, 1)),
        }

    def min_clearance(self):
        """Najmniejszy odstep osi torow wsrod par dalszych niz NEAR_STUDS po torze.

        Certyfikat do recznego sprawdzenia wyniku: wartosc < TRACK_WIDTH
        oznacza fizyczny konflikt, ktorego overlap() nie powinien przepuscic.
        Zwraca (odleglosc, elementA, elementB).
        """
        T, m = self.place()
        if m >= POS_TOL:
            return (0.0, None, None)
        pts = []
        for i, t in enumerate(self.types):
            if T[i] is None:
                continue
            for (_, _, line) in ELEMENTS[t].centerline:
                for p in line:
                    pts.append((apply(T[i], p), i))
        near = self.near_pairs()
        cell = TRACK_WIDTH * 3
        grid = defaultdict(list)
        for k, ((x, y), _) in enumerate(pts):
            grid[(int(x // cell), int(y // cell))].append(k)
        best = (float("inf"), None, None)
        for (cx, cy), bucket in grid.items():
            cand = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    cand.extend(grid.get((cx + dx, cy + dy), ()))
            for k in bucket:
                (x1, y1), i1 = pts[k]
                for m2 in cand:
                    if m2 <= k:
                        continue
                    (x2, y2), i2 = pts[m2]
                    if i1 == i2 or (min(i1, i2), max(i1, i2)) in near:
                        continue
                    d = math.hypot(x1 - x2, y1 - y2)
                    if d < best[0]:
                        best = (d, i1, i2)
        return best
