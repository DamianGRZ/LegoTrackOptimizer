"""
Zarodki populacji i operatory ZACHOWUJACE DOMKNIECIE.

Kazdy operator przeksztalca uklad domkniety w uklad domkniety.
Nie ma tu naprawy ani kary za otwarte tory - one po prostu nie powstaja.
Jedyne, co moze pojsc zle, to geometria (nachodzenie), i to lapie fitness.
"""
from __future__ import annotations

import random

from .geometry import (ANG, ELEM_LEN, ELEMENTS, STRAIGHT_LEN, SWITCH_BODY,
                       advance, compose, inverse, port_pose)
from .layout import POS_TOL, Layout
from .settings import NEUTRAL_BLOCKS, Settings
from .table import TransformTable

FLIP = {"CL": "CR", "CR": "CL", "S16": "S16", "S24": "S24"}
SWITCHES = ("WL", "WR")
STRAIGHTS = ("S16", "S24")


# --- zarodki ----------------------------------------------------------------

def oval(n_straight: int = 6, straight: str = "S16") -> Layout:
    """Klasyczny owal: 8 lukow + n prostych + 8 lukow + n prostych."""
    seq = ["CL"] * 8 + [straight] * n_straight + ["CL"] * 8 + [straight] * n_straight
    return Layout.from_cycle(seq)


def circle() -> Layout:
    return Layout.from_cycle(["CL"] * 16)


def figure_eight() -> Layout:
    """Osemka: skrzyzowanie i dwa lustrzane platki; jedna trasa przecinajaca sama siebie."""
    # 2 proste + pol krzyzowania = promien luku, wiec 3/4 okregu trafia w os poprzeczna
    lobe = ["S16"] * 2 + ["CR"] * 12 + ["S16"] * 2
    lay = _rebuild_chain(Layout(["XX"]), [], lobe, (0, 1), (0, 2))
    return _rebuild_chain(lay, [], [FLIP[t] for t in lobe], (0, 3), (0, 0))


# --- pomocnicze: lancuchy ---------------------------------------------------

def chain_from(lay: Layout, anchor):
    """Idz od portu `anchor` przez elementy 2-portowe.

    Zwraca (ids, znormalizowane_typy, koncowy_port) albo None.
    """
    nodeset = set(lay.nodes())
    ids, types = [], []
    cur = lay.match.get(anchor)
    while cur is not None and cur[0] not in nodeset:
        j, q = cur
        ids.append(j)
        types.append(lay.types[j] if q == 0 else FLIP[lay.types[j]])
        cur = lay.match.get((j, 1 - q))
    if cur is None:
        return None
    return ids, types, cur


def all_chains(lay: Layout):
    """Wszystkie maksymalne lancuchy elementow 2-portowych.

    Element (kotwica_wejscia, ids, typy, kotwica_wyjscia).
    Dla ukladu bez wezlow zwraca jeden lancuch cykliczny (kotwice = None).
    """
    nodes = lay.nodes()
    if not nodes:
        if not lay.types:
            return []
        # cykliczny: rozwin od (0, 0)
        ids, types = [], []
        start = (0, 0)
        cur = start
        while True:
            i, p = cur
            ids.append(i)
            types.append(lay.types[i] if p == 0 else FLIP[lay.types[i]])
            nxt = lay.match.get((i, 1 - p))
            if nxt is None or nxt == start:
                break
            cur = nxt
        return [(None, ids, types, None)]

    out, seen = [], set()
    for i in nodes:
        for p in range(ELEMENTS[lay.types[i]].n_ports):
            if (i, p) in seen:
                continue
            r = chain_from(lay, (i, p))
            if r is None:
                continue
            ids, types, end = r
            seen.add((i, p))
            seen.add(end)
            out.append(((i, p), ids, types, end))
    return out


def net_transform(types) -> tuple:
    t = (0.0, 0.0, 0)
    for e in types:
        t = compose(t, advance(e))
    return t


def _rebuild_chain(lay: Layout, ids, new_types, head, tail) -> Layout:
    """Podmien elementy `ids` (lancuch) na `new_types`.

    head/tail to porty na koncach (sasiad przed pierwszym i za ostatnim).
    """
    new = lay.copy()
    for i in ids:
        for p in range(ELEMENTS[new.types[i]].n_ports):
            new.unlink((i, p))
        new.types[i] = None                    # oznacz jako usuniety
    fresh = [new.add(t) for t in new_types]
    prev = head
    for k, fid in enumerate(fresh):
        new.link(prev, (fid, 0))
        prev = (fid, 1)
    new.link(prev, tail)
    return new.compact()


# --- operatory --------------------------------------------------------------

def mut_swap_segment(lay: Layout, table: TransformTable, rng: random.Random,
                     max_win: int = 6, tries: int = 15) -> Layout | None:
    """Podmien fragment lancucha na rownowazny z tablicy przeksztalcen.

    Dwa ciagi o tym samym przeksztalceniu netto sa wymienne bez wzgledu na
    otoczenie, wiec domkniecie calego ukladu jest zachowane automatycznie.
    """
    chains = [c for c in all_chains(lay) if c[1]]
    if not chains:
        return None
    for _ in range(tries):
        head, ids, types, tail = rng.choice(chains)
        n = len(ids)
        cyclic = head is None
        w = rng.randint(1, min(max_win, n))
        a = rng.randrange(n) if cyclic else rng.randrange(n - w + 1)
        win_types = [types[(a + k) % n] for k in range(w)]

        t = net_transform(win_types)
        alts = [s for s in table.lookup(t, max_len=table.max_len)
                if list(s) != win_types]
        if not alts:
            continue
        new_seq = list(rng.choice(alts))

        if cyclic:
            rot = [types[(a + k) % n] for k in range(n)]
            full = new_seq + rot[w:]
            cand = Layout.from_cycle(full)
        else:
            full = types[:a] + new_seq + types[a + w:]
            cand = _rebuild_chain(lay, ids, full, head, tail)
        if cand.is_matched() and cand.is_planar_closed():
            return cand
    return None


def mut_antipodal_insert(lay: Layout, rng: random.Random, block=None,
                         blocks=NEUTRAL_BLOCKS, min_chain: int = 4) -> Layout | None:
    """Wstaw ten sam blok neutralny katowo w dwoch miejscach roznych o 180 st.

    Netto przesuniecie sie znosi: R(a)d + R(a+180)d = 0, wiec przeksztalcenie
    lancucha nie zmienia sie i reszta ukladu pozostaje nietknieta.
    """
    block = list(block or rng.choice(blocks))
    if net_transform(block)[2] % ANG != 0:
        return None
    chains = [c for c in all_chains(lay) if len(c[1]) >= min_chain]
    if not chains:
        return None
    head, ids, types, tail = rng.choice(chains)
    n = len(types)

    # skumulowany kurs przed kazda pozycja
    head_ang, a = [], 0
    for e in types:
        head_ang.append(a % ANG)
        a = (a + advance(e)[2]) % ANG

    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)
             if (head_ang[j] - head_ang[i]) % ANG == 8]
    if not pairs:
        return None
    i, j = rng.choice(pairs)
    full = types[:i] + block + types[i:j] + block + types[j:]
    if head is None:
        return Layout.from_cycle(full)
    return _rebuild_chain(lay, ids, full, head, tail)


def put_switch(lay: Layout, i: int, tname: str, rev: bool) -> Layout:
    """Podmien element 2-portowy i na rozjazd.

    Droga glowna rozjazdu (port0<->port1) ma to samo przeksztalcenie co S16,
    wiec podmiana nie rusza geometrii reszty ukladu.
    `rev` odwraca zwrot rozjazdu - bez tego oba rozjazdy patrza w te sama
    strone i galaz musi byc petla nawrotna zamiast zwyklej mijanki.
    """
    A = lay.match[(i, 0)]
    B = lay.match[(i, 1)]
    new = lay.copy()
    new.unlink((i, 0))
    new.unlink((i, 1))
    new.types[i] = tname
    if rev:
        new.link(A, (i, 1))
        new.link(B, (i, 0))
    else:
        new.link(A, (i, 0))
        new.link(B, (i, 1))
    return new


def straight_runs(lay: Layout, studs: float):
    """Ciagi kolejnych prostych o lacznej dlugosci dokladnie `studs`.

    Zwraca liste (ids, p, q): ids to klocki ciagu po kolei, p to port
    pierwszego klocka prowadzacy do drugiego, q to port ostatniego klocka
    prowadzacy do przedostatniego. Reszta ukladu wisi na (ids[0], 1 - p)
    i (ids[-1], 1 - q). Ten sam ciag widziany od drugiego konca nie jest
    powtarzany. 32 study to zawsze dwie S16, 48 to trzy S16 albo dwie S24.
    """
    out, seen = [], set()
    for i, t in enumerate(lay.types):
        if t not in STRAIGHTS:
            continue
        for p in (0, 1):
            ids, q, total = [i], 1 - p, ELEM_LEN[t]
            cur = (i, p)
            while total < studs:
                nb = lay.match.get(cur)
                if nb is None or nb[0] in ids or lay.types[nb[0]] not in STRAIGHTS:
                    break
                ids.append(nb[0])
                q = nb[1]
                total += ELEM_LEN[lay.types[nb[0]]]
                cur = (nb[0], 1 - q)
            if total != studs:
                continue
            key = frozenset(((ids[0], 1 - p), (ids[-1], 1 - q)))
            if key in seen:
                continue
            seen.add(key)
            out.append((ids, p, q))
    return out


def s16_pairs(lay: Layout):
    """Pary sasiednich S16 pod rozjazd: korpus 32 study to dokladnie dwie S16."""
    return [(ids[0], p, ids[1], q) for ids, p, q in straight_runs(lay, SWITCH_BODY)]


def switch_frame(lay: Layout, T0, i: int, p: int, rev: bool):
    """Uklad odniesienia rozjazdu zastepujacego pare S16 (wejscie od
    zewnetrznego portu elementu i)."""
    outer = port_pose("S16", 1 - p)
    Tsw = compose(compose(T0[i], outer), (0.0, 0.0, 8))
    if rev:
        Tsw = compose(Tsw, (SWITCH_BODY, 0.0, 8))
    return Tsw


def put_switch_pair(lay: Layout, i: int, p: int, j: int, q: int,
                    tname: str, rev: bool) -> Layout | None:
    """Podmien pare sasiednich S16 na jeden rozjazd (32 study) w slocie i.

    Element j zostaje osierocony (bez polaczen) - caller MUSI na koncu
    wywolac compact(), ktore go usunie.
    """
    E1, E2 = (i, 1 - p), (j, 1 - q)
    A = lay.match.get(E1)
    B = lay.match.get(E2)
    if A is None or B is None:
        return None
    new = lay.copy()
    for e in (i, j):
        for pp in range(2):
            new.unlink((e, pp))
    new.types[i] = tname
    if rev:
        new.link(A, (i, 1))
        new.link(B, (i, 0))
    else:
        new.link(A, (i, 0))
        new.link(B, (i, 1))
    return new


def mut_add_branch(lay: Layout, table: TransformTable, rng: random.Random,
                   attempts: int = 30, n_seqs: int = 6, accept: str = "first") -> Layout | None:
    """Wstaw PARE rozjazdow i polacz ich odgalezienia lancuchem z tablicy.

    Rozjazdy zawsze parami -> parzystosc zachowana, wolnych portow brak
    z konstrukcji. Kazdy rozjazd zastepuje pare sasiednich S16 (korpus
    32 study). Przeszukiwane sa oba typy (WL/WR) i oba zwroty kazdego
    rozjazdu - tylko czesc kombinacji jest realizowalna geometrycznie.
    """
    pairs = s16_pairs(lay)
    if len(pairs) < 2:
        return None

    combos = [(ti, tj, ri, rj)
              for ti in SWITCHES for tj in SWITCHES
              for ri in (False, True) for rj in (False, True)]

    # Podmiana pary S16 -> rozjazd nie zmienia geometrii reszty ukladu,
    # wiec rozmieszczenie liczymy RAZ, poza petla po kombinacjach.
    T0, mism0 = lay.place()
    if mism0 >= POS_TOL:
        return None
    base_ov = lay.overlap()
    base_rt = lay.count_routes()
    best, best_score = None, None

    for _ in range(attempts):
        pa, pb = rng.sample(pairs, 2)
        i1, p1, j1, q1 = pa
        i2, p2, j2, q2 = pb
        if len({i1, j1, i2, j2}) < 4:
            continue
        if T0[i1] is None or T0[i2] is None:
            continue
        rng.shuffle(combos)
        for ti, tj, ri, rj in combos:
            Ti = switch_frame(lay, T0, i1, p1, ri)
            Tj = switch_frame(lay, T0, i2, p2, rj)
            Pi = compose(Ti, port_pose(ti, 2))
            Pj = compose(Tj, port_pose(tj, 2))
            N = compose(inverse(Pi), compose(Pj, (0.0, 0.0, 8)))

            seqs = table.lookup(N, max_len=table.max_len, min_len=1)
            if not seqs:
                continue
            base = put_switch_pair(lay, i1, p1, j1, q1, ti, ri)
            if base is None:
                continue
            base = put_switch_pair(base, i2, p2, j2, q2, tj, rj)
            if base is None:
                continue
            rng.shuffle(seqs)
            for seq in seqs[:n_seqs]:
                out = base.copy()
                prev = (i1, 2)
                for e in seq:
                    k = out.add(e)
                    out.link(prev, (k, 0))
                    prev = (k, 1)
                out.link(prev, (i2, 2))
                out = out.compact()
                if not (out.is_matched() and out.is_planar_closed()):
                    continue
                ov, rt = out.overlap(), out.count_routes()
                # preferuj galaz, ktora realnie dodaje trase i nie psuje geometrii
                score = (rt - base_rt, -(ov - base_ov))
                if best is None or score > best_score:
                    best, best_score = out, score
                if accept == "first" and score[0] > 0 and ov <= base_ov:
                    return out
    return best


def mut_remove_branch(lay: Layout, rng: random.Random) -> Layout | None:
    """Usun galaz laczaca odgalezienia dwoch rozjazdow.

    Kazdy rozjazd (32 study) wraca do PARY prostych S16 - odwrotnosc
    put_switch_pair.
    """
    sw = [i for i, t in enumerate(lay.types) if t in SWITCHES]
    if len(sw) < 2:
        return None
    rng.shuffle(sw)
    for i in sw:
        r = chain_from(lay, (i, 2))
        if r is None:
            continue
        ids, _, (j, q) = r
        if q != 2 or lay.types[j] not in SWITCHES or j == i:
            continue
        new = lay.copy()
        for k in ids:
            for p in range(ELEMENTS[new.types[k]].n_ports):
                new.unlink((k, p))
            new.types[k] = None
        new.unlink((i, 2))
        new.unlink((j, 2))
        for e in (i, j):
            b_side = new.match.get((e, 1))
            new.unlink((e, 1))
            new.types[e] = "S16"
            k = new.add("S16")
            new.link((e, 1), (k, 0))
            if b_side is not None:
                new.link((k, 1), b_side)
        out = new.compact()
        if out.is_matched():
            return out
    return None


def mutate(lay: Layout, table: TransformTable, rng: random.Random,
           cfg: Settings = Settings()) -> Layout:
    """Jedna proba mutacji; przy niepowodzeniu zwraca oryginal."""
    ops = [
        lambda: mut_swap_segment(lay, table, rng, cfg.max_win, cfg.swap_tries),
        lambda: mut_antipodal_insert(lay, rng, None, cfg.neutral_blocks,
                                     cfg.antipodal_min_chain),
        lambda: mut_add_branch(lay, table, rng, cfg.branch_attempts, cfg.branch_seqs,
                               cfg.accept),
        lambda: mut_remove_branch(lay, rng),
        lambda: mut_add_crossing_pair(lay, table, rng, cfg.xx_attempts, cfg.xx_limit,
                                      cfg.xx_scan, cfg.accept),
        lambda: mut_remove_crossing(lay, rng),
        lambda: mut_add_dbl_crossover(lay, table, rng, cfg.dc_attempts, cfg.dc_limit,
                                      cfg.dc_scan, cfg.accept),
        lambda: mut_remove_dbl_crossover(lay, rng, cfg.dc_refill),
    ]
    if cfg.mutate_draws is None:
        order = list(range(len(ops)))
        rng.shuffle(order)
    else:
        order = rng.choices(range(len(ops)), weights=cfg.weights, k=cfg.mutate_draws)
    for k in order:
        out = ops[k]()
        if out is not None and out.is_matched() and out.is_planar_closed():
            return out
    return lay


def crossover(a: Layout, b: Layout, table: TransformTable, rng: random.Random,
              cfg: Settings = Settings()) -> Layout:
    """Wymiana galezi: przenies jedna galaz z b do a.

    Prototyp: dla ukladow bez galezi degeneruje sie do wymiany fragmentu
    lancucha o tym samym przeksztalceniu netto. Fragment z a ma do `max_win`
    klockow, fragment z b do glebokosci tablicy - tak samo jak podmiana.
    """
    ca = [c for c in all_chains(a) if len(c[1]) >= cfg.cx_min_chain]
    cb = [c for c in all_chains(b) if len(c[1]) >= cfg.cx_min_chain]
    if not ca or not cb:
        return a
    for _ in range(cfg.cx_tries):
        ha, ida, ta, tla = rng.choice(ca)
        _, _, tb, _ = rng.choice(cb)
        na, nb = len(ta), len(tb)
        wa = rng.randint(1, min(cfg.max_win, na))
        aa = rng.randrange(na - wa + 1)
        seg = ta[aa:aa + wa]
        t = net_transform(seg)
        # szukaj w b fragmentu o tym samym przeksztalceniu
        for wb in range(1, min(table.max_len, nb) + 1):
            for ab in range(nb - wb + 1):
                cand = tb[ab:ab + wb]
                tc = net_transform(cand)
                if (abs(tc[0] - t[0]) < cfg.cx_tol and abs(tc[1] - t[1]) < cfg.cx_tol
                        and tc[2] == t[2] and cand != seg):
                    full = ta[:aa] + cand + ta[aa + wa:]
                    out = (Layout.from_cycle(full) if ha is None
                           else _rebuild_chain(a, ida, full, ha, tla))
                    if out.is_matched() and out.is_planar_closed():
                        return out
    return a


# --- skrzyzowania -----------------------------------------------------------

def crossing_connectors(max_n: int = 12, straights=("S16", "S24")):
    """Lancuchy o przeksztalceniu netto dokladnie (-16, 0, 0).

    Tyle wlasnie wymaga polaczenie portow 2 i 3 skrzyzowania: jego os
    poprzeczna zachowuje sie jak S16, wiec domkniecie drugiej petli musi
    "cofnac" o jedna prosta. Tego NIE da sie znalezc w tablicy - najkrotsze
    rozwiazanie ma 21 elementow, a tablica siega 7. Ale rodzina jest znana
    analitycznie: cykliczny owal minus jedna prosta 16.

        [S16]*(a-1) + [C]*8 + <prosty odcinek tej samej dlugosci> + [C]*8

    Zwraca liste posortowana rosnaco po dlugosci.
    """
    out = []
    for c in ("CL", "CR"):
        for a in range(1, max_n + 1):
            run = ["S16"] * a
            out.append(tuple(run[1:] + [c] * 8 + run + [c] * 8))
            if "S24" in straights:
                # oba odcinki tej samej dlugosci, ale z prostych 24
                for b in range(1, max_n // 2 + 1):
                    if 24 * b % 16:
                        continue
                    k = 24 * b // 16
                    if k < 1:
                        continue
                    out.append(tuple(["S16"] * (k - 1) + [c] * 8 +
                                     ["S24"] * b + [c] * 8))
    return sorted(set(out), key=len)


_XCONN = None


def mut_add_crossing_pair(lay: Layout, table: TransformTable, rng: random.Random,
                          attempts: int = 12, limit: int = 2, scan: int = 1000,
                          accept: str = "first") -> Layout | None:
    """Wstaw PARE skrzyzowan i przeprowadz przez nie druga zamknieta petle.

    Skrzyzowania musza byc parzyste z tego samego powodu co rozjazdy, tylko
    dowod jest inny: dwie zamkniete krzywe na plaszczyznie przecinaja sie
    PARZYSTA liczbe razy. Pojedyncze skrzyzowanie zostawia wiec zawsze jedno
    nielegalne przeciecie - sprawdzone wyczerpujaco, minimum kolizji = 4.

    Druga petla wchodzi w XX_i portem 2, wychodzi portem 3, biegnie do XX_j
    i wraca. Osie skrzyzowan sa niezalezne, wiec pociag nie przechodzi
    miedzy petlami - to dwie osobne trasy dzielace tylko miejsce.
    """
    cand_ids = [i for i, t in enumerate(lay.types) if t == "S16"]
    if len(cand_ids) < 2:
        return None
    T0, mism0 = lay.place()
    if mism0 >= POS_TOL:
        return None

    REV = (STRAIGHT_LEN, 0.0, 8)
    base_ov = lay.overlap()
    base_rt = lay.count_routes()
    best, best_score = None, None

    for _ in range(attempts):
        i, j = rng.sample(cand_ids, 2)
        if T0[i] is None or T0[j] is None:
            continue
        for ri, rj in ((0, 0), (0, 1), (1, 0), (1, 1)):
            Ti = compose(T0[i], REV) if ri else T0[i]
            Tj = compose(T0[j], REV) if rj else T0[j]
            P2i = compose(Ti, port_pose("XX", 2))
            P3i = compose(Ti, port_pose("XX", 3))
            P2j = compose(Tj, port_pose("XX", 2))
            P3j = compose(Tj, port_pose("XX", 3))
            N1 = compose(inverse(P3i), compose(P2j, (0.0, 0.0, 8)))
            N2 = compose(inverse(P3j), compose(P2i, (0.0, 0.0, 8)))

            c1 = table.connectors(N1, limit=limit, scan=scan, rng=rng)
            if not c1:
                continue
            c2 = table.connectors(N2, limit=limit, scan=scan, rng=rng)
            if not c2:
                continue

            base = put_switch(lay, i, "XX", bool(ri))
            base = put_switch(base, j, "XX", bool(rj))
            for s1 in c1:
                for s2 in c2:
                    out = base.copy()
                    prev = (i, 3)
                    for e in s1:
                        k = out.add(e)
                        out.link(prev, (k, 0))
                        prev = (k, 1)
                    out.link(prev, (j, 2))
                    prev = (j, 3)
                    for e in s2:
                        k = out.add(e)
                        out.link(prev, (k, 0))
                        prev = (k, 1)
                    out.link(prev, (i, 2))
                    if not (out.is_matched() and out.is_planar_closed()):
                        continue
                    ov, rt = out.overlap(), out.count_routes()
                    score = (rt - base_rt, -(ov - base_ov))
                    if best is None or score > best_score:
                        best, best_score = out, score
                    if accept == "first" and score[0] > 0 and ov <= base_ov:
                        return out
    return best


def cross_loop(lay: Layout, i: int):
    """Petla biegnaca osiami poprzecznymi od (i, 2) z powrotem do (i, 3).

    Na kazdym napotkanym skrzyzowaniu przechodzi na jego druga os (2 <-> 3).
    Zwraca (ids klockow lancuchow, ids skrzyzowan) albo None, gdy petla
    trafi na wezel inny niz os poprzeczna skrzyzowania.
    """
    ids, crossings, cur = [], [i], (i, 2)
    while True:
        r = chain_from(lay, cur)
        if r is None:
            return None
        seg, _, end = r
        ids += seg
        if end == (i, 3):
            return ids, crossings
        k, p = end
        if lay.types[k] != "XX" or p not in (2, 3):
            return None
        crossings.append(k)
        cur = (k, 5 - p)


def mut_remove_crossing(lay: Layout, rng: random.Random) -> Layout | None:
    """Usun petle wiszaca na osiach poprzecznych i wszystkie jej skrzyzowania.

    Kazde takie skrzyzowanie wraca do postaci S16: porty 0/1 leza dokladnie
    tam, gdzie porty prostej, wiec reszta ukladu sie nie rusza.
    """
    xs = [i for i, t in enumerate(lay.types) if t == "XX"]
    if not xs:
        return None
    rng.shuffle(xs)
    for i in xs:
        r = cross_loop(lay, i)
        if r is None:
            continue
        ids, crossings = r
        new = lay.copy()
        for k in ids:
            for p in range(ELEMENTS[new.types[k]].n_ports):
                new.unlink((k, p))
            new.types[k] = None
        for k in crossings:
            new.unlink((k, 2))
            new.unlink((k, 3))
            new.types[k] = "S16"
        out = new.compact()
        if out.is_matched():
            return out
    return None


# --- podwojny rozjazd krzyzowy ---------------------------------------------

def put_dc(lay: Layout, ids, p: int, q: int, rev: bool) -> Layout | None:
    """Podmien ciag prostych `ids` na klocek DC w slocie ids[0].

    Porty 0/1 klocka przejmuja polaczenia z reszta ukladu (`rev` zamienia
    je miejscami), porty 2/3 zostaja wolne. Pozostale klocki ciagu zostaja
    osierocone - caller MUSI na koncu wywolac compact(). Wzor: put_switch_pair.
    """
    A = lay.match.get((ids[0], 1 - p))
    B = lay.match.get((ids[-1], 1 - q))
    if A is None or B is None:
        return None
    new = lay.copy()
    for e in ids:
        new.unlink((e, 0))
        new.unlink((e, 1))
    new.types[ids[0]] = "DC"
    new.link(A, (ids[0], 1 if rev else 0))
    new.link(B, (ids[0], 0 if rev else 1))
    return new


def mut_add_dbl_crossover(lay: Layout, table: TransformTable, rng: random.Random,
                          attempts: int = 12, limit: int = 8, scan: int = 4000,
                          accept: str = "first") -> Layout | None:
    """Wstaw podwojny rozjazd krzyzowy i domknij jego drugi tor osobna petla.

    Tor 1 klocka zastepuje 48 studow prostych i przenosi tak samo jak one,
    wiec reszta ukladu sie nie rusza. Porty 3 -> 2 laczy lancuch z tablicy
    (jak w mut_add_crossing_pair): powstaje druga zamknieta petla, a skosy
    klocka daja nowe trasy przez obie. Potrzebne przesuniecie wynika z samego
    klocka, wiec jest jedno dla kazdego polozenia. Wszystkie cztery porty
    dostaja partnera za jednym razem, wiec jedna sztuka wystarczy. Drugi tor
    moze lezec po obu stronach toru 1 (`rev`); zla strona wjezdza w reszte
    ukladu i odpada na overlap().
    """
    runs = straight_runs(lay, ELEM_LEN["DC"])
    if not runs:
        return None
    N = compose(inverse(port_pose("DC", 3)), compose(port_pose("DC", 2), (0.0, 0.0, 8)))
    seqs = table.connectors(N, limit=limit, scan=scan, rng=rng)
    if not seqs:
        return None
    base_ov = lay.overlap()
    base_rt = lay.count_routes()
    best, best_score = None, None

    for _ in range(attempts):
        ids, p, q = rng.choice(runs)
        i = ids[0]
        for rev in (False, True):
            base = put_dc(lay, ids, p, q, rev)
            if base is None:
                continue
            for seq in seqs:
                out = base.copy()
                prev = (i, 3)
                for e in seq:
                    k = out.add(e)
                    out.link(prev, (k, 0))
                    prev = (k, 1)
                out.link(prev, (i, 2))
                out = out.compact()
                if not (out.is_matched() and out.is_planar_closed()):
                    continue
                ov, rt = out.overlap(), out.count_routes()
                score = (rt - base_rt, -(ov - base_ov))
                if best is None or score > best_score:
                    best, best_score = out, score
                if accept == "first" and score[0] > 0 and ov <= base_ov:
                    return out
    return best


def mut_remove_dbl_crossover(lay: Layout, rng: random.Random,
                             refill: str = "random") -> Layout | None:
    """Usun podwojny rozjazd krzyzowy razem z cala skladowa na jego drugim
    torze (takze wezlami, ktore na niej urosly); tor 1 dostaje z powrotem
    48 studow prostych: trzy S16 albo dwie S24, losowo. DC, ktorego drugi
    tor wraca do toru glownego, jest pomijany. Odpowiednik mut_remove_crossing."""
    xs = [i for i, t in enumerate(lay.types) if t == "DC"]
    if not xs:
        return None
    rng.shuffle(xs)
    for i in xs:
        new = lay.copy()
        start = new.unlink((i, 2))
        new.unlink((i, 3))
        if start is None:
            continue
        ids, stack = set(), [start[0]]
        while stack:
            u = stack.pop()
            if u in ids:
                continue
            ids.add(u)
            stack += [new.match[(u, p)][0]
                      for p in range(ELEMENTS[new.types[u]].n_ports)
                      if (u, p) in new.match]
        if i in ids:
            continue
        for k in ids:
            for p in range(ELEMENTS[new.types[k]].n_ports):
                new.unlink((k, p))
            new.types[k] = None
        b_side = new.match.get((i, 1))
        new.unlink((i, 1))
        fill = ("S16",) * 3 if refill == "s16" else rng.choice((("S16",) * 3, ("S24",) * 2))
        new.types[i] = fill[0]
        prev = (i, 1)
        for t in fill[1:]:
            k = new.add(t)
            new.link(prev, (k, 0))
            prev = (k, 1)
        if b_side is not None:
            new.link(prev, b_side)
        out = new.compact()
        if out.is_matched():
            return out
    return None
