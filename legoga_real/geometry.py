"""
SE(2) geometry + definicje elementow toru.

Konwencje
---------
* Jednostka dlugosci: stud (1 stud = 8 mm).
* Kat jest liczba calkowita mod 16; 1 jednostka = 22.5 stopnia.
  Dzieki temu skladanie obrotow jest DOKLADNE (bez kumulacji bledu float),
  a tylko pozycje sa zmiennoprzecinkowe.
* Poza (pose) = (x, y, a) gdzie a in Z_16.
* Port ma pozycje i kierunek ZEWNETRZNY (outward) - tam, gdzie "wystaje" tor.
  Port 0 kazdego elementu jest w (0,0) i patrzy w kierunku 8 (czyli -x).

!!!  UWAGA - LICZBY DO PODMIANY  !!!
Geometria SWITCH_* i CROSS ponizej to PRZYBLIZENIE (rozjazd traktowany jako
prosta-16 z odgalezieniem rownym jednemu lukowi). Rzeczywiste LEGO ma inne
wymiary. Przed uzyciem produkcyjnym podmien wartosci na dane z biblioteki
czesci BlueBrick / TrixBrix. Cala geometria siedzi w tym jednym pliku.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

# --- stale geometryczne -----------------------------------------------------

ANG = 16                      # liczba jednostek kata w pelnym obrocie
STRAIGHT_LEN = 16.0           # prosta standardowa LEGO
STRAIGHT_LONG_LEN = 24.0      # prosta 24 (4DBrix 2.04.066) - realna czesc kitu
CURVE_R = 40.0                # promien luku LEGO (study)
CURVE_UNITS = 1               # luk = 1 jednostka kata = 22.5 stopnia

# Rozjazd R40 (data/track_pieces_v2.yaml, 4DBrix 2.04.021/018): korpus 32
# study wzdluz drogi glownej, zmierzony port odgalezienia (32.75, +-13.0)
# z kursem wyjscia 22.5 st. Odgalezienie jest lukiem R40 na calej dlugosci:
# dwa luki 3-4-5 (+36.87 st., potem -14.37 st.).
SWITCH_BODY = 32.0
SWITCH_C_X, SWITCH_C_Y = 32.75, 13.0

TRACK_WIDTH = 8.0             # szerokosc toru - do detekcji kolizji


_A = 2.0 * math.pi / ANG      # radiany na jednostke
_SW_A1 = math.asin(0.6)       # kat pierwszego luku odgalezienia (3-4-5)
# dlugosc osi kazdego elementu (study) - do liczenia odleglosci PO TORZE
ARC_LEN = CURVE_R * _A * CURVE_UNITS
ELEM_LEN = {"S16": STRAIGHT_LEN, "S24": STRAIGHT_LONG_LEN,
            "CL": ARC_LEN, "CR": ARC_LEN,
            "WL": SWITCH_BODY, "WR": SWITCH_BODY, "XX": STRAIGHT_LEN}


def rad(a: int) -> float:
    return (a % ANG) * _A


# tablice trygonometryczne - katy sa dyskretne, wiec liczymy je raz
_COS = tuple(math.cos(k * _A) for k in range(ANG))
_SIN = tuple(math.sin(k * _A) for k in range(ANG))


# --- SE(2) ------------------------------------------------------------------

def compose(t1, t2):
    """Zloz przeksztalcenia: najpierw t1, potem t2 (w ukladzie t1)."""
    x1, y1, a1 = t1
    x2, y2, a2 = t2
    a1 %= ANG
    c, s = _COS[a1], _SIN[a1]
    return (x1 + c * x2 - s * y2,
            y1 + s * x2 + c * y2,
            (a1 + a2) % ANG)


def inverse(t):
    x, y, a = t
    a %= ANG
    c, s = _COS[a], _SIN[a]
    return (-(c * x + s * y), -(-s * x + c * y), (-a) % ANG)


def apply(t, p):
    """Przeksztalc punkt (x, y)."""
    x, y, a = t
    a %= ANG
    c, s = _COS[a], _SIN[a]
    return (x + c * p[0] - s * p[1], y + s * p[0] + c * p[1])


IDENT = (0.0, 0.0, 0)


# --- elementy ---------------------------------------------------------------

@dataclass(frozen=True)
class ElementType:
    name: str
    ports: tuple                  # ((x, y, dir_out), ...)
    routes: tuple                 # ((portA, portB), ...) - przejezdne pary
    centerline: tuple             # ((portA, portB, [punkty...]), ...) do kolizji
    is_node: bool = False         # czy to wierzchol grafu zredukowanego

    @property
    def n_ports(self) -> int:
        return len(self.ports)


def _arc_points(sign: int, n: int = 4):
    """Punkty na osi luku od portu 0 do portu 1."""
    pts = []
    for i in range(n + 1):
        b = rad(CURVE_UNITS) * i / n
        pts.append((CURVE_R * math.sin(b), sign * CURVE_R * (1 - math.cos(b))))
    return tuple(pts)


def _line_points(x0, y0, x1, y1, n: int = 3):
    return tuple(((x0 + (x1 - x0) * i / n), (y0 + (y1 - y0) * i / n))
                 for i in range(n + 1))


_ARC_END_X = CURVE_R * math.sin(rad(CURVE_UNITS))
_ARC_END_Y = CURVE_R * (1 - math.cos(rad(CURVE_UNITS)))

ELEMENTS: dict[str, ElementType] = {}


def _reg(e: ElementType) -> ElementType:
    ELEMENTS[e.name] = e
    return e


S16 = _reg(ElementType(
    name="S16",
    ports=((0.0, 0.0, 8), (STRAIGHT_LEN, 0.0, 0)),
    routes=((0, 1),),
    centerline=((0, 1, _line_points(0, 0, STRAIGHT_LEN, 0)),),
))

S24 = _reg(ElementType(
    name="S24",
    ports=((0.0, 0.0, 8), (STRAIGHT_LONG_LEN, 0.0, 0)),
    routes=((0, 1),),
    centerline=((0, 1, _line_points(0, 0, STRAIGHT_LONG_LEN, 0, 5)),),
))

CL = _reg(ElementType(
    name="CL",
    ports=((0.0, 0.0, 8), (_ARC_END_X, _ARC_END_Y, CURVE_UNITS)),
    routes=((0, 1),),
    centerline=((0, 1, _arc_points(+1)),),
))

CR = _reg(ElementType(
    name="CR",
    ports=((0.0, 0.0, 8), (_ARC_END_X, -_ARC_END_Y, -CURVE_UNITS % ANG)),
    routes=((0, 1),),
    centerline=((0, 1, _arc_points(-1)),),
))

# Rozjazd: port 0 = iglica, port 1 = droga prosta, port 2 = odgalezienie.
# Pociag NIE przejedzie 1 <-> 2. Stale geometrii: sekcja na gorze pliku.

def _switch_diverge_points(sign: int, n1: int = 5, n2: int = 4):
    """Os odgalezienia rozjazdu: luk +_SW_A1, luk w przeciwna strone do kursu
    22.5 st., na koncu zmierzony port (SWITCH_C_X, SWITCH_C_Y)."""
    pts = [(CURVE_R * math.sin(_SW_A1 * i / n1),
            sign * CURVE_R * (1.0 - math.cos(_SW_A1 * i / n1)))
           for i in range(n1 + 1)]
    # srodek drugiego luku: koniec pierwszego + promien wzdluz prawej normalnej
    cx = 2.0 * CURVE_R * math.sin(_SW_A1)
    cy = CURVE_R * (1.0 - 2.0 * math.cos(_SW_A1))
    h_end = rad(CURVE_UNITS)
    for i in range(1, n2 + 1):
        h = _SW_A1 + (h_end - _SW_A1) * i / n2    # kurs maleje do 22.5 st.
        pts.append((cx - CURVE_R * math.sin(h),
                    sign * (cy + CURVE_R * math.cos(h))))
    # pochodna 3-4-5 konczy w (32.69, 12.96); port zmierzony jest zrodlem prawdy
    pts.append((SWITCH_C_X, sign * SWITCH_C_Y))
    return tuple(pts)


WL = _reg(ElementType(
    name="WL",
    ports=((0.0, 0.0, 8),
           (SWITCH_BODY, 0.0, 0),
           (SWITCH_C_X, SWITCH_C_Y, CURVE_UNITS)),
    routes=((0, 1), (0, 2)),
    centerline=((0, 1, _line_points(0, 0, SWITCH_BODY, 0, 4)),
                (0, 2, _switch_diverge_points(+1))),
    is_node=True,
))

WR = _reg(ElementType(
    name="WR",
    ports=((0.0, 0.0, 8),
           (SWITCH_BODY, 0.0, 0),
           (SWITCH_C_X, -SWITCH_C_Y, -CURVE_UNITS % ANG)),
    routes=((0, 1), (0, 2)),
    centerline=((0, 1, _line_points(0, 0, SWITCH_BODY, 0, 4)),
                (0, 2, _switch_diverge_points(-1))),
    is_node=True,
))

# Skrzyzowanie 90 st.: dwie NIEZALEZNE sciezki 0<->1 oraz 2<->3.
_H = STRAIGHT_LEN / 2.0
XX = _reg(ElementType(
    name="XX",
    ports=((0.0, 0.0, 8),
           (STRAIGHT_LEN, 0.0, 0),
           (_H, -_H, 12),
           (_H, _H, 4)),
    routes=((0, 1), (2, 3)),
    centerline=((0, 1, _line_points(0, 0, STRAIGHT_LEN, 0)),
                (2, 3, _line_points(_H, -_H, _H, _H))),
    is_node=True,
))

# Elementy 2-portowe uzywane przez generator tablicy przeksztalcen.
CHAIN_ELEMENTS = ("S16", "S24", "CL", "CR")


def advance(name: str):
    """Przeksztalcenie 'wejscie portem 0 -> wyjscie portem 1' dla elementu 2-portowego."""
    e = ELEMENTS[name]
    x, y, d = e.ports[1]
    return (x, y, d)


def other_port(name: str, port: int):
    """Dla elementu 2-portowego: drugi port."""
    return 1 - port


def port_pose(name: str, port: int):
    p = ELEMENTS[name].ports[port]
    return (p[0], p[1], p[2])
