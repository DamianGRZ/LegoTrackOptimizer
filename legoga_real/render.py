"""Rysowanie wynikow prototypu: podsypka, szyny, zlacza i znaczniki w matplotlib.

Uklad to graf polaczen portow, wiec kazdy FIZYCZNY klocek rysowany jest dokladnie
raz, z wlasnej ramki wyznaczonej przez place(). Wymiary pochodza z geometry.py,
tutaj sa tylko style rysunku.
"""
import math
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.patches import Patch, Polygon

from .geometry import (
    ANG, CURVE_R, CURVE_UNITS, DC_LEN, DC_SPACING, STRAIGHT_LEN, STRAIGHT_LONG_LEN,
    SWITCH_BODY, TRACK_WIDTH, _SW_A1, apply, compose, port_pose,
)

# Bez interfejsu: tylko pliki PNG (Tk i watki Pool wywracaja Tcl).
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402

# kolor na fizyczny klocek
COLORS = {"S16": "#3498db", "S24": "#2980b9", "R40": "#27ae60", "XX": "#9b59b6",
          "WL": "#e74c3c", "WR": "#e67e22", "DC": "#8e44ad"}
# typ -> fizyczny klocek; CL i CR to ten sam klocek ulozony w dwie strony
PHYSICAL = {"CL": "R40", "CR": "R40"}
# (fizyczny klocek, etykieta) w kolejnosci wierszy legendy
LEGEND = (("S16", "STRAIGHT_16"), ("S24", "STRAIGHT_24"), ("R40", "R40_CURVE"),
          ("XX", "CROSS_90"), ("WL", "R40_SWITCH_LEFT"), ("WR", "R40_SWITCH_RIGHT"),
          ("DC", "DOUBLE_CROSSOVER"))
MARKER_ANCHOR = {"WL": (16.0, 0.0), "WR": (16.0, 0.0), "XX": (8.0, 0.0)}

CURVE_ANGLE = 360.0 / ANG * CURVE_UNITS       # stopnie na jeden luk
ARC1_ANGLE = math.degrees(_SW_A1)             # pierwszy luk odgalezienia rozjazdu
ARC2_ANGLE = ARC1_ANGLE - CURVE_ANGLE         # drugi luk wraca do kursu wyjscia

RAIL_OFFSET = 2.5                # polowa rozstawu szyn (study)
COL_RAIL = "#95a5a6"
RAIL_LW = 1.8
N_ARC_PTS = 60
SWITCH_ARC_PTS = 30              # odgalezienie rozjazdu jest krotkie, wystarczy mniej punktow
SWITCH_BRANCH_WIDTH_STUD = 6.4   # wezsza podsypka, zeby odgalezienie odroznialo sie od prostej
DC_CROSS_WIDTH_STUD = 3.6        # cienka podsypka skosow podwojnego rozjazdu
AXIS_LABEL_FONTSIZE = 11

# Zlacze ma kolor tla, zeby klocki dalo sie policzyc okiem. Szerokosc w punktach,
# nie w studach: kreska musi byc widoczna przy kazdym powiekszeniu.
JOINT_COLOR = "white"
JOINT_LW = 1.2


def arc_points(cx, cy, r, start_deg, sweep_deg, n=N_ARC_PTS):
    """Tablice (x, y) wzdluz luku okregu."""
    angles = np.linspace(np.radians(start_deg),
                         np.radians(start_deg + sweep_deg), n)
    return cx + r * np.cos(angles), cy + r * np.sin(angles)


def offset_path(x, y, offset):
    """Lamana przesunieta o +-offset studow prostopadle do kierunku drogi."""
    dx = np.gradient(x)
    dy = np.gradient(y)
    length = np.sqrt(dx**2 + dy**2)
    length[length == 0] = 1e-9
    nx = -dy / length
    ny = dx / length
    return x + offset * nx, y + offset * ny


def _draw_track_bed(ax, x, y, color, alpha=0.88, zorder=2, width_stud=None):
    """Podsypka jako wielokat wokol drogi; szerokosc w studach, domyslnie TRACK_WIDTH."""
    half = float(TRACK_WIDTH if width_stud is None else width_stud) / 2.0
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.size < 2:
        return
    lx, ly = offset_path(x_arr, y_arr, +half)
    rx, ry = offset_path(x_arr, y_arr, -half)
    verts = list(zip(lx, ly)) + list(zip(rx[::-1], ry[::-1]))
    ax.add_patch(Polygon(verts, closed=True, facecolor=color,
                         edgecolor='none', alpha=alpha, zorder=zorder))


def _draw_joint(ax, x, y, theta_deg):
    """Kreska w poprzek podsypki na granicy klockow, prostopadla do kursu."""
    perp = np.radians(theta_deg + 90.0)
    half = TRACK_WIDTH / 2.0
    dx, dy = half * np.cos(perp), half * np.sin(perp)
    ax.plot([x - dx, x + dx], [y - dy, y + dy], color=JOINT_COLOR,
            lw=JOINT_LW, solid_capstyle='butt', zorder=4)


def _draw_rails(ax, x, y, offset=RAIL_OFFSET, zorder=3):
    """Dwie szyny po obu stronach osi."""
    lx, ly = offset_path(x, y, +offset)
    rx, ry = offset_path(x, y, -offset)
    ax.plot(lx, ly, color=COL_RAIL, lw=RAIL_LW, solid_capstyle='butt', zorder=zorder)
    ax.plot(rx, ry, color=COL_RAIL, lw=RAIL_LW, solid_capstyle='butt', zorder=zorder)


def _local_to_world(x0, y0, theta0):
    """Funkcja przeliczajaca wspolrzedne klocka (lx, ly) na wspolrzedne rysunku."""
    theta_rad = np.radians(theta0)
    cos_t, sin_t = np.cos(theta_rad), np.sin(theta_rad)

    def to_world(lx, ly):
        return (x0 + lx * cos_t - ly * sin_t, y0 + lx * sin_t + ly * cos_t)

    return to_world


def _draw_segment(ax, p, q, color, **bed_kw):
    """Prosty odcinek podsypki z szynami miedzy punktami p i q."""
    xs = np.array([p[0], q[0]])
    ys = np.array([p[1], q[1]])
    _draw_track_bed(ax, xs, ys, color, **bed_kw)
    _draw_rails(ax, xs, ys)


def _draw_arc(ax, cx, cy, radius, start_ang, sweep_deg, color, n=N_ARC_PTS,
              width_stud=None, alpha=0.88):
    """Luk podsypki z szynami wokol srodka (cx, cy).

    Podsypka to pierscien miedzy dwoma wspolsrodkowymi lukami, a nie przesuniecie
    osi: normalne z gradientu sa jednostronne na koncach lamanej i zostawialyby
    klin tla na styku dwoch lukow. Krawedzie pierscienia leza dokladnie na promieniach.
    """
    half = float(TRACK_WIDTH if width_stud is None else width_stud) / 2.0
    x_out, y_out = arc_points(cx, cy, radius + half, start_ang, sweep_deg, n=n)
    x_in, y_in = arc_points(cx, cy, radius - half, start_ang, sweep_deg, n=n)
    verts = list(zip(x_out, y_out)) + list(zip(x_in[::-1], y_in[::-1]))
    ax.add_patch(Polygon(verts, closed=True, facecolor=color,
                         edgecolor='none', alpha=alpha, zorder=2))
    xi, yi = arc_points(cx, cy, radius - RAIL_OFFSET, start_ang, sweep_deg, n=n)
    xo, yo = arc_points(cx, cy, radius + RAIL_OFFSET, start_ang, sweep_deg, n=n)
    ax.plot(xi, yi, color=COL_RAIL, lw=RAIL_LW, solid_capstyle='butt', zorder=3)
    ax.plot(xo, yo, color=COL_RAIL, lw=RAIL_LW, solid_capstyle='butt', zorder=3)


def _draw_straight_piece(ax, x0, y0, theta0, length, color):
    """Prosta o dlugosci `length` od (x0, y0) w kierunku `theta0` (stopnie)."""
    theta_rad = np.radians(theta0)
    x1 = x0 + length * np.cos(theta_rad)
    y1 = y0 + length * np.sin(theta_rad)
    _draw_segment(ax, (x0, y0), (x1, y1), color)


def _draw_curve_piece(ax, x0, y0, theta0, radius, sweep_deg, color):
    """Luk o promieniu `radius`; dodatni `sweep_deg` skreca w lewo, ujemny w prawo."""
    # Srodek luku lezy prostopadle do kursu wejscia, po stronie skretu.
    center_offset_angle = theta0 + 90 if sweep_deg > 0 else theta0 - 90
    center_offset_rad = np.radians(center_offset_angle)
    cx = x0 + radius * np.cos(center_offset_rad)
    cy = y0 + radius * np.sin(center_offset_rad)

    start_ang = np.degrees(np.arctan2(y0 - cy, x0 - cx))

    _draw_arc(ax, cx, cy, radius, start_ang, sweep_deg, color)


def _draw_switch_piece(ax, x0, y0, theta0, direction, color):
    """Rozjazd: droga glowna i odgalezienie z dwoch lukow; `direction` to 'left' albo 'right'."""
    tw = _local_to_world(x0, y0, theta0)

    _draw_segment(ax, (x0, y0), tw(SWITCH_BODY, 0.0), color)

    flip = 1 if direction == 'left' else -1

    # Pierwszy luk: na zewnatrz.
    c1x, c1y = tw(0.0, flip * CURVE_R)
    a1_start = theta0 - 90 * flip
    a1_sweep = flip * ARC1_ANGLE
    _draw_arc(ax, c1x, c1y, CURVE_R, a1_start, a1_sweep, color,
              n=SWITCH_ARC_PTS, width_stud=SWITCH_BRANCH_WIDTH_STUD)

    # Punkt styku na koncu pierwszego luku.
    a1_end_rad = np.radians(a1_start + a1_sweep)
    jx = c1x + CURVE_R * np.cos(a1_end_rad)
    jy = c1y + CURVE_R * np.sin(a1_end_rad)
    j_heading = theta0 + flip * ARC1_ANGLE

    # Drugi luk: z powrotem, w przeciwna strone.
    j_heading_rad = np.radians(j_heading)
    perp_dir = j_heading_rad - flip * np.pi / 2
    c2x = jx + CURVE_R * np.cos(perp_dir)
    c2y = jy + CURVE_R * np.sin(perp_dir)
    a2_start = np.degrees(np.arctan2(jy - c2y, jx - c2x))
    a2_sweep = -flip * ARC2_ANGLE
    _draw_arc(ax, c2x, c2y, CURVE_R, a2_start, a2_sweep, color,
              n=SWITCH_ARC_PTS, width_stud=SWITCH_BRANCH_WIDTH_STUD)


def _draw_cross90_piece(ax, x0, y0, theta0, color):
    """Skrzyzowanie: dwie prostopadle osie przecinajace sie w polowie."""
    tw = _local_to_world(x0, y0, theta0)
    half = STRAIGHT_LEN / 2

    _draw_segment(ax, tw(0, 0), tw(STRAIGHT_LEN, 0), color)
    _draw_segment(ax, tw(half, -half), tw(half, half), color)


def _draw_double_crossover_piece(ax, x0, y0, theta0, color):
    """Podwojny rozjazd krzyzowy: dwa rownolegle tory i dwa skosy miedzy nimi.

    Tory biegna przez cala dlugosc, skosy sa cienkie i zajmuja tylko srodkowa
    jedna trzecia, wiec klocek czyta sie jak dwa tory i jedno krzyzowanie. Cztery
    porty sa zaznaczone wprost.
    """
    tw = _local_to_world(x0, y0, theta0)
    length, lat = DC_LEN, DC_SPACING
    start, end = length / 3, 2 * length / 3

    _draw_segment(ax, tw(0.0, 0.0), tw(length, 0.0), color)
    _draw_segment(ax, tw(0.0, lat), tw(length, lat), color)

    _draw_segment(ax, tw(start, 0.0), tw(end, lat), color,
                  width_stud=DC_CROSS_WIDTH_STUD, alpha=0.75)
    _draw_segment(ax, tw(start, lat), tw(end, 0.0), color,
                  width_stud=DC_CROSS_WIDTH_STUD, alpha=0.75)

    for px, py in ((0.0, 0.0), (length, 0.0), (0.0, lat), (length, lat)):
        wx, wy = tw(px, py)
        ax.plot(wx, wy, marker='o', ms=4.5, mfc=color, mec='white',
                mew=0.7, zorder=6)


_PIECE_DRAWERS = {
    "S16": lambda ax, x0, y0, th, c: _draw_straight_piece(ax, x0, y0, th, STRAIGHT_LEN, c),
    "S24": lambda ax, x0, y0, th, c: _draw_straight_piece(ax, x0, y0, th, STRAIGHT_LONG_LEN, c),
    "CL": lambda ax, x0, y0, th, c: _draw_curve_piece(ax, x0, y0, th, CURVE_R, CURVE_ANGLE, c),
    "CR": lambda ax, x0, y0, th, c: _draw_curve_piece(ax, x0, y0, th, CURVE_R, -CURVE_ANGLE, c),
    "WL": lambda ax, x0, y0, th, c: _draw_switch_piece(ax, x0, y0, th, 'left', c),
    "WR": lambda ax, x0, y0, th, c: _draw_switch_piece(ax, x0, y0, th, 'right', c),
    "XX": _draw_cross90_piece,
    "DC": _draw_double_crossover_piece,
}


def _draw_piece(ax, tname, x0, y0, theta):
    """Jeden klocek typu `tname` z ramki (x0, y0, theta); theta w stopniach."""
    color = COLORS[PHYSICAL.get(tname, tname)]
    _PIECE_DRAWERS[tname](ax, x0, y0, theta, color)


def _junction_marker(physical):
    """Ksztalt znacznika: 'D' rozjazd, 's' skrzyzowanie, None dla pozostalych."""
    if physical in ("WL", "WR"):
        return "D"
    return "s" if physical == "XX" else None


def _draw_boundary(ax, max_size):
    """Granica stolu jako przerywany prostokat wokol srodka rysunku."""
    width, height = max_size
    half_w, half_h = width / 2.0, height / 2.0
    rect_x = [-half_w, half_w, half_w, -half_w, -half_w]
    rect_y = [-half_h, -half_h, half_h, half_h, -half_h]
    ax.plot(rect_x, rect_y, "k--", linewidth=1, alpha=0.5)


def _setup_axes(ax):
    """Rowna skala osi, podpisy w studach i lekka siatka."""
    ax.set_aspect("equal")
    ax.set_xlabel("X (studs)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Y (studs)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.grid(True, alpha=0.3)


def draw_layout(ax, lay, max_size):
    """Kazdy fizyczny klocek raz, z jego ramki wyznaczonej przez place(),
    obrocony tak, jak ocenial go GA (najlepsze dopasowanie do stolu)."""
    k, _, _ = lay.fit(max_size)
    T = [t if t is None else compose((0.0, 0.0, k), t) for t in lay.place()[0]]
    xs, ys = [], []
    for t_frame in T:
        if t_frame is not None:
            xs.append(t_frame[0])
            ys.append(t_frame[1])
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0

    def world(t_frame):
        return (t_frame[0] - cx, t_frame[1] - cy, (t_frame[2] % 16) * 22.5)

    for i, tname in enumerate(lay.types):
        if T[i] is None:
            continue
        x0, y0, theta = world(T[i])
        _draw_piece(ax, tname, x0, y0, theta)

    # zlacza: jedna kreska na kazda pare polaczonych portow
    seen = set()
    for (i, p), (j, q) in lay.match.items():
        key = frozenset(((i, p), (j, q)))
        if key in seen or T[i] is None:
            continue
        seen.add(key)
        px, py, pd = compose(T[i], port_pose(lay.types[i], p))
        _draw_joint(ax, px - cx, py - cy, ((pd + 8) % 16) * 22.5)

    # znaczniki: romb na rozjezdzie, kwadrat na srodku skrzyzowania
    for i, tname in enumerate(lay.types):
        anchor = MARKER_ANCHOR.get(tname)
        if anchor is None or T[i] is None:
            continue
        physical = PHYSICAL.get(tname, tname)
        mx, my = apply(T[i], anchor)
        ax.plot(mx - cx, my - cy, _junction_marker(physical), color=COLORS[physical],
                markersize=9, markeredgecolor="black", markeredgewidth=1.2, zorder=8)

    _draw_boundary(ax, max_size)
    _setup_axes(ax)


def legend_handles(lay, inventory):
    counts = {}
    for tname in lay.types:
        physical = PHYSICAL.get(tname, tname)
        counts[physical] = counts.get(physical, 0) + 1
    handles = []
    for physical, name in LEGEND:
        label = f"{name:<17}{counts.get(physical, 0):>3}/{inventory.get(physical, 0)}"
        color = COLORS[physical]
        marker = _junction_marker(physical)
        if marker is None:
            handles.append(Patch(facecolor=color, edgecolor="black", label=label))
        else:
            handles.append(plt.Line2D([0], [0], marker=marker, color=color,
                                      linestyle="-", linewidth=3, markersize=6,
                                      markeredgecolor="black", label=label))
    return handles


def save_gallery(rows, path, max_size, cols: int = 3):
    """Wszystkie uklady na jednym PNG, bez legendy; numer w tytule panelu
    to numer pliku z save_layouts. Wiersz to (lay, elem, trasy, kolizje, rozmiar)."""
    n_rows = (len(rows) + cols - 1) // cols
    fig, axes = plt.subplots(n_rows, cols, figsize=(8 * cols, 7.5 * n_rows))
    axes = list(np.atleast_1d(axes).flatten())
    for k, (ax, (lay, used, routes, ov, size)) in enumerate(zip(axes, rows), 1):
        draw_layout(ax, lay, max_size)
        ax.set_title(f"#{k}: {used:.0f} elem, {routes:.0f} tras, kolizje {ov:.0f}, "
                     f"{size[0]:.0f}x{size[1]:.0f} studow", fontsize=12)
    for ax in axes[len(rows):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


def save_layouts(rows, path, inventory, max_size):
    """Kazdy uklad na osobnym PNG `<path>_<k>`, z legenda na prawo od rysunku.
    Zwraca liste sciezek."""
    stem, suffix = Path(path).with_suffix(""), Path(path).suffix or ".png"
    paths = []
    for k, (lay, used, routes, ov, size) in enumerate(rows, 1):
        fig, ax = plt.subplots(figsize=(9, 8.5))
        draw_layout(ax, lay, max_size)
        ax.set_title(f"#{k}: {used:.0f} elem, {routes:.0f} tras, kolizje {ov:.0f}, "
                     f"{size[0]:.0f}x{size[1]:.0f} studow", fontsize=12)
        ax.legend(handles=legend_handles(lay, inventory), loc="upper left",
                  bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0,
                  prop={"family": "monospace", "size": 8})
        out = f"{stem}_{k}{suffix}"
        fig.savefig(out, dpi=110, bbox_inches="tight")
        plt.close(fig)
        paths.append(out)
    return paths
