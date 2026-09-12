"""Narysuj wyniki prototypu legoga silnikiem renderera V1 (track_renderer).

Uklad prototypu to graf polaczen portow, wiec nie przechodzi przez
MultiPathLayout (druga, osobna petla jest tam niewyrazalna). Zamiast tego
kazdy FIZYCZNY klocek rysowany jest raz, z jego wlasnej ramki, wspolnym
silnikiem _draw_piece / _draw_joint - podsypka, szyny, kolory i styl V1.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from src.config import BoundaryConfig
from src.encoding import (
    STRAIGHT_16, STRAIGHT_24, R40_CURVE, CROSS_90,
    R40_SWITCH_LEFT, R40_SWITCH_RIGHT,
)
# track_renderer wymusza backend Agg przy imporcie, wiec tu nie trzeba.
from src.visualization.track_renderer import (
    _draw_boundary, _draw_joint, _draw_piece, _junction_marker, _setup_axes,
    get_piece_color,
)

from legoga_real.geometry import apply, compose, port_pose

# typ legoga -> (indeks katalogowy V1, flip dla _draw_piece)
PIECE_MAP = {
    "S16": (STRAIGHT_16, 0),
    "S24": (STRAIGHT_24, 0),
    "CL": (R40_CURVE, 0),
    "CR": (R40_CURVE, 1),
    "WL": (R40_SWITCH_LEFT, 0),
    "WR": (R40_SWITCH_RIGHT, 0),
    "XX": (CROSS_90, 0),
}
LEGEND_NAMES = {
    STRAIGHT_16: "STRAIGHT_16", STRAIGHT_24: "STRAIGHT_24",
    R40_CURVE: "R40_CURVE", CROSS_90: "CROSS_90",
    R40_SWITCH_LEFT: "R40_SWITCH_LEFT", R40_SWITCH_RIGHT: "R40_SWITCH_RIGHT",
}
MARKER_ANCHOR = {"WL": (16.0, 0.0), "WR": (16.0, 0.0), "XX": (8.0, 0.0)}


def draw_layout_v1(ax, lay, boundary):
    """Kazdy fizyczny klocek raz, z jego ramki wyznaczonej przez place()."""
    T, _ = lay.place()
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
        idx, flip = PIECE_MAP[tname]
        _draw_piece(ax, idx, x0, y0, theta, flip=flip)

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
        idx, _ = PIECE_MAP[tname]
        mx, my = apply(T[i], anchor)
        ax.plot(mx - cx, my - cy, _junction_marker(idx), color=get_piece_color(idx),
                markersize=9, markeredgecolor="black", markeredgewidth=1.2, zorder=8)

    _draw_boundary(ax, boundary)
    _setup_axes(ax)


def legend_handles(lay, inventory):
    counts = {}
    for tname in lay.types:
        idx, _ = PIECE_MAP[tname]
        counts[idx] = counts.get(idx, 0) + 1
    cap = {
        STRAIGHT_16: inventory["S16"], STRAIGHT_24: inventory["S24"],
        R40_CURVE: inventory["R40"],
        R40_SWITCH_LEFT: inventory["WL"], R40_SWITCH_RIGHT: inventory["WR"],
        CROSS_90: inventory["XX"],
    }
    handles = []
    for idx in sorted(LEGEND_NAMES):
        label = f"{LEGEND_NAMES[idx]:<17}{counts.get(idx, 0):>3}/{cap[idx]}"
        color = get_piece_color(idx)
        marker = _junction_marker(idx)
        if marker is None:
            handles.append(Patch(facecolor=color, edgecolor="black", label=label))
        else:
            handles.append(plt.Line2D([0], [0], marker=marker, color=color,
                                      linestyle="-", linewidth=3, markersize=6,
                                      markeredgecolor="black", label=label))
    return handles


def save_gallery(rows, path, inventory, max_size, cols: int = 3):
    """Galeria ukladow z frontu: (lay, elem, trasy, kolizje, rozmiar) na panel."""
    half_w, half_h = max_size[0] / 2.0, max_size[1] / 2.0
    boundary = BoundaryConfig(min_x=-half_w, max_x=half_w, min_y=-half_h, max_y=half_h)
    n_rows = (len(rows) + cols - 1) // cols
    fig, axes = plt.subplots(n_rows, cols, figsize=(8 * cols, 7.5 * n_rows))
    axes = list(np.atleast_1d(axes).flatten())
    for ax, (lay, used, routes, ov, size) in zip(axes, rows):
        draw_layout_v1(ax, lay, boundary)
        ax.set_title(f"{used:.0f} elem, {routes:.0f} tras, kolizje {ov:.0f}, "
                     f"{size[0]:.0f}x{size[1]:.0f} studow", fontsize=12)
    for ax in axes[len(rows):]:
        ax.axis("off")
    axes[0].legend(handles=legend_handles(rows[0][0], inventory),
                   loc="lower left", prop={"family": "monospace", "size": 8})
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path
