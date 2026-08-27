"""Visualization functions for track layouts and optimization results."""

from pathlib import Path
from typing import Optional, Union

import matplotlib
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Patch, Polygon

from src.config import BoundaryConfig
from src.catalog import TrackCatalog
from src.geometry import Layout
from src.types import MultiPathLayout
from src.encoding import (
    SWITCH_INDICES,
    STRAIGHT_16 as PIECE_IDX_STRAIGHT_16,
    STRAIGHT_24 as PIECE_IDX_STRAIGHT_24,
    R40_CURVE as PIECE_IDX_R40_CURVE,
    CROSS_90 as PIECE_IDX_CROSS_90,
    R40_SWITCH_LEFT as PIECE_IDX_SWITCH_LEFT,
    R40_SWITCH_RIGHT as PIECE_IDX_SWITCH_RIGHT,
    DOUBLE_CROSSOVER as PIECE_IDX_DBL_CROSSOVER,
)

# Geometry helpers (arc/offset paths, rail constants) from the track model.
from src.lego_track_models import (
    R40,
    RAIL_OFFSET,
    CURVE_ANGLE,
    SWITCH_LEN,
    ARC1_ANGLE,
    ARC2_ANGLE,
    COL_RAIL,
    BED_WIDTH_STUD,
    RAIL_LW,
    N_ARC_PTS,
    arc_points,
    offset_path,
)

# Headless pipeline: PNGs only (Tk + Pool result-handler threads crash Tcl).
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402

# Color palette per piece type.
PIECE_COLORS = {
    PIECE_IDX_STRAIGHT_16: "#3498db",
    PIECE_IDX_STRAIGHT_24: "#2980b9",
    PIECE_IDX_R40_CURVE: "#27ae60",
    PIECE_IDX_CROSS_90: "#9b59b6",
    PIECE_IDX_SWITCH_LEFT: "#e74c3c",
    PIECE_IDX_SWITCH_RIGHT: "#e67e22",
    PIECE_IDX_DBL_CROSSOVER: "#8e44ad",
}

# Fallback colors for unknown pieces
FALLBACK_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

# Piece index constants for rendering are imported from src.encoding (single
# source of truth: the canonical 0..6 chromosome mapping).

# DOUBLE_CROSSOVER body dimensions (studs): 48 long x 16 wide, port A at
# piece-local (0, 0) and port C at (0, 16). Mirrors src.templates.DC_*.
DC_LENGTH_STUDS = 48.0
DC_LATERAL_STUDS = 16.0

# Rendering resolution / overlay widths (studs).
SWITCH_ARC_PTS = 30              # arc points for the (short) switch S-curve branches
SWITCH_BRANCH_WIDTH_STUD = 6.4   # thinner bed for the switch diverge S-curve
DC_CROSS_WIDTH_STUD = 3.6        # thin bed for the double-crossover scissors roads


def get_piece_color(piece_idx: int) -> str:
    if piece_idx in PIECE_COLORS:
        return PIECE_COLORS[piece_idx]
    return FALLBACK_COLORS[piece_idx % len(FALLBACK_COLORS)]


def _draw_track_bed(ax, x, y, color, alpha=0.88, zorder=2, width_stud=None):
    """Fill the colored track bed as a polygon centered on the path.

    Width is in DATA units (studs) and defaults to BED_WIDTH_STUD = 8
    (full physical piece width). Callers may override width_stud to draw
    visually-distinct thinner overlays (e.g. switch diverge segments at
    ~6 stud, double-crossover diagonal routes at ~5 stud) so multiple
    routes through the same physical area remain readable."""
    half = float(BED_WIDTH_STUD if width_stud is None else width_stud) / 2.0
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.size < 2:
        return
    lx, ly = offset_path(x_arr, y_arr, +half)
    rx, ry = offset_path(x_arr, y_arr, -half)
    verts = list(zip(lx, ly)) + list(zip(rx[::-1], ry[::-1]))
    ax.add_patch(Polygon(verts, closed=True, facecolor=color,
                         edgecolor='none', alpha=alpha, zorder=zorder))


def _draw_rails(ax, x, y, offset=RAIL_OFFSET, zorder=3):
    """Draw two thin gray rails offset from centerline."""
    lx, ly = offset_path(x, y, +offset)
    rx, ry = offset_path(x, y, -offset)
    ax.plot(lx, ly, color=COL_RAIL, lw=RAIL_LW, solid_capstyle='butt', zorder=zorder)
    ax.plot(rx, ry, color=COL_RAIL, lw=RAIL_LW, solid_capstyle='butt', zorder=zorder)


def _local_to_world(x0, y0, theta0):
    """Return a closure mapping piece-local (lx, ly) studs to world coords (SE(2))."""
    theta_rad = np.radians(theta0)
    cos_t, sin_t = np.cos(theta_rad), np.sin(theta_rad)

    def to_world(lx, ly):
        return (x0 + lx * cos_t - ly * sin_t, y0 + lx * sin_t + ly * cos_t)

    return to_world


def _draw_segment(ax, p, q, color, draw_rails_flag=True, **bed_kw):
    """Draw a straight bed (+ optional rails) between two world points p and q."""
    xs = np.array([p[0], q[0]])
    ys = np.array([p[1], q[1]])
    _draw_track_bed(ax, xs, ys, color, **bed_kw)
    if draw_rails_flag:
        _draw_rails(ax, xs, ys)


def _draw_arc(ax, cx, cy, radius, start_ang, sweep_deg, color,
              draw_rails_flag=True, n=N_ARC_PTS, **bed_kw):
    """Draw an arc bed (+ optional inner/outer rails) about center (cx, cy)."""
    xc, yc = arc_points(cx, cy, radius, start_ang, sweep_deg, n=n)
    _draw_track_bed(ax, xc, yc, color, **bed_kw)
    if draw_rails_flag:
        xi, yi = arc_points(cx, cy, radius - RAIL_OFFSET, start_ang, sweep_deg, n=n)
        xo, yo = arc_points(cx, cy, radius + RAIL_OFFSET, start_ang, sweep_deg, n=n)
        ax.plot(xi, yi, color=COL_RAIL, lw=RAIL_LW, solid_capstyle='butt', zorder=3)
        ax.plot(xo, yo, color=COL_RAIL, lw=RAIL_LW, solid_capstyle='butt', zorder=3)


def _draw_straight_piece(ax, x0, y0, theta0, length, color, draw_rails_flag=True):
    """Draw a straight track piece with proper geometry.

    Args:
        ax: Matplotlib axes.
        x0, y0: Start position (studs).
        theta0: Entry heading (degrees).
        length: Piece length (studs).
        color: Track bed color.
        draw_rails_flag: Whether to draw rails.
    """
    theta_rad = np.radians(theta0)
    x1 = x0 + length * np.cos(theta_rad)
    y1 = y0 + length * np.sin(theta_rad)
    _draw_segment(ax, (x0, y0), (x1, y1), color, draw_rails_flag)


def _draw_curve_piece(ax, x0, y0, theta0, radius, sweep_deg, color, draw_rails_flag=True):
    """Draw a curved track piece with proper arc geometry.

    Args:
        ax: Matplotlib axes.
        x0, y0: Start position (studs).
        theta0: Entry heading (degrees).
        radius: Curve radius (studs).
        sweep_deg: Sweep angle (positive = left/CCW, negative = right/CW).
        color: Track bed color.
        draw_rails_flag: Whether to draw rails.
    """
    # Arc center is perpendicular to entry heading: left turn (positive sweep)
    # -> center +90deg; right turn (negative sweep) -> center -90deg.
    center_offset_angle = theta0 + 90 if sweep_deg > 0 else theta0 - 90
    center_offset_rad = np.radians(center_offset_angle)
    cx = x0 + radius * np.cos(center_offset_rad)
    cy = y0 + radius * np.sin(center_offset_rad)

    # Start angle is from center to entry point
    start_ang = np.degrees(np.arctan2(y0 - cy, x0 - cx))

    _draw_arc(ax, cx, cy, radius, start_ang, sweep_deg, color, draw_rails_flag)


def _draw_switch_piece(ax, x0, y0, theta0, direction, color_main, color_branch,
                       draw_rails_flag=True):
    """Draw a switch piece with main straight path and diverging branch (S-curve).

    Args:
        ax: Matplotlib axes.
        x0, y0: Start position (studs).
        theta0: Entry heading (degrees).
        direction: 'left' or 'right'.
        color_main: Main path color.
        color_branch: Branch path color.
        draw_rails_flag: Whether to draw rails.
    """
    tw = _local_to_world(x0, y0, theta0)

    # Main straight path (SWITCH_LEN studs)
    _draw_segment(ax, (x0, y0), tw(SWITCH_LEN, 0.0), color_main, draw_rails_flag)

    flip = 1 if direction == 'left' else -1

    # Arc 1: outward curve
    c1x, c1y = tw(0.0, flip * R40)
    a1_start = theta0 - 90 * flip
    a1_sweep = flip * ARC1_ANGLE
    _draw_arc(ax, c1x, c1y, R40, a1_start, a1_sweep, color_branch, draw_rails_flag,
              n=SWITCH_ARC_PTS, width_stud=SWITCH_BRANCH_WIDTH_STUD)

    # Junction point at end of Arc 1
    a1_end_rad = np.radians(a1_start + a1_sweep)
    jx = c1x + R40 * np.cos(a1_end_rad)
    jy = c1y + R40 * np.sin(a1_end_rad)
    j_heading = theta0 + flip * ARC1_ANGLE

    # Arc 2: inward curve (opposite direction)
    j_heading_rad = np.radians(j_heading)
    perp_dir = j_heading_rad - flip * np.pi / 2
    c2x = jx + R40 * np.cos(perp_dir)
    c2y = jy + R40 * np.sin(perp_dir)
    a2_start = np.degrees(np.arctan2(jy - c2y, jx - c2x))
    a2_sweep = -flip * ARC2_ANGLE
    _draw_arc(ax, c2x, c2y, R40, a2_start, a2_sweep, color_branch, draw_rails_flag,
              n=SWITCH_ARC_PTS, width_stud=SWITCH_BRANCH_WIDTH_STUD)


def _draw_switch_with_install(ax, x0, y0, theta0, catalog_direction,
                              color_main, color_branch, draw_rails_flag, installed_reversed):
    """Render a switch accounting for installation orientation.

    `_draw_switch_piece` always sprouts the diverge S-curve from the train's
    entry point. That's correct for normal install (port C is forward of port
    A entry), but a reversed-install switch has port C at the OPPOSITE end of
    the body. Calling `_draw_switch_piece` directly would put the diverge in
    the wrong location.

    Trick: for reversed install, render the switch from the FAR end with
    heading rotated 180°. The resulting through line is visually the same
    32-stud segment, but the natural diverge S-curve now sprouts from what
    was the FAR end (the "actual port A" end in main frame), curving back
    toward the body start where port C truly sits in main frame.
    """
    if installed_reversed:
        theta_rad = np.radians(theta0)
        x_far = x0 + SWITCH_LEN * np.cos(theta_rad)
        y_far = y0 + SWITCH_LEN * np.sin(theta_rad)
        _draw_switch_piece(ax, x_far, y_far, theta0 + 180.0, catalog_direction,
                           color_main, color_branch, draw_rails_flag)
    else:
        _draw_switch_piece(ax, x0, y0, theta0, catalog_direction,
                           color_main, color_branch, draw_rails_flag)


def _draw_cross90_piece(ax, x0, y0, theta0, color, draw_rails_flag=True):
    """Draw a 90-degree crossing with two perpendicular routes.

    Args:
        ax: Matplotlib axes.
        x0, y0: Start position (studs).
        theta0: Entry heading (degrees).
        color: Track bed color.
        draw_rails_flag: Whether to draw rails.
    """
    tw = _local_to_world(x0, y0, theta0)

    # Horizontal route (main path): 0 to 16 studs along entry heading.
    _draw_segment(ax, tw(0, 0), tw(16, 0), color, draw_rails_flag)

    # Vertical route (perpendicular): centered at (8, 0), spanning -8 to +8.
    _draw_segment(ax, tw(8, -8), tw(8, 8), color, draw_rails_flag)


def _draw_double_crossover_piece(ax, x0, y0, theta0, color, draw_rails_flag=True):
    """Draw a double crossover as a clean 4-PORT scissors crossover.

    The piece (4DBrix 210.1, 48x16 studs) has exactly FOUR ports — A,B on
    track 1 (y=0) and C,D on track 2 (y=16) — with two straight-through routes
    (A-B, C-D) and two crossover roads (A->D, C->B) meeting at ONE central
    at-grade crossing (the catalog ``DOUBLE_CROSSOVER`` routes). The two
    parallel tracks are drawn full-length; the crossover roads are thin
    diagonals over the central third only, so the piece reads as "2 tracks +
    1 scissors" instead of a tangle of stub ends. The four ports are marked
    explicitly.

    Args:
        ax: Matplotlib axes.
        x0, y0: Start position (studs) - entry of track 1 (port A).
        theta0: Entry heading (degrees).
        color: Track bed color.
        draw_rails_flag: Whether to draw rails.
    """
    tw = _local_to_world(x0, y0, theta0)
    length, lat = DC_LENGTH_STUDS, DC_LATERAL_STUDS

    # Two parallel through-tracks: A->B (y=0) and C->D (y=lat).
    _draw_segment(ax, tw(0.0, 0.0), tw(length, 0.0), color, draw_rails_flag)
    _draw_segment(ax, tw(0.0, lat), tw(length, lat), color, draw_rails_flag)

    # Central scissors: crossover roads A->D and C->B, thin and confined to the
    # middle third so they cross once at the center without stubbing the mainlines.
    _draw_segment(ax, tw(16.0, 0.0), tw(32.0, lat), color, draw_rails_flag,
                  width_stud=DC_CROSS_WIDTH_STUD, alpha=0.75)
    _draw_segment(ax, tw(16.0, lat), tw(32.0, 0.0), color, draw_rails_flag,
                  width_stud=DC_CROSS_WIDTH_STUD, alpha=0.75)

    # Mark the FOUR ports (A, B, C, D) so the piece reads 4-port, not 6.
    for px, py in ((0.0, 0.0), (length, 0.0), (0.0, lat), (length, lat)):
        wx, wy = tw(px, py)
        ax.plot(wx, wy, marker='o', ms=4.5, mfc=color, mec='white',
                mew=0.7, zorder=6)


def _dc_body_poses(piece_sequence, states):
    """Port-A world poses for each PHYSICAL double crossover threaded by a path.

    A DOUBLE_CROSSOVER is one physical piece, but a closed loop covers its four
    ports by traversing it twice, so index 6 appears at two non-adjacent slots.
    Drawing the body per-occurrence paints it twice and -- for the traversal that
    enters at port C -- 16 studs off to the wrong side. We recover port A's pose
    from each traversal's entry+exit geometry and de-duplicate, so
    ``_draw_double_crossover_piece`` is called exactly once per physical piece.

    A traversal's lateral exit offset (train frame) identifies the entry port:
    +16 -> entered port A (cross_1_to_2); -16 -> entered port C (cross_2_to_1),
    so port A sits 16 studs back along the piece-local +y axis. Through routes
    (offset 0) enter at port A by convention; the both-through cover is emitted
    by no current seed, so its port-C twin is not separately disambiguated.

    Args:
        piece_sequence: Per-slot catalog indices for the path.
        states: (n+1, 3) cumulative [x, y, theta_deg] for the path.

    Returns:
        List of unique (x, y, theta_deg) port-A poses, one per physical DC.
    """
    poses = []
    seen = set()
    for i, piece_idx in enumerate(piece_sequence):
        if piece_idx != PIECE_IDX_DBL_CROSSOVER or i + 1 >= len(states):
            continue
        ex, ey, eth = float(states[i][0]), float(states[i][1]), float(states[i][2])
        th = np.radians(eth)
        dx, dy = states[i + 1][0] - ex, states[i + 1][1] - ey
        lateral = -np.sin(th) * dx + np.cos(th) * dy
        if lateral < -1.0:  # entered port C: port A is 16 stud back along +y
            ax_ = ex + DC_LATERAL_STUDS * np.sin(th)
            ay_ = ey - DC_LATERAL_STUDS * np.cos(th)
        else:               # entered port A (cross_1_to_2 or a through route)
            ax_, ay_ = ex, ey
        # Round before the modulo so an FK-drifted 359.9999 deg wraps to 0, not
        # 360 -- otherwise the two traversals of one piece key differently.
        key = (round(ax_, 1), round(ay_, 1), round(round(eth, 1) % 360.0, 1))
        if key in seen:
            continue
        seen.add(key)
        poses.append((float(ax_), float(ay_), eth))
    return poses


def _draw_dc_bodies(ax, piece_sequence, states):
    """Draw every physical double crossover threaded by a path exactly once."""
    color = get_piece_color(PIECE_IDX_DBL_CROSSOVER)
    for dcx, dcy, dcth in _dc_body_poses(piece_sequence, states):
        _draw_double_crossover_piece(ax, dcx, dcy, dcth, color)


def _r40_flip_from_dtheta(piece_idx: int, dtheta_deg: float) -> int:
    """Recover the per-slot flip bit for an R40_CURVE from its FK heading change.

    Heading change in (-180, +180]: positive ≈ +22.5° → flip=0 (LEFT);
    negative ≈ -22.5° → flip=1 (RIGHT). Non-R40 pieces always return 0.
    """
    if piece_idx != PIECE_IDX_R40_CURVE:
        return 0
    d = float(dtheta_deg)
    while d > 180.0:
        d -= 360.0
    while d <= -180.0:
        d += 360.0
    return 1 if d < 0 else 0


def _draw_piece(ax, piece_idx, x0, y0, theta0, draw_rails_flag=True, installed_reversed=False,
                flip=0):
    """Draw a single piece with proper geometry based on piece type.

    Args:
        ax: Matplotlib axes.
        piece_idx: Piece index (post-refactor 0..6).
        x0, y0: Start position (studs).
        theta0: Entry heading (degrees).
        draw_rails_flag: Whether to draw rails.
        installed_reversed: Only meaningful for switches (see prior docstring).
        flip: For R40_CURVE only — 0 draws a LEFT arc (+22.5°), 1 draws RIGHT
            (-22.5°). Other piece types ignore this flag.
    """
    color = get_piece_color(piece_idx)

    if piece_idx == PIECE_IDX_STRAIGHT_16:
        _draw_straight_piece(ax, x0, y0, theta0, 16.0, color, draw_rails_flag)
    elif piece_idx == PIECE_IDX_STRAIGHT_24:
        _draw_straight_piece(ax, x0, y0, theta0, 24.0, color, draw_rails_flag)
    elif piece_idx == PIECE_IDX_R40_CURVE:
        sweep = CURVE_ANGLE if not flip else -CURVE_ANGLE
        _draw_curve_piece(ax, x0, y0, theta0, R40, sweep, color, draw_rails_flag)
    elif piece_idx == PIECE_IDX_SWITCH_LEFT:
        _draw_switch_with_install(ax, x0, y0, theta0, 'left', color, color,
                                  draw_rails_flag, installed_reversed)
    elif piece_idx == PIECE_IDX_SWITCH_RIGHT:
        _draw_switch_with_install(ax, x0, y0, theta0, 'right', color, color,
                                  draw_rails_flag, installed_reversed)
    elif piece_idx == PIECE_IDX_CROSS_90:
        _draw_cross90_piece(ax, x0, y0, theta0, color, draw_rails_flag)
    elif piece_idx == PIECE_IDX_DBL_CROSSOVER:
        _draw_double_crossover_piece(ax, x0, y0, theta0, color, draw_rails_flag)
    else:
        # Unknown/inactive index: nothing to draw (valid layouts only carry 0..6).
        pass


def get_piece_short_name(piece_idx: int, catalog: TrackCatalog) -> str:
    """Catalog id for a piece index: the same name run_info.md prints."""
    return catalog.index_to_id[piece_idx]


def _draw_piece_sequence(ax, pieces, states, *, reversed_flags=None, limit=None):
    """Draw the piece geometry for one traversal sequence.

    Shared by all three render paths so a piece-rendering fix lands once
    instead of in three near-identical loops. A DOUBLE_CROSSOVER is one
    physical piece threaded twice; its body is drawn once via _draw_dc_bodies,
    never per traversal.

    Args:
        ax: Matplotlib axes.
        pieces: Per-slot catalog indices (list or ndarray), length n.
        states: (n+1, 3) [x, y, theta] poses; theta in degrees.
        reversed_flags: Optional per-slot bool; True installs that slot's
            switch reversed. None means no slot is reversed.
        limit: Draw only the first `limit` slots (and their DC bodies);
            None draws all.
    """
    n = len(pieces) if limit is None else min(limit, len(pieces))
    for i in range(n):
        if i + 1 >= len(states):
            break
        piece_idx = int(pieces[i])
        if piece_idx == PIECE_IDX_DBL_CROSSOVER:
            continue
        flip = _r40_flip_from_dtheta(piece_idx, states[i + 1, 2] - states[i, 2])
        installed_reversed = bool(reversed_flags[i]) if reversed_flags is not None else False
        _draw_piece(ax, piece_idx, states[i, 0], states[i, 1], states[i, 2],
                    draw_rails_flag=True, installed_reversed=installed_reversed, flip=flip)
    _draw_dc_bodies(ax, [int(p) for p in pieces[:n]], states)


def _draw_boundary(ax, boundary, *, label=None):
    """Draw the boundary rectangle as a dashed line, if a boundary is given."""
    if boundary is None:
        return
    rect_x = [boundary.min_x, boundary.max_x, boundary.max_x, boundary.min_x, boundary.min_x]
    rect_y = [boundary.min_y, boundary.min_y, boundary.max_y, boundary.max_y, boundary.min_y]
    ax.plot(rect_x, rect_y, "k--", linewidth=1, alpha=0.5, label=label)


def _setup_axes(ax, *, label_fontsize=None):
    """Equal aspect, axis labels (studs), and a light grid."""
    ax.set_aspect("equal")
    label_kw = {"fontsize": label_fontsize} if label_fontsize is not None else {}
    ax.set_xlabel("X (studs)", **label_kw)
    ax.set_ylabel("Y (studs)", **label_kw)
    ax.grid(True, alpha=0.3)


def _metrics_box(ax, text, *, fontsize):
    """Draw the standard top-left metrics text box."""
    ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=fontsize,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))


def plot_layout(
    layout: Layout,
    catalog: TrackCatalog,
    boundary: Optional[BoundaryConfig] = None,
    title: str = "Track Layout",
    save_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """Plot track layout with colored pieces and connection dots.

    Each piece type has a different color. Switches are drawn as their S-curve
    body plus a diamond marker; crossings get a square marker.
    Start is marked with green square, end with red circle.

    Args:
        layout: Track layout to plot.
        catalog: Piece catalog; supplies the id per index for the legend.
        boundary: Optional boundary configuration to draw.
        title: Plot title.
        save_path: Optional path to save plot as PNG.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    if layout.n_pieces == 0:
        ax.text(0.5, 0.5, "Empty Layout", ha="center", va="center", fontsize=16)
        ax.set_title(title)
        return fig

    x = layout.states[:, 0]
    y = layout.states[:, 1]

    # Legend + special-position bookkeeping (by index for consistent ordering).
    indices = [int(v) for v in layout.indices]
    piece_indices_seen = set(indices)
    switch_positions = [i for i, p in enumerate(indices) if p in SWITCH_INDICES]
    crossing_positions = [i for i, p in enumerate(indices) if p == PIECE_IDX_CROSS_90]

    _draw_piece_sequence(ax, indices, layout.states)

    for i in switch_positions:
        color = get_piece_color(indices[i])
        mid_x = (x[i] + x[i + 1]) / 2
        mid_y = (y[i] + y[i + 1]) / 2
        ax.plot(mid_x, mid_y, "D", color=color, markersize=8, markeredgecolor="black",
                markeredgewidth=1, zorder=6)

    for i in crossing_positions:
        color = get_piece_color(indices[i])
        mid_x = (x[i] + x[i + 1]) / 2
        mid_y = (y[i] + y[i + 1]) / 2
        ax.plot(mid_x, mid_y, "s", color=color, markersize=8, markeredgecolor="black",
                markeredgewidth=1, zorder=6)

    ax.plot(x[0], y[0], "gs", markersize=12, label="Start", zorder=10)
    ax.plot(x[-1], y[-1], "ro", markersize=12, label="End", zorder=10)

    _draw_boundary(ax, boundary, label="Boundary")
    _setup_axes(ax, label_fontsize=11)
    ax.set_title(title, fontsize=14)

    legend_elements = []
    for piece_idx in sorted(piece_indices_seen):
        color = get_piece_color(piece_idx)
        name = get_piece_short_name(piece_idx, catalog)
        if piece_idx in SWITCH_INDICES:
            legend_elements.append(
                plt.Line2D([0], [0], marker="D", color=color, linestyle="-",
                           linewidth=3, markersize=6, markeredgecolor="black", label=name)
            )
        elif piece_idx == PIECE_IDX_CROSS_90:
            # Square marker for CROSS_90 (matches the square position markers).
            legend_elements.append(
                plt.Line2D([0], [0], marker="s", color=color, linestyle="-",
                           linewidth=3, markersize=6, markeredgecolor="black", label=name)
            )
        else:
            # Straights, curves, and the DOUBLE_CROSSOVER (drawn as its own body)
            # get a filled swatch.
            legend_elements.append(Patch(facecolor=color, edgecolor="black", label=name))

    legend_elements.extend([
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="g", markersize=10,
                   label="Start"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="r", markersize=10,
                   label="End"),
    ])

    if boundary is not None:
        legend_elements.append(plt.Line2D([0], [0], color="k", linestyle="--", label="Boundary"))

    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

    # Add metrics text. "Crossings" counts CROSS_90 pieces (the square-marked
    # ones); DOUBLE_CROSSOVERs are reported separately, deduped to physical
    # pieces (one DC is threaded twice but is one body).
    n_switches = sum(1 for i in layout.indices if int(i) in SWITCH_INDICES)
    n_crossings = sum(1 for i in layout.indices if int(i) == PIECE_IDX_CROSS_90)
    n_crossovers = len(_dc_body_poses(indices, layout.states))
    metrics_lines = [
        f"Pieces: {layout.n_physical_pieces}",
        f"Switches: {n_switches}",
        f"Crossings: {n_crossings}",
    ]
    if n_crossovers:
        metrics_lines.append(f"Crossovers: {n_crossovers}")
    metrics_lines += [
        f"Closure Error: {layout.closure_error:.2f} studs",
        f"Angle Error: {layout.angle_error:.2f}°",
        f"Area: {layout.area:.0f} studs²",
    ]
    _metrics_box(ax, "\n".join(metrics_lines), fontsize=10)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, format='png', dpi=150, bbox_inches='tight')

    return fig


def plot_multi_path_layout(
    layout: MultiPathLayout,
    catalog: TrackCatalog,
    boundary: Optional[BoundaryConfig] = None,
    title: str = "Multi-Path Track Layout",
    save_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """Plot multi-path track layout showing all traversal paths.

    Each path is drawn with a different color. The main path (all straight-through)
    is drawn in blue, branch paths in other colors.

    Args:
        layout: MultiPathLayout to plot.
        catalog: Unused; kept for call-compatibility with the render pipeline.
        boundary: Optional boundary configuration to draw.
        title: Plot title.
        save_path: Optional path to save plot as PNG.

    Returns:
        Matplotlib figure.
    """
    n_paths = len(layout.paths)

    if n_paths == 0:
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.text(0.5, 0.5, "Empty Layout", ha="center", va="center", fontsize=16)
        ax.set_title(title)
        return fig

    # Create subplots: one for combined view, one for each path
    n_cols = min(3, n_paths + 1)
    n_rows = (n_paths + 1 + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))

    axes = np.asarray(axes).flatten()

    # Hide unused subplots
    for i in range(n_paths + 1, len(axes)):
        axes[i].set_visible(False)

    # Combined view (first subplot)
    ax_combined = axes[0]
    _plot_combined_paths(ax_combined, layout, boundary, title)

    # Individual path views
    for i, path in enumerate(layout.paths):
        if i + 1 < len(axes):
            ax = axes[i + 1]
            _plot_single_path(ax, path, boundary, i)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, format='png', dpi=150, bbox_inches='tight')

    return fig


def _plot_combined_paths(ax, layout: MultiPathLayout, boundary, title):
    """Plot main path with branch sections highlighted differently."""
    # Draw main path (path 0) with proper piece geometry
    main_path = layout.paths[0] if layout.paths else None
    if main_path is None or len(main_path.states) <= 1:
        return

    x = main_path.states[:, 0]
    y = main_path.states[:, 1]

    # OUT positions of all switch pairs — these switches are installed reversed.
    out_positions = {sp.out_position for sp in layout.switch_pairs}
    reversed_flags = [i in out_positions for i in range(len(main_path.piece_sequence))]
    _draw_piece_sequence(ax, main_path.piece_sequence, main_path.states,
                         reversed_flags=reversed_flags)

    ax.plot(x[0], y[0], "gs", markersize=10, zorder=10)
    ax.plot(x[-1], y[-1], "ro", markersize=10, zorder=10)

    # Mark switch positions on main path. The pair holds two opposite-handed
    # switches; read the actual piece index from the augmented main_loop_pieces.
    for sp in layout.switch_pairs:
        for pos in (sp.in_position, sp.out_position):
            if pos < len(x) - 1 and pos < len(layout.main_loop_pieces):
                sw_idx = layout.main_loop_pieces[pos]
                sw_color = get_piece_color(sw_idx)
                mid_x = (x[pos] + x[pos + 1]) / 2
                mid_y = (y[pos] + y[pos + 1]) / 2
                ax.plot(mid_x, mid_y, "D", color=sw_color, markersize=10,
                        markeredgecolor="black", markeredgewidth=1.5, zorder=8)

    # Draw the DIVERGENT section of each branched switch pair, deduplicated.
    # Source-of-truth for the slice is path.divergent_ranges, populated by the
    # decoder while building each path. Slicing piece_sequence/states by the
    # comparator-inferred range was incorrect because branch and main paths
    # have different lengths past the first IN switch and never realign.
    for seg_idx, (pieces, states) in enumerate(layout.get_branch_segments()):
        if len(states) <= 1:
            continue

        # Skip the IN switch (pieces[0]) and OUT switch (pieces[-1]); they're
        # already drawn at their main-loop positions by the main-path render
        # above. Drawing them again here at the branch-end positions (which
        # differ from the main-loop positions) would visually duplicate them
        # at conflicting angles. Just draw the branch INTERNAL pieces.
        for k in range(1, len(pieces) - 1):
            sx, sy, stheta = states[k]
            _, _, stheta_next = states[k + 1] if k + 1 < len(states) else (sx, sy, stheta)
            flip = _r40_flip_from_dtheta(pieces[k], stheta_next - stheta)
            _draw_piece(ax, pieces[k], sx, sy, stheta, draw_rails_flag=True, flip=flip)

        ax.plot(
            states[:, 0], states[:, 1],
            color='#e74c3c', linewidth=2.5, linestyle='-',
            alpha=0.6, zorder=5,
            label='Branch' if seg_idx == 0 else None,
        )

    # Overlay post-merge FK drift as a diagnostic indicator. These points
    # represent physically real pieces (already drawn solid above) whose FK
    # positions drifted because a branch's merge route did not align with the
    # main loop. Drawn dotted gray to distinguish from real geometry.
    drift_segments = layout.get_post_merge_drift()
    for seg_idx, (_, drift_states) in enumerate(drift_segments):
        ax.plot(
            drift_states[:, 0], drift_states[:, 1],
            color='#888888', linewidth=1.0, linestyle=':',
            alpha=0.6, zorder=3,
            label='FK drift (failed merge)' if seg_idx == 0 else None,
        )

    _draw_boundary(ax, boundary)
    _setup_axes(ax)
    ax.set_title(f"{title}\n(All {len(layout.paths)} Paths)")
    ax.legend(fontsize=8, loc="upper right")

    # Metrics
    n_switches = sum(1 for p in layout.main_loop_pieces if p in SWITCH_INDICES)
    metrics = (
        f"Pieces: {layout.n_physical_pieces}\n"
        f"Switches: {n_switches}\n"
        f"Switch Pairs: {layout.n_switch_pairs}\n"
        f"Max Closure: {layout.max_closure_error:.2f}\n"
        f"Max Angle: {layout.max_angle_error:.2f}°"
    )
    _metrics_box(ax, metrics, fontsize=9)


def _plot_single_path(ax, path, boundary, path_idx):
    """Plot a single traversal path with switch markers."""
    if len(path.states) <= 1:
        ax.text(0.5, 0.5, "Empty Path", ha="center", va="center")
        return

    x = path.states[:, 0]
    y = path.states[:, 1]

    # Identify switch positions in this path
    switch_positions = []
    for i, piece_idx in enumerate(path.piece_sequence):
        if piece_idx in SWITCH_INDICES:
            switch_positions.append(i)

    # Drift boundary: only show diagnostic dotted-gray drift trajectory when
    # the path actually fails to close. A path that closes (small closure /
    # angle error) has no meaningful drift — render the whole thing solid.
    DRIFT_POS_THRESHOLD = 1.0  # studs
    DRIFT_ANG_THRESHOLD = 1.0  # degrees
    has_drift = (
        path.closure_error > DRIFT_POS_THRESHOLD
        or path.angle_error > DRIFT_ANG_THRESHOLD
    )
    drift_start = (
        min(end for _, end in path.divergent_ranges.values()) + 1
        if (has_drift and path.divergent_ranges)
        else len(path.piece_sequence)
    )

    # Plot solid pieces only up to the drift boundary. Switches in a path
    # alternate IN, OUT, IN, OUT (one pair contributes its entry switch then
    # its exit switch); the 2nd, 4th, ... switches are OUTs and were installed
    # reversed, so the renderer flips their visual diverge direction.
    switches_seen = 0
    reversed_flags = []
    for piece_idx in path.piece_sequence:
        rev = False
        if piece_idx in SWITCH_INDICES:
            switches_seen += 1
            rev = (switches_seen % 2 == 0)
        reversed_flags.append(rev)

    n_solid = min(drift_start, len(path.piece_sequence))
    _draw_piece_sequence(ax, path.piece_sequence, path.states,
                         reversed_flags=reversed_flags, limit=n_solid)

    # Overlay drift trajectory (if any) as a dotted gray polyline
    if drift_start < len(x) - 1:
        ax.plot(
            x[drift_start:], y[drift_start:],
            color='#888888', linewidth=1.0, linestyle=':',
            alpha=0.6, zorder=3,
            label='FK drift (failed merge)',
        )

    # Mark switch positions with diamond markers — only for non-drifted switches
    for sw_pos in switch_positions:
        if sw_pos >= drift_start or sw_pos >= len(x) - 1:
            continue
        piece_idx = path.piece_sequence[sw_pos]
        sw_color = get_piece_color(piece_idx)
        mid_x = (x[sw_pos] + x[sw_pos + 1]) / 2
        mid_y = (y[sw_pos] + y[sw_pos + 1]) / 2
        ax.plot(mid_x, mid_y, "D", color=sw_color, markersize=8,
                markeredgecolor="black", markeredgewidth=1, zorder=6)

    # Mark start/end (end may be a drifted position — that's intentional, it
    # visualizes closure-failure magnitude).
    ax.plot(x[0], y[0], "gs", markersize=8, zorder=10)
    ax.plot(x[-1], y[-1], "ro", markersize=8, zorder=10)

    _draw_boundary(ax, boundary)
    _setup_axes(ax)

    closed_str = "CLOSED" if path.is_closed else "OPEN"
    n_switches = len(switch_positions)
    ax.set_title(f"Path {path_idx}: {path.describe_route()}\n"
                 f"({path.n_pieces} pieces, {n_switches} switches, {closed_str})")

    # Metrics
    metrics = (
        f"Closure: {path.closure_error:.2f}\n"
        f"Angle: {path.angle_error:.2f}°"
    )
    _metrics_box(ax, metrics, fontsize=8)
