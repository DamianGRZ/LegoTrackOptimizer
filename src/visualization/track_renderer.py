"""Visualization functions for track layouts and optimization results."""

import math
from pathlib import Path
from typing import Optional, Sequence, Union

import matplotlib
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Patch, Polygon

from src.config import BoundaryConfig
from src.catalog import TrackCatalog
from src.types import MultiPathLayout, TraversalPath
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
from src.run_info import count_pieces

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

# Track-piece geometry (studs/degrees). Exact geometry from community
# research (MacFreek, L-Gauge, Transponderings, Fx Bricks).
R40 = 40.0                          # curve radius in studs
RAIL_OFFSET = 2.5                   # half gauge (5 studs / 2)
CURVE_ANGLE = 22.5                  # degrees per R40 piece
SWITCH_LEN = 32.0                   # switch straight path length
ARC1_ANGLE = math.degrees(math.atan2(3, 4))  # 36.87° (3-4-5 triple)
ARC2_ANGLE = ARC1_ANGLE - CURVE_ANGLE         # 14.37°

# Full physical piece width in studs. A siding runs 16 studs centre-to-centre
# from the main line, so two 8-wide beds leave an 8-stud gap between their edges.
BED_WIDTH_STUD = 8.0

# Rail style.
COL_RAIL = '#95a5a6'               # gray
RAIL_LW = 1.8                      # rail line width
N_ARC_PTS = 60                     # arc interpolation points

# DOUBLE_CROSSOVER body dimensions (studs): 48 long x 16 wide, port A at
# piece-local (0, 0) and port C at (0, 16). Mirrors src.templates.DC_*.
DC_LENGTH_STUDS = 48.0
DC_LATERAL_STUDS = 16.0

# Rendering resolution / overlay widths (studs).
SWITCH_ARC_PTS = 30              # arc points for the (short) switch S-curve branches
SWITCH_BRANCH_WIDTH_STUD = 6.4   # thinner bed for the switch diverge S-curve
DC_CROSS_WIDTH_STUD = 3.6        # thin bed for the double-crossover scissors roads

# Track-axes type sizes, in typographic points.
AXIS_LABEL_FONTSIZE = 11
TITLE_FONTSIZE = 14

# Piece-joint seam: background-colored line across the bed at every piece
# boundary, so pieces can be counted by eye. Width in points, not studs —
# it must stay visible at any zoom level.
JOINT_COLOR = "white"
JOINT_LW = 1.2


def get_piece_color(piece_idx: int) -> str:
    if piece_idx in PIECE_COLORS:
        return PIECE_COLORS[piece_idx]
    return FALLBACK_COLORS[piece_idx % len(FALLBACK_COLORS)]


def arc_points(cx, cy, r, start_deg, sweep_deg, n=N_ARC_PTS):
    """Generate (x, y) arrays along a circular arc."""
    angles = np.linspace(np.radians(start_deg),
                         np.radians(start_deg + sweep_deg), n)
    return cx + r * np.cos(angles), cy + r * np.sin(angles)


def offset_path(x, y, offset):
    """Offset a polyline by +/-offset studs perpendicular to path direction."""
    dx = np.gradient(x)
    dy = np.gradient(y)
    length = np.sqrt(dx**2 + dy**2)
    length[length == 0] = 1e-9
    nx = -dy / length  # perpendicular normal
    ny = dx / length
    return x + offset * nx, y + offset * ny


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


def _draw_joint(ax, x, y, theta_deg):
    """Seam across the bed at a piece boundary, perpendicular to the heading.

    Drawn at piece boundaries only (FK states), never inside a piece body.
    """
    perp = np.radians(theta_deg + 90.0)
    half = BED_WIDTH_STUD / 2.0
    dx, dy = half * np.cos(perp), half * np.sin(perp)
    ax.plot([x - dx, x + dx], [y - dy, y + dy], color=JOINT_COLOR,
            lw=JOINT_LW, solid_capstyle='butt', zorder=4)


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


def _draw_segment(ax, p, q, color, **bed_kw):
    """Draw a straight bed with its rails between two world points p and q."""
    xs = np.array([p[0], q[0]])
    ys = np.array([p[1], q[1]])
    _draw_track_bed(ax, xs, ys, color, **bed_kw)
    _draw_rails(ax, xs, ys)


def _draw_arc(ax, cx, cy, radius, start_ang, sweep_deg, color, n=N_ARC_PTS,
              width_stud=None, alpha=0.88):
    """Draw an arc bed with its inner/outer rails about center (cx, cy).

    The bed is the ring between two concentric arcs, NOT an offset of the
    centerline polyline: gradient normals are one-sided at a polyline's ends,
    tilting the end edges by half a chord angle and leaving a background
    wedge at every curve-curve joint. Ring edges lie exactly on the radii,
    so consecutive pieces tile seamlessly.
    """
    half = float(BED_WIDTH_STUD if width_stud is None else width_stud) / 2.0
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
    """Draw a straight track piece with proper geometry.

    Args:
        ax: Matplotlib axes.
        x0, y0: Start position (studs).
        theta0: Entry heading (degrees).
        length: Piece length (studs).
        color: Track bed color.
    """
    theta_rad = np.radians(theta0)
    x1 = x0 + length * np.cos(theta_rad)
    y1 = y0 + length * np.sin(theta_rad)
    _draw_segment(ax, (x0, y0), (x1, y1), color)


def _draw_curve_piece(ax, x0, y0, theta0, radius, sweep_deg, color):
    """Draw a curved track piece with proper arc geometry.

    Args:
        ax: Matplotlib axes.
        x0, y0: Start position (studs).
        theta0: Entry heading (degrees).
        radius: Curve radius (studs).
        sweep_deg: Sweep angle (positive = left/CCW, negative = right/CW).
        color: Track bed color.
    """
    # Arc center is perpendicular to entry heading: left turn (positive sweep)
    # -> center +90deg; right turn (negative sweep) -> center -90deg.
    center_offset_angle = theta0 + 90 if sweep_deg > 0 else theta0 - 90
    center_offset_rad = np.radians(center_offset_angle)
    cx = x0 + radius * np.cos(center_offset_rad)
    cy = y0 + radius * np.sin(center_offset_rad)

    # Start angle is from center to entry point
    start_ang = np.degrees(np.arctan2(y0 - cy, x0 - cx))

    _draw_arc(ax, cx, cy, radius, start_ang, sweep_deg, color)


def _draw_switch_piece(ax, x0, y0, theta0, direction, color_main, color_branch):
    """Draw a switch piece with main straight path and diverging branch (S-curve).

    Args:
        ax: Matplotlib axes.
        x0, y0: Start position (studs).
        theta0: Entry heading (degrees).
        direction: 'left' or 'right'.
        color_main: Main path color.
        color_branch: Branch path color.
    """
    tw = _local_to_world(x0, y0, theta0)

    # Main straight path (SWITCH_LEN studs)
    _draw_segment(ax, (x0, y0), tw(SWITCH_LEN, 0.0), color_main)

    flip = 1 if direction == 'left' else -1

    # Arc 1: outward curve
    c1x, c1y = tw(0.0, flip * R40)
    a1_start = theta0 - 90 * flip
    a1_sweep = flip * ARC1_ANGLE
    _draw_arc(ax, c1x, c1y, R40, a1_start, a1_sweep, color_branch,
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
    _draw_arc(ax, c2x, c2y, R40, a2_start, a2_sweep, color_branch,
              n=SWITCH_ARC_PTS, width_stud=SWITCH_BRANCH_WIDTH_STUD)


def _draw_switch_with_install(ax, x0, y0, theta0, catalog_direction,
                              color_main, color_branch, installed_reversed):
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
                           color_main, color_branch)
    else:
        _draw_switch_piece(ax, x0, y0, theta0, catalog_direction,
                           color_main, color_branch)


def _draw_cross90_piece(ax, x0, y0, theta0, color):
    """Draw a 90-degree crossing with two perpendicular routes.

    Args:
        ax: Matplotlib axes.
        x0, y0: Start position (studs).
        theta0: Entry heading (degrees).
        color: Track bed color.
    """
    tw = _local_to_world(x0, y0, theta0)

    # Horizontal route (main path): 0 to 16 studs along entry heading.
    _draw_segment(ax, tw(0, 0), tw(16, 0), color)

    # Vertical route (perpendicular): centered at (8, 0), spanning -8 to +8.
    _draw_segment(ax, tw(8, -8), tw(8, 8), color)


def _draw_double_crossover_piece(ax, x0, y0, theta0, color):
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
    """
    tw = _local_to_world(x0, y0, theta0)
    length, lat = DC_LENGTH_STUDS, DC_LATERAL_STUDS

    # Two parallel through-tracks: A->B (y=0) and C->D (y=lat).
    _draw_segment(ax, tw(0.0, 0.0), tw(length, 0.0), color)
    _draw_segment(ax, tw(0.0, lat), tw(length, lat), color)

    # Central scissors: crossover roads A->D and C->B, thin and confined to the
    # middle third so they cross once at the center without stubbing the mainlines.
    _draw_segment(ax, tw(16.0, 0.0), tw(32.0, lat), color,
                  width_stud=DC_CROSS_WIDTH_STUD, alpha=0.75)
    _draw_segment(ax, tw(16.0, lat), tw(32.0, 0.0), color,
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


def _draw_piece(ax, piece_idx, x0, y0, theta0, installed_reversed=False, flip=0):
    """Draw a single piece with proper geometry based on piece type.

    Args:
        ax: Matplotlib axes.
        piece_idx: Piece index (post-refactor 0..6).
        x0, y0: Start position (studs).
        theta0: Entry heading (degrees).
        installed_reversed: Only meaningful for switches (see prior docstring).
        flip: For R40_CURVE only — 0 draws a LEFT arc (+22.5°), 1 draws RIGHT
            (-22.5°). Other piece types ignore this flag.
    """
    color = get_piece_color(piece_idx)

    if piece_idx == PIECE_IDX_STRAIGHT_16:
        _draw_straight_piece(ax, x0, y0, theta0, 16.0, color)
    elif piece_idx == PIECE_IDX_STRAIGHT_24:
        _draw_straight_piece(ax, x0, y0, theta0, 24.0, color)
    elif piece_idx == PIECE_IDX_R40_CURVE:
        sweep = CURVE_ANGLE if not flip else -CURVE_ANGLE
        _draw_curve_piece(ax, x0, y0, theta0, R40, sweep, color)
    elif piece_idx == PIECE_IDX_SWITCH_LEFT:
        _draw_switch_with_install(ax, x0, y0, theta0, 'left', color, color,
                                  installed_reversed)
    elif piece_idx == PIECE_IDX_SWITCH_RIGHT:
        _draw_switch_with_install(ax, x0, y0, theta0, 'right', color, color,
                                  installed_reversed)
    elif piece_idx == PIECE_IDX_CROSS_90:
        _draw_cross90_piece(ax, x0, y0, theta0, color)
    elif piece_idx == PIECE_IDX_DBL_CROSSOVER:
        _draw_double_crossover_piece(ax, x0, y0, theta0, color)
    else:
        # Unknown/inactive index: nothing to draw (valid layouts only carry 0..6).
        pass


def _draw_piece_sequence(ax, pieces, states, *, reversed_flags=None):
    """Draw the piece geometry for one traversal sequence.

    The single shared drawing loop, so a piece-rendering fix lands once.
    A DOUBLE_CROSSOVER is one physical piece threaded twice; its body is
    drawn once via _draw_dc_bodies, never per traversal.

    Args:
        ax: Matplotlib axes.
        pieces: Per-slot catalog indices (list or ndarray), length n.
        states: (n+1, 3) [x, y, theta] poses; theta in degrees.
        reversed_flags: Optional per-slot bool; True installs that slot's
            switch reversed. None means no slot is reversed.
    """
    for i in range(len(pieces)):
        if i + 1 >= len(states):
            break
        piece_idx = int(pieces[i])
        _draw_joint(ax, states[i, 0], states[i, 1], states[i, 2])
        if piece_idx == PIECE_IDX_DBL_CROSSOVER:
            continue
        flip = _r40_flip_from_dtheta(piece_idx, states[i + 1, 2] - states[i, 2])
        installed_reversed = bool(reversed_flags[i]) if reversed_flags is not None else False
        _draw_piece(ax, piece_idx, states[i, 0], states[i, 1], states[i, 2],
                    installed_reversed=installed_reversed, flip=flip)
    _draw_dc_bodies(ax, [int(p) for p in pieces], states)


def _draw_boundary(ax, boundary):
    """Draw the boundary rectangle as a dashed line, if a boundary is given."""
    if boundary is None:
        return
    rect_x = [boundary.min_x, boundary.max_x, boundary.max_x, boundary.min_x, boundary.min_x]
    rect_y = [boundary.min_y, boundary.min_y, boundary.max_y, boundary.max_y, boundary.min_y]
    ax.plot(rect_x, rect_y, "k--", linewidth=1, alpha=0.5)


def _setup_axes(ax):
    """Equal aspect, axis labels (studs), and a light grid."""
    ax.set_aspect("equal")
    ax.set_xlabel("X (studs)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Y (studs)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.grid(True, alpha=0.3)


# =============================================================================
# Unified single-figure renderer: full-size track view + info panel
# =============================================================================

PANEL_FIGSIZE = (14.0, 10.0)
TRACK_WIDTH_RATIO = 4.0
PANEL_WIDTH_RATIO = 1.15
PANEL_METRICS_FONTSIZE = 10
PANEL_LEGEND_FONTSIZE = 9
PANEL_TEXT_XY = (0.02, 0.98)   # metrics-block anchor in panel-axes fraction
PANEL_LEGEND_GAP = 0.02        # axes-fraction gap between metrics block and legend
BRANCH_COLOR = "#e74c3c"
DRIFT_COLOR = "#888888"


def _junction_marker(piece_idx: int) -> Optional[str]:
    """Marker shape for a junction piece: 'D' switch, 's' crossing, None otherwise.

    Single source for both the on-track marker and its legend symbol.
    """
    if piece_idx in SWITCH_INDICES:
        return "D"
    if piece_idx == PIECE_IDX_CROSS_90:
        return "s"
    return None


def _cross_marker_points(layout: MultiPathLayout) -> list[tuple[float, float]]:
    """One (x, y) per PHYSICAL CROSS_90: the crossing-center world pose carried by
    each CrossJunction record. A committed crossing spans two main-loop slots, so
    scanning slots would mark the same piece twice."""
    return [(center_x, center_y)
            for center_x, center_y, _ in (cj.origin for cj in layout.cross_junctions)]


def _drift_segments(layout: MultiPathLayout, closure_tolerance: float,
                    angle_tolerance: float) -> list[np.ndarray]:
    """Post-merge drift states of the paths that FAIL closure at these tolerances.

    The thresholds must be the optimizer's own, or the figure can flag a layout
    the evaluation pipeline considers closed.
    """
    failed = {
        idx for idx, path in enumerate(layout.paths)
        if path.closure_error > closure_tolerance or path.angle_error > angle_tolerance
    }
    return [states for idx, states in layout.get_post_merge_drift() if idx in failed]


def _draw_junction_markers(ax, layout: MultiPathLayout, main_path: TraversalPath) -> None:
    """Diamonds on switch slots, one square per physical crossing center."""
    x, y = main_path.states[:, 0], main_path.states[:, 1]
    for sp in layout.switch_pairs:
        for pos in (sp.in_position, sp.out_position):
            if pos < len(x) - 1 and pos < len(layout.main_loop_pieces):
                piece_idx = layout.main_loop_pieces[pos]
                ax.plot((x[pos] + x[pos + 1]) / 2, (y[pos] + y[pos + 1]) / 2,
                        _junction_marker(piece_idx), color=get_piece_color(piece_idx),
                        markersize=10, markeredgecolor="black", markeredgewidth=1.5,
                        zorder=8)
    cross_color = get_piece_color(PIECE_IDX_CROSS_90)
    for cx, cy in _cross_marker_points(layout):
        ax.plot(cx, cy, _junction_marker(PIECE_IDX_CROSS_90), color=cross_color,
                markersize=8, markeredgecolor="black", markeredgewidth=1, zorder=6)


def _draw_branches(ax, layout: MultiPathLayout) -> bool:
    """Draw each siding's internal pieces plus overlay; IN/OUT switches are skipped
    (already drawn at their main-loop positions). Returns True if any was drawn."""
    drawn = False
    for pieces, states in layout.get_branch_segments():
        if len(states) <= 1:
            continue
        drawn = True
        for k in range(1, len(pieces) - 1):
            sx, sy, stheta = states[k]
            _, _, theta_next = states[k + 1] if k + 1 < len(states) else states[k]
            flip = _r40_flip_from_dtheta(pieces[k], theta_next - stheta)
            _draw_joint(ax, sx, sy, stheta)
            _draw_piece(ax, pieces[k], sx, sy, stheta, flip=flip)
        ax.plot(states[:, 0], states[:, 1], color=BRANCH_COLOR, linewidth=2.5,
                alpha=0.6, zorder=5)
    return drawn


def _draw_track_view(ax, layout: MultiPathLayout, main_path: TraversalPath, boundary, *,
                     closure_tolerance: float, angle_tolerance: float) -> tuple[bool, bool]:
    """Full track on one axes: main loop (with reversed OUT switches), DC bodies,
    branches, junction markers, tolerance-gated FK drift, start/end, boundary.

    Returns (has_branch, has_drift) for the legend.
    """
    out_positions = {sp.out_position for sp in layout.switch_pairs}
    reversed_flags = [i in out_positions for i in range(len(main_path.piece_sequence))]
    _draw_piece_sequence(ax, main_path.piece_sequence, main_path.states,
                         reversed_flags=reversed_flags)

    _draw_junction_markers(ax, layout, main_path)
    has_branch = _draw_branches(ax, layout)

    drift = _drift_segments(layout, closure_tolerance, angle_tolerance)
    for drift_states in drift:
        ax.plot(drift_states[:, 0], drift_states[:, 1], color=DRIFT_COLOR,
                linewidth=1.0, linestyle=":", alpha=0.6, zorder=3)

    x, y = main_path.states[:, 0], main_path.states[:, 1]
    ax.plot(x[0], y[0], "gs", markersize=12, zorder=10)
    ax.plot(x[-1], y[-1], "ro", markersize=12, zorder=10)
    _draw_boundary(ax, boundary)
    return has_branch, bool(drift)


def _panel_metrics_lines(layout: MultiPathLayout, inventory: Optional[dict[str, int]],
                         objectives: Optional[Sequence[float]], cv: Optional[float],
                         closure_tolerance: float, angle_tolerance: float,
                         objective_labels: Optional[Sequence[str]] = None,
                         objective_signs: Optional[Sequence[float]] = None) -> list[str]:
    """Metric rows for the info panel. Objective rows need values, names and
    signs together, the constraint row needs cv; errors are shown against their
    tolerances."""
    used = layout.n_physical_pieces
    total = sum(inventory.values()) if inventory else None
    lines = [f"Pieces: {used}" if total is None else f"Pieces: {used}/{total}"]
    if total is not None:
        lines.append(f"Utilization: {used / total:.1%}")
    if objectives is not None and objective_labels and objective_signs:
        # A search score is not a share of the kit — shown as a bare number so
        # it never reads as a rival utilization percentage.
        lines += [f"{label}: {sign * float(value):.4g}" for label, sign, value
                  in zip(objective_labels, objective_signs, objectives)]
    lines.append(f"Number of unique Paths: {layout.n_paths}")
    lines.append(f"Closure error: {layout.max_closure_error:.2f} / "
                 f"{closure_tolerance:.1f} studs")
    lines.append(f"Angle error: {layout.max_angle_error:.2f} / {angle_tolerance:.1f}°")
    if cv is not None:
        lines.append(f"Constraint violation: {cv:.2f}")
    return lines


def _legend_handles(layout: MultiPathLayout, catalog: TrackCatalog,
                    inventory: Optional[dict[str, int]], *, has_branch: bool,
                    has_drift: bool, has_boundary: bool) -> list[Patch | plt.Line2D]:
    """One row per catalog type (used/capacity, zero-usage rows included), then
    Start/End and the overlays actually present on the figure."""
    counts = count_pieces(layout)
    capacity = catalog.inventory_by_index(inventory) if inventory else {}
    # Name column sized from the catalog: a longer piece id must not silently
    # push the count column out of alignment.
    name_width = max(len(piece_id) for piece_id in catalog.index_to_id.values()) + 1
    handles = []
    for piece_idx in sorted(catalog.index_to_id):
        name = catalog.index_to_id[piece_idx]
        used = counts.get(piece_idx, 0)
        cap = capacity.get(piece_idx)
        label = f"{name:<{name_width}}{used:>4}" + (f"/{cap}" if cap is not None else "")
        color = get_piece_color(piece_idx)
        marker = _junction_marker(piece_idx)
        if marker is None:
            handles.append(Patch(facecolor=color, edgecolor="black", label=label))
        else:
            handles.append(plt.Line2D([0], [0], marker=marker, color=color,
                                      linestyle="-", linewidth=3, markersize=6,
                                      markeredgecolor="black", label=label))
    handles += [
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="g",
                   markersize=10, label="Start"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="r",
                   markersize=10, label="End"),
    ]
    if has_branch:
        handles.append(plt.Line2D([0], [0], color=BRANCH_COLOR, linewidth=2.5,
                                  label="Branch"))
    if has_drift:
        handles.append(plt.Line2D([0], [0], color=DRIFT_COLOR, linestyle=":",
                                  label="FK drift (failed merge)"))
    if has_boundary:
        handles.append(plt.Line2D([0], [0], color="k", linestyle="--", label="Boundary"))
    return handles


def _render_info_panel(fig, ax_panel, metrics: list[str],
                       handles: list[Patch | plt.Line2D]) -> None:
    """Metrics block at the panel top; legend anchored below the block's MEASURED
    bottom edge, so adding a metrics row can never overlap the legend.

    Measuring needs a renderer, not a rendered figure: axes positions are already
    settled by tight_layout, so the text extent is final before anything is drawn.
    """
    text_x, text_y = PANEL_TEXT_XY
    metrics_text = ax_panel.text(
        text_x, text_y, "\n".join(metrics), transform=ax_panel.transAxes,
        fontsize=PANEL_METRICS_FONTSIZE, verticalalignment="top", family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )
    bbox = metrics_text.get_window_extent(fig.canvas.get_renderer())
    _, legend_top = ax_panel.transAxes.inverted().transform((bbox.x0, bbox.y0))
    ax_panel.legend(handles=handles, loc="upper left",
                    bbox_to_anchor=(text_x, legend_top - PANEL_LEGEND_GAP),
                    frameon=False,
                    prop={"family": "monospace", "size": PANEL_LEGEND_FONTSIZE})


def plot_layout(
    layout: MultiPathLayout,
    catalog: TrackCatalog,
    boundary: Optional[BoundaryConfig] = None,
    title: str = "Track Layout",
    save_path: Optional[Union[str, Path]] = None,
    *,
    closure_tolerance: float,
    angle_tolerance: float,
    inventory: Optional[dict[str, int]] = None,
    objectives: Optional[Sequence[float]] = None,
    cv: Optional[float] = None,
    objective_labels: Optional[Sequence[str]] = None,
    objective_signs: Optional[Sequence[float]] = None,
) -> Figure:
    """Render one full-size track view with a metrics + legend panel beside it.

    The closure/angle tolerances are keyword-only and required: the drift
    overlay must be gated by the optimizer's own thresholds, never a renderer
    default. Objective rows appear only when values, names and signs are all
    given, the constraint row only when ``cv`` is given (pass it for infeasible
    individuals).

    Args:
        layout: Decoded multi-path layout.
        catalog: Piece catalog; supplies ids, colors, and index order.
        boundary: Optional boundary rectangle to draw.
        title: Plot title; metrics live in the panel, not here.
        save_path: Optional path to save the figure as PNG.
        closure_tolerance: Position closure tolerance in studs, from config.
        angle_tolerance: Angle closure tolerance in degrees, from config.
        inventory: Optional {piece_id: count} capacities for panel and legend.
        objectives: Optional raw pymoo F row, as stored (minimized).
        cv: Optional constraint violation of the rendered individual.
        objective_labels: One name per objective, in F order.
        objective_signs: Factor per objective turning its stored minimum into
            the value the panel shows.

    Returns:
        Matplotlib figure.
    """
    fig = plt.figure(figsize=PANEL_FIGSIZE)
    main_path = layout.paths[0] if layout.paths else None

    if main_path is None or len(main_path.states) <= 1:
        ax = fig.add_subplot()
        ax.text(0.5, 0.5, "Empty Layout", ha="center", va="center", fontsize=16)
        ax.set_title(title)
    else:
        grid = fig.add_gridspec(1, 2, width_ratios=(TRACK_WIDTH_RATIO, PANEL_WIDTH_RATIO))
        ax = fig.add_subplot(grid[0, 0])
        ax_panel = fig.add_subplot(grid[0, 1])
        ax_panel.set_axis_off()

        has_branch, has_drift = _draw_track_view(
            ax, layout, main_path, boundary,
            closure_tolerance=closure_tolerance, angle_tolerance=angle_tolerance,
        )
        _setup_axes(ax)
        ax.set_title(title, fontsize=TITLE_FONTSIZE)

        metrics = _panel_metrics_lines(layout, inventory, objectives, cv,
                                       closure_tolerance, angle_tolerance,
                                       objective_labels=objective_labels,
                                       objective_signs=objective_signs)
        handles = _legend_handles(layout, catalog, inventory, has_branch=has_branch,
                                  has_drift=has_drift, has_boundary=boundary is not None)
        fig.tight_layout()
        _render_info_panel(fig, ax_panel, metrics, handles)

    if save_path is not None:
        fig.savefig(save_path, format="png", dpi=150, bbox_inches="tight")
    return fig
