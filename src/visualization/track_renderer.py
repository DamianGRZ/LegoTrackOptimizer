"""Visualization functions for track layouts and optimization results."""

from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from src.config import BoundaryConfig
from src.catalog import TrackCatalog
from src.geometry import Layout
from src.types import MultiPathLayout

# Import geometry helpers from track models (do not modify these)
from src.lego_track_models import (
    R40,
    RAIL_OFFSET,
    CURVE_ANGLE,
    SWITCH_LEN,
    ARC1_ANGLE,
    ARC2_ANGLE,
    COL_RAIL,
    BED_LW,
    RAIL_LW,
    N_ARC_PTS,
    arc_points,
    offset_path,
)

# Color palette for piece types (index-based)
# Indices: 0=STRAIGHT_16, 1=STRAIGHT_24, 2=R40_LEFT, 3=R40_RIGHT,
#          4=CROSS_90, 5=SW_LEFT_IN, 6=SW_LEFT_OUT, 7=SW_RIGHT_IN, 8=SW_RIGHT_OUT,
#          9=DOUBLE_CROSSOVER
PIECE_COLORS = {
    0: "#3498db",  # STRAIGHT_16 - Blue
    1: "#2980b9",  # STRAIGHT_24 - Darker Blue
    2: "#27ae60",  # R40_LEFT - Green
    3: "#16a085",  # R40_RIGHT - Teal
    4: "#9b59b6",  # CROSS_90 - Purple
    5: "#e74c3c",  # SW_LEFT_IN - Red
    6: "#c0392b",  # SW_LEFT_OUT - Dark Red
    7: "#e67e22",  # SW_RIGHT_IN - Orange
    8: "#d35400",  # SW_RIGHT_OUT - Dark Orange
    9: "#8e44ad",  # DOUBLE_CROSSOVER - Deep Purple
}

# Fallback colors for unknown pieces
FALLBACK_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

# Switch piece indices
SWITCH_INDICES = {5, 6, 7, 8}
CROSSING_INDICES = {4, 9}

# Shortened names for legend
PIECE_SHORT_NAMES = {
    0: "STRAIGHT_16",
    1: "STRAIGHT_24",
    2: "R40_LEFT",
    3: "R40_RIGHT",
    4: "CROSS_90",
    5: "SW_LEFT_IN",
    6: "SW_LEFT_OUT",
    7: "SW_RIGHT_IN",
    8: "SW_RIGHT_OUT",
    9: "DBL_CROSSOVER",
}


# Piece type constants for rendering
PIECE_IDX_STRAIGHT_16 = 0
PIECE_IDX_STRAIGHT_24 = 1
PIECE_IDX_R40_LEFT = 2
PIECE_IDX_R40_RIGHT = 3
PIECE_IDX_CROSS_90 = 4
PIECE_IDX_SW_LEFT_IN = 5
PIECE_IDX_SW_LEFT_OUT = 6
PIECE_IDX_SW_RIGHT_IN = 7
PIECE_IDX_SW_RIGHT_OUT = 8
PIECE_IDX_DBL_CROSSOVER = 9

# Piece lengths for straight pieces
STRAIGHT_LENGTHS = {
    PIECE_IDX_STRAIGHT_16: 16.0,
    PIECE_IDX_STRAIGHT_24: 24.0,
}


def get_piece_color(piece_idx: int) -> str:
    """Get color for a piece by index."""
    if piece_idx in PIECE_COLORS:
        return PIECE_COLORS[piece_idx]
    return FALLBACK_COLORS[piece_idx % len(FALLBACK_COLORS)]


def _draw_track_bed(ax, x, y, color, lw=BED_LW, alpha=0.88, zorder=2):
    """Draw thick colored track bed."""
    ax.plot(x, y, color=color, lw=lw, solid_capstyle='butt',
            alpha=alpha, zorder=zorder)


def _draw_rails(ax, x, y, offset=RAIL_OFFSET, zorder=3):
    """Draw two thin gray rails offset from centerline."""
    lx, ly = offset_path(x, y, +offset)
    rx, ry = offset_path(x, y, -offset)
    ax.plot(lx, ly, color=COL_RAIL, lw=RAIL_LW, solid_capstyle='butt', zorder=zorder)
    ax.plot(rx, ry, color=COL_RAIL, lw=RAIL_LW, solid_capstyle='butt', zorder=zorder)


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

    x = np.array([x0, x1])
    y = np.array([y0, y1])

    _draw_track_bed(ax, x, y, color)
    if draw_rails_flag:
        _draw_rails(ax, x, y)


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
    # Arc center is perpendicular to entry heading
    # For left turn (positive sweep): center is to the left (+90 deg)
    # For right turn (negative sweep): center is to the right (-90 deg)
    if sweep_deg > 0:
        # Left turn - center to the left of entry direction
        center_offset_angle = theta0 + 90
    else:
        # Right turn - center to the right of entry direction
        center_offset_angle = theta0 - 90

    center_offset_rad = np.radians(center_offset_angle)
    cx = x0 + radius * np.cos(center_offset_rad)
    cy = y0 + radius * np.sin(center_offset_rad)

    # Start angle is from center to entry point
    start_ang = np.degrees(np.arctan2(y0 - cy, x0 - cx))

    # Generate arc points for centerline
    xc, yc = arc_points(cx, cy, radius, start_ang, sweep_deg, n=N_ARC_PTS)

    _draw_track_bed(ax, xc, yc, color)

    if draw_rails_flag:
        # Inner and outer rails
        xi, yi = arc_points(cx, cy, radius - RAIL_OFFSET, start_ang, sweep_deg, n=N_ARC_PTS)
        xo, yo = arc_points(cx, cy, radius + RAIL_OFFSET, start_ang, sweep_deg, n=N_ARC_PTS)
        ax.plot(xi, yi, color=COL_RAIL, lw=RAIL_LW, solid_capstyle='butt', zorder=3)
        ax.plot(xo, yo, color=COL_RAIL, lw=RAIL_LW, solid_capstyle='butt', zorder=3)


def _draw_switch_piece(ax, x0, y0, theta0, direction, color_main, color_branch, draw_rails_flag=True):
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
    theta_rad = np.radians(theta0)
    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)

    def transform_point(lx, ly):
        """Transform local coordinates to world."""
        return (x0 + lx * cos_t - ly * sin_t,
                y0 + lx * sin_t + ly * cos_t)

    # Main straight path (32 studs)
    x_main = np.array([x0, x0 + SWITCH_LEN * cos_t])
    y_main = np.array([y0, y0 + SWITCH_LEN * sin_t])
    _draw_track_bed(ax, x_main, y_main, color_main)
    if draw_rails_flag:
        _draw_rails(ax, x_main, y_main)

    flip = 1 if direction == 'left' else -1

    # Arc 1: outward curve
    c1_local = (0.0, flip * R40)
    c1x, c1y = transform_point(*c1_local)
    a1_start = theta0 - 90 * flip
    a1_sweep = flip * ARC1_ANGLE

    x1, y1 = arc_points(c1x, c1y, R40, a1_start, a1_sweep, n=30)
    _draw_track_bed(ax, x1, y1, color_branch, lw=8)

    if draw_rails_flag:
        xi1, yi1 = arc_points(c1x, c1y, R40 - RAIL_OFFSET, a1_start, a1_sweep, n=30)
        xo1, yo1 = arc_points(c1x, c1y, R40 + RAIL_OFFSET, a1_start, a1_sweep, n=30)
        ax.plot(xi1, yi1, color=COL_RAIL, lw=RAIL_LW, solid_capstyle='butt', zorder=3)
        ax.plot(xo1, yo1, color=COL_RAIL, lw=RAIL_LW, solid_capstyle='butt', zorder=3)

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

    x2, y2 = arc_points(c2x, c2y, R40, a2_start, a2_sweep, n=30)
    _draw_track_bed(ax, x2, y2, color_branch, lw=8)

    if draw_rails_flag:
        xi2, yi2 = arc_points(c2x, c2y, R40 - RAIL_OFFSET, a2_start, a2_sweep, n=30)
        xo2, yo2 = arc_points(c2x, c2y, R40 + RAIL_OFFSET, a2_start, a2_sweep, n=30)
        ax.plot(xi2, yi2, color=COL_RAIL, lw=RAIL_LW, solid_capstyle='butt', zorder=3)
        ax.plot(xo2, yo2, color=COL_RAIL, lw=RAIL_LW, solid_capstyle='butt', zorder=3)


def _draw_cross90_piece(ax, x0, y0, theta0, color, draw_rails_flag=True):
    """Draw a 90-degree crossing with two perpendicular routes.

    Args:
        ax: Matplotlib axes.
        x0, y0: Start position (studs).
        theta0: Entry heading (degrees).
        color: Track bed color.
        draw_rails_flag: Whether to draw rails.
    """
    theta_rad = np.radians(theta0)
    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)

    def transform_point(lx, ly):
        """Transform local coordinates to world."""
        return (x0 + lx * cos_t - ly * sin_t,
                y0 + lx * sin_t + ly * cos_t)

    # Horizontal route (main path): 0 to 16 studs along entry heading
    h_start = transform_point(0, 0)
    h_end = transform_point(16, 0)
    x_horiz = np.array([h_start[0], h_end[0]])
    y_horiz = np.array([h_start[1], h_end[1]])
    _draw_track_bed(ax, x_horiz, y_horiz, color)
    if draw_rails_flag:
        _draw_rails(ax, x_horiz, y_horiz)

    # Vertical route (perpendicular): centered at (8, 0), spanning -8 to +8 perpendicular
    v_start = transform_point(8, -8)
    v_end = transform_point(8, 8)
    x_vert = np.array([v_start[0], v_end[0]])
    y_vert = np.array([v_start[1], v_end[1]])
    _draw_track_bed(ax, x_vert, y_vert, color)
    if draw_rails_flag:
        _draw_rails(ax, x_vert, y_vert)


def _draw_double_crossover_piece(ax, x0, y0, theta0, color, draw_rails_flag=True):
    """Draw a double crossover with two parallel tracks and diagonal crossings.

    Args:
        ax: Matplotlib axes.
        x0, y0: Start position (studs) - entry of track 1.
        theta0: Entry heading (degrees).
        color: Track bed color.
        draw_rails_flag: Whether to draw rails.
    """
    theta_rad = np.radians(theta0)
    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)

    def transform_point(lx, ly):
        """Transform local coordinates to world."""
        return (x0 + lx * cos_t - ly * sin_t,
                y0 + lx * sin_t + ly * cos_t)

    # Track 1 (main): 0 to 48 studs at y=0
    t1_start = transform_point(0, 0)
    t1_end = transform_point(48, 0)
    x_track1 = np.array([t1_start[0], t1_end[0]])
    y_track1 = np.array([t1_start[1], t1_end[1]])
    _draw_track_bed(ax, x_track1, y_track1, color)
    if draw_rails_flag:
        _draw_rails(ax, x_track1, y_track1)

    # Track 2 (parallel): 0 to 48 studs at y=16
    t2_start = transform_point(0, 16)
    t2_end = transform_point(48, 16)
    x_track2 = np.array([t2_start[0], t2_end[0]])
    y_track2 = np.array([t2_start[1], t2_end[1]])
    _draw_track_bed(ax, x_track2, y_track2, color)
    if draw_rails_flag:
        _draw_rails(ax, x_track2, y_track2)

    # Diagonal crossovers (X-pattern in the middle section)
    # Crossover 1: from track 1 to track 2 (going up-right)
    # Approximate geometry: diagonal from ~(12, 0) to ~(36, 16)
    c1_start = transform_point(12, 0)
    c1_end = transform_point(36, 16)
    x_cross1 = np.array([c1_start[0], c1_end[0]])
    y_cross1 = np.array([c1_start[1], c1_end[1]])
    _draw_track_bed(ax, x_cross1, y_cross1, color, lw=6, alpha=0.7)
    if draw_rails_flag:
        _draw_rails(ax, x_cross1, y_cross1)

    # Crossover 2: from track 2 to track 1 (going down-right)
    # Approximate geometry: diagonal from ~(12, 16) to ~(36, 0)
    c2_start = transform_point(12, 16)
    c2_end = transform_point(36, 0)
    x_cross2 = np.array([c2_start[0], c2_end[0]])
    y_cross2 = np.array([c2_start[1], c2_end[1]])
    _draw_track_bed(ax, x_cross2, y_cross2, color, lw=6, alpha=0.7)
    if draw_rails_flag:
        _draw_rails(ax, x_cross2, y_cross2)


def _draw_piece(ax, piece_idx, x0, y0, theta0, draw_rails_flag=True):
    """Draw a single piece with proper geometry based on piece type.

    Args:
        ax: Matplotlib axes.
        piece_idx: Piece index (0-9).
        x0, y0: Start position (studs).
        theta0: Entry heading (degrees).
        draw_rails_flag: Whether to draw rails.
    """
    color = get_piece_color(piece_idx)

    if piece_idx == PIECE_IDX_STRAIGHT_16:
        _draw_straight_piece(ax, x0, y0, theta0, 16.0, color, draw_rails_flag)
    elif piece_idx == PIECE_IDX_STRAIGHT_24:
        _draw_straight_piece(ax, x0, y0, theta0, 24.0, color, draw_rails_flag)
    elif piece_idx == PIECE_IDX_R40_LEFT:
        _draw_curve_piece(ax, x0, y0, theta0, R40, CURVE_ANGLE, color, draw_rails_flag)
    elif piece_idx == PIECE_IDX_R40_RIGHT:
        _draw_curve_piece(ax, x0, y0, theta0, R40, -CURVE_ANGLE, color, draw_rails_flag)
    elif piece_idx in (PIECE_IDX_SW_LEFT_IN, PIECE_IDX_SW_LEFT_OUT):
        color_branch = PIECE_COLORS.get(piece_idx, color)
        _draw_switch_piece(ax, x0, y0, theta0, 'left', color, color_branch, draw_rails_flag)
    elif piece_idx in (PIECE_IDX_SW_RIGHT_IN, PIECE_IDX_SW_RIGHT_OUT):
        color_branch = PIECE_COLORS.get(piece_idx, color)
        _draw_switch_piece(ax, x0, y0, theta0, 'right', color, color_branch, draw_rails_flag)
    elif piece_idx == PIECE_IDX_CROSS_90:
        _draw_cross90_piece(ax, x0, y0, theta0, color, draw_rails_flag)
    elif piece_idx == PIECE_IDX_DBL_CROSSOVER:
        _draw_double_crossover_piece(ax, x0, y0, theta0, color, draw_rails_flag)
    else:
        # Fallback: simple line to next state (will be handled by caller)
        pass


def get_piece_short_name(piece_idx: int) -> str:
    """Get short name for a piece by index."""
    return PIECE_SHORT_NAMES.get(piece_idx, f"PIECE_{piece_idx}")


def plot_layout(
    layout: Layout,
    catalog: TrackCatalog,
    boundary: Optional[BoundaryConfig] = None,
    title: str = "Track Layout",
    save_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """Plot track layout with colored pieces and connection dots.

    Each piece type has a different color. Switches are shown with thicker lines
    and diamond markers. Crossings shown with square markers.
    Start is marked with green square, end with red circle.

    Args:
        layout: Track layout to plot.
        catalog: Track catalog for piece names.
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

    # Extract coordinates
    x = layout.states[:, 0]
    y = layout.states[:, 1]
    theta = layout.states[:, 2]

    # Track piece types for legend (by index for consistent ordering)
    piece_indices_seen = set()
    switch_positions = []
    crossing_positions = []

    # Plot each piece with proper geometry
    for i in range(layout.n_pieces):
        piece_idx = int(layout.indices[i])

        # Track special positions
        if piece_idx in SWITCH_INDICES:
            switch_positions.append(i)
        elif piece_idx in CROSSING_INDICES:
            crossing_positions.append(i)

        # Draw piece with proper geometry (arcs for curves, etc.)
        _draw_piece(ax, piece_idx, x[i], y[i], theta[i], draw_rails_flag=True)

        piece_indices_seen.add(piece_idx)

    # Mark switch positions with diamond markers
    for i in switch_positions:
        piece_idx = int(layout.indices[i])
        color = get_piece_color(piece_idx)
        mid_x = (x[i] + x[i + 1]) / 2
        mid_y = (y[i] + y[i + 1]) / 2
        ax.plot(mid_x, mid_y, "D", color=color, markersize=8, markeredgecolor="black",
                markeredgewidth=1, zorder=6)

    # Mark crossing positions with square markers
    for i in crossing_positions:
        piece_idx = int(layout.indices[i])
        color = get_piece_color(piece_idx)
        mid_x = (x[i] + x[i + 1]) / 2
        mid_y = (y[i] + y[i + 1]) / 2
        ax.plot(mid_x, mid_y, "s", color=color, markersize=8, markeredgecolor="black",
                markeredgewidth=1, zorder=6)

    # Mark start as green square and end as red circle
    ax.plot(x[0], y[0], "gs", markersize=12, label="Start", zorder=10)
    ax.plot(x[-1], y[-1], "ro", markersize=12, label="End", zorder=10)

    # Draw boundary if provided
    if boundary is not None:
        rect_x = [boundary.min_x, boundary.max_x, boundary.max_x, boundary.min_x, boundary.min_x]
        rect_y = [boundary.min_y, boundary.min_y, boundary.max_y, boundary.max_y, boundary.min_y]
        ax.plot(rect_x, rect_y, "k--", linewidth=1, alpha=0.5, label="Boundary")

    # Set equal aspect ratio
    ax.set_aspect("equal")

    # Labels and grid
    ax.set_xlabel("X (studs)", fontsize=11)
    ax.set_ylabel("Y (studs)", fontsize=11)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)

    # Create legend with piece types (sorted by index)
    legend_elements = []
    for piece_idx in sorted(piece_indices_seen):
        color = get_piece_color(piece_idx)
        name = get_piece_short_name(piece_idx)
        if piece_idx in SWITCH_INDICES:
            # Diamond marker for switches
            legend_elements.append(
                plt.Line2D([0], [0], marker="D", color=color, linestyle="-",
                          linewidth=3, markersize=6, markeredgecolor="black", label=name)
            )
        elif piece_idx in CROSSING_INDICES:
            # Square marker for crossings
            legend_elements.append(
                plt.Line2D([0], [0], marker="s", color=color, linestyle="-",
                          linewidth=3, markersize=6, markeredgecolor="black", label=name)
            )
        else:
            legend_elements.append(Patch(facecolor=color, edgecolor="black", label=name))

    legend_elements.extend([
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="g", markersize=10, label="Start"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="r", markersize=10, label="End"),
    ])

    if boundary is not None:
        legend_elements.append(plt.Line2D([0], [0], color="k", linestyle="--", label="Boundary"))

    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

    # Add metrics text
    n_switches = len([i for i in layout.indices if int(i) in SWITCH_INDICES])
    n_crossings = len([i for i in layout.indices if int(i) in CROSSING_INDICES])
    metrics_text = (
        f"Pieces: {layout.n_pieces}\n"
        f"Switches: {n_switches}\n"
        f"Crossings: {n_crossings}\n"
        f"Closure Error: {layout.closure_error:.2f} studs\n"
        f"Angle Error: {layout.angle_error:.2f}°\n"
        f"Area: {layout.area:.0f} studs²"
    )
    ax.text(
        0.02,
        0.98,
        metrics_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()

    # Save to PNG if path provided
    if save_path is not None:
        fig.savefig(save_path, format='png', dpi=150, bbox_inches='tight')

    return fig


# Path colors for multi-path visualization
PATH_COLORS = [
    "#1f77b4",  # Blue - main path
    "#ff7f0e",  # Orange - branch path 1
    "#2ca02c",  # Green - branch path 2
    "#d62728",  # Red - branch path 3
    "#9467bd",  # Purple
    "#8c564b",  # Brown
]


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
        catalog: Track catalog for piece names.
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

    if n_paths == 1:
        axes = np.array([axes]).flatten()
    else:
        axes = np.array(axes).flatten()

    # Hide unused subplots
    for i in range(n_paths + 1, len(axes)):
        axes[i].set_visible(False)

    # Combined view (first subplot)
    ax_combined = axes[0]
    _plot_combined_paths(ax_combined, layout, catalog, boundary, title)

    # Individual path views
    for i, path in enumerate(layout.paths):
        if i + 1 < len(axes):
            ax = axes[i + 1]
            _plot_single_path(ax, path, catalog, boundary, i)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, format='png', dpi=150, bbox_inches='tight')

    return fig


def _plot_combined_paths(ax, layout: MultiPathLayout, catalog, boundary, title):
    """Plot main path with branch sections highlighted differently."""
    # Draw main path (path 0) with proper piece geometry
    main_path = layout.paths[0] if layout.paths else None
    if main_path is None or len(main_path.states) <= 1:
        return

    x = main_path.states[:, 0]
    y = main_path.states[:, 1]
    theta = main_path.states[:, 2]

    for i in range(len(main_path.piece_sequence)):
        if i < len(x) - 1:
            piece_idx = main_path.piece_sequence[i]
            _draw_piece(ax, piece_idx, x[i], y[i], theta[i], draw_rails_flag=True)

    ax.plot(x[0], y[0], "gs", markersize=10, zorder=10)
    ax.plot(x[-1], y[-1], "ro", markersize=10, zorder=10)

    # Mark switch positions on main path (from switch pairs, not raw scan)
    for sp in layout.switch_pairs:
        for pos, idx in [(sp.in_position, sp.in_switch_idx), (sp.out_position, sp.out_switch_idx)]:
            if pos < len(x) - 1:
                sw_color = get_piece_color(idx)
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

        for k, piece_idx in enumerate(pieces):
            sx, sy, stheta = states[k]
            _draw_piece(ax, piece_idx, sx, sy, stheta, draw_rails_flag=True)

        ax.plot(
            states[:, 0], states[:, 1],
            color='#e74c3c', linewidth=2.5, linestyle='-',
            alpha=0.6, zorder=5,
            label='Branch' if seg_idx == 0 else None,
        )

    # Draw boundary
    if boundary is not None:
        rect_x = [boundary.min_x, boundary.max_x, boundary.max_x, boundary.min_x, boundary.min_x]
        rect_y = [boundary.min_y, boundary.min_y, boundary.max_y, boundary.max_y, boundary.min_y]
        ax.plot(rect_x, rect_y, "k--", linewidth=1, alpha=0.5)

    ax.set_aspect("equal")
    ax.set_xlabel("X (studs)")
    ax.set_ylabel("Y (studs)")
    ax.set_title(f"{title}\n(All {len(layout.paths)} Paths)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")

    # Metrics
    n_switches = sum(1 for p in layout.main_loop_pieces if p in SWITCH_INDICES)
    metrics = (
        f"Pieces: {layout.n_pieces}\n"
        f"Switches: {n_switches}\n"
        f"Switch Pairs: {layout.n_switch_pairs}\n"
        f"Max Closure: {layout.max_closure_error:.2f}\n"
        f"Max Angle: {layout.max_angle_error:.2f}°"
    )
    ax.text(0.02, 0.98, metrics, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))


def _plot_single_path(ax, path, catalog, boundary, path_idx):
    """Plot a single traversal path with switch markers."""
    if len(path.states) <= 1:
        ax.text(0.5, 0.5, "Empty Path", ha="center", va="center")
        return

    x = path.states[:, 0]
    y = path.states[:, 1]
    theta = path.states[:, 2]

    # Identify switch positions in this path
    switch_positions = []
    for i, piece_idx in enumerate(path.piece_sequence):
        if piece_idx in SWITCH_INDICES:
            switch_positions.append(i)

    # Plot each piece with proper geometry
    for i in range(len(path.piece_sequence)):
        if i < len(x) - 1:
            piece_idx = path.piece_sequence[i]
            _draw_piece(ax, piece_idx, x[i], y[i], theta[i], draw_rails_flag=True)

    # Mark switch positions with diamond markers
    for sw_pos in switch_positions:
        if sw_pos < len(x) - 1:
            piece_idx = path.piece_sequence[sw_pos]
            sw_color = get_piece_color(piece_idx)
            mid_x = (x[sw_pos] + x[sw_pos + 1]) / 2
            mid_y = (y[sw_pos] + y[sw_pos + 1]) / 2
            ax.plot(mid_x, mid_y, "D", color=sw_color, markersize=8,
                   markeredgecolor="black", markeredgewidth=1, zorder=6)

    # Mark start/end
    ax.plot(x[0], y[0], "gs", markersize=8, zorder=10)
    ax.plot(x[-1], y[-1], "ro", markersize=8, zorder=10)

    # Draw boundary
    if boundary is not None:
        rect_x = [boundary.min_x, boundary.max_x, boundary.max_x, boundary.min_x, boundary.min_x]
        rect_y = [boundary.min_y, boundary.min_y, boundary.max_y, boundary.max_y, boundary.min_y]
        ax.plot(rect_x, rect_y, "k--", linewidth=1, alpha=0.5)

    ax.set_aspect("equal")
    ax.set_xlabel("X (studs)")
    ax.set_ylabel("Y (studs)")

    closed_str = "CLOSED" if path.is_closed else "OPEN"
    n_switches = len(switch_positions)
    ax.set_title(f"Path {path_idx}: {path.describe_route()}\n"
                 f"({path.n_pieces} pieces, {n_switches} switches, {closed_str})")
    ax.grid(True, alpha=0.3)

    # Metrics
    metrics = (
        f"Closure: {path.closure_error:.2f}\n"
        f"Angle: {path.angle_error:.2f}°"
    )
    ax.text(0.02, 0.98, metrics, transform=ax.transAxes, fontsize=8,
            verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
