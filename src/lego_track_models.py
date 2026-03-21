#!/usr/bin/env python3
"""
LEGO Track Piece Visualization — Matplotlib Models
====================================================
Exact geometry from community research (MacFreek, L-Gauge, Transponderings, Fx Bricks).

Pieces from YAML:
  - Straight (53401): 16 studs, 0°
  - Curved R40 (53400): 22.5° arc, R=40 studs
  - Switch Left (53407): 32 studs, branch 22.5° via compound S-curve
  - Switch Right (53404): mirror of left
  
Port coordinates (stud units, origin at Port A):
  Straight:  A=(0,0,0°)  B=(16,0,0°)
  Curve:     A=(0,0,0°)  B=(15.307,3.045,22.5°) [left-turning]
  Switch L:  A=(0,0,0°)  B=(32,0,0°)  C=(32.693,12.955,22.5°)
  Switch R:  A=(0,0,0°)  B=(32,0,0°)  C=(32.693,-12.955,-22.5°)

Branch S-curve: Arc1 = 36.87° outward (R40), Arc2 = 14.37° inward (R40)
Based on 3-4-5 Pythagorean triple scaled to R=40.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import math

# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════
R40 = 40.0                          # curve radius in studs
RAIL_OFFSET = 2.5                   # half gauge (5 studs / 2)
CURVE_ANGLE = 22.5                  # degrees per R40 piece
SWITCH_LEN = 32.0                   # switch straight path length
ARC1_ANGLE = math.degrees(math.atan2(3, 4))  # 36.87° (3-4-5 triple)
ARC2_ANGLE = ARC1_ANGLE - CURVE_ANGLE         # 14.37°

# Colors matching our matplotlib GUI
COL_STRAIGHT_BED = '#2c3e50'        # dark blue-gray
COL_CURVE_BED    = '#e67e22'        # orange
COL_SWITCH_MAIN  = '#8e44ad'        # purple
COL_BRANCH_BED   = '#27ae60'        # green
COL_RAIL         = '#95a5a6'        # gray
COL_PORT_A       = '#e74c3c'        # red
COL_PORT_B       = '#2980b9'        # blue
COL_PORT_C       = '#27ae60'        # green
COL_DIM          = '#7f8c8d'        # dimension gray

BED_LW = 10                         # track bed line width
RAIL_LW = 1.8                       # rail line width
N_ARC_PTS = 60                      # arc interpolation points


# ═══════════════════════════════════════════════════════════════
# GEOMETRY HELPERS
# ═══════════════════════════════════════════════════════════════

def arc_points(cx, cy, r, start_deg, sweep_deg, n=N_ARC_PTS):
    """Generate (x, y) arrays along a circular arc."""
    angles = np.linspace(np.radians(start_deg),
                         np.radians(start_deg + sweep_deg), n)
    return cx + r * np.cos(angles), cy + r * np.sin(angles)


def offset_path(x, y, offset):
    """Offset a polyline by ±offset studs perpendicular to path direction."""
    dx = np.gradient(x)
    dy = np.gradient(y)
    length = np.sqrt(dx**2 + dy**2)
    length[length == 0] = 1e-9
    nx = -dy / length  # perpendicular normal
    ny =  dx / length
    return x + offset * nx, y + offset * ny


def draw_track_bed(ax, x, y, color, lw=BED_LW, alpha=0.88, zorder=2):
    """Draw thick colored track bed."""
    ax.plot(x, y, color=color, lw=lw, solid_capstyle='butt',
            alpha=alpha, zorder=zorder)


def draw_rails(ax, x, y, offset=RAIL_OFFSET, zorder=3):
    """Draw two thin gray rails offset from centerline."""
    lx, ly = offset_path(x, y, +offset)
    rx, ry = offset_path(x, y, -offset)
    ax.plot(lx, ly, color=COL_RAIL, lw=RAIL_LW, solid_capstyle='butt', zorder=zorder)
    ax.plot(rx, ry, color=COL_RAIL, lw=RAIL_LW, solid_capstyle='butt', zorder=zorder)


def draw_port(ax, x, y, angle_deg, label, color, size=12):
    """Draw port marker: filled circle + direction arrow + label."""
    # Port circle
    ax.plot(x, y, 'o', color='white', markersize=size + 2,
            markeredgecolor=color, markeredgewidth=2.2, zorder=8)
    ax.text(x, y, label, ha='center', va='center', fontsize=8,
            fontweight='bold', color=color, zorder=9)

    # Direction arrow (outward from port)
    arad = np.radians(angle_deg)
    arrow_len = 5
    ax.annotate('', xy=(x + arrow_len * np.cos(arad),
                        y + arrow_len * np.sin(arad)),
                xytext=(x + 1.8 * np.cos(arad),
                        y + 1.8 * np.sin(arad)),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=1.8, mutation_scale=12),
                zorder=8)


def draw_dim_line(ax, x1, y1, x2, y2, label, offset_perp=0, fontsize=8):
    """Draw a dimension annotation line."""
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='<->', color=COL_DIM,
                                lw=0.8, connectionstyle='arc3,rad=0'))
    ax.text(mx, my + offset_perp, label, ha='center', va='bottom',
            fontsize=fontsize, color=COL_DIM, fontstyle='italic',
            fontfamily='monospace')


def setup_subplot(ax, title, subtitle, xlim, ylim):
    """Configure a subplot with grid and labels."""
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.25, linewidth=0.4, color='#cccccc')
    ax.set_axisbelow(True)

    # Heavier axes at origin
    ax.axhline(0, color='#bbbbbb', lw=0.6, alpha=0.5, zorder=0)
    ax.axvline(0, color='#bbbbbb', lw=0.6, alpha=0.5, zorder=0)

    ax.set_xlabel('X (studs)', fontsize=8, color='#888')
    ax.set_ylabel('Y (studs)', fontsize=8, color='#888')
    ax.set_title(title, fontsize=11, fontweight='bold', color='#2c3e50', pad=14)
    ax.text(0.5, 0.97, subtitle, transform=ax.transAxes,
            ha='center', va='top', fontsize=7, color='#999',
            fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='none', alpha=0.7))
    ax.tick_params(labelsize=7, colors='#aaa')


# ═══════════════════════════════════════════════════════════════
# PIECE RENDERERS
# ═══════════════════════════════════════════════════════════════

def draw_straight(ax):
    """Straight Track — 53401 — 16 studs, 0° turn."""
    setup_subplot(ax, 'Straight Track — 53401',
                  'Port A → B  |  Δθ = 0°  |  16 studs',
                  (-6, 22), (-8, 8))

    # Centerline path
    x = np.array([0, 16.0])
    y = np.array([0, 0.0])

    draw_track_bed(ax, x, y, COL_STRAIGHT_BED)
    draw_rails(ax, x, y)

    # Ports
    draw_port(ax, 0, 0, 180, 'A', COL_PORT_A)
    draw_port(ax, 16, 0, 0, 'B', COL_PORT_B)

    # Dimension
    draw_dim_line(ax, 0, -5.5, 16, -5.5, '16 studs', offset_perp=0.5)

    # Track width dimension
    ax.annotate('', xy=(8, RAIL_OFFSET), xytext=(8, -RAIL_OFFSET),
                arrowprops=dict(arrowstyle='<->', color='#bbb', lw=0.6))
    ax.text(8.3, 0, '5', fontsize=6, color='#bbb', va='center',
            fontfamily='monospace')


def draw_curve_r40(ax):
    """Curved Track R40 — 53400 — 22.5° arc, R=40 studs (left-turning)."""
    setup_subplot(ax, 'Curved Track R40 — 53400',
                  'Port A → B  |  Δθ = 22.5°  |  R = 40 studs',
                  (-5, 22), (-4, 12))

    # Arc center: entry at (0,0) heading +X, left turn → center at (0, +R)
    cx, cy = 0.0, R40
    start_ang = -90.0  # from center to entry point
    sweep = CURVE_ANGLE  # +22.5° counter-clockwise (left turn)

    # Centerline arc
    xc, yc = arc_points(cx, cy, R40, start_ang, sweep)
    draw_track_bed(ax, xc, yc, COL_CURVE_BED)

    # Inner / outer rails
    xi, yi = arc_points(cx, cy, R40 - RAIL_OFFSET, start_ang, sweep)
    xo, yo = arc_points(cx, cy, R40 + RAIL_OFFSET, start_ang, sweep)
    ax.plot(xi, yi, color=COL_RAIL, lw=RAIL_LW, solid_capstyle='butt', zorder=3)
    ax.plot(xo, yo, color=COL_RAIL, lw=RAIL_LW, solid_capstyle='butt', zorder=3)

    # Port B position
    end_ang_rad = np.radians(start_ang + sweep)
    bx = cx + R40 * np.cos(end_ang_rad)
    by = cy + R40 * np.sin(end_ang_rad)
    b_heading = CURVE_ANGLE  # exit heading for left turn

    # Ports
    draw_port(ax, 0, 0, 180, 'A', COL_PORT_A)
    draw_port(ax, bx, by, b_heading, 'B', COL_PORT_B)

    # Radius line (dashed) — only show portion within viewport
    ax.plot([0, 0], [0, 10], '--', color='#ccc', lw=0.7, zorder=1)
    ax.text(1, 6, 'R = 40', fontsize=8, color=COL_DIM,
            fontfamily='monospace', fontstyle='italic')

    # Angle arc indicator (small arc near entry)
    ang_r = 6
    ang_x, ang_y = arc_points(0, 0, ang_r, 0, CURVE_ANGLE, 30)
    ax.plot(ang_x, ang_y, '-', color=COL_CURVE_BED, lw=1.2, alpha=0.6)
    mid_ang = np.radians(CURVE_ANGLE / 2)
    ax.text((ang_r + 2.5) * np.cos(mid_ang),
            (ang_r + 2.5) * np.sin(mid_ang),
            '22.5°', fontsize=7.5, color=COL_CURVE_BED, fontweight='bold',
            ha='center', va='center')

    # Note: 16 pieces = full circle
    ax.text(0.02, 0.02, '16 pieces = 360°', transform=ax.transAxes,
            fontsize=6.5, color='#bbb', fontfamily='monospace')

    # Port B coordinate label
    ax.text(bx + 1.2, by + 1.5, f'({bx:.1f}, {by:.1f})',
            fontsize=6, color=COL_PORT_B, fontfamily='monospace')


def draw_switch(ax, direction='left'):
    """Switch Track — compound S-curve diverging branch.

    Left: 53407, Port C at (32.693, +12.955, +22.5°)
    Right: 53404, Port C at (32.693, -12.955, -22.5°)

    S-curve: Arc1 = 36.87° outward (R40), Arc2 = 14.37° inward (R40)
    """
    flip = 1 if direction == 'left' else -1
    part_id = '53407' if direction == 'left' else '53404'

    ylim = (-6, 18) if direction == 'left' else (-18, 6)
    setup_subplot(ax, f'Switch {"Left" if direction == "left" else "Right"} — {part_id}',
                  f'Ports A→B (main) + C (div)  |  Δθ_C = {flip * 22.5:+.1f}°',
                  (-6, 40), ylim)

    # ── MAIN STRAIGHT PATH ────────────────────────────────────
    mx = np.array([0, SWITCH_LEN])
    my = np.array([0, 0])
    draw_track_bed(ax, mx, my, COL_SWITCH_MAIN)
    draw_rails(ax, mx, my)

    # ── DIVERGING BRANCH (S-curve) ────────────────────────────
    # Arc 1: outward curve, center perpendicular to entry on diverging side
    c1x, c1y = 0.0, flip * R40
    a1_start = -90.0 * flip  # angle from center to entry (0,0)
    a1_sweep = flip * ARC1_ANGLE  # 36.87° toward diverging side

    x1, y1 = arc_points(c1x, c1y, R40, a1_start, a1_sweep)
    draw_track_bed(ax, x1, y1, COL_BRANCH_BED, lw=8)

    # Rails for arc 1
    xi1, yi1 = arc_points(c1x, c1y, R40 - RAIL_OFFSET, a1_start, a1_sweep)
    xo1, yo1 = arc_points(c1x, c1y, R40 + RAIL_OFFSET, a1_start, a1_sweep)
    ax.plot(xi1, yi1, color=COL_RAIL, lw=RAIL_LW, solid_capstyle='butt', zorder=3)
    ax.plot(xo1, yo1, color=COL_RAIL, lw=RAIL_LW, solid_capstyle='butt', zorder=3)

    # Junction point at end of Arc 1
    a1_end_rad = np.radians(a1_start + a1_sweep)
    jx = c1x + R40 * np.cos(a1_end_rad)  # should be ≈ 24
    jy = c1y + R40 * np.sin(a1_end_rad)  # should be ≈ 8 * flip
    j_heading = flip * ARC1_ANGLE  # heading at junction

    # Arc 2: inward curve (opposite direction), correcting back toward 22.5°
    # Center perpendicular to RIGHT of heading (for left switch)
    j_heading_rad = np.radians(j_heading)
    perp_dir = j_heading_rad - flip * np.pi / 2  # perpendicular toward straight side
    c2x = jx + R40 * np.cos(perp_dir)
    c2y = jy + R40 * np.sin(perp_dir)

    # Start angle of Arc 2 from its center
    a2_start = np.degrees(np.arctan2(jy - c2y, jx - c2x))
    a2_sweep = -flip * ARC2_ANGLE  # opposite direction from Arc 1

    x2, y2 = arc_points(c2x, c2y, R40, a2_start, a2_sweep)
    draw_track_bed(ax, x2, y2, COL_BRANCH_BED, lw=8)

    # Rails for arc 2
    xi2, yi2 = arc_points(c2x, c2y, R40 - RAIL_OFFSET, a2_start, a2_sweep)
    xo2, yo2 = arc_points(c2x, c2y, R40 + RAIL_OFFSET, a2_start, a2_sweep)
    ax.plot(xi2, yi2, color=COL_RAIL, lw=RAIL_LW, solid_capstyle='butt', zorder=3)
    ax.plot(xo2, yo2, color=COL_RAIL, lw=RAIL_LW, solid_capstyle='butt', zorder=3)

    # ── PORT C position ───────────────────────────────────────
    pc_x = 48 - R40 * np.sin(np.radians(CURVE_ANGLE))  # ≈ 32.693
    pc_y = flip * (16 - R40 * (1 - np.cos(np.radians(CURVE_ANGLE))))  # ≈ ±12.955
    pc_heading = flip * CURVE_ANGLE

    # ── PORTS ─────────────────────────────────────────────────
    draw_port(ax, 0, 0, 180, 'A', COL_PORT_A)
    draw_port(ax, SWITCH_LEN, 0, 0, 'B', COL_PORT_B)
    draw_port(ax, pc_x, pc_y, pc_heading, 'C', COL_PORT_C)

    # ── DIVERGENCE POINT marker ───────────────────────────────
    ax.plot(0, 0, 's', color='white', markersize=6,
            markeredgecolor=COL_PORT_A, markeredgewidth=1.5, zorder=7)

    # ── ANNOTATIONS ───────────────────────────────────────────
    # Dimension line for main path
    dim_y = -4 if direction == 'left' else 4
    draw_dim_line(ax, 0, dim_y, SWITCH_LEN, dim_y, '32 studs',
                  offset_perp=0.5 * flip)

    # Arc labels
    mid1_ang = np.radians(a1_start + a1_sweep * 0.5)
    ax.text(c1x + (R40 + 4 * flip) * np.cos(mid1_ang),
            c1y + (R40 + 4 * flip) * np.sin(mid1_ang),
            f'Arc1\n{ARC1_ANGLE:.1f}°', fontsize=6, color=COL_BRANCH_BED,
            ha='center', va='center', fontfamily='monospace', fontweight='bold')

    mid2_ang = np.radians(a2_start + a2_sweep * 0.5)
    ax.text(c2x + (R40 + 4 * flip) * np.cos(mid2_ang),
            c2y + (R40 + 4 * flip) * np.sin(mid2_ang),
            f'Arc2\n{ARC2_ANGLE:.1f}°', fontsize=6, color=COL_BRANCH_BED,
            ha='center', va='center', fontfamily='monospace', fontweight='bold')

    # Port C coordinate label
    ax.text(pc_x + 1, pc_y + 2 * flip,
            f'({pc_x:.1f}, {pc_y:.1f})\n{pc_heading:+.1f}°',
            fontsize=6, color=COL_PORT_C, fontfamily='monospace',
            ha='left', va='center')

    # S-curve explanation
    note_y = 0.02 if direction == 'left' else 0.95
    ax.text(0.02, note_y,
            f'S-curve: 2× R40 arcs (3-4-5 triple)',
            transform=ax.transAxes, fontsize=6, color='#bbb',
            fontfamily='monospace', va='bottom' if direction == 'left' else 'top')


# ═══════════════════════════════════════════════════════════════
# MAIN FIGURE (only runs when script is executed directly)
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle('LEGO Track Pieces — Matplotlib Models (YAML Spec)',
                 fontsize=14, fontweight='bold', color='#2c3e50', y=0.98)
    fig.text(0.5, 0.955,
             'Exact geometry from community research · Stud coordinates · Port-based',
             ha='center', fontsize=9, color='#999', fontfamily='monospace')

    # Draw each piece
    draw_straight(axes[0, 0])
    draw_curve_r40(axes[0, 1])
    draw_switch(axes[1, 0], direction='left')
    draw_switch(axes[1, 1], direction='right')

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=COL_STRAIGHT_BED, label='Straight bed'),
        mpatches.Patch(facecolor=COL_CURVE_BED, label='Curve bed'),
        mpatches.Patch(facecolor=COL_SWITCH_MAIN, label='Switch main'),
        mpatches.Patch(facecolor=COL_BRANCH_BED, label='Branch (S-curve)'),
        mpatches.Patch(facecolor=COL_RAIL, label='Rails (±2.5 studs)'),
        mpatches.Patch(facecolor=COL_PORT_A, label='Port A (entry)'),
        mpatches.Patch(facecolor=COL_PORT_B, label='Port B (exit)'),
        mpatches.Patch(facecolor=COL_PORT_C, label='Port C (diverge)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
               fontsize=8, framealpha=0.9, edgecolor='#ddd',
               bbox_to_anchor=(0.5, 0.005))

    plt.tight_layout(rect=[0, 0.055, 1, 0.945])
    plt.savefig('lego_track_models.png', dpi=180,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    print("Saved lego_track_models.png")
    plt.close()
