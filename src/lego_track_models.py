"""LEGO Track Piece geometry constants for visualization rendering.

Exact geometry from community research (MacFreek, L-Gauge, Transponderings, Fx Bricks).
"""

import math

import numpy as np

# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════
R40 = 40.0                          # curve radius in studs
RAIL_OFFSET = 2.5                   # half gauge (5 studs / 2)
CURVE_ANGLE = 22.5                  # degrees per R40 piece
SWITCH_LEN = 32.0                   # switch straight path length
ARC1_ANGLE = math.degrees(math.atan2(3, 4))  # 36.87° (3-4-5 triple)
ARC2_ANGLE = ARC1_ANGLE - CURVE_ANGLE         # 14.37°

COL_RAIL = '#95a5a6'                # gray
# Full physical piece width in studs: adjacent parallel tracks at the
# canonical 8-stud siding spacing touch with no false overlap.
BED_WIDTH_STUD = 8.0
RAIL_LW = 1.8                      # rail line width
N_ARC_PTS = 60                     # arc interpolation points


# ═══════════════════════════════════════════════════════════════
# GEOMETRY HELPERS
# ═══════════════════════════════════════════════════════════════

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
