"""Self-intersection detection for track layouts.

Detects crossing segments and provides crossing-pair data for CROSS_90 repair.
"""

from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

CROSS_90_INDEX = 4


def find_crossing_pairs(
    states: NDArray[np.float64],
    piece_indices: Optional[List[int]] = None,
    min_separation: int = 3,
) -> List[Tuple[int, int, float]]:
    """Find all self-intersecting segment pairs with heading angles.

    Same detection as count_segment_crossings() but returns the actual
    crossing pairs with angle information for crossing repair.

    Args:
        states: (n+1, 3) FK states array [x, y, theta].
        piece_indices: Piece index at each position (skips existing CROSS_90).
        min_separation: Minimum index distance between segments to check.

    Returns:
        List of (pos_i, pos_j, angle_diff) sorted by proximity to 90 deg.
        angle_diff is in [0, 90] — how close the crossing is to perpendicular.
    """
    n = len(states) - 1
    if n < min_separation + 1:
        return []

    cross_positions = set()
    if piece_indices is not None:
        for i, idx in enumerate(piece_indices):
            if idx == CROSS_90_INDEX:
                cross_positions.add(i)

    x = states[:, 0]
    y = states[:, 1]
    theta = states[:, 2]
    pairs = []

    for i in range(n - min_separation):
        if i in cross_positions:
            continue
        ax, ay = x[i], y[i]
        bx, by = x[i + 1], y[i + 1]
        i_min_x, i_max_x = min(ax, bx), max(ax, bx)
        i_min_y, i_max_y = min(ay, by), max(ay, by)

        for j in range(i + min_separation, n):
            if j in cross_positions:
                continue
            cx, cy = x[j], y[j]
            dx, dy = x[j + 1], y[j + 1]

            j_min_x, j_max_x = min(cx, dx), max(cx, dx)
            j_min_y, j_max_y = min(cy, dy), max(cy, dy)

            if i_max_x < j_min_x or j_max_x < i_min_x:
                continue
            if i_max_y < j_min_y or j_max_y < i_min_y:
                continue

            if _segments_intersect(ax, ay, bx, by, cx, cy, dx, dy):
                raw_diff = abs(theta[i] - theta[j]) % 180
                angle_diff = min(raw_diff, 180 - raw_diff)
                pairs.append((i, j, angle_diff))

    pairs.sort(key=lambda t: abs(t[2] - 90.0))
    return pairs


def count_segment_crossings(
    states: NDArray[np.float64],
    piece_indices: Optional[List[int]] = None,
    min_separation: int = 3,
) -> int:
    """Count track segments that cross each other without a CROSS_90 piece.

    Args:
        states: (n+1, 3) FK states array [x, y, theta].
        piece_indices: Piece index at each position (for CROSS_90 exemption).
        min_separation: Minimum index distance between segments to check.

    Returns:
        Number of invalid (non-CROSS_90) segment crossings.
    """
    n = len(states) - 1
    if n < min_separation + 1:
        return 0

    cross_positions = set()
    if piece_indices is not None:
        for i, idx in enumerate(piece_indices):
            if idx == CROSS_90_INDEX:
                cross_positions.add(i)

    x = states[:, 0]
    y = states[:, 1]
    crossings = 0

    for i in range(n - min_separation):
        ax, ay = x[i], y[i]
        bx, by = x[i + 1], y[i + 1]
        i_min_x, i_max_x = min(ax, bx), max(ax, bx)
        i_min_y, i_max_y = min(ay, by), max(ay, by)

        for j in range(i + min_separation, n):
            cx, cy = x[j], y[j]
            dx, dy = x[j + 1], y[j + 1]

            j_min_x, j_max_x = min(cx, dx), max(cx, dx)
            j_min_y, j_max_y = min(cy, dy), max(cy, dy)

            if i_max_x < j_min_x or j_max_x < i_min_x:
                continue
            if i_max_y < j_min_y or j_max_y < i_min_y:
                continue

            if _segments_intersect(ax, ay, bx, by, cx, cy, dx, dy):
                if i in cross_positions or j in cross_positions:
                    continue
                crossings += 1

    return crossings


def _segments_intersect(
    ax: float, ay: float, bx: float, by: float,
    cx: float, cy: float, dx: float, dy: float,
) -> bool:
    """Check if segment (a,b) intersects segment (c,d) using cross products."""
    def cross(ox, oy, px, py, qx, qy):
        return (px - ox) * (qy - oy) - (py - oy) * (qx - ox)

    d1 = cross(cx, cy, dx, dy, ax, ay)
    d2 = cross(cx, cy, dx, dy, bx, by)
    d3 = cross(ax, ay, bx, by, cx, cy)
    d4 = cross(ax, ay, bx, by, dx, dy)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True

    return False
