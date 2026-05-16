"""Self-intersection detection for track layouts.

Detects crossing segments and provides crossing-pair data for CROSS_90 repair.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

CROSS_90_INDEX = 3
DOUBLE_CROSSOVER_INDEX = 6

# Port set covered by each catalog route of DOUBLE_CROSSOVER (yaml order):
#   0 = track1_through (A,B)
#   1 = track2_through (C,D)
#   2 = cross_1_to_2   (A,D)
#   3 = cross_2_to_1   (C,B)
# Port indexing: A=0, B=1, C=2, D=3.
_DC_ROUTE_PORTS: Dict[int, frozenset] = {
    0: frozenset({0, 1}),
    1: frozenset({2, 3}),
    2: frozenset({0, 3}),
    3: frozenset({2, 1}),
}


_EXEMPT_PIECE_INDICES = frozenset({CROSS_90_INDEX, DOUBLE_CROSSOVER_INDEX})


def _exempt_positions(piece_indices: Optional[List[int]]) -> set:
    """Set of chromosome slots whose internal route is an intentional crossing.

    CROSS_90 hosts perpendicular routes; DOUBLE_CROSSOVER hosts diagonals
    (cross_1_to_2 + cross_2_to_1) that cross inside the piece body. Both
    must be excluded from self-intersection counting and repair targeting.
    """
    if not piece_indices:
        return set()
    return {i for i, idx in enumerate(piece_indices) if idx in _EXEMPT_PIECE_INDICES}


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
        piece_indices: Piece index at each position (skips existing CROSS_90
            and DOUBLE_CROSSOVER slots — their internal crossings are
            intentional).
        min_separation: Minimum index distance between segments to check.

    Returns:
        List of (pos_i, pos_j, angle_diff) sorted by proximity to 90 deg.
        angle_diff is in [0, 90] — how close the crossing is to perpendicular.
    """
    n = len(states) - 1
    if n < min_separation + 1:
        return []

    cross_positions = _exempt_positions(piece_indices)

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

    cross_positions = _exempt_positions(piece_indices)

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


def count_dangling_cross_ports(
    states: NDArray[np.float64],
    piece_indices: List[int],
    pos_tol: float = 4.0,
    ang_tol_deg: float = 15.0,
) -> int:
    """Count CROSS_90 chromosome slots that lack a perpendicular partner.

    Each CROSS_90 piece has 4 ports. To use ALL 4 ports the train must
    traverse the same physical world location TWICE: once through the
    horizontal route (W<->E) and once through the perpendicular vertical
    route (N<->S). In our chromosome encoding, that means BOTH a CROSS_90
    slot AND another slot (CROSS_90 or FK-equivalent STRAIGHT_16) must
    exist at the same world midpoint with perpendicular heading.

    Returns the count of CROSS_90 slots WITHOUT a perpendicular partner;
    those represent crosses with 2 dangling ports and are unbuildable.

    Args:
        states: (n+1, 3) FK states array [x, y, theta_deg].
        piece_indices: piece index at each chromosome slot.
        pos_tol: how close two midpoints must be to count as the same
            world location (studs).
        ang_tol_deg: how close to perpendicular two headings must be.
    """
    if len(piece_indices) == 0:
        return 0

    midpoints: List[Tuple[float, float, float]] = []
    cross_slots: List[int] = []
    for i, piece_idx in enumerate(piece_indices):
        x_start, y_start, theta_deg = states[i]
        # Train midpoint approx (cross center if CROSS_90, segment center
        # otherwise). 8 stud forward in train direction works for any
        # 16-stud-FK piece (CROSS_90, STRAIGHT_16, R40 close enough).
        theta_rad = np.radians(theta_deg)
        mx = x_start + 8.0 * np.cos(theta_rad)
        my = y_start + 8.0 * np.sin(theta_rad)
        midpoints.append((mx, my, theta_deg))
        if piece_idx == CROSS_90_INDEX:
            cross_slots.append(i)

    dangling = 0
    for ci in cross_slots:
        cx, cy, ctheta = midpoints[ci]
        partner_found = False
        for j, (jx, jy, jtheta) in enumerate(midpoints):
            if j == ci:
                continue
            if abs(jx - cx) > pos_tol or abs(jy - cy) > pos_tol:
                continue
            ang_diff = (jtheta - ctheta + 180.0) % 360.0 - 180.0
            # Partner must be perpendicular: |ang_diff| in (90 - tol, 90 + tol)
            if abs(abs(ang_diff) - 90.0) > ang_tol_deg:
                continue
            partner_found = True
            break
        if not partner_found:
            dangling += 1

    return dangling


def count_dangling_double_crossover_ports(
    main_loop_pieces: List[int],
    main_loop_routes: Dict[int, int],
    dbl_crossover_records,
) -> int:
    """Count dangling ports across all DOUBLE_CROSSOVER chromosome slots.

    Each physical DOUBLE_CROSSOVER must be traversed by route pairs that union
    to all 4 ports {A,B,C,D}; otherwise the layout is unbuildable. The decoder
    guarantees this for every record it produces, so the typical return value
    is 0. The function still flags two pathological cases for defence in
    depth:

      1. A DOUBLE_CROSSOVER slot in ``main_loop_pieces`` with no record
         covering it — only one route is used, 2 ports dangle (counted as 2).
      2. A record whose two routes do not jointly cover all 4 ports —
         contributes ``4 - |port_union|`` per record.

    Args:
        main_loop_pieces: Augmented main-loop piece indices.
        main_loop_routes: Map main-loop position -> catalog route index for
            DOUBLE_CROSSOVER slots.
        dbl_crossover_records: Sequence of DblCrossover records (typed loosely
            to avoid a cross-module import from src.types).

    Returns:
        Dangling-port count (0 = no dangling).
    """
    accounted: set = set()
    dangling = 0
    for rec in dbl_crossover_records or ():
        p1, p2 = rec.positions
        accounted.update((p1, p2))
        r1, r2 = rec.routes
        covered = (_DC_ROUTE_PORTS.get(r1, frozenset())
                   | _DC_ROUTE_PORTS.get(r2, frozenset()))
        dangling += 4 - len(covered)

    solo_slots = sum(
        1 for pos, piece in enumerate(main_loop_pieces)
        if piece == DOUBLE_CROSSOVER_INDEX and pos not in accounted
    )
    # Each solo slot represents a single traversal — 2 ports used, 2 dangle.
    dangling += 2 * solo_slots
    return dangling


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
