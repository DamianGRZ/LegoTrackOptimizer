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


def _proper_crossing_pairs(
    states: NDArray[np.float64],
    min_separation: int,
    exempt: set,
) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Index arrays (ii, jj) of segment pairs that properly cross.

    Two-stage vectorized scan:
      1. Cheap (n, n) candidate mask: index band (j >= i + min_separation),
         exemptions, and bounding-box overlap — the same prefilter the scalar
         loop used.
      2. Strict orientation sign test evaluated only on the (typically sparse)
         candidate pairs. Endpoint touches do not count, matching the original
         scalar test exactly.

    Pairs are returned in (i, j)-lexicographic order.
    """
    n = len(states) - 1
    x = states[:, 0]
    y = states[:, 1]

    x_lo = np.minimum(x[:-1], x[1:])
    x_hi = np.maximum(x[:-1], x[1:])
    y_lo = np.minimum(y[:-1], y[1:])
    y_hi = np.maximum(y[:-1], y[1:])

    idx = np.arange(n)
    candidates = (idx[None, :] - idx[:, None]) >= min_separation
    candidates &= (x_hi[:, None] >= x_lo[None, :]) & (x_hi[None, :] >= x_lo[:, None])
    candidates &= (y_hi[:, None] >= y_lo[None, :]) & (y_hi[None, :] >= y_lo[:, None])

    if exempt:
        ex = np.fromiter(exempt, dtype=np.int64)
        ex = ex[ex < n]
        candidates[ex, :] = False
        candidates[:, ex] = False

    ii, jj = np.nonzero(candidates)  # row-major == (i, j)-lexicographic
    if len(ii) == 0:
        return ii, jj

    ax, ay = x[ii], y[ii]
    bx, by = x[ii + 1], y[ii + 1]
    cx, cy = x[jj], y[jj]
    dx, dy = x[jj + 1], y[jj + 1]

    cd_x, cd_y = dx - cx, dy - cy
    ab_x, ab_y = bx - ax, by - ay
    d1 = cd_x * (ay - cy) - cd_y * (ax - cx)
    d2 = cd_x * (by - cy) - cd_y * (bx - cx)
    d3 = ab_x * (cy - ay) - ab_y * (cx - ax)
    d4 = ab_x * (dy - ay) - ab_y * (dx - ax)

    proper = (
        (((d1 > 0) & (d2 < 0)) | ((d1 < 0) & (d2 > 0)))
        & (((d3 > 0) & (d4 < 0)) | ((d3 < 0) & (d4 > 0)))
    )
    return ii[proper], jj[proper]


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
        Ties keep (i, j)-lexicographic order (stable sort), matching the
        original scan order.
    """
    n = len(states) - 1
    if n < min_separation + 1:
        return []

    ii, jj = _proper_crossing_pairs(
        states, min_separation, _exempt_positions(piece_indices),
    )
    if len(ii) == 0:
        return []

    theta = states[:, 2]
    raw_diff = np.abs(theta[ii] - theta[jj]) % 180.0
    angle_diff = np.minimum(raw_diff, 180.0 - raw_diff)

    order = np.argsort(np.abs(angle_diff - 90.0), kind="stable")
    return [(int(ii[k]), int(jj[k]), float(angle_diff[k])) for k in order]


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

    ii, _jj = _proper_crossing_pairs(
        states, min_separation, _exempt_positions(piece_indices),
    )
    return int(len(ii))


def _cross_midpoint(state: NDArray[np.float64]) -> Tuple[float, float, float]:
    """World midpoint (8 stud forward) and heading for a 16-stud-FK slot."""
    x_start, y_start, theta_deg = float(state[0]), float(state[1]), float(state[2])
    theta_rad = np.radians(theta_deg)
    return (x_start + 8.0 * np.cos(theta_rad),
            y_start + 8.0 * np.sin(theta_rad),
            theta_deg)


def cross_pair_perpendicular(
    states: NDArray[np.float64],
    i: int,
    j: int,
    pos_tol: float = 4.0,
    ang_tol_deg: float = 15.0,
) -> bool:
    """True iff slots i and j share a world midpoint and cross at ~90 deg.

    This is the single definition of "a valid CROSS_90 crossing" used by BOTH
    the decoder (to validate a cross-junction descriptor before committing) and
    count_dangling_cross_ports (to confirm a placed CROSS_90 has its partner).
    Sharing it guarantees a decoder-validated crossing is never counted dangling.
    """
    ix, iy, ith = _cross_midpoint(states[i])
    jx, jy, jth = _cross_midpoint(states[j])
    if abs(ix - jx) > pos_tol or abs(iy - jy) > pos_tol:
        return False
    ang_diff = (jth - ith + 180.0) % 360.0 - 180.0
    return abs(abs(ang_diff) - 90.0) <= ang_tol_deg


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

    n_slots = len(piece_indices)
    cross_slots = [i for i, idx in enumerate(piece_indices) if idx == CROSS_90_INDEX]

    dangling = 0
    for ci in cross_slots:
        partner_found = any(
            cross_pair_perpendicular(states, ci, j, pos_tol, ang_tol_deg)
            for j in range(n_slots) if j != ci
        )
        if not partner_found:
            dangling += 1

    return dangling


def count_dangling_double_crossover_ports(
    main_loop_pieces: List[int],
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
