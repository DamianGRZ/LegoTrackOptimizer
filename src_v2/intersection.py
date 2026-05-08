"""Segment-intersection detection for V2 port-pair layouts.

Operates on a PortGraph: for each active edge, computes the world endpoints
(port-A pose -> port-B pose of each piece), then pairwise-tests segments for
spatial intersection. The resulting hit list drives :func:`introduce_crossing`
in :mod:`src_v2.structural_mutations`.

V2-native — does not depend on src/intersection.py.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from .se2 import pose_compose
from .types import PortEdge, PortGraph


# (row_idx, slot_a, port_a, slot_b, port_b, world_xy_a, world_xy_b)
EdgeEndpoint = Tuple[int, int, int, int, int, Tuple[float, float], Tuple[float, float]]

# (edge_a_endpoint, edge_b_endpoint, perpendicularity_error_deg)
PerpendicularHit = Tuple[EdgeEndpoint, EdgeEndpoint, float]


def edge_world_endpoints(
    graph: PortGraph,
    catalog,
) -> Dict[int, Tuple[Tuple[float, float], Tuple[float, float]]]:
    """For each port-graph edge, return its two world-space endpoints.

    Returns a dict keyed by edge index in ``graph.edges`` to (xy_a, xy_b).
    """
    spec = catalog.spec
    out: Dict[int, Tuple[Tuple[float, float], Tuple[float, float]]] = {}

    for i, edge in enumerate(graph.edges):
        pose_a = graph.slot_poses.get(edge.slot_a)
        pose_b = graph.slot_poses.get(edge.slot_b)
        if pose_a is None or pose_b is None:
            continue
        piece_a = spec.by_id.get(graph.slot_pieces.get(edge.slot_a, ""))
        piece_b = spec.by_id.get(graph.slot_pieces.get(edge.slot_b, ""))
        if piece_a is None or piece_b is None:
            continue

        port_a_def = piece_a.ports.get(edge.port_a)
        port_b_def = piece_b.ports.get(edge.port_b)
        if port_a_def is None or port_b_def is None:
            continue

        port_a_world = pose_compose(pose_a, (port_a_def.dx, port_a_def.dy, port_a_def.dtheta))
        port_b_world = pose_compose(pose_b, (port_b_def.dx, port_b_def.dy, port_b_def.dtheta))
        out[i] = ((port_a_world[0], port_a_world[1]),
                  (port_b_world[0], port_b_world[1]))

    return out


def _segment_orientation(
    px: float, py: float, qx: float, qy: float, rx: float, ry: float,
) -> float:
    """Cross product of (q - p) x (r - p); positive = CCW, negative = CW."""
    return (qx - px) * (ry - py) - (qy - py) * (rx - px)


def _segments_intersect(
    a: Tuple[float, float], b: Tuple[float, float],
    c: Tuple[float, float], d: Tuple[float, float],
) -> bool:
    """True iff segment (a, b) crosses segment (c, d) strictly (no endpoint touch)."""
    d1 = _segment_orientation(c[0], c[1], d[0], d[1], a[0], a[1])
    d2 = _segment_orientation(c[0], c[1], d[0], d[1], b[0], b[1])
    d3 = _segment_orientation(a[0], a[1], b[0], b[1], c[0], c[1])
    d4 = _segment_orientation(a[0], a[1], b[0], b[1], d[0], d[1])
    return ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0))


def _segment_angle_deg(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Angle of segment a -> b in degrees, range (-180, 180]."""
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))


def find_perpendicular_intersections(
    graph: PortGraph,
    catalog,
    perpendicularity_tolerance_deg: float = 20.0,
) -> List[PerpendicularHit]:
    """Find pairs of edges whose segments cross at near-perpendicular angles.

    Returns a list of ``(endpoint_a, endpoint_b, perp_error_deg)`` sorted by
    smallest perpendicularity error first (closest to 90 deg crossings come
    first — best candidates for CROSS_90 introduction).
    """
    endpoints = edge_world_endpoints(graph, catalog)
    if len(endpoints) < 2:
        return []

    # Build endpoint records keyed by edge index
    records: List[EdgeEndpoint] = []
    for idx, (a, b) in endpoints.items():
        edge = graph.edges[idx]
        records.append((idx, edge.slot_a,
                        _port_idx(catalog, graph.slot_pieces.get(edge.slot_a, ""), edge.port_a),
                        edge.slot_b,
                        _port_idx(catalog, graph.slot_pieces.get(edge.slot_b, ""), edge.port_b),
                        a, b))

    hits: List[PerpendicularHit] = []
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            r_i, r_j = records[i], records[j]
            # Skip edges that share a slot (adjacent in topology, not a crossing)
            if {r_i[1], r_i[3]} & {r_j[1], r_j[3]}:
                continue
            if not _segments_intersect(r_i[5], r_i[6], r_j[5], r_j[6]):
                continue
            angle_i = _segment_angle_deg(r_i[5], r_i[6])
            angle_j = _segment_angle_deg(r_j[5], r_j[6])
            diff = abs(angle_i - angle_j) % 180.0
            perp_error = abs(diff - 90.0)
            if perp_error <= perpendicularity_tolerance_deg:
                hits.append((r_i, r_j, perp_error))

    hits.sort(key=lambda h: h[2])
    return hits


def _port_idx(catalog, piece_id: str, port_name: str) -> Optional[int]:
    spec = catalog.spec
    if spec is None or not piece_id:
        return None
    piece_spec = spec.by_id.get(piece_id)
    if piece_spec is None:
        return None
    names = list(piece_spec.ports)
    return names.index(port_name) if port_name in names else None
<<<<<<< Updated upstream


# =============================================================================
# Uncovered-intersection count for problem.G[4]
# =============================================================================


_CROSSING_PIECES: frozenset = frozenset({"CROSS_90", "DOUBLE_CROSSOVER"})
"""Piece kinds whose presence at a crossing geometrically *covers* it."""


def _piece_chord_segments(
    graph: PortGraph, catalog,
) -> Dict[int, Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Each non-crossing piece's port-A → port-B chord in world coordinates.

    CROSS_90 and DOUBLE_CROSSOVER are excluded — their presence at a
    geometric intersection is what makes the intersection legal, so they
    don't contribute "ordinary" track segments. Curves are approximated by
    their A-B chord (under-counts arc bow but is conservative for v0).
    """
    spec = catalog.spec
    segments: Dict[int, Tuple[Tuple[float, float], Tuple[float, float]]] = {}
    for slot, piece_id in graph.slot_pieces.items():
        if piece_id in _CROSSING_PIECES:
            continue
        slot_pose = graph.slot_poses.get(slot)
        if slot_pose is None:
            continue
        piece_spec = spec.by_id.get(piece_id)
        if piece_spec is None or "A" not in piece_spec.ports or "B" not in piece_spec.ports:
            continue
        port_a = piece_spec.ports["A"]
        port_b = piece_spec.ports["B"]
        a_world = pose_compose(slot_pose, (port_a.dx, port_a.dy, port_a.dtheta))
        b_world = pose_compose(slot_pose, (port_b.dx, port_b.dy, port_b.dtheta))
        segments[slot] = ((a_world[0], a_world[1]), (b_world[0], b_world[1]))
    return segments


def count_uncovered_intersections(graph: PortGraph, catalog) -> int:
    """Count piece-pair geometric intersections not covered by a crossing piece.

    For every pair of non-crossing slots, the A-B chord segments are tested
    for strict intersection (endpoint touches don't count, so adjacent
    end-to-end pieces are excluded by construction). Crossings (CROSS_90,
    DOUBLE_CROSSOVER) are dropped from the segment set because their
    presence is what *legalizes* a crossing — two ordinary segments meeting
    inside a CROSS_90's footprint count, but that's caught indirectly when
    the surrounding non-crossing edges intersect each other in the same
    region without the crossing slot present.
    """
    segments = _piece_chord_segments(graph, catalog)
    slots = list(segments.keys())
    return sum(
        1
        for i, slot_i in enumerate(slots)
        for slot_j in slots[i + 1:]
        if _segments_intersect(*segments[slot_i], *segments[slot_j])
    )
=======
>>>>>>> Stashed changes
