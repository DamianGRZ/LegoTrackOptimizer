"""Port-graph decoder: chromosome -> PortGraph.

Algorithm:

1. Parse the chromosome into raw piece slots, raw port-pair edges, and anchor.
2. Validate edges against catalog and slot state — drop self-loops, edges
   referencing inactive slots, edges with invalid port indices for the
   referenced piece kind, and edges that double-book a port. The number
   dropped is reported via ``PortGraph.dropped_edge_count``.
3. Compute connected components via union-find on sanitized edges.
4. For each component, BFS pose propagation from a deterministic anchor slot:
   compose the slot pose with each piece-local port offset to get the port's
   world pose, then back out the partner slot's pose by composing with the
   inverse of the partner port's offset (with a 180 deg flip because mating
   ports face opposite directions).
5. When BFS reaches an already-posed slot via a different walk, record the
   pose mismatch as a cycle closure residual.
6. Identify loose ports (declared by a piece spec but not present in any edge).
7. Return the assembled :class:`PortGraph`.

This module produces the intermediate ``PortGraph`` only; conversion to
``MultiPathLayout`` (path enumeration, switch_pair derivation for
visualization) happens in a downstream step yet to be implemented.
"""

from __future__ import annotations

import math
<<<<<<< Updated upstream
from collections import defaultdict, deque
from dataclasses import dataclass
from itertools import pairwise
from typing import Dict, Iterator, List, Optional, Set, Tuple

import networkx as nx
=======
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

>>>>>>> Stashed changes
from numpy.typing import NDArray

from .catalog import TrackCatalog
from .catalog.specs import TrackCatalogSpec, TrackPieceSpec
from .encoding import (
    INACTIVE,
    PortPairDimensions,
    get_anchor,
<<<<<<< Updated upstream
    get_slot_flip,
    get_slot_rotate,
=======
>>>>>>> Stashed changes
    iter_active_pairs,
    iter_active_slots,
)
from .se2 import IDENTITY, Pose, pose_compose, pose_inverse
from .types import CycleResidual, PortEdge, PortGraph


# =============================================================================
# Decoder Configuration
# =============================================================================


@dataclass
class DecoderConfig:
    """Decoder tolerances and boundary."""

    closure_position_tolerance: float = 4.0
    closure_angle_tolerance_deg: float = 5.0
    boundary_min_x: float = -100.0
    boundary_max_x: float = 100.0
    boundary_min_y: float = -100.0
    boundary_max_y: float = 100.0


# =============================================================================
# Public entry point
# =============================================================================


def decode_chromosome(
    x: NDArray,
    dims: PortPairDimensions,
    catalog: TrackCatalog,
    config: Optional[DecoderConfig] = None,
) -> PortGraph:
    """Decode a port-pair chromosome into a :class:`PortGraph`.

    Args:
        x: Chromosome array of length ``dims.n_var``.
        dims: Port-pair dimensions used to encode ``x``.
        catalog: Track catalog. Must have been loaded from V2 YAML so that
            ``catalog.spec`` provides per-piece port definitions and routes.
        config: Decoder configuration. Defaults are used when None.
    """
    if config is None:
        config = DecoderConfig()

    spec = catalog.spec
    if spec is None:
        raise ValueError(
            "decode_chromosome requires a catalog loaded from V2 yaml; "
            "catalog.spec is None."
        )

    slot_pieces, slot_indices = _read_slots(x, dims, catalog)
<<<<<<< Updated upstream
    slot_flips = _read_slot_flips(x, dims, slot_pieces, spec)
    slot_rotates = _read_slot_rotates(x, dims, slot_pieces, spec)
=======
>>>>>>> Stashed changes
    raw_edges = _read_raw_edges(x, dims, catalog, spec, slot_pieces)
    sanitized_edges, dropped = _sanitize_edges(raw_edges)

    components = _connected_components(slot_pieces, sanitized_edges)

    anchor = _anchor_pose_from_chromosome(x, dims)
    slot_poses, residuals = _propagate_poses(
<<<<<<< Updated upstream
        slot_pieces, slot_flips, slot_rotates,
        sanitized_edges, components, spec, anchor,
    )

    # Phase 3: shift the laid-out poses so the bbox center lands at boundary
    # center plus the chromosome's anchor (off_x, off_y). The chromosome's
    # FK origin xy is therefore irrelevant to placement -- it gets cancelled
    # by the recentering step. This guarantees layouts respect the boundary
    # constraint regardless of where FK happened to start.
    slot_poses = _auto_center_layout(
        slot_poses,
        boundary_min_x=config.boundary_min_x,
        boundary_max_x=config.boundary_max_x,
        boundary_min_y=config.boundary_min_y,
        boundary_max_y=config.boundary_max_y,
        anchor_offset_xy=(anchor[0], anchor[1]),
=======
        slot_pieces, sanitized_edges, components, spec, anchor,
>>>>>>> Stashed changes
    )

    loose_ports = _find_loose_ports(slot_pieces, sanitized_edges, spec)

<<<<<<< Updated upstream
    branch_labels = _compute_branch_labels(
        components, sanitized_edges, slot_pieces, spec,
    )

    return PortGraph(
        slot_pieces=slot_pieces,
        slot_indices=slot_indices,
        slot_flips=slot_flips,
        slot_rotates=slot_rotates,
=======
    return PortGraph(
        slot_pieces=slot_pieces,
        slot_indices=slot_indices,
>>>>>>> Stashed changes
        edges=sanitized_edges,
        slot_poses=slot_poses,
        closure_residuals=residuals,
        loose_ports=loose_ports,
        connected_components=components,
        dropped_edge_count=dropped,
<<<<<<< Updated upstream
        branch_labels=branch_labels,
    )


def _read_slot_flips(
    x: NDArray,
    dims: PortPairDimensions,
    slot_pieces: Dict[int, str],
    spec: TrackCatalogSpec,
) -> Dict[int, int]:
    """Read per-slot flip bits, forcing 0 on slots whose piece is asymmetric.

    Decoder-side enforcement is belt-and-braces: repair *should* have already
    cleared flips on asymmetric pieces, but we don't trust that and we do
    not want a stray 1 to silently produce a "mirrored switch" geometry that
    has no physical equivalent.
    """
    flips: Dict[int, int] = {}
    for slot_idx, piece_id in slot_pieces.items():
        raw = get_slot_flip(x, dims, slot_idx)
        ps = spec.by_id.get(piece_id)
        flips[slot_idx] = raw if (ps is not None and ps.symmetric) else 0
    return flips


def _read_slot_rotates(
    x: NDArray,
    dims: PortPairDimensions,
    slot_pieces: Dict[int, str],
    spec: TrackCatalogSpec,
) -> Dict[int, int]:
    """Read per-slot rotate bits, forcing 0 on non-rotatable pieces.

    Symmetric guard analogous to ``_read_slot_flips`` — keeps the decoder
    immune to chromosomes that haven't been through repair yet.
    """
    rotates: Dict[int, int] = {}
    for slot_idx, piece_id in slot_pieces.items():
        raw = get_slot_rotate(x, dims, slot_idx)
        ps = spec.by_id.get(piece_id)
        rotates[slot_idx] = raw if (ps is not None and ps.rotatable) else 0
    return rotates


def _piece_body_length(piece_spec: TrackPieceSpec) -> float:
    """Body length used as the rotation pivot for the rotate bit.

    Switches expose ``body_length_studs``; straights/crossings use
    ``length_studs``. Falls back to 0.0 (rotation becomes a no-op) for
    pieces that define neither — including curves, which are not declared
    rotatable anyway.
    """
    if piece_spec.body_length_studs is not None:
        return float(piece_spec.body_length_studs)
    if piece_spec.length_studs is not None:
        return float(piece_spec.length_studs)
    return 0.0


=======
    )


>>>>>>> Stashed changes
# =============================================================================
# Step 1 — read slots
# =============================================================================


def _read_slots(
    x: NDArray, dims: PortPairDimensions, catalog: TrackCatalog,
) -> Tuple[Dict[int, str], Dict[int, int]]:
    """Read active slots, returning (slot_pieces, slot_indices) maps."""
    slot_pieces: Dict[int, str] = {}
    slot_indices: Dict[int, int] = {}
    index_to_id = catalog.index_to_id
    for slot_idx, piece_index in iter_active_slots(x, dims):
        piece_id = index_to_id.get(piece_index)
        if piece_id is None:
            # Slot references unknown piece index — skip rather than crash.
            continue
        slot_pieces[slot_idx] = piece_id
        slot_indices[slot_idx] = piece_index
    return slot_pieces, slot_indices


# =============================================================================
# Step 2 — read and validate edges
# =============================================================================


def _read_raw_edges(
    x: NDArray,
    dims: PortPairDimensions,
    catalog: TrackCatalog,
    spec: TrackCatalogSpec,
    slot_pieces: Dict[int, str],
) -> List[PortEdge]:
    """Read raw port-pair rows, converting integer port indices to names.

    Edges referencing inactive slots or out-of-range port indices are dropped
    silently here; they will be counted via ``_sanitize_edges``.
    """
    raw: List[PortEdge] = []
    for _, sa, pa, sb, pb in iter_active_pairs(x, dims):
        port_a_name = _port_name(sa, pa, slot_pieces, spec)
        port_b_name = _port_name(sb, pb, slot_pieces, spec)
        if port_a_name is None or port_b_name is None:
            continue
        raw.append(PortEdge(slot_a=sa, port_a=port_a_name,
                            slot_b=sb, port_b=port_b_name))
    return raw


def _port_name(
    slot_idx: int,
    port_idx: int,
    slot_pieces: Dict[int, str],
    spec: TrackCatalogSpec,
) -> Optional[str]:
    """Map a (slot_idx, port_idx) to a port name, or None if invalid."""
    piece_id = slot_pieces.get(slot_idx)
    if piece_id is None:
        return None
    piece_spec = spec.by_id.get(piece_id)
    if piece_spec is None:
        return None
    port_names = list(piece_spec.ports)
    if port_idx < 0 or port_idx >= len(port_names):
        return None
    return port_names[port_idx]


def _sanitize_edges(raw_edges: List[PortEdge]) -> Tuple[List[PortEdge], int]:
    """Drop self-loops and double-booked ports.

    Self-loop: an edge with slot_a == slot_b.
    Double-booking: a (slot, port) pair appearing in more than one edge —
    the first occurrence wins, later ones are dropped.

    Returns:
        (sanitized_edges, dropped_count) where dropped_count includes raw
        edges that were absent from the input due to upstream filtering.
    """
    used_ports: Set[Tuple[int, str]] = set()
    sanitized: List[PortEdge] = []
    dropped = 0

    for edge in raw_edges:
        if edge.slot_a == edge.slot_b:
            dropped += 1
            continue
        key_a = (edge.slot_a, edge.port_a)
        key_b = (edge.slot_b, edge.port_b)
        if key_a in used_ports or key_b in used_ports:
            dropped += 1
            continue
        used_ports.add(key_a)
        used_ports.add(key_b)
        sanitized.append(edge)

    return sanitized, dropped


# =============================================================================
# Step 3 — connected components (union-find)
# =============================================================================


def _connected_components(
    slot_pieces: Dict[int, str], edges: List[PortEdge],
) -> List[Set[int]]:
    """Return slot-index sets, one per connected component.

    Isolated slots (no edges) form singleton components.
    """
    parent: Dict[int, int] = {s: s for s in slot_pieces}

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        # Path compression
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for edge in edges:
        if edge.slot_a in parent and edge.slot_b in parent:
            union(edge.slot_a, edge.slot_b)

    groups: Dict[int, Set[int]] = {}
    for slot in slot_pieces:
        root = find(slot)
        groups.setdefault(root, set()).add(slot)

    return list(groups.values())


# =============================================================================
# Step 4 — BFS pose propagation
# =============================================================================


def _anchor_pose_from_chromosome(
    x: NDArray, dims: PortPairDimensions,
) -> Pose:
    """Read anchor genes and convert theta from degrees to radians."""
    ax, ay, atheta_deg = get_anchor(x, dims)
    return (float(ax), float(ay), math.radians(float(atheta_deg)))


def _port_local_pose(
<<<<<<< Updated upstream
    piece_spec: TrackPieceSpec,
    port_name: str,
    flip: int = 0,
    rotate: int = 0,
) -> Pose:
    """Pose of ``port_name`` in the piece-local frame (port A at origin).

    Two orientation modifiers, applied in order rotate → flip:

    - ``rotate=1`` (only meaningful for ``rotatable`` pieces): 180°
      in-plane rotation around the piece body center ``(L/2, 0)`` where
      ``L`` is the piece body length. Coordinates transform as
      ``(dx, dy, dtheta) → (L - dx, -dy, dtheta + π)``. After this the
      what-was-port-B sits at (0, 0, 0) — the rotated piece "starts" from
      the far end. For the IN→OUT switch case this puts the frog on the
      throat side, exactly the OUT geometry.
    - ``flip=1`` (only meaningful for ``symmetric`` pieces): Y-mirror
      across the longitudinal axis: ``dy → -dy``, ``dtheta → -dtheta``.

    Switches use rotate, curves use flip; the two operations don't combine
    on any current catalog piece. Returned theta is in radians.
    """
    port = piece_spec.ports[port_name]
    dx, dy, dtheta = port.dx, port.dy, port.dtheta

    if rotate:
        L = _piece_body_length(piece_spec)
        dx = L - dx
        dy = -dy
        dtheta = dtheta + math.pi

    if flip:
        dy = -dy
        dtheta = -dtheta

    return (dx, dy, dtheta)
=======
    piece_spec: TrackPieceSpec, port_name: str,
) -> Pose:
    """Pose of ``port_name`` in the piece-local frame (port A at origin).

    Returned theta is in radians (matches V2 PortDef convention).
    """
    port = piece_spec.ports[port_name]
    return (port.dx, port.dy, port.dtheta)
>>>>>>> Stashed changes


COMPONENT_TILING_GAP_STUDS: float = 20.0
"""Horizontal gap between tiled components (decoder layout)."""


<<<<<<< Updated upstream
def _auto_center_layout(
    slot_poses: Dict[int, Pose],
    *,
    boundary_min_x: float,
    boundary_max_x: float,
    boundary_min_y: float,
    boundary_max_y: float,
    anchor_offset_xy: Tuple[float, float],
) -> Dict[int, Pose]:
    """Phase 3 auto-centering: translate ``slot_poses`` so the bbox center
    lands at the boundary center plus a small chromosome-supplied offset.

    Theta is preserved (the FK orientation already encodes the layout's
    rotation about its own center). With Phase 3's anchor reinterpretation
    the chromosome's xy genes are bounded to +/-5% of boundary width/height,
    so the additive offset only fine-tunes placement; auto-centering does
    the heavy lifting.
    """
    if not slot_poses:
        return slot_poses
    xs = [p[0] for p in slot_poses.values()]
    ys = [p[1] for p in slot_poses.values()]
    layout_cx = (min(xs) + max(xs)) / 2.0
    layout_cy = (min(ys) + max(ys)) / 2.0
    bcx = (boundary_min_x + boundary_max_x) / 2.0
    bcy = (boundary_min_y + boundary_max_y) / 2.0
    dx = bcx - layout_cx + float(anchor_offset_xy[0])
    dy = bcy - layout_cy + float(anchor_offset_xy[1])
    return {s: (p[0] + dx, p[1] + dy, p[2]) for s, p in slot_poses.items()}


def _propagate_poses(
    slot_pieces: Dict[int, str],
    slot_flips: Dict[int, int],
    slot_rotates: Dict[int, int],
=======
def _propagate_poses(
    slot_pieces: Dict[int, str],
>>>>>>> Stashed changes
    edges: List[PortEdge],
    components: List[Set[int]],
    spec: TrackCatalogSpec,
    anchor: Pose,
) -> Tuple[Dict[int, Pose], List[CycleResidual]]:
    """BFS each component from its own anchor pose, tiling them horizontally.

    The first component is anchored at the chromosome's anchor pose. Each
    subsequent component is placed to the right of the previous component's
    rightmost slot by ``COMPONENT_TILING_GAP_STUDS``, with no rotation of its
    own. This prevents multi-component layouts from stacking on top of each
    other in world space, which would otherwise force the boundary constraint
    to do all the separation work.

    When an edge reaches a slot whose pose is already known via a different
    walk, the (signed) residual is recorded as a :class:`CycleResidual`.
    """
    slot_poses: Dict[int, Pose] = {}
    all_residuals: List[CycleResidual] = []

    incidence: Dict[int, List[PortEdge]] = {s: [] for s in slot_pieces}
    for edge in edges:
        incidence.setdefault(edge.slot_a, []).append(edge)
        incidence.setdefault(edge.slot_b, []).append(edge)

    cursor_x: Optional[float] = None  # rightmost extent of placed components

    for i, component in enumerate(sorted(components, key=lambda c: min(c))):
        # BFS at origin (or chromosome anchor for the first component) to get
        # the component's relative pose layout.
        bfs_anchor = anchor if i == 0 else IDENTITY
        component_poses, residuals = _bfs_component(
<<<<<<< Updated upstream
            min(component), component, slot_pieces, slot_flips, slot_rotates,
            incidence, spec, bfs_anchor,
=======
            min(component), component, slot_pieces, incidence, spec, bfs_anchor,
>>>>>>> Stashed changes
        )
        all_residuals.extend(residuals)

        if not component_poses:
            continue

        if i > 0:
            # Translate so the component's bounding-box left edge lands at cursor_x
            # and its bottom edge aligns with the chromosome anchor's y.
            xs = [p[0] for p in component_poses.values()]
            ys = [p[1] for p in component_poses.values()]
            shift_x = cursor_x - min(xs)
            shift_y = anchor[1] - min(ys)
            component_poses = {
                s: (p[0] + shift_x, p[1] + shift_y, p[2])
                for s, p in component_poses.items()
            }

        slot_poses.update(component_poses)

        xs = [p[0] for p in component_poses.values()]
        cursor_x = max(xs) + COMPONENT_TILING_GAP_STUDS

    return slot_poses, all_residuals


def _bfs_component(
    anchor_slot: int,
    component: Set[int],
    slot_pieces: Dict[int, str],
<<<<<<< Updated upstream
    slot_flips: Dict[int, int],
    slot_rotates: Dict[int, int],
=======
>>>>>>> Stashed changes
    incidence: Dict[int, List[PortEdge]],
    spec: TrackCatalogSpec,
    anchor_pose: Pose,
) -> Tuple[Dict[int, Pose], List[CycleResidual]]:
    """BFS pose propagation within a single connected component."""
    poses: Dict[int, Pose] = {anchor_slot: anchor_pose}
    residuals: List[CycleResidual] = []
    queue = deque([anchor_slot])
    visited_edges: Set[int] = set()

    while queue:
        slot_a = queue.popleft()
        if slot_a not in component:
            continue
        piece_a = spec.by_id[slot_pieces[slot_a]]
<<<<<<< Updated upstream
        flip_a = slot_flips.get(slot_a, 0)
        rotate_a = slot_rotates.get(slot_a, 0)
=======
>>>>>>> Stashed changes
        pose_a = poses[slot_a]

        for edge in incidence[slot_a]:
            edge_id = id(edge)
            if edge_id in visited_edges:
                continue
            visited_edges.add(edge_id)

            other_slot, other_port = edge.other_endpoint(slot_a)
            this_port = edge.port_a if edge.slot_a == slot_a else edge.port_b

            # V2 mating convention: the partner piece's local +x aligns with
            # this side's outgoing port heading (no 180 deg flip — see file
            # header).
<<<<<<< Updated upstream
            port_local = _port_local_pose(piece_a, this_port, flip_a, rotate_a)
            port_world = pose_compose(pose_a, port_local)

            partner_piece = spec.by_id[slot_pieces[other_slot]]
            partner_flip = slot_flips.get(other_slot, 0)
            partner_rotate = slot_rotates.get(other_slot, 0)
            partner_port_local = _port_local_pose(
                partner_piece, other_port, partner_flip, partner_rotate,
            )
=======
            port_local = _port_local_pose(piece_a, this_port)
            port_world = pose_compose(pose_a, port_local)

            partner_piece = spec.by_id[slot_pieces[other_slot]]
            partner_port_local = _port_local_pose(partner_piece, other_port)
>>>>>>> Stashed changes
            partner_pose = pose_compose(port_world, pose_inverse(partner_port_local))

            if other_slot in poses:
                existing = poses[other_slot]
                dx = partner_pose[0] - existing[0]
                dy = partner_pose[1] - existing[1]
                dtheta = _wrap_pi(partner_pose[2] - existing[2])
                residuals.append(CycleResidual(
                    slot_a=slot_a, slot_b=other_slot,
                    dx=dx, dy=dy, dtheta=dtheta,
                ))
            else:
                poses[other_slot] = partner_pose
                queue.append(other_slot)

    return poses, residuals


def _wrap_pi(theta: float) -> float:
    """Wrap an angle in radians to (-pi, pi]."""
    wrapped = (theta + math.pi) % (2 * math.pi) - math.pi
    if wrapped == -math.pi:
        wrapped = math.pi
    return wrapped


# =============================================================================
# Step 5 — loose-port detection
# =============================================================================


def port_graph_to_layout(graph, catalog: TrackCatalog):
<<<<<<< Updated upstream
    """Walk every connected component in a PortGraph and produce a Layout.

    For each component the walk follows the same "tree-style follow any
    unvisited edge" pattern: visits each slot once and uses ``id(edge)``
    to avoid revisiting. Components are emitted in size-descending order;
    each gets its own closing-duplicate state at its tail, and the
    component start indices are recorded in ``component_breaks`` so the
    renderer can lay each component's curves out independently.

    Multi-component is the v0+Phase-5 reality: NSGA-II finds layouts with
    disjoint closed loops + open chains. Walking only the largest dropped
    those extra components from the rendered PNG even though they counted
    toward the reported piece total.
=======
    """Walk the largest cycle in a PortGraph and produce a legacy Layout.

    Used for visualization: src_v2.visualization.plot_layout consumes a
    sequential Layout (indices + (n+1,3) states array of degree-theta poses).
    The walk visits each slot in the largest component once, following edges
    that progress to unvisited slots, using port A as the entry-side
    convention so the FK chain reads naturally.
>>>>>>> Stashed changes
    """
    import numpy as np
    from .geometry import Layout

    if not graph.connected_components:
<<<<<<< Updated upstream
        return Layout(
            indices=np.array([], dtype=np.int32), states=np.zeros((1, 3)),
        )

    edge_incidence: Dict[int, List[PortEdge]] = {}
    for edge in graph.edges:
        edge_incidence.setdefault(edge.slot_a, []).append(edge)
        edge_incidence.setdefault(edge.slot_b, []).append(edge)

    components_sorted = sorted(graph.connected_components, key=len, reverse=True)

    all_indices: list = []
    all_states: list = []
    all_flips: list = []
    component_breaks: list = []

    for component in components_sorted:
        walk = _walk_component(component, edge_incidence)
        keep = [s for s in walk if s in graph.slot_indices and s in graph.slot_poses]
        if not keep:
            continue
        comp_indices = [graph.slot_indices[s] for s in keep]
        comp_states = [
            [graph.slot_poses[s][0], graph.slot_poses[s][1],
             math.degrees(graph.slot_poses[s][2])]
            for s in keep
        ]
        comp_flips = [int(graph.slot_flips.get(s, 0)) for s in keep]
        component_breaks.append(len(all_indices))
        all_indices.extend(comp_indices)
        all_states.extend(comp_states)
        all_flips.extend(comp_flips)
        all_states.append(comp_states[0])  # closing duplicate per component

    if not all_indices:
        return Layout(
            indices=np.array([], dtype=np.int32), states=np.zeros((1, 3)),
        )

    return Layout(
        indices=np.array(all_indices, dtype=np.int32),
        states=np.array(all_states, dtype=np.float64),
        component_breaks=np.array(component_breaks, dtype=np.int32),
        flips=np.array(all_flips, dtype=np.int8),
    )


def _walk_component(component, edge_incidence) -> List[int]:
    """DFS that visits **every** slot in the component, including branches.

    The previous single-path follow stopped when the current slot ran out
    of unvisited neighbours, so at a switch (3+ ports) only one branch
    was emitted and 2+ pieces per branching node were silently dropped.
    DFS pushes branch successors onto a stack so all of them eventually
    get visited.

    The order is deterministic (sorted slot ids), so the renderer's curve
    ``next_theta`` lookup remains stable across re-runs even though it is
    no longer guaranteed to be along the train-traversable route at
    branching slots. Visualisation correctness is preferred over perfect
    curve direction at switches; an open-loop drawn at the wrong sweep
    is better than missing pieces entirely.
    """
    walk: List[int] = []
    visited: Set[int] = set()
    stack: List[int] = [min(component)]
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        walk.append(current)
        for edge in edge_incidence.get(current, []):
            if edge.slot_a not in component or edge.slot_b not in component:
                continue
            other = edge.slot_b if edge.slot_a == current else edge.slot_a
            if other not in visited:
                stack.append(other)
    # Pick up any slots not reached via edges (isolated singletons).
    for slot in sorted(component):
        if slot not in visited:
            walk.append(slot)
            visited.add(slot)
    return walk
=======
        return Layout(indices=np.array([], dtype=np.int32), states=np.zeros((1, 3)))

    largest = max(graph.connected_components, key=len)

    incidence: Dict[int, List[PortEdge]] = {}
    for edge in graph.edges:
        if edge.slot_a in largest and edge.slot_b in largest:
            incidence.setdefault(edge.slot_a, []).append(edge)
            incidence.setdefault(edge.slot_b, []).append(edge)

    start = min(largest)
    walk: List[int] = [start]
    visited: Set[int] = {start}
    visited_edges: Set[int] = set()

    current = start
    while True:
        next_slot = None
        for edge in incidence.get(current, []):
            if id(edge) in visited_edges:
                continue
            other = edge.slot_b if edge.slot_a == current else edge.slot_a
            if other not in visited:
                next_slot = other
                visited_edges.add(id(edge))
                break

        if next_slot is None:
            break
        walk.append(next_slot)
        visited.add(next_slot)
        current = next_slot

    indices = np.array(
        [graph.slot_indices[s] for s in walk if s in graph.slot_indices],
        dtype=np.int32,
    )

    states_list = []
    for slot in walk:
        pose = graph.slot_poses.get(slot)
        if pose is None:
            continue
        states_list.append([pose[0], pose[1], math.degrees(pose[2])])

    if not states_list:
        return Layout(indices=indices, states=np.zeros((1, 3)))

    states_list.append(states_list[0])  # closed-loop final state = first
    states = np.array(states_list, dtype=np.float64)
    return Layout(indices=indices, states=states)
>>>>>>> Stashed changes


def _find_loose_ports(
    slot_pieces: Dict[int, str],
    edges: List[PortEdge],
    spec: TrackCatalogSpec,
) -> List[Tuple[int, str]]:
    """Return (slot_idx, port_name) for every port not present in any edge."""
    used: Set[Tuple[int, str]] = set()
    for edge in edges:
        used.add((edge.slot_a, edge.port_a))
        used.add((edge.slot_b, edge.port_b))

    loose: List[Tuple[int, str]] = []
    for slot_idx, piece_id in slot_pieces.items():
        piece_spec = spec.by_id.get(piece_id)
        if piece_spec is None:
            continue
        for port_name in piece_spec.ports:
            if (slot_idx, port_name) not in used:
                loose.append((slot_idx, port_name))
    return loose
<<<<<<< Updated upstream


# =============================================================================
# Step 6 — branch labels (cycle membership per slot+route)
# =============================================================================


def _compute_branch_labels(
    components: List[Set[int]],
    edges: List[PortEdge],
    slot_pieces: Dict[int, str],
    spec: TrackCatalogSpec,
) -> Dict[Tuple[int, str], int]:
    """Map each (slot, route_name) to the id of the cycle it lies on.

    Iterates the fundamental cycle basis of every connected component;
    each cycle receives the next ``cycle_id``. A slot in a cycle adopts the
    catalog route whose port set matches the slot's ports touched by the
    cycle. Switches in a passing siding land on two cycles (``through`` on
    the main loop, ``diverging`` on the branch) and so receive two labels.
    """
    return {
        (slot, route): cycle_id
        for cycle_id, cycle in enumerate(_iter_cycles(components, edges))
        for slot, route in _slot_route_labels(cycle, slot_pieces, spec)
    }


def _iter_cycles(
    components: List[Set[int]], edges: List[PortEdge],
) -> Iterator[Set[PortEdge]]:
    """Yield each fundamental cycle as a set of its edges, per component.

    Multi-edges between the same slot pair (rare in practice) collapse to
    one edge in the simple graph fed to ``nx.cycle_basis``; the lookup
    table picks the canonical representative.
    """
    for component in components:
        # Sort for determinism (Rule 3, PLAN §10.3 I1): set iteration order
        # is unspecified per Python language ref + PEP 456; PYTHONHASHSEED
        # only salts str/bytes hashes, not int. Without sort, lookup-table
        # collision tie-breaks (last-write-wins) and nx.cycle_basis cycle
        # ordering vary across CPython versions and worker processes.
        inside = sorted(
            (e for e in edges if {e.slot_a, e.slot_b} <= component),
            key=lambda e: (e.slot_a, e.port_a, e.slot_b, e.port_b),
        )
        lookup = {frozenset({e.slot_a, e.slot_b}): e for e in inside}
        graph = nx.Graph((e.slot_a, e.slot_b) for e in inside)
        cycles = sorted(nx.cycle_basis(graph), key=lambda c: tuple(sorted(c)))
        for cycle in cycles:
            yield {
                lookup[frozenset({u, v})]
                for u, v in pairwise(cycle + [cycle[0]])
            }


def _slot_route_labels(
    cycle: Set[PortEdge],
    slot_pieces: Dict[int, str],
    spec: TrackCatalogSpec,
) -> Iterator[Tuple[int, str]]:
    """Yield (slot, route_name) for slots whose ports-on-cycle match a route."""
    slot_ports: Dict[int, Set[str]] = defaultdict(set)
    for edge in cycle:
        slot_ports[edge.slot_a].add(edge.port_a)
        slot_ports[edge.slot_b].add(edge.port_b)
    for slot, ports in slot_ports.items():
        piece_spec = spec.by_id.get(slot_pieces.get(slot, ""))
        if piece_spec is None:
            continue
        route = next(
            (name for name, seq in piece_spec.routes.items() if set(seq) == ports),
            None,
        )
        if route is not None:
            yield slot, route
=======
>>>>>>> Stashed changes
