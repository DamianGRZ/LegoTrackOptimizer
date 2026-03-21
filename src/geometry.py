"""Track layout geometry using forward kinematics (FK) and multi-phase topology."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .data import TrackCatalog
from .topology import (
    BranchDef,
    BranchGeometry,
    LayoutChromosome,
    LayoutGeometry,
    LoopDef,
    LoopGeometry,
)


def compute_fk_chain(fk_deltas: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute cumulative forward kinematics chain from piece deltas.

    Args:
        fk_deltas: (n, 3) array of [dx, dy, dtheta] for each piece.

    Returns:
        (n+1, 3) array of cumulative states [x, y, theta].
        State 0 is origin [0, 0, 0], state i+1 is after piece i.
    """
    n = len(fk_deltas)
    states = np.zeros((n + 1, 3), dtype=np.float64)

    for i in range(n):
        dx, dy, dtheta = fk_deltas[i]
        theta_rad = np.radians(states[i, 2])
        cos_t = np.cos(theta_rad)
        sin_t = np.sin(theta_rad)

        # Transform local displacement to world coordinates
        states[i + 1, 0] = states[i, 0] + dx * cos_t - dy * sin_t
        states[i + 1, 1] = states[i, 1] + dx * sin_t + dy * cos_t
        states[i + 1, 2] = states[i, 2] + dtheta

    return states


@dataclass
class Layout:
    """Legacy single-loop layout representation (Phase 1 compatibility)."""

    indices: NDArray[np.int32]
    states: NDArray[np.float64]

    @property
    def n_pieces(self) -> int:
        """Number of pieces in layout."""
        return len(self.indices)

    @property
    def final_state(self) -> NDArray[np.float64]:
        """Final state [x, y, theta]."""
        return self.states[-1]

    @property
    def closure_error(self) -> float:
        """Euclidean distance from final position to starting position."""
        dx = self.final_state[0] - self.states[0, 0]
        dy = self.final_state[1] - self.states[0, 1]
        return float(np.sqrt(dx ** 2 + dy ** 2))

    @property
    def angle_error(self) -> float:
        """Normalized angular deviation from 360 degrees."""
        total = abs(self.final_state[2])
        if total == 0:
            return 360.0
        # Get remainder after full circles
        remainder = total % 360
        # Error is distance to nearest multiple of 360
        return min(remainder, 360 - remainder)

    @property
    def total_angle(self) -> float:
        """Total accumulated angle in degrees."""
        return float(self.final_state[2])

    @property
    def bounding_box(self) -> Tuple[float, float, float, float]:
        """Bounding box (min_x, min_y, max_x, max_y)."""
        if len(self.states) == 0:
            return (0.0, 0.0, 0.0, 0.0)

        min_x = float(np.min(self.states[:, 0]))
        max_x = float(np.max(self.states[:, 0]))
        min_y = float(np.min(self.states[:, 1]))
        max_y = float(np.max(self.states[:, 1]))
        return (min_x, min_y, max_x, max_y)

    @property
    def area(self) -> float:
        """Bounding box area."""
        min_x, min_y, max_x, max_y = self.bounding_box
        return (max_x - min_x) * (max_y - min_y)

    def is_closed(self, pos_tol: float = 0.5, angle_tol: float = 5.0) -> bool:
        """Check if layout is closed within tolerances."""
        return self.closure_error <= pos_tol and self.angle_error <= angle_tol


def build_layout(chromosome: NDArray, catalog: TrackCatalog) -> Layout:
    """Build Layout from chromosome array.

    Args:
        chromosome: Array of piece indices (-1 for empty slots).
        catalog: Track catalog for FK lookup.

    Returns:
        Layout with computed states.
    """
    chromosome = np.asarray(chromosome, dtype=np.int32)
    valid_mask = chromosome >= 0
    indices = chromosome[valid_mask]

    if len(indices) == 0:
        return Layout(indices=np.array([], dtype=np.int32), states=np.zeros((1, 3)))

    fk_deltas = catalog.get_fk(indices)
    states = compute_fk_chain(fk_deltas)

    return Layout(indices=indices, states=states)


def compute_closure_metrics(states: NDArray[np.float64]) -> Tuple[float, float]:
    """Compute closure error and angle error from state trajectory.

    Args:
        states: (n+1, 3) array of states [x, y, theta].

    Returns:
        (closure_error, angle_error) tuple.
    """
    if len(states) <= 1:
        return (0.0, 360.0)

    final = states[-1]
    closure_error = float(np.sqrt(final[0] ** 2 + final[1] ** 2))
    total_angle = abs(final[2])
    if total_angle == 0:
        angle_error = 360.0
    else:
        remainder = total_angle % 360
        angle_error = min(remainder, 360 - remainder)

    return (closure_error, angle_error)


def estimate_angular_deficit(
    indices: NDArray, catalog: TrackCatalog, target: float = 360.0
) -> float:
    """Calculate degrees needed to close loop.

    Args:
        indices: Piece indices array.
        catalog: Track catalog.
        target: Target total angle (360 for single loop).

    Returns:
        Degrees needed (positive = need more left turns).
    """
    fk_deltas = catalog.get_fk(indices)
    total_angle = np.sum(fk_deltas[:, 2])
    return target - abs(total_angle)


def compute_layout_geometry(
    chromosome: LayoutChromosome,
    catalog: TrackCatalog,
    initial_pose: Optional[NDArray[np.float64]] = None,
) -> LayoutGeometry:
    """Compute complete geometry for all loops.

    Args:
        chromosome: Layout chromosome with loops and branches.
        catalog: Track catalog.
        initial_pose: Optional starting pose [x, y, theta].

    Returns:
        LayoutGeometry with all computed loop geometries.
    """
    if initial_pose is None:
        initial_pose = np.zeros(3, dtype=np.float64)

    loops = []
    for loop_def in chromosome.loops:
        loop_geom = compute_loop_geometry(loop_def, catalog, initial_pose)
        loops.append(loop_geom)

    # TODO: Phase 4 - collision detection between loops
    collisions: list = []

    return LayoutGeometry(loops=loops, collisions=collisions)


def compute_loop_geometry(
    loop_def: LoopDef,
    catalog: TrackCatalog,
    initial_pose: NDArray[np.float64],
) -> LoopGeometry:
    """Compute geometry for single loop with branches.

    Args:
        loop_def: Loop definition.
        catalog: Track catalog.
        initial_pose: Starting pose [x, y, theta].

    Returns:
        LoopGeometry with computed states.
    """
    piece_indices = np.array(loop_def.main_sequence, dtype=np.int32)
    n_pieces = len(piece_indices)

    if n_pieces == 0:
        return LoopGeometry(
            loop_id=loop_def.loop_id,
            piece_indices=np.array([], dtype=np.int32),
            route_indices=np.array([], dtype=np.int32),
            states=np.zeros((1, 3)),
            arc_lengths=np.array([]),
            radii=np.array([]),
            speed_limits=np.array([]),
            closure_error=0.0,
            angle_error=360.0,
        )

    # Build route indices (default = 0)
    route_indices = np.zeros(n_pieces, dtype=np.int32)
    for spec in loop_def.route_specs:
        if 0 <= spec.piece_position < n_pieces:
            route_indices[spec.piece_position] = spec.route_idx

    # Get FK deltas with routes
    fk_deltas = catalog.get_fk_with_routes(piece_indices, route_indices)

    # Compute state trajectory
    local_states = compute_fk_chain(fk_deltas)
    states = _apply_initial_pose(local_states, initial_pose)

    # Compute closure metrics
    closure_error, angle_error = compute_closure_metrics(local_states)

    # Get per-piece properties
    arc_lengths = catalog.get_arc_lengths(piece_indices)
    radii = catalog.get_radii(piece_indices)
    speed_limits = catalog.get_speed_limits(piece_indices)

    # Compute bounding box
    bbox_min = np.min(states[:, :2], axis=0)
    bbox_max = np.max(states[:, :2], axis=0)

    # Compute branch geometries
    branches = []
    for branch_def in loop_def.branches:
        diverge_idx = branch_def.diverge_position
        if 0 <= diverge_idx < len(states):
            diverge_pose = states[diverge_idx]
            branch_geom = compute_branch_geometry(
                branch_def, catalog, diverge_pose, states, diverge_idx
            )
            branches.append(branch_geom)

    return LoopGeometry(
        loop_id=loop_def.loop_id,
        piece_indices=piece_indices,
        route_indices=route_indices,
        states=states,
        arc_lengths=arc_lengths,
        radii=radii,
        speed_limits=speed_limits,
        closure_error=closure_error,
        angle_error=angle_error,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        branches=branches,
    )


def compute_branch_geometry(
    branch_def: BranchDef,
    catalog: TrackCatalog,
    diverge_pose: NDArray[np.float64],
    parent_states: NDArray[np.float64],
    diverge_switch_idx: Optional[int] = None,
) -> BranchGeometry:
    """Compute geometry for single branch.

    Args:
        branch_def: Branch definition.
        catalog: Track catalog.
        diverge_pose: Pose at divergence point.
        parent_states: States from parent loop.
        diverge_switch_idx: Index of divergence switch in parent.

    Returns:
        BranchGeometry with computed states.
    """
    piece_indices = np.array(branch_def.piece_indices, dtype=np.int32)
    n_pieces = len(piece_indices)

    if n_pieces == 0:
        return BranchGeometry(
            branch_id=branch_def.branch_id,
            parent_loop_id=branch_def.parent_loop_id,
            piece_indices=np.array([], dtype=np.int32),
            route_indices=np.array([], dtype=np.int32),
            states=diverge_pose.reshape(1, 3),
            arc_lengths=np.array([]),
            radii=np.array([]),
            speed_limits=np.array([]),
            is_dead_end=branch_def.is_dead_end,
        )

    # Build route indices
    route_indices = np.zeros(n_pieces, dtype=np.int32)
    for spec in branch_def.route_specs:
        if 0 <= spec.piece_position < n_pieces:
            route_indices[spec.piece_position] = spec.route_idx

    # Compute branch start pose using divergent FK from switch
    # The diverge_pose is the state BEFORE the switch, we need to apply
    # the divergent route FK to get the actual branch start
    branch_start = diverge_pose.copy()

    # If there's a diverge route spec, apply the divergent FK
    if branch_def.diverge_route.route_idx > 0 and diverge_switch_idx is not None:
        # Get the switch piece index from parent loop
        # Apply divergent route FK instead of default
        pass  # TODO: Phase 3 - apply divergent FK

    # Get FK deltas with routes
    fk_deltas = catalog.get_fk_with_routes(piece_indices, route_indices)

    # Compute state trajectory from branch start
    local_states = compute_fk_chain(fk_deltas)
    states = _apply_initial_pose(local_states, branch_start)

    # Get per-piece properties
    arc_lengths = catalog.get_arc_lengths(piece_indices)
    radii = catalog.get_radii(piece_indices)
    speed_limits = catalog.get_speed_limits(piece_indices)

    # Compute rejoin error if not dead-end
    rejoin_error = None
    rejoin_angle_error = None
    if not branch_def.is_dead_end and branch_def.rejoin_position is not None:
        # Find rejoin target in parent loop
        if branch_def.rejoin_position < len(parent_states):
            target = parent_states[branch_def.rejoin_position]
            end_state = states[-1]
            rejoin_error = float(
                np.sqrt((end_state[0] - target[0]) ** 2 + (end_state[1] - target[1]) ** 2)
            )
            rejoin_angle_error = abs(end_state[2] - target[2])

    return BranchGeometry(
        branch_id=branch_def.branch_id,
        parent_loop_id=branch_def.parent_loop_id,
        piece_indices=piece_indices,
        route_indices=route_indices,
        states=states,
        arc_lengths=arc_lengths,
        radii=radii,
        speed_limits=speed_limits,
        is_dead_end=branch_def.is_dead_end,
        rejoin_error=rejoin_error,
        rejoin_angle_error=rejoin_angle_error,
    )


def _apply_initial_pose(
    states: NDArray[np.float64], initial_pose: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Transform states from local to global frame.

    Args:
        states: Local states (n+1, 3).
        initial_pose: Initial pose [x, y, theta].

    Returns:
        Global states (n+1, 3).
    """
    x0, y0, theta0 = initial_pose
    theta_rad = np.radians(theta0)
    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)

    result = states.copy()

    # Rotate and translate each state
    for i in range(len(states)):
        x_local = states[i, 0]
        y_local = states[i, 1]
        result[i, 0] = x0 + x_local * cos_t - y_local * sin_t
        result[i, 1] = y0 + x_local * sin_t + y_local * cos_t
        result[i, 2] = theta0 + states[i, 2]

    return result


def _normalize_angle_error(total_angle: float, target: float = 360.0) -> float:
    """Normalize angle error to [0, target/2].

    Args:
        total_angle: Total accumulated angle.
        target: Target angle (360 for closed loop).

    Returns:
        Normalized error (distance to nearest multiple of target).
    """
    if total_angle == 0:
        return target
    remainder = abs(total_angle) % target
    return min(remainder, target - remainder)


def build_layout_from_chromosome(
    chromosome: LayoutChromosome, catalog: TrackCatalog
) -> Layout:
    """Build legacy Layout from LayoutChromosome (Phase 1 compatibility).

    Args:
        chromosome: Layout chromosome.
        catalog: Track catalog.

    Returns:
        Layout from main loop.
    """
    main_loop = chromosome.get_main_loop()
    if not main_loop:
        return Layout(indices=np.array([], dtype=np.int32), states=np.zeros((1, 3)))

    indices = np.array(main_loop.main_sequence, dtype=np.int32)
    fk_deltas = catalog.get_fk(indices)
    states = compute_fk_chain(fk_deltas)

    return Layout(indices=indices, states=states)


def layout_to_geometry(layout: Layout, loop_id: int = 0) -> LayoutGeometry:
    """Convert legacy Layout to LayoutGeometry (Phase 1 compatibility).

    Args:
        layout: Legacy Layout object.
        loop_id: Loop ID for the geometry.

    Returns:
        LayoutGeometry wrapping the layout.
    """
    bbox = layout.bounding_box
    loop = LoopGeometry(
        loop_id=loop_id,
        piece_indices=layout.indices,
        route_indices=np.zeros(len(layout.indices), dtype=np.int32),
        states=layout.states,
        arc_lengths=np.zeros(len(layout.indices)),  # Would need catalog
        radii=np.full(len(layout.indices), np.inf),
        speed_limits=np.full(len(layout.indices), 1.57),
        closure_error=layout.closure_error,
        angle_error=layout.angle_error,
        bbox_min=np.array([bbox[0], bbox[1]]),
        bbox_max=np.array([bbox[2], bbox[3]]),
    )

    return LayoutGeometry(loops=[loop])