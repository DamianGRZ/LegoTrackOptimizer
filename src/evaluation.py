"""Track layout evaluation: speed profiles, objectives, and constraints."""

from dataclasses import dataclass
from typing import Dict

import numpy as np
from numpy.typing import NDArray

from .config import BoundaryConfig, PhysicsConfig
from .data import TrackCatalog
from .geometry import Layout


@dataclass
class SpeedProfile:
    """Time-optimal speed profile along track layout."""

    speeds: NDArray[np.float64]  # (n,) speed at each segment in m/s
    avg_speed: float  # m/s
    lap_time: float  # seconds
    total_distance: float  # meters
    max_speed: float  # m/s
    min_speed: float  # m/s


def compute_speed_profile(
    layout: Layout,
    catalog: TrackCatalog,
    physics: PhysicsConfig,
) -> SpeedProfile:
    """Compute time-optimal speed profile using 3-pass algorithm.

    Derailing prevention is built into Pass 1 via curvature limits.
    No separate derailing constraint needed - physics model ensures safety.

    Algorithm (from Locomotive_dynamics.md):
    1. Pass 1: Curvature limits - v_limit[i] = SF × √(μ × g × R[i]) PREVENTS DERAILING
    2. Pass 2: Forward acceleration - respect a_max = 3.92 m/s²
    3. Pass 3: Backward braking - respect a_brake = 2.45 m/s²
    4. Double-unroll method for closed loops

    Args:
        layout: Track layout with geometry
        catalog: Track catalog for piece properties
        physics: Physics configuration (SF=0.8, mu=0.30, etc.)

    Returns:
        SpeedProfile with speeds, avg_speed, lap_time, etc.
    """
    if layout.n_pieces == 0:
        return SpeedProfile(
            speeds=np.array([]),
            avg_speed=0.0,
            lap_time=0.0,
            total_distance=0.0,
            max_speed=0.0,
            min_speed=0.0,
        )

    # Get piece properties and convert units
    stud_to_m = physics.stud_mm / 1000.0
    arc_lengths = catalog.get_arc_lengths(layout.indices) * stud_to_m  # meters
    radii_m = catalog.get_radii(layout.indices) / 1000.0  # meters
    speed_limits = catalog.get_speed_limits(layout.indices)  # m/s

    # Pass 1: Curvature speed limits (vectorized) - THIS PREVENTS DERAILING
    v_curve = np.where(
        np.isinf(radii_m),
        physics.motor_top_speed,
        physics.safety_factor * np.sqrt(physics.friction_coeff * physics.gravity * radii_m),
    )
    v_limit = np.minimum(v_curve, speed_limits)

    # Apply 3-pass algorithm
    is_closed = layout.is_closed(pos_tol=1.0, angle_tol=10.0)
    speeds = _compute_speeds_double_unroll(v_limit, arc_lengths, physics) if is_closed else _compute_speeds_open(v_limit, arc_lengths, physics)

    # Compute metrics
    total_distance = float(np.sum(arc_lengths))
    lap_time = _compute_lap_time(speeds, arc_lengths)
    avg_speed = total_distance / lap_time if lap_time > 0 else 0.0

    return SpeedProfile(
        speeds=speeds,
        avg_speed=avg_speed,
        lap_time=lap_time,
        total_distance=total_distance,
        max_speed=float(np.max(speeds)) if len(speeds) > 0 else 0.0,
        min_speed=float(np.min(speeds)) if len(speeds) > 0 else 0.0,
    )


def _compute_speeds_double_unroll(
    v_limit: NDArray[np.float64],
    arc_lengths: NDArray[np.float64],
    physics: PhysicsConfig,
) -> NDArray[np.float64]:
    """Compute speeds for closed loop using double-unroll method."""
    n = len(v_limit)
    v_limit_double = np.concatenate([v_limit, v_limit])
    arc_lengths_double = np.concatenate([arc_lengths, arc_lengths])

    v_fwd = _forward_pass(v_limit_double, arc_lengths_double, physics.max_accel)
    v_bwd = _backward_pass(v_fwd, arc_lengths_double, physics.brake_decel)

    return v_bwd[n : 2 * n]


def _compute_speeds_open(
    v_limit: NDArray[np.float64],
    arc_lengths: NDArray[np.float64],
    physics: PhysicsConfig,
) -> NDArray[np.float64]:
    """Compute speeds for open track (no wrap-around)."""
    v_fwd = _forward_pass(v_limit, arc_lengths, physics.max_accel)
    return _backward_pass(v_fwd, arc_lengths, physics.brake_decel)


def _forward_pass(
    v_limit: NDArray[np.float64],
    arc_lengths: NDArray[np.float64],
    a_max: float,
) -> NDArray[np.float64]:
    """Forward pass: Apply acceleration limits."""
    n = len(v_limit)
    v_fwd = np.zeros(n)
    v_fwd[0] = v_limit[0]

    for i in range(1, n):
        v_accel = np.sqrt(v_fwd[i - 1] ** 2 + 2 * a_max * arc_lengths[i - 1])
        v_fwd[i] = min(v_limit[i], v_accel)

    return v_fwd


def _backward_pass(
    v_fwd: NDArray[np.float64],
    arc_lengths: NDArray[np.float64],
    a_brake: float,
) -> NDArray[np.float64]:
    """Backward pass: Apply braking limits."""
    n = len(v_fwd)
    v_bwd = np.zeros(n)
    v_bwd[-1] = v_fwd[-1]

    for i in range(n - 2, -1, -1):
        v_brake = np.sqrt(v_bwd[i + 1] ** 2 + 2 * a_brake * arc_lengths[i])
        v_bwd[i] = min(v_fwd[i], v_brake)

    return v_bwd


def _compute_lap_time(speeds: NDArray[np.float64], arc_lengths: NDArray[np.float64]) -> float:
    """Compute lap time from speeds and arc lengths."""
    safe_speeds = np.where(speeds > 0, speeds, 0.001)
    return float(np.sum(arc_lengths / safe_speeds))


def compute_objectives(
    layout: Layout,
    speed_profile: SpeedProfile,
    catalog: TrackCatalog,
    total_inventory: int,
) -> NDArray[np.float64]:
    """Compute 2 objectives (all minimized).

    F[0] = -utilization (maximize piece usage)
    F[1] = -avg_speed (maximize speed)

    Area removed as objective - it caused circles to be non-dominated by ovals
    despite lower utilization/speed. With 2 objectives, ovals now dominate circles
    on both dimensions (higher utilization AND higher speed from straights).

    Derailing is prevented by physics model in speed profile, not a constraint.
    Inventory limits enforced by G[3] constraint.

    Args:
        layout: Track layout
        speed_profile: Computed speed profile (already prevents derailing)
        catalog: Track catalog
        total_inventory: Total number of pieces available

    Returns:
        Array of shape (2,) with objective values
    """
    utilization = layout.n_pieces / total_inventory if total_inventory > 0 else 0.0

    return np.array(
        [-utilization, -speed_profile.avg_speed],
        dtype=np.float64,
    )


def compute_constraints(
    layout: Layout,
    chromosome: NDArray,
    inventory: Dict[str, int],
    catalog: TrackCatalog,
    closure_tol: float,
    angle_tol: float,
    boundary: BoundaryConfig,
) -> NDArray[np.float64]:
    """Compute 5 inequality constraints (g <= 0 is feasible).

    G[0] = (closure_error - tolerance) / tolerance
    G[1] = (angle_error - tolerance) / tolerance
    G[2] = boundary_violation / diagonal
    G[3] = inventory_excess (count violations) - USER INVENTORY CONSTRAINT
    G[4] = orphan_switch_count

    Note: No derailing constraint - speed profile already ensures safe speeds.

    Args:
        layout: Track layout
        chromosome: Chromosome array
        inventory: Inventory limits {piece_id: count}
        catalog: Track catalog
        closure_tol: Closure tolerance in studs
        angle_tol: Angle tolerance in degrees
        boundary: Boundary configuration

    Returns:
        Array of shape (5,) with constraint values
    """
    g_closure = (layout.closure_error - closure_tol) / closure_tol
    g_angle = (layout.angle_error - angle_tol) / angle_tol
    g_boundary = _compute_boundary_violation(layout, boundary)
    g_inventory = _compute_inventory_excess(chromosome, inventory, catalog)
    g_orphan = _compute_orphan_switches(chromosome)

    return np.array([g_closure, g_angle, g_boundary, g_inventory, g_orphan], dtype=np.float64)


def _compute_boundary_violation(layout: Layout, boundary: BoundaryConfig) -> float:
    """Compute normalized boundary violation constraint."""
    x = layout.states[:, 0]
    y = layout.states[:, 1]

    violations = np.maximum(0, np.concatenate([
        boundary.min_x - x,
        x - boundary.max_x,
        boundary.min_y - y,
        y - boundary.max_y,
    ]))

    max_violation = np.max(violations) if len(violations) > 0 else 0.0
    return float(max_violation / boundary.diagonal)


def _compute_inventory_excess(
    chromosome: NDArray,
    inventory: Dict[str, int],
    catalog: TrackCatalog,
) -> float:
    """Compute inventory excess constraint (count of violations)."""
    valid_indices = chromosome[chromosome >= 0]
    unique_indices, counts = np.unique(valid_indices, return_counts=True)

    excess = 0
    for piece_idx, count in zip(unique_indices, counts):
        piece = catalog[int(piece_idx)]
        if piece and piece.id in inventory:
            excess += max(0, count - inventory[piece.id])

    return float(excess)


def _compute_orphan_switches(chromosome: NDArray) -> float:
    """Compute orphan switch count (unpaired IN/OUT switches)."""
    valid_indices = chromosome[chromosome >= 0]
    unique_indices, counts = np.unique(valid_indices, return_counts=True)
    usage = dict(zip(unique_indices, counts))

    # Switch indices: 5=LEFT_IN, 6=LEFT_OUT, 7=RIGHT_IN, 8=RIGHT_OUT
    switch_pairs = [(5, 6), (7, 8)]
    orphan_count = sum(abs(usage.get(in_idx, 0) - usage.get(out_idx, 0)) for in_idx, out_idx in switch_pairs)

    return float(orphan_count)
