"""Speed profile computation for track layouts.

Time-optimal speed profiling using a 3-pass algorithm with friction ellipse
constraints. This is the physics scoring module — it turns geometry into speed.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .physics import DEFAULT_TRAIN_CONFIG, TrainConfig, available_accel, v_eff_array
from ..catalog import TrackCatalog
from ..geometry import Layout


@dataclass(frozen=True)
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
    train_config: TrainConfig = DEFAULT_TRAIN_CONFIG,
    safety_margin: float = 1.0,
) -> SpeedProfile:
    """Compute time-optimal speed profile using 3-pass algorithm.

    Algorithm:
    1. Pass 1: Curvature limits via v_eff_array(R) at mu_design
    2. Pass 2: Forward acceleration - respect train_config.max_accel
    3. Pass 3: Backward braking - respect train_config.brake_decel
    4. Double-unroll method for closed loops

    Args:
        layout: Track layout with geometry
        catalog: Track catalog for piece properties (also provides stud_mm)
        train_config: Portable locomotive physics (default: DEFAULT_TRAIN_CONFIG)
        safety_margin: Multiplier (in (0, 1]) applied to every Pass-1 cap so
            the operating speed stays strictly below the derailment cap.
            Default 1.0 applies no margin.

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
    stud_to_m = catalog.stud_mm / 1000.0
    arc_lengths = catalog.get_arc_lengths(layout.indices) * stud_to_m  # meters
    # Route-aware radii when the layout carries a traversed-route index per
    # piece (branch/diagonal segments are curve-limited); else default-route.
    # A whole MultiPathLayout has no single route, so it falls back to per-piece.
    route_indices = getattr(layout, "route_indices", None)
    radii_m = (
        catalog.get_route_radii(layout.indices, route_indices)
        if route_indices is not None
        else catalog.get_radii(layout.indices)
    ) / 1000.0

    # Pass 1: per-segment derailment and motor caps, derived here from the
    # train physics alone — the catalog supplies geometry, never a speed, so
    # one place decides how fast the train may go. safety_margin (default 1.0)
    # scales every cap so operating speed stays strictly below them.
    v_limit = v_eff_array(train_config, radii_m) * safety_margin

    # Apply 3-pass algorithm (friction ellipse reduces accel/brake on curves)
    is_closed = layout.is_closed(pos_tol=1.0, angle_tol=10.0)
    speeds = (
        _compute_speeds_double_unroll(v_limit, arc_lengths, radii_m, train_config)
        if is_closed
        else _compute_speeds_open(v_limit, arc_lengths, radii_m, train_config)
    )

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
    radii_m: NDArray[np.float64],
    train_config: TrainConfig,
) -> NDArray[np.float64]:
    """Compute speeds for closed loop using double-unroll method."""
    n = len(v_limit)
    v_limit_double = np.concatenate([v_limit, v_limit])
    arc_lengths_double = np.concatenate([arc_lengths, arc_lengths])
    radii_double = np.concatenate([radii_m, radii_m])

    v_fwd = _forward_pass(v_limit_double, arc_lengths_double, radii_double, train_config)
    v_bwd = _backward_pass(v_fwd, arc_lengths_double, radii_double, train_config)

    return v_bwd[n : 2 * n]


def _compute_speeds_open(
    v_limit: NDArray[np.float64],
    arc_lengths: NDArray[np.float64],
    radii_m: NDArray[np.float64],
    train_config: TrainConfig,
) -> NDArray[np.float64]:
    """Compute speeds for open track (no wrap-around)."""
    v_fwd = _forward_pass(v_limit, arc_lengths, radii_m, train_config)
    return _backward_pass(v_fwd, arc_lengths, radii_m, train_config)


def _forward_pass(
    v_limit: NDArray[np.float64],
    arc_lengths: NDArray[np.float64],
    radii_m: NDArray[np.float64],
    train_config: TrainConfig,
) -> NDArray[np.float64]:
    """Forward pass: apply acceleration limits with friction ellipse."""
    n = len(v_limit)
    v_fwd = np.zeros(n)
    v_fwd[0] = v_limit[0]

    for i in range(1, n):
        a_max = available_accel(train_config, float(v_fwd[i - 1]), float(radii_m[i - 1]))
        v_accel = np.sqrt(v_fwd[i - 1] ** 2 + 2 * a_max * arc_lengths[i - 1])
        v_fwd[i] = min(v_limit[i], v_accel)

    return v_fwd


def _backward_pass(
    v_fwd: NDArray[np.float64],
    arc_lengths: NDArray[np.float64],
    radii_m: NDArray[np.float64],
    train_config: TrainConfig,
) -> NDArray[np.float64]:
    """Backward pass: apply braking limits with friction ellipse."""
    n = len(v_fwd)
    v_bwd = np.zeros(n)
    v_bwd[-1] = v_fwd[-1]

    for i in range(n - 2, -1, -1):
        a_brake = available_accel(
            train_config, float(v_bwd[i + 1]), float(radii_m[i]), is_braking=True
        )
        v_brake = np.sqrt(v_bwd[i + 1] ** 2 + 2 * a_brake * arc_lengths[i])
        v_bwd[i] = min(v_fwd[i], v_brake)

    return v_bwd


def _compute_lap_time(speeds: NDArray[np.float64], arc_lengths: NDArray[np.float64]) -> float:
    """Compute lap time from speeds and arc lengths."""
    safe_speeds = np.where(speeds > 0, speeds, 0.001)
    return float(np.sum(arc_lengths / safe_speeds))
