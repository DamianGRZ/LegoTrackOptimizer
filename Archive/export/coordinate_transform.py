"""Coordinate transformation from LEGO Track Optimizer to BlueBrick/NCP system.

Our coordinate system:
- Origin at (0, 0), first piece entry point
- X-axis: forward direction
- Y-axis: left positive (standard math convention, CCW rotation)
- Angles: degrees, counter-clockwise positive
- Units: studs (1 stud = 8mm)
- states[i] gives position BEFORE piece i starts (entry point)

BlueBrick/NCP coordinate system:
- Parts positioned at their center
- Y-axis: down positive (screen coordinates)
- Angles: degrees, clockwise positive (due to Y flip)
- Units: studs

Transformation:
1. Compute piece center from entry position + half FK delta
2. Negate Y coordinate (flip Y-axis)
3. Negate angle (invert rotation direction)
"""

from dataclasses import dataclass

import numpy as np

from .part_mapping import PartMapping


@dataclass
class BlueBrickPose:
    """Piece pose in BlueBrick/NCP coordinate system.

    Attributes:
        x: X position in studs.
        y: Y position in studs (Y-axis flipped from our system).
        angle: Rotation angle in degrees (sign inverted from our system).
    """

    x: float
    y: float
    angle: float


def transform_to_bluebrick(
    entry_state: tuple[float, float, float],
    fk_delta: tuple[float, float, float],
    mapping: PartMapping,
) -> BlueBrickPose:
    """Transform piece from our coordinate system to BlueBrick/NCP.

    Computes the piece center position and orientation in BlueBrick coordinates.

    Args:
        entry_state: (x, y, theta) position before piece starts, in our coords.
        fk_delta: (dx, dy, dtheta) forward kinematics delta for the piece.
        mapping: Part mapping containing orientation adjustment.

    Returns:
        BlueBrickPose with center position and angle in BlueBrick coords.
    """
    x, y, theta = entry_state
    dx, dy, dtheta = fk_delta

    # Convert entry angle to radians for trig
    theta_rad = np.radians(theta)
    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)

    # Compute piece center in our coordinate system
    # Center is at entry + half of FK delta transformed to world frame
    half_dx = dx / 2
    half_dy = dy / 2
    center_x = x + half_dx * cos_t - half_dy * sin_t
    center_y = y + half_dx * sin_t + half_dy * cos_t
    center_theta = theta + dtheta / 2

    # Apply part-specific center offset (in local frame) if any
    if mapping.center_offset_x != 0 or mapping.center_offset_y != 0:
        center_x += mapping.center_offset_x * cos_t - mapping.center_offset_y * sin_t
        center_y += mapping.center_offset_x * sin_t + mapping.center_offset_y * cos_t

    # Transform to BlueBrick coordinate system:
    # - X stays the same
    # - Y is negated (flip Y-axis)
    # - Angle is negated (invert rotation due to Y flip)
    bb_x = center_x
    bb_y = -center_y
    bb_angle = -center_theta + mapping.orientation_diff

    # Normalize angle to [-180, 180] range
    bb_angle = ((bb_angle + 180) % 360) - 180

    return BlueBrickPose(x=bb_x, y=bb_y, angle=bb_angle)


def transform_point_to_bluebrick(x: float, y: float) -> tuple[float, float]:
    """Transform a single point from our coords to BlueBrick coords.

    Args:
        x: X coordinate in our system (studs).
        y: Y coordinate in our system (studs).

    Returns:
        (x, y) in BlueBrick coordinates.
    """
    return (x, -y)


def transform_angle_to_bluebrick(angle: float) -> float:
    """Transform an angle from our system to BlueBrick.

    Args:
        angle: Angle in degrees (CCW positive in our system).

    Returns:
        Angle in degrees (CW positive in BlueBrick).
    """
    return -angle
