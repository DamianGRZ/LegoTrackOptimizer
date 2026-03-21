"""Template-based passing siding definitions for guaranteed geometric closure.

Standard LEGO passing siding pattern:
    [IN_switch] -> [approach_curve] -> [straights×N] -> [return_curve] -> [OUT_switch]

The template approach guarantees that branch paths geometrically connect
the IN switch diverge exit to the OUT switch merge entry, eliminating
the ~0% feasibility rate of random branch piece placement.

Key insight from Monty's Trains (Track Planning Part 1):
    "a curved track per switch allows diverging route to align parallel to main route"
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray


# =============================================================================
# TEMPLATE DATA STRUCTURES
# =============================================================================


@dataclass(frozen=True)
class PassingSidingTemplate:
    """Defines geometry of a standard passing siding.

    A passing siding diverges from the main line, runs parallel for some
    distance, then merges back. The template specifies:
    - Which switch types (LEFT or RIGHT)
    - Which curves to use for alignment
    - The geometric parameters for closure computation

    Attributes:
        name: Human-readable name for the template.
        handedness: "LEFT" or "RIGHT" indicating diverge direction.
        in_switch_idx: Piece index for IN switch (diverges from main).
        out_switch_idx: Piece index for OUT switch (merges back to main).
        approach_curve_idx: Curve after diverge to become parallel.
        return_curve_idx: Curve before merge to approach main line angle.
        straight_idx: Straight piece for parallel section.
        diverge_fk: (dx, dy, dtheta) for IN switch diverge route.
        merge_fk: (dx, dy, dtheta) for OUT switch merge route.
    """

    name: str
    handedness: str
    in_switch_idx: int
    out_switch_idx: int
    approach_curve_idx: int
    return_curve_idx: int
    straight_idx: int
    diverge_fk: Tuple[float, float, float]
    merge_fk: Tuple[float, float, float]


# =============================================================================
# STANDARD TEMPLATES (from track_pieces.yaml FK values)
# =============================================================================

# Piece indices from track_pieces.yaml piece_index mapping
STRAIGHT_16 = 0
R40_LEFT = 2
R40_RIGHT = 3
R40_SWITCH_LEFT_IN = 5
R40_SWITCH_LEFT_OUT = 6
R40_SWITCH_RIGHT_IN = 7
R40_SWITCH_RIGHT_OUT = 8

# FK values from track_pieces.yaml (32-stud switches)
# LEFT_IN diverge: (31.0, 6.2, 22.5)  - turns left 22.5°
# LEFT_OUT merge:  (31.0, -6.2, -22.5) - turns right 22.5° to rejoin
# RIGHT_IN diverge: (31.0, -6.2, -22.5) - turns right 22.5°
# RIGHT_OUT merge:  (31.0, 6.2, 22.5)  - turns left 22.5° to rejoin

LEFT_SIDING = PassingSidingTemplate(
    name="left_passing_siding",
    handedness="LEFT",
    in_switch_idx=R40_SWITCH_LEFT_IN,
    out_switch_idx=R40_SWITCH_LEFT_OUT,
    # After LEFT_IN diverge (heading +22.5°), use R40_RIGHT to go parallel
    approach_curve_idx=R40_RIGHT,
    # Before LEFT_OUT merge, use R40_LEFT to approach at +22.5°
    return_curve_idx=R40_LEFT,
    straight_idx=STRAIGHT_16,
    diverge_fk=(31.0, 6.2, 22.5),
    merge_fk=(31.0, -6.2, -22.5),
)

RIGHT_SIDING = PassingSidingTemplate(
    name="right_passing_siding",
    handedness="RIGHT",
    in_switch_idx=R40_SWITCH_RIGHT_IN,
    out_switch_idx=R40_SWITCH_RIGHT_OUT,
    # After RIGHT_IN diverge (heading -22.5°), use R40_LEFT to go parallel
    approach_curve_idx=R40_LEFT,
    # Before RIGHT_OUT merge, use R40_RIGHT to approach at -22.5°
    return_curve_idx=R40_RIGHT,
    straight_idx=STRAIGHT_16,
    diverge_fk=(31.0, -6.2, -22.5),
    merge_fk=(31.0, 6.2, 22.5),
)

# Reverse templates: OUT acts as entry (diverge), IN acts as exit (merge).
# Enables parallel tracks where the siding is traversed in the opposite direction.
# The branch curves are mirrored (approach/return swap).

LEFT_SIDING_REVERSE = PassingSidingTemplate(
    name="left_passing_siding_reverse",
    handedness="LEFT_REVERSE",
    in_switch_idx=R40_SWITCH_LEFT_OUT,   # OUT acts as entry (diverge)
    out_switch_idx=R40_SWITCH_LEFT_IN,   # IN acts as exit (merge)
    approach_curve_idx=R40_LEFT,         # Mirrored: LEFT to go parallel
    return_curve_idx=R40_RIGHT,          # Mirrored: RIGHT to approach merge
    straight_idx=STRAIGHT_16,
    diverge_fk=(31.0, -6.2, -22.5),     # OUT's merge FK used as diverge
    merge_fk=(31.0, 6.2, 22.5),         # IN's diverge FK used as merge
)

RIGHT_SIDING_REVERSE = PassingSidingTemplate(
    name="right_passing_siding_reverse",
    handedness="RIGHT_REVERSE",
    in_switch_idx=R40_SWITCH_RIGHT_OUT,  # OUT acts as entry (diverge)
    out_switch_idx=R40_SWITCH_RIGHT_IN,  # IN acts as exit (merge)
    approach_curve_idx=R40_RIGHT,        # Mirrored: RIGHT to go parallel
    return_curve_idx=R40_LEFT,           # Mirrored: LEFT to approach merge
    straight_idx=STRAIGHT_16,
    diverge_fk=(31.0, 6.2, 22.5),       # OUT's merge FK used as diverge
    merge_fk=(31.0, -6.2, -22.5),       # IN's diverge FK used as merge
)

# Template lookup by handedness index (0=LEFT, 1=RIGHT, 2=LEFT_REVERSE, 3=RIGHT_REVERSE)
TEMPLATES = {
    0: LEFT_SIDING,
    1: RIGHT_SIDING,
    2: LEFT_SIDING_REVERSE,
    3: RIGHT_SIDING_REVERSE,
}


# =============================================================================
# BRANCH PIECE COMPUTATION
# =============================================================================


def compute_branch_pieces(template: PassingSidingTemplate, n_straights: int) -> List[int]:
    """Generate piece sequence for a passing siding branch.

    The branch structure is:
        [approach_curve] + [straight × N] + [return_curve]

    This creates a path that:
    1. Turns to become parallel to main line (approach_curve)
    2. Runs parallel for N straights
    3. Turns back toward main line angle (return_curve)

    Args:
        template: Passing siding template defining piece types.
        n_straights: Number of straight pieces in parallel section.

    Returns:
        List of piece indices for the branch (excluding switches).
    """
    return (
        [template.approach_curve_idx]
        + [template.straight_idx] * n_straights
        + [template.return_curve_idx]
    )


def compute_branch_piece_count(n_straights: int) -> int:
    """Total pieces in branch (approach + straights + return)."""
    return 2 + n_straights


# =============================================================================
# GEOMETRY COMPUTATION
# =============================================================================

# R40 curve FK values (from track_pieces.yaml)
R40_LEFT_FK = (15.307, 3.045, 22.5)
R40_RIGHT_FK = (15.307, -3.045, -22.5)
STRAIGHT_16_FK = (16.0, 0.0, 0.0)

# FK lookup by piece index
PIECE_FK = {
    STRAIGHT_16: STRAIGHT_16_FK,
    R40_LEFT: R40_LEFT_FK,
    R40_RIGHT: R40_RIGHT_FK,
    R40_SWITCH_LEFT_IN: (32.0, 0.0, 0.0),  # Default (straight-through, 32-stud switch)
    R40_SWITCH_LEFT_OUT: (32.0, 0.0, 0.0),
    R40_SWITCH_RIGHT_IN: (32.0, 0.0, 0.0),
    R40_SWITCH_RIGHT_OUT: (32.0, 0.0, 0.0),
}


def apply_fk(state: Tuple[float, float, float], fk: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Apply FK delta to a state (x, y, theta).

    FK transformation:
        x' = x + dx * cos(theta) - dy * sin(theta)
        y' = y + dx * sin(theta) + dy * cos(theta)
        theta' = theta + dtheta

    Args:
        state: Current pose (x, y, theta) in degrees.
        fk: FK delta (dx, dy, dtheta).

    Returns:
        New pose (x', y', theta').
    """
    x, y, theta = state
    dx, dy, dtheta = fk

    theta_rad = np.radians(theta)
    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)

    x_new = x + dx * cos_t - dy * sin_t
    y_new = y + dx * sin_t + dy * cos_t
    theta_new = theta + dtheta

    return (float(x_new), float(y_new), float(theta_new))


def compute_branch_endpoint(
    in_switch_state: Tuple[float, float, float],
    template: PassingSidingTemplate,
    n_straights: int,
) -> Tuple[float, float, float]:
    """Compute where the branch ends (before merge).

    Traces FK through:
    1. IN switch diverge
    2. Approach curve
    3. N straights
    4. Return curve

    Args:
        in_switch_state: Pose at IN switch entry (x, y, theta).
        template: Passing siding template.
        n_straights: Number of straights in parallel section.

    Returns:
        Branch endpoint pose (x, y, theta) ready for OUT switch merge.
    """
    state = in_switch_state

    # 1. IN switch diverge
    state = apply_fk(state, template.diverge_fk)

    # 2. Approach curve
    approach_fk = PIECE_FK[template.approach_curve_idx]
    state = apply_fk(state, approach_fk)

    # 3. N straights
    straight_fk = PIECE_FK[template.straight_idx]
    for _ in range(n_straights):
        state = apply_fk(state, straight_fk)

    # 4. Return curve
    return_fk = PIECE_FK[template.return_curve_idx]
    state = apply_fk(state, return_fk)

    return state


def compute_required_main_distance(
    template: PassingSidingTemplate,
    n_straights: int,
) -> float:
    """Compute main loop X-distance the siding spans.

    This is used to find where the OUT switch should be placed.
    The branch runs parallel to the main line, so the X-distance
    is approximately the sum of X-components.

    Args:
        template: Passing siding template.
        n_straights: Number of straights in parallel section.

    Returns:
        Approximate X-distance along main loop axis (studs).
    """
    # Start at origin, compute branch endpoint
    start_state = (0.0, 0.0, 0.0)
    end_state = compute_branch_endpoint(start_state, template, n_straights)

    # X-distance is the total forward travel
    # For parallel siding, this equals the main loop distance needed
    total_x = end_state[0]

    return total_x


def compute_out_switch_alignment_error(
    in_switch_state: Tuple[float, float, float],
    out_switch_state: Tuple[float, float, float],
    template: PassingSidingTemplate,
    n_straights: int,
) -> Tuple[float, float]:
    """Compute error between branch endpoint and OUT switch merge entry.

    For a valid passing siding, the branch must arrive at the OUT switch's
    merge entry port (port 2). This function computes the position and
    angle error.

    Args:
        in_switch_state: Pose at IN switch entry.
        out_switch_state: Pose at OUT switch entry.
        template: Passing siding template.
        n_straights: Number of straights in parallel section.

    Returns:
        (position_error, angle_error) tuple in (studs, degrees).
    """
    # Compute where branch actually ends
    branch_end = compute_branch_endpoint(in_switch_state, template, n_straights)

    # Expected: OUT switch merge entry should receive the branch
    # The merge FK (15.3073, ±3.0448, ±22.5) is applied FROM port 2 TO port 1
    # So port 2 is at (16 - 15.3073, ±3.0448) = (0.6927, ±3.0448) in switch local frame

    # Transform OUT switch port 2 to world frame
    out_x, out_y, out_theta = out_switch_state

    # Port 2 position relative to switch entry (from track_pieces.yaml, 32-stud switches)
    # LEFT_OUT: port 2 at (1.0, 6.2) heading 22.5°
    # RIGHT_OUT: port 2 at (1.0, -6.2) heading -22.5°
    if template.handedness == "LEFT":
        port2_local = (1.0, 6.2, 22.5)
    else:
        port2_local = (1.0, -6.2, -22.5)

    # Transform to world coordinates
    port2_x = out_x + port2_local[0] * np.cos(np.radians(out_theta)) - port2_local[1] * np.sin(np.radians(out_theta))
    port2_y = out_y + port2_local[0] * np.sin(np.radians(out_theta)) + port2_local[1] * np.cos(np.radians(out_theta))
    port2_theta = out_theta + port2_local[2]

    # Compute errors
    position_error = np.sqrt((branch_end[0] - port2_x) ** 2 + (branch_end[1] - port2_y) ** 2)

    # Angle error (normalize to [-180, 180])
    angle_diff = branch_end[2] - port2_theta
    while angle_diff > 180:
        angle_diff -= 360
    while angle_diff < -180:
        angle_diff += 360
    angle_error = abs(angle_diff)

    return (float(position_error), float(angle_error))


def is_valid_siding(
    in_switch_state: Tuple[float, float, float],
    out_switch_state: Tuple[float, float, float],
    template: PassingSidingTemplate,
    n_straights: int,
    position_tolerance: float = 2.0,
    angle_tolerance: float = 5.0,
) -> bool:
    """Check if a passing siding configuration is geometrically valid.

    Args:
        in_switch_state: Pose at IN switch entry.
        out_switch_state: Pose at OUT switch entry.
        template: Passing siding template.
        n_straights: Number of straights in parallel section.
        position_tolerance: Max position error (studs).
        angle_tolerance: Max angle error (degrees).

    Returns:
        True if siding geometry closes within tolerances.
    """
    pos_error, angle_error = compute_out_switch_alignment_error(
        in_switch_state, out_switch_state, template, n_straights
    )
    return pos_error <= position_tolerance and angle_error <= angle_tolerance


# =============================================================================
# INVENTORY HELPERS
# =============================================================================


def get_siding_inventory_requirements(
    template: PassingSidingTemplate,
    n_straights: int,
) -> dict[int, int]:
    """Get piece counts needed for a passing siding.

    Args:
        template: Passing siding template.
        n_straights: Number of straights in parallel section.

    Returns:
        Dict mapping piece_index -> required count.
    """
    requirements: dict[int, int] = {}

    # Switches
    requirements[template.in_switch_idx] = requirements.get(template.in_switch_idx, 0) + 1
    requirements[template.out_switch_idx] = requirements.get(template.out_switch_idx, 0) + 1

    # Curves
    requirements[template.approach_curve_idx] = requirements.get(template.approach_curve_idx, 0) + 1
    requirements[template.return_curve_idx] = requirements.get(template.return_curve_idx, 0) + 1

    # Straights
    if n_straights > 0:
        requirements[template.straight_idx] = requirements.get(template.straight_idx, 0) + n_straights

    return requirements


def check_siding_inventory(
    template: PassingSidingTemplate,
    n_straights: int,
    available_inventory: dict[int, int],
    used_inventory: dict[int, int],
) -> bool:
    """Check if inventory has pieces for a passing siding.

    Args:
        template: Passing siding template.
        n_straights: Number of straights in parallel section.
        available_inventory: Available pieces by index.
        used_inventory: Already used pieces by index.

    Returns:
        True if sufficient inventory remains.
    """
    requirements = get_siding_inventory_requirements(template, n_straights)

    for piece_idx, needed in requirements.items():
        available = available_inventory.get(piece_idx, 0)
        used = used_inventory.get(piece_idx, 0)
        if used + needed > available:
            return False

    return True
