"""Self-intersection detection for track layouts.

Finds where track passes near itself at compatible angles for switch connections.
This enables topology to emerge naturally - sidings, figure-8s, loop-to-loop.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .encoding import (
    SWITCH_LEFT_IN,
    SWITCH_LEFT_OUT,
    SWITCH_RIGHT_IN,
    SWITCH_RIGHT_OUT,
)


# Switch divergence angle (R40 switches)
SWITCH_DIVERGE_ANGLE = 22.5  # degrees


@dataclass
class SwitchOpportunity:
    """A potential switch connection point where track passes near itself."""

    # Position in piece sequence where IN switch would go
    in_position: int

    # Position where OUT switch would go (where track returns)
    out_position: int

    # Handedness: 'left' or 'right'
    handedness: str

    # Quality metrics
    position_error: float  # Distance between switch ports (studs)
    heading_error: float   # Angular mismatch (degrees)

    # Switch piece indices to use
    in_switch_idx: int
    out_switch_idx: int

    @property
    def quality_score(self) -> float:
        """Lower is better - combined position and heading error."""
        return self.position_error + self.heading_error * 0.1

    @property
    def is_valid(self) -> bool:
        """Check if this opportunity meets connection tolerances."""
        return self.position_error < 4.0 and self.heading_error < 10.0


@dataclass
class IntersectionResult:
    """Result of self-intersection analysis for a track layout."""

    # All potential switch opportunities found
    opportunities: List[SwitchOpportunity]

    # Number of switches in the piece sequence
    switch_count: int

    # Switches that have valid connections
    connected_switches: List[Tuple[int, int]]  # (in_pos, out_pos) pairs

    # Switches with loose port 2 (no valid connection)
    loose_port_count: int

    @property
    def has_valid_topology(self) -> bool:
        """True if all switches have valid connections."""
        return self.loose_port_count == 0


def find_self_intersections(
    states: NDArray[np.float64],
    piece_indices: NDArray[np.int32],
    position_tolerance: float = 8.0,
    angle_tolerance: float = 15.0,
    min_separation: int = 4,
) -> List[SwitchOpportunity]:
    """Find positions where track passes near itself at compatible angles.

    These are potential switch connection points where:
    - Two track positions are geometrically close
    - The heading difference matches switch diverge angle (±22.5°)

    Args:
        states: (n+1, 3) array of track states [x, y, theta].
        piece_indices: (n,) array of piece indices at each position.
        position_tolerance: Max distance between potential connection points (studs).
        angle_tolerance: Max heading error from ideal ±22.5° (degrees).
        min_separation: Minimum pieces between IN and OUT positions.

    Returns:
        List of SwitchOpportunity sorted by quality (best first).
    """
    n = len(piece_indices)
    if n < min_separation + 2:
        return []

    opportunities = []

    # Check all pairs of positions
    for i in range(n - min_separation):
        pos_i = states[i, :2]
        heading_i = states[i, 2]

        for j in range(i + min_separation, n):
            pos_j = states[j, :2]
            heading_j = states[j, 2]

            # Check position proximity
            distance = np.linalg.norm(pos_i - pos_j)
            if distance > position_tolerance:
                continue

            # Check heading compatibility for switch connection
            # For a switch pair: IN diverges at +22.5° (left) or -22.5° (right)
            # OUT merges from the same angle
            heading_diff = _normalize_angle(heading_j - heading_i)

            # Check for LEFT switch opportunity (diverge +22.5°)
            left_error = abs(_normalize_angle(heading_diff - SWITCH_DIVERGE_ANGLE))
            if left_error < angle_tolerance:
                opportunities.append(SwitchOpportunity(
                    in_position=i,
                    out_position=j,
                    handedness='left',
                    position_error=distance,
                    heading_error=left_error,
                    in_switch_idx=SWITCH_LEFT_IN,
                    out_switch_idx=SWITCH_LEFT_OUT,
                ))

            # Check for RIGHT switch opportunity (diverge -22.5°)
            right_error = abs(_normalize_angle(heading_diff + SWITCH_DIVERGE_ANGLE))
            if right_error < angle_tolerance:
                opportunities.append(SwitchOpportunity(
                    in_position=i,
                    out_position=j,
                    handedness='right',
                    position_error=distance,
                    heading_error=right_error,
                    in_switch_idx=SWITCH_RIGHT_IN,
                    out_switch_idx=SWITCH_RIGHT_OUT,
                ))

    # Sort by quality (best opportunities first)
    opportunities.sort(key=lambda x: x.quality_score)

    return opportunities


def analyze_switch_connections(
    states: NDArray[np.float64],
    piece_indices: NDArray[np.int32],
    position_tolerance: float = 4.0,
    angle_tolerance: float = 10.0,
) -> IntersectionResult:
    """Analyze switch connections in a track layout.

    Switches must be paired: each IN switch needs a matching OUT switch
    of the same handedness. This validates structural correctness.

    For actual geometric connection (passing siding), the IN and OUT
    are connected via a branch (curve-straights-curve), not directly.

    Args:
        states: (n+1, 3) array of track states [x, y, theta].
        piece_indices: (n,) array of piece indices.
        position_tolerance: Max connection distance (studs).
        angle_tolerance: Max heading error (degrees).

    Returns:
        IntersectionResult with connection analysis.
    """
    n = len(piece_indices)

    # Count switches by type
    left_in_count = sum(1 for idx in piece_indices if idx == SWITCH_LEFT_IN)
    left_out_count = sum(1 for idx in piece_indices if idx == SWITCH_LEFT_OUT)
    right_in_count = sum(1 for idx in piece_indices if idx == SWITCH_RIGHT_IN)
    right_out_count = sum(1 for idx in piece_indices if idx == SWITCH_RIGHT_OUT)

    switch_count = left_in_count + left_out_count + right_in_count + right_out_count

    # Find all potential self-intersection opportunities
    opportunities = find_self_intersections(
        states, piece_indices,
        position_tolerance=position_tolerance * 2,
        angle_tolerance=angle_tolerance * 1.5,
    )

    # Pair switches by handedness
    # Each IN needs a matching OUT of same handedness
    left_pairs = min(left_in_count, left_out_count)
    right_pairs = min(right_in_count, right_out_count)

    # Build connected pairs list (positional matching for tracking)
    connected_switches = []
    in_positions = []
    out_positions = []

    for i, idx in enumerate(piece_indices):
        if idx == SWITCH_LEFT_IN or idx == SWITCH_RIGHT_IN:
            in_positions.append((i, 'left' if idx == SWITCH_LEFT_IN else 'right'))
        elif idx == SWITCH_LEFT_OUT or idx == SWITCH_RIGHT_OUT:
            out_positions.append((i, 'left' if idx == SWITCH_LEFT_OUT else 'right'))

    # Match IN to OUT by handedness (in order of appearance)
    used_outs = set()
    for in_pos, in_hand in in_positions:
        for out_pos, out_hand in out_positions:
            if out_pos in used_outs:
                continue
            if in_hand == out_hand and out_pos > in_pos:  # OUT must come after IN
                connected_switches.append((in_pos, out_pos))
                used_outs.add(out_pos)
                break

    # Loose ports = unpaired switches
    # Left unpaired: |left_in - left_out|
    # Right unpaired: |right_in - right_out|
    loose_port_count = (
        abs(left_in_count - left_out_count) +
        abs(right_in_count - right_out_count)
    )

    return IntersectionResult(
        opportunities=opportunities,
        switch_count=switch_count,
        connected_switches=connected_switches,
        loose_port_count=loose_port_count,
    )


def compute_port2_pose(
    state: NDArray[np.float64],
    piece_idx: int,
) -> Optional[NDArray[np.float64]]:
    """Compute world pose of port 2 (divergent/merge port) for a switch.

    Args:
        state: [x, y, theta] state at the switch entry.
        piece_idx: Piece index of the switch.

    Returns:
        [x, y, theta] pose of port 2 in world frame, or None if not a switch.
    """
    if not _is_switch(piece_idx):
        return None

    if piece_idx == SWITCH_LEFT_IN:
        # Port 2 at (15.3073, 3.0448), heading +22.5° in local frame
        return _transform_local_to_world(state, 15.3073, 3.0448, 22.5)
    elif piece_idx == SWITCH_RIGHT_IN:
        # Port 2 at (15.3073, -3.0448), heading -22.5° in local frame
        return _transform_local_to_world(state, 15.3073, -3.0448, -22.5)
    elif piece_idx == SWITCH_LEFT_OUT:
        # Port 2 at (0.6927, 3.0448), heading +22.5° in local frame
        return _transform_local_to_world(state, 0.6927, 3.0448, 22.5)
    elif piece_idx == SWITCH_RIGHT_OUT:
        # Port 2 at (0.6927, -3.0448), heading -22.5° in local frame
        return _transform_local_to_world(state, 0.6927, -3.0448, -22.5)

    return None


# =============================================================================
# Helper Functions
# =============================================================================

def _normalize_angle(angle: float) -> float:
    """Normalize angle to [-180, 180] range."""
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle


def _is_switch(piece_idx: int) -> bool:
    """Check if piece is any type of switch."""
    return piece_idx in {SWITCH_LEFT_IN, SWITCH_LEFT_OUT,
                         SWITCH_RIGHT_IN, SWITCH_RIGHT_OUT}


def _is_in_switch(piece_idx: int) -> bool:
    """Check if piece is an IN switch (diverging)."""
    return piece_idx in {SWITCH_LEFT_IN, SWITCH_RIGHT_IN}


def _is_out_switch(piece_idx: int) -> bool:
    """Check if piece is an OUT switch (merging)."""
    return piece_idx in {SWITCH_LEFT_OUT, SWITCH_RIGHT_OUT}


def _transform_local_to_world(
    state: NDArray[np.float64],
    local_x: float,
    local_y: float,
    local_theta: float,
) -> NDArray[np.float64]:
    """Transform local coordinates to world frame.

    Args:
        state: [x, y, theta] in world frame.
        local_x, local_y, local_theta: Position and heading in local frame.

    Returns:
        [x, y, theta] in world frame.
    """
    theta_rad = np.radians(state[2])
    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)

    world_x = state[0] + local_x * cos_t - local_y * sin_t
    world_y = state[1] + local_x * sin_t + local_y * cos_t
    world_theta = state[2] + local_theta

    return np.array([world_x, world_y, world_theta])


def _get_divergent_port_pose(
    state: NDArray[np.float64],
    piece_idx: int,
) -> NDArray[np.float64]:
    """Get world pose of divergent port for IN switch."""
    if piece_idx == SWITCH_LEFT_IN:
        return _transform_local_to_world(state, 15.3073, 3.0448, 22.5)
    elif piece_idx == SWITCH_RIGHT_IN:
        return _transform_local_to_world(state, 15.3073, -3.0448, -22.5)
    return state.copy()


def _get_merge_port_pose(
    state: NDArray[np.float64],
    piece_idx: int,
) -> NDArray[np.float64]:
    """Get world pose of merge port for OUT switch."""
    if piece_idx == SWITCH_LEFT_OUT:
        return _transform_local_to_world(state, 0.6927, 3.0448, 22.5)
    elif piece_idx == SWITCH_RIGHT_OUT:
        return _transform_local_to_world(state, 0.6927, -3.0448, -22.5)
    return state.copy()
