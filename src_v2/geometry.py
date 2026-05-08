"""Track layout geometry using forward kinematics (FK) and multi-phase topology."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .catalog import TrackCatalog


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
    """Track-layout representation backing the matplotlib renderer.

    The trailing entry of ``states`` is the "closing duplicate" — a copy of
    the first state, used by curve drawing to figure out the rotation
    direction of the *last* piece. So ``len(states) == n_pieces + 1`` for
    a single-component layout.

    Multi-component layouts (Phase 5+: GA finds disconnected closed loops
    or open chains): set ``component_breaks`` to the piece indices where
    each component starts (e.g. ``[0, 12, 28]`` for three components of
    sizes 12, 16, ...). Each component contributes its own closing
    duplicate inside ``states``, so ``len(states) == n_pieces + n_components``.
    The renderer iterates per-component to keep curve directions correct
    across component boundaries.
    """

    indices: NDArray[np.int32]
    states: NDArray[np.float64]
    component_breaks: Optional[NDArray[np.int32]] = None
    # Per-piece flip bit (only meaningful for symmetric pieces; e.g.
    # R40_CURVE uses it to pick left vs right arc). Set by the
    # port-graph→layout walk so the renderer can ignore the
    # next-state-theta heuristic — that heuristic gives the wrong
    # sign whenever the walk traverses a slot from port B to port A
    # (e.g. when DFS pops back into the cycle from the alternative
    # neighbour at the start slot), producing a 1-curve-per-corner
    # mis-render.
    flips: Optional[NDArray[np.int8]] = None

    @property
    def n_pieces(self) -> int:
        """Number of pieces in layout."""
        return len(self.indices)

    @property
    def n_components(self) -> int:
        """Number of disjoint pieces-walks packed into this Layout."""
        if self.component_breaks is None:
            return 1 if self.n_pieces > 0 else 0
        return len(self.component_breaks)

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


