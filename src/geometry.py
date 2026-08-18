"""Track layout geometry: vectorized forward kinematics (FK), closure metrics,
and the single-loop ``Layout`` view consumed by the train physics and renderer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .catalog import TrackCatalog


def compute_fk_chain(fk_deltas: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute cumulative forward kinematics chain from piece deltas.

    Vectorized: the heading entering piece i depends only on the cumulative
    sum of preceding dtheta values, so all headings are computed in one
    ``cumsum``, the local->world rotation in one trig batch, and the positions
    in two more cumsums. Matches the sequential loop bit-for-bit (cumsum
    accumulates in the same order).

    Args:
        fk_deltas: (n, 3) array of [dx, dy, dtheta] for each piece.

    Returns:
        (n+1, 3) array of cumulative states [x, y, theta].
        State 0 is origin [0, 0, 0], state i+1 is after piece i.
    """
    fk_deltas = np.asarray(fk_deltas, dtype=np.float64)
    n = len(fk_deltas)
    states = np.zeros((n + 1, 3), dtype=np.float64)
    if n == 0:
        return states

    theta = np.cumsum(fk_deltas[:, 2])
    theta_entering = np.radians(np.concatenate(([0.0], theta[:-1])))
    cos_t = np.cos(theta_entering)
    sin_t = np.sin(theta_entering)

    # Local displacement rotated into world coordinates, then accumulated.
    world_dx = fk_deltas[:, 0] * cos_t - fk_deltas[:, 1] * sin_t
    world_dy = fk_deltas[:, 0] * sin_t + fk_deltas[:, 1] * cos_t

    states[1:, 0] = np.cumsum(world_dx)
    states[1:, 1] = np.cumsum(world_dy)
    states[1:, 2] = theta
    return states


@dataclass
class Layout:
    """Single-loop layout view (one piece sequence + FK states).

    A simpler representation than MultiPathLayout, consumed by the train
    physics (per-route views) and the switchless renderer path.
    """

    indices: NDArray[np.int32]
    states: NDArray[np.float64]
    # Catalog route index per piece (parallel to indices); when present the speed
    # profiler scores each segment on its traversed route instead of the default.
    route_indices: Optional[NDArray[np.int32]] = None

    @property
    def n_pieces(self) -> int:
        """Number of pieces in layout."""
        return len(self.indices)

    @property
    def n_physical_pieces(self) -> int:
        """Physical pieces. A single-loop view holds no descriptor records, so
        it equals the slot count; defined so both renderer paths share a name."""
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
        # Distance to the nearest multiple of 360
        remainder = total % 360
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
