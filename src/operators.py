"""Segment-aware genetic operators for partitioned chromosome encoding.

Crossover:
- PartitionedCrossover: One-point crossover on main loop, uniform swap on
  junction slots, uniform swap on start position.

Mutation:
- PartitionedMutation: Weighted sub-operator selection with separate mutation
  strategies for main loop genes (80%) and junction genes (20%).
"""

from __future__ import annotations

from typing import List

import numpy as np
from numpy.typing import NDArray
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation

from .encoding import (
    GENES_PER_JUNCTION,
    INACTIVE,
    MAIN_LOOP_PIECE_INDICES,
    MAX_MAIN_LOOP_PIECE,
    PartitionedDimensions,
)

# Valid main-loop piece types as a sorted array for fast random selection
_MAIN_LOOP_TYPES = np.array(sorted(MAIN_LOOP_PIECE_INDICES), dtype=np.int16)


# =============================================================================
# Crossover
# =============================================================================

class PartitionedCrossover(Crossover):
    """Segment-aware crossover respecting chromosome partition boundaries.

    - Main loop [0, n_main): one-point crossover (random cut, swap tails).
    - Junction slots [junc_start, junc_end): uniform per-slot swap (each
      4-gene junction descriptor taken from one parent at random).
    - Start position [start_pos_start, end): uniform from either parent.

    Args:
        dims: Chromosome partition dimensions.
        prob: Crossover probability per mating.
    """

    def __init__(self, dims: PartitionedDimensions, prob: float = 0.9) -> None:
        super().__init__(n_parents=2, n_offsprings=2, prob=prob)
        self.dims = dims

    def _do(self, problem, X, **kwargs) -> NDArray:
        # pymoo convention: X shape (n_parents, n_matings, n_var)
        _, n_matings, n_var = X.shape
        Y = np.empty((self.n_offsprings, n_matings, n_var), dtype=X.dtype)

        dims = self.dims

        for k in range(n_matings):
            p1 = X[0, k]
            p2 = X[1, k]
            c1 = p1.copy()
            c2 = p2.copy()

            # --- Main loop: one-point crossover ---
            if dims.n_main > 1:
                cut = np.random.randint(1, dims.n_main)
                c1[cut:dims.n_main] = p2[cut:dims.n_main]
                c2[cut:dims.n_main] = p1[cut:dims.n_main]

            # --- Junctions: uniform per-slot swap ---
            for j in range(dims.max_junctions):
                if np.random.random() < 0.5:
                    base = dims.junc_start + j * GENES_PER_JUNCTION
                    end = base + GENES_PER_JUNCTION
                    c1[base:end] = p2[base:end]
                    c2[base:end] = p1[base:end]

            # --- Start position: uniform from either parent ---
            if np.random.random() < 0.5:
                sp = dims.start_pos_start
                c1[sp:sp + 2] = p2[sp:sp + 2]
                c2[sp:sp + 2] = p1[sp:sp + 2]

            Y[0, k] = c1
            Y[1, k] = c2

        return Y


# =============================================================================
# Mutation Sub-operators (Main Loop)
# =============================================================================

def _mutate_piece_type(x: NDArray, dims: PartitionedDimensions) -> None:
    """Change a random active main-loop position to a different piece type."""
    active = np.where(x[:dims.n_main] != INACTIVE)[0]
    if len(active) == 0:
        return
    pos = active[np.random.randint(len(active))]
    old = x[pos]
    choices = _MAIN_LOOP_TYPES[_MAIN_LOOP_TYPES != old]
    if len(choices) == 0:
        return
    x[pos] = choices[np.random.randint(len(choices))]


def _activate_position(x: NDArray, dims: PartitionedDimensions) -> None:
    """Set a random inactive main-loop position to a random piece type."""
    inactive = np.where(x[:dims.n_main] == INACTIVE)[0]
    if len(inactive) == 0:
        return
    pos = inactive[np.random.randint(len(inactive))]
    x[pos] = _MAIN_LOOP_TYPES[np.random.randint(len(_MAIN_LOOP_TYPES))]


def _deactivate_position(x: NDArray, dims: PartitionedDimensions) -> None:
    """Set a random active main-loop position to INACTIVE."""
    active = np.where(x[:dims.n_main] != INACTIVE)[0]
    if len(active) <= 4:  # keep minimum for closure
        return
    pos = active[np.random.randint(len(active))]
    x[pos] = INACTIVE


def _swap_positions(x: NDArray, dims: PartitionedDimensions) -> None:
    """Swap two random main-loop positions (active or inactive)."""
    if dims.n_main < 2:
        return
    i, j = np.random.choice(dims.n_main, size=2, replace=False)
    x[i], x[j] = x[j], x[i]


# =============================================================================
# Mutation Sub-operators (Junction)
# =============================================================================

def _toggle_active(x: NDArray, dims: PartitionedDimensions) -> None:
    """Flip the active flag on a random junction slot."""
    if dims.max_junctions == 0:
        return
    slot = np.random.randint(dims.max_junctions)
    base = dims.junc_start + slot * GENES_PER_JUNCTION
    x[base] = 1 - x[base]  # 0 -> 1, 1 -> 0


def _reposition_junction(x: NDArray, dims: PartitionedDimensions) -> None:
    """Shift position of a random junction by a small delta."""
    if dims.max_junctions == 0:
        return
    slot = np.random.randint(dims.max_junctions)
    base = dims.junc_start + slot * GENES_PER_JUNCTION
    delta = np.random.randint(-5, 6)  # -5 to +5, excluding 0 is unlikely but fine
    new_pos = int(np.clip(x[base + 1] + delta, 0, dims.n_main - 1))
    x[base + 1] = new_pos


def _change_handedness(x: NDArray, dims: PartitionedDimensions) -> None:
    """Set handedness of a random junction to a valid template index."""
    if dims.max_junctions == 0:
        return
    from .templates import TEMPLATES
    slot = np.random.randint(dims.max_junctions)
    base = dims.junc_start + slot * GENES_PER_JUNCTION
    x[base + 2] = np.random.randint(0, len(TEMPLATES))


def _adjust_straights(x: NDArray, dims: PartitionedDimensions) -> None:
    """Adjust n_straights of a random junction by +-1..3."""
    if dims.max_junctions == 0:
        return
    slot = np.random.randint(dims.max_junctions)
    base = dims.junc_start + slot * GENES_PER_JUNCTION
    delta = np.random.choice([-3, -2, -1, 1, 2, 3])
    new_val = int(np.clip(x[base + 3] + delta, 0, dims.total_straights))
    x[base + 3] = new_val


# =============================================================================
# Combined Mutation
# =============================================================================

# Main loop sub-operators with equal weights
_MAIN_LOOP_OPS = [_mutate_piece_type, _activate_position, _deactivate_position, _swap_positions]
_MAIN_LOOP_WEIGHTS = np.array([0.30, 0.25, 0.20, 0.25])
_MAIN_LOOP_WEIGHTS /= _MAIN_LOOP_WEIGHTS.sum()

# Junction sub-operators with equal weights
_JUNCTION_OPS = [_toggle_active, _reposition_junction, _change_handedness, _adjust_straights]
_JUNCTION_WEIGHTS = np.array([0.25, 0.25, 0.25, 0.25])
_JUNCTION_WEIGHTS /= _JUNCTION_WEIGHTS.sum()


class PartitionedMutation(Mutation):
    """Segment-aware mutation with weighted sub-operator selection.

    Each individual that passes the probability gate receives exactly one
    mutation drawn from two categories:

    - **Main loop mutations** (80% when junctions exist, 100% otherwise):
      piece_type_change, activate_position, deactivate_position, swap_positions.
    - **Junction mutations** (20% when junctions exist):
      toggle_active, reposition, change_handedness, adjust_straights.

    Args:
        dims: Chromosome partition dimensions.
        prob: Per-individual mutation probability.
    """

    def __init__(self, dims: PartitionedDimensions, prob: float = 0.3) -> None:
        super().__init__(prob=prob)
        self.dims = dims

    def _do(self, problem, X, **kwargs) -> NDArray:
        has_junctions = self.dims.max_junctions > 0

        for i in range(len(X)):
            # Select category: main loop vs junction
            if has_junctions and np.random.random() < 0.2:
                op_idx = np.random.choice(len(_JUNCTION_OPS), p=_JUNCTION_WEIGHTS)
                _JUNCTION_OPS[op_idx](X[i], self.dims)
            else:
                op_idx = np.random.choice(len(_MAIN_LOOP_OPS), p=_MAIN_LOOP_WEIGHTS)
                _MAIN_LOOP_OPS[op_idx](X[i], self.dims)

        return X


# =============================================================================
# Convenience: NoOpCrossover (kept for backward compatibility)
# =============================================================================

class NoOpCrossover(Crossover):
    """Identity crossover -- returns parents unchanged."""

    def __init__(self, **kwargs) -> None:
        super().__init__(n_parents=2, n_offsprings=2, **kwargs)

    def _do(self, problem, X, **kwargs) -> NDArray:
        return X
