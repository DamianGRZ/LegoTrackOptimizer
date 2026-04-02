"""Genetic operators for CGP-inspired integer chromosome encoding.

Mutation operators:
- PieceTypeMutation: Change piece type at a random node
- ConnectionMutation: Rewire port2/port3 connections (CGP-style)
- NodeInsertionMutation: Activate an inactive node
- NodeDeletionMutation: Deactivate an active node
- TrackMutation: Combined operator selecting from above with probabilities

Crossover operators:
- UniformNodeCrossover: Per-node uniform crossover (preserves node tuple integrity)
- OnePointCrossover: Standard one-point crossover on flat array
- NoOpCrossover: Returns parents unchanged
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation

from .encoding import (
    CROSSING_INDICES,
    FOUR_PORT_PIECES,
    GENES_PER_NODE,
    INACTIVE,
    IN_SWITCH_INDICES,
    OUT_SWITCH_INDICES,
    SIMPLE_PIECE_INDICES,
    SWITCH_INDICES,
    THREE_PORT_PIECES,
    ChromosomeDimensions,
    get_all_piece_types,
    get_node,
    get_piece_type,
    get_port2_conn,
    set_node,
    set_piece_type,
    set_port2_conn,
    set_port3_conn,
)


# =============================================================================
# Mutation Operators
# =============================================================================

class TrackMutation(Mutation):
    """Combined mutation operator for integer CGP chromosomes.

    Selects from sub-operators with configurable probabilities:
    - piece_type: Change piece type at random active node (0.35)
    - connection: Rewire a port2/port3 connection (0.15)
    - insert: Activate an inactive node with random piece (0.20)
    - delete: Deactivate a random active node (0.10)
    - swap: Swap two active nodes' piece types (0.10)
    - shift: Move a piece from one position to another (0.10)
    """

    def __init__(self, dims: ChromosomeDimensions,
                 max_piece_index: int = 9,
                 prob: float = 0.3,
                 **kwargs):
        super().__init__(prob=prob, **kwargs)
        self.dims = dims
        self.max_piece_index = max_piece_index

        self.op_probs = np.array([0.35, 0.15, 0.20, 0.10, 0.10, 0.10])
        self.op_probs /= self.op_probs.sum()

    def _do(self, problem, X, **kwargs):
        for i in range(len(X)):
            op = np.random.choice(6, p=self.op_probs)
            if op == 0:
                self._mutate_piece_type(X[i])
            elif op == 1:
                self._mutate_connection(X[i])
            elif op == 2:
                self._insert_node(X[i])
            elif op == 3:
                self._delete_node(X[i])
            elif op == 4:
                self._swap_nodes(X[i])
            else:
                self._shift_node(X[i])
        return X

    def _mutate_piece_type(self, x: NDArray) -> None:
        """Change piece type at a random active node."""
        active = self._get_active_positions(x)
        if not active:
            return

        pos = active[np.random.randint(len(active))]
        old_type = get_piece_type(x, pos)

        # Pick a different piece type
        new_type = np.random.randint(-1, self.max_piece_index + 1)
        attempts = 0
        while new_type == old_type and attempts < 10:
            new_type = np.random.randint(-1, self.max_piece_index + 1)
            attempts += 1

        set_piece_type(x, pos, new_type)

        # Clear connection genes if new type doesn't support them
        if new_type == INACTIVE or (new_type not in THREE_PORT_PIECES and new_type not in FOUR_PORT_PIECES):
            set_port2_conn(x, pos, INACTIVE)
            set_port3_conn(x, pos, INACTIVE)

    def _mutate_connection(self, x: NDArray) -> None:
        """Rewire a port2 or port3 connection (CGP-style connection mutation)."""
        # Find nodes with active connections or that could have connections
        candidates = []
        for i in range(self.dims.n_nodes):
            pt = get_piece_type(x, i)
            if pt in THREE_PORT_PIECES or pt in FOUR_PORT_PIECES:
                candidates.append(i)

        if not candidates:
            return

        pos = candidates[np.random.randint(len(candidates))]
        pt = get_piece_type(x, pos)

        # Mutate port2
        new_target = np.random.randint(-1, self.dims.n_nodes)
        set_port2_conn(x, pos, new_target)

        # Also mutate port3 for 4-port pieces
        if pt in FOUR_PORT_PIECES:
            new_target3 = np.random.randint(-1, self.dims.n_nodes)
            set_port3_conn(x, pos, new_target3)

    def _insert_node(self, x: NDArray) -> None:
        """Activate an inactive node with a random piece."""
        inactive = self._get_inactive_positions(x)
        if not inactive:
            return

        pos = inactive[np.random.randint(len(inactive))]
        new_type = np.random.randint(0, self.max_piece_index + 1)
        set_node(x, pos, new_type)

    def _delete_node(self, x: NDArray) -> None:
        """Deactivate a random active node."""
        active = self._get_active_positions(x)
        if len(active) <= 4:  # Keep minimum for closure
            return

        pos = active[np.random.randint(len(active))]
        set_node(x, pos, INACTIVE)

    def _swap_nodes(self, x: NDArray) -> None:
        """Swap piece types between two random active nodes."""
        active = self._get_active_positions(x)
        if len(active) < 2:
            return

        idx = np.random.choice(len(active), size=2, replace=False)
        p1, p2 = active[idx[0]], active[idx[1]]

        t1, c1_p2, c1_p3 = get_node(x, p1)
        t2, c2_p2, c2_p3 = get_node(x, p2)
        set_node(x, p1, t2, c2_p2, c2_p3)
        set_node(x, p2, t1, c1_p2, c1_p3)

    def _shift_node(self, x: NDArray) -> None:
        """Move a piece from one active position to an inactive position."""
        active = self._get_active_positions(x)
        inactive = self._get_inactive_positions(x)
        if not active or not inactive:
            return

        src = active[np.random.randint(len(active))]
        dst = inactive[np.random.randint(len(inactive))]

        t, p2, p3 = get_node(x, src)
        set_node(x, dst, t, p2, p3)
        set_node(x, src, INACTIVE)

    def _get_active_positions(self, x: NDArray) -> List[int]:
        return [i for i in range(self.dims.n_nodes) if get_piece_type(x, i) != INACTIVE]

    def _get_inactive_positions(self, x: NDArray) -> List[int]:
        return [i for i in range(self.dims.n_nodes) if get_piece_type(x, i) == INACTIVE]


# =============================================================================
# Crossover Operators
# =============================================================================

class NoOpCrossover(Crossover):
    """Identity crossover — returns parents unchanged."""

    def __init__(self, **kwargs):
        super().__init__(n_parents=2, n_offsprings=2, **kwargs)

    def _do(self, problem, X, **kwargs):
        return X


class UniformNodeCrossover(Crossover):
    """Per-node uniform crossover preserving node tuple integrity.

    For each node position, the offspring inherits the complete
    (type, port2, port3) tuple from one parent with probability 0.5.
    """

    def __init__(self, dims: ChromosomeDimensions, prob: float = 0.9, **kwargs):
        super().__init__(n_parents=2, n_offsprings=2, prob=prob, **kwargs)
        self.dims = dims

    def _do(self, problem, X, **kwargs):
        # pymoo crossover contract:
        # Input X: (n_parents, n_matings, n_var)  [axes swapped by pymoo]
        # Output:  (n_offsprings, n_matings, n_var)
        n_parents, n_matings, n_var = X.shape
        Y = np.full((self.n_offsprings, n_matings, n_var), INACTIVE, dtype=X.dtype)

        for k in range(n_matings):
            p1 = X[0, k]
            p2 = X[1, k]

            c1 = p1.copy()
            c2 = p2.copy()

            for i in range(self.dims.n_nodes):
                if np.random.random() < 0.5:
                    base = i * GENES_PER_NODE
                    end = base + GENES_PER_NODE
                    c1[base:end], c2[base:end] = p2[base:end].copy(), p1[base:end].copy()

            Y[0, k] = c1
            Y[1, k] = c2

        return Y


class OnePointNodeCrossover(Crossover):
    """One-point crossover at node boundaries."""

    def __init__(self, dims: ChromosomeDimensions, prob: float = 0.9, **kwargs):
        super().__init__(n_parents=2, n_offsprings=2, prob=prob, **kwargs)
        self.dims = dims

    def _do(self, problem, X, **kwargs):
        n_parents, n_matings, n_var = X.shape
        Y = np.full((self.n_offsprings, n_matings, n_var), INACTIVE, dtype=X.dtype)

        for k in range(n_matings):
            p1 = X[0, k]
            p2 = X[1, k]

            cut_node = np.random.randint(1, self.dims.n_nodes)
            cut_gene = cut_node * GENES_PER_NODE

            Y[0, k] = np.concatenate([p1[:cut_gene], p2[cut_gene:]])
            Y[1, k] = np.concatenate([p2[:cut_gene], p1[cut_gene:]])

        return Y


# Backward-compatible aliases
SwitchPreservingCrossover = UniformNodeCrossover
SegmentSelectiveCrossover = UniformNodeCrossover
