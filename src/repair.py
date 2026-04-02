"""Repair operators for CGP-inspired integer chromosomes.

Operates directly on integer gene values — no decode/re-encode needed.

Operators:
- InventoryRepair: Remove pieces exceeding inventory limits
- SwitchPairingRepair: Ensure IN/OUT switches are properly paired
- ConnectionRepair: Fix dangling port2/port3 connections
- ClosureRepair: Add/remove curves to approach angular closure
- TrackRepairPipeline: Combines all repairs in sequence
"""

from typing import Dict, List, Optional, Set

import numpy as np
from numpy.typing import NDArray
from pymoo.core.repair import Repair

from .encoding import (
    CROSSING_INDICES,
    FOUR_PORT_PIECES,
    GENES_PER_NODE,
    INACTIVE,
    IN_SWITCH_INDICES,
    OUT_SWITCH_INDICES,
    STRAIGHT_16,
    SWITCH_INDICES,
    SWITCH_PAIRS,
    THREE_PORT_PIECES,
    ChromosomeDimensions,
    get_all_piece_types,
    get_all_port2_conns,
    get_node,
    get_piece_type,
    get_port2_conn,
    get_port3_conn,
    set_node,
    set_piece_type,
    set_port2_conn,
    set_port3_conn,
)


# =============================================================================
# Inventory Repair
# =============================================================================

class InventoryRepair(Repair):
    """Remove excess pieces that violate inventory limits."""

    def __init__(self, inventory_by_index: Dict[int, int],
                 dims: ChromosomeDimensions, **kwargs):
        super().__init__(**kwargs)
        self.inventory_by_index = inventory_by_index
        self.dims = dims

    def _do(self, problem, X, **kwargs):
        for i in range(len(X)):
            self._repair_chromosome(X[i])
        return X

    def _repair_chromosome(self, x: NDArray) -> None:
        """Remove excess pieces from end of sequence."""
        usage: Dict[int, int] = {}

        for i in range(self.dims.n_nodes):
            pt = get_piece_type(x, i)
            if pt == INACTIVE or pt < 0:
                continue

            usage[pt] = usage.get(pt, 0) + 1
            limit = self.inventory_by_index.get(pt, 0)

            if usage[pt] > limit:
                set_node(x, i, INACTIVE)
                usage[pt] -= 1


# =============================================================================
# Switch Pairing Repair
# =============================================================================

class SwitchPairingRepair(Repair):
    """Ensure IN/OUT switches are properly paired.

    For each switch pair type (LEFT_IN↔LEFT_OUT, RIGHT_IN↔RIGHT_OUT),
    count occurrences. If unbalanced, replace excess with STRAIGHT_16.
    """

    def __init__(self, dims: ChromosomeDimensions, **kwargs):
        super().__init__(**kwargs)
        self.dims = dims

    def _do(self, problem, X, **kwargs):
        for i in range(len(X)):
            self._repair_chromosome(X[i])
        return X

    def _repair_chromosome(self, x: NDArray) -> None:
        for in_idx, out_idx in SWITCH_PAIRS:
            in_positions = []
            out_positions = []

            for i in range(self.dims.n_nodes):
                pt = get_piece_type(x, i)
                if pt == in_idx:
                    in_positions.append(i)
                elif pt == out_idx:
                    out_positions.append(i)

            # Remove excess from whichever has more
            while len(in_positions) > len(out_positions) and in_positions:
                pos = in_positions.pop()
                set_node(x, pos, STRAIGHT_16)

            while len(out_positions) > len(in_positions) and out_positions:
                pos = out_positions.pop()
                set_node(x, pos, STRAIGHT_16)


# =============================================================================
# Connection Repair
# =============================================================================

class ConnectionRepair(Repair):
    """Fix dangling or invalid port connections.

    - 2-port pieces should have port2=-1, port3=-1
    - 3-port pieces should have port3=-1
    - Connection targets must reference active nodes within bounds
    """

    def __init__(self, dims: ChromosomeDimensions, **kwargs):
        super().__init__(**kwargs)
        self.dims = dims

    def _do(self, problem, X, **kwargs):
        for i in range(len(X)):
            self._repair_chromosome(X[i])
        return X

    def _repair_chromosome(self, x: NDArray) -> None:
        for i in range(self.dims.n_nodes):
            pt = get_piece_type(x, i)
            p2 = get_port2_conn(x, i)
            p3 = get_port3_conn(x, i)

            if pt == INACTIVE:
                if p2 != INACTIVE:
                    set_port2_conn(x, i, INACTIVE)
                if p3 != INACTIVE:
                    set_port3_conn(x, i, INACTIVE)
                continue

            # 2-port pieces: no extra connections
            if pt not in THREE_PORT_PIECES and pt not in FOUR_PORT_PIECES:
                if p2 != INACTIVE:
                    set_port2_conn(x, i, INACTIVE)
                if p3 != INACTIVE:
                    set_port3_conn(x, i, INACTIVE)
                continue

            # 3-port pieces: port3 should be -1
            if pt in THREE_PORT_PIECES and p3 != INACTIVE:
                set_port3_conn(x, i, INACTIVE)

            # Validate connection targets are in bounds and active
            if p2 != INACTIVE:
                if p2 < 0 or p2 >= self.dims.n_nodes or get_piece_type(x, p2) == INACTIVE:
                    set_port2_conn(x, i, INACTIVE)

            if p3 != INACTIVE:
                if p3 < 0 or p3 >= self.dims.n_nodes or get_piece_type(x, p3) == INACTIVE:
                    set_port3_conn(x, i, INACTIVE)


# =============================================================================
# Closure Repair
# =============================================================================

class ClosureRepair(Repair):
    """Fix angular closure by adding or removing curves.

    If total angle deficit exists, append curves.
    If excess, remove curves from the end.
    """

    def __init__(self, dims: ChromosomeDimensions,
                 catalog_fk_table: NDArray,
                 inventory_by_index: Dict[int, int],
                 curve_index: int = 2,  # R40_LEFT
                 curve_angle: float = 22.5,
                 **kwargs):
        super().__init__(**kwargs)
        self.dims = dims
        self.fk_table = catalog_fk_table
        self.inventory_by_index = inventory_by_index
        self.curve_index = curve_index
        self.curve_angle = curve_angle

    def _do(self, problem, X, **kwargs):
        for i in range(len(X)):
            self._repair_chromosome(X[i])
        return X

    def _repair_chromosome(self, x: NDArray) -> None:
        # Compute current total angle
        total_angle = 0.0
        usage: Dict[int, int] = {}
        last_active = -1

        for i in range(self.dims.n_nodes):
            pt = get_piece_type(x, i)
            if pt == INACTIVE or pt < 0:
                continue
            if pt < len(self.fk_table):
                total_angle += self.fk_table[pt, 2]
            usage[pt] = usage.get(pt, 0) + 1
            last_active = i

        deficit = 360.0 - total_angle
        if abs(deficit) < 1.0:
            return

        # Add curves if deficit > curve_angle
        curve_available = (
            self.inventory_by_index.get(self.curve_index, 0)
            - usage.get(self.curve_index, 0)
        )

        if deficit > self.curve_angle and curve_available > 0:
            n_add = min(int(deficit / self.curve_angle), curve_available, 4)
            added = 0
            for i in range(last_active + 1, self.dims.n_nodes):
                if added >= n_add:
                    break
                if get_piece_type(x, i) == INACTIVE:
                    set_node(x, i, self.curve_index)
                    added += 1

        # Remove curves if deficit < -curve_angle
        elif deficit < -self.curve_angle:
            n_remove = min(int(abs(deficit) / self.curve_angle), 4)
            removed = 0
            for i in range(self.dims.n_nodes - 1, -1, -1):
                if removed >= n_remove:
                    break
                pt = get_piece_type(x, i)
                if pt == self.curve_index:
                    set_node(x, i, INACTIVE)
                    removed += 1


# =============================================================================
# Combined Pipeline
# =============================================================================

class TrackRepairPipeline(Repair):
    """Combined repair pipeline: connections → switches → inventory → closure."""

    def __init__(self, dims: ChromosomeDimensions,
                 inventory_by_index: Dict[int, int],
                 catalog_fk_table: Optional[NDArray] = None,
                 enable_closure_repair: bool = True,
                 enable_switch_repair: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.dims = dims

        self.connection_repair = ConnectionRepair(dims)
        self.switch_repair = SwitchPairingRepair(dims) if enable_switch_repair else None
        self.inventory_repair = InventoryRepair(inventory_by_index, dims)
        self.closure_repair = (
            ClosureRepair(dims, catalog_fk_table, inventory_by_index)
            if enable_closure_repair and catalog_fk_table is not None
            else None
        )

    def _do(self, problem, X, **kwargs):
        # Order: connections first (structural), then switches, inventory, closure
        X = self.connection_repair._do(problem, X, **kwargs)

        if self.switch_repair is not None:
            X = self.switch_repair._do(problem, X, **kwargs)

        X = self.inventory_repair._do(problem, X, **kwargs)

        if self.closure_repair is not None:
            X = self.closure_repair._do(problem, X, **kwargs)

        return X
