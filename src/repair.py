"""Repair operators for partitioned chromosome encoding.

Operates on the three-ingredient chromosome layout:
  [Main Loop: N genes] [Junction Descriptors: J×4 genes] [Start Position: 2 genes]

Operators:
- MainLoopClosureRepair: Adjust R40 curves to approach 360° angular closure
- JunctionValidityRepair: Clamp junction genes to valid ranges
- InventoryRepair: Enforce piece inventory limits across main loop and junctions
- TrackRepairPipeline: Chains all repairs in correct order
"""

from collections import Counter
from typing import Dict, Optional

import numpy as np
from numpy.typing import NDArray
from pymoo.core.repair import Repair

from .encoding import (
    INACTIVE,
    GENES_PER_JUNCTION,
    PartitionedDimensions,
    PieceIndex,
    get_active_junctions,
    get_active_main_pieces,
    get_junction,
    get_main_loop_types,
    set_junction,
    set_main_loop_type,
)

R40_LEFT = int(PieceIndex.R40_LEFT)
R40_RIGHT = int(PieceIndex.R40_RIGHT)
STRAIGHT_16 = int(PieceIndex.STRAIGHT_16)
STRAIGHT_24 = int(PieceIndex.STRAIGHT_24)

# Angles per R40 curve piece
R40_ANGLE = 22.5
TARGET_ANGLE = 360.0
CLOSURE_TOLERANCE = 1.0  # degrees


# =============================================================================
# Main Loop Closure Repair
# =============================================================================

class MainLoopClosureRepair(Repair):
    """Adjust R40 curves in the main loop to approach 360 degree closure.

    Computes total angle from active main loop pieces using the FK table.
    If angle deficit exists, appends R40_LEFT curves into empty slots.
    If angle excess exists, removes R40 curves from the end.
    """

    def __init__(
        self,
        dims: PartitionedDimensions,
        catalog_fk_table: NDArray,
        inventory_by_index: Dict[int, int],
        max_corrections: int = 4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dims = dims
        self.fk_table = catalog_fk_table
        self.inventory_by_index = inventory_by_index
        self.max_corrections = max_corrections

    def _do(self, problem, X, **kwargs):
        for i in range(len(X)):
            self._repair_chromosome(X[i])
        return X

    def _repair_chromosome(self, x: NDArray) -> None:
        types = get_main_loop_types(x, self.dims)
        active_mask = types != INACTIVE

        if not np.any(active_mask):
            return

        # Compute total angle from FK table
        active_types = types[active_mask]
        total_angle = self._compute_total_angle(active_types)
        deficit = TARGET_ANGLE - total_angle

        if abs(deficit) < CLOSURE_TOLERANCE:
            return

        # Count current R40 usage for inventory check
        usage = Counter(int(t) for t in active_types)

        if deficit > R40_ANGLE:
            self._add_curves(x, types, usage, deficit)
        elif deficit < -R40_ANGLE:
            self._remove_curves(x, types, deficit)

    def _compute_total_angle(self, active_types: NDArray) -> float:
        """Sum dtheta for all active pieces via FK table lookup."""
        total = 0.0
        for pt in active_types:
            pt_int = int(pt)
            if 0 <= pt_int < len(self.fk_table):
                total += self.fk_table[pt_int, 2]
        return total

    def _add_curves(
        self,
        x: NDArray,
        types: NDArray,
        usage: Counter,
        deficit: float,
    ) -> None:
        """Add R40 curves to reduce angular deficit."""
        # Choose curve direction based on deficit sign (positive = need left turns)
        curve_idx = R40_LEFT if deficit > 0 else R40_RIGHT
        curve_angle = R40_ANGLE if deficit > 0 else -R40_ANGLE

        available = (
            self.inventory_by_index.get(curve_idx, 0) - usage.get(curve_idx, 0)
        )
        if available <= 0:
            return

        n_needed = int(abs(deficit) / R40_ANGLE)
        n_add = min(n_needed, available, self.max_corrections)

        # Find empty slots after last active piece
        added = 0
        for pos in range(self.dims.n_main):
            if added >= n_add:
                break
            if types[pos] == INACTIVE:
                set_main_loop_type(x, self.dims, pos, curve_idx)
                types[pos] = curve_idx
                added += 1

    def _remove_curves(
        self,
        x: NDArray,
        types: NDArray,
        deficit: float,
    ) -> None:
        """Remove R40 curves to reduce angular excess."""
        # Excess angle: deficit is negative, remove curves matching the excess direction
        # If total > 360 (deficit < 0), remove curves contributing to excess
        curve_idx = R40_LEFT if deficit < 0 else R40_RIGHT

        n_needed = int(abs(deficit) / R40_ANGLE)
        n_remove = min(n_needed, self.max_corrections)

        removed = 0
        for pos in range(self.dims.n_main - 1, -1, -1):
            if removed >= n_remove:
                break
            if int(types[pos]) == curve_idx:
                set_main_loop_type(x, self.dims, pos, INACTIVE)
                types[pos] = INACTIVE
                removed += 1


# =============================================================================
# Junction Validity Repair
# =============================================================================

class JunctionValidityRepair(Repair):
    """Clamp junction descriptor genes to valid ranges.

    - Clamps position to [0, n_active_main - 1]
    - Clamps n_straights to available straight inventory
    - Deactivates junctions when switch inventory is insufficient
    """

    def __init__(
        self,
        dims: PartitionedDimensions,
        inventory_by_index: Dict[int, int],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dims = dims
        self.inventory_by_index = inventory_by_index

    def _do(self, problem, X, **kwargs):
        for i in range(len(X)):
            self._repair_chromosome(X[i])
        return X

    def _repair_chromosome(self, x: NDArray) -> None:
        if self.dims.max_junctions == 0:
            return

        # Count active main loop pieces for position clamping
        active_main = get_active_main_pieces(x, self.dims)
        n_active = len(active_main)

        if n_active == 0:
            # No main loop pieces: deactivate all junctions
            for k in range(self.dims.max_junctions):
                active, pos, hand, n_str = get_junction(x, self.dims, k)
                if active:
                    set_junction(x, self.dims, k, 0, pos, hand, n_str)
            return

        max_pos = n_active - 1

        # Track total straights claimed by active junctions
        total_straights_claimed = 0
        active_junction_count = 0

        for k in range(self.dims.max_junctions):
            active, pos, hand, n_str = get_junction(x, self.dims, k)
            if not active:
                continue

            active_junction_count += 1
            modified = False

            # Clamp position
            if pos < 0 or pos > max_pos:
                pos = np.clip(pos, 0, max_pos)
                modified = True

            # Clamp handedness to valid range [0, 3]
            if hand < 0 or hand > 3:
                hand = np.clip(hand, 0, 3)
                modified = True

            # Clamp n_straights to available inventory
            remaining_straights = self.dims.total_straights - total_straights_claimed
            if n_str < 0:
                n_str = 0
                modified = True
            elif n_str > remaining_straights:
                n_str = max(0, remaining_straights)
                modified = True

            total_straights_claimed += n_str

            if modified:
                set_junction(x, self.dims, k, active, pos, hand, n_str)

        # Deactivate excess junctions if switch inventory is insufficient
        # Each junction needs one Type A and one Type B physical switch
        self._deactivate_excess_junctions(x, active_junction_count)

    def _deactivate_excess_junctions(self, x: NDArray, n_active: int) -> None:
        """Deactivate junctions beyond available switch inventory.

        Each passing siding is opposite-handed: 1 LEFT + 1 RIGHT switch.
        Max pair count = min(LEFT_count, RIGHT_count).
        """
        from .encoding import SWITCH_LEFT, SWITCH_RIGHT

        left_count = self.inventory_by_index.get(int(SWITCH_LEFT), 0)
        right_count = self.inventory_by_index.get(int(SWITCH_RIGHT), 0)
        max_pairs = min(left_count, right_count)

        if n_active <= max_pairs:
            return

        # Deactivate junctions from the end until we're within budget
        deactivated = 0
        target = n_active - max_pairs
        for k in range(self.dims.max_junctions - 1, -1, -1):
            if deactivated >= target:
                break
            active, pos, hand, n_str = get_junction(x, self.dims, k)
            if active:
                set_junction(x, self.dims, k, 0, pos, hand, n_str)
                deactivated += 1


# =============================================================================
# Inventory Repair
# =============================================================================

class InventoryRepair(Repair):
    """Enforce inventory limits across main loop and junction branch pieces.

    Counts total piece usage from:
    - Main loop genes (active piece types)
    - Active junctions (each junction consumes 2 switches + N straights)

    When any piece type exceeds inventory, deactivates excess main loop genes
    from the end of the sequence.
    """

    def __init__(
        self,
        dims: PartitionedDimensions,
        inventory_by_index: Dict[int, int],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dims = dims
        self.inventory_by_index = inventory_by_index

    def _do(self, problem, X, **kwargs):
        for i in range(len(X)):
            self._repair_chromosome(X[i])
        return X

    def _repair_chromosome(self, x: NDArray) -> None:
        # Count main loop usage
        usage: Dict[int, int] = {}
        for pos in range(self.dims.n_main):
            pt = int(x[pos])
            if pt == INACTIVE:
                continue
            usage[pt] = usage.get(pt, 0) + 1

        # Each active junction is a passing siding consuming 1 LEFT + 1 RIGHT
        # switch (opposite-handed pair) regardless of which side it diverges to.
        from .encoding import SWITCH_LEFT, SWITCH_RIGHT
        left_sw, right_sw = int(SWITCH_LEFT), int(SWITCH_RIGHT)

        for k in range(self.dims.max_junctions):
            active, pos, hand, n_str = get_junction(x, self.dims, k)
            if not active:
                continue

            usage[left_sw] = usage.get(left_sw, 0) + 1
            usage[right_sw] = usage.get(right_sw, 0) + 1

            if n_str > 0:
                usage[STRAIGHT_16] = usage.get(STRAIGHT_16, 0) + n_str

        # Check for violations and deactivate excess main loop pieces from end
        violations = {
            pt: count - self.inventory_by_index.get(pt, 0)
            for pt, count in usage.items()
            if count > self.inventory_by_index.get(pt, 0)
        }

        if not violations:
            return

        # Remove excess from main loop, scanning from end
        for pos in range(self.dims.n_main - 1, -1, -1):
            if not violations:
                break

            pt = int(x[pos])
            if pt == INACTIVE:
                continue

            excess = violations.get(pt, 0)
            if excess > 0:
                set_main_loop_type(x, self.dims, pos, INACTIVE)
                violations[pt] -= 1
                if violations[pt] <= 0:
                    del violations[pt]


# =============================================================================
# Combined Pipeline
# =============================================================================

class TrackRepairPipeline(Repair):
    """Chains repair operators: JunctionValidity -> Inventory -> MainLoopClosure.

    Order rationale:
    1. JunctionValidityRepair first — clamps junction genes so downstream
       inventory counting is accurate.
    2. InventoryRepair — removes excess pieces, affecting angle totals.
    3. MainLoopClosureRepair last — adjusts curves based on final piece set.
    """

    def __init__(
        self,
        dims: PartitionedDimensions,
        inventory_by_index: Dict[int, int],
        catalog_fk_table: NDArray,
        enable_closure_repair: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.junction_repair = JunctionValidityRepair(dims, inventory_by_index)
        self.inventory_repair = InventoryRepair(dims, inventory_by_index)
        self.closure_repair = (
            MainLoopClosureRepair(dims, catalog_fk_table, inventory_by_index)
            if enable_closure_repair
            else None
        )

    def _do(self, problem, X, **kwargs):
        X = self.junction_repair._do(problem, X, **kwargs)
        X = self.inventory_repair._do(problem, X, **kwargs)
        if self.closure_repair is not None:
            X = self.closure_repair._do(problem, X, **kwargs)
        return X
