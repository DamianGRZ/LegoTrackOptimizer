"""Repair operators for partitioned chromosome encoding.

Operates on the three-ingredient chromosome layout:
  [Main Loop: N genes] [Junction Descriptors: J×4 genes] [Start Position: 2 genes]

Operators:
- MainLoopClosureRepair: R40 curves -> 360° angular closure, then drop straights
  to shrink the residual dx/dy positional gap
- JunctionValidityRepair: Clamp junction genes to valid ranges
- InventoryRepair: Enforce piece inventory limits across main loop and junctions
- BoundaryAwareRepair: Re-center or shrink layouts that exceed the boundary box
- TrackRepairPipeline: Chains all repairs in correct order
"""

import math
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from pymoo.core.repair import Repair

from .encoding import (
    INACTIVE,
    SWITCH_LEFT,
    SWITCH_RIGHT,
    PartitionedDimensions,
    PieceIndex,
    fk_rows_with_flips,
    get_active_cross_junctions,
    get_active_double_crossovers,
    get_active_junctions,
    get_active_main_pieces,
    get_junction,
    get_main_loop_flips,
    get_main_loop_types,
    set_flip,
    set_junction,
    set_main_loop_type,
)
from .geometry import compute_fk_chain
from .intersection import cross_pair_perpendicular, find_crossing_pairs

R40_CURVE = int(PieceIndex.R40_CURVE)
STRAIGHT_16 = int(PieceIndex.STRAIGHT_16)
STRAIGHT_24 = int(PieceIndex.STRAIGHT_24)

# Angles per R40 curve piece
R40_ANGLE = 22.5
CLOSURE_TOLERANCE = 1.0  # degrees

# The feasible closed topologies this R40-only kit can build are a single loop
# (+/-360, one winding) or a self-crossing figure-8 (turning sum 0). Repair
# drives the heading sum toward whichever is nearest ON THE SAME SIDE as the
# current sum, so a right-handed loop (-360) is never dragged toward +360. A
# double-wrap (+/-720) needs two nested laps, which one radius can't build (its
# seed family is a stub, proven infeasible), so it is deliberately not a target.
ANGLE_STEP = 360.0

SIDING_MARGIN = 16.0  # conservative per-side reach of an active passing siding (studs)


def _curve_flip(angle: float) -> int:
    """Flip of the R40 curve turning toward ``angle``'s sign (0=left, 1=right)."""
    return 0 if angle >= 0.0 else 1


def _main_loop_states(x: NDArray, dims: PartitionedDimensions,
                      fk_table: NDArray) -> NDArray:
    """FK states (origin-based) of the active main-loop pieces, flips applied."""
    types = x[:dims.n_main]
    flips = x[dims.main_flips_start:dims.main_flips_end]
    mask = types != INACTIVE
    if not mask.any():
        return np.zeros((1, 3), dtype=np.float64)
    return compute_fk_chain(fk_rows_with_flips(fk_table, types[mask], flips[mask]))


_STRAIGHT_TYPES = (STRAIGHT_16, STRAIGHT_24)
# Axis -> the two anti-parallel headings (degrees) we shrink along.
_AXIS_HEADINGS = {"x": (0.0, 180.0), "y": (90.0, 270.0)}
_HEADING_TOL = 1.0  # deg; headings are exact multiples of 22.5


def _active_straight_headings(
    x: NDArray, dims: PartitionedDimensions, fk_table: NDArray,
) -> List[Tuple[int, int, float]]:
    """List of (slot, piece_type, world_heading_deg) for each active straight."""
    types = x[:dims.n_main]
    flips = x[dims.main_flips_start:dims.main_flips_end]
    out: List[Tuple[int, int, float]] = []
    heading = 0.0
    for slot in range(dims.n_main):
        pt = int(types[slot])
        if pt == INACTIVE:
            continue
        if pt in _STRAIGHT_TYPES:
            out.append((slot, pt, heading))
        dtheta = float(fk_table[pt, 2])
        if pt == R40_CURVE and int(flips[slot]) == 1:
            dtheta = -dtheta
        heading += dtheta
    return out


def _find_antiparallel_pairs(
    headings: List[Tuple[int, int, float]], axis: str,
) -> List[Tuple[int, int]]:
    """Pairs (slot_a, slot_b) of SAME-type straights on opposite headings of `axis`."""
    h_lo, h_hi = _AXIS_HEADINGS[axis]
    side_a, side_b = {}, {}  # piece_type -> [slots]
    for slot, pt, h in headings:
        hm = h % 360.0
        if abs((hm - h_lo + 180) % 360 - 180) <= _HEADING_TOL:
            side_a.setdefault(pt, []).append(slot)
        elif abs((hm - h_hi + 180) % 360 - 180) <= _HEADING_TOL:
            side_b.setdefault(pt, []).append(slot)
    pairs = []
    for pt in set(side_a) & set(side_b):
        for a, b in zip(side_a[pt], side_b[pt]):  # same-type, equal length
            pairs.append((a, b))
    return pairs


# =============================================================================
# Main Loop Closure Repair
# =============================================================================

class MainLoopClosureRepair(Repair):
    """Close the main loop in two stages: angular, then translational.

    Stage 1 (angle): sum the signed dtheta of active main-loop pieces, then add
    or remove R40 curves to drive that sum onto the nearest feasible closed
    target on the same side (+/-360 for a loop, 0 for a figure-8).

    Stage 2 (position): once the loop is angularly closed, drop straights to
    shrink the residual dx/dy gap between the loop's tail and its start. A
    straight turns by 0 degrees, so removing it translates the tail without
    disturbing the 360-degree sum (see ``_close_translation``).
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

    def _target_angle(self, x: NDArray, total_angle: float) -> float:
        """Nearest feasible closed-turning target for this genome.

        A single loop closes at the full winding on the same side as the current
        turning (+360 for a left loop, -360 for a right one), so an already-closed
        loop of either handedness is left untouched. The figure-8 target 0
        competes only when the turning is closer to 0 than to a full winding AND
        the loop actually crosses itself, so a near-flat plain genome is still
        closed to +/-360 rather than frozen open.
        """
        full = ANGLE_STEP if total_angle >= 0.0 else -ANGLE_STEP
        if abs(total_angle) < abs(full - total_angle) and self._has_crossing(x):
            return 0.0
        return full

    def _has_crossing(self, x: NDArray) -> bool:
        """True when the decoded loop will contain a crossing.

        A descriptor crossing (cross-junction / double-crossover gene) is a cheap
        gene read; an emergent one — a perpendicular STRAIGHT_16-on-STRAIGHT_16
        overlap the decoder converts to CROSS_90 — needs the decoder's own
        self-intersection scan. The caller only reaches the emergent scan for
        near-zero-turning genomes, so plain loops never pay for it.
        """
        if (get_active_cross_junctions(x, self.dims)
                or get_active_double_crossovers(x, self.dims)):
            return True
        types = get_main_loop_types(x, self.dims)
        flips = get_main_loop_flips(x, self.dims)
        mask = types != INACTIVE
        if int(mask.sum()) < 4:
            return False
        active = [int(t) for t in types[mask]]
        states = compute_fk_chain(fk_rows_with_flips(self.fk_table, types[mask], flips[mask]))
        return any(
            active[pi] == STRAIGHT_16 and active[pj] == STRAIGHT_16
            and cross_pair_perpendicular(states, pi, pj)
            for pi, pj, _ in find_crossing_pairs(states, active)
        )

    def _reroutes_main_loop(self, x: NDArray) -> bool:
        """True when a descriptor re-lengthens main-loop slots at decode time.

        A double-crossover reroutes two slots; a passing siding replaces two
        straights with 32-stud switch bodies. Either way the raw main-loop dx/dy
        gap is a decode artifact, so Stage-2 translational closure must not act
        on it (dropping straights would break an otherwise valid loop).
        """
        return bool(get_active_double_crossovers(x, self.dims)
                    or get_active_junctions(x, self.dims))

    def _repair_chromosome(self, x: NDArray) -> None:
        types = get_main_loop_types(x, self.dims)
        flips = get_main_loop_flips(x, self.dims)
        active_mask = types != INACTIVE

        if not np.any(active_mask):
            return

        total_angle = self._compute_total_angle(types[active_mask], flips[active_mask])
        target = self._target_angle(x, total_angle)
        deficit = target - total_angle

        # Stage 1: angular closure. Add or remove whole R40 curves by turning
        # magnitude (|target| vs |total|), handedness from the sign, so loops of
        # either chirality close instead of only left-handed ones.
        if abs(deficit) >= R40_ANGLE - CLOSURE_TOLERANCE:
            n_curves = round(abs(deficit) / R40_ANGLE)
            if abs(target) >= abs(total_angle):
                usage = Counter(int(t) for t in types[active_mask])
                self._add_curves(x, types, flips, usage, n_curves, _curve_flip(target))
            else:
                self._remove_curves(x, types, flips, n_curves, _curve_flip(total_angle))
            active_mask = types != INACTIVE
            total_angle = self._compute_total_angle(types[active_mask], flips[active_mask])

        # Stage 2: translational closure — only on an angularly closed loop whose
        # raw main-loop gap is real. Straights carry dtheta=0, so dropping one
        # shifts the tail without disturbing the angular sum. Skipped when a
        # descriptor re-lengthens slots at decode (DC or passing siding), where
        # the raw gap is an artifact (see _reroutes_main_loop).
        if (abs(target - total_angle) < CLOSURE_TOLERANCE
                and not self._reroutes_main_loop(x)):
            self._close_translation(x)

    def _compute_total_angle(self, active_types: NDArray, active_flips: NDArray) -> float:
        """Sum signed dtheta across active pieces, honoring R40_CURVE flips."""
        types = np.asarray(active_types, dtype=np.int32)
        flips = np.asarray(active_flips, dtype=np.int32)
        valid = (types >= 0) & (types < len(self.fk_table))
        if not np.any(valid):
            return 0.0
        fk = fk_rows_with_flips(self.fk_table, types[valid], flips[valid])
        return float(fk[:, 2].sum())

    def _add_curves(
        self,
        x: NDArray,
        types: NDArray,
        flips: NDArray,
        usage: Counter,
        n_curves: int,
        flip: int,
    ) -> None:
        """Fill up to ``n_curves`` empty slots with R40 curves of ``flip``."""
        available = self.inventory_by_index.get(R40_CURVE, 0) - usage.get(R40_CURVE, 0)
        n_add = min(n_curves, available, self.max_corrections)

        added = 0
        for pos in range(self.dims.n_main):
            if added >= n_add:
                break
            if types[pos] == INACTIVE:
                set_main_loop_type(x, self.dims, pos, R40_CURVE)
                set_flip(x, self.dims, pos, flip)
                types[pos] = R40_CURVE
                flips[pos] = flip
                added += 1

    def _remove_curves(
        self,
        x: NDArray,
        types: NDArray,
        flips: NDArray,
        n_curves: int,
        flip: int,
    ) -> None:
        """Drop up to ``n_curves`` R40 curves of ``flip`` from the loop tail."""
        n_remove = min(n_curves, self.max_corrections)

        removed = 0
        for pos in range(self.dims.n_main - 1, -1, -1):
            if removed >= n_remove:
                break
            if int(types[pos]) == R40_CURVE and int(flips[pos]) == flip:
                set_main_loop_type(x, self.dims, pos, INACTIVE)
                set_flip(x, self.dims, pos, 0)
                types[pos] = INACTIVE
                flips[pos] = 0
                removed += 1

    def _close_translation(self, x: NDArray) -> None:
        """Drop straights so the positional gap (dx, dy) shrinks toward closure.

        A straight at world heading ``h`` displaces the loop tail by
        ``L*(cos h, sin h)`` and turns by 0 degrees, so removing it subtracts
        exactly that vector from the closure gap and leaves every other piece's
        heading — and the 360-degree angular sum — untouched. Greedily drop the
        straight whose removal most reduces ``|gap|``, up to ``max_corrections``.
        """
        gap = _main_loop_states(x, self.dims, self.fk_table)[-1, :2].astype(np.float64)

        # World-displacement vector of each active straight.
        candidates = [
            (slot, np.array([math.cos(math.radians(h)), math.sin(math.radians(h))],
                            dtype=np.float64) * float(self.fk_table[ptype, 0]))
            for slot, ptype, h in _active_straight_headings(x, self.dims, self.fk_table)
        ]

        removed = 0
        while removed < self.max_corrections and candidates:
            gap_norm = float(np.hypot(*gap))
            best_i, best_reduction = -1, 1e-6
            for i, (_, disp) in enumerate(candidates):
                reduction = gap_norm - float(np.hypot(*(gap - disp)))
                if reduction > best_reduction:
                    best_i, best_reduction = i, reduction
            if best_i < 0:
                break  # no straight brings the tail closer to the start

            slot, disp = candidates.pop(best_i)
            set_main_loop_type(x, self.dims, slot, INACTIVE)
            set_flip(x, self.dims, slot, 0)
            gap -= disp
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

    def repair_chromosome(self, x: NDArray) -> None:
        """Public single-chromosome clamp, for downstream operators to re-run."""
        self._repair_chromosome(x)

    def _repair_chromosome(self, x: NDArray) -> None:
        if self.dims.max_junctions == 0:
            return

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
# Boundary-Aware Repair
# =============================================================================

class BoundaryAwareRepair(Repair):
    """Rescue out-of-bounds layouts: re-center if they fit, else symmetric shrink.

    Branch 1 (translate): if the main-loop span fits the box, zero start_x/start_y
        so the decoder's _auto_center places the loop at box center (max margin).
    Branch 2 (shrink): if an axis is genuinely too big, deactivate same-type
        anti-parallel straight pairs (closure- and angle-preserving), then translate.
    """

    def __init__(
        self,
        dims: PartitionedDimensions,
        catalog_fk_table: NDArray,
        *,
        siding_margin: float = SIDING_MARGIN,
        boundary_tolerance: float = 0.0,
        junction_repair: Optional["JunctionValidityRepair"] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dims = dims
        self.fk_table = catalog_fk_table
        self.siding_margin = siding_margin
        self.boundary_tolerance = boundary_tolerance
        self.junction_repair = junction_repair  # re-clamp after shrink (set by pipeline)

    def _do(self, problem, X, **kwargs):
        for i in range(len(X)):
            self._repair_chromosome(X[i])
        return X

    def _box_dims(self) -> Tuple[float, float]:
        return (self.dims.boundary_max_x - self.dims.boundary_min_x,
                self.dims.boundary_max_y - self.dims.boundary_min_y)

    def _effective_box(self, x: NDArray) -> Tuple[float, float]:
        box_w, box_h = self._box_dims()
        # If any siding junction is active, reserve a conservative margin.
        margin = 0.0
        for k in range(self.dims.max_junctions):
            if int(get_junction(x, self.dims, k)[0]) == 1:
                margin = self.siding_margin
                break
        return max(0.0, box_w - 2 * margin), max(0.0, box_h - 2 * margin)

    def _repair_chromosome(self, x: NDArray) -> None:
        # An active DC descriptor reroutes two slots at decode time, so the
        # raw main-loop span measured below is an artifact — decline
        # gracefully and leave boundary enforcement to the G[3] penalty.
        if get_active_double_crossovers(x, self.dims):
            return
        states = _main_loop_states(x, self.dims, self.fk_table)
        if states.shape[0] <= 1:
            return
        w = float(states[:, 0].max() - states[:, 0].min())
        h = float(states[:, 1].max() - states[:, 1].min())
        box_w_eff, box_h_eff = self._effective_box(x)

        start_x = float(x[self.dims.start_pos_start])
        start_y = float(x[self.dims.start_pos_start + 1])
        slack_x = (box_w_eff - w) / 2.0
        slack_y = (box_h_eff - h) / 2.0
        tol = self.boundary_tolerance

        x_too_big = w > box_w_eff
        y_too_big = h > box_h_eff
        x_offset_out = abs(start_x) > slack_x + tol
        y_offset_out = abs(start_y) > slack_y + tol

        if not (x_too_big or y_too_big or x_offset_out or y_offset_out):
            return  # already in bounds

        # A genuinely-too-big axis must shrink; a fitting loop only needs centering.
        shrank = (x_too_big or y_too_big) and self._shrink(x, w, h, box_w_eff, box_h_eff)

        # Translate: zero the fine-tuning offset so _auto_center fully centers.
        x[self.dims.start_pos_start] = 0
        x[self.dims.start_pos_start + 1] = 0

        if shrank and self.junction_repair is not None:
            # Deactivating straights shifted active-piece indices: re-clamp junctions.
            self.junction_repair.repair_chromosome(x)

    def _shrink(self, x: NDArray, w: float, h: float,
                box_w_eff: float, box_h_eff: float) -> bool:
        headings = _active_straight_headings(x, self.dims, self.fk_table)
        removed_any = False
        for axis, span, box in (("x", w, box_w_eff), ("y", h, box_h_eff)):
            deficit = span - box
            if deficit <= 0:
                continue
            pairs = _find_antiparallel_pairs(headings, axis)
            if not pairs:
                continue  # decline gracefully — never break closure
            # Length L = dx of the straight type in the first pair.
            a, b = pairs[0]
            ptype = int(x[a])
            length = float(self.fk_table[ptype, 0])
            n_pairs = min(len(pairs), math.ceil(deficit / max(length, 1e-6)))
            for a, b in pairs[:n_pairs]:
                set_main_loop_type(x, self.dims, a, INACTIVE)
                set_flip(x, self.dims, a, 0)
                set_main_loop_type(x, self.dims, b, INACTIVE)
                set_flip(x, self.dims, b, 0)
                removed_any = True
        return removed_any


# =============================================================================
# Combined Pipeline
# =============================================================================

class TrackRepairPipeline(Repair):
    """Chains: JunctionValidity -> Inventory -> MainLoopClosure -> BoundaryAware.

    Order rationale:
    1. JunctionValidityRepair first — clamps junction genes so downstream
       inventory counting is accurate.
    2. InventoryRepair — removes excess pieces, affecting angle totals.
    3. MainLoopClosureRepair — adjusts curves based on final piece set.
    4. BoundaryAwareRepair last — re-centers or shrinks the layout once the
       piece set is finalised.
    """

    def __init__(
        self,
        dims: PartitionedDimensions,
        inventory_by_index: Dict[int, int],
        catalog_fk_table: NDArray,
        enable_closure_repair: bool = True,
        enable_boundary_repair: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.junction_repair = JunctionValidityRepair(dims, inventory_by_index)
        self.inventory_repair = InventoryRepair(dims, inventory_by_index)
        self.closure_repair = (
            MainLoopClosureRepair(dims, catalog_fk_table, inventory_by_index)
            if enable_closure_repair else None
        )
        self.boundary_repair = (
            BoundaryAwareRepair(dims, catalog_fk_table,
                                junction_repair=self.junction_repair)
            if enable_boundary_repair else None
        )

    def _do(self, problem, X, **kwargs):
        X = self.junction_repair._do(problem, X, **kwargs)
        X = self.inventory_repair._do(problem, X, **kwargs)
        if self.closure_repair is not None:
            X = self.closure_repair._do(problem, X, **kwargs)
        if self.boundary_repair is not None:
            X = self.boundary_repair._do(problem, X, **kwargs)
        return X
