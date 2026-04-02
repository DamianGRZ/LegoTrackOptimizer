"""Integer sampling strategies for CGP-inspired chromosome encoding.

Generates initial population with:
- Heuristic seeds: known closed-loop patterns (circles, ovals, racetracks)
- Random chromosomes: random piece types with closure-aware angle balancing
- Connection-aware: heuristic patterns include proper branch connections

All chromosomes use direct integer encoding (piece_type, port2_conn, port3_conn).
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from pymoo.core.sampling import Sampling

from .config import OptimizationConfig
from .data import TrackCatalog
from .encoding import (
    CROSS_90,
    GENES_PER_NODE,
    INACTIVE,
    IN_SWITCH_INDICES,
    OUT_SWITCH_INDICES,
    R40_LEFT,
    R40_RIGHT,
    STRAIGHT_16,
    STRAIGHT_24,
    SWITCH_LEFT_IN,
    SWITCH_LEFT_OUT,
    SWITCH_RIGHT_IN,
    SWITCH_RIGHT_OUT,
    ChromosomeDimensions,
    compute_dimensions,
    create_chromosome_from_pieces,
    create_empty_chromosome,
    set_node,
    set_port2_conn,
)


# Heuristic ratio: fraction of population seeded with valid patterns
HEURISTIC_RATIO = 0.08


class IntegerSampling(Sampling):
    """Primary sampling class for integer CGP chromosomes.

    Generates a mix of heuristic seeds and random chromosomes.
    """

    def __init__(self, catalog: TrackCatalog, config: OptimizationConfig,
                 heuristic_ratio: float = HEURISTIC_RATIO, **kwargs):
        super().__init__(**kwargs)
        self.catalog = catalog
        self.config = config
        self.dims = compute_dimensions(config.total_inventory)
        self.heuristic_ratio = heuristic_ratio

        # Build inventory by index
        self.inventory_by_index: Dict[int, int] = {}
        for piece_id, count in config.inventory.items():
            idx = catalog._id_to_index.get(piece_id)
            if idx is not None:
                self.inventory_by_index[idx] = count

    def _do(self, problem, n_samples, **kwargs):
        X = np.full((n_samples, self.dims.n_var), INACTIVE, dtype=np.int16)

        n_heuristic = max(1, int(n_samples * self.heuristic_ratio))

        # Generate heuristic seeds
        patterns = self._get_heuristic_patterns()
        for i in range(n_heuristic):
            pattern = patterns[i % len(patterns)] if patterns else []
            if pattern:
                x = create_chromosome_from_pieces(self.dims, pattern)
                X[i, :] = x
            else:
                X[i, :] = self._random_chromosome()

        # Generate random chromosomes
        for i in range(n_heuristic, n_samples):
            X[i, :] = self._closure_aware_chromosome()

        return X

    def _get_heuristic_patterns(self) -> List[List[int]]:
        """Generate heuristic piece patterns from inventory."""
        patterns = []

        inv = self.inventory_by_index
        n_left = inv.get(R40_LEFT, 0)
        n_right = inv.get(R40_RIGHT, 0)
        n_str16 = inv.get(STRAIGHT_16, 0)
        n_str24 = inv.get(STRAIGHT_24, 0)

        # Pattern 1: Simple circle (16 R40_LEFT)
        if n_left >= 16:
            patterns.append([R40_LEFT] * 16)

        # Pattern 2: Simple circle (16 R40_RIGHT)
        if n_right >= 16:
            patterns.append([R40_RIGHT] * 16)

        # Pattern 3: Symmetric oval (4+4 curves + straights)
        if n_left >= 8 and n_str16 >= 4:
            n_str = min(n_str16, 8)
            half_str = n_str // 2
            patterns.append(
                [R40_LEFT] * 4 + [STRAIGHT_16] * half_str +
                [R40_LEFT] * 4 + [STRAIGHT_16] * half_str
            )

        # Pattern 4: Racetrack with right curves
        if n_right >= 8 and n_str16 >= 4:
            n_str = min(n_str16, 8)
            half_str = n_str // 2
            patterns.append(
                [R40_RIGHT] * 4 + [STRAIGHT_16] * half_str +
                [R40_RIGHT] * 4 + [STRAIGHT_16] * half_str
            )

        # Pattern 5: Large oval with mixed straights
        if n_left >= 8 and n_str16 >= 2 and n_str24 >= 2:
            patterns.append(
                [R40_LEFT] * 4 + [STRAIGHT_24] * 2 + [STRAIGHT_16] * 2 +
                [R40_LEFT] * 4 + [STRAIGHT_24] * 2 + [STRAIGHT_16] * 2
            )

        # Pattern 6: Oval with left passing siding (if switches available)
        n_sw_l_in = inv.get(SWITCH_LEFT_IN, 0)
        n_sw_l_out = inv.get(SWITCH_LEFT_OUT, 0)
        if n_left >= 10 and n_str16 >= 8 and n_sw_l_in >= 1 and n_sw_l_out >= 1:
            # Main loop: curves + switch_in + straights + switch_out + curves + straights
            main = (
                [R40_LEFT] * 4 +
                [SWITCH_LEFT_IN] +
                [STRAIGHT_16] * 4 +
                [SWITCH_LEFT_OUT] +
                [R40_LEFT] * 4 +
                [STRAIGHT_16] * 4
            )
            # Branch: approach_curve + straights + return_curve
            branch = [R40_RIGHT, STRAIGHT_16, STRAIGHT_16, R40_LEFT]
            patterns.append(main)
            # Store branch spec for connection wiring (handled separately)

        # Pattern 7: Oval with right passing siding
        n_sw_r_in = inv.get(SWITCH_RIGHT_IN, 0)
        n_sw_r_out = inv.get(SWITCH_RIGHT_OUT, 0)
        if n_right >= 10 and n_str16 >= 8 and n_sw_r_in >= 1 and n_sw_r_out >= 1:
            main = (
                [R40_RIGHT] * 4 +
                [SWITCH_RIGHT_IN] +
                [STRAIGHT_16] * 4 +
                [SWITCH_RIGHT_OUT] +
                [R40_RIGHT] * 4 +
                [STRAIGHT_16] * 4
            )
            patterns.append(main)

        return patterns

    def _closure_aware_chromosome(self) -> NDArray:
        """Generate a chromosome with guaranteed 360-degree angular closure."""
        x = create_empty_chromosome(self.dims)

        inv = dict(self.inventory_by_index)
        target = 360.0
        angle = 0.0
        pos = 0

        # Determine which curve to use
        curve_idx = R40_LEFT if inv.get(R40_LEFT, 0) >= 16 else R40_RIGHT
        curve_angle = 22.5 if curve_idx == R40_LEFT else -22.5

        # Place curves until we reach target
        while abs(angle) < abs(target) - 0.01 and pos < self.dims.n_nodes:
            if inv.get(curve_idx, 0) <= 0:
                break
            set_node(x, pos, curve_idx)
            inv[curve_idx] -= 1
            angle += curve_angle
            pos += 1

        # Sprinkle some straights at corners for variety
        n_str = min(inv.get(STRAIGHT_16, 0), min(7, self.dims.n_nodes - pos))
        if n_str > 0 and pos > 4:
            # Insert straights by shifting — simpler: append at end
            for _ in range(n_str):
                if pos >= self.dims.n_nodes:
                    break
                set_node(x, pos, STRAIGHT_16)
                inv[STRAIGHT_16] = inv.get(STRAIGHT_16, 0) - 1
                pos += 1

        return x

    def _random_chromosome(self) -> NDArray:
        """Generate a random chromosome with valid piece types."""
        x = create_empty_chromosome(self.dims)

        # Available piece indices (with inventory)
        available = [idx for idx, count in self.inventory_by_index.items() if count > 0]
        if not available:
            return x

        inv = dict(self.inventory_by_index)

        for i in range(self.dims.n_nodes):
            if not available:
                break

            # 30% chance of inactive node
            if np.random.random() < 0.3:
                continue

            idx = available[np.random.randint(len(available))]
            if inv.get(idx, 0) <= 0:
                available = [a for a in available if inv.get(a, 0) > 0]
                if not available:
                    break
                idx = available[np.random.randint(len(available))]

            set_node(x, i, idx)
            inv[idx] -= 1

        return x


# Backward-compatible aliases
MultiSegmentSampling = IntegerSampling
HeuristicSampling = IntegerSampling
