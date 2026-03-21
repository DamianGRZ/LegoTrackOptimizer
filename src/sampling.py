"""Random-key sampling strategies for generating initial population.

Provides sampling classes for random-key [0,1] chromosome encoding:
- RandomKeySampling: Primary sampling class for RK encoding
- MultiSegmentSampling: Alias for RandomKeySampling (backward compatibility)
- HeuristicSampling: Legacy sampling for flat integer chromosome encoding

Random-key approach:
- All chromosome values are [0.0, 1.0]
- Heuristic patterns are converted to RK values via piece_index_to_rk()
- Random chromosomes use closure-aware angle balancing
- Decoder maps RK values to pieces dynamically

IMPORTANT: All heuristic patterns undergo round-trip validation to ensure
they close after RK encoding/decoding.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from pymoo.core.sampling import Sampling

from .config import OptimizationConfig
from .data import TrackCatalog
from .encoding import (
    B_MAX,
    B_SLOT,
    INACTIVE,
    L_MAX,
    N_VAR,
    MAIN_LOOP_START,
    POSITION_START,
    RK_INACTIVE_THRESHOLD,
    create_chromosome_from_pattern,
    create_empty_chromosome,
    piece_index_to_rk,
    rk_to_piece_index,
    set_branch_slot,
    set_branch_template_params,
    set_switch_mask_value,
)

# Legacy piece index mapping - kept for backward compatibility
# New code should use catalog.index_to_id which is dynamically generated
# from track_pieces.yaml and automatically adapts to new pieces.
_LEGACY_INDEX_TO_ID = {
    0: "STRAIGHT_16",
    1: "STRAIGHT_24",
    2: "R40_LEFT",
    3: "R40_RIGHT",
    4: "CROSS_90",
    5: "R40_SWITCH_LEFT_IN",
    6: "R40_SWITCH_LEFT_OUT",
    7: "R40_SWITCH_RIGHT_IN",
    8: "R40_SWITCH_RIGHT_OUT",
    9: "DOUBLE_CROSSOVER",
}

# Alias for backward compatibility (deprecated - use catalog.index_to_id)
INDEX_TO_ID = _LEGACY_INDEX_TO_ID


# =============================================================================
# New Multi-Segment Sampling (Phase 1)
# =============================================================================

class MultiSegmentSampling(Sampling):
    """Sampling for multi-segment chromosome encoding.

    15% heuristic patterns (valid closed loops), 85% random chromosomes.
    Generates chromosomes in the N_VAR=218 gene format with template-based
    branch encoding (4 genes per branch slot).

    Heuristic patterns:
    - Simple circles (16 R40 curves)
    - Symmetric ovals (R40 curves + straights)
    - Racetracks (4 corners + straights)
    - Ovals with passing sidings (template-based branches)
    """

    HEURISTIC_RATIO = 0.20  # 20% heuristic, 80% random (research recommends 20-30%)

    def __init__(self, catalog: TrackCatalog, config: OptimizationConfig):
        """Initialize multi-segment sampling.

        Args:
            catalog: Track catalog for piece properties.
            config: Optimization configuration with inventory limits.
        """
        super().__init__()
        self.catalog = catalog  # Store for round-trip validation and FK lookup
        self.config = config
        self.inventory = config.inventory

        # Use dynamic index_to_id from catalog (auto-adapts to new pieces)
        self.index_to_id = catalog.index_to_id

        # Build valid piece list from inventory (sorted for consistency with decoder)
        self.valid_pieces = []
        for piece_id, count in self.inventory.items():
            idx = catalog._id_to_index.get(piece_id)
            if idx is not None and count > 0:
                self.valid_pieces.append(idx)
        self.valid_pieces.sort()  # Must match decoder's sorted available_pieces

        # Inventory by index for validation
        self.inventory_by_index = {}
        for piece_id, count in self.inventory.items():
            idx = catalog._id_to_index.get(piece_id)
            if idx is not None:
                self.inventory_by_index[idx] = count

        # FK angles for closure-aware sampling
        self.fk_angles = {
            idx: abs(catalog._fk_table[idx, 2])
            for idx in self.valid_pieces
        }

        # Categorize pieces for closure-aware generation
        self.curve_indices = [idx for idx in self.valid_pieces if self.fk_angles[idx] > 0]
        self.straight_indices = [idx for idx in self.valid_pieces if self.fk_angles[idx] == 0]

    def _do(self, problem, n_samples, **kwargs) -> NDArray:
        """Generate population with heuristic and random chromosomes.

        Args:
            problem: pymoo problem instance.
            n_samples: Number of chromosomes to generate.
            **kwargs: Additional arguments.

        Returns:
            Population array of shape (n_samples, N_VAR).
        """
        n_heuristic = int(n_samples * self.HEURISTIC_RATIO)
        n_random = n_samples - n_heuristic

        heuristic_samples = self._generate_heuristic_population(n_heuristic)
        random_samples = self._generate_random_population(n_random)

        # Combine populations
        population = np.vstack([heuristic_samples, random_samples])

        # Add random starting positions
        self._add_random_positions(population)

        return population

    def _add_random_positions(self, population: NDArray) -> None:
        """Add random starting positions to chromosomes (in-place).

        In RK encoding, position genes are [0,1] values that are scaled
        to world coordinates by the decoder.

        Args:
            population: Population array of shape (n, N_VAR).
        """
        n_samples = len(population)

        # Generate random position keys in [0, 1]
        rk_x = np.random.uniform(0.0, 1.0, size=n_samples)
        rk_y = np.random.uniform(0.0, 1.0, size=n_samples)

        # Set position genes (RK values)
        population[:, POSITION_START] = rk_x
        population[:, POSITION_START + 1] = rk_y

    def _generate_heuristic_population(self, n_samples: int) -> NDArray:
        """Generate heuristic chromosomes with VALIDATED patterns.

        Converts integer piece patterns to RK chromosomes using
        piece_index_to_rk() for proper encoding. Each pattern undergoes
        round-trip validation to ensure it closes after decoding.

        Args:
            n_samples: Number of heuristic samples.

        Returns:
            Array of shape (n_samples, N_VAR) with [0,1] values.
        """
        population = np.zeros((n_samples, N_VAR), dtype=np.float64)

        pattern_generators = [
            self._simple_circle,
            self._symmetric_oval,
            self._racetrack,
            self._large_oval,
            self._double_oval,
            self._oval_with_left_siding,
            self._oval_with_right_siding,
        ]

        for i in range(n_samples):
            generator = pattern_generators[i % len(pattern_generators)]
            pattern = generator()

            if pattern is not None:
                # Convert integer pattern to RK chromosome
                chromosome = create_chromosome_from_pattern(
                    piece_indices=list(pattern),
                    available_pieces=self.valid_pieces,
                    start_rk_x=0.5,  # Center of boundary
                    start_rk_y=0.5,
                )

                # Round-trip validation: decode and check closure
                if self._validate_closure(chromosome, pattern):
                    population[i] = chromosome
                else:
                    # Pattern failed round-trip - use closure-aware random
                    population[i] = self._closure_aware_chromosome()
            else:
                population[i] = self._closure_aware_chromosome()

        return population

    def _validate_closure(
        self, chromosome: NDArray, expected_pattern: NDArray, tolerance: float = 5.0
    ) -> bool:
        """Validate that chromosome decodes to expected pattern and closes.

        Performs round-trip validation: decode RK chromosome and verify
        the pieces match expected and angular closure is within tolerance.

        Args:
            chromosome: RK chromosome to validate.
            expected_pattern: Original piece indices pattern.
            tolerance: Angular tolerance in degrees (default 5.0).

        Returns:
            True if pattern closes, False otherwise.
        """
        # Decode main loop genes to get piece sequence
        piece_keys = chromosome[MAIN_LOOP_START:MAIN_LOOP_START + L_MAX]
        decoded_pieces = []

        # Simulate decoder's piece selection (simplified)
        inventory_used = {}
        for rk_value in piece_keys:
            if rk_value < RK_INACTIVE_THRESHOLD:
                continue

            # Build available pieces from remaining inventory
            available = []
            for idx in sorted(self.inventory_by_index.keys()):
                remaining = self.inventory_by_index[idx] - inventory_used.get(idx, 0)
                if remaining > 0:
                    available.append(idx)

            if not available:
                break

            piece_idx = rk_to_piece_index(float(rk_value), available)
            if piece_idx >= 0:
                decoded_pieces.append(piece_idx)
                inventory_used[piece_idx] = inventory_used.get(piece_idx, 0) + 1

        if len(decoded_pieces) < 4:
            return False

        # Compute total angle
        total_angle = sum(
            abs(self.catalog._fk_table[idx, 2]) for idx in decoded_pieces
        )

        # Check for 360-degree closure
        angle_error = min(abs(total_angle % 360), 360 - abs(total_angle % 360))
        return angle_error <= tolerance

    def _generate_random_population(self, n_samples: int) -> NDArray:
        """Generate random chromosomes.

        Args:
            n_samples: Number of random samples.

        Returns:
            Array of shape (n_samples, N_VAR).
        """
        population = np.zeros((n_samples, N_VAR), dtype=np.float64)

        for i in range(n_samples):
            population[i] = self._random_chromosome()

        return population

    def _simple_circle(self) -> Optional[NDArray]:
        """Generate simple circle pattern: 16 R40_LEFT pieces.

        Returns:
            Main loop array or None if invalid.
        """
        pattern = np.array([2] * 16, dtype=np.int32)  # 16x R40_LEFT

        if not self._validate_inventory(pattern):
            return None

        return pattern

    def _symmetric_oval(self) -> Optional[NDArray]:
        """Generate symmetric oval: R40 curves + straights.

        Pattern: 4 corners of R40_LEFT with straights between.

        Returns:
            Main loop array or None if invalid.
        """
        pattern = np.array(
            [2, 2, 2, 2] +  # Corner 1 (90 deg)
            [0, 0] +        # Straight side
            [2, 2, 2, 2] +  # Corner 2 (90 deg)
            [0, 0] +        # Straight side
            [2, 2, 2, 2] +  # Corner 3 (90 deg)
            [0, 0] +        # Straight side
            [2, 2, 2, 2] +  # Corner 4 (90 deg)
            [0, 0],         # Straight side
            dtype=np.int32,
        )

        if not self._validate_inventory(pattern):
            return None

        return pattern

    def _racetrack(self) -> Optional[NDArray]:
        """Generate racetrack pattern: 4 corner groups + long straights.

        Returns:
            Main loop array or None if invalid.
        """
        pattern = np.array(
            [0, 0, 0, 0] +  # Long straight
            [2, 2, 2, 2] +  # Corner 1 (90 deg)
            [0, 0, 0, 0] +  # Long straight
            [2, 2, 2, 2] +  # Corner 2 (90 deg)
            [0, 0, 0, 0] +  # Long straight
            [2, 2, 2, 2] +  # Corner 3 (90 deg)
            [0, 0, 0, 0] +  # Long straight
            [2, 2, 2, 2],   # Corner 4 (90 deg)
            dtype=np.int32,
        )

        if not self._validate_inventory(pattern):
            return None

        return pattern

    def _large_oval(self) -> Optional[NDArray]:
        """Generate large oval pattern: uses more pieces for expanded layouts.

        Pattern: 4 corners of R40_LEFT (4 each) + longer straights (6 each side).
        Total: 16 curves + 24 straights = 40 pieces.

        Returns:
            Main loop array or None if invalid.
        """
        pattern = np.array(
            [2, 2, 2, 2] +      # Corner 1 (90 deg)
            [0, 0, 0, 0, 0, 0] +  # Long straight side (6 pieces)
            [2, 2, 2, 2] +      # Corner 2 (90 deg)
            [0, 0, 0, 0, 0, 0] +  # Long straight side
            [2, 2, 2, 2] +      # Corner 3 (90 deg)
            [0, 0, 0, 0, 0, 0] +  # Long straight side
            [2, 2, 2, 2] +      # Corner 4 (90 deg)
            [0, 0, 0, 0, 0, 0],   # Long straight side
            dtype=np.int32,
        )

        if not self._validate_inventory(pattern):
            return None

        return pattern

    def _double_oval(self) -> Optional[NDArray]:
        """Generate double oval pattern: figure-8 using all curve inventory.

        Pattern: Two connected ovals sharing a crossing point.
        Total: 32 curves + straights.

        Returns:
            Main loop array or None if invalid.
        """
        # First loop: left curves
        # Second loop: right curves (connected at crossing)
        pattern = np.array(
            # First oval (left turns)
            [2, 2, 2, 2] +  # 90 deg corner
            [0, 0] +        # Short straight
            [2, 2, 2, 2] +  # 90 deg corner
            [0, 0] +        # Short straight
            [2, 2, 2, 2] +  # 90 deg corner
            [0, 0] +        # Short straight
            [2, 2, 2, 2] +  # 90 deg corner (back to start)
            # Continue with second oval (right turns)
            [0, 0] +        # Short straight
            [3, 3, 3, 3] +  # 90 deg corner (right)
            [0, 0] +        # Short straight
            [3, 3, 3, 3] +  # 90 deg corner (right)
            [0, 0] +        # Short straight
            [3, 3, 3, 3] +  # 90 deg corner (right)
            [0, 0] +        # Short straight
            [3, 3, 3, 3],   # 90 deg corner (right)
            dtype=np.int32,
        )

        if not self._validate_inventory(pattern):
            return None

        return pattern

    def _oval_with_left_siding(self) -> Optional[NDArray]:
        """Generate oval with left-hand passing siding with switches in main loop.

        Places LEFT_IN (5) and LEFT_OUT (6) switches directly in the main loop.
        The decoder's Pass 2 will detect and pair them geometrically.

        Returns:
            Piece index array for conversion to RK chromosome, or None if invalid.
        """
        # LEFT_IN=5, LEFT_OUT=6, STRAIGHT_16=0, R40_LEFT=2
        main_loop = np.array(
            [2, 2, 2, 2] +    # Corner 1 (90 deg) - positions 0-3
            [5] +             # LEFT_IN switch - position 4
            [0, 0] +          # Straights - positions 5-6
            [6] +             # LEFT_OUT switch - position 7
            [0] +             # Straight - position 8
            [2, 2, 2, 2] +    # Corner 2 (90 deg)
            [0, 0] +          # Straight section
            [2, 2, 2, 2] +    # Corner 3 (90 deg)
            [0, 0] +          # Straight section
            [2, 2, 2, 2],     # Corner 4 (90 deg)
            dtype=np.int32,
        )

        if not self._validate_inventory(main_loop):
            return None

        # Check branch inventory (R40_RIGHT for approach, R40_LEFT for return)
        if not (self.inventory_by_index.get(3, 0) > 0 and
                self.inventory_by_index.get(2, 0) > 0):
            return None

        return main_loop

    def _oval_with_right_siding(self) -> Optional[NDArray]:
        """Generate oval with right-hand passing siding with switches in main loop.

        Places RIGHT_IN (7) and RIGHT_OUT (8) switches directly in the main loop.
        The decoder's Pass 2 will detect and pair them geometrically.

        Returns:
            Piece index array for conversion to RK chromosome, or None if invalid.
        """
        # RIGHT_IN=7, RIGHT_OUT=8, STRAIGHT_16=0, R40_RIGHT=3
        main_loop = np.array(
            [3, 3, 3, 3] +    # Corner 1 (90 deg right) - positions 0-3
            [7] +             # RIGHT_IN switch - position 4
            [0, 0] +          # Straights - positions 5-6
            [8] +             # RIGHT_OUT switch - position 7
            [0] +             # Straight - position 8
            [3, 3, 3, 3] +    # Corner 2 (90 deg right)
            [0, 0] +          # Straight section
            [3, 3, 3, 3] +    # Corner 3 (90 deg right)
            [0, 0] +          # Straight section
            [3, 3, 3, 3],     # Corner 4 (90 deg right)
            dtype=np.int32,
        )

        if not self._validate_inventory(main_loop):
            return None

        # Check branch inventory (R40_LEFT for approach, R40_RIGHT for return)
        if not (self.inventory_by_index.get(2, 0) > 0 and
                self.inventory_by_index.get(3, 0) > 0):
            return None
        return main_loop

    def _closure_aware_chromosome(self) -> NDArray:
        """Generate chromosome with guaranteed 360-degree angular closure.

        Uses angle-balanced piece selection to ensure the track can close.
        This is the PRIMARY method for generating feasible initial solutions.

        Strategy:
        1. Build piece sequence that sums to exactly 360 degrees
        2. Add optional straights for variety
        3. Convert to RK chromosome

        Returns:
            RK chromosome with closure-viable piece sequence.
        """
        # Get curve angle (22.5 deg for R40)
        curve_angle = 22.5  # R40 curves

        # Basic closed loop: 16 curves = 360 degrees
        # Add variety: use mix of left/right curves and straights
        pieces = []
        target_angle = 360.0
        accumulated_angle = 0.0

        # Determine available curves
        left_curve = 2  # R40_LEFT
        right_curve = 3  # R40_RIGHT
        straight = 0  # STRAIGHT_16

        # Check inventory
        left_available = self.inventory_by_index.get(left_curve, 0)
        right_available = self.inventory_by_index.get(right_curve, 0)
        straight_available = self.inventory_by_index.get(straight, 0)

        left_used = 0
        right_used = 0
        straight_used = 0

        # Build piece sequence with angle tracking
        # Use predominantly one direction for simpler closure
        use_left = np.random.random() < 0.5

        while abs(accumulated_angle - target_angle) > 0.01:
            remaining = target_angle - accumulated_angle

            if remaining >= curve_angle:
                # Need more turning - add curve
                if use_left and left_used < left_available:
                    pieces.append(left_curve)
                    accumulated_angle += curve_angle
                    left_used += 1
                elif not use_left and right_used < right_available:
                    pieces.append(right_curve)
                    accumulated_angle += curve_angle
                    right_used += 1
                elif left_used < left_available:
                    pieces.append(left_curve)
                    accumulated_angle += curve_angle
                    left_used += 1
                elif right_used < right_available:
                    pieces.append(right_curve)
                    accumulated_angle += curve_angle
                    right_used += 1
                else:
                    # No curves available - fail gracefully
                    break
            elif remaining > 0:
                # Small remaining angle - need to add curves to complete
                # This shouldn't happen if we're careful
                if use_left and left_used < left_available:
                    pieces.append(left_curve)
                    accumulated_angle += curve_angle
                    left_used += 1
                elif right_used < right_available:
                    pieces.append(right_curve)
                    accumulated_angle += curve_angle
                    right_used += 1
                else:
                    break

            # Safety limit
            if len(pieces) > 50:
                break

        # Add some straights for variety (after corners)
        n_straights_to_add = min(
            np.random.randint(0, 8),  # 0-7 straights
            straight_available
        )

        # Insert straights at corner transitions (after every 4 curves)
        if len(pieces) >= 8 and n_straights_to_add > 0:
            # Find corner positions (every 4 curves)
            insert_positions = [4, 8, 12]
            np.random.shuffle(insert_positions)

            for pos in insert_positions[:n_straights_to_add]:
                if pos <= len(pieces):
                    pieces.insert(pos, straight)

        # Convert to RK chromosome
        return create_chromosome_from_pattern(
            piece_indices=pieces,
            available_pieces=self.valid_pieces,
            start_rk_x=0.5,
            start_rk_y=0.5,
        )

    def _random_chromosome(self) -> NDArray:
        """Generate random chromosome with closure-aware piece selection.

        UPDATED: Now uses closure-aware generation to ensure feasibility.
        Creates a minimal viable track that can form closed loops.

        Returns:
            RK chromosome with closure-viable piece sequence.
        """
        # Use closure-aware generation as the base
        x = self._closure_aware_chromosome()

        # 30% chance to activate a branch slot
        if np.random.random() < 0.30:
            self._add_random_branch_genes(x)

        return x

    def _add_random_branch_genes(self, x: NDArray) -> None:
        """Add random branch template parameters.

        Sets one or more branch slots with random valid parameters.
        """
        # Check if switches are available
        left_available = (
            self.inventory_by_index.get(5, 0) > 0 and  # LEFT_IN
            self.inventory_by_index.get(6, 0) > 0 and  # LEFT_OUT
            self.inventory_by_index.get(3, 0) > 0 and  # R40_RIGHT for approach
            self.inventory_by_index.get(2, 0) > 0      # R40_LEFT for return
        )
        right_available = (
            self.inventory_by_index.get(7, 0) > 0 and  # RIGHT_IN
            self.inventory_by_index.get(8, 0) > 0 and  # RIGHT_OUT
            self.inventory_by_index.get(2, 0) > 0 and  # R40_LEFT for approach
            self.inventory_by_index.get(3, 0) > 0      # R40_RIGHT for return
        )

        if not (left_available or right_available):
            return

        # Choose handedness based on availability
        if left_available and right_available:
            handedness = np.random.randint(0, 2)
        elif left_available:
            handedness = 0  # LEFT
        else:
            handedness = 1  # RIGHT

        # Random IN position (first half of main loop is more likely to work)
        in_pos = np.random.randint(8, 50)

        # Random number of straights (0-4)
        n_straights = np.random.randint(0, 5)

        # Set branch template params using RK encoding
        set_branch_template_params(
            x, slot_idx=0,
            in_pos=in_pos,
            handedness=handedness,
            n_straights=n_straights,
            active=1
        )

    def _validate_inventory(self, pattern: NDArray) -> bool:
        """Check if pattern satisfies inventory constraints.

        Args:
            pattern: Array of piece indices.

        Returns:
            True if valid, False otherwise.
        """
        unique_indices, counts = np.unique(pattern, return_counts=True)
        usage = dict(zip(unique_indices, counts))

        for piece_idx, count in usage.items():
            # Use dynamic index_to_id from catalog
            piece_id = self.index_to_id.get(int(piece_idx))
            if piece_id is None:
                return False
            if piece_id not in self.inventory:
                return False
            if count > self.inventory[piece_id]:
                return False

        return True


# =============================================================================
# Legacy Heuristic Sampling (kept for backward compatibility)
# =============================================================================

class HeuristicSampling(Sampling):
    """Legacy sampling seeding 20% of population with valid closed loop patterns.

    Note: This class is kept for backward compatibility.
    New code should use MultiSegmentSampling.

    Generates heuristic patterns (simple circles, ovals, racetracks) that are
    geometrically valid and satisfy inventory constraints. Remaining 80% are
    random chromosomes for diversity.

    Research recommends 20-30% heuristic seeds; >50% can collapse diversity.

    Chromosome: [piece_idx_1, ..., piece_idx_N, start_x, start_y]
    """

    HEURISTIC_RATIO = 0.20  # 20% heuristic, 80% random

    def __init__(self, catalog: TrackCatalog, config: OptimizationConfig, n_piece_vars: int):
        """Initialize heuristic sampling.

        Args:
            catalog: Track catalog for piece properties.
            config: Optimization configuration with inventory limits.
            n_piece_vars: Number of piece variables (excluding position vars).
        """
        super().__init__()
        self.config = config
        self.inventory = config.inventory
        self.total_inventory = config.total_inventory
        self.n_piece_vars = n_piece_vars

        # Use dynamic index_to_id from catalog (auto-adapts to new pieces)
        # Store a copy to avoid deepcopy issues in pymoo
        self.index_to_id = dict(catalog.index_to_id)

    def _do(self, problem, n_samples, **kwargs) -> NDArray:
        """Generate population with heuristic and random chromosomes.

        Args:
            problem: pymoo problem instance.
            n_samples: Number of chromosomes to generate.
            **kwargs: Additional arguments.

        Returns:
            Population array of shape (n_samples, n_var) where n_var = n_pieces + 2.
        """
        n_heuristic = int(n_samples * self.HEURISTIC_RATIO)
        n_random = n_samples - n_heuristic

        heuristic_samples = self._generate_heuristic_population(n_heuristic)
        random_samples = self._generate_random_population(n_random)

        # Combine and add random starting positions
        pieces_only = np.vstack([heuristic_samples, random_samples])
        return self._add_random_positions(pieces_only)

    def _add_random_positions(self, pieces: NDArray) -> NDArray:
        """Add random starting positions to piece chromosomes.

        Args:
            pieces: Array of shape (n, n_piece_vars) with piece indices.

        Returns:
            Array of shape (n, n_piece_vars + 2) with random start positions.
        """
        n_samples = len(pieces)

        # Generate random starting positions within boundary
        start_x = np.random.uniform(
            self.config.boundary.min_x, self.config.boundary.max_x, size=n_samples
        )
        start_y = np.random.uniform(
            self.config.boundary.min_y, self.config.boundary.max_y, size=n_samples
        )

        # Concatenate pieces with starting positions
        return np.column_stack([pieces, start_x, start_y])

    def _generate_heuristic_population(self, n_samples: int) -> NDArray:
        """Generate heuristic chromosomes with valid patterns.

        Args:
            n_samples: Number of heuristic samples.

        Returns:
            Array of shape (n_samples, n_var).
        """
        patterns = []
        pattern_generators = [
            self._simple_circle,
            self._symmetric_oval,
            self._racetrack,
        ]

        for i in range(n_samples):
            # Cycle through pattern types
            generator = pattern_generators[i % len(pattern_generators)]
            pattern = generator()

            # Validate and add to population
            if pattern is not None:
                patterns.append(pattern)
            else:
                # Fallback to random if pattern invalid
                patterns.append(self._random_chromosome())

        return np.array(patterns, dtype=np.int32)

    def _generate_random_population(self, n_samples: int) -> NDArray:
        """Generate random chromosomes.

        Args:
            n_samples: Number of random samples.

        Returns:
            Array of shape (n_samples, n_var).
        """
        return np.array([self._random_chromosome() for _ in range(n_samples)], dtype=np.int32)

    def _simple_circle(self) -> Optional[NDArray]:
        """Generate simple circle pattern: 16 R40_LEFT pieces.

        Returns:
            Chromosome array or None if invalid.
        """
        pattern = [2] * 16  # 16x R40_LEFT forms 360 deg circle

        if not self._validate_inventory(pattern):
            return None

        return self._pattern_to_chromosome(pattern)

    def _symmetric_oval(self) -> Optional[NDArray]:
        """Generate symmetric oval: R40 curves + straights.

        Pattern: 4 R40_LEFT + 2 STRAIGHT_16 + 4 R40_LEFT + 2 STRAIGHT_16
        Forms oval with 180 deg at each end.

        Returns:
            Chromosome array or None if invalid.
        """
        pattern = (
            [2] * 4  # Left turn (90 deg)
            + [0] * 2  # Straight side
            + [2] * 4  # Left turn (90 deg)
            + [0] * 2  # Straight side
            + [2] * 4  # Left turn (90 deg)
            + [0] * 2  # Straight side
            + [2] * 4  # Left turn (90 deg)
            + [0] * 2  # Straight side
        )  # 16 R40_LEFT + 8 STRAIGHT_16

        if not self._validate_inventory(pattern):
            return None

        return self._pattern_to_chromosome(pattern)

    def _racetrack(self) -> Optional[NDArray]:
        """Generate racetrack pattern: 4 corner groups + long straights.

        Pattern: Long straight + 4 R40_LEFT + long straight + 4 R40_LEFT (x2)

        Returns:
            Chromosome array or None if invalid.
        """
        pattern = (
            [0] * 4  # Long straight
            + [2] * 4  # Corner 1 (90 deg)
            + [0] * 4  # Long straight
            + [2] * 4  # Corner 2 (90 deg)
            + [0] * 4  # Long straight
            + [2] * 4  # Corner 3 (90 deg)
            + [0] * 4  # Long straight
            + [2] * 4  # Corner 4 (90 deg)
        )  # 16 R40_LEFT + 16 STRAIGHT_16

        if not self._validate_inventory(pattern):
            return None

        return self._pattern_to_chromosome(pattern)

    def _pattern_to_chromosome(self, pattern: List[int]) -> NDArray:
        """Convert pattern to chromosome with empty slots.

        Args:
            pattern: List of piece indices.

        Returns:
            Chromosome array of length n_piece_vars (no position yet).
        """
        chromosome = np.full(self.n_piece_vars, -1, dtype=np.int32)
        chromosome[: len(pattern)] = pattern
        return chromosome

    def _random_chromosome(self) -> NDArray:
        """Generate random chromosome respecting inventory limits.

        Returns:
            Chromosome array of length n_piece_vars (no position yet).
        """
        chromosome = np.full(self.n_piece_vars, -1, dtype=np.int32)

        # Build list of available pieces using dynamic index_to_id
        available_pieces = []
        for piece_idx, piece_id in self.index_to_id.items():
            if piece_id in self.inventory:
                count = self.inventory[piece_id]
                available_pieces.extend([piece_idx] * count)

        # Shuffle and fill random number of pieces
        np.random.shuffle(available_pieces)
        n_pieces = np.random.randint(0, len(available_pieces) + 1)
        chromosome[:n_pieces] = available_pieces[:n_pieces]

        # Shuffle positions
        np.random.shuffle(chromosome)

        return chromosome

    def _validate_inventory(self, pattern: List[int]) -> bool:
        """Check if pattern satisfies inventory constraints.

        Args:
            pattern: List of piece indices.

        Returns:
            True if pattern is valid, False otherwise.
        """
        unique_indices, counts = np.unique(pattern, return_counts=True)
        usage = dict(zip(unique_indices, counts))

        for piece_idx, count in usage.items():
            # Use dynamic index_to_id from catalog
            piece_id = self.index_to_id.get(piece_idx)
            if piece_id is None:
                return False
            if piece_id not in self.inventory:
                return False
            if count > self.inventory[piece_id]:
                return False

        return True


# =============================================================================
# Class Aliases
# =============================================================================

# RandomKeySampling is the primary name for the RK encoding sampler
RandomKeySampling = MultiSegmentSampling
