"""Sampling strategies for generating initial population.

Provides two sampling classes:
- HeuristicSampling: Legacy sampling for flat chromosome encoding
- MultiSegmentSampling: New sampling for multi-segment encoding (Phase 1)
"""

from typing import Dict, List, Optional

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
    POSITION_START,
    create_chromosome_from_main_loop,
    create_empty_chromosome,
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
        self.config = config
        self.inventory = config.inventory

        # Use dynamic index_to_id from catalog (auto-adapts to new pieces)
        self.index_to_id = catalog.index_to_id

        # Build valid piece list from inventory
        self.valid_pieces = []
        for piece_id, count in self.inventory.items():
            idx = catalog._id_to_index.get(piece_id)
            if idx is not None and count > 0:
                self.valid_pieces.append(idx)

        # Inventory by index for validation
        self.inventory_by_index = {}
        for piece_id, count in self.inventory.items():
            idx = catalog._id_to_index.get(piece_id)
            if idx is not None:
                self.inventory_by_index[idx] = count

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

        Args:
            population: Population array of shape (n, N_VAR).
        """
        n_samples = len(population)

        # Generate random starting positions within boundary
        start_x = np.random.uniform(
            self.config.boundary.min_x,
            self.config.boundary.max_x,
            size=n_samples,
        )
        start_y = np.random.uniform(
            self.config.boundary.min_y,
            self.config.boundary.max_y,
            size=n_samples,
        )

        # Set position genes
        population[:, POSITION_START] = start_x
        population[:, POSITION_START + 1] = start_y

    def _generate_heuristic_population(self, n_samples: int) -> NDArray:
        """Generate heuristic chromosomes with valid patterns.

        Args:
            n_samples: Number of heuristic samples.

        Returns:
            Array of shape (n_samples, N_VAR).
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
                population[i] = create_chromosome_from_main_loop(pattern)
            else:
                population[i] = self._random_chromosome()

        return population

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
        """Generate oval with left-hand passing siding with actual switches in main loop.

        Places actual switch pieces in the main loop:
        - LEFT_IN (5) at position 4
        - LEFT_OUT (6) at position 7 (3 straights apart for branch geometry)
        - Decoder will match these and compute connecting branch pieces

        Returns:
            Full chromosome with main loop containing switches, or None if invalid.
        """
        # Main loop: oval with LEFT_IN and LEFT_OUT switches placed
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

        # Check main loop inventory
        if not self._validate_inventory(main_loop):
            return None

        # Check if we have curves for the branch (R40_RIGHT to go parallel, R40_LEFT to return)
        r40_right_available = self.inventory_by_index.get(3, 0) > 0
        r40_left_available = self.inventory_by_index.get(2, 0) > 0

        if not all([r40_right_available, r40_left_available]):
            return None

        # Create chromosome with main loop
        x = create_chromosome_from_main_loop(main_loop)

        return x

    def _oval_with_right_siding(self) -> Optional[NDArray]:
        """Generate oval with right-hand passing siding with actual switches in main loop.

        Places actual switch pieces in the main loop:
        - RIGHT_IN (7) at position 4
        - RIGHT_OUT (8) at position 7 (3 straights apart for branch geometry)
        - Decoder will match these and compute connecting branch pieces

        Returns:
            Full chromosome with main loop containing switches, or None if invalid.
        """
        # Main loop: clockwise oval with RIGHT_IN and RIGHT_OUT switches placed
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

        # Check main loop inventory
        if not self._validate_inventory(main_loop):
            return None

        # Check if we have curves for the branch (R40_LEFT to go parallel, R40_RIGHT to return)
        r40_left_available = self.inventory_by_index.get(2, 0) > 0
        r40_right_available = self.inventory_by_index.get(3, 0) > 0

        if not all([r40_left_available, r40_right_available]):
            return None

        # Create chromosome with main loop
        x = create_chromosome_from_main_loop(main_loop)

        return x

    def _random_chromosome(self) -> NDArray:
        """Generate random chromosome respecting inventory limits.

        Also generates switch mask and branch slot genes with 30% probability.
        """
        x = create_empty_chromosome()

        if not self.valid_pieces:
            return x

        # Build pool of available pieces
        available_pieces = []
        for piece_idx, count in self.inventory_by_index.items():
            available_pieces.extend([piece_idx] * count)

        if not available_pieces:
            return x

        # Shuffle and select random number of pieces
        np.random.shuffle(available_pieces)
        n_pieces = np.random.randint(4, min(len(available_pieces), L_MAX) + 1)
        pieces = available_pieces[:n_pieces]

        # Fill main loop
        x[:len(pieces)] = pieces

        # 30% chance to add switch and branch genes
        if np.random.random() < 0.30:
            self._add_switch_branch_genes(x, pieces, available_pieces[n_pieces:])

        return x

    def _add_switch_branch_genes(
        self,
        x: NDArray,
        main_loop_pieces: List[int],
        remaining_pieces: List[int],
    ) -> None:
        """Add template-based branch slots to chromosome.

        Template-based branch encoding:
        - Find STRAIGHT_16 positions that can become switch IN positions
        - Set branch template params: [IN_pos, handedness, n_straights, active]
        - Decoder computes branch pieces and OUT position from geometry

        Branch slot format: [IN_pos, handedness (0=LEFT, 1=RIGHT), n_straights, active]
        """
        # Find positions with STRAIGHT_16 (index 0) that could be replaced with switches
        straight_positions = [i for i, p in enumerate(main_loop_pieces) if p == 0]

        # Need at least 3 consecutive straights for a passing siding
        # (IN switch, parallel section, OUT switch)
        if len(straight_positions) < 3:
            return

        # Check if switches are available (check both LEFT and RIGHT)
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

        # Pick a random IN position (must have room for siding after it)
        valid_in_positions = [p for p in straight_positions if p < len(main_loop_pieces) - 4]
        if not valid_in_positions:
            return

        in_pos = np.random.choice(valid_in_positions)

        # Random number of straights in parallel section (0-3)
        n_straights = np.random.randint(0, 4)

        # Set branch template params
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
