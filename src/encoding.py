"""Partitioned chromosome encoding for track layout optimization.

Kia-inspired ingredient-based encoding with three segments:

  [Main Loop: N genes] [Junction Descriptors: J×4 genes] [Start Position: 2 genes]

- Ingredient 1 (Main Loop): piece type indices for the sequential track backbone
- Ingredient 2 (Junctions): metadata descriptors for passing sidings (switch pairs)
- Ingredient 3 (Start Position): layout offset within boundary

All dimensions scale dynamically from inventory configuration.
Branch content is template-determined (guaranteed closure by construction).
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# =============================================================================
# Constants
# =============================================================================

INACTIVE = -1
GENES_PER_JUNCTION = 4  # passing siding: (active, position, handedness, n_straights)
GENES_PER_CROSS_JUNCTION = 3  # cross junction: (active, position_W, handedness)

# Passing-siding junction gene offsets within a 4-gene descriptor
JUNC_ACTIVE = 0
JUNC_POSITION = 1
JUNC_HANDEDNESS = 2
JUNC_N_STRAIGHTS = 3

# Cross-junction descriptor gene offsets
CJ_ACTIVE = 0
CJ_POSITION_W = 1   # position of switch_for_W in main_loop_pieces
CJ_HANDEDNESS = 2   # 0 = LEFT switches, 1 = RIGHT switches


# =============================================================================
# Piece Index Constants (from track_pieces.yaml piece_index mapping)
# =============================================================================

class PieceIndex(IntEnum):
    """Track piece indices from catalog."""
    STRAIGHT_16 = 0
    STRAIGHT_24 = 1
    R40_LEFT = 2
    R40_RIGHT = 3
    CROSS_90 = 4
    SWITCH_LEFT = 5
    SWITCH_RIGHT = 6
    DOUBLE_CROSSOVER = 7


# Convenience aliases
STRAIGHT_16 = PieceIndex.STRAIGHT_16
STRAIGHT_24 = PieceIndex.STRAIGHT_24
R40_LEFT = PieceIndex.R40_LEFT
R40_RIGHT = PieceIndex.R40_RIGHT
CROSS_90 = PieceIndex.CROSS_90
SWITCH_LEFT = PieceIndex.SWITCH_LEFT
SWITCH_RIGHT = PieceIndex.SWITCH_RIGHT
DOUBLE_CROSSOVER = PieceIndex.DOUBLE_CROSSOVER

# Piece categories
SIMPLE_PIECE_INDICES = {STRAIGHT_16, STRAIGHT_24, R40_LEFT, R40_RIGHT}
SWITCH_INDICES = {SWITCH_LEFT, SWITCH_RIGHT}
CROSSING_INDICES = {CROSS_90, DOUBLE_CROSSOVER}

# Main loop piece types: simple pieces + crossings (no switches, no double crossover)
MAIN_LOOP_PIECE_INDICES = {STRAIGHT_16, STRAIGHT_24, R40_LEFT, R40_RIGHT}
MAX_MAIN_LOOP_PIECE = max(MAIN_LOOP_PIECE_INDICES)

# Switch ID set used by inventory computation (one entry per handedness now).
SWITCH_IDS = {"R40_SWITCH_LEFT", "R40_SWITCH_RIGHT"}


# =============================================================================
# Chromosome Dimensions (Dynamic, From Inventory)
# =============================================================================

@dataclass(frozen=True)
class PartitionedDimensions:
    """Chromosome layout dimensions, computed from inventory at runtime.

    Segments:
        [0, n_main)                                    — main loop piece types
        [n_main, junc_end)                             — passing-siding junction descriptors (J × 4)
        [junc_end, cross_junc_end)                     — cross-junction descriptors (K × 3)
        [cross_junc_end, cross_junc_end + 2)           — start position (x, y)
    """
    n_main: int              # Max main loop pieces (non-switch inventory)
    max_junctions: int       # Max passing-siding junction slots
    max_cross_junctions: int # Max cross-junction slots (from CROSS_90 inventory)
    total_straights: int     # Total straight pieces in inventory
    boundary_min_x: float
    boundary_max_x: float
    boundary_min_y: float
    boundary_max_y: float

    @property
    def junc_start(self) -> int:
        return self.n_main

    @property
    def junc_end(self) -> int:
        return self.n_main + self.max_junctions * GENES_PER_JUNCTION

    @property
    def cross_junc_start(self) -> int:
        return self.junc_end

    @property
    def cross_junc_end(self) -> int:
        return self.cross_junc_start + self.max_cross_junctions * GENES_PER_CROSS_JUNCTION

    @property
    def start_pos_start(self) -> int:
        return self.cross_junc_end

    @property
    def n_var(self) -> int:
        return self.cross_junc_end + 2


def compute_dimensions(config, catalog) -> PartitionedDimensions:
    """Compute chromosome dimensions from inventory configuration.

    Args:
        config: OptimizationConfig with inventory and boundary.
        catalog: TrackCatalog for piece_index lookup.

    Returns:
        PartitionedDimensions with all segment sizes derived from inventory.
    """
    inv = config.inventory

    # Main loop: all non-switch pieces
    n_main = 0
    for piece_id, count in inv.items():
        idx = catalog._id_to_index.get(piece_id)
        if idx is not None and idx not in SWITCH_INDICES:
            n_main += count
    n_main = max(1, n_main)

    # Each passing siding is opposite-handed: consumes 1 LEFT + 1 RIGHT switch.
    # So available passing-siding junctions = min of the two counts.
    left_count = inv.get("R40_SWITCH_LEFT", 0)
    right_count = inv.get("R40_SWITCH_RIGHT", 0)
    max_junctions = min(left_count, right_count)

    # Cross-junctions: each consumes 4 switches (all same handedness, LEFT or
    # RIGHT) + 4 R40 curves of opposite handedness + 1 CROSS_90.
    # Max cross junctions per handedness is floor(switch_count / 4) limited
    # also by CROSS_90 availability and corresponding curve count.
    cross_count = inv.get("CROSS_90", 0)
    r40_left_count = inv.get("R40_LEFT", 0)
    r40_right_count = inv.get("R40_RIGHT", 0)
    cross_junc_left = min(left_count // 4, r40_right_count // 4)   # LEFT-junction needs R40_RIGHT spurs
    cross_junc_right = min(right_count // 4, r40_left_count // 4)  # RIGHT-junction needs R40_LEFT spurs
    max_cross_junctions = min(cross_count, cross_junc_left + cross_junc_right)

    # Straights available for branches
    total_straights = inv.get("STRAIGHT_16", 0) + inv.get("STRAIGHT_24", 0)

    return PartitionedDimensions(
        n_main=n_main,
        max_junctions=max_junctions,
        max_cross_junctions=max_cross_junctions,
        total_straights=total_straights,
        boundary_min_x=config.boundary.min_x,
        boundary_max_x=config.boundary.max_x,
        boundary_min_y=config.boundary.min_y,
        boundary_max_y=config.boundary.max_y,
    )


# =============================================================================
# Bounds Generation
# =============================================================================

def generate_bounds(dims: PartitionedDimensions) -> Tuple[NDArray, NDArray]:
    """Generate per-gene lower and upper bounds.

    Returns:
        Tuple of (xl, xu) integer bound arrays of length n_var.
    """
    xl = np.full(dims.n_var, INACTIVE, dtype=np.int16)
    xu = np.full(dims.n_var, INACTIVE, dtype=np.int16)

    # Main loop: [-1, MAX_MAIN_LOOP_PIECE]
    xl[:dims.n_main] = INACTIVE
    xu[:dims.n_main] = MAX_MAIN_LOOP_PIECE

    # Junction descriptors (passing siding)
    for k in range(dims.max_junctions):
        base = dims.junc_start + k * GENES_PER_JUNCTION
        xl[base + JUNC_ACTIVE] = 0
        xu[base + JUNC_ACTIVE] = 1
        xl[base + JUNC_POSITION] = 0
        xu[base + JUNC_POSITION] = dims.n_main - 1
        xl[base + JUNC_HANDEDNESS] = 0
        xu[base + JUNC_HANDEDNESS] = 1  # LEFT, RIGHT
        xl[base + JUNC_N_STRAIGHTS] = 0
        xu[base + JUNC_N_STRAIGHTS] = dims.total_straights

    # Cross-junction descriptors
    for k in range(dims.max_cross_junctions):
        base = dims.cross_junc_start + k * GENES_PER_CROSS_JUNCTION
        xl[base + CJ_ACTIVE] = 0
        xu[base + CJ_ACTIVE] = 1
        xl[base + CJ_POSITION_W] = 0
        xu[base + CJ_POSITION_W] = dims.n_main - 1
        xl[base + CJ_HANDEDNESS] = 0
        xu[base + CJ_HANDEDNESS] = 1  # LEFT, RIGHT switch family

    # Start position
    xl[dims.start_pos_start] = int(dims.boundary_min_x)
    xu[dims.start_pos_start] = int(dims.boundary_max_x)
    xl[dims.start_pos_start + 1] = int(dims.boundary_min_y)
    xu[dims.start_pos_start + 1] = int(dims.boundary_max_y)

    return xl, xu


# =============================================================================
# Main Loop Gene Access
# =============================================================================

def get_main_loop_types(x: NDArray, dims: PartitionedDimensions) -> NDArray:
    """Read main loop piece types as array. Values: -1 (inactive) or 0-9."""
    return x[:dims.n_main].copy()


def set_main_loop_type(x: NDArray, dims: PartitionedDimensions,
                       pos: int, piece_type: int) -> None:
    """Set a single main loop piece type (in-place)."""
    x[pos] = piece_type


def get_active_main_pieces(x: NDArray, dims: PartitionedDimensions) -> NDArray:
    """Get main loop piece types, filtering out INACTIVE."""
    types = x[:dims.n_main]
    return types[types != INACTIVE].copy()


# =============================================================================
# Junction Gene Access
# =============================================================================

def get_junction(x: NDArray, dims: PartitionedDimensions,
                 slot: int) -> Tuple[int, int, int, int]:
    """Read a junction descriptor.

    Returns:
        (active, position, handedness, n_straights) tuple.
    """
    base = dims.junc_start + slot * GENES_PER_JUNCTION
    return (
        int(x[base + JUNC_ACTIVE]),
        int(x[base + JUNC_POSITION]),
        int(x[base + JUNC_HANDEDNESS]),
        int(x[base + JUNC_N_STRAIGHTS]),
    )


def set_junction(x: NDArray, dims: PartitionedDimensions, slot: int,
                 active: int, position: int, handedness: int,
                 n_straights: int) -> None:
    """Write a junction descriptor (in-place)."""
    base = dims.junc_start + slot * GENES_PER_JUNCTION
    x[base + JUNC_ACTIVE] = active
    x[base + JUNC_POSITION] = position
    x[base + JUNC_HANDEDNESS] = handedness
    x[base + JUNC_N_STRAIGHTS] = n_straights


def get_active_junctions(
    x: NDArray, dims: PartitionedDimensions,
) -> List[Tuple[int, int, int, int, int]]:
    """Get active junction descriptors, sorted by position.

    Returns:
        List of (slot, active, position, handedness, n_straights) tuples,
        sorted by position ascending.
    """
    junctions = []
    for k in range(dims.max_junctions):
        active, position, handedness, n_straights = get_junction(x, dims, k)
        if active:
            junctions.append((k, active, position, handedness, n_straights))
    junctions.sort(key=lambda j: j[2])  # Sort by position
    return junctions


# =============================================================================
# Cross-Junction Gene Access
# =============================================================================

def get_cross_junction(x: NDArray, dims: PartitionedDimensions,
                       slot: int) -> Tuple[int, int, int]:
    """Read a cross-junction descriptor.

    Returns:
        (active, position_W, handedness) tuple.
    """
    base = dims.cross_junc_start + slot * GENES_PER_CROSS_JUNCTION
    return (
        int(x[base + CJ_ACTIVE]),
        int(x[base + CJ_POSITION_W]),
        int(x[base + CJ_HANDEDNESS]),
    )


def set_cross_junction(x: NDArray, dims: PartitionedDimensions, slot: int,
                       active: int, position_W: int, handedness: int) -> None:
    """Write a cross-junction descriptor (in-place)."""
    base = dims.cross_junc_start + slot * GENES_PER_CROSS_JUNCTION
    x[base + CJ_ACTIVE] = active
    x[base + CJ_POSITION_W] = position_W
    x[base + CJ_HANDEDNESS] = handedness


def get_active_cross_junctions(
    x: NDArray, dims: PartitionedDimensions,
) -> List[Tuple[int, int, int, int]]:
    """Get active cross-junction descriptors, sorted by position_W.

    Returns:
        List of (slot, active, position_W, handedness) tuples.
    """
    junctions = []
    for k in range(dims.max_cross_junctions):
        active, position_W, handedness = get_cross_junction(x, dims, k)
        if active:
            junctions.append((k, active, position_W, handedness))
    junctions.sort(key=lambda j: j[2])  # sort by position_W
    return junctions


# =============================================================================
# Start Position Access
# =============================================================================

def get_start_position(x: NDArray, dims: PartitionedDimensions) -> Tuple[float, float]:
    """Read start position (x, y)."""
    return (float(x[dims.start_pos_start]), float(x[dims.start_pos_start + 1]))


# =============================================================================
# Chromosome Construction
# =============================================================================

def create_empty_chromosome(dims: PartitionedDimensions) -> NDArray:
    """Create a chromosome with all genes inactive/zero."""
    x = np.full(dims.n_var, INACTIVE, dtype=np.int16)
    # Zero out junction descriptors (active=0 by default)
    for k in range(dims.max_junctions):
        set_junction(x, dims, k, active=0, position=0, handedness=0, n_straights=0)
    # Zero start position
    x[dims.start_pos_start] = 0
    x[dims.start_pos_start + 1] = 0
    return x


def create_chromosome_from_pieces(
    dims: PartitionedDimensions,
    main_loop_pieces: List[int],
    junctions: Optional[List[Tuple[int, int, int, int]]] = None,
    cross_junctions: Optional[List[Tuple[int, int, int]]] = None,
) -> NDArray:
    """Create a chromosome from a piece sequence and optional descriptors.

    Args:
        dims: Partitioned dimensions.
        main_loop_pieces: Piece type indices for the main loop.
        junctions: Optional (active, position, handedness, n_straights) tuples
            for passing-siding descriptors.
        cross_junctions: Optional (active, position_W, handedness) tuples for
            cross-junction descriptors.

    Returns:
        Chromosome array.
    """
    x = create_empty_chromosome(dims)
    n = min(len(main_loop_pieces), dims.n_main)
    x[:n] = main_loop_pieces[:n]

    if junctions:
        for k, junc in enumerate(junctions):
            if k >= dims.max_junctions:
                break
            set_junction(x, dims, k, *junc)

    if cross_junctions:
        for k, cj in enumerate(cross_junctions):
            if k >= dims.max_cross_junctions:
                break
            set_cross_junction(x, dims, k, *cj)

    return x


# =============================================================================
# Validation
# =============================================================================

def validate_chromosome(x: NDArray, dims: PartitionedDimensions) -> List[str]:
    """Validate chromosome gene values are within bounds.

    Returns:
        List of error messages (empty if valid).
    """
    errors = []

    if len(x) != dims.n_var:
        errors.append(f"Length {len(x)} != expected {dims.n_var}")
        return errors

    # Main loop bounds
    for i in range(dims.n_main):
        val = int(x[i])
        if val != INACTIVE and (val < 0 or val > MAX_MAIN_LOOP_PIECE):
            errors.append(f"Main loop [{i}]: {val} out of range [-1, {MAX_MAIN_LOOP_PIECE}]")

    # Junction bounds
    for k in range(dims.max_junctions):
        active, pos, hand, n_str = get_junction(x, dims, k)
        if active not in (0, 1):
            errors.append(f"Junction {k}: active={active} not 0 or 1")
        if pos < 0 or pos >= dims.n_main:
            errors.append(f"Junction {k}: position={pos} out of [0, {dims.n_main - 1}]")
        if hand < 0 or hand > 1:
            errors.append(f"Junction {k}: handedness={hand} out of [0, 3]")
        if n_str < 0 or n_str > dims.total_straights:
            errors.append(f"Junction {k}: n_straights={n_str} out of [0, {dims.total_straights}]")

    return errors


# =============================================================================
# Chromosome Statistics
# =============================================================================

def chromosome_stats(x: NDArray, dims: PartitionedDimensions) -> Dict:
    """Compute summary statistics for a chromosome."""
    main_types = get_main_loop_types(x, dims)
    active_main = main_types[main_types != INACTIVE]

    piece_counts: Dict[int, int] = {}
    for pt in active_main:
        pt_int = int(pt)
        piece_counts[pt_int] = piece_counts.get(pt_int, 0) + 1

    active_juncs = get_active_junctions(x, dims)

    return {
        "n_main_genes": dims.n_main,
        "n_active_main": len(active_main),
        "n_junctions": dims.max_junctions,
        "n_active_junctions": len(active_juncs),
        "piece_counts": piece_counts,
        "n_var": dims.n_var,
    }
