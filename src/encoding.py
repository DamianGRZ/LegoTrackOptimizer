"""Random-key chromosome encoding for track layout optimization.

Research-based encoding using Bean's random-key approach:
- All gene values in [0.0, 1.0]
- Deterministic decoder maps genes to discrete piece selections
- Standard pymoo operators (SBX, PM) work without modification
- Smoother fitness landscape for better convergence

Chromosome structure (fixed-length array, N_VAR=218):
- Piece selection keys:  [0, 100)    - RK values for piece type selection
- Port priority keys:    [100, 200)  - RK values for construction order
- Branch template keys:  [200, 216)  - 4 slots × 4 RK genes each
- Start position keys:   [216, 218)  - RK values scaled to boundary

Branch slot layout: [in_pos_key, handedness_key, n_straights_key, active_key]
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# =============================================================================
# Segment Size Constants
# =============================================================================

L_MAX = 100      # Piece selection positions
S_MAX = 100      # Port priority positions (legacy alias: switch mask)
P_MAX = 100      # Port priority positions
B_MAX = 4        # Number of branch slots
B_SLOT = 4       # Genes per branch slot: [in_pos, handedness, n_straights, active]
C_MAX = 0        # Crossing overlay disabled

# Computed dimensions
N_MAIN_LOOP = L_MAX
N_SWITCH_MASK = S_MAX
N_PIECE_KEYS = L_MAX
N_PRIORITY_KEYS = P_MAX
N_BRANCH_GENES = B_MAX * B_SLOT  # 16 genes
N_CROSSING_GENES = C_MAX * 2      # 0
N_PIECE_GENES = N_PIECE_KEYS + N_PRIORITY_KEYS + N_BRANCH_GENES + N_CROSSING_GENES  # 216
N_POSITION_GENES = 2  # start_x, start_y
N_VAR = N_PIECE_GENES + N_POSITION_GENES  # 218

# Segment boundaries (start indices)
MAIN_LOOP_START = 0
MAIN_LOOP_END = L_MAX
SWITCH_MASK_START = L_MAX
SWITCH_MASK_END = L_MAX + S_MAX
PRIORITY_START = L_MAX
PRIORITY_END = L_MAX + P_MAX
BRANCH_SLOTS_START = PRIORITY_END
BRANCH_SLOTS_END = BRANCH_SLOTS_START + N_BRANCH_GENES
BRANCH_START = BRANCH_SLOTS_START
BRANCH_END = BRANCH_SLOTS_END
CROSSING_START = BRANCH_SLOTS_END
CROSSING_END = N_PIECE_GENES
POSITION_START = N_PIECE_GENES
POSITION_END = N_VAR

# Random-key bounds (all [0, 1])
XL = np.zeros(N_VAR, dtype=np.float64)
XU = np.ones(N_VAR, dtype=np.float64)

# Sentinel value for inactive decoded genes (used in phenotype, not genotype)
INACTIVE = -1

# RK encoding uses 0.0 to mark inactive slots (decoder skips these)
RK_INACTIVE = 0.0
RK_INACTIVE_THRESHOLD = 0.001  # Values below this are treated as inactive


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
    SWITCH_LEFT_IN = 5
    SWITCH_LEFT_OUT = 6
    SWITCH_RIGHT_IN = 7
    SWITCH_RIGHT_OUT = 8
    DOUBLE_CROSSOVER = 9


# Legacy constants for backward compatibility
STRAIGHT_16 = PieceIndex.STRAIGHT_16
STRAIGHT_24 = PieceIndex.STRAIGHT_24
R40_LEFT = PieceIndex.R40_LEFT
R40_RIGHT = PieceIndex.R40_RIGHT
CROSS_90 = PieceIndex.CROSS_90
SWITCH_LEFT_IN = PieceIndex.SWITCH_LEFT_IN
SWITCH_LEFT_OUT = PieceIndex.SWITCH_LEFT_OUT
SWITCH_RIGHT_IN = PieceIndex.SWITCH_RIGHT_IN
SWITCH_RIGHT_OUT = PieceIndex.SWITCH_RIGHT_OUT
DOUBLE_CROSSOVER = PieceIndex.DOUBLE_CROSSOVER

# Piece categories
SIMPLE_PIECE_INDICES = {STRAIGHT_16, STRAIGHT_24, R40_LEFT, R40_RIGHT}
COMPLEX_PIECE_INDICES = {CROSS_90, SWITCH_LEFT_IN, SWITCH_LEFT_OUT,
                         SWITCH_RIGHT_IN, SWITCH_RIGHT_OUT, DOUBLE_CROSSOVER}
SWITCH_INDICES = {SWITCH_LEFT_IN, SWITCH_LEFT_OUT, SWITCH_RIGHT_IN, SWITCH_RIGHT_OUT}

# Switch pairs for balancing: (IN_index, OUT_index)
SWITCH_PAIRS = [
    (SWITCH_LEFT_IN, SWITCH_LEFT_OUT),
    (SWITCH_RIGHT_IN, SWITCH_RIGHT_OUT),
]


# =============================================================================
# Random-Key Mapping Functions
# =============================================================================

def rk_to_piece_index(rk_value: float, available_pieces: List[int]) -> int:
    """Map [0,1] value to available piece index.

    Uses Bean's random-key approach: gene value selects from
    dynamically-updated available piece list.

    Args:
        rk_value: Random-key value in [0, 1].
        available_pieces: List of available piece indices.

    Returns:
        Selected piece index, or INACTIVE if no pieces available.
    """
    if not available_pieces:
        return INACTIVE
    idx = int(rk_value * len(available_pieces))
    idx = min(idx, len(available_pieces) - 1)
    return available_pieces[idx]


def rk_to_branch_params(rk_values: NDArray) -> Tuple[int, int, int, bool]:
    """Map 4 RK values to branch template parameters.

    Args:
        rk_values: Array of 4 random-key values in [0, 1].

    Returns:
        Tuple of (in_pos, handedness, n_straights, active).
        - in_pos: Main loop position [0, L_MAX-1]
        - handedness: 0=LEFT, 1=RIGHT
        - n_straights: Number of straights [0, 8]
        - active: True if branch should be used
    """
    in_pos = int(rk_values[0] * L_MAX)
    in_pos = min(in_pos, L_MAX - 1)
    # Handedness: 0=LEFT, 1=RIGHT, 2=LEFT_REVERSE, 3=RIGHT_REVERSE
    handedness = int(rk_values[1] * 4)
    handedness = min(handedness, 3)
    n_straights = int(rk_values[2] * 9)
    n_straights = min(n_straights, 8)
    active = rk_values[3] >= 0.5
    return (in_pos, handedness, n_straights, active)


def rk_to_position(rk_x: float, rk_y: float, boundary) -> Tuple[float, float]:
    """Map [0,1] position values to boundary coordinates.

    Args:
        rk_x: Random-key value for X position.
        rk_y: Random-key value for Y position.
        boundary: BoundaryConfig with min/max coordinates.

    Returns:
        Tuple of (x, y) world coordinates.
    """
    x = boundary.min_x + rk_x * (boundary.max_x - boundary.min_x)
    y = boundary.min_y + rk_y * (boundary.max_y - boundary.min_y)
    return (x, y)


def piece_index_to_rk(piece_idx: int, available_pieces: List[int]) -> float:
    """Convert piece index to random-key value (inverse mapping).

    Used to encode known patterns as RK chromosomes.

    Args:
        piece_idx: Piece index to encode.
        available_pieces: List of available piece indices.

    Returns:
        Random-key value that would decode to this piece.
    """
    if not available_pieces or piece_idx not in available_pieces:
        return 0.5  # Default to middle of range
    bucket_idx = available_pieces.index(piece_idx)
    n_buckets = len(available_pieces)
    # Return center of bucket
    return (bucket_idx + 0.5) / n_buckets


def position_to_rk(x: float, y: float, boundary) -> Tuple[float, float]:
    """Convert world coordinates to random-key values (inverse mapping).

    Args:
        x: World X coordinate.
        y: World Y coordinate.
        boundary: BoundaryConfig with min/max coordinates.

    Returns:
        Tuple of (rk_x, rk_y) in [0, 1].
    """
    width = boundary.max_x - boundary.min_x
    height = boundary.max_y - boundary.min_y
    rk_x = (x - boundary.min_x) / width if width > 0 else 0.5
    rk_y = (y - boundary.min_y) / height if height > 0 else 0.5
    return (np.clip(rk_x, 0, 1), np.clip(rk_y, 0, 1))


# =============================================================================
# Segment Slicing Functions
# =============================================================================

def get_piece_keys(x: NDArray) -> NDArray:
    """Extract piece selection keys from chromosome.

    Args:
        x: Full chromosome array of length N_VAR.

    Returns:
        Piece selection keys of length L_MAX.
    """
    return x[MAIN_LOOP_START:MAIN_LOOP_END]


def get_priority_keys(x: NDArray) -> NDArray:
    """Extract port priority keys from chromosome.

    Args:
        x: Full chromosome array of length N_VAR.

    Returns:
        Priority keys of length P_MAX.
    """
    return x[PRIORITY_START:PRIORITY_END]


def get_branch_keys(x: NDArray) -> NDArray:
    """Extract branch template keys from chromosome.

    Args:
        x: Full chromosome array of length N_VAR.

    Returns:
        Branch keys of length B_MAX * B_SLOT.
    """
    return x[BRANCH_START:BRANCH_END]


def get_branch_slot_keys(x: NDArray, slot_idx: int) -> NDArray:
    """Extract a single branch slot's keys from chromosome.

    Args:
        x: Full chromosome array of length N_VAR.
        slot_idx: Branch slot index (0 to B_MAX-1).

    Returns:
        Single branch slot keys of length B_SLOT.
    """
    if slot_idx < 0 or slot_idx >= B_MAX:
        raise ValueError(f"slot_idx must be 0 to {B_MAX-1}, got {slot_idx}")
    start = BRANCH_START + slot_idx * B_SLOT
    return x[start:start + B_SLOT]


def get_position_keys(x: NDArray) -> Tuple[float, float]:
    """Extract position keys from chromosome.

    Args:
        x: Full chromosome array of length N_VAR.

    Returns:
        Tuple of (rk_x, rk_y) position keys.
    """
    return float(x[POSITION_START]), float(x[POSITION_START + 1])


# =============================================================================
# Legacy Aliases for Backward Compatibility
# =============================================================================

def get_main_loop(x: NDArray) -> NDArray:
    """Extract main loop segment (piece selection keys).

    Args:
        x: Full chromosome array of length N_VAR.

    Returns:
        Main loop segment of length L_MAX.
    """
    return x[MAIN_LOOP_START:MAIN_LOOP_END]


def get_switch_mask(x: NDArray) -> NDArray:
    """Extract switch mask segment (priority keys).

    Args:
        x: Full chromosome array of length N_VAR.

    Returns:
        Switch mask segment of length S_MAX.
    """
    return x[SWITCH_MASK_START:SWITCH_MASK_END]


def get_branch_slots(x: NDArray) -> NDArray:
    """Extract branch slots segment.

    Args:
        x: Full chromosome array of length N_VAR.

    Returns:
        Branch slots segment of length B_MAX * B_SLOT.
    """
    return x[BRANCH_SLOTS_START:BRANCH_SLOTS_END]


def get_branch_slot(x: NDArray, slot_idx: int) -> NDArray:
    """Extract a single branch slot.

    Args:
        x: Full chromosome array of length N_VAR.
        slot_idx: Branch slot index (0 to B_MAX-1).

    Returns:
        Single branch slot of length B_SLOT.
    """
    if slot_idx < 0 or slot_idx >= B_MAX:
        raise ValueError(f"slot_idx must be 0 to {B_MAX-1}, got {slot_idx}")
    start = BRANCH_SLOTS_START + slot_idx * B_SLOT
    return x[start:start + B_SLOT]


def get_crossing_overlay(x: NDArray) -> NDArray:
    """Extract crossing overlay segment (disabled).

    Args:
        x: Full chromosome array of length N_VAR.

    Returns:
        Empty array (crossing overlay disabled).
    """
    return x[CROSSING_START:CROSSING_END]


def get_start_position(x: NDArray) -> Tuple[float, float]:
    """Extract starting position from chromosome.

    In RK encoding, these are [0,1] keys.

    Args:
        x: Full chromosome array of length N_VAR.

    Returns:
        Tuple of (rk_x, rk_y).
    """
    return float(x[POSITION_START]), float(x[POSITION_START + 1])


def set_main_loop(x: NDArray, values: NDArray) -> None:
    """Set main loop segment in chromosome (in-place).

    Args:
        x: Full chromosome array of length N_VAR.
        values: Values to set (length <= L_MAX), should be [0,1].
    """
    n = min(len(values), L_MAX)
    x[MAIN_LOOP_START:MAIN_LOOP_START + n] = np.clip(values[:n], 0, 1)


def set_start_position(x: NDArray, start_x: float, start_y: float) -> None:
    """Set starting position in chromosome (in-place).

    Args:
        x: Full chromosome array of length N_VAR.
        start_x: Starting X position key [0,1].
        start_y: Starting Y position key [0,1].
    """
    x[POSITION_START] = np.clip(start_x, 0, 1)
    x[POSITION_START + 1] = np.clip(start_y, 0, 1)


# =============================================================================
# Chromosome Construction
# =============================================================================

def create_empty_chromosome(dtype=np.float64) -> NDArray:
    """Create a chromosome with minimal active genes.

    Main loop genes start at 0.0 (inactive) - ComplexificationMutation
    will gradually activate them, creating NEAT-style minimal-to-complex
    evolution. Other segments (priority, branch, position) are random.

    Args:
        dtype: Data type for the array.

    Returns:
        Chromosome array of length N_VAR with main loop inactive.
    """
    x = np.zeros(N_VAR, dtype=dtype)

    # Main loop genes [0, 100) stay at 0.0 (inactive)
    # Priority, branch, and position genes are random
    x[PRIORITY_START:PRIORITY_END] = np.random.uniform(0, 1, P_MAX)
    x[BRANCH_START:BRANCH_END] = np.random.uniform(0, 1, N_BRANCH_GENES)
    x[POSITION_START:POSITION_END] = np.random.uniform(0, 1, N_POSITION_GENES)

    return x


def create_chromosome_from_pattern(
    piece_indices: List[int],
    available_pieces: List[int],
    start_rk_x: float = 0.5,
    start_rk_y: float = 0.5,
    dtype=np.float64,
) -> NDArray:
    """Create chromosome from piece index pattern.

    Converts integer piece pattern to RK chromosome. Unused main loop slots
    are set to 0.0 (inactive marker) so decoder skips them.

    Args:
        piece_indices: List of piece indices to encode.
        available_pieces: List of available piece indices for RK mapping.
        start_rk_x: Starting X position key [0,1].
        start_rk_y: Starting Y position key [0,1].
        dtype: Data type for the array.

    Returns:
        Full chromosome array of length N_VAR.
    """
    x = np.random.uniform(0, 1, N_VAR).astype(dtype)

    # Set ALL main loop slots to inactive (0.0) first
    x[MAIN_LOOP_START:MAIN_LOOP_END] = RK_INACTIVE

    # Encode piece selection keys for the pattern
    for i, piece_idx in enumerate(piece_indices):
        if i >= L_MAX:
            break
        x[MAIN_LOOP_START + i] = piece_index_to_rk(piece_idx, available_pieces)

    # Set position keys
    x[POSITION_START] = start_rk_x
    x[POSITION_START + 1] = start_rk_y

    return x


def create_chromosome_from_main_loop(
    main_loop: NDArray,
    start_x: float = 0.0,
    start_y: float = 0.0,
    dtype=np.float64,
) -> NDArray:
    """Create chromosome from main loop values.

    Legacy function - interprets inputs as RK values if <= 1,
    otherwise attempts conversion.

    Args:
        main_loop: Array of values for main loop.
        start_x: Starting X value.
        start_y: Starting Y value.
        dtype: Data type for the array.

    Returns:
        Full chromosome array of length N_VAR.
    """
    x = np.random.uniform(0, 1, N_VAR).astype(dtype)
    n = min(len(main_loop), L_MAX)

    # If values are > 1, assume they're piece indices and scale
    if np.any(main_loop > 1):
        x[MAIN_LOOP_START:MAIN_LOOP_START + n] = np.clip(main_loop[:n] / 10.0, 0, 1)
    else:
        x[MAIN_LOOP_START:MAIN_LOOP_START + n] = np.clip(main_loop[:n], 0, 1)

    x[POSITION_START] = np.clip(start_x, 0, 1) if start_x <= 1 else 0.5
    x[POSITION_START + 1] = np.clip(start_y, 0, 1) if start_y <= 1 else 0.5
    return x


# =============================================================================
# Bounds Generation
# =============================================================================

@dataclass
class SegmentBounds:
    """Bounds for chromosome segments."""
    xl: NDArray  # Lower bounds
    xu: NDArray  # Upper bounds


def generate_bounds(
    max_piece_index: int = 9,
    boundary_min_x: float = -100.0,
    boundary_max_x: float = 100.0,
    boundary_min_y: float = -100.0,
    boundary_max_y: float = 100.0,
) -> SegmentBounds:
    """Generate segment-specific lower and upper bounds.

    In random-key encoding, all bounds are [0, 1].

    Args:
        max_piece_index: Ignored in RK encoding (kept for API compatibility).
        boundary_*: Ignored in RK encoding (kept for API compatibility).

    Returns:
        SegmentBounds with xl=0 and xu=1 for all genes.
    """
    xl = np.zeros(N_VAR, dtype=np.float64)
    xu = np.ones(N_VAR, dtype=np.float64)
    return SegmentBounds(xl=xl, xu=xu)


# =============================================================================
# Validation
# =============================================================================

def validate_chromosome(x: NDArray, max_piece_index: int = 9) -> bool:
    """Validate chromosome structure.

    Args:
        x: Chromosome array.
        max_piece_index: Ignored in RK encoding.

    Returns:
        True if valid, False otherwise.
    """
    if len(x) != N_VAR:
        return False
    # All values should be in [0, 1]
    return bool(np.all((x >= 0) & (x <= 1)))


def count_active_genes(x: NDArray) -> Dict[str, int]:
    """Count 'active' genes in each segment.

    In RK encoding, all genes are potentially active.

    Args:
        x: Chromosome array.

    Returns:
        Dict with counts per segment.
    """
    return {
        'main_loop': L_MAX,
        'switch_mask': S_MAX,
        'branch_slots': N_BRANCH_GENES,
        'crossing_overlay': 0,
    }


def get_active_main_loop(x: NDArray) -> NDArray:
    """Get piece selection keys (all are 'active' in RK encoding).

    Args:
        x: Chromosome array.

    Returns:
        Array of piece selection keys.
    """
    return get_main_loop(x)


# =============================================================================
# Branch Slot Helpers
# =============================================================================

def get_branch_template_params(x: NDArray, slot_idx: int) -> Tuple[int, int, int, bool]:
    """Get decoded template parameters for a branch slot.

    Args:
        x: Chromosome array.
        slot_idx: Branch slot index (0 to B_MAX-1).

    Returns:
        Tuple of (in_pos, handedness, n_straights, active).
    """
    slot = get_branch_slot(x, slot_idx)
    return rk_to_branch_params(slot)


def set_branch_template_params(
    x: NDArray,
    slot_idx: int,
    in_pos: int,
    handedness: int,
    n_straights: int,
    active: int,
) -> None:
    """Set template parameters for a branch slot using RK values.

    Args:
        x: Chromosome array (modified in-place).
        slot_idx: Branch slot index (0 to B_MAX-1).
        in_pos: Main loop position for IN switch.
        handedness: 0=LEFT, 1=RIGHT.
        n_straights: Number of straights (0-8).
        active: 1 if enabled, 0 if disabled.
    """
    if slot_idx < 0 or slot_idx >= B_MAX:
        raise ValueError(f"slot_idx must be 0 to {B_MAX-1}, got {slot_idx}")
    start = BRANCH_SLOTS_START + slot_idx * B_SLOT

    # Convert integer params to RK values
    x[start] = (in_pos + 0.5) / L_MAX if in_pos >= 0 else 0.0
    x[start + 1] = 0.75 if handedness else 0.25
    x[start + 2] = (n_straights + 0.5) / 9.0
    x[start + 3] = 0.75 if active else 0.25


def set_branch_slot(x: NDArray, slot_idx: int, values: NDArray) -> None:
    """Set a single branch slot in chromosome (in-place).

    Args:
        x: Full chromosome array of length N_VAR.
        slot_idx: Branch slot index (0 to B_MAX-1).
        values: Values to set (length <= B_SLOT).
    """
    if slot_idx < 0 or slot_idx >= B_MAX:
        raise ValueError(f"slot_idx must be 0 to {B_MAX-1}, got {slot_idx}")
    start = BRANCH_SLOTS_START + slot_idx * B_SLOT
    n = min(len(values), B_SLOT)
    x[start:start + n] = np.clip(values[:n], 0, 1)


def get_branch_in_position(x: NDArray, slot_idx: int) -> int:
    """Get IN switch position for a branch slot.

    Args:
        x: Chromosome array.
        slot_idx: Branch slot index (0 to B_MAX-1).

    Returns:
        IN switch position in main loop.
    """
    in_pos, _, _, _ = get_branch_template_params(x, slot_idx)
    return in_pos


def get_branch_handedness(x: NDArray, slot_idx: int) -> int:
    """Get branch handedness (LEFT=0, RIGHT=1).

    Args:
        x: Chromosome array.
        slot_idx: Branch slot index (0 to B_MAX-1).

    Returns:
        0 for LEFT, 1 for RIGHT.
    """
    _, handedness, _, _ = get_branch_template_params(x, slot_idx)
    return handedness


def get_branch_n_straights(x: NDArray, slot_idx: int) -> int:
    """Get number of straight pieces in parallel section.

    Args:
        x: Chromosome array.
        slot_idx: Branch slot index (0 to B_MAX-1).

    Returns:
        Number of straights (0-8).
    """
    _, _, n_straights, _ = get_branch_template_params(x, slot_idx)
    return n_straights


def get_branch_active_flag(x: NDArray, slot_idx: int) -> bool:
    """Get branch active flag.

    Args:
        x: Chromosome array.
        slot_idx: Branch slot index (0 to B_MAX-1).

    Returns:
        True if active flag is set.
    """
    _, _, _, active = get_branch_template_params(x, slot_idx)
    return active


def is_branch_active(x: NDArray, slot_idx: int) -> bool:
    """Check if a branch slot is active.

    Args:
        x: Chromosome array.
        slot_idx: Branch slot index (0 to B_MAX-1).

    Returns:
        True if branch is active.
    """
    return get_branch_active_flag(x, slot_idx)


def is_branch_valid(x: NDArray, slot_idx: int) -> bool:
    """Check if a branch slot defines a valid template-based branch.

    Args:
        x: Chromosome array.
        slot_idx: Branch slot index (0 to B_MAX-1).

    Returns:
        True if branch slot is potentially valid.
    """
    return is_branch_active(x, slot_idx)


def find_active_branch_slots(x: NDArray) -> List[int]:
    """Find all active branch slot indices.

    Args:
        x: Chromosome array.

    Returns:
        List of active branch slot indices.
    """
    return [i for i in range(B_MAX) if is_branch_active(x, i)]


def find_inactive_branch_slots(x: NDArray) -> List[int]:
    """Find all inactive branch slot indices.

    Args:
        x: Chromosome array.

    Returns:
        List of inactive branch slot indices.
    """
    return [i for i in range(B_MAX) if not is_branch_active(x, i)]


# Legacy aliases
def get_branch_src_switch(x: NDArray, slot_idx: int) -> int:
    """Legacy alias for get_branch_in_position."""
    return get_branch_in_position(x, slot_idx)


def get_branch_out_position(x: NDArray, slot_idx: int) -> int:
    """DEPRECATED: OUT position computed from geometry."""
    return INACTIVE


def get_branch_rejoin_target(x: NDArray, slot_idx: int) -> int:
    """Legacy alias for get_branch_out_position."""
    return INACTIVE


def get_branch_pieces(x: NDArray, slot_idx: int) -> NDArray:
    """DEPRECATED: Branch pieces computed from template."""
    return np.array([], dtype=np.int32)


# =============================================================================
# Switch/Priority Helpers
# =============================================================================

def set_switch_mask_value(x: NDArray, position: int, value: float) -> None:
    """Set priority key at a position.

    Args:
        x: Chromosome array.
        position: Position in priority segment (0 to S_MAX-1).
        value: Priority value [0, 1].
    """
    if 0 <= position < S_MAX:
        x[SWITCH_MASK_START + position] = np.clip(value, 0, 1)


def get_switch_mask_value(x: NDArray, position: int) -> float:
    """Get priority key at a position.

    Args:
        x: Chromosome array.
        position: Position in priority segment (0 to S_MAX-1).

    Returns:
        Priority value [0, 1].
    """
    if 0 <= position < S_MAX:
        return float(x[SWITCH_MASK_START + position])
    return 0.5


# =============================================================================
# Crossing Overlay (disabled)
# =============================================================================

def get_crossing_pair(x: NDArray, pair_idx: int) -> Tuple[int, int]:
    """Get crossing pair (disabled).

    Args:
        x: Chromosome array.
        pair_idx: Crossing pair index.

    Returns:
        Tuple of (INACTIVE, INACTIVE).
    """
    return (INACTIVE, INACTIVE)


def set_crossing_pair(x: NDArray, pair_idx: int, pos1: int, pos2: int) -> None:
    """Set crossing pair (disabled - no-op).

    Args:
        x: Chromosome array.
        pair_idx: Crossing pair index.
        pos1: First position.
        pos2: Second position.
    """
    pass


def is_crossing_active(x: NDArray, pair_idx: int) -> bool:
    """Check if crossing is active (always False - disabled).

    Args:
        x: Chromosome array.
        pair_idx: Crossing pair index.

    Returns:
        False (crossing disabled).
    """
    return False
