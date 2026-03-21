"""Multi-segment chromosome encoding for track layout optimization.

Chromosome structure (fixed-length array, N_VAR=282):
- Main loop:       [0, 100)       - piece indices for construction sequence (L_MAX=100)
- Switch mask:     [100, 200)     - switch type at main-loop position (S_MAX=100)
- Branch slots:    [200, 280)     - B_MAX=4 slots of B_SLOT=20 genes each
- Start position:  [280, 282)     - start_x, start_y coordinates

Branch slot layout: [IN_pos, OUT_pos, branch_pieces×18]

Inactive genes use sentinel value -1.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray


# =============================================================================
# Segment Size Constants (simplified template-based encoding)
# =============================================================================

L_MAX = 100      # Main loop positions (max pieces in main loop)
S_MAX = 100      # Switch mask positions (mirrors main loop)
B_MAX = 4        # Number of branch slots
B_SLOT = 4       # Genes per branch slot: [IN_pos, handedness, n_straights, active]
C_MAX = 0        # Crossing overlay disabled

# Computed dimensions
N_MAIN_LOOP = L_MAX
N_SWITCH_MASK = S_MAX
N_BRANCH_GENES = B_MAX * B_SLOT  # 16 genes for 4 branch slots
N_CROSSING_GENES = C_MAX * 2      # 0
N_PIECE_GENES = N_MAIN_LOOP + N_SWITCH_MASK + N_BRANCH_GENES + N_CROSSING_GENES  # 216
N_POSITION_GENES = 2  # start_x, start_y
N_VAR = N_PIECE_GENES + N_POSITION_GENES  # 218

# Template-based branch encoding:
# Each branch slot has 4 genes:
#   [0] IN_pos: Main loop position for IN switch (0 to L_MAX-1, or -1 if inactive)
#   [1] handedness: 0=LEFT, 1=RIGHT (determines switch type and curves)
#   [2] n_straights: Number of straight pieces in parallel section (0-8)
#   [3] active: 1 if branch is active, 0 if disabled

# Segment boundaries (start indices)
MAIN_LOOP_START = 0
MAIN_LOOP_END = L_MAX
SWITCH_MASK_START = L_MAX
SWITCH_MASK_END = L_MAX + S_MAX
BRANCH_SLOTS_START = L_MAX + S_MAX
BRANCH_SLOTS_END = L_MAX + S_MAX + N_BRANCH_GENES
CROSSING_START = BRANCH_SLOTS_END
CROSSING_END = N_PIECE_GENES
POSITION_START = N_PIECE_GENES
POSITION_END = N_VAR

# Sentinel value for inactive genes
INACTIVE = -1

# =============================================================================
# Piece Index Constants (from track_pieces.yaml piece_index mapping)
# =============================================================================

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

# Simple pieces (straights and basic curves)
SIMPLE_PIECE_INDICES = {STRAIGHT_16, STRAIGHT_24, R40_LEFT, R40_RIGHT}

# Complex pieces (switches, crossings, etc.)
COMPLEX_PIECE_INDICES = {CROSS_90, SWITCH_LEFT_IN, SWITCH_LEFT_OUT,
                         SWITCH_RIGHT_IN, SWITCH_RIGHT_OUT, DOUBLE_CROSSOVER}

# All switch piece indices
SWITCH_INDICES = {SWITCH_LEFT_IN, SWITCH_LEFT_OUT, SWITCH_RIGHT_IN, SWITCH_RIGHT_OUT}

# Switch pairs for balancing: (IN_index, OUT_index)
SWITCH_PAIRS = [
    (SWITCH_LEFT_IN, SWITCH_LEFT_OUT),    # Left pair
    (SWITCH_RIGHT_IN, SWITCH_RIGHT_OUT),  # Right pair
]


# =============================================================================
# Segment Slicing Functions
# =============================================================================

def get_main_loop(x: NDArray) -> NDArray:
    """Extract main loop segment from chromosome.

    Args:
        x: Full chromosome array of length N_VAR.

    Returns:
        Main loop segment of length L_MAX.
    """
    return x[MAIN_LOOP_START:MAIN_LOOP_END]


def get_switch_mask(x: NDArray) -> NDArray:
    """Extract switch mask segment from chromosome.

    Args:
        x: Full chromosome array of length N_VAR.

    Returns:
        Switch mask segment of length S_MAX.
    """
    return x[SWITCH_MASK_START:SWITCH_MASK_END]


def get_branch_slots(x: NDArray) -> NDArray:
    """Extract branch slots segment from chromosome.

    Args:
        x: Full chromosome array of length N_VAR.

    Returns:
        Branch slots segment of length B_MAX * B_SLOT.
    """
    return x[BRANCH_SLOTS_START:BRANCH_SLOTS_END]


def get_branch_slot(x: NDArray, slot_idx: int) -> NDArray:
    """Extract a single branch slot from chromosome.

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
    """Extract crossing overlay segment from chromosome.

    Args:
        x: Full chromosome array of length N_VAR.

    Returns:
        Crossing overlay segment of length C_MAX * 2.
    """
    return x[CROSSING_START:CROSSING_END]


def get_start_position(x: NDArray) -> Tuple[float, float]:
    """Extract starting position from chromosome.

    Args:
        x: Full chromosome array of length N_VAR.

    Returns:
        Tuple of (start_x, start_y).
    """
    return float(x[POSITION_START]), float(x[POSITION_START + 1])


def set_main_loop(x: NDArray, values: NDArray) -> None:
    """Set main loop segment in chromosome (in-place).

    Args:
        x: Full chromosome array of length N_VAR.
        values: Values to set (length <= L_MAX).
    """
    n = min(len(values), L_MAX)
    x[MAIN_LOOP_START:MAIN_LOOP_START + n] = values[:n]


def set_start_position(x: NDArray, start_x: float, start_y: float) -> None:
    """Set starting position in chromosome (in-place).

    Args:
        x: Full chromosome array of length N_VAR.
        start_x: Starting X coordinate.
        start_y: Starting Y coordinate.
    """
    x[POSITION_START] = start_x
    x[POSITION_START + 1] = start_y


# =============================================================================
# Chromosome Construction
# =============================================================================

def create_empty_chromosome(dtype=np.float64) -> NDArray:
    """Create an empty chromosome with all inactive genes.

    Args:
        dtype: Data type for the array.

    Returns:
        Chromosome array of length N_VAR filled with INACTIVE (-1).
    """
    x = np.full(N_VAR, INACTIVE, dtype=dtype)
    # Position genes should default to 0, not -1
    x[POSITION_START:POSITION_END] = 0.0
    return x


def create_chromosome_from_main_loop(
    main_loop: NDArray,
    start_x: float = 0.0,
    start_y: float = 0.0,
    dtype=np.float64,
) -> NDArray:
    """Create chromosome from main loop pieces only.

    Args:
        main_loop: Array of piece indices for main loop.
        start_x: Starting X coordinate.
        start_y: Starting Y coordinate.
        dtype: Data type for the array.

    Returns:
        Full chromosome array of length N_VAR.
    """
    x = create_empty_chromosome(dtype)
    n = min(len(main_loop), L_MAX)
    x[MAIN_LOOP_START:MAIN_LOOP_START + n] = main_loop[:n]
    set_start_position(x, start_x, start_y)
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
    max_piece_index: int,
    boundary_min_x: float = -100.0,
    boundary_max_x: float = 100.0,
    boundary_min_y: float = -100.0,
    boundary_max_y: float = 100.0,
) -> SegmentBounds:
    """Generate segment-specific lower and upper bounds.

    Args:
        max_piece_index: Maximum valid piece index from catalog.
        boundary_min_x: Minimum X boundary.
        boundary_max_x: Maximum X boundary.
        boundary_min_y: Minimum Y boundary.
        boundary_max_y: Maximum Y boundary.

    Returns:
        SegmentBounds with xl and xu arrays.
    """
    xl = np.full(N_VAR, -1.0, dtype=np.float64)
    xu = np.full(N_VAR, float(max_piece_index), dtype=np.float64)

    # Main loop: -1 (inactive) to max_piece_index
    xl[MAIN_LOOP_START:MAIN_LOOP_END] = -1.0
    xu[MAIN_LOOP_START:MAIN_LOOP_END] = float(max_piece_index)

    # Switch mask: -1 (no switch) to max_piece_index
    xl[SWITCH_MASK_START:SWITCH_MASK_END] = -1.0
    xu[SWITCH_MASK_START:SWITCH_MASK_END] = float(max_piece_index)

    # Branch slots: Template-based encoding [IN_pos, handedness, n_straights, active]
    for slot_idx in range(B_MAX):
        slot_start = BRANCH_SLOTS_START + slot_idx * B_SLOT
        # IN_pos: -1 (inactive) to L_MAX-1
        xl[slot_start] = -1.0
        xu[slot_start] = float(L_MAX - 1)
        # handedness: 0 (LEFT) to 1 (RIGHT)
        xl[slot_start + 1] = 0.0
        xu[slot_start + 1] = 1.0
        # n_straights: 0 to 8
        xl[slot_start + 2] = 0.0
        xu[slot_start + 2] = 8.0
        # active: 0 (disabled) to 1 (enabled)
        xl[slot_start + 3] = 0.0
        xu[slot_start + 3] = 1.0

    # Crossing overlay: -1 (inactive) to L_MAX-1 (main loop positions)
    xl[CROSSING_START:CROSSING_END] = -1.0
    xu[CROSSING_START:CROSSING_END] = float(L_MAX - 1)

    # Start position: boundary limits
    xl[POSITION_START] = boundary_min_x
    xu[POSITION_START] = boundary_max_x
    xl[POSITION_START + 1] = boundary_min_y
    xu[POSITION_START + 1] = boundary_max_y

    return SegmentBounds(xl=xl, xu=xu)


# =============================================================================
# Validation
# =============================================================================

def validate_chromosome(x: NDArray, max_piece_index: int) -> bool:
    """Validate chromosome structure.

    Args:
        x: Chromosome array.
        max_piece_index: Maximum valid piece index.

    Returns:
        True if valid, False otherwise.
    """
    if len(x) != N_VAR:
        return False

    # Check main loop values are in valid range
    main_loop = get_main_loop(x)
    if not np.all((main_loop >= INACTIVE) & (main_loop <= max_piece_index)):
        return False

    return True


def count_active_genes(x: NDArray) -> Dict[str, int]:
    """Count active (non-sentinel) genes in each segment.

    Args:
        x: Chromosome array.

    Returns:
        Dict with counts per segment.
    """
    return {
        'main_loop': int(np.sum(get_main_loop(x) >= 0)),
        'switch_mask': int(np.sum(get_switch_mask(x) >= 0)),
        'branch_slots': int(np.sum(get_branch_slots(x) >= 0)),
        'crossing_overlay': int(np.sum(get_crossing_overlay(x) >= 0)),
    }


def get_active_main_loop(x: NDArray) -> NDArray:
    """Get only the active (non-sentinel) genes from main loop.

    Args:
        x: Chromosome array.

    Returns:
        Array of active piece indices only.
    """
    main_loop = get_main_loop(x)
    return main_loop[main_loop >= 0]


# =============================================================================
# Branch Slot Helpers (Template-Based Architecture)
#
# Branch slot layout: [IN_pos, handedness, n_straights, active]
# - IN_pos: Main loop index where IN switch diverges (-1 if inactive)
# - handedness: 0=LEFT, 1=RIGHT (determines switch type and curve direction)
# - n_straights: Number of straight pieces in parallel section (0-8)
# - active: 1 if branch should be used, 0 if disabled
#
# OUT_pos is COMPUTED from geometry (not encoded) - the decoder finds where
# the branch naturally reconnects to the main loop based on template geometry.
#
# A valid branch requires:
# 1. active == 1 (branch is enabled)
# 2. IN_pos >= 0 and < main_loop_length
# 3. Sufficient inventory for switches and branch pieces
# 4. Main loop has room for OUT switch at computed position
# =============================================================================

def set_branch_slot(x: NDArray, slot_idx: int, values: NDArray) -> None:
    """Set a single branch slot in chromosome (in-place).

    Args:
        x: Full chromosome array of length N_VAR.
        slot_idx: Branch slot index (0 to B_MAX-1).
        values: Values to set (length <= B_SLOT).
            Format: [IN_position, OUT_position, branch_piece×18]
    """
    if slot_idx < 0 or slot_idx >= B_MAX:
        raise ValueError(f"slot_idx must be 0 to {B_MAX-1}, got {slot_idx}")
    start = BRANCH_SLOTS_START + slot_idx * B_SLOT
    n = min(len(values), B_SLOT)
    x[start:start + n] = values[:n]


def get_branch_in_position(x: NDArray, slot_idx: int) -> int:
    """Get IN switch position for a branch slot.

    The IN switch is where the branch diverges from the main loop.

    Args:
        x: Chromosome array.
        slot_idx: Branch slot index (0 to B_MAX-1).

    Returns:
        IN switch position in main loop, or INACTIVE if branch is inactive.
    """
    slot = get_branch_slot(x, slot_idx)
    return int(slot[0])


def get_branch_handedness(x: NDArray, slot_idx: int) -> int:
    """Get branch handedness (LEFT=0, RIGHT=1).

    Args:
        x: Chromosome array.
        slot_idx: Branch slot index (0 to B_MAX-1).

    Returns:
        0 for LEFT, 1 for RIGHT.
    """
    slot = get_branch_slot(x, slot_idx)
    return int(slot[1])


def get_branch_n_straights(x: NDArray, slot_idx: int) -> int:
    """Get number of straight pieces in parallel section.

    Args:
        x: Chromosome array.
        slot_idx: Branch slot index (0 to B_MAX-1).

    Returns:
        Number of straights (0-8).
    """
    slot = get_branch_slot(x, slot_idx)
    return max(0, min(8, int(slot[2])))


def get_branch_active_flag(x: NDArray, slot_idx: int) -> bool:
    """Get branch active flag.

    Args:
        x: Chromosome array.
        slot_idx: Branch slot index (0 to B_MAX-1).

    Returns:
        True if active flag is 1.
    """
    slot = get_branch_slot(x, slot_idx)
    return int(slot[3]) >= 1


def get_branch_template_params(x: NDArray, slot_idx: int) -> Tuple[int, int, int, bool]:
    """Get all template parameters for a branch slot.

    Args:
        x: Chromosome array.
        slot_idx: Branch slot index (0 to B_MAX-1).

    Returns:
        Tuple of (in_pos, handedness, n_straights, active).
    """
    slot = get_branch_slot(x, slot_idx)
    in_pos = int(slot[0])
    handedness = int(slot[1]) % 2  # Ensure 0 or 1
    n_straights = max(0, min(8, int(slot[2])))
    active = int(slot[3]) >= 1
    return (in_pos, handedness, n_straights, active)


def set_branch_template_params(
    x: NDArray,
    slot_idx: int,
    in_pos: int,
    handedness: int,
    n_straights: int,
    active: int,
) -> None:
    """Set template parameters for a branch slot.

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
    x[start] = in_pos
    x[start + 1] = handedness % 2
    x[start + 2] = max(0, min(8, n_straights))
    x[start + 3] = 1 if active else 0


# Legacy function - kept for backward compatibility
def get_branch_out_position(x: NDArray, slot_idx: int) -> int:
    """DEPRECATED: OUT position is now computed from geometry.

    In the new template-based encoding, OUT position is determined by the
    decoder based on template geometry, not encoded in the chromosome.

    Returns:
        Always returns INACTIVE (-1) in new encoding.
    """
    return INACTIVE


# Legacy function - kept for backward compatibility
def get_branch_pieces(x: NDArray, slot_idx: int) -> NDArray:
    """DEPRECATED: Branch pieces are now computed from template.

    In the new template-based encoding, branch pieces are determined by the
    template (approach_curve + n_straights × STRAIGHT + return_curve).

    Returns:
        Empty array in new encoding.
    """
    return np.array([], dtype=np.int32)


def is_branch_active(x: NDArray, slot_idx: int) -> bool:
    """Check if a branch slot is active.

    A branch is active if active_flag is 1 AND IN_position >= 0.

    Args:
        x: Chromosome array.
        slot_idx: Branch slot index (0 to B_MAX-1).

    Returns:
        True if branch is active.
    """
    in_pos, _, _, active = get_branch_template_params(x, slot_idx)
    return active and in_pos >= 0


def is_branch_valid(x: NDArray, slot_idx: int) -> bool:
    """Check if a branch slot defines a valid template-based branch.

    A valid branch requires:
    1. active flag is 1
    2. IN_position >= 0

    Note: Geometric validity (OUT position fits in main loop) is checked
    by the decoder, not here.

    Args:
        x: Chromosome array.
        slot_idx: Branch slot index (0 to B_MAX-1).

    Returns:
        True if branch slot is potentially valid.
    """
    in_pos, _, _, active = get_branch_template_params(x, slot_idx)
    return active and in_pos >= 0


# Legacy aliases for backward compatibility
def get_branch_src_switch(x: NDArray, slot_idx: int) -> int:
    """Legacy alias for get_branch_in_position."""
    return get_branch_in_position(x, slot_idx)


def get_branch_rejoin_target(x: NDArray, slot_idx: int) -> int:
    """Legacy alias for get_branch_out_position."""
    return get_branch_out_position(x, slot_idx)


def set_switch_mask_value(x: NDArray, position: int, switch_type: int) -> None:
    """Set switch type at a main loop position.

    Args:
        x: Chromosome array.
        position: Position in main loop (0 to L_MAX-1).
        switch_type: Switch type index, or INACTIVE for no switch.
    """
    if position < 0 or position >= S_MAX:
        raise ValueError(f"position must be 0 to {S_MAX-1}, got {position}")
    x[SWITCH_MASK_START + position] = switch_type


def get_switch_mask_value(x: NDArray, position: int) -> int:
    """Get switch type at a main loop position.

    Args:
        x: Chromosome array.
        position: Position in main loop (0 to L_MAX-1).

    Returns:
        Switch type index, or INACTIVE if no switch.
    """
    if position < 0 or position >= S_MAX:
        return INACTIVE
    return int(x[SWITCH_MASK_START + position])


def find_active_branch_slots(x: NDArray) -> List[int]:
    """Find all active branch slot indices.

    Args:
        x: Chromosome array.

    Returns:
        List of active branch slot indices.
    """
    active = []
    for i in range(B_MAX):
        if is_branch_active(x, i):
            active.append(i)
    return active


def find_inactive_branch_slots(x: NDArray) -> List[int]:
    """Find all inactive branch slot indices.

    Args:
        x: Chromosome array.

    Returns:
        List of inactive branch slot indices.
    """
    inactive = []
    for i in range(B_MAX):
        if not is_branch_active(x, i):
            inactive.append(i)
    return inactive


# =============================================================================
# Crossing Overlay Helpers (disabled - C_MAX=0)
# =============================================================================

def get_crossing_pair(x: NDArray, pair_idx: int) -> Tuple[int, int]:
    """Get a crossing pair (two position indices).

    Note: Crossing overlay is currently disabled (C_MAX=0).

    Args:
        x: Chromosome array.
        pair_idx: Crossing pair index (0 to C_MAX-1).

    Returns:
        Tuple of (position1, position2) in main loop.
    """
    if C_MAX == 0:
        return (INACTIVE, INACTIVE)
    if pair_idx < 0 or pair_idx >= C_MAX:
        raise ValueError(f"pair_idx must be 0 to {C_MAX-1}, got {pair_idx}")
    start = CROSSING_START + pair_idx * 2
    return int(x[start]), int(x[start + 1])


def set_crossing_pair(x: NDArray, pair_idx: int, pos1: int, pos2: int) -> None:
    """Set a crossing pair (in-place).

    Note: Crossing overlay is currently disabled (C_MAX=0).

    Args:
        x: Chromosome array.
        pair_idx: Crossing pair index (0 to C_MAX-1).
        pos1: First position in main loop.
        pos2: Second position in main loop.
    """
    if C_MAX == 0:
        return  # No-op when crossings disabled
    if pair_idx < 0 or pair_idx >= C_MAX:
        raise ValueError(f"pair_idx must be 0 to {C_MAX-1}, got {pair_idx}")
    start = CROSSING_START + pair_idx * 2
    x[start] = pos1
    x[start + 1] = pos2


def is_crossing_active(x: NDArray, pair_idx: int) -> bool:
    """Check if a crossing pair is active.

    Note: Crossing overlay is currently disabled (C_MAX=0).

    Args:
        x: Chromosome array.
        pair_idx: Crossing pair index (0 to C_MAX-1).

    Returns:
        True if crossing is active (both positions >= 0).
    """
    if C_MAX == 0:
        return False
    pos1, pos2 = get_crossing_pair(x, pair_idx)
    return pos1 >= 0 and pos2 >= 0
