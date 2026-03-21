"""4-phase construction decoder for multi-segment chromosomes.

Decodes integer chromosomes into Layout objects via construction-based approach.
Guarantees >99% feasibility by construction - any random chromosome produces
an evaluable Layout.

Architecture: Paired-Switch Continuous Paths
- Each branch requires paired switches (IN diverges, OUT merges)
- All traversal paths are computed as continuous FK chains
- With N switch pairs, there are 2^N possible paths through the track
- Constraint: ALL paths must form closed loops

Phases:
1. Main loop construction - turtle-graphics FK with inventory/angular budget checks
2. Switch pair extraction - identify paired IN/OUT switches from chromosome
3. Path enumeration - generate all 2^N traversal paths
4. Per-path FK computation - continuous FK chain for each path
"""

from dataclasses import dataclass, field
from itertools import product
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .data import TrackCatalog
from .encoding import (
    B_MAX,
    C_MAX,
    INACTIVE,
    L_MAX,
    get_active_main_loop,
    get_branch_in_position,
    get_branch_slot,
    get_branch_slots,
    get_branch_template_params,
    get_crossing_overlay,
    get_crossing_pair,
    get_main_loop,
    get_start_position,
    get_switch_mask,
    is_branch_active,
    is_branch_valid,
    is_crossing_active,
)
from .geometry import Layout, compute_fk_chain
from .templates import (
    LEFT_SIDING,
    RIGHT_SIDING,
    TEMPLATES,
    STRAIGHT_16,
    check_siding_inventory,
    compute_branch_pieces,
    compute_required_main_distance,
)
from .topology import MultiPathLayout, SwitchPair, TraversalPath


# =============================================================================
# Decoder Configuration
# =============================================================================

@dataclass
class DecoderConfig:
    """Configuration for the construction decoder."""

    # Closure tolerances (for early closure detection)
    position_tolerance: float = 8.0   # studs
    angle_tolerance: float = 15.0     # degrees

    # Angular budget
    target_angle: float = 360.0       # degrees for closed loop
    max_cumulative_angle: float = 1440.0  # Allow up to 4 full loops (for complex layouts)

    # Inventory mode
    skip_on_exhausted: bool = True    # Skip gene if piece type exhausted
    substitute_similar: bool = False  # Substitute with similar piece (Phase 2)

    # Collision detection
    collision_radius: float = 3.0     # studs - minimum distance between pieces


@dataclass
class BranchState:
    """State for a branch (pushed when switch is placed)."""

    # Diverging port turtle state
    x: float
    y: float
    theta: float  # degrees

    # Branch metadata
    src_switch_idx: int        # Position in main loop where switch was placed
    switch_piece_idx: int      # Piece index of the switch
    branch_slot_idx: int       # Which branch slot this corresponds to


@dataclass
class DecoderState:
    """Intermediate state during decoding."""

    # Current turtle state
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0  # degrees

    # Accumulated values
    cumulative_angle: float = 0.0
    pieces_placed: int = 0

    # Inventory tracking
    inventory_used: Dict[int, int] = field(default_factory=dict)

    # Placed piece data
    piece_indices: List[int] = field(default_factory=list)
    states: List[Tuple[float, float, float]] = field(default_factory=list)

    # Switch and branch tracking
    switch_positions: List[int] = field(default_factory=list)  # Main loop positions with switches
    branch_stack: List[BranchState] = field(default_factory=list)  # L-system branch stack
    branch_pieces: Dict[int, List[int]] = field(default_factory=dict)  # {slot_idx: [piece indices]}
    branch_states: Dict[int, List[Tuple[float, float, float]]] = field(default_factory=dict)

    # Crossing positions
    crossing_positions: List[Tuple[int, int]] = field(default_factory=list)

    # Port connectivity tracking
    # Tracks main loop positions where switches are placed (their port 2 is potentially loose)
    switch_port2_positions: List[int] = field(default_factory=list)
    # Positions where port 2 is connected (part of a valid switch pair)
    connected_port2_positions: set = field(default_factory=set)

    def use_piece(self, piece_idx: int, add_to_main_loop: bool = True) -> None:
        """Record piece usage.

        Args:
            piece_idx: Piece index to record.
            add_to_main_loop: If True, add to piece_indices (main loop).
                              If False, only track inventory.
        """
        self.inventory_used[piece_idx] = self.inventory_used.get(piece_idx, 0) + 1
        if add_to_main_loop:
            self.piece_indices.append(piece_idx)
        self.pieces_placed += 1

    def get_usage(self, piece_idx: int) -> int:
        """Get current usage count for piece type."""
        return self.inventory_used.get(piece_idx, 0)

    def push_branch(self, branch: BranchState) -> None:
        """Push a branch state onto the stack (L-system '[')."""
        self.branch_stack.append(branch)

    def pop_branch(self) -> Optional[BranchState]:
        """Pop a branch state from the stack (L-system ']')."""
        if self.branch_stack:
            return self.branch_stack.pop()
        return None


# =============================================================================
# Main Decoder Function
# =============================================================================

def decode_chromosome(
    chromosome: NDArray,
    catalog: TrackCatalog,
    inventory: Dict[str, int],
    config: Optional[DecoderConfig] = None,
) -> Layout:
    """Decode chromosome into Layout via paired-switch architecture.

    New Architecture:
    1. Build main loop (piece sequence without switches)
    2. Extract switch pairs from branch slots
    3. Enumerate all 2^N traversal paths
    4. Compute continuous FK for each path
    5. Return MultiPathLayout (backward compatible with Layout)

    Args:
        chromosome: Full chromosome array of length N_VAR.
        catalog: Track catalog with piece properties.
        inventory: Available inventory {piece_id: count}.
        config: Decoder configuration (uses defaults if None).

    Returns:
        MultiPathLayout with all traversal paths (Layout-compatible).
    """
    if config is None:
        config = DecoderConfig()

    # Convert inventory to index-based for faster lookup
    inventory_by_index = _convert_inventory_to_index(inventory, catalog)

    # Phase 1: Main loop construction
    state = _decode_main_loop(chromosome, catalog, inventory_by_index, config)

    # Phase 2: Extract switch pairs from branch slots
    switch_pairs = _extract_switch_pairs(chromosome, state, catalog, inventory_by_index, config)

    # Phase 3: Enumerate and compute all traversal paths
    multi_path = _build_multi_path_layout(
        chromosome, state, switch_pairs, catalog, inventory_by_index, config
    )

    # Apply starting position translation
    start_x, start_y = get_start_position(chromosome)
    multi_path.start_position = (start_x, start_y)
    for path in multi_path.paths:
        if len(path.states) > 0:
            path.states[:, 0] += start_x
            path.states[:, 1] += start_y

    return multi_path


def decode_chromosome_legacy(
    chromosome: NDArray,
    catalog: TrackCatalog,
    inventory: Dict[str, int],
    config: Optional[DecoderConfig] = None,
) -> Layout:
    """Legacy decoder returning simple Layout (no multi-path support).

    Use this for backward compatibility with code that expects Layout.
    """
    multi_path = decode_chromosome(chromosome, catalog, inventory, config)

    # Convert to simple Layout from main path
    main_path = multi_path.get_main_path()
    if main_path is None or main_path.n_pieces == 0:
        return Layout(
            indices=np.array([], dtype=np.int32),
            states=np.zeros((1, 3), dtype=np.float64),
        )

    return Layout(
        indices=np.array(main_path.piece_sequence, dtype=np.int32),
        states=main_path.states.copy(),
    )


# =============================================================================
# Phase 1: Main Loop Construction
# =============================================================================

def _decode_main_loop(
    chromosome: NDArray,
    catalog: TrackCatalog,
    inventory_by_index: Dict[int, int],
    config: DecoderConfig,
) -> DecoderState:
    """Phase 1: Build main loop from chromosome genes.

    Process genes left-to-right using turtle-graphics FK.
    Skip genes that:
    - Are inactive (-1)
    - Reference exhausted inventory
    - Would overshoot angular budget

    NOTE: Early closure detection is DISABLED. All valid genes are processed
    to ensure switches and branches encoded in the chromosome are included.
    The optimizer handles closure via fitness constraints.

    Args:
        chromosome: Full chromosome array.
        catalog: Track catalog.
        inventory_by_index: Available inventory by piece index.
        config: Decoder configuration.

    Returns:
        DecoderState with placed pieces and states.
    """
    state = DecoderState()
    state.states.append((0.0, 0.0, 0.0))  # Initial state at origin

    main_loop = get_main_loop(chromosome)

    for gene in main_loop:
        gene_val = int(gene)

        # Skip inactive genes
        if gene_val == INACTIVE:
            continue

        # Skip invalid piece indices
        if gene_val < 0 or gene_val >= len(catalog._fk_table):
            continue

        # Check inventory
        available = inventory_by_index.get(gene_val, 0)
        used = state.get_usage(gene_val)
        if config.skip_on_exhausted and used >= available:
            continue

        # Get FK deltas for this piece
        fk = catalog._fk_table[gene_val]
        dx, dy, dtheta = fk[0], fk[1], fk[2]

        # Check angular budget (allow complex layouts with multiple loops)
        new_cumulative = state.cumulative_angle + abs(dtheta)
        if new_cumulative > config.max_cumulative_angle:
            continue  # Hard limit to prevent infinite layouts

        # Apply FK transformation
        theta_rad = np.radians(state.theta)
        cos_t = np.cos(theta_rad)
        sin_t = np.sin(theta_rad)

        new_x = state.x + dx * cos_t - dy * sin_t
        new_y = state.y + dx * sin_t + dy * cos_t
        new_theta = state.theta + dtheta

        # Update state
        state.x = new_x
        state.y = new_y
        state.theta = new_theta
        state.cumulative_angle += abs(dtheta)

        # Record piece placement
        state.use_piece(gene_val)
        state.states.append((new_x, new_y, new_theta))

        # Track switch positions (port 2 is potentially loose)
        # Switch indices: 5=LEFT_IN, 6=LEFT_OUT, 7=RIGHT_IN, 8=RIGHT_OUT
        if gene_val in (5, 6, 7, 8):
            position = len(state.piece_indices) - 1  # Current position
            state.switch_port2_positions.append(position)

        # NOTE: Early closure check removed - process ALL genes to ensure
        # switches and branches are included. Optimizer handles closure.

    return state


# =============================================================================
# Phase 2: Template-Based Siding Extraction
# =============================================================================

# Switch piece indices (from track_pieces.yaml)
SWITCH_PIECE_IDS = {
    "R40_SWITCH_LEFT_IN": 5,
    "R40_SWITCH_LEFT_OUT": 6,
    "R40_SWITCH_RIGHT_IN": 7,
    "R40_SWITCH_RIGHT_OUT": 8,
}


def _extract_switch_pairs(
    chromosome: NDArray,
    state: DecoderState,
    catalog: TrackCatalog,
    inventory_by_index: Dict[int, int],
    config: DecoderConfig,
) -> List[SwitchPair]:
    """Extract switch pairs from branch slots AND existing switches in main loop.

    Two-pass approach:
    1. Template-based: Read branch slots and inject switches into main loop
    2. Legacy detection: Scan main loop for switches already placed and pair them

    Args:
        chromosome: Full chromosome array with branch slot params.
        state: Decoder state with main loop pieces.
        catalog: Track catalog.
        inventory_by_index: Available inventory by piece index.
        config: Decoder configuration.

    Returns:
        List of valid SwitchPair objects with computed branch pieces.
    """
    switch_pairs = []
    n_main_loop = len(state.piece_indices)

    if n_main_loop == 0:
        return switch_pairs

    pair_id = 0
    used_positions = set()  # Track positions already used in pairs

    # =========================================================================
    # Pass 1: Template-based branch slot processing
    # =========================================================================
    for slot_idx in range(B_MAX):
        # Read template parameters from branch slot
        in_pos, handedness, n_straights, active = get_branch_template_params(chromosome, slot_idx)

        # Skip inactive branches
        if not active or in_pos < 0:
            continue

        # Validate IN_pos is within main loop
        if in_pos >= n_main_loop:
            continue

        # Get template for this handedness (0=LEFT, 1=RIGHT)
        template = TEMPLATES.get(handedness)
        if template is None:
            continue

        # Compute required main loop distance for this siding
        required_distance = compute_required_main_distance(template, n_straights)

        # Find OUT position based on template geometry
        out_pos = _find_out_position_for_siding(
            in_pos, required_distance, state.piece_indices, catalog
        )

        # Skip if no valid OUT position found
        if out_pos is None or out_pos >= n_main_loop:
            continue

        # Check inventory for switch pair
        in_switch_idx = template.in_switch_idx
        out_switch_idx = template.out_switch_idx

        in_avail = inventory_by_index.get(in_switch_idx, 0) - state.get_usage(in_switch_idx)
        out_avail = inventory_by_index.get(out_switch_idx, 0) - state.get_usage(out_switch_idx)

        if in_avail < 1 or out_avail < 1:
            continue  # No switch inventory

        # Generate branch pieces from template
        branch_pieces = compute_branch_pieces(template, n_straights)

        # Check inventory for branch pieces
        if not _check_branch_inventory(branch_pieces, inventory_by_index, state.inventory_used):
            continue

        # All checks passed - inject switches into main loop
        original_in_piece = state.piece_indices[in_pos]
        original_out_piece = state.piece_indices[out_pos]

        state.piece_indices[in_pos] = in_switch_idx
        state.piece_indices[out_pos] = out_switch_idx

        # Track switch usage
        state.use_piece(in_switch_idx, add_to_main_loop=False)
        state.use_piece(out_switch_idx, add_to_main_loop=False)

        # Return original pieces to inventory if they were valid
        if original_in_piece >= 0:
            state.inventory_used[original_in_piece] = max(0, state.inventory_used.get(original_in_piece, 1) - 1)
        if original_out_piece >= 0:
            state.inventory_used[original_out_piece] = max(0, state.inventory_used.get(original_out_piece, 1) - 1)

        # Consume branch inventory
        for piece_idx in branch_pieces:
            state.inventory_used[piece_idx] = state.inventory_used.get(piece_idx, 0) + 1

        # Create switch pair
        pair = SwitchPair(
            pair_id=pair_id,
            in_position=in_pos,
            out_position=out_pos,
            in_switch_idx=in_switch_idx,
            out_switch_idx=out_switch_idx,
            branch_pieces=branch_pieces,
        )
        switch_pairs.append(pair)
        pair_id += 1

        # Mark positions as used
        used_positions.add(in_pos)
        used_positions.add(out_pos)

        # Mark port 2 as connected for both switches
        state.connected_port2_positions.add(in_pos)
        state.connected_port2_positions.add(out_pos)

    # =========================================================================
    # Pass 2: Legacy switch detection from main loop
    # =========================================================================
    # Scan for switches already placed in main loop and try to pair them
    # Process both left and right handedness
    for handedness in [0, 1]:  # 0=LEFT, 1=RIGHT
        template = TEMPLATES.get(handedness)
        if template is None:
            continue

        in_switch_idx = template.in_switch_idx
        out_switch_idx = template.out_switch_idx

        # Find all IN and OUT switch positions in main loop
        in_positions = []
        out_positions = []

        for pos, piece_idx in enumerate(state.piece_indices):
            if pos in used_positions:
                continue  # Already used in a pair
            if piece_idx == in_switch_idx:
                in_positions.append(pos)
            elif piece_idx == out_switch_idx:
                out_positions.append(pos)

        # Match IN switches with compatible OUT switches
        for in_pos in sorted(in_positions):
            best_out_pos = None
            best_n_straights = None
            best_distance_error = float('inf')

            for out_pos in sorted(out_positions):
                # OUT must come after IN
                if out_pos <= in_pos:
                    continue
                # Don't reuse OUT switches
                if out_pos in used_positions:
                    continue

                # Compute main loop distance between IN and OUT
                main_distance = _compute_main_loop_distance(
                    in_pos, out_pos, state.piece_indices, catalog
                )

                # Try different n_straights to find best match
                for n_straights in range(9):  # 0 to 8 straights
                    required_distance = compute_required_main_distance(template, n_straights)
                    distance_error = abs(main_distance - required_distance)

                    # Check if this is a better match (within tolerance)
                    if distance_error < best_distance_error and distance_error < 8.0:
                        # Check branch inventory
                        branch_pieces = compute_branch_pieces(template, n_straights)
                        has_inventory = _check_branch_inventory(
                            branch_pieces, inventory_by_index, state.inventory_used
                        )
                        if has_inventory:
                            best_out_pos = out_pos
                            best_n_straights = n_straights
                            best_distance_error = distance_error

            # Create pair if match found
            if best_out_pos is not None:
                branch_pieces = compute_branch_pieces(template, best_n_straights)

                pair = SwitchPair(
                    pair_id=pair_id,
                    in_position=in_pos,
                    out_position=best_out_pos,
                    in_switch_idx=in_switch_idx,
                    out_switch_idx=out_switch_idx,
                    branch_pieces=branch_pieces,
                )
                switch_pairs.append(pair)
                pair_id += 1

                # Mark positions as used
                used_positions.add(in_pos)
                used_positions.add(best_out_pos)

                # Mark port 2 as connected for both switches
                state.connected_port2_positions.add(in_pos)
                state.connected_port2_positions.add(best_out_pos)

                # Consume branch inventory
                for piece_idx in branch_pieces:
                    state.inventory_used[piece_idx] = state.inventory_used.get(piece_idx, 0) + 1

    # Track all switch positions for loose port counting
    for pos, piece_idx in enumerate(state.piece_indices):
        if piece_idx in (5, 6, 7, 8):  # Switch indices
            if pos not in state.switch_port2_positions:
                state.switch_port2_positions.append(pos)

    return switch_pairs


def _find_out_position_for_siding(
    in_pos: int,
    required_distance: float,
    piece_indices: List[int],
    catalog: TrackCatalog,
) -> Optional[int]:
    """Find OUT switch position based on template geometry.

    Scans forward from IN position, accumulating arc length until
    the required distance is reached.

    Args:
        in_pos: IN switch position in main loop.
        required_distance: Required main loop distance for branch (studs).
        piece_indices: Main loop piece indices.
        catalog: Track catalog.

    Returns:
        OUT switch position, or None if no valid position found.
    """
    if in_pos < 0 or in_pos >= len(piece_indices):
        return None

    accumulated = 0.0
    tolerance = 8.0  # studs - allow slack for geometric matching

    # Start after IN switch position
    for pos in range(in_pos + 1, len(piece_indices)):
        piece_idx = piece_indices[pos]
        if piece_idx < 0:
            continue

        # Get arc length for this piece
        arc_length = float(catalog.get_arc_lengths(np.array([piece_idx]))[0])
        accumulated += arc_length

        # Check if we've reached the required distance
        if abs(accumulated - required_distance) <= tolerance:
            return pos

        if accumulated > required_distance + tolerance:
            break  # Overshot

    return None


def _match_switch_pairs(
    in_positions: List[int],
    out_positions: List[int],
    in_switch_idx: int,
    out_switch_idx: int,
    handedness: int,
    state: DecoderState,
    catalog: TrackCatalog,
    inventory_by_index: Dict[int, int],
    config: DecoderConfig,
) -> List[SwitchPair]:
    """Match IN switches with compatible OUT switches.

    For each IN switch, find the nearest OUT switch that:
    1. Comes AFTER the IN switch in the main loop
    2. Is at approximately the right distance for a valid branch

    Args:
        in_positions: Main loop positions of IN switches.
        out_positions: Main loop positions of OUT switches.
        in_switch_idx: Piece index of IN switch type.
        out_switch_idx: Piece index of OUT switch type.
        handedness: 0=LEFT, 1=RIGHT.
        state: Decoder state.
        catalog: Track catalog.
        inventory_by_index: Available inventory.
        config: Decoder configuration.

    Returns:
        List of matched SwitchPair objects.
    """
    pairs = []
    template = TEMPLATES.get(handedness)
    if template is None:
        return pairs

    used_out_positions = set()
    pair_id = 0

    for in_pos in sorted(in_positions):
        # Find best matching OUT switch
        best_out_pos = None
        best_n_straights = None
        best_distance_error = float('inf')

        for out_pos in sorted(out_positions):
            # OUT must come after IN
            if out_pos <= in_pos:
                continue
            # Don't reuse OUT switches
            if out_pos in used_out_positions:
                continue

            # Compute main loop distance between IN and OUT
            main_distance = _compute_main_loop_distance(
                in_pos, out_pos, state.piece_indices, catalog
            )

            # Try different n_straights to find best match
            for n_straights in range(9):  # 0 to 8 straights
                required_distance = compute_required_main_distance(template, n_straights)
                distance_error = abs(main_distance - required_distance)

                # Check if this is a better match (within tolerance)
                if distance_error < best_distance_error and distance_error < 4.0:
                    # Check branch inventory (curves for approach/return)
                    branch_pieces = compute_branch_pieces(template, n_straights)
                    has_inventory = _check_branch_inventory(
                        branch_pieces, inventory_by_index, state.inventory_used
                    )
                    if has_inventory:
                        best_out_pos = out_pos
                        best_n_straights = n_straights
                        best_distance_error = distance_error

        # Create pair if match found
        if best_out_pos is not None:
            branch_pieces = compute_branch_pieces(template, best_n_straights)

            pair = SwitchPair(
                pair_id=pair_id,
                in_position=in_pos,
                out_position=best_out_pos,
                in_switch_idx=in_switch_idx,
                out_switch_idx=out_switch_idx,
                branch_pieces=branch_pieces,
            )
            pairs.append(pair)
            used_out_positions.add(best_out_pos)
            pair_id += 1

            # Mark port 2 as connected for both switches
            state.connected_port2_positions.add(in_pos)
            state.connected_port2_positions.add(best_out_pos)

            # Consume branch inventory
            for piece_idx in branch_pieces:
                state.inventory_used[piece_idx] = state.inventory_used.get(piece_idx, 0) + 1

    return pairs


def _compute_main_loop_distance(
    in_pos: int,
    out_pos: int,
    piece_indices: List[int],
    catalog: TrackCatalog,
) -> float:
    """Compute arc length along main loop between two positions.

    Args:
        in_pos: Start position.
        out_pos: End position.
        piece_indices: Main loop piece indices.
        catalog: Track catalog.

    Returns:
        Total arc length in studs.
    """
    total = 0.0
    for pos in range(in_pos, out_pos):
        if pos < len(piece_indices):
            piece_idx = piece_indices[pos]
            if piece_idx >= 0:
                arc_length = float(catalog.get_arc_lengths(np.array([piece_idx]))[0])
                total += arc_length
    return total


def _check_branch_inventory(
    branch_pieces: List[int],
    available: Dict[int, int],
    used: Dict[int, int],
) -> bool:
    """Check if branch pieces are available in inventory.

    Args:
        branch_pieces: List of piece indices for branch.
        available: Available inventory by piece index.
        used: Already used inventory.

    Returns:
        True if all branch pieces are available.
    """
    # Count required pieces
    required = {}
    for piece_idx in branch_pieces:
        required[piece_idx] = required.get(piece_idx, 0) + 1

    # Check availability
    for piece_idx, count in required.items():
        avail = available.get(piece_idx, 0)
        already_used = used.get(piece_idx, 0)
        if already_used + count > avail:
            return False

    return True


def _find_out_position(
    in_pos: int,
    required_distance: float,
    piece_indices: List[int],
    catalog: TrackCatalog,
) -> Optional[int]:
    """Find main loop position for OUT switch based on required distance.

    Scans forward from IN position, accumulating arc length until
    the required distance is reached. The OUT switch should be placed
    at the position where accumulated distance matches branch length.

    Args:
        in_pos: IN switch position in main loop.
        required_distance: Required X-distance for branch (studs).
        piece_indices: Main loop piece indices.
        catalog: Track catalog for arc length lookup.

    Returns:
        OUT switch position, or None if no valid position found.
    """
    if in_pos < 0 or in_pos >= len(piece_indices):
        return None

    accumulated = 0.0
    tolerance = 4.0  # studs - allow some slack for geometric matching

    # Start after IN switch position
    for pos in range(in_pos + 1, len(piece_indices)):
        piece_idx = piece_indices[pos]
        if piece_idx < 0:
            continue

        # Get arc length for this piece (approximate as FK dx for straights)
        arc_length = float(catalog.get_arc_lengths(np.array([piece_idx]))[0])
        accumulated += arc_length

        # Check if we've reached the required distance
        if abs(accumulated - required_distance) <= tolerance:
            # Found a position at approximately the right distance
            return pos

        if accumulated > required_distance + tolerance:
            # Overshot - no valid position
            break

    return None


def _consume_siding_inventory(
    template,
    n_straights: int,
    state: DecoderState,
) -> None:
    """Mark siding pieces as used in decoder state.

    Args:
        template: Passing siding template.
        n_straights: Number of straights in parallel section.
        state: Decoder state to update.
    """
    # Mark switches as used
    state.use_piece(template.in_switch_idx, add_to_main_loop=False)
    state.use_piece(template.out_switch_idx, add_to_main_loop=False)

    # Mark branch pieces as used
    state.use_piece(template.approach_curve_idx, add_to_main_loop=False)
    state.use_piece(template.return_curve_idx, add_to_main_loop=False)
    for _ in range(n_straights):
        state.use_piece(template.straight_idx, add_to_main_loop=False)


# =============================================================================
# Phase 3: Multi-Path Layout Construction
# =============================================================================

def _build_multi_path_layout(
    chromosome: NDArray,
    state: DecoderState,
    switch_pairs: List[SwitchPair],
    catalog: TrackCatalog,
    inventory_by_index: Dict[int, int],
    config: DecoderConfig,
) -> MultiPathLayout:
    """Build MultiPathLayout with all traversal paths.

    For N switch pairs, generates 2^N paths. Each path is a continuous
    FK chain representing one way to traverse the track.

    Args:
        chromosome: Full chromosome array.
        state: Decoder state with main loop.
        switch_pairs: Valid switch pairs.
        catalog: Track catalog.
        inventory_by_index: Available inventory.
        config: Decoder configuration.

    Returns:
        MultiPathLayout with all traversal paths computed.
    """
    main_loop_pieces = list(state.piece_indices)
    n_switch_pairs = len(switch_pairs)

    # Compute loose port count: switches with unconnected port 2
    # All switch positions minus those with connected port 2
    loose_port_count = len(state.switch_port2_positions) - len(state.connected_port2_positions)

    # If no switch pairs, just return single main loop path
    if n_switch_pairs == 0:
        main_path = _compute_single_path(
            main_loop_pieces, [], tuple(), catalog
        )
        return MultiPathLayout(
            main_loop_pieces=main_loop_pieces,
            switch_pairs=[],
            paths=[main_path],
            loose_port_count=loose_port_count,
        )

    # Generate all 2^N path combinations
    paths = []
    for path_id, choices in enumerate(product([0, 1], repeat=n_switch_pairs)):
        path = _compute_single_path(
            main_loop_pieces, switch_pairs, choices, catalog
        )
        path.path_id = path_id
        paths.append(path)

    return MultiPathLayout(
        main_loop_pieces=main_loop_pieces,
        switch_pairs=switch_pairs,
        paths=paths,
        loose_port_count=loose_port_count,
    )


def _compute_single_path(
    main_loop_pieces: List[int],
    switch_pairs: List[SwitchPair],
    route_choices: Tuple[int, ...],
    catalog: TrackCatalog,
) -> TraversalPath:
    """Compute FK for a single traversal path.

    Args:
        main_loop_pieces: Base main loop piece indices.
        switch_pairs: Switch pairs (sorted by in_position).
        route_choices: Binary tuple (0=straight, 1=branch) per switch pair.
        catalog: Track catalog.

    Returns:
        TraversalPath with computed states and closure metrics.
    """
    if not main_loop_pieces:
        return TraversalPath(
            path_id=0,
            route_choices=route_choices,
            piece_sequence=[],
            states=np.zeros((1, 3), dtype=np.float64),
            closure_error=0.0,
            angle_error=360.0,
        )

    # Sort switch pairs by in_position for correct ordering
    sorted_pairs = sorted(switch_pairs, key=lambda p: p.in_position)

    # Build piece sequence and route indices for this path
    piece_sequence = []
    route_indices = []
    current_pos = 0

    for i, pair in enumerate(sorted_pairs):
        choice = route_choices[i] if i < len(route_choices) else 0

        # Add main loop pieces up to IN switch position (all default route)
        segment = main_loop_pieces[current_pos:pair.in_position]
        piece_sequence.extend(segment)
        route_indices.extend([0] * len(segment))

        if choice == 0:
            # Straight-through: both switches use route 0 (default)
            piece_sequence.append(pair.in_switch_idx)
            route_indices.append(0)
            segment = main_loop_pieces[pair.in_position + 1:pair.out_position]
            piece_sequence.extend(segment)
            route_indices.extend([0] * len(segment))
            piece_sequence.append(pair.out_switch_idx)
            route_indices.append(0)
        else:
            # Branch: IN switch diverges (route 1), OUT switch merges (route 1)
            piece_sequence.append(pair.in_switch_idx)
            route_indices.append(1)
            piece_sequence.extend(pair.branch_pieces)
            route_indices.extend([0] * len(pair.branch_pieces))
            piece_sequence.append(pair.out_switch_idx)
            route_indices.append(1)

        current_pos = pair.out_position + 1

    # Add remaining main loop pieces after last switch pair
    segment = main_loop_pieces[current_pos:]
    piece_sequence.extend(segment)
    route_indices.extend([0] * len(segment))

    # Compute FK chain using route-aware catalog lookup
    states = _compute_path_fk(piece_sequence, route_indices, catalog)

    # Compute closure metrics
    closure_error, angle_error = _compute_closure_metrics(states)

    return TraversalPath(
        path_id=0,
        route_choices=route_choices,
        piece_sequence=piece_sequence,
        states=states,
        closure_error=closure_error,
        angle_error=angle_error,
    )


def _compute_path_fk(
    piece_sequence: List[int],
    route_indices: List[int],
    catalog: TrackCatalog,
) -> NDArray[np.float64]:
    """Compute FK states for a path using route-aware catalog lookup.

    Each piece uses the FK for its assigned route index:
    - route 0 = default (straight-through for switches, normal for other pieces)
    - route 1 = diverge (for IN switches) or merge (for OUT switches)

    Args:
        piece_sequence: Ordered piece indices for this path.
        route_indices: Route index per piece (parallel to piece_sequence).
        catalog: Track catalog with get_fk_route().

    Returns:
        States array (n+1, 3) with [x, y, theta].
    """
    if not piece_sequence:
        return np.zeros((1, 3), dtype=np.float64)

    n = len(piece_sequence)
    states = np.zeros((n + 1, 3), dtype=np.float64)

    for i in range(n):
        fk = catalog.get_fk_route(piece_sequence[i], route_indices[i])
        dx, dy, dtheta = fk[0], fk[1], fk[2]

        theta_rad = np.radians(states[i, 2])
        cos_t = np.cos(theta_rad)
        sin_t = np.sin(theta_rad)

        states[i + 1, 0] = states[i, 0] + dx * cos_t - dy * sin_t
        states[i + 1, 1] = states[i, 1] + dx * sin_t + dy * cos_t
        states[i + 1, 2] = states[i, 2] + dtheta

    return states


def _compute_closure_metrics(states: NDArray[np.float64]) -> Tuple[float, float]:
    """Compute closure error and angle error from states.

    Args:
        states: FK states array (n+1, 3).

    Returns:
        (closure_error, angle_error) tuple.
    """
    if len(states) <= 1:
        return (0.0, 360.0)

    final = states[-1]
    closure_error = float(np.sqrt(final[0]**2 + final[1]**2))

    total_angle = abs(final[2])
    if total_angle == 0:
        angle_error = 360.0
    else:
        remainder = total_angle % 360
        angle_error = min(remainder, 360 - remainder)

    return (closure_error, angle_error)


def _normalize_angle(angle: float) -> float:
    """Normalize angle to [-180, 180) range.

    Args:
        angle: Angle in degrees.

    Returns:
        Normalized angle.
    """
    while angle >= 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle


# =============================================================================
# Phase 4: Crossing Overlay (Placeholder for Future Implementation)
# =============================================================================

# Crossing piece indices
CROSS_90_INDEX = 4
DOUBLE_CROSSOVER_INDEX = 9

# Note: Phase 4 crossing overlay is deferred.
# Crossings require two paths to physically intersect at perpendicular angles.
# This would create additional loops in the track graph.
# For now, crossings are not supported in the multi-path architecture.


# =============================================================================
# Helper Functions
# =============================================================================

def _convert_inventory_to_index(
    inventory: Dict[str, int],
    catalog: TrackCatalog,
) -> Dict[int, int]:
    """Convert inventory from piece_id to piece_index.

    Args:
        inventory: Inventory by piece ID {piece_id: count}.
        catalog: Track catalog.

    Returns:
        Inventory by piece index {piece_index: count}.
    """
    result = {}
    for piece_id, count in inventory.items():
        idx = catalog._id_to_index.get(piece_id)
        if idx is not None:
            result[idx] = count
    return result


# =============================================================================
# Utility Functions
# =============================================================================

def compute_angular_deficit(
    chromosome: NDArray,
    catalog: TrackCatalog,
    target: float = 360.0,
) -> float:
    """Compute angular deficit for main loop closure.

    Args:
        chromosome: Full chromosome array.
        catalog: Track catalog.
        target: Target total angle (360 for single loop).

    Returns:
        Degrees needed (positive = need more turning).
    """
    main_loop = get_active_main_loop(chromosome)
    if len(main_loop) == 0:
        return target

    fk_deltas = catalog.get_fk(main_loop.astype(np.int32))
    total_angle = np.sum(np.abs(fk_deltas[:, 2]))
    return target - total_angle


def estimate_pieces_for_closure(
    deficit: float,
    catalog: TrackCatalog,
    curve_index: int = 2,  # R40_LEFT default
) -> int:
    """Estimate number of curve pieces needed to close angular deficit.

    Args:
        deficit: Angular deficit in degrees.
        catalog: Track catalog.
        curve_index: Piece index for typical curve.

    Returns:
        Estimated number of pieces needed.
    """
    if deficit <= 0:
        return 0

    # Get angle per curve piece
    fk = catalog._fk_table[curve_index]
    angle_per_piece = abs(fk[2])

    if angle_per_piece <= 0:
        return 0

    return int(np.ceil(deficit / angle_per_piece))
