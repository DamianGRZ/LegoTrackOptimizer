"""Random-key construction decoder for multi-segment chromosomes.

Decodes [0,1] random-key chromosomes into Layout objects via construction-based approach.
Implements Bean's random-key decoding with port-priority queue.
Guarantees >99% feasibility by construction - any random chromosome produces
an evaluable Layout.

Architecture: Port-Priority Construction with Paired-Switch Continuous Paths
- Each branch requires paired switches (IN diverges, OUT merges)
- All traversal paths are computed as continuous FK chains
- With N switch pairs, there are 2^N possible paths through the track
- Constraint: ALL paths must form closed loops

Phases:
1. Main loop construction - port-priority FK with dynamic inventory selection
2. Switch pair extraction - identify paired IN/OUT switches from chromosome
3. Path enumeration - generate all 2^N traversal paths
4. Per-path FK computation - continuous FK chain for each path
"""

from dataclasses import dataclass, field
from enum import IntEnum
from itertools import product
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from numpy.typing import NDArray

from .data import TrackCatalog
from .encoding import (
    B_MAX,
    C_MAX,
    INACTIVE,
    L_MAX,
    RK_INACTIVE_THRESHOLD,
    get_active_main_loop,
    get_branch_in_position,
    get_branch_slot,
    get_branch_slots,
    get_branch_template_params,
    get_crossing_overlay,
    get_crossing_pair,
    get_main_loop,
    get_piece_keys,
    get_priority_keys,
    get_start_position,
    get_switch_mask,
    is_branch_active,
    is_branch_valid,
    is_crossing_active,
    rk_to_piece_index,
    rk_to_position,
)
from .geometry import Layout, compute_fk_chain
from .intersection import CROSS_90_INDEX, find_crossing_pairs
# templates.py no longer used by decoder — port-based branch computation (Phase B)
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

    # Boundary for RK position scaling
    boundary_min_x: float = -100.0
    boundary_max_x: float = 100.0
    boundary_min_y: float = -100.0
    boundary_max_y: float = 100.0

    def scale_position(self, rk_x: float, rk_y: float) -> Tuple[float, float]:
        """Scale [0,1] RK position values to world coordinates."""
        x = self.boundary_min_x + rk_x * (self.boundary_max_x - self.boundary_min_x)
        y = self.boundary_min_y + rk_y * (self.boundary_max_y - self.boundary_min_y)
        return (x, y)


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


# =============================================================================
# Port-Priority Queue Structures (Research-Based)
# =============================================================================

class PortType(IntEnum):
    """Port classification for construction decoder."""
    STANDARD = 0       # Normal 2-port piece output
    SWITCH_DIVERGE = 1 # Switch port 2 (diverging branch)
    SWITCH_MERGE = 2   # Switch port for merging (OUT switch)


@dataclass
class OpenPort:
    """An open connection point awaiting extension or closure.

    Used in port-priority construction where gene values control
    which open port to extend next.
    """
    x: float           # World X position
    y: float           # World Y position
    theta: float       # Heading in degrees
    priority: float    # Gene-driven priority (higher = process first)
    source_idx: int    # Index in constructed sequence
    port_type: PortType = PortType.STANDARD

    def distance_to(self, other: 'OpenPort') -> float:
        """Compute Euclidean distance to another port."""
        return float(np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2))

    def can_connect_to(self, other: 'OpenPort', pos_tol: float = 8.0, ang_tol: float = 15.0) -> bool:
        """Check if this port can connect to another (for closure).

        Ports can connect if they are close together and facing
        opposite directions (180 degrees apart).
        """
        if self.distance_to(other) > pos_tol:
            return False
        angle_diff = abs((self.theta - other.theta + 180) % 360 - 180)
        return abs(angle_diff - 180) < ang_tol


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

    # Port-priority queue (research-based)
    open_ports: List[OpenPort] = field(default_factory=list)

    def pop_highest_priority_port(self) -> Optional[OpenPort]:
        """Remove and return the highest priority open port."""
        if not self.open_ports:
            return None
        self.open_ports.sort(key=lambda p: -p.priority)
        return self.open_ports.pop(0)

    def find_closure_candidate(self, pos_tol: float = 8.0, ang_tol: float = 15.0) -> Optional[Tuple[OpenPort, OpenPort]]:
        """Find two open ports that can connect (closure-seeking).

        Returns the first pair of ports that can connect, or None if no
        closure opportunity exists.
        """
        for i, p1 in enumerate(self.open_ports):
            for p2 in self.open_ports[i + 1:]:
                if p1.can_connect_to(p2, pos_tol, ang_tol):
                    return (p1, p2)
        return None

    def add_open_port(self, port: OpenPort) -> None:
        """Add an open port to the queue."""
        self.open_ports.append(port)

    def remove_port(self, port: OpenPort) -> None:
        """Remove a specific port from the queue."""
        if port in self.open_ports:
            self.open_ports.remove(port)

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

    # Phase 1.5: Replace self-intersections with CROSS_90 pieces
    _apply_crossing_repair(state, catalog, inventory_by_index)

    # Phase 2: Extract switch pairs from branch slots
    switch_pairs = _extract_switch_pairs(chromosome, state, catalog, inventory_by_index, config)

    # Phase 3: Enumerate and compute all traversal paths
    multi_path = _build_multi_path_layout(
        chromosome, state, switch_pairs, catalog, inventory_by_index, config
    )

    # Phase 4: Build secondary loops through crossings
    secondary_loops = _build_secondary_loops(
        chromosome, state, catalog, inventory_by_index, config
    )
    for sec_path in secondary_loops:
        multi_path.paths.append(sec_path)
    # Store secondary closure for constraint
    if secondary_loops:
        multi_path.secondary_closure_error = min(p.closure_error for p in secondary_loops)
    else:
        multi_path.secondary_closure_error = 0.0

    # Auto-center layout within boundary instead of using RK start position.
    # This eliminates boundary violations for correctly-shaped layouts and
    # removes 2 genes from the effective search space.
    # Closure and angle are translation-invariant — centering doesn't affect them.
    all_x = np.concatenate([p.states[:, 0] for p in multi_path.paths if len(p.states) > 0])
    all_y = np.concatenate([p.states[:, 1] for p in multi_path.paths if len(p.states) > 0])

    if len(all_x) > 0:
        layout_cx = (all_x.min() + all_x.max()) / 2
        layout_cy = (all_y.min() + all_y.max()) / 2
        boundary_cx = (config.boundary_min_x + config.boundary_max_x) / 2
        boundary_cy = (config.boundary_min_y + config.boundary_max_y) / 2
        shift_x = boundary_cx - layout_cx
        shift_y = boundary_cy - layout_cy

        multi_path.start_position = (shift_x, shift_y)
        for path in multi_path.paths:
            if len(path.states) > 0:
                path.states[:, 0] += shift_x
                path.states[:, 1] += shift_y

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
    """Phase 1: Build main loop from chromosome genes using Random-Key decoding.

    Process genes left-to-right using turtle-graphics FK.
    Skip genes where no pieces remain available.

    Args:
        chromosome: Full chromosome array with [0,1] RK values.
        catalog: Track catalog.
        inventory_by_index: Available inventory by piece index.
        config: Decoder configuration.

    Returns:
        DecoderState with placed pieces and states.
    """
    state = DecoderState()
    state.states.append((0.0, 0.0, 0.0))  # Initial state at origin

    # Track signed angle for closure-aware piece filtering
    signed_angle = 0.0

    piece_keys = get_piece_keys(chromosome)

    for rk_value in piece_keys:
        if rk_value < RK_INACTIVE_THRESHOLD:
            continue

        available_pieces = _get_available_pieces(
            inventory_by_index, state.inventory_used,
            fk_table=catalog._fk_table,
            signed_angle=signed_angle,
            target_angle=config.target_angle,
        )
        if not available_pieces:
            continue

        # Exclude fork/merge pieces from natural placement — reserved for injection
        available_pieces = [
            p for p in available_pieces
            if p not in catalog._fork_indices and p not in catalog._merge_indices
        ]
        if not available_pieces:
            continue

        piece_idx = rk_to_piece_index(float(rk_value), available_pieces)

        if piece_idx == INACTIVE or piece_idx < 0:
            continue

        if piece_idx >= len(catalog._fk_table):
            continue

        fk = catalog._fk_table[piece_idx]
        dx, dy, dtheta = fk[0], fk[1], fk[2]

        new_cumulative = state.cumulative_angle + abs(dtheta)
        if new_cumulative > config.max_cumulative_angle:
            continue

        theta_rad = np.radians(state.theta)
        cos_t = np.cos(theta_rad)
        sin_t = np.sin(theta_rad)

        new_x = state.x + dx * cos_t - dy * sin_t
        new_y = state.y + dx * sin_t + dy * cos_t
        new_theta = state.theta + dtheta

        # No position-budget check — BRKGA evolves compact layouts
        # through closure-scaled objective and heuristic seeds

        state.x = new_x
        state.y = new_y
        state.theta = new_theta
        state.cumulative_angle += abs(dtheta)
        signed_angle += dtheta

        state.use_piece(piece_idx)
        state.states.append((new_x, new_y, new_theta))

        if piece_idx in (5, 6, 7, 8):
            position = len(state.piece_indices) - 1
            state.switch_port2_positions.append(position)

        if piece_idx == 4:
            position = len(state.piece_indices) - 1
            state.crossing_positions.append((position, piece_idx))

        # NOTE: Early closure check removed - process ALL genes to ensure
        # switches and branches are included. Optimizer handles closure.

    return state


# =============================================================================
# Phase 1.5: Crossing Repair — replace self-intersections with CROSS_90
# =============================================================================


def _apply_crossing_repair(
    state: DecoderState,
    catalog: TrackCatalog,
    inventory_by_index: Dict[int, int],
    angle_tolerance: float = 15.0,
) -> None:
    """Replace self-intersections with CROSS_90 pieces.

    When the main loop crosses itself at ~90°, replace the piece at one
    crossing position with CROSS_90 and remove the piece at the other
    (they are physically the same crossing piece). Recomputes FK chain.

    Mutates state in-place.
    """
    n = len(state.piece_indices)
    if n < 4:
        return

    cross_avail = (
        inventory_by_index.get(CROSS_90_INDEX, 0)
        - state.get_usage(CROSS_90_INDEX)
    )
    if cross_avail <= 0:
        return

    # Build numpy states array from tuple list
    states_array = np.array(state.states, dtype=np.float64)

    pairs = find_crossing_pairs(states_array, state.piece_indices)
    if not pairs:
        return

    # Greedy assignment: best 90° crossings first
    claimed: set = set()
    replacements: list = []  # (pos_i → CROSS_90)
    removals: list = []      # pos_j to delete

    for pos_i, pos_j, angle_diff in pairs:
        if abs(angle_diff - 90.0) > angle_tolerance:
            continue
        if cross_avail <= 0:
            break
        if pos_i in claimed or pos_j in claimed:
            continue

        claimed.add(pos_i)
        claimed.add(pos_j)
        replacements.append(pos_i)
        removals.append(pos_j)
        cross_avail -= 1

    if not replacements:
        return

    # --- Apply inventory changes ---
    for pos_i, pos_j in zip(replacements, removals):
        orig_i = state.piece_indices[pos_i]
        orig_j = state.piece_indices[pos_j]

        # Return originals
        state.inventory_used[orig_i] = state.inventory_used.get(orig_i, 0) - 1
        state.inventory_used[orig_j] = state.inventory_used.get(orig_j, 0) - 1

        # Consume one CROSS_90
        state.inventory_used[CROSS_90_INDEX] = (
            state.inventory_used.get(CROSS_90_INDEX, 0) + 1
        )

        # Replace piece at pos_i
        state.piece_indices[pos_i] = CROSS_90_INDEX

    # --- Remove pieces at removal positions (reverse order) ---
    for pos_j in sorted(removals, reverse=True):
        state.piece_indices.pop(pos_j)
        # states list has n+1 entries; entry pos_j+1 corresponds to piece pos_j's exit
        if pos_j + 1 < len(state.states):
            state.states.pop(pos_j + 1)

    # --- Shift switch/crossing position tracking ---
    def _shift(positions: list, removed: list) -> list:
        removed_sorted = sorted(removed)
        result = []
        for p in positions:
            if p in removed:
                continue
            shift = sum(1 for r in removed_sorted if r < p)
            result.append(p - shift)
        return result

    state.switch_port2_positions = _shift(
        state.switch_port2_positions, removals,
    )
    old_cross = state.crossing_positions
    shifted_cross_pos = _shift([p for p, _ in old_cross], removals)
    state.crossing_positions = [
        (p, idx) for p, idx in zip(shifted_cross_pos, [idx for _, idx in old_cross])
    ]

    # Add new crossing positions from replacements
    shifted_replacements = _shift(replacements, removals)
    for p in shifted_replacements:
        state.crossing_positions.append((p, CROSS_90_INDEX))

    # --- Recompute FK chain ---
    fk_deltas = np.array(
        [catalog._fk_table[idx] for idx in state.piece_indices],
        dtype=np.float64,
    )
    new_states = compute_fk_chain(fk_deltas)

    # Rebuild state.states as list of tuples
    state.states = [(float(new_states[i, 0]), float(new_states[i, 1]),
                     float(new_states[i, 2])) for i in range(len(new_states))]

    # Update turtle state to final position
    final = new_states[-1]
    state.x = float(final[0])
    state.y = float(final[1])
    state.theta = float(final[2])

    # Recompute cumulative angle
    state.cumulative_angle = float(np.sum(np.abs(fk_deltas[:, 2])))
    state.pieces_placed = len(state.piece_indices)


# =============================================================================
# Phase 2: Port-Based Switch Pair Extraction
# =============================================================================


def _extract_switch_pairs(
    chromosome: NDArray,
    state: DecoderState,
    catalog: TrackCatalog,
    inventory_by_index: Dict[int, int],
    config: DecoderConfig,
) -> List[SwitchPair]:
    """Extract switch pairs using port-based role classification.

    Unified single-pass approach (replaces template-based Pass 1 + legacy Pass 2):
    1. Scan main loop for fork pieces (catalog._fork_indices)
    2. For each fork, search downstream for compatible merge (catalog._compatible_pairs)
    3. Check main section straightness, compute branch from FK, verify inventory

    Args:
        chromosome: Full chromosome array (branch slots still influence placement).
        state: Decoder state with main loop pieces.
        catalog: Track catalog with role classification.
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
    used_positions: Set[int] = set()

    # =========================================================================
    # Pass 1: Branch slot injection (chromosome-driven switch placement)
    # Reads branch slot genes [200-216] to inject switches at straight sections
    # =========================================================================
    for slot_idx in range(B_MAX):
        in_pos, handedness, n_straights, active = get_branch_template_params(
            chromosome, slot_idx
        )
        if not active or in_pos < 0 or in_pos >= n_main_loop:
            continue

        # Determine fork/merge indices from handedness using catalog roles
        if handedness in (0, 2):  # LEFT or LEFT_REVERSE
            fork_candidates = [
                idx for idx in catalog._fork_indices
                if catalog.get_alternate_route(idx) is not None
                and catalog.get_alternate_route(idx).dtheta > 0
            ]
        else:  # RIGHT or RIGHT_REVERSE
            fork_candidates = [
                idx for idx in catalog._fork_indices
                if catalog.get_alternate_route(idx) is not None
                and catalog.get_alternate_route(idx).dtheta < 0
            ]

        if not fork_candidates:
            continue
        fork_idx = fork_candidates[0]
        compatible = catalog._compatible_pairs.get(fork_idx, [])
        if not compatible:
            continue
        merge_idx = compatible[0]

        # Find IN position: nearest straight that can be replaced
        actual_in = _find_straight_near(state.piece_indices, in_pos, used_positions)
        if actual_in is None:
            continue

        # Check switch inventory
        fork_avail = inventory_by_index.get(fork_idx, 0) - state.inventory_used.get(fork_idx, 0)
        merge_avail = inventory_by_index.get(merge_idx, 0) - state.inventory_used.get(merge_idx, 0)
        if fork_avail < 1 or merge_avail < 1:
            continue

        # Find OUT position: near-straight section downstream
        out_pos = _find_out_for_injection(
            actual_in, state.piece_indices, used_positions,
            fk_table=catalog._fk_table,
        )
        if out_pos is None:
            continue

        # Compute branch from FK
        main_dist = _compute_main_fk_distance(
            state.piece_indices, actual_in, out_pos, catalog._fk_table
        )
        branch = compute_branch_from_fk(fork_idx, merge_idx, main_dist, catalog)
        if branch is None:
            continue

        # Check branch inventory
        if not _check_branch_inventory(branch, inventory_by_index, state.inventory_used):
            continue

        # INJECT: replace straights with switches
        original_in = state.piece_indices[actual_in]
        original_out = state.piece_indices[out_pos]
        state.piece_indices[actual_in] = fork_idx
        state.piece_indices[out_pos] = merge_idx

        # Return original pieces to inventory
        if original_in >= 0:
            state.inventory_used[original_in] = max(
                0, state.inventory_used.get(original_in, 1) - 1
            )
        if original_out >= 0:
            state.inventory_used[original_out] = max(
                0, state.inventory_used.get(original_out, 1) - 1
            )

        # Track switch usage
        state.inventory_used[fork_idx] = state.inventory_used.get(fork_idx, 0) + 1
        state.inventory_used[merge_idx] = state.inventory_used.get(merge_idx, 0) + 1

        # Create switch pair
        pair = SwitchPair(
            pair_id=pair_id,
            in_position=actual_in,
            out_position=out_pos,
            in_switch_idx=fork_idx,
            out_switch_idx=merge_idx,
            branch_pieces=branch,
        )
        switch_pairs.append(pair)
        pair_id += 1
        used_positions.add(actual_in)
        used_positions.add(out_pos)
        state.connected_port2_positions.add(actual_in)
        state.connected_port2_positions.add(out_pos)

        # Consume branch inventory
        for p in branch:
            state.inventory_used[p] = state.inventory_used.get(p, 0) + 1

    # =========================================================================
    # Pass 2: Scan for naturally-placed switches (fallback)
    # =========================================================================

    # Find all fork and merge positions in main loop
    fork_positions = []
    merge_positions = []

    for pos, piece_idx in enumerate(state.piece_indices):
        if piece_idx in catalog._fork_indices:
            fork_positions.append((pos, piece_idx))
        elif piece_idx in catalog._merge_indices:
            merge_positions.append((pos, piece_idx))

    # For each fork, find the best compatible merge downstream
    for fork_pos, fork_idx in fork_positions:
        if fork_pos in used_positions:
            continue

        compatible_merges = catalog._compatible_pairs.get(fork_idx, [])
        if not compatible_merges:
            continue

        best_pair = None
        best_gap = float('inf')

        for merge_pos, merge_idx in merge_positions:
            # Merge must come after fork and not already used
            if merge_pos <= fork_pos or merge_pos in used_positions:
                continue
            if merge_idx not in compatible_merges:
                continue

            # Check main section straightness (critical — prevents branch endpoint gap)
            if not _main_section_is_straight(
                state.piece_indices, fork_pos, merge_pos, catalog._fk_table
            ):
                continue

            # Compute main FK distance between fork and merge
            main_dist = _compute_main_fk_distance(
                state.piece_indices, fork_pos, merge_pos, catalog._fk_table
            )

            # Compute branch pieces from FK
            branch = compute_branch_from_fk(fork_idx, merge_idx, main_dist, catalog)
            if branch is None:
                continue

            # Check branch inventory
            if not _check_branch_inventory(branch, inventory_by_index, state.inventory_used):
                continue

            # Compute endpoint gap for ranking candidates
            gap = abs(main_dist - sum(float(catalog._fk_table[p, 0]) for p in branch))
            if gap < best_gap:
                best_gap = gap
                best_pair = (merge_pos, merge_idx, branch, main_dist)

        if best_pair is None:
            continue

        merge_pos, merge_idx, branch_pieces, main_dist = best_pair

        # Create switch pair
        pair = SwitchPair(
            pair_id=pair_id,
            in_position=fork_pos,
            out_position=merge_pos,
            in_switch_idx=fork_idx,
            out_switch_idx=merge_idx,
            branch_pieces=branch_pieces,
        )
        switch_pairs.append(pair)
        pair_id += 1

        # Mark positions as used
        used_positions.add(fork_pos)
        used_positions.add(merge_pos)

        # Mark port 2 as connected
        state.connected_port2_positions.add(fork_pos)
        state.connected_port2_positions.add(merge_pos)

        # Consume branch inventory
        for piece_idx in branch_pieces:
            state.inventory_used[piece_idx] = state.inventory_used.get(piece_idx, 0) + 1

    # Track all multi-port positions for loose port counting
    for pos, piece_idx in enumerate(state.piece_indices):
        if piece_idx in catalog._fork_indices or piece_idx in catalog._merge_indices:
            if pos not in state.switch_port2_positions:
                state.switch_port2_positions.append(pos)

    return switch_pairs


# =============================================================================
# Port-Based Branch Computation (Phase B — replaces template-based approach)
# =============================================================================


def _find_straight_near(
    piece_indices: List[int],
    target_pos: int,
    used_positions: Set[int],
    search_range: int = 5,
) -> Optional[int]:
    """Find nearest straight piece to target position for switch injection."""
    STRAIGHTS = {0, 1}  # STR_16, STR_24
    n = len(piece_indices)
    for offset in range(search_range):
        for candidate in [target_pos + offset, target_pos - offset]:
            if 0 <= candidate < n and candidate not in used_positions:
                if piece_indices[candidate] in STRAIGHTS:
                    return candidate
    return None


def _find_out_for_injection(
    in_pos: int,
    piece_indices: List[int],
    used_positions: Set[int],
    fk_table: Optional[NDArray] = None,
    min_gap: int = 3,
    max_gap: int = 12,
) -> Optional[int]:
    """Find OUT position downstream from IN with near-straight section between.

    The OUT position must be a straight piece. The section between IN and OUT
    must have ≤5° net angle (allows some curves if they cancel).
    """
    STRAIGHTS = {0, 1}
    n = len(piece_indices)

    for out_pos in range(in_pos + min_gap, min(in_pos + max_gap, n)):
        if out_pos in used_positions:
            continue
        if piece_indices[out_pos] not in STRAIGHTS:
            continue
        # Check section straightness (allows canceling curves)
        if fk_table is not None:
            if _main_section_is_straight(piece_indices, in_pos, out_pos, fk_table, tolerance=5.0):
                return out_pos
        else:
            # Fallback: require all-straight
            section = piece_indices[in_pos + 1 : out_pos]
            if section and all(p in STRAIGHTS for p in section):
                return out_pos

    return None


def _main_section_is_straight(
    main_pieces: List[int],
    fork_pos: int,
    merge_pos: int,
    fk_table: NDArray,
    tolerance: float = 5.0,
) -> bool:
    """Check that main loop between fork and merge has ~0° net angle.

    Branch geometry assumes the main section is straight (parallel to branch).
    If the main section curves, the branch endpoint will miss the merge port.

    Args:
        main_pieces: Main loop piece indices.
        fork_pos: Position of fork (IN switch) in main loop.
        merge_pos: Position of merge (OUT switch) in main loop.
        fk_table: FK table from catalog.
        tolerance: Maximum allowed net angle in degrees.

    Returns:
        True if main section is approximately straight.
    """
    between = main_pieces[fork_pos + 1 : merge_pos]
    if not between:
        return True
    total_angle = sum(float(fk_table[int(p), 2]) for p in between)
    return abs(total_angle) <= tolerance


def _compute_main_fk_distance(
    main_pieces: List[int],
    fork_pos: int,
    merge_pos: int,
    fk_table: NDArray,
) -> float:
    """Compute forward FK distance along main loop between fork and merge.

    Args:
        main_pieces: Main loop piece indices.
        fork_pos: Fork position.
        merge_pos: Merge position.
        fk_table: FK table from catalog.

    Returns:
        Total forward distance in studs (sum of dx values).
    """
    between = main_pieces[fork_pos + 1 : merge_pos]
    return sum(float(fk_table[int(p), 0]) for p in between)


def compute_branch_from_fk(
    fork_idx: int,
    merge_idx: int,
    main_distance: float,
    catalog: 'TrackCatalog',
) -> Optional[List[int]]:
    """Compute branch pieces from FK deltas of fork/merge alternate routes.

    Replaces template-based compute_branch_pieces(). Derives everything
    from piece FK data — works for ANY fork/merge combination.

    Branch structure: [approach_curve] + [N × STRAIGHT_16] + [return_curve]
    - Approach curve: matches fork's diverge angle direction
    - Return curve: matches merge's convergence angle direction

    Args:
        fork_idx: Fork piece index (e.g., SWITCH_LEFT_IN=5).
        merge_idx: Merge piece index (e.g., SWITCH_LEFT_OUT=6).
        main_distance: FK distance along main loop between fork and merge.
        catalog: Track catalog.

    Returns:
        List of piece indices for the branch, or None if impossible.
    """
    fork_alt = catalog.get_alternate_route(fork_idx)
    merge_alt = catalog.get_alternate_route(merge_idx)
    if not fork_alt or not merge_alt:
        return None

    # Determine approach/return curve from diverge angle
    # Fork diverges left (+dtheta) → approach with R40_RIGHT (matching curve on branch side)
    # Fork diverges right (-dtheta) → approach with R40_LEFT
    R40_LEFT = 2
    R40_RIGHT = 3
    STRAIGHT_16 = 0

    if fork_alt.dtheta > 0:
        approach_idx = R40_RIGHT
        return_idx = R40_LEFT
    else:
        approach_idx = R40_LEFT
        return_idx = R40_RIGHT

    # Compute how many straights fill the forward gap
    approach_dx = float(catalog._fk_table[approach_idx, 0])
    return_dx = float(catalog._fk_table[return_idx, 0])
    straight_dx = float(catalog._fk_table[STRAIGHT_16, 0])

    if straight_dx <= 0:
        return None

    available = main_distance - approach_dx - return_dx
    if available < -straight_dx:
        return None  # Branch can't fit

    n_straights = max(0, round(available / straight_dx))

    return [approach_idx] + [STRAIGHT_16] * n_straights + [return_idx]


def _verify_branch_endpoint(
    branch_pieces: List[int],
    fork_idx: int,
    merge_idx: int,
    main_distance: float,
    catalog: 'TrackCatalog',
    tolerance: float = 8.0,
) -> bool:
    """Verify branch FK endpoint is within tolerance of merge port.

    Computes the full branch FK chain (diverge + branch + converge)
    and checks if the endpoint matches what the merge port expects.

    Args:
        branch_pieces: Branch piece indices.
        fork_idx: Fork piece index.
        merge_idx: Merge piece index.
        main_distance: Main loop FK distance between fork and merge.
        catalog: Track catalog.
        tolerance: Maximum allowed endpoint gap in studs.

    Returns:
        True if branch endpoint is within tolerance.
    """
    from .geometry import compute_fk_chain

    fork_alt = catalog.get_alternate_route(fork_idx)
    merge_alt = catalog.get_alternate_route(merge_idx)
    if not fork_alt or not merge_alt:
        return False

    # Build full branch FK chain: diverge route + branch pieces
    # Start from fork's diverge endpoint
    indices = np.array(branch_pieces, dtype=np.int32)
    deltas = catalog._fk_table[indices]

    # Prepend the fork's diverge delta
    fork_delta = np.array([[fork_alt.dx, fork_alt.dy, fork_alt.dtheta]])
    full_deltas = np.vstack([fork_delta, deltas])
    states = compute_fk_chain(full_deltas)

    branch_end_x = states[-1, 0]
    branch_end_y = states[-1, 1]

    # Expected: merge port entry is at (main_distance + fork_default_dx, 0)
    # because the main loop goes straight for main_distance from fork to merge
    fork_default = catalog._fk_table[fork_idx]
    expected_x = fork_default[0] + main_distance
    expected_y = 0.0  # Merge entry is on the main loop axis

    gap = np.sqrt((branch_end_x - expected_x) ** 2 + (branch_end_y - expected_y) ** 2)
    return gap <= tolerance


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
    required: Dict[int, int] = {}
    for piece_idx in branch_pieces:
        required[piece_idx] = required.get(piece_idx, 0) + 1

    for piece_idx, count in required.items():
        avail = available.get(piece_idx, 0)
        already_used = used.get(piece_idx, 0)
        if already_used + count > avail:
            return False

    return True


# =============================================================================
# Phase 3: Multi-Path Layout Construction
# =============================================================================

# =============================================================================
# Phase 3: Secondary Loop Construction (CROSS_90 Support)
# =============================================================================

SECONDARY_LOOP_START = 100  # Genes [100-150) for secondary loop piece keys
SECONDARY_LOOP_SIZE = 50


def _build_secondary_loops(
    chromosome: NDArray,
    state: DecoderState,
    catalog: TrackCatalog,
    inventory_by_index: Dict[int, int],
    config: DecoderConfig,
) -> List[TraversalPath]:
    """Build secondary loops through crossing pieces in the main loop.

    For each CROSS_90, constructs an independent closed loop using
    the crossing's perpendicular route (route 1) and remaining inventory.

    Args:
        chromosome: Full chromosome array.
        state: Decoder state with main loop pieces and FK states.
        catalog: Track catalog.
        inventory_by_index: Available inventory.
        config: Decoder configuration.

    Returns:
        List of secondary TraversalPath objects.
    """
    secondary_paths = []

    if not state.crossing_positions:
        return secondary_paths

    # Get secondary loop piece keys from chromosome
    sec_keys = chromosome[SECONDARY_LOOP_START:SECONDARY_LOOP_START + SECONDARY_LOOP_SIZE]

    for cross_pos, cross_idx in state.crossing_positions:
        # Get main loop FK state at crossing position
        if cross_pos >= len(state.states):
            continue

        piece_x, piece_y, piece_theta = state.states[cross_pos]

        # Port 2 world position: local (8, -8) rotated by piece heading
        theta_rad = np.radians(piece_theta)
        cos_t, sin_t = np.cos(theta_rad), np.sin(theta_rad)
        port2_x = piece_x + 8.0 * cos_t + 8.0 * sin_t  # (8, -8) rotated
        port2_y = piece_y + 8.0 * sin_t - 8.0 * cos_t

        # Secondary loop entry heading: train enters from south heading north
        entry_heading = piece_theta + 90.0

        # Build secondary loop from remaining inventory
        sec_pieces = _construct_secondary_loop(
            sec_keys, cross_idx, catalog, inventory_by_index,
            state.inventory_used, config,
        )

        if not sec_pieces:
            continue

        # Build FK chain for secondary loop
        # First piece: crossing route 1 (FK 16,0,0 in train's frame)
        piece_indices = [cross_idx] + sec_pieces
        route_indices = [1] + [0] * len(sec_pieces)  # Route 1 for crossing, 0 for others

        # Compute FK chain starting at port 2 position/heading
        fk_deltas = catalog.get_fk_with_routes(
            np.array(piece_indices, dtype=np.int32),
            np.array(route_indices, dtype=np.int32),
        )
        states = compute_fk_chain(fk_deltas)

        # Translate to world position
        states[:, 0] += port2_x
        states[:, 1] += port2_y
        # Rotate by entry heading (FK chain starts at heading 0, we need entry_heading)
        if abs(entry_heading) > 0.01:
            rot_rad = np.radians(entry_heading)
            cos_r, sin_r = np.cos(rot_rad), np.sin(rot_rad)
            # Rotate around port2 position
            dx = states[:, 0] - port2_x
            dy = states[:, 1] - port2_y
            states[:, 0] = port2_x + dx * cos_r - dy * sin_r
            states[:, 1] = port2_y + dx * sin_r + dy * cos_r
            states[:, 2] += entry_heading

        # Compute closure
        closure_error, angle_error = _compute_closure_metrics(states)

        path = TraversalPath(
            path_id=len(secondary_paths) + 100,  # Offset to distinguish from main paths
            route_choices=(),
            piece_sequence=piece_indices,
            states=states,
            closure_error=closure_error,
            angle_error=angle_error,
        )
        secondary_paths.append(path)

        # Mark crossing ports as connected (removes loose port penalty)
        state.connected_port2_positions.add(cross_pos)

        # Track secondary loop piece usage
        for p in sec_pieces:
            state.inventory_used[p] = state.inventory_used.get(p, 0) + 1

    return secondary_paths


def _construct_secondary_loop(
    piece_keys: NDArray,
    cross_idx: int,
    catalog: TrackCatalog,
    inventory_by_index: Dict[int, int],
    used: Dict[int, int],
    config: DecoderConfig,
) -> List[int]:
    """Construct secondary loop piece sequence from RK keys.

    Similar to main loop construction but uses remaining inventory
    and excludes crossing pieces (already placed).
    """
    pieces = []
    local_used = dict(used)  # Copy to track secondary usage
    signed_angle = 0.0
    CROSS_INDICES = catalog._crossing_indices

    for rk_value in piece_keys:
        if rk_value < RK_INACTIVE_THRESHOLD:
            continue

        available = _get_available_pieces(
            inventory_by_index, local_used,
            fk_table=catalog._fk_table,
            signed_angle=signed_angle,
            target_angle=config.target_angle,
        )
        if not available:
            continue

        # Exclude crossing and switch pieces from secondary loop
        available = [
            p for p in available
            if p not in CROSS_INDICES
            and p not in catalog._fork_indices
            and p not in catalog._merge_indices
        ]
        if not available:
            continue

        piece_idx = rk_to_piece_index(float(rk_value), available)
        if piece_idx == INACTIVE or piece_idx < 0:
            continue

        signed_angle += catalog._fk_table[piece_idx, 2]
        local_used[piece_idx] = local_used.get(piece_idx, 0) + 1
        pieces.append(piece_idx)

    return pieces


# =============================================================================
# Multi-Path Layout Construction
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

    # Loose port count: only count genuinely broken connections.
    # Unpaired switches are NOT loose — they operate in straight-through mode
    # (port 2 simply not used, like a switch with the lever set to "straight").
    # Only crossings have truly loose ports (perpendicular route can't connect).
    crossing_loose = len(state.crossing_positions) * 2
    loose_port_count = crossing_loose

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

    # Collect all positions absorbed by switches (skip during main loop traversal)
    absorbed_set = set()
    for pair in sorted_pairs:
        absorbed_set.update(pair.absorbed_positions)

    # Build piece sequence and route indices for this path
    piece_sequence = []
    route_indices = []
    current_pos = 0

    for i, pair in enumerate(sorted_pairs):
        choice = route_choices[i] if i < len(route_choices) else 0

        # Add main loop pieces up to IN switch position, skipping absorbed
        for pos in range(current_pos, pair.in_position):
            if pos not in absorbed_set:
                piece_sequence.append(main_loop_pieces[pos])
                route_indices.append(0)

        if choice == 0:
            # Straight-through: both switches use route 0 (default)
            piece_sequence.append(pair.in_switch_idx)
            route_indices.append(0)
            for pos in range(pair.in_position + 1, pair.out_position):
                if pos not in absorbed_set:
                    piece_sequence.append(main_loop_pieces[pos])
                    route_indices.append(0)
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

    # Add remaining main loop pieces after last switch pair, skipping absorbed
    for pos in range(current_pos, len(main_loop_pieces)):
        if pos not in absorbed_set:
            piece_sequence.append(main_loop_pieces[pos])
            route_indices.append(0)

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
# Phase 4: Crossing Overlay (Not Supported with R40 Curves)
# =============================================================================

# Crossing piece indices
CROSS_90_INDEX = 4
DOUBLE_CROSSOVER_INDEX = 9

# NOTE: CROSS_90 figure-8 layouts are geometrically impossible with R40 curves.
# The crossing's perpendicular port is only 8 studs offset, but R40 turns
# create 68+ stud offsets. Crossings always have 2 loose ports and are
# effectively banned by the loose_port constraint.


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


def _get_available_pieces(
    inventory_by_index: Dict[int, int],
    used: Dict[int, int],
    fk_table: Optional[NDArray] = None,
    signed_angle: float = 0.0,
    target_angle: float = 360.0,
) -> List[int]:
    """Build list of available piece indices from remaining inventory.

    Strict angle-budget enforcement: reject any piece that makes the target
    angle unreachable with remaining inventory. At each step, computes the
    min/max achievable angle from remaining pieces AFTER placing this one.
    If target falls outside [min, max], the piece is rejected.

    Args:
        inventory_by_index: Total available inventory {piece_idx: count}.
        used: Already used inventory {piece_idx: count}.
        fk_table: FK table for angle checking (optional).
        signed_angle: Current accumulated signed angle in degrees.
        target_angle: Target angle for closure (360.0 for closed loop).

    Returns:
        Sorted list of piece indices that have remaining inventory.
    """
    available = []
    for piece_idx, total in inventory_by_index.items():
        already_used = used.get(piece_idx, 0)
        if already_used < total:
            available.append(piece_idx)

    if not available or fk_table is None:
        return sorted(available)

    # Compute remaining angle capacity in each direction (from ALL remaining pieces)
    remaining_positive = 0.0  # Max additional positive angle achievable
    remaining_negative = 0.0  # Max additional negative angle achievable (negative number)
    for idx in available:
        angle = fk_table[idx, 2]
        remaining_count = inventory_by_index.get(idx, 0) - used.get(idx, 0)
        if angle > 0:
            remaining_positive += angle * remaining_count
        elif angle < 0:
            remaining_negative += angle * remaining_count

    filtered = []
    for idx in available:
        piece_angle = fk_table[idx, 2]

        # Straights never change angle — always OK
        if piece_angle == 0:
            filtered.append(idx)
            continue

        # After placing this piece, what's the new signed angle?
        new_signed = signed_angle + piece_angle
        # How much more do we need?
        still_needed = target_angle - new_signed

        # Remaining capacity AFTER using one of this piece
        if piece_angle > 0:
            future_pos = remaining_positive - piece_angle
            future_neg = remaining_negative
        else:
            future_pos = remaining_positive
            future_neg = remaining_negative - piece_angle

        # Can we still reach target? Target must be within [min_achievable, max_achievable]
        min_achievable = still_needed - future_pos  # Best case: use all remaining positive
        max_achievable = still_needed - future_neg  # Best case: use all remaining negative

        # If 0 is within [min_achievable, max_achievable], we CAN reach exactly target
        # min_achievable <= 0 means we have enough positive capacity
        # max_achievable >= 0 means we have enough negative capacity (or need is positive)
        if min_achievable <= 22.5 and max_achievable >= -22.5:
            filtered.append(idx)

    return sorted(filtered) if filtered else sorted(available)


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
