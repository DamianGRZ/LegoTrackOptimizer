"""CGP-inspired construction decoder for integer-encoded chromosomes.

Decodes integer node-tuple chromosomes into MultiPathLayout objects.
Each node has (piece_type, port2_conn, port3_conn) — branches and
secondary loops emerge from the connection genes.

Phases:
1. Main loop construction — sequential scan, skip inactive nodes and branch nodes
2. Branch extraction — follow port2_conn connections from switches
3. Self-intersection detection — identify geometric crossings
4. Path enumeration — generate all 2^N traversal paths with continuous FK
5. Secondary loops — follow port2/port3 connections from crossings
"""

from dataclasses import dataclass, field
from itertools import product
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from numpy.typing import NDArray

from .data import TrackCatalog
from .encoding import (
    CROSSING_INDICES,
    GENES_PER_NODE,
    IN_SWITCH_INDICES,
    INACTIVE,
    OUT_SWITCH_INDICES,
    SWITCH_INDICES,
    ChromosomeDimensions,
    get_all_piece_types,
    get_all_port2_conns,
    get_all_port3_conns,
    get_branch_nodes,
    get_node,
    get_piece_type,
    get_port2_conn,
    get_port3_conn,
)
from .geometry import Layout, compute_fk_chain
from .intersection import CROSS_90_INDEX, find_crossing_pairs
from .topology import MultiPathLayout, SwitchPair, TraversalPath


# =============================================================================
# Decoder Configuration
# =============================================================================

@dataclass
class DecoderConfig:
    """Configuration for the construction decoder."""

    # Closure tolerances
    position_tolerance: float = 8.0   # studs
    angle_tolerance: float = 15.0     # degrees

    # Angular budget
    target_angle: float = 360.0       # degrees for closed loop
    max_cumulative_angle: float = 1440.0  # Allow up to 4 full loops

    # Boundary for auto-centering
    boundary_min_x: float = -100.0
    boundary_max_x: float = 100.0
    boundary_min_y: float = -100.0
    boundary_max_y: float = 100.0


# =============================================================================
# Decoder State
# =============================================================================

@dataclass
class DecoderState:
    """Intermediate state during decoding."""

    # Placed piece data (main loop)
    piece_indices: List[int] = field(default_factory=list)
    states: List[Tuple[float, float, float]] = field(default_factory=list)

    # Inventory tracking
    inventory_used: Dict[int, int] = field(default_factory=dict)

    # Switch tracking
    switch_positions: List[int] = field(default_factory=list)

    # Crossing positions: (main_loop_position, piece_index)
    crossing_positions: List[Tuple[int, int]] = field(default_factory=list)

    # Port 2 connectivity tracking
    connected_port2_positions: Set[int] = field(default_factory=set)

    def use_piece(self, piece_idx: int) -> None:
        """Record piece usage in main loop."""
        self.inventory_used[piece_idx] = self.inventory_used.get(piece_idx, 0) + 1
        self.piece_indices.append(piece_idx)

    def get_usage(self, piece_idx: int) -> int:
        """Get current usage count for piece type."""
        return self.inventory_used.get(piece_idx, 0)


# =============================================================================
# Main Decoder Function
# =============================================================================

def decode_chromosome(
    chromosome: NDArray,
    catalog: TrackCatalog,
    inventory: Dict[str, int],
    config: Optional[DecoderConfig] = None,
    dims: Optional[ChromosomeDimensions] = None,
) -> MultiPathLayout:
    """Decode integer chromosome into MultiPathLayout.

    Architecture:
    1. Identify branch nodes (referenced by switch connection genes)
    2. Build main loop from sequential active nodes (excluding branch nodes)
    3. Extract switch pairs from connection genes
    4. Detect self-intersections, optionally replace with CROSS_90
    5. Enumerate all 2^N traversal paths
    6. Build secondary loops through crossings
    7. Auto-center within boundary

    Args:
        chromosome: Integer chromosome array (n_nodes * 3 genes).
        catalog: Track catalog with piece properties.
        inventory: Available inventory {piece_id: count}.
        config: Decoder configuration.
        dims: Chromosome dimensions (computed from inventory if None).

    Returns:
        MultiPathLayout with all traversal paths.
    """
    if config is None:
        config = DecoderConfig()

    if dims is None:
        total_inv = sum(inventory.values())
        dims = ChromosomeDimensions(n_nodes=total_inv, n_var=total_inv * GENES_PER_NODE)

    inventory_by_index = _convert_inventory_to_index(inventory, catalog)

    # Step 1: Identify which nodes are branch nodes (referenced by switch port2)
    branch_node_set = _identify_branch_nodes(chromosome, dims)

    # Step 2: Build main loop (sequential active nodes, excluding branch nodes)
    state = _decode_main_loop(chromosome, dims, catalog, inventory_by_index, config, branch_node_set)

    # Step 3: Extract switch pairs from connection genes
    switch_pairs = _extract_switch_pairs(
        chromosome, dims, state, catalog, inventory_by_index
    )

    # Step 4: Self-intersection detection and CROSS_90 repair
    _apply_crossing_repair(state, catalog, inventory_by_index)

    # Step 5: Build multi-path layout with 2^N path enumeration
    multi_path = _build_multi_path_layout(state, switch_pairs, catalog)

    # Step 6: Build secondary loops through crossings
    secondary_loops = _build_secondary_loops(
        chromosome, dims, state, catalog, inventory_by_index, branch_node_set
    )
    for sec_path in secondary_loops:
        multi_path.paths.append(sec_path)
    if secondary_loops:
        multi_path.secondary_closure_error = min(p.closure_error for p in secondary_loops)
    else:
        multi_path.secondary_closure_error = 0.0

    # Step 7: Auto-center layout within boundary
    _auto_center(multi_path, config)

    return multi_path


def decode_chromosome_legacy(
    chromosome: NDArray,
    catalog: TrackCatalog,
    inventory: Dict[str, int],
    config: Optional[DecoderConfig] = None,
    dims: Optional[ChromosomeDimensions] = None,
) -> Layout:
    """Legacy decoder returning simple Layout (no multi-path support)."""
    multi_path = decode_chromosome(chromosome, catalog, inventory, config, dims)

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
# Step 1: Identify Branch Nodes
# =============================================================================

def _identify_branch_nodes(
    chromosome: NDArray,
    dims: ChromosomeDimensions,
) -> Set[int]:
    """Identify nodes that are part of branches (referenced by switch port2_conn).

    Scans for IN switches whose port2_conn points to a valid node.
    The chain of nodes from port2_conn to the OUT switch's port2_conn
    constitutes the branch.

    Returns:
        Set of node indices that are branch nodes (not part of main loop).
    """
    branch_nodes: Set[int] = set()
    types = get_all_piece_types(chromosome, dims.n_nodes)
    p2conns = get_all_port2_conns(chromosome, dims.n_nodes)

    # Find all IN switches with valid port2 connections
    for i in range(dims.n_nodes):
        if types[i] in IN_SWITCH_INDICES and p2conns[i] != INACTIVE:
            branch_start = int(p2conns[i])
            if 0 <= branch_start < dims.n_nodes:
                # Find matching OUT switch that has port2 pointing into this range
                branch_end = _find_branch_end(types, p2conns, branch_start, dims.n_nodes)
                # Mark all nodes in [branch_start, branch_end] as branch nodes
                for j in range(branch_start, branch_end + 1):
                    if j < dims.n_nodes and types[j] != INACTIVE:
                        branch_nodes.add(j)

    # Also check crossings — their port2/port3 targets are secondary loop nodes
    p3conns = get_all_port3_conns(chromosome, dims.n_nodes)
    for i in range(dims.n_nodes):
        if types[i] in CROSSING_INDICES:
            for conn in (p2conns[i], p3conns[i]):
                if conn != INACTIVE and 0 <= conn < dims.n_nodes:
                    # Secondary loop nodes — mark them too
                    sec_end = _find_secondary_end(types, conn, dims.n_nodes)
                    for j in range(int(conn), sec_end + 1):
                        if j < dims.n_nodes and types[j] != INACTIVE:
                            branch_nodes.add(j)

    return branch_nodes


def _find_branch_end(
    types: NDArray,
    p2conns: NDArray,
    branch_start: int,
    n_nodes: int,
) -> int:
    """Find the last node of a branch chain.

    The branch end is determined by finding an OUT switch whose port2_conn
    points to a node in the branch range.
    """
    # Search for an OUT switch referencing a node >= branch_start
    for i in range(n_nodes):
        if types[i] in OUT_SWITCH_INDICES and p2conns[i] != INACTIVE:
            end = int(p2conns[i])
            if end >= branch_start and end < n_nodes:
                return end

    # Fallback: scan forward from branch_start until inactive node
    end = branch_start
    for j in range(branch_start, n_nodes):
        if types[j] == INACTIVE:
            break
        end = j
    return end


def _find_secondary_end(
    types: NDArray,
    start: int,
    n_nodes: int,
) -> int:
    """Find end of secondary loop node chain starting at 'start'."""
    end = start
    for j in range(start, n_nodes):
        if types[j] == INACTIVE:
            break
        end = j
    return end


# =============================================================================
# Step 2: Main Loop Construction
# =============================================================================

def _decode_main_loop(
    chromosome: NDArray,
    dims: ChromosomeDimensions,
    catalog: TrackCatalog,
    inventory_by_index: Dict[int, int],
    config: DecoderConfig,
    branch_nodes: Set[int],
) -> DecoderState:
    """Build main loop from sequential active nodes, skipping branch nodes.

    Reads piece_type genes directly (integer, no RK mapping).
    Applies turtle-graphics FK for each active main loop piece.
    Skips inactive nodes (-1) and nodes claimed by branches.
    """
    state = DecoderState()
    state.states.append((0.0, 0.0, 0.0))

    x, y, theta = 0.0, 0.0, 0.0
    cumulative_angle = 0.0

    for i in range(dims.n_nodes):
        if i in branch_nodes:
            continue

        piece_idx = get_piece_type(chromosome, i)
        if piece_idx == INACTIVE or piece_idx < 0:
            continue

        if piece_idx >= len(catalog._fk_table):
            continue

        # Check cumulative angle budget
        fk = catalog._fk_table[piece_idx]
        dtheta = fk[2]
        if cumulative_angle + abs(dtheta) > config.max_cumulative_angle:
            continue

        # Apply FK
        dx, dy = fk[0], fk[1]
        theta_rad = np.radians(theta)
        cos_t = np.cos(theta_rad)
        sin_t = np.sin(theta_rad)

        x = x + dx * cos_t - dy * sin_t
        y = y + dx * sin_t + dy * cos_t
        theta = theta + dtheta
        cumulative_angle += abs(dtheta)

        state.use_piece(piece_idx)
        state.states.append((x, y, theta))

        # Track switch positions in main loop
        if piece_idx in SWITCH_INDICES:
            position = len(state.piece_indices) - 1
            state.switch_positions.append(position)

        # Track crossing positions
        if piece_idx in CROSSING_INDICES:
            position = len(state.piece_indices) - 1
            state.crossing_positions.append((position, piece_idx))

    return state


# =============================================================================
# Step 3: Extract Switch Pairs From Connection Genes
# =============================================================================

def _extract_switch_pairs(
    chromosome: NDArray,
    dims: ChromosomeDimensions,
    state: DecoderState,
    catalog: TrackCatalog,
    inventory_by_index: Dict[int, int],
) -> List[SwitchPair]:
    """Extract switch pairs using port2_conn connection genes.

    For each IN switch in the main loop:
    1. Read its port2_conn to find the branch start node
    2. Find a matching OUT switch with port2_conn pointing to branch end
    3. Collect branch pieces from the chain between start and end
    4. Validate inventory and geometric compatibility
    """
    switch_pairs = []
    n_main = len(state.piece_indices)
    if n_main == 0:
        return switch_pairs

    # Build mapping: node_idx → main_loop_position for switches
    # We need to find which chromosome nodes correspond to main loop positions
    main_loop_node_map = _build_main_loop_node_map(chromosome, dims, state)

    pair_id = 0
    used_in_positions: Set[int] = set()
    used_out_positions: Set[int] = set()

    types = get_all_piece_types(chromosome, dims.n_nodes)
    p2conns = get_all_port2_conns(chromosome, dims.n_nodes)

    # Find IN switches with port2 connections
    for node_i in range(dims.n_nodes):
        if types[node_i] not in IN_SWITCH_INDICES:
            continue
        if p2conns[node_i] == INACTIVE:
            continue

        # Map this chromosome node to its main loop position
        main_pos_in = main_loop_node_map.get(node_i)
        if main_pos_in is None or main_pos_in in used_in_positions:
            continue

        branch_start = int(p2conns[node_i])
        if branch_start < 0 or branch_start >= dims.n_nodes:
            continue

        fork_idx = int(types[node_i])

        # Find matching OUT switch whose port2_conn references branch end
        found_out = None
        for node_j in range(dims.n_nodes):
            if types[node_j] not in OUT_SWITCH_INDICES:
                continue
            if p2conns[node_j] == INACTIVE:
                continue

            main_pos_out = main_loop_node_map.get(node_j)
            if main_pos_out is None or main_pos_out in used_out_positions:
                continue
            if main_pos_out <= main_pos_in:
                continue

            merge_idx = int(types[node_j])
            if not _are_compatible_switches(fork_idx, merge_idx):
                continue

            branch_end = int(p2conns[node_j])
            if branch_end < branch_start or branch_end >= dims.n_nodes:
                continue

            found_out = (node_j, main_pos_out, merge_idx, branch_end)
            break

        if found_out is None:
            continue

        node_j, main_pos_out, merge_idx, branch_end = found_out

        # Collect branch pieces from [branch_start, branch_end]
        branch_pieces = []
        for k in range(branch_start, branch_end + 1):
            if k < dims.n_nodes:
                pt = int(types[k])
                if pt != INACTIVE and pt >= 0:
                    branch_pieces.append(pt)

        if not branch_pieces:
            continue

        # Check branch piece inventory
        if not _check_branch_inventory(branch_pieces, inventory_by_index, state.inventory_used):
            continue

        # Create switch pair
        pair = SwitchPair(
            pair_id=pair_id,
            in_position=main_pos_in,
            out_position=main_pos_out,
            in_switch_idx=fork_idx,
            out_switch_idx=merge_idx,
            branch_pieces=branch_pieces,
        )
        switch_pairs.append(pair)
        pair_id += 1

        used_in_positions.add(main_pos_in)
        used_out_positions.add(main_pos_out)
        state.connected_port2_positions.add(main_pos_in)
        state.connected_port2_positions.add(main_pos_out)

        # Consume branch inventory
        for p in branch_pieces:
            state.inventory_used[p] = state.inventory_used.get(p, 0) + 1

    return switch_pairs


def _build_main_loop_node_map(
    chromosome: NDArray,
    dims: ChromosomeDimensions,
    state: DecoderState,
) -> Dict[int, int]:
    """Build mapping from chromosome node index to main loop position.

    Returns:
        Dict mapping node_idx → main_loop_position for all main loop pieces.
    """
    node_map: Dict[int, int] = {}
    main_pos = 0

    # Reconstruct which chromosome nodes ended up in the main loop
    # by matching piece types in order
    main_pieces = list(state.piece_indices)
    branch_nodes = _identify_branch_nodes(chromosome, dims)

    for i in range(dims.n_nodes):
        if i in branch_nodes:
            continue
        pt = get_piece_type(chromosome, i)
        if pt == INACTIVE or pt < 0:
            continue
        if main_pos < len(main_pieces):
            node_map[i] = main_pos
            main_pos += 1

    return node_map


def _are_compatible_switches(in_type: int, out_type: int) -> bool:
    """Check if IN and OUT switch types are compatible for a siding."""
    from .encoding import SWITCH_PAIRS
    for in_idx, out_idx in SWITCH_PAIRS:
        if in_type == in_idx and out_type == out_idx:
            return True
        if in_type == out_idx and out_type == in_idx:
            return True
    return False


def _check_branch_inventory(
    branch_pieces: List[int],
    inventory_by_index: Dict[int, int],
    used: Dict[int, int],
) -> bool:
    """Check if branch pieces are available in inventory."""
    temp_used = dict(used)
    for p in branch_pieces:
        temp_used[p] = temp_used.get(p, 0) + 1
        if temp_used[p] > inventory_by_index.get(p, 0):
            return False
    return True


# =============================================================================
# Step 4: Self-Intersection → CROSS_90 Repair
# =============================================================================

def _apply_crossing_repair(
    state: DecoderState,
    catalog: TrackCatalog,
    inventory_by_index: Dict[int, int],
    angle_tolerance: float = 15.0,
) -> None:
    """Replace self-intersections with CROSS_90 pieces.

    When the main loop crosses itself at ~90 degrees, replace the piece at one
    crossing position with CROSS_90 and remove the piece at the other.
    Recomputes FK chain. Mutates state in-place.
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

    states_array = np.array(state.states, dtype=np.float64)
    pairs = find_crossing_pairs(states_array, state.piece_indices)
    if not pairs:
        return

    claimed: Set[int] = set()
    replacements: List[int] = []
    removals: List[int] = []

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

    # Apply inventory changes
    for pos_i, pos_j in zip(replacements, removals):
        orig_i = state.piece_indices[pos_i]
        orig_j = state.piece_indices[pos_j]
        state.inventory_used[orig_i] = max(0, state.inventory_used.get(orig_i, 0) - 1)
        state.inventory_used[orig_j] = max(0, state.inventory_used.get(orig_j, 0) - 1)
        state.inventory_used[CROSS_90_INDEX] = state.inventory_used.get(CROSS_90_INDEX, 0) + 1
        state.piece_indices[pos_i] = CROSS_90_INDEX

    # Remove pieces at removal positions (reverse order)
    for pos_j in sorted(removals, reverse=True):
        state.piece_indices.pop(pos_j)
        if pos_j + 1 < len(state.states):
            state.states.pop(pos_j + 1)

    # Shift position tracking
    def _shift(positions, removed):
        removed_sorted = sorted(removed)
        return [
            p - sum(1 for r in removed_sorted if r < p)
            for p in positions if p not in removed
        ]

    state.switch_positions = _shift(state.switch_positions, removals)
    old_cross = state.crossing_positions
    shifted_pos = _shift([p for p, _ in old_cross], removals)
    state.crossing_positions = list(zip(shifted_pos, [idx for _, idx in old_cross]))

    # Add new crossing positions
    shifted_repl = _shift(replacements, removals)
    for p in shifted_repl:
        state.crossing_positions.append((p, CROSS_90_INDEX))

    # Recompute FK chain
    fk_deltas = np.array(
        [catalog._fk_table[idx] for idx in state.piece_indices],
        dtype=np.float64,
    )
    new_states = compute_fk_chain(fk_deltas)
    state.states = [
        (float(new_states[i, 0]), float(new_states[i, 1]), float(new_states[i, 2]))
        for i in range(len(new_states))
    ]


# =============================================================================
# Step 5: Multi-Path Layout Construction
# =============================================================================

def _build_multi_path_layout(
    state: DecoderState,
    switch_pairs: List[SwitchPair],
    catalog: TrackCatalog,
) -> MultiPathLayout:
    """Build MultiPathLayout with all 2^N traversal paths."""
    main_loop_pieces = list(state.piece_indices)
    n_switch_pairs = len(switch_pairs)

    # Loose ports: crossings contribute 2 each (perpendicular ports)
    crossing_loose = len(state.crossing_positions) * 2
    loose_port_count = crossing_loose

    if n_switch_pairs == 0:
        main_path = _compute_single_path(main_loop_pieces, [], tuple(), catalog)
        return MultiPathLayout(
            main_loop_pieces=main_loop_pieces,
            switch_pairs=[],
            paths=[main_path],
            loose_port_count=loose_port_count,
        )

    paths = []
    for path_id, choices in enumerate(product([0, 1], repeat=n_switch_pairs)):
        path = _compute_single_path(main_loop_pieces, switch_pairs, choices, catalog)
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
    """Compute FK for a single traversal path."""
    if not main_loop_pieces:
        return TraversalPath(
            path_id=0,
            route_choices=route_choices,
            piece_sequence=[],
            states=np.zeros((1, 3), dtype=np.float64),
            closure_error=0.0,
            angle_error=360.0,
        )

    sorted_pairs = sorted(switch_pairs, key=lambda p: p.in_position)
    absorbed_set: Set[int] = set()
    for pair in sorted_pairs:
        absorbed_set.update(pair.absorbed_positions)

    piece_sequence: List[int] = []
    route_indices: List[int] = []
    current_pos = 0

    for i, pair in enumerate(sorted_pairs):
        choice = route_choices[i] if i < len(route_choices) else 0

        # Main loop pieces up to IN switch
        for pos in range(current_pos, pair.in_position):
            if pos not in absorbed_set:
                piece_sequence.append(main_loop_pieces[pos])
                route_indices.append(0)

        if choice == 0:
            # Straight-through
            piece_sequence.append(pair.in_switch_idx)
            route_indices.append(0)
            for pos in range(pair.in_position + 1, pair.out_position):
                if pos not in absorbed_set:
                    piece_sequence.append(main_loop_pieces[pos])
                    route_indices.append(0)
            piece_sequence.append(pair.out_switch_idx)
            route_indices.append(0)
        else:
            # Branch path
            piece_sequence.append(pair.in_switch_idx)
            route_indices.append(1)
            piece_sequence.extend(pair.branch_pieces)
            route_indices.extend([0] * len(pair.branch_pieces))
            piece_sequence.append(pair.out_switch_idx)
            route_indices.append(1)

        current_pos = pair.out_position + 1

    # Remaining main loop pieces
    for pos in range(current_pos, len(main_loop_pieces)):
        if pos not in absorbed_set:
            piece_sequence.append(main_loop_pieces[pos])
            route_indices.append(0)

    states = _compute_path_fk(piece_sequence, route_indices, catalog)
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
    """Compute FK states for a path using route-aware catalog lookup."""
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
    """Compute closure error and angle error from FK states."""
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


# =============================================================================
# Step 6: Secondary Loops Through Crossings
# =============================================================================

def _build_secondary_loops(
    chromosome: NDArray,
    dims: ChromosomeDimensions,
    state: DecoderState,
    catalog: TrackCatalog,
    inventory_by_index: Dict[int, int],
    branch_nodes: Set[int],
) -> List[TraversalPath]:
    """Build secondary loops through crossing pieces.

    For each CROSS_90 in the main loop, follows port2/port3 connections
    to build an independent closed loop using the perpendicular route.
    """
    secondary_paths: List[TraversalPath] = []

    if not state.crossing_positions:
        return secondary_paths

    types = get_all_piece_types(chromosome, dims.n_nodes)

    for cross_pos, cross_idx in state.crossing_positions:
        if cross_pos >= len(state.states):
            continue

        # Find the chromosome node for this crossing
        cross_node = _find_crossing_node(chromosome, dims, cross_pos, state, branch_nodes)
        if cross_node is None:
            continue

        p2_target = get_port2_conn(chromosome, cross_node)
        if p2_target == INACTIVE or p2_target < 0 or p2_target >= dims.n_nodes:
            continue

        # Collect secondary loop pieces from connection chain
        sec_pieces = []
        for j in range(int(p2_target), dims.n_nodes):
            pt = int(types[j])
            if pt == INACTIVE:
                break
            sec_pieces.append(pt)

        if not sec_pieces:
            continue

        # Check inventory
        temp_used = dict(state.inventory_used)
        valid = True
        for p in sec_pieces:
            temp_used[p] = temp_used.get(p, 0) + 1
            if temp_used[p] > inventory_by_index.get(p, 0):
                valid = False
                break
        if not valid:
            continue

        # Get crossing position and heading for secondary loop entry
        piece_x, piece_y, piece_theta = state.states[cross_pos]

        # Port 2 world position: local (8, -8) rotated by piece heading
        theta_rad = np.radians(piece_theta)
        cos_t, sin_t = np.cos(theta_rad), np.sin(theta_rad)
        port2_x = piece_x + 8.0 * cos_t + 8.0 * sin_t
        port2_y = piece_y + 8.0 * sin_t - 8.0 * cos_t
        entry_heading = piece_theta + 90.0

        # Build FK chain: first piece = crossing route 1, rest = route 0
        piece_indices = [cross_idx] + sec_pieces
        route_indices = [1] + [0] * len(sec_pieces)

        fk_deltas = catalog.get_fk_with_routes(
            np.array(piece_indices, dtype=np.int32),
            np.array(route_indices, dtype=np.int32),
        )
        sec_states = compute_fk_chain(fk_deltas)

        # Transform to world coordinates
        sec_states[:, 0] += port2_x
        sec_states[:, 1] += port2_y
        if abs(entry_heading) > 0.01:
            rot_rad = np.radians(entry_heading)
            cos_r, sin_r = np.cos(rot_rad), np.sin(rot_rad)
            dx = sec_states[:, 0] - port2_x
            dy = sec_states[:, 1] - port2_y
            sec_states[:, 0] = port2_x + dx * cos_r - dy * sin_r
            sec_states[:, 1] = port2_y + dx * sin_r + dy * cos_r
            sec_states[:, 2] += entry_heading

        closure_error, angle_error = _compute_closure_metrics(sec_states)

        path = TraversalPath(
            path_id=len(secondary_paths) + 100,
            route_choices=(),
            piece_sequence=piece_indices,
            states=sec_states,
            closure_error=closure_error,
            angle_error=angle_error,
        )
        secondary_paths.append(path)

        state.connected_port2_positions.add(cross_pos)
        for p in sec_pieces:
            state.inventory_used[p] = state.inventory_used.get(p, 0) + 1

    return secondary_paths


def _find_crossing_node(
    chromosome: NDArray,
    dims: ChromosomeDimensions,
    cross_pos: int,
    state: DecoderState,
    branch_nodes: Set[int],
) -> Optional[int]:
    """Find the chromosome node index for a crossing at main loop position cross_pos."""
    main_pos = 0
    for i in range(dims.n_nodes):
        if i in branch_nodes:
            continue
        pt = get_piece_type(chromosome, i)
        if pt == INACTIVE or pt < 0:
            continue
        if main_pos == cross_pos:
            return i
        main_pos += 1
    return None


# =============================================================================
# Step 7: Auto-Center
# =============================================================================

def _auto_center(multi_path: MultiPathLayout, config: DecoderConfig) -> None:
    """Auto-center layout within boundary."""
    all_x_parts = [p.states[:, 0] for p in multi_path.paths if len(p.states) > 0]
    all_y_parts = [p.states[:, 1] for p in multi_path.paths if len(p.states) > 0]

    if not all_x_parts:
        return

    all_x = np.concatenate(all_x_parts)
    all_y = np.concatenate(all_y_parts)

    if len(all_x) == 0:
        return

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


# =============================================================================
# Helper Functions
# =============================================================================

def _convert_inventory_to_index(
    inventory: Dict[str, int],
    catalog: TrackCatalog,
) -> Dict[int, int]:
    """Convert inventory from piece_id to piece_index."""
    result = {}
    for piece_id, count in inventory.items():
        idx = catalog._id_to_index.get(piece_id)
        if idx is not None:
            result[idx] = count
    return result


# Legacy constants for backward compatibility with intersection.py
CROSS_90_INDEX = 4
DOUBLE_CROSSOVER_INDEX = 9
