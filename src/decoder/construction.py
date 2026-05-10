"""Construction-based decoder: chromosome → MultiPathLayout.

Algorithm:
1. Read main loop piece types, filter INACTIVE → active_pieces
2. Read active junctions, sort by position, compute branch pieces from templates
3. Inject switches into main loop copy (replace, not insert)
4. Self-intersection repair (inject CROSS_90 at ~90° crossings)
5. Compute FK for augmented main loop
6. Enumerate 2^J traversal paths
7. Auto-center within boundary, return MultiPathLayout
"""

from itertools import product
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from numpy.typing import NDArray

from src.catalog import TrackCatalog
from src.decoder.types import DecoderConfig, InventoryTracker, ValidatedJunction
from src.encoding import (
    INACTIVE,
    SWITCH_INDICES,
    PartitionedDimensions,
    get_active_junctions,
    get_main_loop_types,
    get_start_position,
)
from src.geometry import compute_closure_metrics, compute_fk_chain
from src.intersection import CROSS_90_INDEX, find_crossing_pairs
from src.templates import (
    TEMPLATES,
    compute_branch_pieces,
    compute_required_main_distance,
    get_siding_inventory_requirements,
    is_valid_siding,
    switch_indices_for,
)
from src.types import MultiPathLayout, SwitchPair, TraversalPath


# =============================================================================
# Main Decoder Function
# =============================================================================

def decode_chromosome(
    x: NDArray,
    catalog: TrackCatalog,
    inventory: Dict[str, int],
    dims: PartitionedDimensions,
    config: Optional[DecoderConfig] = None,
) -> MultiPathLayout:
    """Decode a partitioned chromosome into a MultiPathLayout.

    Args:
        x: Chromosome array of length dims.n_var.
        catalog: Track catalog with FK tables and piece properties.
        inventory: Available pieces {piece_id: count}.
        dims: Partitioned chromosome dimensions.
        config: Decoder configuration (defaults created if None).

    Returns:
        MultiPathLayout with all traversal paths computed.
    """
    if config is None:
        config = DecoderConfig()

    tracker = InventoryTracker(inventory, catalog)

    # Step 1: Read main loop active pieces
    main_pieces = _read_main_loop(x, dims, tracker)

    if not main_pieces:
        return _empty_layout()

    # Step 2: Read and validate junctions
    junctions = _read_junctions(x, dims, main_pieces, tracker, catalog, config)

    # Step 3: Inject switches into main loop, build switch pairs
    augmented_pieces, switch_pairs = _inject_switches(
        main_pieces, junctions, tracker, catalog, config,
    )

    # Step 4: Self-intersection repair (CROSS_90 injection)
    augmented_pieces = _apply_crossing_repair(augmented_pieces, tracker, catalog, config)

    # Step 5 + 6: Build multi-path layout with FK and 2^J paths
    multi_path = _build_multi_path_layout(augmented_pieces, switch_pairs, catalog)

    # Step 7: Auto-center within boundary
    start_x, start_y = get_start_position(x, dims)
    _auto_center(multi_path, config, start_x, start_y)

    return multi_path


# =============================================================================
# Step 1: Read Main Loop
# =============================================================================

def _read_main_loop(
    x: NDArray,
    dims: PartitionedDimensions,
    tracker: InventoryTracker,
) -> List[int]:
    """Read active main loop pieces, consuming from inventory.

    Filters INACTIVE genes and pieces that exceed inventory.

    Returns:
        List of valid piece indices for the main loop.
    """
    raw_types = get_main_loop_types(x, dims)
    pieces: List[int] = []

    for val in raw_types:
        idx = int(val)
        if idx == INACTIVE:
            continue
        if not tracker.can_use(idx):
            continue
        tracker.use(idx)
        pieces.append(idx)

    return pieces


# =============================================================================
# Step 2: Read and Validate Junctions
# =============================================================================

def _read_junctions(
    x: NDArray,
    dims: PartitionedDimensions,
    main_pieces: List[int],
    tracker: InventoryTracker,
    catalog: TrackCatalog,
    config: DecoderConfig,
) -> List[ValidatedJunction]:
    """Read active junctions, validate inventory, deactivate if insufficient.

    Junctions are processed in position order. Each junction greedily
    claims inventory for its switches, curves, and straights.

    Returns:
        List of validated junctions sorted by position.
    """
    n_main = len(main_pieces)
    if n_main < 4:
        return []

    raw_junctions = get_active_junctions(x, dims)
    validated: List[ValidatedJunction] = []

    for slot, _active, position, handedness, n_straights in raw_junctions:
        # Clamp position to valid main loop range
        position = max(0, min(position, n_main - 1))

        # Clamp handedness to valid template range (0=LEFT, 1=RIGHT)
        handedness = handedness % len(TEMPLATES)

        template = TEMPLATES[handedness]

        # Clamp n_straights to reasonable range
        n_straights = max(0, min(n_straights, dims.total_straights))

        branch_pieces = compute_branch_pieces(template, n_straights)
        requirements = get_siding_inventory_requirements(template, n_straights)

        # Greedy inventory check
        if not tracker.can_use_batch(requirements):
            continue

        # Reserve the inventory
        tracker.use_batch(requirements)

        validated.append(ValidatedJunction(
            slot=slot,
            position=position,
            handedness=handedness,
            n_straights=n_straights,
            template=template,
            branch_pieces=branch_pieces,
            siding_requirements=requirements,
        ))

    return validated


# =============================================================================
# Step 3: Inject Switches Into Main Loop
# =============================================================================

def _inject_switches(
    main_pieces: List[int],
    junctions: List[ValidatedJunction],
    tracker: InventoryTracker,
    catalog: TrackCatalog,
    config: DecoderConfig,
) -> Tuple[List[int], List[SwitchPair]]:
    """Replace main loop pieces at junction positions with switches.

    For each validated junction:
    1. Replace piece at IN position with IN switch
    2. Walk FK from IN to find OUT position matching required distance
    3. Replace piece at OUT position with OUT switch
    4. Validate geometric alignment via is_valid_siding()
    5. Revert and deactivate if invalid

    Returns:
        (augmented_pieces, switch_pairs) where augmented_pieces has
        switches injected in place.
    """
    if not junctions:
        return list(main_pieces), []

    augmented = list(main_pieces)
    switch_pairs: List[SwitchPair] = []
    used_positions: Set[int] = set()
    n_main = len(augmented)

    # Track cumulative offset from prior injections (none — we replace, not insert)
    pair_id = 0

    for junc in junctions:
        in_pos = junc.position

        # Skip if position already claimed or out of range
        if in_pos in used_positions or in_pos >= n_main:
            _release_junction_inventory(junc, tracker)
            continue

        template = junc.template
        required_dist = compute_required_main_distance(template, junc.n_straights)

        # Find OUT position by walking FK from IN position
        out_pos = _find_out_position(augmented, in_pos, required_dist, catalog)

        if out_pos is None or out_pos >= n_main or out_pos <= in_pos:
            _release_junction_inventory(junc, tracker)
            continue

        if out_pos in used_positions:
            _release_junction_inventory(junc, tracker)
            continue

        # IN-state is the train state when entering the IN switch — same
        # before/after injection (computed from positions 0..in_pos-1).
        in_state = _compute_state_at_position(augmented, in_pos, catalog)

        # OUT-state must reflect post-injection geometry: the IN switch's
        # body length differs from whatever piece it replaces, shifting
        # everything downstream. Tentatively inject IN, then compute, then
        # revert if the alignment check fails.
        entry_switch_idx, exit_switch_idx = switch_indices_for(template)
        orig_in = augmented[in_pos]
        augmented[in_pos] = entry_switch_idx
        out_state = _compute_state_at_position(augmented, out_pos, catalog)

        if not is_valid_siding(
            in_state, out_state, template, junc.n_straights,
            position_tolerance=config.siding_position_tolerance,
            angle_tolerance=config.siding_angle_tolerance,
        ):
            augmented[in_pos] = orig_in  # revert tentative IN injection
            _release_junction_inventory(junc, tracker)
            continue

        # Release the replaced pieces and complete the OUT injection.
        orig_out = augmented[out_pos]
        tracker.release(orig_in)
        tracker.release(orig_out)
        augmented[out_pos] = exit_switch_idx

        used_positions.add(in_pos)
        used_positions.add(out_pos)

        # Mark positions between IN and OUT as occupied (no overlapping junctions)
        for p in range(in_pos, out_pos + 1):
            used_positions.add(p)

        switch_pairs.append(SwitchPair(
            pair_id=pair_id,
            in_position=in_pos,
            out_position=out_pos,
            handedness=template.handedness,
            branch_pieces=junc.branch_pieces,
            merge_fk=template.merge_fk,
        ))
        pair_id += 1

    return augmented, switch_pairs


def _find_out_position(
    pieces: List[int],
    in_pos: int,
    required_dist: float,
    catalog: TrackCatalog,
) -> Optional[int]:
    """Find the main-loop position to place the OUT switch.

    `required_dist` is the X contribution of main pieces strictly between IN
    and OUT (not counting the switch bodies). We walk forward from in_pos+1
    and find the position pos where cumulative X distance from IN body END
    to pos's BODY START best matches `required_dist`. The OUT switch then
    replaces whatever piece was at pos.

    Args:
        pieces: Current main loop piece list.
        in_pos: IN switch position.
        required_dist: Target X distance of main pieces between switches.
        catalog: Track catalog for FK lookup.

    Returns:
        Best matching OUT position, or None if no valid position found.
    """
    n = len(pieces)
    if in_pos + 2 >= n:
        return None

    in_state = _compute_state_at_position(pieces, in_pos, catalog)
    base_theta = in_state[2]

    cumulative_x = 0.0  # distance from IN body end to start of current pos
    best_pos = None
    best_error = float('inf')

    for pos in range(in_pos + 1, n):
        # Check error at pos's START (before adding pos's own dx contribution)
        error = abs(cumulative_x - required_dist)
        if error < best_error:
            best_error = error
            best_pos = pos

        idx = pieces[pos]
        if idx < 0 or idx >= len(catalog._fk_table):
            continue

        fk = catalog._fk_table[idx]
        dx, dy, dtheta = fk[0], fk[1], fk[2]
        theta_rad = np.radians(base_theta)
        cos_t = np.cos(theta_rad)
        sin_t = np.sin(theta_rad)
        local_x = dx * cos_t - dy * sin_t
        cumulative_x += abs(local_x)
        base_theta += dtheta

        # Stop searching once we've well past the target
        if cumulative_x > required_dist + 32.0:
            break

    # Accept if within 20% + 2 stud tolerance
    if best_pos is not None and best_error <= max(required_dist * 0.2, 0.0) + 2.0:
        return best_pos

    return None


def _compute_state_at_position(
    pieces: List[int],
    position: int,
    catalog: TrackCatalog,
) -> Tuple[float, float, float]:
    """Compute FK state (x, y, theta) at a given main loop position.

    Runs FK chain from origin through pieces[0:position].

    Returns:
        (x, y, theta) tuple at the entry of the piece at `position`.
    """
    if position == 0:
        return (0.0, 0.0, 0.0)

    indices = np.array(pieces[:position], dtype=np.int32)
    fk_deltas = catalog.get_fk(indices)
    states = compute_fk_chain(fk_deltas)

    final = states[-1]
    return (float(final[0]), float(final[1]), float(final[2]))


def _release_junction_inventory(
    junc: ValidatedJunction,
    tracker: InventoryTracker,
) -> None:
    """Return a junction's reserved inventory when it fails validation."""
    for idx, count in junc.siding_requirements.items():
        tracker.release(idx, count)


# =============================================================================
# Step 4: Self-Intersection Repair
# =============================================================================

def _apply_crossing_repair(
    pieces: List[int],
    tracker: InventoryTracker,
    catalog: TrackCatalog,
    config: DecoderConfig,
) -> List[int]:
    """Replace ~90° self-intersections with CROSS_90 pieces.

    Detects segment crossings near 90 degrees and replaces the piece
    at one crossing position with CROSS_90 if inventory allows.
    Does not remove pieces (maintains stable indices for switch pairs).

    Returns:
        Modified piece list with CROSS_90 replacements.
    """
    n = len(pieces)
    if n < 4:
        return pieces

    cross_remaining = tracker.remaining(CROSS_90_INDEX)
    if cross_remaining <= 0:
        return pieces

    # Compute FK states for crossing detection
    indices = np.array(pieces, dtype=np.int32)
    fk_deltas = catalog.get_fk(indices)
    states = compute_fk_chain(fk_deltas)

    pairs = find_crossing_pairs(states, pieces)
    if not pairs:
        return pieces

    result = list(pieces)
    claimed: Set[int] = set()
    replacements = 0

    for pos_i, pos_j, angle_diff in pairs:
        if abs(angle_diff - 90.0) > config.crossing_angle_tolerance:
            continue
        if cross_remaining <= 0:
            break
        if pos_i in claimed or pos_j in claimed:
            continue
        # Skip switch positions
        if pieces[pos_i] in SWITCH_INDICES or pieces[pos_j] in SWITCH_INDICES:
            continue

        # Replace pos_i with CROSS_90
        orig = result[pos_i]
        tracker.release(orig)
        tracker.use(CROSS_90_INDEX)
        result[pos_i] = CROSS_90_INDEX

        claimed.add(pos_i)
        claimed.add(pos_j)
        cross_remaining -= 1
        replacements += 1

    return result


# =============================================================================
# Step 5 + 6: Multi-Path Layout Construction
# =============================================================================

def _build_multi_path_layout(
    main_pieces: List[int],
    switch_pairs: List[SwitchPair],
    catalog: TrackCatalog,
) -> MultiPathLayout:
    """Build MultiPathLayout with all 2^J traversal paths.

    Args:
        main_pieces: Augmented main loop pieces (with switches injected).
        switch_pairs: Validated switch pairs with branch pieces.
        catalog: Track catalog for FK computation.

    Returns:
        MultiPathLayout with all paths computed.
    """
    n_pairs = len(switch_pairs)

    if n_pairs == 0:
        main_path = _compute_single_path(main_pieces, [], tuple(), catalog)
        return MultiPathLayout(
            main_loop_pieces=main_pieces,
            switch_pairs=[],
            paths=[main_path],
            loose_port_count=0,
        )

    # Sort pairs by position for deterministic path construction
    sorted_pairs = sorted(switch_pairs, key=lambda p: p.in_position)

    paths: List[TraversalPath] = []
    for path_id, choices in enumerate(product([0, 1], repeat=n_pairs)):
        path = _compute_single_path(main_pieces, sorted_pairs, choices, catalog)
        path.path_id = path_id
        paths.append(path)

    return MultiPathLayout(
        main_loop_pieces=main_pieces,
        switch_pairs=sorted_pairs,
        paths=paths,
        loose_port_count=0,
    )


def _compute_single_path(
    main_pieces: List[int],
    switch_pairs: List[SwitchPair],
    route_choices: Tuple[int, ...],
    catalog: TrackCatalog,
) -> TraversalPath:
    """Compute FK chain for a single traversal path.

    For each switch pair, either traverse straight-through (choice=0)
    or take the branch (choice=1). The branch path uses the diverge
    route for IN switch, branch pieces, then merge route for OUT switch.

    Args:
        main_pieces: Augmented main loop with switches.
        switch_pairs: Sorted switch pairs.
        route_choices: Binary tuple, one per switch pair.
        catalog: Track catalog for FK lookup.

    Returns:
        TraversalPath with piece sequence, states, and closure metrics.
    """
    if not main_pieces:
        return TraversalPath(
            path_id=0,
            route_choices=route_choices,
            piece_sequence=[],
            states=np.zeros((1, 3), dtype=np.float64),
            closure_error=0.0,
            angle_error=360.0,
        )

    # Catalog route indices for switch pieces (per YAML route order):
    #   0 = through    (A->B, straight main traversal)
    #   1 = diverging  (A->C, used at the entry switch on a branch path)
    # The exit switch on a branch path needs C->A in the train's entry-local
    # frame for a REVERSED installation — not expressible by catalog routes.
    # We register pair.merge_fk in `fk_overrides` for that one piece index.
    THROUGH, DIVERGING = 0, 1

    piece_sequence: List[int] = []
    route_indices: List[int] = []
    divergent_ranges: Dict[int, Tuple[int, int]] = {}
    fk_overrides: Dict[int, Tuple[float, float, float]] = {}
    current_pos = 0

    for i, pair in enumerate(switch_pairs):
        choice = route_choices[i] if i < len(route_choices) else 0

        # Add main loop pieces before this switch pair
        for pos in range(current_pos, pair.in_position):
            piece_sequence.append(main_pieces[pos])
            route_indices.append(THROUGH)

        in_seq_idx = len(piece_sequence)
        # Switch piece indices were placed in main_pieces by _inject_switches.
        entry_switch = main_pieces[pair.in_position]
        exit_switch = main_pieces[pair.out_position]

        if choice == 0:
            # Straight-through both switches; main loop pieces between them.
            piece_sequence.append(entry_switch)
            route_indices.append(THROUGH)
            for pos in range(pair.in_position + 1, pair.out_position):
                piece_sequence.append(main_pieces[pos])
                route_indices.append(THROUGH)
            piece_sequence.append(exit_switch)
            route_indices.append(THROUGH)
        else:
            # Branch: entry diverges (A->C), branch template pieces, then
            # exit merges via reversed-install C->A (pair.merge_fk override).
            piece_sequence.append(entry_switch)
            route_indices.append(DIVERGING)
            piece_sequence.extend(pair.branch_pieces)
            route_indices.extend([THROUGH] * len(pair.branch_pieces))
            piece_sequence.append(exit_switch)
            exit_seq_idx = len(piece_sequence) - 1
            route_indices.append(THROUGH)  # placeholder; override takes precedence
            fk_overrides[exit_seq_idx] = pair.merge_fk
            divergent_ranges[i] = (in_seq_idx, exit_seq_idx)

        current_pos = pair.out_position + 1

    # Remaining main loop pieces after last switch pair
    for pos in range(current_pos, len(main_pieces)):
        piece_sequence.append(main_pieces[pos])
        route_indices.append(THROUGH)

    # Compute FK
    states = _compute_path_fk(piece_sequence, route_indices, catalog, fk_overrides)
    closure_error, angle_error = compute_closure_metrics(states)

    return TraversalPath(
        path_id=0,
        route_choices=route_choices,
        piece_sequence=piece_sequence,
        states=states,
        closure_error=closure_error,
        angle_error=angle_error,
        divergent_ranges=divergent_ranges,
    )


def _compute_path_fk(
    piece_sequence: List[int],
    route_indices: List[int],
    catalog: TrackCatalog,
    fk_overrides: Optional[Dict[int, Tuple[float, float, float]]] = None,
) -> NDArray[np.float64]:
    """Compute FK states for a piece sequence with route selection.

    Args:
        piece_sequence: Ordered piece indices.
        route_indices: Route index per piece (used unless overridden).
        catalog: Track catalog.
        fk_overrides: Map of sequence index -> (dx, dy, dtheta). When present
                      for an index, takes precedence over the catalog route.
                      Used for OUT-position branch traversal where the catalog
                      can't express reversed-installation FK (see SwitchPair).

    Returns:
        (n+1, 3) state array [x, y, theta].
    """
    if not piece_sequence:
        return np.zeros((1, 3), dtype=np.float64)

    n = len(piece_sequence)
    states = np.zeros((n + 1, 3), dtype=np.float64)

    for i in range(n):
        if fk_overrides is not None and i in fk_overrides:
            dx, dy, dtheta = fk_overrides[i]
        else:
            fk = catalog.get_fk_route(piece_sequence[i], route_indices[i])
            dx, dy, dtheta = fk[0], fk[1], fk[2]

        theta_rad = np.radians(states[i, 2])
        cos_t = np.cos(theta_rad)
        sin_t = np.sin(theta_rad)

        states[i + 1, 0] = states[i, 0] + dx * cos_t - dy * sin_t
        states[i + 1, 1] = states[i, 1] + dx * sin_t + dy * cos_t
        states[i + 1, 2] = states[i, 2] + dtheta

    return states


# =============================================================================
# Step 7: Auto-Center
# =============================================================================

def _auto_center(
    multi_path: MultiPathLayout,
    config: DecoderConfig,
    start_x: float,
    start_y: float,
) -> None:
    """Shift layout so its center aligns with boundary center plus start offset.

    The start_x/start_y genes from the chromosome act as a fine-tuning
    offset on top of auto-centering.
    """
    all_x_parts = [p.states[:, 0] for p in multi_path.paths if len(p.states) > 1]
    all_y_parts = [p.states[:, 1] for p in multi_path.paths if len(p.states) > 1]

    if not all_x_parts:
        multi_path.start_position = (start_x, start_y)
        return

    all_x = np.concatenate(all_x_parts)
    all_y = np.concatenate(all_y_parts)

    layout_cx = (all_x.min() + all_x.max()) / 2
    layout_cy = (all_y.min() + all_y.max()) / 2
    boundary_cx = (config.boundary_min_x + config.boundary_max_x) / 2
    boundary_cy = (config.boundary_min_y + config.boundary_max_y) / 2

    shift_x = boundary_cx - layout_cx + start_x
    shift_y = boundary_cy - layout_cy + start_y

    multi_path.start_position = (shift_x, shift_y)
    for path in multi_path.paths:
        if len(path.states) > 1:
            path.states[:, 0] += shift_x
            path.states[:, 1] += shift_y


# =============================================================================
# Helpers
# =============================================================================

def _empty_layout() -> MultiPathLayout:
    """Return an empty MultiPathLayout for degenerate chromosomes."""
    empty_path = TraversalPath(
        path_id=0,
        route_choices=tuple(),
        piece_sequence=[],
        states=np.zeros((1, 3), dtype=np.float64),
        closure_error=0.0,
        angle_error=360.0,
    )
    return MultiPathLayout(
        main_loop_pieces=[],
        switch_pairs=[],
        paths=[empty_path],
        loose_port_count=0,
    )
