"""Construction-based decoder: chromosome → MultiPathLayout.

Pipeline:
- Read main loop piece types, filter INACTIVE → active_pieces
- Read active junctions, sort by position, compute branch pieces from templates
- Inject switches into main loop copy (replace, not insert)
- Inject cross-junctions (CROSS_90 at descriptor-named perpendicular slots)
- Inject double-crossovers (DOUBLE_CROSSOVER replacing two straight slots)
- Self-intersection repair (inject CROSS_90 at emergent ~90° crossings)
- Compute FK for the augmented main loop and enumerate 2^J traversal paths
- Auto-center within boundary, return MultiPathLayout
"""

from itertools import product
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from numpy.typing import NDArray

from src.catalog import TrackCatalog
from src.decoder.types import DecoderConfig, InventoryTracker, ValidatedJunction
from src.encoding import (
    INACTIVE,
    R40_CURVE,
    STRAIGHT_16,
    SWITCH_INDICES,
    PartitionedDimensions,
    fk_rows_with_flips,
    get_active_cross_junctions,
    get_active_double_crossovers,
    get_active_junctions,
    get_main_loop_flips,
    get_main_loop_types,
    get_start_position,
)
from src.geometry import compute_closure_metrics, compute_fk_chain
from src.intersection import (
    CROSS_90_INDEX,
    cross_pair_perpendicular,
    find_crossing_pairs,
)
from src.templates import (
    DC_PIECE_IDX,
    TEMPLATES,
    compute_branch_pieces,
    compute_required_main_distance,
    dbl_crossover_piece_origin,
    dbl_crossover_routes_cover_all_ports,
    get_dbl_crossover_inventory_requirements,
    get_siding_inventory_requirements,
    is_valid_siding,
    switch_indices_for,
)
from src.types import CrossJunction, DblCrossover, MultiPathLayout, SwitchPair, TraversalPath


# =============================================================================
# FK with per-slot flip (R40_CURVE handedness chosen at placement time)
# =============================================================================

def get_fk_with_flip(
    catalog: TrackCatalog, piece_idx: int, flip: int = 0,
) -> NDArray[np.float64]:
    """FK row for ``piece_idx``, negated when ``flip`` is set on an R40_CURVE.

    Other piece types ignore the flip bit (catalog row is symmetric or has
    its own per-route handling). Returned array shape is (3,): [dx, dy, dtheta_deg].
    """
    fk = catalog._fk_table[piece_idx].copy()
    if piece_idx == int(R40_CURVE) and flip:
        fk[1] *= -1.0
        fk[2] *= -1.0
    return fk


def fk_array_with_flips(
    catalog: TrackCatalog,
    pieces: List[int],
    flips: Optional[List[int]] = None,
) -> NDArray[np.float64]:
    """Vectorized FK lookup for a main-loop slice, applying per-slot flips."""
    return fk_rows_with_flips(catalog._fk_table, pieces, flips)


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

    # Trace of every descriptor the decode skips, attached to the layout for
    # the per-category run report.
    drop_log: List[str] = []

    main_pieces, main_flips = _read_main_loop(x, dims, tracker)

    if not main_pieces:
        return _empty_layout()

    junctions = _read_junctions(
        x, dims, main_pieces, tracker, drop_log=drop_log,
    )

    augmented_pieces, augmented_flips, switch_pairs = _inject_switches(
        main_pieces, junctions, tracker, catalog, config,
        main_flips=main_flips, drop_log=drop_log,
    )

    # Deliberate cross-junctions (CROSS_90 self-crossings)
    cross_junctions = _inject_cross_junctions(
        augmented_pieces, x, dims, tracker, catalog,
        main_flips=augmented_flips, drop_log=drop_log,
    )

    dbl_crossovers, dbl_route_map = _inject_double_crossovers(
        augmented_pieces, x, dims, tracker, catalog, config,
        main_flips=augmented_flips, drop_log=drop_log,
    )

    augmented_pieces, augmented_flips = _apply_crossing_repair(
        augmented_pieces, tracker, catalog, flips=augmented_flips,
    )

    # Build multi-path layout with FK and 2^J paths
    multi_path = _build_multi_path_layout(
        augmented_pieces, augmented_flips, switch_pairs, catalog, dbl_route_map,
    )
    multi_path.cross_junctions = cross_junctions
    multi_path.dbl_crossovers = dbl_crossovers
    multi_path.main_loop_routes = dbl_route_map
    multi_path.drop_log = drop_log

    start_x, start_y = get_start_position(x, dims)
    _auto_center(multi_path, config, start_x, start_y)

    return multi_path


# =============================================================================
# Read Main Loop
# =============================================================================

def _read_main_loop(
    x: NDArray,
    dims: PartitionedDimensions,
    tracker: InventoryTracker,
) -> Tuple[List[int], List[int]]:
    """Read active main loop pieces and their per-slot flips.

    Returns:
        (pieces, flips): parallel lists. Flips are kept only for slots that
        survive the inventory check, so indexing aligns with ``pieces``.
    """
    raw_types = get_main_loop_types(x, dims)
    raw_flips = get_main_loop_flips(x, dims)
    pieces: List[int] = []
    flips: List[int] = []

    for slot, val in enumerate(raw_types):
        idx = int(val)
        if idx == INACTIVE:
            continue
        if not tracker.can_use(idx):
            continue
        tracker.use(idx)
        pieces.append(idx)
        flips.append(int(raw_flips[slot]))

    return pieces, flips


# =============================================================================
# Read and Validate Junctions
# =============================================================================

def _read_junctions(
    x: NDArray,
    dims: PartitionedDimensions,
    main_pieces: List[int],
    tracker: InventoryTracker,
    drop_log: Optional[List[str]] = None,
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

        branch_pieces, branch_flips = compute_branch_pieces(template, n_straights)
        requirements = get_siding_inventory_requirements(template, n_straights)

        # Greedy inventory check
        if not tracker.can_use_batch(requirements):
            if drop_log is not None:
                drop_log.append(
                    f"junction[{slot}] (pos {position}): insufficient inventory for siding"
                )
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
            branch_flips=branch_flips,
            siding_requirements=requirements,
        ))

    return validated


# =============================================================================
# Inject Switches Into Main Loop
# =============================================================================

def _inject_switches(
    main_pieces: List[int],
    junctions: List[ValidatedJunction],
    tracker: InventoryTracker,
    catalog: TrackCatalog,
    config: DecoderConfig,
    main_flips: Optional[List[int]] = None,
    drop_log: Optional[List[str]] = None,
) -> Tuple[List[int], List[int], List[SwitchPair]]:
    """Replace main loop pieces at junction positions with switches.

    Returns:
        (augmented_pieces, augmented_flips, switch_pairs). The flips array is
        parallel to ``augmented_pieces``; slots overwritten by a switch get
        flip=0 (switches are not flip-aware).
    """
    if main_flips is None:
        main_flips = [0] * len(main_pieces)
    if not junctions:
        return list(main_pieces), list(main_flips), []

    augmented = list(main_pieces)
    augmented_flips = list(main_flips)
    switch_pairs: List[SwitchPair] = []
    used_positions: Set[int] = set()
    n_main = len(augmented)

    pair_id = 0

    def _log_drop(junc, reason):
        if drop_log is not None:
            drop_log.append(f"junction[{junc.slot}] (pos {junc.position}): {reason}")

    for junc in junctions:
        in_pos = junc.position

        if in_pos in used_positions or in_pos >= n_main:
            _log_drop(junc, "IN position occupied or out of range")
            _release_junction_inventory(junc, tracker)
            continue

        template = junc.template
        required_dist = compute_required_main_distance(template, junc.n_straights)

        out_pos = _find_out_position(augmented, augmented_flips, in_pos, required_dist, catalog)

        if out_pos is None or out_pos >= n_main or out_pos <= in_pos:
            _log_drop(junc, "no OUT position at the required main distance")
            _release_junction_inventory(junc, tracker)
            continue

        if out_pos in used_positions:
            _log_drop(junc, f"OUT position {out_pos} occupied")
            _release_junction_inventory(junc, tracker)
            continue

        in_state = _state_at_position(augmented, augmented_flips, in_pos, catalog)

        entry_switch_idx, exit_switch_idx = switch_indices_for(template)
        orig_in = augmented[in_pos]
        orig_in_flip = augmented_flips[in_pos]
        augmented[in_pos] = entry_switch_idx
        augmented_flips[in_pos] = 0
        out_state = _state_at_position(augmented, augmented_flips, out_pos, catalog)

        if not is_valid_siding(
            in_state, out_state, template, junc.n_straights,
            position_tolerance=config.siding_position_tolerance,
            angle_tolerance=config.siding_angle_tolerance,
        ):
            _log_drop(junc, "siding geometry invalid (branch endpoint mismatch)")
            augmented[in_pos] = orig_in
            augmented_flips[in_pos] = orig_in_flip
            _release_junction_inventory(junc, tracker)
            continue

        orig_out = augmented[out_pos]
        tracker.release(orig_in)
        tracker.release(orig_out)
        augmented[out_pos] = exit_switch_idx
        augmented_flips[out_pos] = 0

        # Mark IN, OUT, and everything between as occupied (no overlapping junctions)
        used_positions.update(range(in_pos, out_pos + 1))

        switch_pairs.append(SwitchPair(
            pair_id=pair_id,
            in_position=in_pos,
            out_position=out_pos,
            handedness=template.handedness,
            branch_pieces=junc.branch_pieces,
            branch_flips=junc.branch_flips,
            merge_fk=template.merge_fk,
        ))
        pair_id += 1

    return augmented, augmented_flips, switch_pairs


# =============================================================================
# Cross-Junction Injection
# =============================================================================

def _inject_cross_junctions(
    main_pieces: List[int],
    x: NDArray,
    dims: PartitionedDimensions,
    tracker: InventoryTracker,
    catalog: TrackCatalog,
    *,
    main_flips: Optional[List[int]] = None,
    drop_log: Optional[List[str]] = None,
) -> List[CrossJunction]:
    """Place a CROSS_90 wherever a descriptor names two perpendicular-coincident slots.

    Each active descriptor (pos_1, pos_2) models ONE physical CROSS_90 traversed
    twice by the loop (a bare self-crossing / figure-8). A descriptor is committed
    iff:
      * both positions are distinct, in range, currently STRAIGHT_16, and free of
        prior switch placements,
      * one CROSS_90 is available, and
      * the FK states at the two slots share a world midpoint and cross at ~90 deg
        (``cross_pair_perpendicular`` — the SAME predicate count_dangling_cross_ports
        uses, so a committed crossing is never counted dangling).

    On commit both slots are set to CROSS_90 and ONE physical CROSS_90 is consumed.
    Because CROSS_90 FK == STRAIGHT_16 FK ([16, 0, 0]), the rewrite preserves the FK
    chain — closure is unaffected. This is the deliberate placement path; emergent
    crossings are handled by ``_apply_crossing_repair``.
    """
    descriptors = get_active_cross_junctions(x, dims)
    if not descriptors:
        return []

    n_main = len(main_pieces)
    if n_main < 2:
        return []
    if main_flips is None:
        main_flips = [0] * n_main

    occupied = {pos for pos, p in enumerate(main_pieces) if p in SWITCH_INDICES}
    states = compute_fk_chain(fk_array_with_flips(catalog, main_pieces, main_flips))
    cross_junctions: List[CrossJunction] = []

    def _log_drop(slot, p1, p2, reason):
        if drop_log is not None:
            drop_log.append(f"CROSS[{slot}] (pos {p1},{p2}): {reason}")

    for slot, _active, p1, p2 in descriptors:
        p1, p2 = (p1, p2) if p1 < p2 else (p2, p1)
        if p1 == p2 or p1 < 0 or p2 >= n_main:
            _log_drop(slot, p1, p2, "invalid positions")
            continue
        if p1 in occupied or p2 in occupied:
            _log_drop(slot, p1, p2, "position occupied")
            continue
        if main_pieces[p1] != STRAIGHT_16 or main_pieces[p2] != STRAIGHT_16:
            _log_drop(slot, p1, p2, "slots not STRAIGHT_16")
            continue
        if not tracker.can_use(CROSS_90_INDEX):
            _log_drop(slot, p1, p2, "no CROSS_90 inventory")
            continue
        # Validate with the dangling-port predicate's own tolerances so a
        # committed crossing is guaranteed feasible (never counted dangling).
        if not cross_pair_perpendicular(states, p1, p2):
            _log_drop(slot, p1, p2, "not perpendicular-coincident")
            continue

        tracker.release(main_pieces[p1])
        tracker.release(main_pieces[p2])
        tracker.use(CROSS_90_INDEX)
        main_pieces[p1] = CROSS_90_INDEX
        main_pieces[p2] = CROSS_90_INDEX
        main_flips[p1] = 0
        main_flips[p2] = 0
        occupied.update((p1, p2))
        origin = (float(states[p1][0]), float(states[p1][1]), float(states[p1][2]))
        cross_junctions.append(CrossJunction(
            slot=slot, positions=(p1, p2), origin=origin,
        ))

    return cross_junctions


# =============================================================================
# Double-Crossover Injection
# =============================================================================

def _inject_double_crossovers(
    main_pieces: List[int],
    x: NDArray,
    dims: PartitionedDimensions,
    tracker: InventoryTracker,
    catalog: TrackCatalog,
    config: DecoderConfig,
    *,
    main_flips: Optional[List[int]] = None,
    drop_log: Optional[List[str]] = None,
) -> Tuple[List[DblCrossover], Dict[int, int]]:
    """Replace pairs of main-loop slots with a physical DOUBLE_CROSSOVER.

    Each active descriptor (pos_1, route_1, pos_2, route_2) is committed iff:
      * routes form a valid 2-route cover of {A,B,C,D} (no dangling),
      * both positions are distinct, in range, currently STRAIGHT_16, and free
        from prior switch/cross-junction placements,
      * the chromosome's FK chain — with the route at pos_1 applied — lands at
        pos_2 such that both port-A world poses agree within tolerance, and
      * DOUBLE_CROSSOVER inventory is available.

    Returns (records, route_map) where route_map maps each replaced main-loop
    position to its catalog-route index so _compute_path_fk can pull the right
    FK row at each traversal.
    """
    descriptors = get_active_double_crossovers(x, dims)
    if not descriptors:
        return [], {}

    n_main = len(main_pieces)
    if n_main < 2:
        return [], {}
    if main_flips is None:
        main_flips = [0] * n_main

    pos_tol = config.siding_position_tolerance
    ang_tol_deg = config.siding_angle_tolerance
    occupied = {pos for pos, p in enumerate(main_pieces) if p in SWITCH_INDICES}
    route_map: Dict[int, int] = {}
    records: List[DblCrossover] = []

    def _log_drop(slot, p1, p2, reason):
        if drop_log is not None:
            drop_log.append(f"DC[{slot}] (pos {p1},{p2}): {reason}")

    for slot, _active, p1, r1, p2, r2 in descriptors:
        # Order by position, keeping each route bound to its own position.
        if p1 > p2:
            (p1, r1), (p2, r2) = (p2, r2), (p1, r1)
        if p1 == p2 or p1 >= n_main or p2 >= n_main:
            _log_drop(slot, p1, p2, "invalid positions")
            continue
        if not dbl_crossover_routes_cover_all_ports(r1, r2):
            _log_drop(slot, p1, p2, f"routes {r1},{r2} do not cover all 4 ports")
            continue
        if p1 in occupied or p2 in occupied:
            _log_drop(slot, p1, p2, "position occupied")
            continue
        if main_pieces[p1] != STRAIGHT_16 or main_pieces[p2] != STRAIGHT_16:
            _log_drop(slot, p1, p2, "slots not STRAIGHT_16")
            continue
        if not tracker.can_use_batch(get_dbl_crossover_inventory_requirements()):
            _log_drop(slot, p1, p2, "no DOUBLE_CROSSOVER inventory")
            continue

        # Build a tentative pieces+route_map view so the FK chain at p2 sees
        # the actual route the train takes at p1.
        tentative = list(main_pieces)
        tentative_flips = list(main_flips)
        tentative_routes = dict(route_map)
        state_p1 = _state_at_position(
            tentative, tentative_flips, p1, catalog, tentative_routes,
        )
        tentative[p1] = DC_PIECE_IDX
        tentative_flips[p1] = 0
        tentative_routes[p1] = r1
        state_p2 = _state_at_position(
            tentative, tentative_flips, p2, catalog, tentative_routes,
        )

        origin_1 = dbl_crossover_piece_origin(state_p1, r1)
        origin_2 = dbl_crossover_piece_origin(state_p2, r2)
        if not _piece_origins_match(origin_1, origin_2, pos_tol, ang_tol_deg):
            _log_drop(slot, p1, p2, "piece origins do not coincide (FK mismatch)")
            continue

        # Consume one DOUBLE_CROSSOVER, record the route at each occupied slot.
        tracker.release(main_pieces[p1])
        tracker.release(main_pieces[p2])
        tracker.use_batch(get_dbl_crossover_inventory_requirements())
        main_pieces[p1] = DC_PIECE_IDX
        main_pieces[p2] = DC_PIECE_IDX
        main_flips[p1] = 0
        main_flips[p2] = 0
        route_map[p1] = r1
        route_map[p2] = r2
        occupied.update((p1, p2))
        records.append(DblCrossover(
            slot=slot,
            positions=(p1, p2),
            routes=(r1, r2),
            origin=origin_1,
        ))

    return records, route_map


def _state_at_position(
    pieces: List[int],
    flips: List[int],
    position: int,
    catalog: TrackCatalog,
    route_map: Optional[Dict[int, int]] = None,
) -> Tuple[float, float, float]:
    """FK state entering ``position`` (origin-based walk of pieces[:position]).

    ``route_map`` maps main-loop positions to catalog-route indices for
    multi-route pieces (DOUBLE_CROSSOVER slots); without it the vectorized
    flip-aware lookup is used.
    """
    if position <= 0:
        return (0.0, 0.0, 0.0)
    if route_map:
        deltas = np.empty((position, 3), dtype=np.float64)
        for i in range(position):
            if i in route_map:
                deltas[i] = catalog.get_fk_route(pieces[i], route_map[i])
            else:
                deltas[i] = get_fk_with_flip(
                    catalog, pieces[i], flips[i] if i < len(flips) else 0,
                )
    else:
        deltas = fk_array_with_flips(catalog, pieces[:position], flips[:position])
    final = compute_fk_chain(deltas)[-1]
    return (float(final[0]), float(final[1]), float(final[2]))


def _piece_origins_match(
    o1: Tuple[float, float, float],
    o2: Tuple[float, float, float],
    pos_tol: float,
    ang_tol_deg: float,
) -> bool:
    """True iff two derived port-A world poses agree within tolerance."""
    if np.hypot(o1[0] - o2[0], o1[1] - o2[1]) > pos_tol:
        return False
    ang_diff = (o1[2] - o2[2] + 180.0) % 360.0 - 180.0
    return abs(ang_diff) <= ang_tol_deg


def _find_out_position(
    pieces: List[int],
    flips: List[int],
    in_pos: int,
    required_dist: float,
    catalog: TrackCatalog,
) -> Optional[int]:
    """Find the main-loop position to place the OUT switch.

    Walks forward from ``in_pos + 1`` accumulating each piece's local-X
    contribution (with the slot's flip applied so R40_CURVE flip=1 contributes
    the same magnitude but opposite dtheta).
    """
    n = len(pieces)
    if in_pos + 2 >= n:
        return None

    in_state = _state_at_position(pieces, flips, in_pos, catalog)
    base_theta = in_state[2]

    cumulative_x = 0.0
    best_pos = None
    best_error = float('inf')

    for pos in range(in_pos + 1, n):
        error = abs(cumulative_x - required_dist)
        if error < best_error:
            best_error = error
            best_pos = pos

        idx = pieces[pos]
        if idx < 0 or idx >= len(catalog._fk_table):
            continue

        fk = get_fk_with_flip(catalog, idx, flips[pos] if pos < len(flips) else 0)
        dx, dy, dtheta = fk[0], fk[1], fk[2]
        theta_rad = np.radians(base_theta)
        cos_t = np.cos(theta_rad)
        sin_t = np.sin(theta_rad)
        local_x = dx * cos_t - dy * sin_t
        cumulative_x += abs(local_x)
        base_theta += dtheta

        if cumulative_x > required_dist + 32.0:
            break

    if best_pos is not None and best_error <= required_dist * 0.2 + 2.0:
        return best_pos

    return None


def _release_junction_inventory(
    junc: ValidatedJunction,
    tracker: InventoryTracker,
) -> None:
    """Return a junction's reserved inventory when it fails validation."""
    for idx, count in junc.siding_requirements.items():
        tracker.release(idx, count)


# =============================================================================
# Self-Intersection Repair
# =============================================================================

def _apply_crossing_repair(
    pieces: List[int],
    tracker: InventoryTracker,
    catalog: TrackCatalog,
    flips: Optional[List[int]] = None,
) -> Tuple[List[int], List[int]]:
    """Mark emergent perpendicular STRAIGHT_16-on-STRAIGHT_16 self-crossings as CROSS_90.

    The by-chance path (complements the deliberate ``_inject_cross_junctions``):
    when the evolved main loop happens to cross itself, a CROSS_90 is placed there.

    FK-neutral policy: a crossing is converted ONLY when both crossing segments are
    STRAIGHT_16 (CROSS_90 FK == STRAIGHT_16 FK == [16, 0, 0], so rewriting one slot
    preserves the chain — closure is untouched) AND the crossing is ~perpendicular
    (``cross_pair_perpendicular``, so the partner segment makes the placed cross
    non-dangling). Crossings involving a curve, or not near 90 deg, are left
    unconverted — they remain a mild ``g_collisions`` penalty rather than being
    rewritten into a closure break or a dangling cross. One CROSS_90 is consumed per
    converted crossing.

    Recomputes FK and re-detects pairs after each conversion so multi-intersection
    layouts converge correctly (each converted slot is exempt from re-detection).
    """
    n = len(pieces)
    if flips is None:
        flips = [0] * n
    if n < 4:
        return pieces, list(flips)

    result = list(pieces)
    result_flips = list(flips)

    while tracker.remaining(CROSS_90_INDEX) > 0:
        states = compute_fk_chain(fk_array_with_flips(catalog, result, result_flips))
        pairs = find_crossing_pairs(states, result)
        target = next(
            (
                (pi, pj) for pi, pj, _ang in pairs
                if result[pi] == STRAIGHT_16 and result[pj] == STRAIGHT_16
                and cross_pair_perpendicular(states, pi, pj)
            ),
            None,
        )
        if target is None:
            break

        pos_i, _pos_j = target
        tracker.release(result[pos_i])
        tracker.use(CROSS_90_INDEX)
        result[pos_i] = CROSS_90_INDEX
        result_flips[pos_i] = 0

    return result, result_flips


# =============================================================================
# Multi-Path Layout Construction
# =============================================================================

def _build_multi_path_layout(
    main_pieces: List[int],
    main_flips: List[int],
    switch_pairs: List[SwitchPair],
    catalog: TrackCatalog,
    main_loop_routes: Optional[Dict[int, int]] = None,
) -> MultiPathLayout:
    """Build MultiPathLayout with all 2^J traversal paths."""
    n_pairs = len(switch_pairs)
    routes = main_loop_routes or {}

    if n_pairs == 0:
        main_path = _compute_single_path(main_pieces, main_flips, [], tuple(), catalog, routes)
        return MultiPathLayout(
            main_loop_pieces=main_pieces,
            switch_pairs=[],
            paths=[main_path],
            loose_port_count=0,
        )

    sorted_pairs = sorted(switch_pairs, key=lambda p: p.in_position)

    paths: List[TraversalPath] = []
    for path_id, choices in enumerate(product([0, 1], repeat=n_pairs)):
        path = _compute_single_path(
            main_pieces, main_flips, sorted_pairs, choices, catalog, routes,
        )
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
    main_flips: List[int],
    switch_pairs: List[SwitchPair],
    route_choices: Tuple[int, ...],
    catalog: TrackCatalog,
    main_loop_routes: Optional[Dict[int, int]] = None,
) -> TraversalPath:
    """Compute FK chain for a single traversal path.

    For each switch pair, either traverse straight-through (choice=0)
    or take the branch (choice=1). The branch path uses the diverge
    route for IN switch, branch pieces, then merge route for OUT switch.

    Args:
        main_pieces: Augmented main loop with switches.
        main_flips: Per-slot R40_CURVE flip bits, parallel to main_pieces.
        switch_pairs: Sorted switch pairs.
        route_choices: Binary tuple, one per switch pair.
        catalog: Track catalog for FK lookup.
        main_loop_routes: Map of main-loop position -> catalog-route index for
            multi-route slots (DOUBLE_CROSSOVER); other slots default to through.

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
    routes = main_loop_routes or {}

    piece_sequence: List[int] = []
    route_indices: List[int] = []
    flip_sequence: List[int] = []
    divergent_ranges: Dict[int, Tuple[int, int]] = {}
    fk_overrides: Dict[int, Tuple[float, float, float]] = {}
    current_pos = 0

    def append_main(pos: int) -> None:
        """Append main_pieces[pos] with its per-position route and flip."""
        piece_sequence.append(main_pieces[pos])
        route_indices.append(routes.get(pos, THROUGH))
        flip_sequence.append(main_flips[pos] if pos < len(main_flips) else 0)

    for i, pair in enumerate(switch_pairs):
        choice = route_choices[i] if i < len(route_choices) else 0

        for pos in range(current_pos, pair.in_position):
            append_main(pos)

        in_seq_idx = len(piece_sequence)
        entry_switch = main_pieces[pair.in_position]
        exit_switch = main_pieces[pair.out_position]

        if choice == 0:
            piece_sequence.append(entry_switch)
            route_indices.append(THROUGH)
            flip_sequence.append(0)
            for pos in range(pair.in_position + 1, pair.out_position):
                append_main(pos)
            piece_sequence.append(exit_switch)
            route_indices.append(THROUGH)
            flip_sequence.append(0)
        else:
            piece_sequence.append(entry_switch)
            route_indices.append(DIVERGING)
            flip_sequence.append(0)
            piece_sequence.extend(pair.branch_pieces)
            route_indices.extend([THROUGH] * len(pair.branch_pieces))
            # Branch flips come from the template (per-curve flip already encoded
            # in SwitchPair.branch_flips by _inject_switches).
            branch_flips = (
                list(pair.branch_flips) if pair.branch_flips
                else [0] * len(pair.branch_pieces)
            )
            if len(branch_flips) < len(pair.branch_pieces):
                branch_flips = branch_flips + [0] * (len(pair.branch_pieces) - len(branch_flips))
            flip_sequence.extend(branch_flips[:len(pair.branch_pieces)])
            piece_sequence.append(exit_switch)
            exit_seq_idx = len(piece_sequence) - 1
            route_indices.append(THROUGH)
            flip_sequence.append(0)
            fk_overrides[exit_seq_idx] = pair.merge_fk
            divergent_ranges[i] = (in_seq_idx, exit_seq_idx)

        current_pos = pair.out_position + 1

    for pos in range(current_pos, len(main_pieces)):
        append_main(pos)

    states = _compute_path_fk(piece_sequence, route_indices, catalog, fk_overrides, flip_sequence)
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
    flips: Optional[List[int]] = None,
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
        flips: Per-piece R40_CURVE flip bits; a set bit negates dy/dtheta for
               R40_CURVE pieces (other piece types ignore it).

    Returns:
        (n+1, 3) state array [x, y, theta].
    """
    if not piece_sequence:
        return np.zeros((1, 3), dtype=np.float64)

    # Gather per-piece local deltas (cheap dict/route lookups), then run the
    # vectorized FK accumulation once.
    n = len(piece_sequence)
    deltas = np.empty((n, 3), dtype=np.float64)

    for i in range(n):
        if fk_overrides is not None and i in fk_overrides:
            deltas[i] = fk_overrides[i]
            continue
        piece = piece_sequence[i]
        route_idx = route_indices[i]
        fk = catalog.get_fk_route(piece, route_idx)
        dx, dy, dtheta = float(fk[0]), float(fk[1]), float(fk[2])
        # Apply per-slot flip for R40_CURVE. Other piece types are
        # symmetric / multi-route and ignore the flip bit.
        if flips is not None and i < len(flips) and flips[i] and piece == int(R40_CURVE):
            dy = -dy
            dtheta = -dtheta
        deltas[i] = (dx, dy, dtheta)

    return compute_fk_chain(deltas)


# =============================================================================
# Auto-Center
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
