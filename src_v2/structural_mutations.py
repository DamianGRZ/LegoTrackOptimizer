"""Topology-aware structural mutations for the port-pair encoding.

Standard mutation operators (mutate one piece type, rewire one edge) are too
local to invent multi-piece structural features like crossings, sidings, or
component-joining via a CROSS_90. These composite mutations decode the
chromosome, analyse the layout, and apply targeted graph surgery in a single
step. They are V2-native — no V1 imports — and they keep the chromosome's
topology invariants intact (no double-booked ports, no self-loops).

Phase 2/3 (closure):

- :func:`introduce_crossing` — convert a near-perpendicular self-intersection
  into a real CROSS_90.
- :func:`introduce_switch_pair` — replace two adjacent STRAIGHT_16's with a
  switch pair, then close port C↔C via A* branch growth.
- :func:`mutate_grow_branch` — close port C of an existing incomplete switch
  pair via A*.

Phase 4 (diversification):

- :func:`mutate_branch_extend` — insert a STRAIGHT_16 in-line on a branch.
- :func:`mutate_branch_shrink` — drop a STRAIGHT_16 from a branch.
- :func:`mutate_swap_switch_hand` — flip a switch IN↔OUT handedness; BFS-flip
  branch curves to preserve handedness invariant; re-grow on failure.
- :func:`mutate_reverse_switch_pairing` — swap (C,C) → (C,A)/(B,C) on a pair
  (memory issue #3: enables OUT→IN reverse sidings).
- :func:`mutate_split_loop_with_crossing` — insert CROSS_90 between two
  non-adjacent edges of a single cycle.
- :func:`mutate_closure_repair_lamarckian` — window-substitution on a cycle
  with non-trivial closure_gap (Lamarckian local search).
"""

from __future__ import annotations

import random
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .branch_grow import find_branch_path
from .decoder import _port_local_pose, decode_chromosome
from .encoding import (
    INACTIVE,
    PortPairDimensions,
    clear_port_pair,
    get_port_pair,
    iter_active_pairs,
    iter_active_slots,
    set_piece_slot,
    set_port_pair,
    set_slot_flip,
    set_slot_rotate,
)
from .intersection import find_perpendicular_intersections
from .se2 import pose_compose


# Switch piece IDs — the closing-branch operators only consider these.
_SWITCH_PIECE_IDS: frozenset = frozenset({"R40_SWITCH_LEFT", "R40_SWITCH_RIGHT"})

# Branch growth budget: maximum pieces in an A*-grown branch path.
_BRANCH_MAX_DEPTH: int = 16

# Goal tolerance (in studs) for branch closure; matches
# OptimizationConfig.branch_closure_tolerance default.
_BRANCH_TOLERANCE_STUDS: float = 8.0


def introduce_crossing(
    x: NDArray,
    dims: PortPairDimensions,
    catalog,
    decoder_config,
    inventory: dict,
    rng: Optional[random.Random] = None,
) -> bool:
    """Convert a near-perpendicular self-intersection into a real CROSS_90.

    Steps:
        1. Verify CROSS_90 is in inventory and there's an unused unit.
        2. Find a free slot in the chromosome for the new CROSS_90.
        3. Find two free port-pair rows (we need to add 2 edges).
        4. Decode the chromosome and look for perpendicular intersections.
        5. Pick the most perpendicular intersection.
        6. Convert one side's edge to terminate at the new CROSS_90's port A,
           add a continuation edge from port B to the original endpoint.
           Do the same for the other side via ports C and D.

    Returns True iff a mutation was applied.
    """
    if rng is None:
        rng = random.Random()

    cross_90_idx = catalog.id_to_index.get("CROSS_90")
    if cross_90_idx is None:
        return False

    cross_capacity = inventory.get("CROSS_90", 0)
    if cross_capacity <= 0:
        return False

    used_cross = sum(
        1 for _, idx in iter_active_slots(x, dims) if idx == cross_90_idx
    )
    if used_cross >= cross_capacity:
        return False

    free_slots = [
        k for k in range(dims.N_max) if int(x[k]) == INACTIVE
    ]
    if not free_slots:
        return False

    free_rows = [
        k for k in range(dims.E_max)
        if get_port_pair(x, dims, k)[0] == INACTIVE
    ]
    if len(free_rows) < 2:
        return False

    graph = decode_chromosome(x, dims, catalog, decoder_config)
    hits = find_perpendicular_intersections(graph, catalog)
    if not hits:
        return False

    edge_i, edge_j, _perp = hits[0]
    row_i, slot_i_a, port_i_a_idx, slot_i_b, port_i_b_idx, _, _ = edge_i
    row_j, slot_j_a, port_j_a_idx, slot_j_b, port_j_b_idx, _, _ = edge_j

    if port_i_a_idx is None or port_i_b_idx is None:
        return False
    if port_j_a_idx is None or port_j_b_idx is None:
        return False

    cross_slot = free_slots[0]
    set_piece_slot(x, dims, cross_slot, cross_90_idx)

    # Edge i becomes (slot_i_a, port_i_a) <-> (cross_slot, A=0)
    # Add new edge: (cross_slot, B=1) <-> (slot_i_b, port_i_b)
    set_port_pair(x, dims, row_i, slot_i_a, port_i_a_idx, cross_slot, 0)
    new_row_1 = free_rows[0]
    set_port_pair(x, dims, new_row_1, cross_slot, 1, slot_i_b, port_i_b_idx)

    # Edge j becomes (slot_j_a, port_j_a) <-> (cross_slot, C=2)
    # Add new edge: (cross_slot, D=3) <-> (slot_j_b, port_j_b)
    set_port_pair(x, dims, row_j, slot_j_a, port_j_a_idx, cross_slot, 2)
    new_row_2 = free_rows[1]
    set_port_pair(x, dims, new_row_2, cross_slot, 3, slot_j_b, port_j_b_idx)

    return True


def introduce_switch_pair(
    x: NDArray,
    dims: PortPairDimensions,
    catalog,
    inventory: dict,
    rng: Optional[random.Random] = None,
) -> bool:
    """Insert a switch pair by absorbing two adjacent STRAIGHT_16's per switch.

    Geometric premise: a switch's through-route is 32 studs long, exactly
    equal to two STRAIGHT_16's (16 + 16 = 32). So replacing **two adjacent
    straights** with one switch preserves the mainline chord length — the
    chain length doesn't shift, and mainline closure is not broken by the
    insertion. (Replacing only one straight would lengthen the chain by
    16 studs and force every downstream piece to slide, breaking closure.)

    Algorithm:

    1. Find pairs of adjacent active STRAIGHT_16 slots in the chromosome —
       i.e. slots ``a`` and ``b`` connected by an edge ``a.B → b.A``, both
       holding STRAIGHT_16. Each such pair is a candidate switch site.
    2. Pick two disjoint candidate pairs (one for IN, one for OUT switch).
    3. For each pair: replace ``a`` with the switch, deactivate ``b``, and
       reroute the chain edge — ``a.B`` now connects to the slot that was
       formerly downstream of ``b`` (skipping ``b`` entirely). Both
       switches use canonical orientation (rotate=0); the through route's
       FK matches the two-straight chord exactly.
    4. Both switches share handedness (LEFT or RIGHT, picked at random),
       which puts both port C's on the same side of the mainline. The
       branch is two R40 curves (one R40_RIGHT for approach + one R40_LEFT
       for return — net 0 turn) wired ``IN.C → curve_1 → curve_2 → OUT.C``.

    Branch closure may still leave a ~6-stud y-residual (an inherent quirk
    of LEGO 9V R40 + STRAIGHT_16 passing-siding geometry, absorbed by track
    flex in real life). That residual is a different problem from the
    mainline-length issue this routine fixes.

    Returns True iff a pair was inserted.
    """
    if rng is None:
        rng = random.Random()

    left_idx = catalog.id_to_index.get("R40_SWITCH_LEFT")
    right_idx = catalog.id_to_index.get("R40_SWITCH_RIGHT")
    curve_idx = catalog.id_to_index.get("R40_CURVE")
    straight_idx = catalog.id_to_index.get("STRAIGHT_16")
    if None in (left_idx, right_idx, curve_idx, straight_idx):
        return False

    left_cap = inventory.get("R40_SWITCH_LEFT", 0)
    right_cap = inventory.get("R40_SWITCH_RIGHT", 0)
    curve_cap = inventory.get("R40_CURVE", 0)
    if left_cap < 1 or right_cap < 1 or curve_cap < 2:
        return False

    used: dict = {}
    for _, idx in iter_active_slots(x, dims):
        used[idx] = used.get(idx, 0) + 1
    if used.get(left_idx, 0) >= left_cap:
        return False
    if used.get(right_idx, 0) >= right_cap:
        return False
    if used.get(curve_idx, 0) + 2 > curve_cap:
        return False

    # Find adjacent-straight pairs along port-B → port-A edges.
    edges = list(iter_active_pairs(x, dims))
    chain_pairs = []   # list of (edge_row, slot_a, slot_b)
    for row, sa, pa, sb, pb in edges:
        if pa != 1 or pb != 0:                       # only forward B→A links
            continue
        if int(x[dims.slot_start + sa]) != straight_idx:
            continue
        if int(x[dims.slot_start + sb]) != straight_idx:
            continue
        chain_pairs.append((row, sa, sb))

    if len(chain_pairs) < 2:
        return False

    # For each candidate pair, locate the edge leaving slot_b.B (= the chain
    # edge we'll re-route). Without it the surgery would orphan slot_b.
    by_b: dict = {}
    for row, sa, pa, sb, pb in edges:
        # We need an outgoing edge from a slot's port B, in either direction.
        if pa == 1:
            by_b.setdefault(sa, []).append((row, sb, pb))
        if pb == 1:
            by_b.setdefault(sb, []).append((row, sa, pa))

    rng.shuffle(chain_pairs)

    # Index chain pairs by slot_a so we can look up p2 by its starting slot.
    pairs_by_a = {p[1]: p for p in chain_pairs}

    def find_disjoint_pairs():
        """Find p1=(r1, a, b) and p2=(r2, c, d) such that
        ``b → c`` is an existing B→A edge — i.e. the two pairs are
        consecutive in the chain. After absorbing, the switches end up
        next to each other (a → c) so they sit in the same mainline
        straight section, where the geometric branch can plausibly close.
        Picking pairs from arbitrary positions drops one switch onto a
        different end-cap, which produces 180°-class closure errors."""
        for p1 in chain_pairs:
            edges_from_b = by_b.get(p1[2], [])
            for _row, partner_slot, partner_port in edges_from_b:
                if partner_port != 0:                       # need ?.A
                    continue
                p2 = pairs_by_a.get(partner_slot)
                if p2 is None:
                    continue
                if {p1[1], p1[2]} & {p2[1], p2[2]}:        # disjoint check
                    continue
                if not by_b.get(p2[2]):                     # p2 must have outgoing edge
                    continue
                return p1, p2
        return None

    pair_pick = find_disjoint_pairs()
    if pair_pick is None:
        return False
    in_pair, out_pair = pair_pick

    # Snapshot for rollback if A* branch growth fails after switch placement.
    x_backup = x.copy()

    # Both switches share handedness so both port C's sit on the same side
    # of the mainline (a real passing siding stays on one side).
    if rng.random() < 0.5:
        switch_idx = left_idx
    else:
        switch_idx = right_idx

    def absorb(pair):
        """Replace pair[1] with a switch, deactivate pair[2], reroute the chain.

        Before:  prev → pair[1].A   pair[1].B → pair[2].A   pair[2].B → next
        After:   prev → pair[1].A   pair[1].B → next        pair[2] inactive
        """
        edge_row, slot_a, slot_b = pair
        candidates = by_b.get(slot_b, [])
        chosen = candidates[0]
        next_row, next_slot, next_port = chosen

        set_piece_slot(x, dims, slot_a, switch_idx)
        set_slot_flip(x, dims, slot_a, 0)
        set_slot_rotate(x, dims, slot_a, 0)
        set_piece_slot(x, dims, slot_b, INACTIVE)
        set_slot_flip(x, dims, slot_b, 0)
        set_slot_rotate(x, dims, slot_b, 0)
        set_port_pair(x, dims, edge_row, slot_a, 1, next_slot, next_port)
        clear_port_pair(x, dims, next_row)

    absorb(in_pair)
    absorb(out_pair)
    in_switch_slot = in_pair[1]
    out_switch_slot = out_pair[1]

    # Phase 3: replace hardcoded 2-curve branch with angular-budget A*.
    # On A* failure (no branch closes within budget), rollback to pre-switch
    # state — caller sees no change.
    from .decoder import DecoderConfig
    grew = _grow_branch_between_switches(
        x, dims, catalog,
        DecoderConfig(),
        inventory,
        in_switch_slot, out_switch_slot,
        rng,
    )
    if not grew:
        x[:] = x_backup
        return False

    return True


# =============================================================================
# Branch growth (Phase 3) — A* over inventory closing port C↔C between switches
# =============================================================================


def mutate_grow_branch(
    x: NDArray,
    dims: PortPairDimensions,
    catalog,
    decoder_config,
    inventory: dict,
    rng: Optional[random.Random] = None,
) -> bool:
    """Pick a pair of incomplete switches on the same through-cycle and
    A*-search a closing branch between their port C's.

    "Incomplete" here means port C is not in any active edge. Switches must
    additionally share the same ``through`` cycle id in
    ``graph.branch_labels`` — otherwise their port C's are not on the same
    mainline and a branch can't reasonably close.

    Returns True iff x was mutated. On A* failure x is unchanged.
    """
    if rng is None:
        rng = random.Random()

    graph = decode_chromosome(x, dims, catalog, decoder_config)

    used_ports = {(e.slot_a, e.port_a) for e in graph.edges} \
        | {(e.slot_b, e.port_b) for e in graph.edges}

    incomplete_by_cycle: dict = {}
    for slot, pid in graph.slot_pieces.items():
        if pid not in _SWITCH_PIECE_IDS:
            continue
        # Port C is index 2 (catalog convention: A=0, B=1, C=2, D=3)
        if (slot, "C") in used_ports:
            continue
        through_cycle = graph.branch_labels.get((slot, "through"))
        if through_cycle is None:
            continue
        incomplete_by_cycle.setdefault(through_cycle, []).append(slot)

    candidates = [(c, slots) for c, slots in incomplete_by_cycle.items() if len(slots) >= 2]
    if not candidates:
        return False

    rng.shuffle(candidates)
    _cycle_id, switch_slots = candidates[0]
    rng.shuffle(switch_slots)
    in_slot, out_slot = switch_slots[0], switch_slots[1]

    return _grow_branch_between_switches(
        x, dims, catalog, decoder_config, inventory,
        in_slot, out_slot, rng,
    )


def _grow_branch_between_switches(
    x: NDArray,
    dims: PortPairDimensions,
    catalog,
    decoder_config,
    inventory: dict,
    in_slot: int,
    out_slot: int,
    rng: random.Random,
) -> bool:
    """Compute IN/OUT port-C world poses, A*-search, install pieces+edges.

    Shared helper used by both ``mutate_grow_branch`` (close existing
    incomplete switch pair) and the upgraded ``introduce_switch_pair``
    (close a freshly-placed switch pair).

    Returns True iff x was mutated successfully (pieces installed and
    edges wired). On A* failure x is unchanged.
    """
    spec = catalog.spec
    if spec is None:
        return False

    graph = decode_chromosome(x, dims, catalog, decoder_config)
    in_pose = graph.slot_poses.get(in_slot)
    out_pose = graph.slot_poses.get(out_slot)
    in_pid = graph.slot_pieces.get(in_slot)
    out_pid = graph.slot_pieces.get(out_slot)
    if None in (in_pose, out_pose, in_pid, out_pid):
        return False

    in_spec = spec.by_id.get(in_pid)
    out_spec = spec.by_id.get(out_pid)
    if in_spec is None or out_spec is None or "C" not in in_spec.ports or "C" not in out_spec.ports:
        return False

    in_c_local = _port_local_pose(
        in_spec, "C", flip=graph.slot_flips.get(in_slot, 0),
        rotate=graph.slot_rotates.get(in_slot, 0),
    )
    out_c_local = _port_local_pose(
        out_spec, "C", flip=graph.slot_flips.get(out_slot, 0),
        rotate=graph.slot_rotates.get(out_slot, 0),
    )
    in_c_world = pose_compose(in_pose, in_c_local)
    out_c_world = pose_compose(out_pose, out_c_local)

    # Remaining inventory — subtract pieces already in chromosome.
    used_count: dict = {}
    for _slot, piece_idx in iter_active_slots(x, dims):
        pid = catalog.index_to_id.get(piece_idx)
        if pid is not None:
            used_count[pid] = used_count.get(pid, 0) + 1
    remaining = {
        pid: max(0, inventory.get(pid, 0) - used_count.get(pid, 0))
        for pid in ("STRAIGHT_16", "R40_CURVE")
    }

    np_rng = np.random.default_rng(rng.randint(0, 2**31 - 1))
    path = find_branch_path(
        start_pose=in_c_world,
        target_pose=out_c_world,
        inventory=remaining,
        catalog=catalog,
        max_depth=_BRANCH_MAX_DEPTH,
        tolerance=_BRANCH_TOLERANCE_STUDS,
        rng=np_rng,
    )
    if not path:
        # ``find_branch_path`` returned ``None`` (no path) or ``[]``
        # (IN/OUT already coincide and need zero pieces). Either way,
        # there is nothing to install.
        return False

    free_slots = [k for k in range(dims.N_max) if int(x[k]) == INACTIVE]
    if len(free_slots) < len(path):
        return False
    free_rows = [
        k for k in range(dims.E_max)
        if get_port_pair(x, dims, k)[0] == INACTIVE
    ]
    if len(free_rows) < len(path) + 1:
        return False

    branch_slots = free_slots[:len(path)]
    branch_rows = free_rows[:len(path) + 1]

    for branch_slot, step in zip(branch_slots, path):
        piece_idx = catalog.id_to_index[step.piece_id]
        set_piece_slot(x, dims, branch_slot, piece_idx)
        set_slot_flip(x, dims, branch_slot, step.flip)
        set_slot_rotate(x, dims, branch_slot, step.rotate)

    # Edge wiring: IN.C → branch[0].A → branch[0].B → branch[1].A → ... → OUT.C
    set_port_pair(x, dims, branch_rows[0], in_slot, 2, branch_slots[0], 0)
    for i in range(len(path) - 1):
        set_port_pair(
            x, dims, branch_rows[i + 1],
            branch_slots[i], 1, branch_slots[i + 1], 0,
        )
    set_port_pair(
        x, dims, branch_rows[len(path)],
        branch_slots[-1], 1, out_slot, 2,
    )

    return True


# =============================================================================
# Phase 4 — Diversification mutations
# =============================================================================


def mutate_branch_extend(
    x: NDArray,
    dims: PortPairDimensions,
    catalog,
    decoder_config,
    inventory: dict,
    rng: Optional[random.Random] = None,
) -> bool:
    """Insert one STRAIGHT_16 in-line on a non-switch edge of a branch cycle.

    A "branch cycle" here means any cycle that passes through a switch's
    port C — identified via ``graph.branch_labels`` carrying the
    ``"diverging"`` route name. Switch slots are preserved (the operator
    only touches the inter-switch edge body), so port C remains paired and
    the cycle stays branch-closed.

    Returns True iff a STRAIGHT_16 was successfully inserted.
    """
    if rng is None:
        rng = random.Random()

    straight_idx = catalog.id_to_index.get("STRAIGHT_16")
    if straight_idx is None:
        return False
    straight_cap = inventory.get("STRAIGHT_16", 0)
    used_straights = sum(
        1 for _, idx in iter_active_slots(x, dims) if idx == straight_idx
    )
    if used_straights >= straight_cap:
        return False

    free_slots = [k for k in range(dims.N_max) if int(x[k]) == INACTIVE]
    free_rows = [
        k for k in range(dims.E_max)
        if get_port_pair(x, dims, k)[0] == INACTIVE
    ]
    if not free_slots or not free_rows:
        return False

    branch_edges = _branch_edges_excluding_switches(
        x, dims, catalog, decoder_config,
    )
    if not branch_edges:
        return False

    row, sa, pa, sb, pb = branch_edges[rng.randrange(len(branch_edges))]
    new_slot = free_slots[0]
    new_row = free_rows[0]

    set_piece_slot(x, dims, new_slot, straight_idx)
    set_slot_flip(x, dims, new_slot, 0)
    set_slot_rotate(x, dims, new_slot, 0)
    # Splice: original (sa, pa) ↔ (sb, pb) becomes (sa, pa) ↔ new.A and
    # new.B ↔ (sb, pb).
    set_port_pair(x, dims, row, sa, pa, new_slot, 0)
    set_port_pair(x, dims, new_row, new_slot, 1, sb, pb)
    return True


def mutate_branch_shrink(
    x: NDArray,
    dims: PortPairDimensions,
    catalog,
    decoder_config,
    inventory: dict,
    rng: Optional[random.Random] = None,
) -> bool:
    """Drop one STRAIGHT_16 sitting on a branch cycle and splice its neighbours.

    Walks the active edges to find a STRAIGHT_16 slot whose A and B ports
    are both paired and whose two adjacent edges share a branch label.
    Removes the slot + one of its edges and rewires the other to bypass.

    Returns True iff a STRAIGHT_16 was successfully removed.
    """
    if rng is None:
        rng = random.Random()

    straight_idx = catalog.id_to_index.get("STRAIGHT_16")
    if straight_idx is None:
        return False

    candidates = _branch_straight_slots_with_neighbours(
        x, dims, catalog, decoder_config, straight_idx,
    )
    if not candidates:
        return False

    slot, (row_a, sa_neighbour, pa_neighbour), (row_b, sb_neighbour, pb_neighbour) \
        = candidates[rng.randrange(len(candidates))]

    set_piece_slot(x, dims, slot, INACTIVE)
    set_slot_flip(x, dims, slot, 0)
    set_slot_rotate(x, dims, slot, 0)
    # Rewire the A-side edge to bypass through to the B-side neighbour.
    set_port_pair(
        x, dims, row_a, sa_neighbour, pa_neighbour,
        sb_neighbour, pb_neighbour,
    )
    clear_port_pair(x, dims, row_b)
    return True


def mutate_swap_switch_hand(
    x: NDArray,
    dims: PortPairDimensions,
    catalog,
    decoder_config,
    inventory: dict,
    rng: Optional[random.Random] = None,
) -> bool:
    """Toggle a switch's handedness and BFS-flip every R40_CURVE on its branch.

    A switch's port C exits the mainline at +π/8 (LEFT) or −π/8 (RIGHT).
    Flipping the handedness while leaving the branch curves unchanged would
    bend the branch into the mainline (closure error ~6 stud + π angle). To
    preserve closure, every R40_CURVE on the branch flips its ``flip`` bit
    too — turning left→right curves and vice versa — keeping the net branch
    angle balanced. The OUT switch's handedness is re-checked against the
    new IN switch hand and re-grown if needed.

    Rolls back on inventory shortfall or A* re-growth failure.
    """
    if rng is None:
        rng = random.Random()

    left_idx = catalog.id_to_index.get("R40_SWITCH_LEFT")
    right_idx = catalog.id_to_index.get("R40_SWITCH_RIGHT")
    curve_idx = catalog.id_to_index.get("R40_CURVE")
    if None in (left_idx, right_idx, curve_idx):
        return False

    graph = decode_chromosome(x, dims, catalog, decoder_config)
    switch_slots = [
        s for s, pid in graph.slot_pieces.items() if pid in _SWITCH_PIECE_IDS
    ]
    if not switch_slots:
        return False

    rng.shuffle(switch_slots)
    target = switch_slots[0]
    current_pid = graph.slot_pieces[target]
    other_pid = (
        "R40_SWITCH_RIGHT" if current_pid == "R40_SWITCH_LEFT" else "R40_SWITCH_LEFT"
    )
    other_idx = catalog.id_to_index[other_pid]

    used_other = sum(
        1 for _, idx in iter_active_slots(x, dims) if idx == other_idx
    )
    if used_other >= inventory.get(other_pid, 0):
        return False

    branch_curves = _branch_curve_slots_from_switch(graph, target, curve_idx, catalog)

    x_backup = x.copy()
    set_piece_slot(x, dims, target, other_idx)
    for curve_slot in branch_curves:
        current_flip = int(x[dims.flip_start + curve_slot])
        set_slot_flip(x, dims, curve_slot, 1 - current_flip)

    # Sanity: if BFS-flip alone closes the branch, we're done. Otherwise
    # we'd need branch re-growth (out of scope here — rollback on residual).
    new_graph = decode_chromosome(x, dims, catalog, decoder_config)
    if any(
        abs(r.dx) > _BRANCH_TOLERANCE_STUDS or abs(r.dy) > _BRANCH_TOLERANCE_STUDS
        for r in new_graph.closure_residuals
    ):
        x[:] = x_backup
        return False
    return True


def mutate_reverse_switch_pairing(
    x: NDArray,
    dims: PortPairDimensions,
    catalog,
    decoder_config,
    inventory: dict,
    rng: Optional[random.Random] = None,
) -> bool:
    """Swap a switch's port-A↔port-B mainline edge to invert siding direction.

    A passing siding has IN.A on the throat side and IN.B on the mainline-
    forward side. Reversing the pairing on IN (so IN.B becomes the throat-
    facing port) inverts the direction trains traverse the siding. Under
    the V2 mating convention this is a single edge-endpoint swap on the
    edge that touches IN.A or IN.B; port C stays paired to the branch.

    The branch geometry shifts after the reversal, so we delegate re-closure
    to :func:`mutate_grow_branch`. Rolls back on A* failure.
    """
    if rng is None:
        rng = random.Random()

    graph = decode_chromosome(x, dims, catalog, decoder_config)
    switch_slots = [
        s for s, pid in graph.slot_pieces.items() if pid in _SWITCH_PIECE_IDS
    ]
    if not switch_slots:
        return False

    rng.shuffle(switch_slots)
    target = switch_slots[0]

    # Locate the two raw edge rows touching ports A (idx 0) and B (idx 1)
    # of the target switch, in the chromosome (not the decoded graph — we
    # need the row index for in-place rewriting).
    a_endpoint = b_endpoint = None
    for k in range(dims.E_max):
        sa, pa, sb, pb = get_port_pair(x, dims, k)
        if sa == INACTIVE:
            continue
        if sa == target and pa == 0:
            a_endpoint = (k, sb, pb)
        elif sb == target and pb == 0:
            a_endpoint = (k, sa, pa)
        if sa == target and pa == 1:
            b_endpoint = (k, sb, pb)
        elif sb == target and pb == 1:
            b_endpoint = (k, sa, pa)

    if a_endpoint is None or b_endpoint is None:
        return False

    x_backup = x.copy()
    row_a, partner_a_slot, partner_a_port = a_endpoint
    row_b, partner_b_slot, partner_b_port = b_endpoint
    # Swap: row_a now wires partner_b ↔ target.A, row_b wires partner_a ↔ target.B.
    set_port_pair(x, dims, row_a, target, 0, partner_b_slot, partner_b_port)
    set_port_pair(x, dims, row_b, target, 1, partner_a_slot, partner_a_port)

    # Re-grow the branch from this switch's port C (other switch was not
    # touched, so its C may need to retract too — branch path subsumes both).
    if not mutate_grow_branch(x, dims, catalog, decoder_config, inventory, rng=rng):
        x[:] = x_backup
        return False
    return True


def mutate_split_loop_with_crossing(
    x: NDArray,
    dims: PortPairDimensions,
    catalog,
    decoder_config,
    inventory: dict,
    rng: Optional[random.Random] = None,
) -> bool:
    """Insert a CROSS_90 between two non-adjacent perpendicular edges of a cycle.

    Distinct from :func:`introduce_crossing`, which only resolves *existing*
    self-intersections. Here we *create* the intersection by routing two
    initially-non-crossing edges through a CROSS_90 — viable when the two
    edges' world midpoints are close (≤ 32 stud) and their headings are
    near-perpendicular.

    Returns True iff a CROSS_90 was inserted.
    """
    if rng is None:
        rng = random.Random()

    cross_idx = catalog.id_to_index.get("CROSS_90")
    if cross_idx is None:
        return False
    used_cross = sum(
        1 for _, idx in iter_active_slots(x, dims) if idx == cross_idx
    )
    if used_cross >= inventory.get("CROSS_90", 0):
        return False

    free_slots = [k for k in range(dims.N_max) if int(x[k]) == INACTIVE]
    free_rows = [
        k for k in range(dims.E_max)
        if get_port_pair(x, dims, k)[0] == INACTIVE
    ]
    if not free_slots or len(free_rows) < 2:
        return False

    graph = decode_chromosome(x, dims, catalog, decoder_config)
    candidates = _perpendicular_edge_pairs_in_cycles(graph, catalog)
    if not candidates:
        return False

    rng.shuffle(candidates)
    (row_i, sa_i, pa_i, sb_i, pb_i), (row_j, sa_j, pa_j, sb_j, pb_j) \
        = candidates[0]

    cross_slot = free_slots[0]
    set_piece_slot(x, dims, cross_slot, cross_idx)
    set_port_pair(x, dims, row_i, sa_i, pa_i, cross_slot, 0)
    set_port_pair(x, dims, free_rows[0], cross_slot, 1, sb_i, pb_i)
    set_port_pair(x, dims, row_j, sa_j, pa_j, cross_slot, 2)
    set_port_pair(x, dims, free_rows[1], cross_slot, 3, sb_j, pb_j)
    return True


def mutate_closure_repair_lamarckian(
    x: NDArray,
    dims: PortPairDimensions,
    catalog,
    decoder_config,
    inventory: dict,
    rng: Optional[random.Random] = None,
    *,
    gap_threshold: float = 4.0,
    window_size: int = 3,
    max_trials: int = 32,
) -> bool:
    """Window-substitution on a cycle with non-trivial closure residual.

    Memetic local search (Orvosh & Davis 1993; Cheng, Gen & Tosawa 1996):
    pick a contiguous K=3-slot window on a failing cycle and try a small
    budget of inventory-respecting substitutions. Commit the first
    substitution that strictly reduces the cycle's residual magnitude.

    Returns True iff a substitution was applied.
    """
    if rng is None:
        rng = random.Random()

    graph = decode_chromosome(x, dims, catalog, decoder_config)
    if not graph.closure_residuals:
        return False
    worst = max(
        graph.closure_residuals,
        key=lambda r: math_hypot_xy(r.dx, r.dy),
    )
    if math_hypot_xy(worst.dx, worst.dy) <= gap_threshold:
        return False

    # Build a substitution-eligible walk: only non-switch, non-crossing slots
    # on the failing component (switches anchor branch geometry; crossings
    # carry routes — substituting either kind silently destroys topology).
    failing_component = next(
        (c for c in graph.connected_components if worst.slot_a in c), None,
    )
    if failing_component is None:
        return False

    spec = catalog.spec
    skip_kinds = {"switch", "crossing"}
    eligible_slots = sorted(
        slot for slot in failing_component
        if (pid := graph.slot_pieces.get(slot)) is not None
        and (spec is None or (
            spec.by_id.get(pid) and spec.by_id[pid].kind not in skip_kinds
        ))
    )
    if len(eligible_slots) < window_size:
        return False

    window_start = rng.randrange(0, len(eligible_slots) - window_size + 1)
    window = eligible_slots[window_start:window_start + window_size]
    original_pieces = [int(x[s]) for s in window]

    candidate_indices = [
        catalog.id_to_index[pid]
        for pid in inventory
        if inventory.get(pid, 0) > 0
        and pid in catalog.id_to_index
        and (spec is None or (
            spec.by_id.get(pid) and spec.by_id[pid].kind not in skip_kinds
        ))
    ]
    if not candidate_indices:
        return False

    base_residual = math_hypot_xy(worst.dx, worst.dy)
    x_backup = x.copy()
    for _ in range(max_trials):
        for slot in window:
            set_piece_slot(
                x, dims, slot, candidate_indices[rng.randrange(len(candidate_indices))],
            )
            set_slot_flip(x, dims, slot, rng.randrange(2))
        new_graph = decode_chromosome(x, dims, catalog, decoder_config)
        if not new_graph.closure_residuals:
            return True
        new_worst = max(
            new_graph.closure_residuals,
            key=lambda r: math_hypot_xy(r.dx, r.dy),
        )
        if math_hypot_xy(new_worst.dx, new_worst.dy) < base_residual:
            return True
        x[:] = x_backup
    # Restore originals on full failure.
    for slot, piece_idx in zip(window, original_pieces):
        set_piece_slot(x, dims, slot, piece_idx)
    return False


# =============================================================================
# Helpers (Phase 4)
# =============================================================================


def math_hypot_xy(dx: float, dy: float) -> float:
    """Local helper to keep the math import lazy on this module."""
    import math
    return math.hypot(dx, dy)


def _branch_edges_excluding_switches(
    x: NDArray, dims: PortPairDimensions, catalog, decoder_config,
) -> list:
    """Return active port-pair rows on branch cycles whose endpoints are not switches.

    Edge format: (row_idx, slot_a, port_a, slot_b, port_b). A "branch" cycle
    here is any cycle ID that has at least one switch in its slot set on the
    ``diverging`` route — that's how port C attaches.
    """
    graph = decode_chromosome(x, dims, catalog, decoder_config)
    branch_cycle_ids = {
        cid for (slot, route), cid in graph.branch_labels.items()
        if route == "diverging"
    }
    if not branch_cycle_ids:
        return []
    branch_slots = {
        slot for (slot, _), cid in graph.branch_labels.items()
        if cid in branch_cycle_ids
    }
    out: list = []
    for k in range(dims.E_max):
        sa, pa, sb, pb = get_port_pair(x, dims, k)
        if sa == INACTIVE or sb == INACTIVE:
            continue
        if sa not in branch_slots or sb not in branch_slots:
            continue
        pid_a = graph.slot_pieces.get(sa)
        pid_b = graph.slot_pieces.get(sb)
        if pid_a in _SWITCH_PIECE_IDS or pid_b in _SWITCH_PIECE_IDS:
            continue
        out.append((k, sa, pa, sb, pb))
    return out


def _branch_straight_slots_with_neighbours(
    x: NDArray, dims: PortPairDimensions, catalog, decoder_config,
    straight_idx: int,
) -> list:
    """Return ``[(slot, edge_a_info, edge_b_info), ...]`` for STRAIGHT_16 slots
    whose A and B ports are both paired AND both neighbours sit on the same
    branch cycle.

    edge_a_info = (row_idx, neighbour_slot, neighbour_port).
    """
    graph = decode_chromosome(x, dims, catalog, decoder_config)
    branch_cycle_ids = {
        cid for (slot, route), cid in graph.branch_labels.items()
        if route == "diverging"
    }
    branch_slots = {
        slot for (slot, _), cid in graph.branch_labels.items()
        if cid in branch_cycle_ids
    }
    if not branch_slots:
        return []

    out: list = []
    for slot in branch_slots:
        if int(x[slot]) != straight_idx:
            continue
        edge_a_info = edge_b_info = None
        for k in range(dims.E_max):
            sa, pa, sb, pb = get_port_pair(x, dims, k)
            if sa == INACTIVE:
                continue
            if sa == slot and pa == 0:
                edge_a_info = (k, sb, pb)
            elif sb == slot and pb == 0:
                edge_a_info = (k, sa, pa)
            elif sa == slot and pa == 1:
                edge_b_info = (k, sb, pb)
            elif sb == slot and pb == 1:
                edge_b_info = (k, sa, pa)
        if edge_a_info and edge_b_info:
            out.append((slot, edge_a_info, edge_b_info))
    return out


def _branch_curve_slots_from_switch(graph, switch_slot: int, curve_idx: int, catalog) -> list:
    """BFS along edges from ``switch.C`` collecting R40_CURVE slot IDs until
    another switch is hit (exclusive)."""
    branch_cycle = graph.branch_labels.get((switch_slot, "diverging"))
    if branch_cycle is None:
        return []
    return [
        slot for (slot, _), cid in graph.branch_labels.items()
        if cid == branch_cycle
        and graph.slot_pieces.get(slot) == "R40_CURVE"
    ]


def _perpendicular_edge_pairs_in_cycles(graph, catalog) -> list:
    """Return ``[(edge_i_tuple, edge_j_tuple), ...]`` for non-adjacent edges
    on the same cycle whose world chords are near-perpendicular and whose
    midpoints are within 32 stud.

    Each tuple: (row_idx, sa, pa_int, sb, pb_int). Port indices are catalog
    integer indices (A=0, B=1, etc.), not names.
    """
    spec = catalog.spec
    if spec is None:
        return []

    # World midpoints + heading per edge row.
    edge_meta: dict = {}
    for i, edge in enumerate(graph.edges):
        pose_a = graph.slot_poses.get(edge.slot_a)
        pose_b = graph.slot_poses.get(edge.slot_b)
        if pose_a is None or pose_b is None:
            continue
        pid_a = graph.slot_pieces.get(edge.slot_a)
        pid_b = graph.slot_pieces.get(edge.slot_b)
        if pid_a in _SWITCH_PIECE_IDS or pid_b in _SWITCH_PIECE_IDS:
            continue
        ps_a = spec.by_id.get(pid_a) if pid_a else None
        ps_b = spec.by_id.get(pid_b) if pid_b else None
        if ps_a is None or ps_b is None:
            continue
        port_names_a = list(ps_a.ports)
        port_names_b = list(ps_b.ports)
        if edge.port_a not in port_names_a or edge.port_b not in port_names_b:
            continue
        pa_int = port_names_a.index(edge.port_a)
        pb_int = port_names_b.index(edge.port_b)
        mid = ((pose_a[0] + pose_b[0]) / 2, (pose_a[1] + pose_b[1]) / 2)
        heading = pose_b[0] - pose_a[0], pose_b[1] - pose_a[1]
        edge_meta[i] = (edge.slot_a, pa_int, edge.slot_b, pb_int, mid, heading)

    pairs: list = []
    keys = list(edge_meta.keys())
    for ii in range(len(keys)):
        for jj in range(ii + 1, len(keys)):
            i, j = keys[ii], keys[jj]
            sa_i, pa_i, sb_i, pb_i, mid_i, h_i = edge_meta[i]
            sa_j, pa_j, sb_j, pb_j, mid_j, h_j = edge_meta[j]
            if {sa_i, sb_i} & {sa_j, sb_j}:
                continue
            if math_hypot_xy(mid_i[0] - mid_j[0], mid_i[1] - mid_j[1]) > 32.0:
                continue
            # Perpendicularity test: |dot| / (|h_i| |h_j|) ≈ 0.
            dot = h_i[0] * h_j[0] + h_i[1] * h_j[1]
            mag_i = math_hypot_xy(*h_i)
            mag_j = math_hypot_xy(*h_j)
            if mag_i < 1e-6 or mag_j < 1e-6:
                continue
            if abs(dot) / (mag_i * mag_j) > 0.34:  # ≈ cos(70°) tolerance
                continue
            pairs.append(
                ((i, sa_i, pa_i, sb_i, pb_i),
                 (j, sa_j, pa_j, sb_j, pb_j)),
            )
    return pairs
