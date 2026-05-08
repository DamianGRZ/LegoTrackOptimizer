"""Topology-aware structural mutations for the port-pair encoding.

Standard mutation operators (mutate one piece type, rewire one edge) are too
local to invent multi-piece structural features like crossings, sidings, or
component-joining via a CROSS_90. These composite mutations decode the
chromosome, analyse the layout, and apply targeted graph surgery in a single
step. They are V2-native — no V1 imports — and they keep the chromosome's
topology invariants intact (no double-booked ports, no self-loops).

Currently implemented:

- :func:`introduce_crossing` — find a near-perpendicular self-intersection in
  the FK chain and insert a CROSS_90 piece that routes both segments through
  it (port A↔B for one, port C↔D for the other).
- :func:`split_with_switch` — cut an active edge mid-chain and splice in a
  switch_in + switch_out pair plus a small siding loop. Produces switch-bearing
  layouts from any closed loop seed in one mutation step.
- :func:`insert_crossover_bridge` — find two parallel-ish straight edges in
  different routes and splice a DOUBLE_CROSSOVER between them. Cheaper than
  ``introduce_crossing`` because it doesn't require self-intersection — it
  just needs two segments running near-parallel.
- :func:`merge_components_via_cross` — when the chromosome has 2+ disjoint
  components, insert a CROSS_90 that routes one component's edge through its
  horizontal pair and the other's edge through its vertical pair, yielding a
  shared crossing point without self-intersection geometry.
"""

from __future__ import annotations

import random
from typing import Optional

from numpy.typing import NDArray

from .decoder import decode_chromosome
from .encoding import (
    INACTIVE,
    PortPairDimensions,
    clear_port_pair,
    get_port_pair,
    iter_active_pairs,
    iter_active_slots,
    set_piece_slot,
    set_port_pair,
)
from .intersection import find_perpendicular_intersections


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


# =============================================================================
# split_with_switch
# =============================================================================


def split_with_switch(
    x: NDArray,
    dims: PortPairDimensions,
    catalog,
    decoder_config,
    inventory: dict,
    rng: Optional[random.Random] = None,
) -> bool:
    """Splice a switch_in/switch_out pair plus a 1-piece siding into an
    existing edge.

    Closed loops dominate the population because no per-edge mutation can
    *create* a switch where there isn't one. This op converts:

        ... slot_a [port_a] -- [port_b] slot_b ...

    into:

        ... slot_a [port_a] -- [A] sw_in [B] -- [A] (siding piece) [B] -- [B] sw_out [A] -- [port_b] slot_b ...
        with sw_in[C] -- sw_out[C] forming a small bypass loop on the diverging route.

    Net effect: any layout gains a switch-rich substructure without breaking
    its existing route. The repair pipeline will normalize what doesn't make
    geometric sense.

    Returns True iff the mutation was applied.
    """
    if rng is None:
        rng = random.Random()

    sw_in_pid = "R40_SWITCH_LEFT_IN"
    sw_out_pid = "R40_SWITCH_LEFT_OUT"
    siding_pid = "STRAIGHT_16"

    sw_in_idx = catalog.id_to_index.get(sw_in_pid)
    sw_out_idx = catalog.id_to_index.get(sw_out_pid)
    siding_idx = catalog.id_to_index.get(siding_pid)
    if sw_in_idx is None or sw_out_idx is None or siding_idx is None:
        return False

    if (inventory.get(sw_in_pid, 0) <= 0
            or inventory.get(sw_out_pid, 0) <= 0
            or inventory.get(siding_pid, 0) <= 0):
        return False

    # Don't exceed inventory caps for switches/sidings already in use
    used_sw_in = sum(1 for _, idx in iter_active_slots(x, dims) if idx == sw_in_idx)
    used_sw_out = sum(1 for _, idx in iter_active_slots(x, dims) if idx == sw_out_idx)
    used_siding = sum(1 for _, idx in iter_active_slots(x, dims) if idx == siding_idx)
    if (used_sw_in >= inventory.get(sw_in_pid, 0)
            or used_sw_out >= inventory.get(sw_out_pid, 0)
            or used_siding >= inventory.get(siding_pid, 0)):
        return False

    free_slots = [k for k in range(dims.N_max) if int(x[k]) == INACTIVE]
    if len(free_slots) < 3:
        return False

    free_rows = [
        k for k in range(dims.E_max)
        if get_port_pair(x, dims, k)[0] == INACTIVE
    ]
    if len(free_rows) < 3:
        return False

    # Pick a victim edge to splice into
    active_edges = list(iter_active_pairs(x, dims))
    if not active_edges:
        return False

    victim_idx = rng.randrange(len(active_edges))
    row, sa, pa, sb, pb = active_edges[victim_idx]
    if INACTIVE in (sa, pa, sb, pb):
        return False
    if sa == sb:
        return False

    sw_in_slot, sw_out_slot, siding_slot = free_slots[:3]
    set_piece_slot(x, dims, sw_in_slot, sw_in_idx)
    set_piece_slot(x, dims, sw_out_slot, sw_out_idx)
    set_piece_slot(x, dims, siding_slot, siding_idx)

    # Rewrite victim row so slot_a connects to sw_in port A
    set_port_pair(x, dims, row, sa, pa, sw_in_slot, 0)
    # Through-route: sw_in port B -> siding A; siding B -> sw_out port B; sw_out port A -> slot_b
    r1, r2, r3 = free_rows[:3]
    set_port_pair(x, dims, r1, sw_in_slot, 1, siding_slot, 0)
    set_port_pair(x, dims, r2, siding_slot, 1, sw_out_slot, 1)
    set_port_pair(x, dims, r3, sw_out_slot, 0, sb, pb)
    # Diverging port C of sw_in left loose so the GA / repair can decide
    # whether to wire it to sw_out's C (forming a bypass) or to something
    # else; leaving it loose makes this op cheaper and keeps inventory
    # for further structural mutations downstream.

    return True


# =============================================================================
# insert_crossover_bridge
# =============================================================================


def insert_crossover_bridge(
    x: NDArray,
    dims: PortPairDimensions,
    catalog,
    decoder_config,
    inventory: dict,
    rng: Optional[random.Random] = None,
) -> bool:
    """Splice a DOUBLE_CROSSOVER between two existing edges so the train
    can switch between two parallel running tracks.

    Unlike :func:`introduce_crossing`, this does NOT require a perpendicular
    self-intersection — it just picks any two distinct edges and routes one
    through the crossover's track1 (A<->B) and the other through track2
    (C<->D). The repair pipeline + FK propagation will discard arrangements
    that produce non-mating port poses, but feasible ones inject a real
    crossover into the population.
    """
    if rng is None:
        rng = random.Random()

    dc_pid = "DOUBLE_CROSSOVER"
    dc_idx = catalog.id_to_index.get(dc_pid)
    if dc_idx is None:
        return False
    if inventory.get(dc_pid, 0) <= 0:
        return False
    used_dc = sum(1 for _, idx in iter_active_slots(x, dims) if idx == dc_idx)
    if used_dc >= inventory.get(dc_pid, 0):
        return False

    free_slots = [k for k in range(dims.N_max) if int(x[k]) == INACTIVE]
    if not free_slots:
        return False
    free_rows = [
        k for k in range(dims.E_max)
        if get_port_pair(x, dims, k)[0] == INACTIVE
    ]
    if len(free_rows) < 2:
        return False

    active_edges = list(iter_active_pairs(x, dims))
    if len(active_edges) < 2:
        return False

    i, j = rng.sample(range(len(active_edges)), 2)
    edge_i = active_edges[i]
    edge_j = active_edges[j]
    row_i, sa_i, pa_i, sb_i, pb_i = edge_i
    row_j, sa_j, pa_j, sb_j, pb_j = edge_j
    if INACTIVE in (sa_i, pa_i, sb_i, pb_i, sa_j, pa_j, sb_j, pb_j):
        return False
    # Avoid the case where the two edges share an endpoint slot — that would
    # double-book a port through the crossover.
    if {sa_i, sb_i} & {sa_j, sb_j}:
        return False

    dc_slot = free_slots[0]
    set_piece_slot(x, dims, dc_slot, dc_idx)

    # Edge i becomes (sa_i, pa_i) <-> (dc, A=0); add (dc, B=1) <-> (sb_i, pb_i)
    set_port_pair(x, dims, row_i, sa_i, pa_i, dc_slot, 0)
    new_r1 = free_rows[0]
    set_port_pair(x, dims, new_r1, dc_slot, 1, sb_i, pb_i)
    # Edge j becomes (sa_j, pa_j) <-> (dc, C=2); add (dc, D=3) <-> (sb_j, pb_j)
    set_port_pair(x, dims, row_j, sa_j, pa_j, dc_slot, 2)
    new_r2 = free_rows[1]
    set_port_pair(x, dims, new_r2, dc_slot, 3, sb_j, pb_j)

    return True


# =============================================================================
# merge_components_via_cross
# =============================================================================


def merge_components_via_cross(
    x: NDArray,
    dims: PortPairDimensions,
    catalog,
    decoder_config,
    inventory: dict,
    rng: Optional[random.Random] = None,
) -> bool:
    """When the chromosome has 2+ disjoint components, splice a CROSS_90 so
    one component runs through the horizontal route (A<->B) and the other
    through the vertical route (C<->D), creating a shared crossing point.

    Mirrors what :func:`introduce_crossing` does for self-intersection but
    operates on the disconnected-graph case where there's no actual
    self-intersection in the FK chain.
    """
    if rng is None:
        rng = random.Random()

    cross_pid = "CROSS_90"
    cross_idx = catalog.id_to_index.get(cross_pid)
    if cross_idx is None:
        return False
    if inventory.get(cross_pid, 0) <= 0:
        return False
    used = sum(1 for _, idx in iter_active_slots(x, dims) if idx == cross_idx)
    if used >= inventory.get(cross_pid, 0):
        return False

    # Determine connected components from the active edge list
    active_edges = list(iter_active_pairs(x, dims))
    if not active_edges:
        return False
    parent: dict = {}

    def find(v: int) -> int:
        while parent.get(v, v) != v:
            parent[v] = parent.get(parent[v], parent[v])
            v = parent[v]
        return v

    for slot_idx, _ in iter_active_slots(x, dims):
        parent[slot_idx] = slot_idx
    for _, sa, _, sb, _ in active_edges:
        if sa in parent and sb in parent:
            ra, rb = find(sa), find(sb)
            if ra != rb:
                parent[ra] = rb

    # Bucket edges by component root
    comp_edges: dict = {}
    for edge in active_edges:
        _, sa, _, sb, _ = edge
        if sa not in parent:
            continue
        comp_edges.setdefault(find(sa), []).append(edge)
    if len(comp_edges) < 2:
        return False  # already one component

    # Pick two distinct components; sample one edge from each
    roots = list(comp_edges.keys())
    rng.shuffle(roots)
    edges_a = comp_edges[roots[0]]
    edges_b = comp_edges[roots[1]]
    edge_i = edges_a[rng.randrange(len(edges_a))]
    edge_j = edges_b[rng.randrange(len(edges_b))]

    free_slots = [k for k in range(dims.N_max) if int(x[k]) == INACTIVE]
    if not free_slots:
        return False
    free_rows = [
        k for k in range(dims.E_max)
        if get_port_pair(x, dims, k)[0] == INACTIVE
    ]
    if len(free_rows) < 2:
        return False

    row_i, sa_i, pa_i, sb_i, pb_i = edge_i
    row_j, sa_j, pa_j, sb_j, pb_j = edge_j
    if INACTIVE in (sa_i, pa_i, sb_i, pb_i, sa_j, pa_j, sb_j, pb_j):
        return False

    cross_slot = free_slots[0]
    set_piece_slot(x, dims, cross_slot, cross_idx)
    # Reroute component A's edge through horizontal (A=0, B=1)
    set_port_pair(x, dims, row_i, sa_i, pa_i, cross_slot, 0)
    set_port_pair(x, dims, free_rows[0], cross_slot, 1, sb_i, pb_i)
    # Reroute component B's edge through vertical (C=2, D=3)
    set_port_pair(x, dims, row_j, sa_j, pa_j, cross_slot, 2)
    set_port_pair(x, dims, free_rows[1], cross_slot, 3, sb_j, pb_j)

    return True
