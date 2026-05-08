"""Port-pair operators: sampling, crossover, mutation.

Three pymoo operators wired to the port-pair encoding:

- :class:`PortPairSampling` produces the initial population using a mix of
  heuristic emitters (closed-loop seeds derived from inventory + boundary)
  and random chromosomes. Default 30% heuristic per Phase 0 design.
- :class:`PortPairCrossover` does one-point crossover within each
  chromosome region (slots, pair-rows, anchor) — preserves more structural
  locality than per-gene uniform.
- :class:`PortPairMutation` selects one of seven weighted sub-operators per
  individual, covering slot, edge, and anchor mutations.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.sampling import Sampling

from .catalog import TrackCatalog
from .config import OptimizationConfig
from .decoder import DecoderConfig
from .encoding import (
    DTYPE,
    GENES_PER_PAIR,
    INACTIVE,
    PortPairDimensions,
    clear_port_pair,
    create_empty_chromosome,
    get_port_pair,
    iter_active_pairs,
    iter_active_slots,
    set_anchor,
    set_piece_slot,
    set_port_pair,
)
from .structural_mutations import (
    insert_crossover_bridge,
    introduce_crossing,
    merge_components_via_cross,
    split_with_switch,
)


Pattern = Tuple[List[Tuple[int, str]], List[Tuple[int, str, int, str]]]
"""(slot_assignments, edge_list) — slot_assignments are (slot_idx, piece_id);
edges are (slot_a, port_a_name, slot_b, port_b_name)."""


# =============================================================================
# Heuristic emitters
# =============================================================================


def _emit_simple_loop(
    catalog: TrackCatalog, inventory: Dict[str, int], dims: PortPairDimensions,
) -> List[Pattern]:
    """16 same-direction R40 in a closed cycle."""
    out: List[Pattern] = []
    for piece_id in ("R40_LEFT", "R40_RIGHT"):
        if inventory.get(piece_id, 0) >= 16 and dims.N_max >= 16:
            slots = [(k, piece_id) for k in range(16)]
            edges = [(k, "B", (k + 1) % 16, "A") for k in range(16)]
            out.append((slots, edges))
    return out


def _emit_oval(
    catalog: TrackCatalog, inventory: Dict[str, int], dims: PortPairDimensions,
) -> List[Pattern]:
    """8 R40 + N straights, repeated for the second half of the oval."""
    out: List[Pattern] = []
    n_straights = inventory.get("STRAIGHT_16", 0)
    for curve_id in ("R40_LEFT", "R40_RIGHT"):
        if inventory.get(curve_id, 0) < 16:
            continue
        for n_str_section in (n_straights // 2, n_straights // 4, n_straights // 6):
            n_str_section = max(1, n_str_section)
            total = 16 + 2 * n_str_section
            if total > dims.N_max or 2 * n_str_section > n_straights:
                continue
            slots: List[Tuple[int, str]] = []
            slot_idx = 0
            for _ in range(8):
                slots.append((slot_idx, curve_id))
                slot_idx += 1
            for _ in range(n_str_section):
                slots.append((slot_idx, "STRAIGHT_16"))
                slot_idx += 1
            for _ in range(8):
                slots.append((slot_idx, curve_id))
                slot_idx += 1
            for _ in range(n_str_section):
                slots.append((slot_idx, "STRAIGHT_16"))
                slot_idx += 1
            n = len(slots)
            edges = [(k, "B", (k + 1) % n, "A") for k in range(n)]
            out.append((slots, edges))
    return out


def _emit_racetrack(
    catalog: TrackCatalog, inventory: Dict[str, int], dims: PortPairDimensions,
) -> List[Pattern]:
    """Four 90 deg corners (4 R40 each) + straights between them."""
    out: List[Pattern] = []
    n_straights = inventory.get("STRAIGHT_16", 0)
    for curve_id in ("R40_LEFT", "R40_RIGHT"):
        if inventory.get(curve_id, 0) < 16:
            continue
        for L, S in [(n_straights // 4, n_straights // 8),
                     (n_straights // 6, n_straights // 6)]:
            L = max(1, L)
            S = max(1, S)
            total = 16 + 2 * (L + S)
            if total > dims.N_max or 2 * (L + S) > n_straights:
                continue
            slots: List[Tuple[int, str]] = []
            slot_idx = 0
            for run_len in (L, S, L, S):
                for _ in range(4):
                    slots.append((slot_idx, curve_id))
                    slot_idx += 1
                for _ in range(run_len):
                    slots.append((slot_idx, "STRAIGHT_16"))
                    slot_idx += 1
            n = len(slots)
            edges = [(k, "B", (k + 1) % n, "A") for k in range(n)]
            out.append((slots, edges))
    return out


def _emit_simple_oval_with_siding(
    catalog: TrackCatalog, inventory: Dict[str, int], dims: PortPairDimensions,
) -> List[Pattern]:
    """Oval main loop with a switch-driven siding branching off port C of each switch."""
    out: List[Pattern] = []
    n_str = inventory.get("STRAIGHT_16", 0)
    if (inventory.get("R40_SWITCH_LEFT_IN", 0) < 1
            or inventory.get("R40_SWITCH_LEFT_OUT", 0) < 1
            or inventory.get("R40_LEFT", 0) < 14):
        return out

    n_section = max(1, min(4, n_str // 4))
    n_branch = max(1, min(2, n_str // 4))

    main_n = 14 + 2 * n_section + 2  # 14 R40 + 2*N straights + 2 switches
    total = main_n + n_branch
    if total > dims.N_max:
        return out

    slots: List[Tuple[int, str]] = []
    slot_idx = 0
    for _ in range(7):
        slots.append((slot_idx, "R40_LEFT")); slot_idx += 1
    in_switch = slot_idx
    slots.append((slot_idx, "R40_SWITCH_LEFT_IN")); slot_idx += 1
    for _ in range(n_section):
        slots.append((slot_idx, "STRAIGHT_16")); slot_idx += 1
    out_switch = slot_idx
    slots.append((slot_idx, "R40_SWITCH_LEFT_OUT")); slot_idx += 1
    for _ in range(7):
        slots.append((slot_idx, "R40_LEFT")); slot_idx += 1
    for _ in range(n_section):
        slots.append((slot_idx, "STRAIGHT_16")); slot_idx += 1
    branch_start = slot_idx
    for _ in range(n_branch):
        slots.append((slot_idx, "STRAIGHT_16")); slot_idx += 1

    edges: List[Tuple[int, str, int, str]] = []
    for k in range(main_n):
        edges.append((k, "B", (k + 1) % main_n, "A"))
    edges.append((in_switch, "C", branch_start, "A"))
    for k in range(n_branch - 1):
        edges.append((branch_start + k, "B", branch_start + k + 1, "A"))
    edges.append((branch_start + n_branch - 1, "B", out_switch, "C"))

    out.append((slots, edges))
    return out


def _emit_figure_8(
    catalog: TrackCatalog, inventory: Dict[str, int], dims: PortPairDimensions,
) -> List[Pattern]:
    """Two cycles sharing a CROSS_90 — figure-8 (V2 capability not in V1)."""
    out: List[Pattern] = []
    if inventory.get("CROSS_90", 0) < 1:
        return out
    n_left = inventory.get("R40_LEFT", 0)
    n_right = inventory.get("R40_RIGHT", 0)
    if 33 > dims.N_max:
        return out

    # Try same-direction (16+16 of one kind), then mixed (16 each)
    plans = []
    if n_left >= 32:
        plans.append(("R40_LEFT", "R40_LEFT"))
    if n_right >= 32:
        plans.append(("R40_RIGHT", "R40_RIGHT"))
    if n_left >= 16 and n_right >= 16:
        plans.append(("R40_LEFT", "R40_RIGHT"))
        plans.append(("R40_RIGHT", "R40_LEFT"))

    for lobe_a, lobe_b in plans:
        slots: List[Tuple[int, str]] = [(0, "CROSS_90")]
        for k in range(1, 17):
            slots.append((k, lobe_a))
        for k in range(17, 33):
            slots.append((k, lobe_b))

        edges: List[Tuple[int, str, int, str]] = []
        edges.append((0, "A", 1, "A"))
        for k in range(1, 16):
            edges.append((k, "B", k + 1, "A"))
        edges.append((16, "B", 0, "B"))
        edges.append((0, "C", 17, "A"))
        for k in range(17, 32):
            edges.append((k, "B", k + 1, "A"))
        edges.append((32, "B", 0, "D"))

        out.append((slots, edges))
    return out


def _emit_dense_crossing_grid(
    catalog: TrackCatalog, inventory: Dict[str, int], dims: PortPairDimensions,
) -> List[Pattern]:
    """Multiple cycles joined through several CROSS_90 pieces.

    Pattern: build a chain of cycles, each sharing a CROSS_90 with its
    neighbour(s). This emitter is what unlocks the 'with_crossing' config
    where the optimal layout uses many CROSS_90 pieces.
    """
    out: List[Pattern] = []
    n_cross = inventory.get("CROSS_90", 0)
    if n_cross < 2:
        return out
    n_left = inventory.get("R40_LEFT", 0)
    n_right = inventory.get("R40_RIGHT", 0)
    if n_left < 16 or n_right < 16:
        return out

    # Two CROSS_90s, two "lobes" + a connecting span using both crossings
    # Layout: cycle_1 (16 R40_LEFT) <-> CROSS_90_a <-> CROSS_90_b <-> cycle_2 (16 R40_RIGHT)
    # Each CROSS_90 wires both routes so the train can traverse perpendicular paths.
    total = 2 + 16 + 16
    if total > dims.N_max:
        return out

    slots: List[Tuple[int, str]] = [
        (0, "CROSS_90"),
        (1, "CROSS_90"),
    ]
    for k in range(2, 18):
        slots.append((k, "R40_LEFT"))
    for k in range(18, 34):
        slots.append((k, "R40_RIGHT"))

    edges: List[Tuple[int, str, int, str]] = []
    # Cycle 1 wraps through CROSS_0 horizontal route + 16 R40_LEFT
    edges.append((0, "A", 2, "A"))
    for k in range(2, 17):
        edges.append((k, "B", k + 1, "A"))
    edges.append((17, "B", 0, "B"))
    # Cycle 2 wraps through CROSS_1 horizontal route + 16 R40_RIGHT
    edges.append((1, "A", 18, "A"))
    for k in range(18, 33):
        edges.append((k, "B", k + 1, "A"))
    edges.append((33, "B", 1, "B"))
    # Vertical routes of the two CROSS_90s connect to each other (degenerate
    # but the GA can mutate this into something non-trivial)
    edges.append((0, "C", 1, "C"))
    edges.append((0, "D", 1, "D"))

    out.append((slots, edges))
    return out


def _emit_multi_loop(
    catalog: TrackCatalog, inventory: Dict[str, int], dims: PortPairDimensions,
) -> List[Pattern]:
    """Two completely disconnected R40 LEFT cycles."""
    out: List[Pattern] = []
    if inventory.get("R40_LEFT", 0) < 32:
        return out
    if 32 > dims.N_max:
        return out
    slots = [(k, "R40_LEFT") for k in range(32)]
    edges: List[Tuple[int, str, int, str]] = []
    for k in range(16):
        edges.append((k, "B", (k + 1) % 16, "A"))
    for k in range(16):
        edges.append((16 + k, "B", 16 + ((k + 1) % 16), "A"))
    out.append((slots, edges))
    return out


def _emit_dogbone(
    catalog: TrackCatalog, inventory: Dict[str, int], dims: PortPairDimensions,
) -> List[Pattern]:
    """Two end-loops connected by a straight spine — switch-rich.

    Topology: 8 R40 (loop A) - 1 switch_in - K straights - 1 switch_out
              - 8 R40 (loop B) - return spine of K straights back to start.
    Forces the GA to work with switches arranged on a long axis instead of
    pure loops.
    """
    out: List[Pattern] = []
    if (inventory.get("R40_SWITCH_LEFT_IN", 0) < 1
            or inventory.get("R40_SWITCH_LEFT_OUT", 0) < 1
            or inventory.get("R40_LEFT", 0) < 16):
        return out
    n_str = inventory.get("STRAIGHT_16", 0)
    if n_str < 4:
        return out

    spine = max(2, min(4, n_str // 2))
    return_spine = spine
    total = 16 + 2 + spine + return_spine
    if total > dims.N_max or spine + return_spine > n_str:
        return out

    slots: List[Tuple[int, str]] = []
    idx = 0
    # loop A
    loop_a_start = idx
    for _ in range(8):
        slots.append((idx, "R40_LEFT")); idx += 1
    sw_in = idx
    slots.append((idx, "R40_SWITCH_LEFT_IN")); idx += 1
    spine_start = idx
    for _ in range(spine):
        slots.append((idx, "STRAIGHT_16")); idx += 1
    sw_out = idx
    slots.append((idx, "R40_SWITCH_LEFT_OUT")); idx += 1
    # loop B
    loop_b_start = idx
    for _ in range(8):
        slots.append((idx, "R40_LEFT")); idx += 1
    return_start = idx
    for _ in range(return_spine):
        slots.append((idx, "STRAIGHT_16")); idx += 1

    edges: List[Tuple[int, str, int, str]] = []
    # loop A: 8 R40 in a half-circle, ends connect to switch through-route
    for k in range(7):
        edges.append((loop_a_start + k, "B", loop_a_start + k + 1, "A"))
    edges.append((loop_a_start + 7, "B", sw_in, "A"))   # close into switch
    # diverging port C of switch_in goes back to loop_a_start to close loop A
    edges.append((sw_in, "C", loop_a_start, "A"))
    # spine through-route
    edges.append((sw_in, "B", spine_start, "A"))
    for k in range(spine - 1):
        edges.append((spine_start + k, "B", spine_start + k + 1, "A"))
    edges.append((spine_start + spine - 1, "B", sw_out, "A"))
    # loop B
    edges.append((sw_out, "B", loop_b_start, "A"))
    for k in range(7):
        edges.append((loop_b_start + k, "B", loop_b_start + k + 1, "A"))
    edges.append((loop_b_start + 7, "B", return_start, "A"))
    # return spine closes back to switch_out's diverging port
    for k in range(return_spine - 1):
        edges.append((return_start + k, "B", return_start + k + 1, "A"))
    edges.append((return_start + return_spine - 1, "B", sw_out, "C"))

    out.append((slots, edges))
    return out


def _emit_double_oval_with_crossover(
    catalog: TrackCatalog, inventory: Dict[str, int], dims: PortPairDimensions,
) -> List[Pattern]:
    """Two parallel ovals joined by a DOUBLE_CROSSOVER — the canonical
    layout pattern that the GA was unable to find from random init.

    Each oval: 8 R40 + 4 STRAIGHT_16. The DOUBLE_CROSSOVER splices into
    the straight runs of both, providing the cross-track route. Port mapping
    assumes A/B = track 1 through, C/D = track 2 through (V2 catalog
    convention).
    """
    out: List[Pattern] = []
    if inventory.get("DOUBLE_CROSSOVER", 0) < 1:
        return out
    n_left = inventory.get("R40_LEFT", 0)
    n_str = inventory.get("STRAIGHT_16", 0)
    if n_left < 16 or n_str < 6:
        return out

    n_str_per_side = max(2, min(4, n_str // 4))
    total = 1 + 16 + 4 * n_str_per_side
    if total > dims.N_max or 4 * n_str_per_side > n_str:
        return out

    slots: List[Tuple[int, str]] = [(0, "DOUBLE_CROSSOVER")]
    idx = 1
    # Oval 1: 8 R40 + 2*n_str_per_side straights split into two runs
    o1_curve_a = idx
    for _ in range(4):
        slots.append((idx, "R40_LEFT")); idx += 1
    o1_str_a = idx
    for _ in range(n_str_per_side):
        slots.append((idx, "STRAIGHT_16")); idx += 1
    o1_curve_b = idx
    for _ in range(4):
        slots.append((idx, "R40_LEFT")); idx += 1
    o1_str_b = idx
    for _ in range(n_str_per_side):
        slots.append((idx, "STRAIGHT_16")); idx += 1
    # Oval 2: same layout
    o2_curve_a = idx
    for _ in range(4):
        slots.append((idx, "R40_LEFT")); idx += 1
    o2_str_a = idx
    for _ in range(n_str_per_side):
        slots.append((idx, "STRAIGHT_16")); idx += 1
    o2_curve_b = idx
    for _ in range(4):
        slots.append((idx, "R40_LEFT")); idx += 1
    o2_str_b = idx
    for _ in range(n_str_per_side):
        slots.append((idx, "STRAIGHT_16")); idx += 1

    edges: List[Tuple[int, str, int, str]] = []

    def chain(start: int, length: int) -> None:
        for k in range(length - 1):
            edges.append((start + k, "B", start + k + 1, "A"))

    # Oval 1: curves_a -> straights_a -> curves_b -> straights_b -> back to curves_a,
    # but interleave the DOUBLE_CROSSOVER track1 ports A/B into one of the straight runs.
    chain(o1_curve_a, 4)
    edges.append((o1_curve_a + 3, "B", o1_str_a, "A"))
    chain(o1_str_a, n_str_per_side)
    edges.append((o1_str_a + n_str_per_side - 1, "B", 0, "A"))   # into DC port A
    edges.append((0, "B", o1_curve_b, "A"))                       # out DC port B
    chain(o1_curve_b, 4)
    edges.append((o1_curve_b + 3, "B", o1_str_b, "A"))
    chain(o1_str_b, n_str_per_side)
    edges.append((o1_str_b + n_str_per_side - 1, "B", o1_curve_a, "A"))

    # Oval 2: same shape, splicing in DC ports C/D
    chain(o2_curve_a, 4)
    edges.append((o2_curve_a + 3, "B", o2_str_a, "A"))
    chain(o2_str_a, n_str_per_side)
    edges.append((o2_str_a + n_str_per_side - 1, "B", 0, "C"))
    edges.append((0, "D", o2_curve_b, "A"))
    chain(o2_curve_b, 4)
    edges.append((o2_curve_b + 3, "B", o2_str_b, "A"))
    chain(o2_str_b, n_str_per_side)
    edges.append((o2_str_b + n_str_per_side - 1, "B", o2_curve_a, "A"))

    out.append((slots, edges))
    return out


def _emit_ladder_yard(
    catalog: TrackCatalog, inventory: Dict[str, int], dims: PortPairDimensions,
) -> List[Pattern]:
    """Mainline + ladder of switches feeding parallel sidings.

    Topology: a main loop tapped by 2-3 switches in series, each diverting
    to a short siding. Closed by joining sidings back through return switches.
    Heavy on switches; produces 'yard' shapes the simple-loop bias misses.
    """
    out: List[Pattern] = []
    n_sw_in = inventory.get("R40_SWITCH_LEFT_IN", 0)
    n_sw_out = inventory.get("R40_SWITCH_LEFT_OUT", 0)
    if n_sw_in < 2 or n_sw_out < 2 or inventory.get("R40_LEFT", 0) < 12:
        return out
    n_str = inventory.get("STRAIGHT_16", 0)
    if n_str < 6:
        return out

    n_pairs = min(2, n_sw_in, n_sw_out)
    siding_len = 2
    main_curves = 12
    spine_per_segment = 2
    total = main_curves + 2 * n_pairs + n_pairs * siding_len + (n_pairs + 1) * spine_per_segment
    if total > dims.N_max:
        return out
    needed_str = n_pairs * siding_len + (n_pairs + 1) * spine_per_segment
    if needed_str > n_str:
        return out

    slots: List[Tuple[int, str]] = []
    idx = 0
    # 12 R40 closed half-loops at each end (6 + 6) — turning back the spine
    end_a = idx
    for _ in range(6):
        slots.append((idx, "R40_LEFT")); idx += 1
    spine_start = idx
    for _ in range(spine_per_segment):
        slots.append((idx, "STRAIGHT_16")); idx += 1
    pair_data: List[Tuple[int, int, int]] = []  # (sw_in_idx, siding_start, sw_out_idx)
    for _ in range(n_pairs):
        sw_in = idx; slots.append((idx, "R40_SWITCH_LEFT_IN")); idx += 1
        sw_out = idx; slots.append((idx, "R40_SWITCH_LEFT_OUT")); idx += 1
        siding = idx
        for _ in range(siding_len):
            slots.append((idx, "STRAIGHT_16")); idx += 1
        pair_data.append((sw_in, siding, sw_out))
        # spine connecting to next pair
        for _ in range(spine_per_segment):
            slots.append((idx, "STRAIGHT_16")); idx += 1
    end_b = idx
    for _ in range(6):
        slots.append((idx, "R40_LEFT")); idx += 1

    edges: List[Tuple[int, str, int, str]] = []
    # close end loop A -> spine start
    for k in range(5):
        edges.append((end_a + k, "B", end_a + k + 1, "A"))
    edges.append((end_a + 5, "B", spine_start, "A"))
    for k in range(spine_per_segment - 1):
        edges.append((spine_start + k, "B", spine_start + k + 1, "A"))
    prev_tail = (spine_start + spine_per_segment - 1, "B")
    cursor = spine_start + spine_per_segment
    for sw_in, siding, sw_out in pair_data:
        edges.append((prev_tail[0], prev_tail[1], sw_in, "A"))
        # through route into out-switch
        edges.append((sw_in, "B", sw_out, "B"))
        # diverging route via siding
        edges.append((sw_in, "C", siding, "A"))
        for k in range(siding_len - 1):
            edges.append((siding + k, "B", siding + k + 1, "A"))
        edges.append((siding + siding_len - 1, "B", sw_out, "C"))
        # advance over the siding pieces and onto the connecting spine
        cursor = siding + siding_len + spine_per_segment
        # spine after the pair
        spine_after = sw_out + 1  # first slot after sw_out is start of siding
        # find connecting spine slots: they come AFTER the siding inside the same block
        spine_link = siding + siding_len
        for k in range(spine_per_segment - 1):
            edges.append((spine_link + k, "B", spine_link + k + 1, "A"))
        edges.append((sw_out, "A", spine_link, "A"))
        prev_tail = (spine_link + spine_per_segment - 1, "B")

    # close into end-loop B
    edges.append((prev_tail[0], prev_tail[1], end_b, "A"))
    for k in range(5):
        edges.append((end_b + k, "B", end_b + k + 1, "A"))
    edges.append((end_b + 5, "B", end_a, "A"))  # close the giant loop

    out.append((slots, edges))
    return out


def _emit_classification_yard(
    catalog: TrackCatalog, inventory: Dict[str, int], dims: PortPairDimensions,
) -> List[Pattern]:
    """Open yard: one switch fans into N parallel sidings, no return loop.

    Intentionally produces loose ports (each siding terminates open) — this
    seed exists to put 'open' topologies in the population so the niching
    operator has non-loop signatures to preserve. The repair pipeline can
    close them via add-edge or the GA can leave them open if they help
    diversity.

    Realistically only useful when the inventory has bumpers or many
    switches; otherwise it'll be filtered for infeasibility but still serve
    as a structural seed for mutation.
    """
    out: List[Pattern] = []
    n_sw = inventory.get("R40_SWITCH_LEFT_IN", 0)
    if n_sw < 2 or inventory.get("STRAIGHT_16", 0) < 6:
        return out
    n_branches = min(3, n_sw)
    branch_len = 2
    trunk = 2
    total = n_branches + trunk + n_branches * branch_len
    if total > dims.N_max:
        return out

    slots: List[Tuple[int, str]] = []
    idx = 0
    trunk_start = idx
    for _ in range(trunk):
        slots.append((idx, "STRAIGHT_16")); idx += 1
    sw_indices: List[int] = []
    branch_indices: List[int] = []
    for _ in range(n_branches):
        sw = idx; slots.append((idx, "R40_SWITCH_LEFT_IN")); idx += 1
        sw_indices.append(sw)
        b = idx
        for _ in range(branch_len):
            slots.append((idx, "STRAIGHT_16")); idx += 1
        branch_indices.append(b)

    edges: List[Tuple[int, str, int, str]] = []
    for k in range(trunk - 1):
        edges.append((trunk_start + k, "B", trunk_start + k + 1, "A"))
    edges.append((trunk_start + trunk - 1, "B", sw_indices[0], "A"))
    for i, sw in enumerate(sw_indices):
        if i + 1 < len(sw_indices):
            edges.append((sw, "B", sw_indices[i + 1], "A"))
        b = branch_indices[i]
        edges.append((sw, "C", b, "A"))
        for k in range(branch_len - 1):
            edges.append((b + k, "B", b + k + 1, "A"))

    out.append((slots, edges))
    return out


def _emit_tri_lobe_crossing(
    catalog: TrackCatalog, inventory: Dict[str, int], dims: PortPairDimensions,
) -> List[Pattern]:
    """Three loops sharing a common CROSS_90 — a clover-like layout
    impossible to discover from oval seeds via per-edge mutation alone.

    Routes: one loop uses CROSS horizontal, second uses vertical, third
    cycles port A->C and B->D (the perpendicular non-default route).
    """
    out: List[Pattern] = []
    if inventory.get("CROSS_90", 0) < 1:
        return out
    n_left = inventory.get("R40_LEFT", 0)
    n_right = inventory.get("R40_RIGHT", 0)
    # Each lobe uses 8 R40 of either direction
    if n_left < 16 or n_right < 8:
        return out
    total = 1 + 8 + 8 + 8
    if total > dims.N_max:
        return out

    slots: List[Tuple[int, str]] = [(0, "CROSS_90")]
    idx = 1
    lobe_a = idx
    for _ in range(8):
        slots.append((idx, "R40_LEFT")); idx += 1
    lobe_b = idx
    for _ in range(8):
        slots.append((idx, "R40_LEFT")); idx += 1
    lobe_c = idx
    for _ in range(8):
        slots.append((idx, "R40_RIGHT")); idx += 1

    edges: List[Tuple[int, str, int, str]] = []
    # Lobe A through horizontal route (A<->B)
    edges.append((0, "A", lobe_a, "A"))
    for k in range(7):
        edges.append((lobe_a + k, "B", lobe_a + k + 1, "A"))
    edges.append((lobe_a + 7, "B", 0, "B"))
    # Lobe B through vertical route (C<->D)
    edges.append((0, "C", lobe_b, "A"))
    for k in range(7):
        edges.append((lobe_b + k, "B", lobe_b + k + 1, "A"))
    edges.append((lobe_b + 7, "B", 0, "D"))
    # Lobe C — leave as a disconnected component; this gives the niching
    # operator a 2-component crossing seed that mutation can later splice.
    for k in range(7):
        edges.append((lobe_c + k, "B", lobe_c + k + 1, "A"))
    edges.append((lobe_c + 7, "B", lobe_c, "A"))

    out.append((slots, edges))
    return out


_HEURISTIC_EMITTERS = (
    _emit_simple_loop,
    _emit_oval,
    _emit_racetrack,
    _emit_simple_oval_with_siding,
    _emit_figure_8,
    _emit_multi_loop,
    _emit_dense_crossing_grid,
    _emit_dogbone,
    _emit_double_oval_with_crossover,
    _emit_ladder_yard,
    _emit_classification_yard,
    _emit_tri_lobe_crossing,
)

# Subset that produces topologically-rich seeds (switches, crossings,
# crossovers, multi-component). The complex_emitter_quota mechanism
# guarantees at least 30%% of heuristic seed slots come from this set even
# when simple-loop emitters dominate by virtue of working on every inventory.
_COMPLEX_HEURISTIC_EMITTERS = frozenset({
    _emit_simple_oval_with_siding,
    _emit_figure_8,
    _emit_dense_crossing_grid,
    _emit_dogbone,
    _emit_double_oval_with_crossover,
    _emit_ladder_yard,
    _emit_classification_yard,
    _emit_tri_lobe_crossing,
})


# =============================================================================
# Sampling
# =============================================================================


class PortPairSampling(Sampling):
    """Initial population: heuristic emitters + random chromosomes."""

    def __init__(
        self,
        dims: PortPairDimensions,
        catalog: TrackCatalog,
        config: OptimizationConfig,
        heuristic_ratio: float = 0.30,
        complex_emitter_quota: float = 0.30,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.dims = dims
        self.catalog = catalog
        self.config = config
        self.heuristic_ratio = heuristic_ratio
        self.complex_emitter_quota = max(0.0, min(1.0, complex_emitter_quota))
        self.rng = np.random.default_rng(seed)
        self.id_to_index = catalog.id_to_index
        self.index_to_id = catalog.index_to_id
        self.patterns, self.complex_patterns, self.simple_patterns = self._build_patterns()

    def _build_patterns(self):
        all_patterns: List[Pattern] = []
        complex_patterns: List[Pattern] = []
        simple_patterns: List[Pattern] = []
        for emitter in _HEURISTIC_EMITTERS:
            emitted = emitter(self.catalog, self.config.inventory, self.dims)
            all_patterns.extend(emitted)
            if emitter in _COMPLEX_HEURISTIC_EMITTERS:
                complex_patterns.extend(emitted)
            else:
                simple_patterns.extend(emitted)
        return all_patterns, complex_patterns, simple_patterns

    def _do(self, problem, n_samples, **kwargs) -> NDArray:
        X = np.full((n_samples, self.dims.n_var), INACTIVE, dtype=DTYPE)
        n_heuristic = (
            max(1, int(n_samples * self.heuristic_ratio)) if self.patterns else 0
        )

        # Build the seed pattern queue: enforce a complex_emitter_quota of
        # heuristic slots even if simple emitters dominate by count. Without
        # this, _emit_simple_loop + _emit_oval + _emit_racetrack run on
        # every inventory and crowd out the structurally-richer seeds.
        pattern_queue: List[Pattern] = []
        if n_heuristic > 0:
            n_complex_target = int(n_heuristic * self.complex_emitter_quota)
            if self.complex_patterns and n_complex_target > 0:
                # Repeat-cycle complex patterns up to the quota.
                for k in range(n_complex_target):
                    pattern_queue.append(
                        self.complex_patterns[k % len(self.complex_patterns)]
                    )
            # Fill remainder by cycling simple patterns (or fall back to all
            # patterns if simple is empty for this inventory).
            fill_pool = self.simple_patterns or self.patterns
            n_remainder = n_heuristic - len(pattern_queue)
            for k in range(n_remainder):
                pattern_queue.append(fill_pool[k % len(fill_pool)])
            # Shuffle so complex/simple aren't grouped at the front of the pop.
            self.rng.shuffle(pattern_queue)

        for i in range(n_samples):
            x = create_empty_chromosome(self.dims)
            if i < n_heuristic and pattern_queue:
                self._populate_from_pattern(x, pattern_queue[i])
            else:
                self._populate_random(x)
            self._apply_random_anchor(x)
            X[i, :] = x

        return X

    def _populate_from_pattern(self, x: NDArray, pattern: Pattern) -> None:
        slots, edges = pattern
        slot_to_piece = {s: p for s, p in slots}

        for slot_idx, piece_id in slots:
            if slot_idx >= self.dims.N_max:
                continue
            piece_index = self.id_to_index.get(piece_id)
            if piece_index is None:
                continue
            set_piece_slot(x, self.dims, slot_idx, piece_index)

        for k, (sa, port_a_name, sb, port_b_name) in enumerate(edges):
            if k >= self.dims.E_max:
                break
            port_a_idx = self._port_idx_for_slot(slot_to_piece, sa, port_a_name)
            port_b_idx = self._port_idx_for_slot(slot_to_piece, sb, port_b_name)
            if port_a_idx is None or port_b_idx is None:
                continue
            set_port_pair(x, self.dims, k, sa, port_a_idx, sb, port_b_idx)

    def _port_idx_for_slot(
        self, slot_to_piece: Dict[int, str], slot_idx: int, port_name: str,
    ) -> Optional[int]:
        piece_id = slot_to_piece.get(slot_idx)
        if piece_id is None:
            return None
        spec = self.catalog.spec
        if spec is None:
            return None
        piece_spec = spec.by_id.get(piece_id)
        if piece_spec is None:
            return None
        names = list(piece_spec.ports)
        return names.index(port_name) if port_name in names else None

    def _populate_random(self, x: NDArray) -> None:
        n_active = int(self.rng.integers(4, self.dims.N_max + 1))
        available = [
            pid for pid, count in self.config.inventory.items()
            if count > 0 and pid in self.id_to_index
        ]
        if not available:
            return
        inv_remaining = dict(self.config.inventory)

        active_count = 0
        for k in range(n_active):
            candidates = [pid for pid in available if inv_remaining.get(pid, 0) > 0]
            if not candidates:
                break
            piece_id = candidates[int(self.rng.integers(len(candidates)))]
            piece_index = self.id_to_index[piece_id]
            set_piece_slot(x, self.dims, k, piece_index)
            inv_remaining[piece_id] -= 1
            active_count += 1

        if active_count < 2:
            return

        spec = self.catalog.spec
        n_edges = min(self.dims.E_max, max(1, int(active_count * 1.2)))
        for k in range(n_edges):
            sa = int(self.rng.integers(0, active_count))
            sb = int(self.rng.integers(0, active_count))
            if sa == sb:
                continue
            piece_a_idx = int(x[sa])
            piece_b_idx = int(x[sb])
            piece_a_id = self.index_to_id.get(piece_a_idx)
            piece_b_id = self.index_to_id.get(piece_b_idx)
            if piece_a_id is None or piece_b_id is None:
                continue
            spec_a = spec.by_id.get(piece_a_id) if spec else None
            spec_b = spec.by_id.get(piece_b_id) if spec else None
            if spec_a is None or spec_b is None:
                continue
            port_a = int(self.rng.integers(0, len(spec_a.ports)))
            port_b = int(self.rng.integers(0, len(spec_b.ports)))
            set_port_pair(x, self.dims, k, sa, port_a, sb, port_b)

    def _apply_random_anchor(self, x: NDArray) -> None:
        b = self.config.boundary
        margin_x = max(1, int((b.max_x - b.min_x) * 0.1))
        margin_y = max(1, int((b.max_y - b.min_y) * 0.1))
        ax = int(self.rng.integers(int(b.min_x) + margin_x, int(b.max_x) - margin_x + 1))
        ay = int(self.rng.integers(int(b.min_y) + margin_y, int(b.max_y) - margin_y + 1))
        atheta = int(self.rng.integers(0, 360))
        set_anchor(x, self.dims, ax, ay, atheta)


# =============================================================================
# Crossover
# =============================================================================


class PortPairCrossover(Crossover):
    """One-point crossover per region (slots, pair-rows, anchor uniform)."""

    def __init__(self, dims: PortPairDimensions, prob: float = 0.9) -> None:
        super().__init__(n_parents=2, n_offsprings=2, prob=prob)
        self.dims = dims

    def _do(self, problem, X, **kwargs) -> NDArray:
        _, n_matings, n_var = X.shape
        Y = np.empty((self.n_offsprings, n_matings, n_var), dtype=X.dtype)
        dims = self.dims

        for k in range(n_matings):
            p1 = X[0, k]
            p2 = X[1, k]
            c1 = p1.copy()
            c2 = p2.copy()

            # Slot region: one-point crossover
            if dims.N_max > 1:
                cut = np.random.randint(1, dims.N_max)
                c1[cut:dims.N_max] = p2[cut:dims.N_max]
                c2[cut:dims.N_max] = p1[cut:dims.N_max]

            # Pair region: one-point crossover at row boundary
            if dims.E_max > 1:
                cut_row = np.random.randint(1, dims.E_max)
                cut_gene = dims.pair_start + cut_row * GENES_PER_PAIR
                c1[cut_gene:dims.pair_end] = p2[cut_gene:dims.pair_end]
                c2[cut_gene:dims.pair_end] = p1[cut_gene:dims.pair_end]

            # Anchor: uniform
            if np.random.random() < 0.5:
                c1[dims.anchor_start:] = p2[dims.anchor_start:]
                c2[dims.anchor_start:] = p1[dims.anchor_start:]

            Y[0, k] = c1
            Y[1, k] = c2

        return Y


# =============================================================================
# Mutation
# =============================================================================


class PortPairMutation(Mutation):
    """Weighted sub-operators: piece-type, activate, deactivate, add edge,
    remove edge, rewire edge, perturb anchor, plus three structural ops
    (split_with_switch, insert_crossover_bridge, merge_components_via_cross)."""

    # First seven entries match the original local mutations; the trailing
    # three are the structural ops that can *create* switches/crossings/
    # crossovers from scratch (the missing capability that locked the GA into
    # ovals). Combined weight on structural ops = 0.18 — small but non-zero,
    # since they are decode-heavy and shouldn't dominate.
    OP_WEIGHTS = np.array([
        0.18, 0.13, 0.09, 0.18, 0.09, 0.13, 0.10,   # local ops (was 0.85 total -> 0.90)
        0.04, 0.04, 0.02,                            # split_switch, crossover_bridge, merge_cross
    ])
    OP_WEIGHTS = OP_WEIGHTS / OP_WEIGHTS.sum()  # normalize so np.choice never throws
    # introduce_crossing was promoted from a mutation sub-op to the repair
    # pipeline (PortPairRepairPipeline), where it runs on every individual
    # rather than probabilistically — mirrors V1's repair-injection design.

    def __init__(
        self,
        dims: PortPairDimensions,
        catalog: TrackCatalog,
        config: OptimizationConfig,
        prob: float = 0.3,
    ) -> None:
        super().__init__(prob=prob)
        self.dims = dims
        self.catalog = catalog
        self.config = config
        self.id_to_index = catalog.id_to_index
        self.index_to_id = catalog.index_to_id
        self.decoder_config = DecoderConfig(
            boundary_min_x=config.boundary.min_x,
            boundary_max_x=config.boundary.max_x,
            boundary_min_y=config.boundary.min_y,
            boundary_max_y=config.boundary.max_y,
        )

    def _do(self, problem, X, **kwargs) -> NDArray:
        ops = (
            self._mutate_piece_type,
            self._activate_slot,
            self._deactivate_slot,
            self._add_edge,
            self._remove_edge,
            self._rewire_edge,
            self._perturb_anchor,
            self._split_with_switch,
            self._insert_crossover_bridge,
            self._merge_components_via_cross,
        )
        for i in range(len(X)):
            op_idx = int(np.random.choice(len(ops), p=self.OP_WEIGHTS))
            ops[op_idx](X[i])
        return X

    # ------------------------------------------------------------------
    # Sub-operators
    # ------------------------------------------------------------------

    def _mutate_piece_type(self, x: NDArray) -> None:
        active = list(iter_active_slots(x, self.dims))
        if not active:
            return
        slot_idx, _ = active[np.random.randint(len(active))]
        available = [
            self.id_to_index[pid] for pid, n in self.config.inventory.items()
            if n > 0 and pid in self.id_to_index
        ]
        if not available:
            return
        new_type = available[np.random.randint(len(available))]
        set_piece_slot(x, self.dims, slot_idx, new_type)

    def _activate_slot(self, x: NDArray) -> None:
        inactive = [
            k for k in range(self.dims.N_max)
            if int(x[k]) == INACTIVE
        ]
        if not inactive:
            return
        slot_idx = inactive[np.random.randint(len(inactive))]
        available = [
            self.id_to_index[pid] for pid, n in self.config.inventory.items()
            if n > 0 and pid in self.id_to_index
        ]
        if not available:
            return
        set_piece_slot(x, self.dims, slot_idx, available[np.random.randint(len(available))])

    def _deactivate_slot(self, x: NDArray) -> None:
        active = list(iter_active_slots(x, self.dims))
        if len(active) <= 4:  # keep min for closure
            return
        slot_idx, _ = active[np.random.randint(len(active))]
        set_piece_slot(x, self.dims, slot_idx, INACTIVE)
        # Cascade: drop edges referencing this slot
        for k in range(self.dims.E_max):
            sa, _, sb, _ = get_port_pair(x, self.dims, k)
            if sa == slot_idx or sb == slot_idx:
                clear_port_pair(x, self.dims, k)

    def _add_edge(self, x: NDArray) -> None:
        # Find an inactive pair row
        free_rows = [
            k for k in range(self.dims.E_max)
            if get_port_pair(x, self.dims, k)[0] == INACTIVE
        ]
        if not free_rows:
            return

        loose = self._find_loose_ports(x)
        if len(loose) < 2:
            return

        i, j = np.random.choice(len(loose), size=2, replace=False)
        slot_a, port_a = loose[i]
        slot_b, port_b = loose[j]
        if slot_a == slot_b:
            return
        row = free_rows[np.random.randint(len(free_rows))]
        set_port_pair(x, self.dims, row, slot_a, port_a, slot_b, port_b)

    def _remove_edge(self, x: NDArray) -> None:
        active_rows = [k for k, *_ in iter_active_pairs(x, self.dims)]
        if not active_rows:
            return
        row = active_rows[np.random.randint(len(active_rows))]
        clear_port_pair(x, self.dims, row)

    def _rewire_edge(self, x: NDArray) -> None:
        active_rows = [(k, sa, pa, sb, pb) for k, sa, pa, sb, pb
                       in iter_active_pairs(x, self.dims)]
        if not active_rows:
            return
        row, sa, pa, sb, pb = active_rows[np.random.randint(len(active_rows))]
        loose = self._find_loose_ports(x)
        if not loose:
            return
        new_endpoint = loose[np.random.randint(len(loose))]
        # Replace one endpoint at random
        if np.random.random() < 0.5:
            set_port_pair(x, self.dims, row, new_endpoint[0], new_endpoint[1], sb, pb)
        else:
            set_port_pair(x, self.dims, row, sa, pa, new_endpoint[0], new_endpoint[1])

    def _perturb_anchor(self, x: NDArray) -> None:
        b = self.config.boundary
        delta_x = np.random.randint(-5, 6)
        delta_y = np.random.randint(-5, 6)
        delta_theta = np.random.choice([-22, -11, 11, 22])  # degrees
        ax = int(np.clip(x[self.dims.anchor_start] + delta_x, b.min_x, b.max_x))
        ay = int(np.clip(x[self.dims.anchor_start + 1] + delta_y, b.min_y, b.max_y))
        atheta = int((x[self.dims.anchor_start + 2] + delta_theta) % 360)
        set_anchor(x, self.dims, ax, ay, atheta)

    # ------------------------------------------------------------------
    # Structural sub-operators (delegate to structural_mutations.py)
    # ------------------------------------------------------------------

    def _split_with_switch(self, x: NDArray) -> None:
        split_with_switch(
            x, self.dims, self.catalog, self.decoder_config,
            self.config.inventory,
        )

    def _insert_crossover_bridge(self, x: NDArray) -> None:
        insert_crossover_bridge(
            x, self.dims, self.catalog, self.decoder_config,
            self.config.inventory,
        )

    def _merge_components_via_cross(self, x: NDArray) -> None:
        merge_components_via_cross(
            x, self.dims, self.catalog, self.decoder_config,
            self.config.inventory,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_loose_ports(self, x: NDArray) -> List[Tuple[int, int]]:
        """Return list of (slot_idx, port_idx) tuples that are unused by edges."""
        used: set = set()
        for _, sa, pa, sb, pb in iter_active_pairs(x, self.dims):
            used.add((sa, pa))
            used.add((sb, pb))

        loose: List[Tuple[int, int]] = []
        spec = self.catalog.spec
        if spec is None:
            return loose

        for slot_idx, piece_index in iter_active_slots(x, self.dims):
            piece_id = self.index_to_id.get(piece_index)
            if piece_id is None:
                continue
            piece_spec = spec.by_id.get(piece_id)
            if piece_spec is None:
                continue
            for port_idx in range(len(piece_spec.ports)):
                if (slot_idx, port_idx) not in used:
                    loose.append((slot_idx, port_idx))
        return loose
