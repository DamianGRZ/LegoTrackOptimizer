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
<<<<<<< Updated upstream
from .config import BoundaryConfig, OptimizationConfig
from .decoder import DecoderConfig
from .encoding import (
    ANCHOR_OFFSET_FRACTION,
    DTYPE,
    GENES_PER_PAIR,
    INACTIVE,
    JUNCTION_GENES,
    JUNCTION_KIND_FIGURE_8_CROSS,
    JUNCTION_KIND_PARALLEL_DC_BRIDGE,
    JUNCTION_KIND_PASSING_SIDING,
    JUNCTION_PARAM_MAX,
    JUNC_ACTIVE_OFFSET,
    JUNC_ANCHOR_OFFSET,
    JUNC_PARAM_A_OFFSET,
    JUNC_PARAM_B_OFFSET,
    PortPairDimensions,
    get_junction,
    get_piece_slot,
    set_junction,
    clear_port_pair,
    create_empty_chromosome,
    get_port_pair,
    get_slot_flip,
    get_slot_rotate,
=======
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
>>>>>>> Stashed changes
    iter_active_pairs,
    iter_active_slots,
    set_anchor,
    set_piece_slot,
    set_port_pair,
<<<<<<< Updated upstream
    set_slot_flip,
    set_slot_rotate,
)
from .structural_mutations import (
    introduce_crossing,
    introduce_switch_pair,
    mutate_branch_extend,
    mutate_branch_shrink,
    mutate_grow_branch,
    mutate_reverse_switch_pairing,
    mutate_split_loop_with_crossing,
    mutate_swap_switch_hand,
)


Pattern = Tuple[
    List[Tuple[int, str]],
    List[Tuple[int, str, int, str]],
    Dict[int, int],
    Dict[int, int],
]
"""(slot_assignments, edge_list, flips, rotates):
- slot_assignments: list of (slot_idx, piece_id)
- edges: list of (slot_a, port_a_name, slot_b, port_b_name)
- flips:   dict slot_idx -> 0 or 1; missing slots default to 0
- rotates: dict slot_idx -> 0 or 1; missing slots default to 0
"""


def _is_symmetric_piece(catalog: TrackCatalog, piece_id: str) -> bool:
    """Return True iff the catalog spec marks this piece as symmetric."""
    spec = catalog.spec
    if spec is None:
        return False
    ps = spec.by_id.get(piece_id)
    return bool(ps and ps.symmetric)


def _is_rotatable_piece(catalog: TrackCatalog, piece_id: str) -> bool:
    """Return True iff the catalog spec marks this piece as rotatable."""
    spec = catalog.spec
    if spec is None:
        return False
    ps = spec.by_id.get(piece_id)
    return bool(ps and ps.rotatable)
=======
)
from .structural_mutations import introduce_crossing


Pattern = Tuple[List[Tuple[int, str]], List[Tuple[int, str, int, str]]]
"""(slot_assignments, edge_list) — slot_assignments are (slot_idx, piece_id);
edges are (slot_a, port_a_name, slot_b, port_b_name)."""
>>>>>>> Stashed changes


# =============================================================================
# Heuristic emitters
# =============================================================================

<<<<<<< Updated upstream
# Stadium boundary-fit constants (ported from V1 sampling.py:48-54).
_OVAL_END_CAP_STUDS: float = 80.0   # ~2*R40 diameter; half-circle end-cap axial extent
_OVAL_BOUNDARY_MARGIN_STUDS: float = 20.0
_STRAIGHT_16_LEN_STUDS: int = 16


def _fit_oval_straights_per_section(
    boundary: BoundaryConfig,
    n_str_avail: int,
    extra_axial: float = 0.0,
    piece_len: int = _STRAIGHT_16_LEN_STUDS,
    end_cap: float = _OVAL_END_CAP_STUDS,
    margin: float = _OVAL_BOUNDARY_MARGIN_STUDS,
) -> int:
    """Per-section straight count for an oval, bounded by both inventory and
    boundary fit. Ported from V1 ``sampling.py:_fit_oval_straights_per_section``
    minus the ``max(1, ...)`` floor: when boundary leaves no room, returns 0
    so callers can distinguish "no stadium fits" from "1-straight stadium".
    """
    m_fit = max(0, int((boundary.width - 2 * margin - end_cap - extra_axial) // piece_len))
    return min(m_fit, n_str_avail // 2)

=======
>>>>>>> Stashed changes

def _emit_simple_loop(
    catalog: TrackCatalog, inventory: Dict[str, int], dims: PortPairDimensions,
) -> List[Pattern]:
<<<<<<< Updated upstream
    """16 same-handed R40_CURVE pieces in a closed circle.

    Two variants emitted: flip=0 (left-turn circle) and flip=1 (right-turn).
    Both consume from the same R40_CURVE inventory bucket, since left/right
    is now selected per-placement via the flip bit.
    """
    out: List[Pattern] = []
    if inventory.get("R40_CURVE", 0) < 16 or dims.N_max < 16:
        return out
    for flip in (0, 1):
        slots = [(k, "R40_CURVE") for k in range(16)]
        edges = [(k, "B", (k + 1) % 16, "A") for k in range(16)]
        flips = {k: flip for k in range(16)}
        rotates: Dict[int, int] = {}
        out.append((slots, edges, flips, rotates))
=======
    """16 same-direction R40 in a closed cycle."""
    out: List[Pattern] = []
    for piece_id in ("R40_LEFT", "R40_RIGHT"):
        if inventory.get(piece_id, 0) >= 16 and dims.N_max >= 16:
            slots = [(k, piece_id) for k in range(16)]
            edges = [(k, "B", (k + 1) % 16, "A") for k in range(16)]
            out.append((slots, edges))
>>>>>>> Stashed changes
    return out


def _emit_oval(
    catalog: TrackCatalog, inventory: Dict[str, int], dims: PortPairDimensions,
) -> List[Pattern]:
<<<<<<< Updated upstream
    """8 R40_CURVE + N straights, repeated for the second half of the oval.

    Both flip variants emitted (left- and right-turning ovals).
    """
    out: List[Pattern] = []
    if inventory.get("R40_CURVE", 0) < 16:
        return out
    n_straights = inventory.get("STRAIGHT_16", 0)
    for flip in (0, 1):
=======
    """8 R40 + N straights, repeated for the second half of the oval."""
    out: List[Pattern] = []
    n_straights = inventory.get("STRAIGHT_16", 0)
    for curve_id in ("R40_LEFT", "R40_RIGHT"):
        if inventory.get(curve_id, 0) < 16:
            continue
>>>>>>> Stashed changes
        for n_str_section in (n_straights // 2, n_straights // 4, n_straights // 6):
            n_str_section = max(1, n_str_section)
            total = 16 + 2 * n_str_section
            if total > dims.N_max or 2 * n_str_section > n_straights:
                continue
            slots: List[Tuple[int, str]] = []
<<<<<<< Updated upstream
            flips: Dict[int, int] = {}
            slot_idx = 0
            for _ in range(8):
                slots.append((slot_idx, "R40_CURVE"))
                flips[slot_idx] = flip
=======
            slot_idx = 0
            for _ in range(8):
                slots.append((slot_idx, curve_id))
>>>>>>> Stashed changes
                slot_idx += 1
            for _ in range(n_str_section):
                slots.append((slot_idx, "STRAIGHT_16"))
                slot_idx += 1
            for _ in range(8):
<<<<<<< Updated upstream
                slots.append((slot_idx, "R40_CURVE"))
                flips[slot_idx] = flip
=======
                slots.append((slot_idx, curve_id))
>>>>>>> Stashed changes
                slot_idx += 1
            for _ in range(n_str_section):
                slots.append((slot_idx, "STRAIGHT_16"))
                slot_idx += 1
            n = len(slots)
            edges = [(k, "B", (k + 1) % n, "A") for k in range(n)]
<<<<<<< Updated upstream
            out.append((slots, edges, flips, {}))
    return out


def _emit_sequential_ring_stadium(
    catalog: TrackCatalog,
    inventory: Dict[str, int],
    dims: PortPairDimensions,
    boundary: BoundaryConfig,
) -> List[Pattern]:
    """V1-style stadium emitter (Phase 2): two banks of 8 R40 curves with a
    symmetric m-straight run between them. Pattern:

        [R40_CURVE x 8, STRAIGHT_16 x m, R40_CURVE x 8, STRAIGHT_16 x m]

    The chain is closed as a "sequential ring" by emitting one port-pair
    per slot transition (slot_i.B <-> slot_{i+1}.A wrap-around). Two flip
    variants (left- vs right-turning) seeded; m sized by V1's fractional
    schedule {1.0, 0.75, 0.5, 0.25} of m_max bounded by both inventory and
    boundary fit.
    """
    out: List[Pattern] = []
    if inventory.get("R40_CURVE", 0) < 16:
        return out

    n_str = inventory.get("STRAIGHT_16", 0)
    m_max = _fit_oval_straights_per_section(boundary, n_str)
    if m_max < 1:
        return out

    sizes = sorted(
        {m_max, max(1, m_max * 3 // 4), max(1, m_max // 2), max(1, m_max // 4)},
        reverse=True,
    )
    for flip in (0, 1):
        for m in sizes:
            n = 16 + 2 * m
            if n > dims.N_max or 2 * m > n_str:
                continue
            slots: List[Tuple[int, str]] = []
            flips: Dict[int, int] = {}
            slot_idx = 0
            for _ in range(8):
                slots.append((slot_idx, "R40_CURVE"))
                flips[slot_idx] = flip
                slot_idx += 1
            for _ in range(m):
                slots.append((slot_idx, "STRAIGHT_16"))
                slot_idx += 1
            for _ in range(8):
                slots.append((slot_idx, "R40_CURVE"))
                flips[slot_idx] = flip
                slot_idx += 1
            for _ in range(m):
                slots.append((slot_idx, "STRAIGHT_16"))
                slot_idx += 1
            edges = [(k, "B", (k + 1) % n, "A") for k in range(n)]
            out.append((slots, edges, flips, {}))
=======
            out.append((slots, edges))
>>>>>>> Stashed changes
    return out


def _emit_racetrack(
    catalog: TrackCatalog, inventory: Dict[str, int], dims: PortPairDimensions,
) -> List[Pattern]:
<<<<<<< Updated upstream
    """Four 90 deg corners (4 R40_CURVE each) + straights between them."""
    out: List[Pattern] = []
    if inventory.get("R40_CURVE", 0) < 16:
        return out
    n_straights = inventory.get("STRAIGHT_16", 0)
    for flip in (0, 1):
=======
    """Four 90 deg corners (4 R40 each) + straights between them."""
    out: List[Pattern] = []
    n_straights = inventory.get("STRAIGHT_16", 0)
    for curve_id in ("R40_LEFT", "R40_RIGHT"):
        if inventory.get(curve_id, 0) < 16:
            continue
>>>>>>> Stashed changes
        for L, S in [(n_straights // 4, n_straights // 8),
                     (n_straights // 6, n_straights // 6)]:
            L = max(1, L)
            S = max(1, S)
            total = 16 + 2 * (L + S)
            if total > dims.N_max or 2 * (L + S) > n_straights:
                continue
            slots: List[Tuple[int, str]] = []
<<<<<<< Updated upstream
            flips: Dict[int, int] = {}
            slot_idx = 0
            for run_len in (L, S, L, S):
                for _ in range(4):
                    slots.append((slot_idx, "R40_CURVE"))
                    flips[slot_idx] = flip
=======
            slot_idx = 0
            for run_len in (L, S, L, S):
                for _ in range(4):
                    slots.append((slot_idx, curve_id))
>>>>>>> Stashed changes
                    slot_idx += 1
                for _ in range(run_len):
                    slots.append((slot_idx, "STRAIGHT_16"))
                    slot_idx += 1
            n = len(slots)
            edges = [(k, "B", (k + 1) % n, "A") for k in range(n)]
<<<<<<< Updated upstream
            out.append((slots, edges, flips, {}))
=======
            out.append((slots, edges))
>>>>>>> Stashed changes
    return out


def _emit_simple_oval_with_siding(
    catalog: TrackCatalog, inventory: Dict[str, int], dims: PortPairDimensions,
) -> List[Pattern]:
<<<<<<< Updated upstream
    """Oval main loop with a passing siding (real LEGO topology).

    A *left-side siding* (branch on +y of mainline) needs:

    - IN switch:  R40_SWITCH_LEFT  rotate=0  → port C at NE (+x +y, +π/8 outward)
    - OUT switch: R40_SWITCH_RIGHT rotate=1  → port C at NW (-x +y, +7π/8 outward)
    - Branch:     2× R40_CURVE flip=1 (right turns, -π/8 each) framing N straights
                  Net branch turn: from +π/8 (IN.C outward) → -π/8 (OUT.C inward)

    A *right-side siding* (branch on -y) is the mirror image — RIGHT IN + LEFT
    OUT, with R40_CURVE flip=0 (left turns) framing the branch.

    Why opposite handedness for IN and OUT: the "IN switch" and "OUT switch" of
    a real-world passing siding are NOT two orientations of the same handed
    switch — they're a left-handed switch + a right-handed switch (or vice
    versa). The 180° rotate bit handles the IN→OUT placement, but the rotated
    LEFT switch puts port C on −y, which is the wrong side of the mainline.
    Using a RIGHT switch + rotate=1 puts port C back on +y, where the branch
    actually runs. Confirmed against 4DBrix / LEGO passing-siding plans.
    """
    out: List[Pattern] = []
    n_str = inventory.get("STRAIGHT_16", 0)
    # 16 mainline curves (8 per end-cap = full half-circle each) + 2 branch curves
    if (inventory.get("R40_SWITCH_LEFT", 0) < 1
            or inventory.get("R40_SWITCH_RIGHT", 0) < 1
            or inventory.get("R40_CURVE", 0) < 18):
        return out

    n_branch = max(1, min(3, n_str // 8))    # branch straights between approach + return
    # Section A (the side WITH the switches) holds: IN body 32 + n_section_a*16
    # straights + OUT body 32 = 64 + 16*n_section_a studs along the mainline.
    # Section B (no switches) is just straights, so 16*n_section_b. For the oval
    # to close horizontally, both sections need equal length, so:
    #     n_section_b = n_section_a + 4   (each switch body = 2 STRAIGHT_16)
    # Pick the smallest n_section_a that fits the branch chord — see analysis
    # below; the branch x-extent forces n_section_a ≈ 3.79 + n_branch.
    n_section_a = n_branch + 4
    n_section_b = n_section_a + 4

    if n_section_a + n_section_b > n_str:
        return out

    # 16 main-loop R40 + n_section_a + n_section_b straights + 2 switches
    main_n = 16 + n_section_a + n_section_b + 2
    branch_n = 2 + n_branch                   # 2 curves + n branch straights
    total = main_n + branch_n
    if total > dims.N_max:
        return out

    # Two variants: left-side siding and right-side siding.
    # Each tuple: (in_hand, out_hand, branch_curve_flip)
    #   - branch_curve_flip=1 means right-turning curves (for left-side siding,
    #     branch on +y, descending pattern from +π/8 to -π/8).
    #   - branch_curve_flip=0 means left-turning curves (right-side siding).
    variants = [
        ("R40_SWITCH_LEFT",  "R40_SWITCH_RIGHT", 1),  # left-side siding (+y)
        ("R40_SWITCH_RIGHT", "R40_SWITCH_LEFT",  0),  # right-side siding (-y)
    ]

    for in_id, out_id, branch_flip in variants:
        slots: List[Tuple[int, str]] = []
        flips: Dict[int, int] = {}
        rotates: Dict[int, int] = {}
        slot_idx = 0

        # First half of oval: 8 R40 curves = 8 × 22.5° = 180° (full end-cap)
        for _ in range(8):
            slots.append((slot_idx, "R40_CURVE"))
            flips[slot_idx] = 0
            slot_idx += 1

        # IN switch (canonical)
        in_switch = slot_idx
        slots.append((slot_idx, in_id))
        rotates[slot_idx] = 0
        slot_idx += 1

        # Section A: mainline straights between IN and OUT switches.
        for _ in range(n_section_a):
            slots.append((slot_idx, "STRAIGHT_16"))
            slot_idx += 1

        # OUT switch (rotated 180° — opposite handedness from IN)
        out_switch = slot_idx
        slots.append((slot_idx, out_id))
        rotates[slot_idx] = 1
        slot_idx += 1

        # Second half of oval: 8 R40 curves
        for _ in range(8):
            slots.append((slot_idx, "R40_CURVE"))
            flips[slot_idx] = 0
            slot_idx += 1

        # Section B: closing mainline straights (4 more than section A to
        # offset the 2 × 32-stud switch bodies that sit only in section A).
        for _ in range(n_section_b):
            slots.append((slot_idx, "STRAIGHT_16"))
            slot_idx += 1

        # Branch: approach curve, n straights, return curve.
        branch_start = slot_idx
        approach_curve = slot_idx
        slots.append((slot_idx, "R40_CURVE"))
        flips[slot_idx] = branch_flip
        slot_idx += 1
        for _ in range(n_branch):
            slots.append((slot_idx, "STRAIGHT_16"))
            slot_idx += 1
        return_curve = slot_idx
        slots.append((slot_idx, "R40_CURVE"))
        flips[slot_idx] = branch_flip
        slot_idx += 1

        # Edges: mainline cyclic (B→A) + branch (IN.C → approach → straights → return → OUT.C)
        edges: List[Tuple[int, str, int, str]] = []
        for k in range(main_n):
            edges.append((k, "B", (k + 1) % main_n, "A"))
        edges.append((in_switch, "C", approach_curve, "A"))
        # approach.B → first branch straight.A
        edges.append((approach_curve, "B", branch_start + 1, "A"))
        for k in range(n_branch - 1):
            edges.append((branch_start + 1 + k, "B", branch_start + 1 + k + 1, "A"))
        # last branch straight.B → return.A
        edges.append((branch_start + n_branch, "B", return_curve, "A"))
        # return.B → OUT.C
        edges.append((return_curve, "B", out_switch, "C"))

        out.append((slots, edges, flips, rotates))
    return out


def _emit_asymmetric_oval_with_siding(
    catalog: TrackCatalog,
    inventory: Dict[str, int],
    dims: PortPairDimensions,
    boundary: BoundaryConfig,
) -> List[Pattern]:
    """Phase 5b: V1's asymmetric-oval-with-siding seed.

    Mainline pattern (Rule 10, two corners of 8 R40 = 360 deg):

        [R40 x 8, STR x (m+2), R40 x 8, STR x m]

    Section A (the +2 side) is the siding host; the +2 STR reserves 32-stud
    surplus so the oval still closes after Phase 5a's switch injection
    consumes 2 STRs from section A. Junction descriptor seeded with
    ``active=1, anchor=section_A_offset, kind=PASSING_SIDING, param_b=handedness``;
    Phase 5a's :class:`JunctionMaterializer` expands it during evaluation.

    Falls back to ``[]`` when the inventory lacks switches or the boundary
    is too small for the (m+2)-stadium plus the 32-stud siding reservation.
    """
    out: List[Pattern] = []
    if (inventory.get("R40_CURVE", 0) < 16
            or inventory.get("R40_SWITCH_LEFT", 0) < 1
            or inventory.get("R40_SWITCH_RIGHT", 0) < 1):
        return out

    n_str = inventory.get("STRAIGHT_16", 0)
    # Reserve 32 studs (extra_axial) for the siding's switch-injection surplus.
    m_max = _fit_oval_straights_per_section(boundary, n_str, extra_axial=32.0)
    if m_max < 1:
        return out

    sizes = sorted(
        {m_max, max(1, m_max * 3 // 4), max(1, m_max // 2), max(1, m_max // 4)},
        reverse=True,
    )

    # Per-handedness seeds: (branch_curve_flip, junction_param_b).
    # branch_curve_flip is for section A's mainline curves NOT branch curves;
    # for our purposes here, both flip variants of the mainline emit roughly
    # equivalent layouts -- we vary the junction's param_b (template variant).
    handedness_variants = (
        (0, 0),  # right-turning mainline + LEFT-handed siding (param_b=0)
        (1, 1),  # left-turning mainline + RIGHT-handed siding (param_b=1)
    )

    for flip, param_b in handedness_variants:
        for m in sizes:
            n_section_a = m + 2
            n_section_b = m
            n = 16 + n_section_a + n_section_b
            if n > dims.N_max:
                continue
            if n_section_a + n_section_b > n_str:
                continue

            slots: List[Tuple[int, str]] = []
            flips: Dict[int, int] = {}
            slot_idx = 0
            for _ in range(8):
                slots.append((slot_idx, "R40_CURVE"))
                flips[slot_idx] = flip
                slot_idx += 1
            section_a_start = slot_idx
            for _ in range(n_section_a):
                slots.append((slot_idx, "STRAIGHT_16"))
                slot_idx += 1
            for _ in range(8):
                slots.append((slot_idx, "R40_CURVE"))
                flips[slot_idx] = flip
                slot_idx += 1
            for _ in range(n_section_b):
                slots.append((slot_idx, "STRAIGHT_16"))
                slot_idx += 1
            edges = [(k, "B", (k + 1) % n, "A") for k in range(n)]

            # Anchor the junction near the centre of section A; param_a varies
            # the branch length so the seed gives the GA several siding sizes.
            anchor_centre = section_a_start + max(0, n_section_a // 2 - 1)
            for n_branch in (1, 2, 3):
                if n_branch * 2 > inventory.get("STRAIGHT_16", 0):
                    continue
                # Phase 4 stores junction params as int16 in [0, 255].
                junction = (1, anchor_centre, JUNCTION_KIND_PASSING_SIDING,
                            n_branch, param_b)
                out.append((slots, edges, flips, {}, [junction]))
    return out


def _emit_figure8_stadium(
    catalog: TrackCatalog,
    inventory: Dict[str, int],
    dims: PortPairDimensions,
    boundary: BoundaryConfig,
) -> List[Pattern]:
    """Phase 6b: figure-8 oval seed.

    Mainline pattern (Rule 10, two corners of 8 R40 = 360 deg): symmetric
    ``[R40 x 8, STR x m, R40 x 8, STR x m]`` oval. Junction descriptor
    seeded with ``active=1, kind=FIGURE_8_CROSS, anchor=mid-straight,
    param_a=lobe_curve_count, param_b=handedness``. Phase 6a's
    :class:`JunctionMaterializer` expands the junction into the full
    figure-8 (cross at anchor + secondary lobe).

    Falls back to ``[]`` when the inventory lacks a CROSS_90 or 16 R40,
    or when the boundary is too small for the oval mainline.
    """
    out: List[Pattern] = []
    if (inventory.get("R40_CURVE", 0) < 16
            or inventory.get("CROSS_90", 0) < 1):
        return out

    n_str = inventory.get("STRAIGHT_16", 0)
    m_max = _fit_oval_straights_per_section(boundary, n_str)
    if m_max < 1:
        return out

    sizes = sorted(
        {m_max, max(1, m_max * 3 // 4), max(1, m_max // 2), max(1, m_max // 4)},
        reverse=True,
    )

    for flip in (0, 1):
        for m in sizes:
            n = 16 + 2 * m
            if n > dims.N_max or 2 * m > n_str:
                continue

            slots: List[Tuple[int, str]] = []
            flips: Dict[int, int] = {}
            slot_idx = 0
            for _ in range(8):
                slots.append((slot_idx, "R40_CURVE"))
                flips[slot_idx] = flip
                slot_idx += 1
            section_a_start = slot_idx
            for _ in range(m):
                slots.append((slot_idx, "STRAIGHT_16"))
                slot_idx += 1
            for _ in range(8):
                slots.append((slot_idx, "R40_CURVE"))
                flips[slot_idx] = flip
                slot_idx += 1
            for _ in range(m):
                slots.append((slot_idx, "STRAIGHT_16"))
                slot_idx += 1
            edges = [(k, "B", (k + 1) % n, "A") for k in range(n)]

            anchor = section_a_start + max(0, m // 2 - 1)
            # Multiple lobe sizes per oval: param_a is the materializer's
            # lobe-internal n_straights, param_b is handedness (0/1).
            for lobe_n_straights in (0, 1, 2):
                junction = (
                    1, anchor, JUNCTION_KIND_FIGURE_8_CROSS,
                    lobe_n_straights, flip,
                )
                out.append((slots, edges, flips, {}, [junction]))
=======
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
>>>>>>> Stashed changes
    return out


def _emit_figure_8(
    catalog: TrackCatalog, inventory: Dict[str, int], dims: PortPairDimensions,
) -> List[Pattern]:
<<<<<<< Updated upstream
    """Two cycles sharing a CROSS_90 — figure-8 (diagonal-quadrant lobes).

    Pre-fix versions of this emitter wired the lobes perpendicularly
    (lobe A: cross.A <-> cross.B; lobe B: cross.C <-> cross.D), which
    requires the two lobes to physically cross each other outside the
    cross — geometrically impossible on a flat-track plane. The closure
    search at ``tools/figure8_closure_search.py`` (run 2026-05-07)
    enumerated 3.3 M tuples and confirmed: zero closing parametrizations
    exist for the perpendicular topology. The fix is to use **diagonal
    quadrant lobes**: each lobe sits in one quadrant of the cross's
    plane, ports connected diagonally rather than perpendicularly.

    Closing parametrization per lobe (from the v2 closure search):

        ``M1 STR_16 + 12 R40 same-handed + M2 STR_16``

    with ``(M1, M2)`` = (2, 2) or (2, 3) depending on which port pair the
    lobe connects. The 12 R40 same-handed contribute 270 deg of net
    rotation, which combined with the cross's port-pair heading offset
    produces a chain end that aligns with the next port's pose within
    < 0.002 stud / 0 deg tolerance.

    Two variants emitted: top-left + bottom-right (both lobes LEFT,
    flip=0) and top-right + bottom-left (both lobes RIGHT, flip=1).
    Opposite-handedness mixes don't correspond to clean diagonal
    topologies and aren't included.
    """
    out: List[Pattern] = []
    if inventory.get("CROSS_90", 0) < 1:
        return out
    if inventory.get("R40_CURVE", 0) < 24:
        return out
    if inventory.get("STRAIGHT_16", 0) < 9:
        return out

    # Lobe A: 16 slots (M1=2, 12 R40, M2=2).
    # Lobe B: 17 slots (M1=2, 12 R40, M2=3).
    # 1 cross + 16 + 17 = 34 chromosome slots.
    if 34 > dims.N_max:
        return out

    # (curve_flip, lobe_A_start_port, lobe_A_end_port, m1_a, m2_a,
    #              lobe_B_start_port, lobe_B_end_port, m1_b, m2_b)
    plans: List[Tuple[int, str, str, int, int, str, str, int, int]] = [
        # Top-left (D->A) + bottom-right (B->C), both LEFT-handed.
        (0, "D", "A", 2, 2, "B", "C", 2, 3),
        # Top-right (B->D) + bottom-left (C->A), both RIGHT-handed.
        (1, "B", "D", 2, 3, "C", "A", 2, 2),
    ]

    def _build_lobe(
        start_idx: int, flip: int, m1: int, m2: int,
    ) -> Tuple[List[Tuple[int, str]], Dict[int, int], List[int]]:
        local_slots: List[Tuple[int, str]] = []
        local_flips: Dict[int, int] = {}
        chain: List[int] = []
        idx = start_idx
        for _ in range(m1):
            local_slots.append((idx, "STRAIGHT_16"))
            chain.append(idx)
            idx += 1
        for _ in range(12):
            local_slots.append((idx, "R40_CURVE"))
            local_flips[idx] = flip
            chain.append(idx)
            idx += 1
        for _ in range(m2):
            local_slots.append((idx, "STRAIGHT_16"))
            chain.append(idx)
            idx += 1
        return local_slots, local_flips, chain

    for plan in plans:
        flip, a_start_port, a_end_port, m1_a, m2_a, \
            b_start_port, b_end_port, m1_b, m2_b = plan
        slots: List[Tuple[int, str]] = [(0, "CROSS_90")]
        flips: Dict[int, int] = {}

        a_local, a_flips, a_chain = _build_lobe(1, flip, m1_a, m2_a)
        slots.extend(a_local)
        flips.update(a_flips)

        b_start = 1 + len(a_chain)
        b_local, b_flips, b_chain = _build_lobe(b_start, flip, m1_b, m2_b)
        slots.extend(b_local)
        flips.update(b_flips)

        edges: List[Tuple[int, str, int, str]] = []
        edges.append((0, a_start_port, a_chain[0], "A"))
        for k in range(len(a_chain) - 1):
            edges.append((a_chain[k], "B", a_chain[k + 1], "A"))
        edges.append((a_chain[-1], "B", 0, a_end_port))
        edges.append((0, b_start_port, b_chain[0], "A"))
        for k in range(len(b_chain) - 1):
            edges.append((b_chain[k], "B", b_chain[k + 1], "A"))
        edges.append((b_chain[-1], "B", 0, b_end_port))

        out.append((slots, edges, flips, {}))
=======
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
>>>>>>> Stashed changes
    return out


def _emit_dense_crossing_grid(
    catalog: TrackCatalog, inventory: Dict[str, int], dims: PortPairDimensions,
) -> List[Pattern]:
<<<<<<< Updated upstream
    """Multiple cycles joined through several CROSS_90 pieces."""
    out: List[Pattern] = []
    if inventory.get("CROSS_90", 0) < 2:
        return out
    if inventory.get("R40_CURVE", 0) < 32:
        return out

=======
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
>>>>>>> Stashed changes
    total = 2 + 16 + 16
    if total > dims.N_max:
        return out

    slots: List[Tuple[int, str]] = [
        (0, "CROSS_90"),
        (1, "CROSS_90"),
    ]
<<<<<<< Updated upstream
    flips: Dict[int, int] = {}
    for k in range(2, 18):
        slots.append((k, "R40_CURVE"))
        flips[k] = 0   # left lobe
    for k in range(18, 34):
        slots.append((k, "R40_CURVE"))
        flips[k] = 1   # right lobe

    edges: List[Tuple[int, str, int, str]] = []
=======
    for k in range(2, 18):
        slots.append((k, "R40_LEFT"))
    for k in range(18, 34):
        slots.append((k, "R40_RIGHT"))

    edges: List[Tuple[int, str, int, str]] = []
    # Cycle 1 wraps through CROSS_0 horizontal route + 16 R40_LEFT
>>>>>>> Stashed changes
    edges.append((0, "A", 2, "A"))
    for k in range(2, 17):
        edges.append((k, "B", k + 1, "A"))
    edges.append((17, "B", 0, "B"))
<<<<<<< Updated upstream
=======
    # Cycle 2 wraps through CROSS_1 horizontal route + 16 R40_RIGHT
>>>>>>> Stashed changes
    edges.append((1, "A", 18, "A"))
    for k in range(18, 33):
        edges.append((k, "B", k + 1, "A"))
    edges.append((33, "B", 1, "B"))
<<<<<<< Updated upstream
    edges.append((0, "C", 1, "C"))
    edges.append((0, "D", 1, "D"))

    out.append((slots, edges, flips, {}))
=======
    # Vertical routes of the two CROSS_90s connect to each other (degenerate
    # but the GA can mutate this into something non-trivial)
    edges.append((0, "C", 1, "C"))
    edges.append((0, "D", 1, "D"))

    out.append((slots, edges))
>>>>>>> Stashed changes
    return out


def _emit_multi_loop(
    catalog: TrackCatalog, inventory: Dict[str, int], dims: PortPairDimensions,
) -> List[Pattern]:
<<<<<<< Updated upstream
    """Two completely disconnected R40_CURVE cycles (one left, one right)."""
    out: List[Pattern] = []
    if inventory.get("R40_CURVE", 0) < 32:
        return out
    if 32 > dims.N_max:
        return out
    slots = [(k, "R40_CURVE") for k in range(32)]
=======
    """Two completely disconnected R40 LEFT cycles."""
    out: List[Pattern] = []
    if inventory.get("R40_LEFT", 0) < 32:
        return out
    if 32 > dims.N_max:
        return out
    slots = [(k, "R40_LEFT") for k in range(32)]
>>>>>>> Stashed changes
    edges: List[Tuple[int, str, int, str]] = []
    for k in range(16):
        edges.append((k, "B", (k + 1) % 16, "A"))
    for k in range(16):
        edges.append((16 + k, "B", 16 + ((k + 1) % 16), "A"))
<<<<<<< Updated upstream
    flips = {k: (0 if k < 16 else 1) for k in range(32)}
    out.append((slots, edges, flips, {}))
    return out


def _emit_dog_bone(
    catalog: TrackCatalog, inventory: Dict[str, int], dims: PortPairDimensions,
) -> List[Pattern]:
    """Two 8-curve end-caps + symmetric straight middle: ~28 pieces, no switches.

    Both end-caps share handedness so the layout closes as a single cycle.
    Each end-cap is a 180° arc (8 × 22.5° = 180°) joined by a straight section
    of length ``n_str/2`` per side. Variants over flip ∈ {0, 1}.
    """
    out: List[Pattern] = []
    if inventory.get("R40_CURVE", 0) < 16:
        return out
    n_str_avail = inventory.get("STRAIGHT_16", 0)
    if n_str_avail < 4:
        return out
    n_per_side = max(2, n_str_avail // 2)
    total = 16 + 2 * n_per_side
    if total > dims.N_max or 2 * n_per_side > n_str_avail:
        return out

    for flip in (0, 1):
        slots: List[Tuple[int, str]] = []
        flips: Dict[int, int] = {}
        slot_idx = 0
        for _ in range(8):
            slots.append((slot_idx, "R40_CURVE"))
            flips[slot_idx] = flip
            slot_idx += 1
        for _ in range(n_per_side):
            slots.append((slot_idx, "STRAIGHT_16"))
            slot_idx += 1
        for _ in range(8):
            slots.append((slot_idx, "R40_CURVE"))
            flips[slot_idx] = flip
            slot_idx += 1
        for _ in range(n_per_side):
            slots.append((slot_idx, "STRAIGHT_16"))
            slot_idx += 1
        n = len(slots)
        edges = [(k, "B", (k + 1) % n, "A") for k in range(n)]
        out.append((slots, edges, flips, {}))
    return out


def _emit_dumbbell(
    catalog: TrackCatalog, inventory: Dict[str, int], dims: PortPairDimensions,
) -> List[Pattern]:
    """Two ovals connected by a chord with one CROSS_90 dividing it.

    Each lobe is 16 curves; the CROSS_90 sits between them and routes the
    two lobes' connecting straights through ports A↔B and C↔D.
    """
    out: List[Pattern] = []
    if inventory.get("CROSS_90", 0) < 1:
        return out
    if inventory.get("R40_CURVE", 0) < 32:
        return out
    n_str_avail = inventory.get("STRAIGHT_16", 0)
    if n_str_avail < 4:
        return out
    if 33 + 4 > dims.N_max:
        return out

    slots: List[Tuple[int, str]] = [(0, "CROSS_90")]
    flips: Dict[int, int] = {}
    for k in range(1, 17):
        slots.append((k, "R40_CURVE"))
        flips[k] = 0
    for k in range(17, 33):
        slots.append((k, "R40_CURVE"))
        flips[k] = 1
    # Two short straight chords linking each lobe back through CROSS_90's
    # opposite-side ports (4 straights total, 2 per chord).
    for k in range(33, 37):
        slots.append((k, "STRAIGHT_16"))

    edges: List[Tuple[int, str, int, str]] = []
    edges.append((0, "A", 1, "A"))
    for k in range(1, 16):
        edges.append((k, "B", k + 1, "A"))
    edges.append((16, "B", 33, "A"))
    edges.append((33, "B", 34, "A"))
    edges.append((34, "B", 0, "B"))
    edges.append((0, "C", 17, "A"))
    for k in range(17, 32):
        edges.append((k, "B", k + 1, "A"))
    edges.append((32, "B", 35, "A"))
    edges.append((35, "B", 36, "A"))
    edges.append((36, "B", 0, "D"))
    out.append((slots, edges, flips, {}))
    return out


_STACKED_OVAL_HEIGHT_STUDS: float = 96.0
"""Minimum boundary height for two R40 stadiums offset by the DC's
16-stud port spacing. Two semicircles of 40 stud radius span 80 stud,
plus the DC north-south offset rounded up = ~96 stud."""


def _emit_parallel_tracks_with_dc(
    catalog: TrackCatalog,
    inventory: Dict[str, int],
    dims: PortPairDimensions,
    boundary: BoundaryConfig,
) -> List[Pattern]:
    """Phase 7b: parallel-tracks stadium seed with a DOUBLE_CROSSOVER bridge.

    Topology (slot 0 is the DC):
      Track 1: DC.B -> R40 x 8 -> STR x m (north) -> R40 x 8 -> STR x m
              (south) -> DC.A
      Track 2: DC.D -> R40 x 8 -> STR x m (north) -> R40 x 8 -> STR x m
              (south) -> DC.C

    Each track is its own 32R40+ stadium loop closed through the DC's
    A/B (track 1) or C/D (track 2) port pair. Per the plan's Phase 7b
    spec the seed is what supplies the parallel-section context that
    Phase 7a's minimal materializer cannot detect.

    A ``PARALLEL_DC_BRIDGE`` junction descriptor is attached anchored at
    slot 0. It is functionally redundant for materialization (anchor is
    already a DC), but Phase 7c-onwards mutations key off the descriptor
    to recognise the DC-bridge structural role.

    Falls back to ``[]`` when inventory lacks DC + 32 R40 + 4 STR, or
    when the boundary can't fit two stacked stadiums.
    """
    out: List[Pattern] = []
    if (inventory.get("DOUBLE_CROSSOVER", 0) < 1
            or inventory.get("R40_CURVE", 0) < 32
            or inventory.get("STRAIGHT_16", 0) < 4):
        return out
    if boundary.height < _STACKED_OVAL_HEIGHT_STUDS:
        return out

    n_str = inventory.get("STRAIGHT_16", 0)
    m_max = _fit_oval_straights_per_section(boundary, n_str)
    if m_max < 1:
        return out

    sizes = sorted(
        {m_max, max(1, m_max * 3 // 4), max(1, m_max // 2), max(1, m_max // 4)},
        reverse=True,
    )

    for m in sizes:
        n = 33 + 4 * m
        if n > dims.N_max or 4 * m > n_str:
            continue

        slots: List[Tuple[int, str]] = [(0, "DOUBLE_CROSSOVER")]
        slot_idx = 1

        # Track 1: NE corner -> north straight -> NW corner -> south straight
        for _ in range(8):
            slots.append((slot_idx, "R40_CURVE"))
            slot_idx += 1
        for _ in range(m):
            slots.append((slot_idx, "STRAIGHT_16"))
            slot_idx += 1
        for _ in range(8):
            slots.append((slot_idx, "R40_CURVE"))
            slot_idx += 1
        for _ in range(m):
            slots.append((slot_idx, "STRAIGHT_16"))
            slot_idx += 1

        # Track 2: same shape, offset 16 stud north via DC.C/D
        for _ in range(8):
            slots.append((slot_idx, "R40_CURVE"))
            slot_idx += 1
        for _ in range(m):
            slots.append((slot_idx, "STRAIGHT_16"))
            slot_idx += 1
        for _ in range(8):
            slots.append((slot_idx, "R40_CURVE"))
            slot_idx += 1
        for _ in range(m):
            slots.append((slot_idx, "STRAIGHT_16"))
            slot_idx += 1

        edges: List[Tuple[int, str, int, str]] = []
        # Chain helper: for slots [a..b], produce sequential B->A edges.
        def _chain(s0: int, count: int) -> None:
            for k in range(count - 1):
                edges.append((s0 + k, "B", s0 + k + 1, "A"))

        # Track 1: 1..(8+m+8+m) = 1..(2m+16). Closes DC.B -> 1.A and (2m+16).B -> DC.A.
        track1_first = 1
        track1_last = 2 * m + 16
        edges.append((0, "B", track1_first, "A"))
        _chain(track1_first, track1_last - track1_first + 1)
        edges.append((track1_last, "B", 0, "A"))

        # Track 2: (2m+17)..(4m+32). Closes DC.D -> (2m+17).A and (4m+32).B -> DC.C.
        track2_first = 2 * m + 17
        track2_last = 4 * m + 32
        edges.append((0, "D", track2_first, "A"))
        _chain(track2_first, track2_last - track2_first + 1)
        edges.append((track2_last, "B", 0, "C"))

        junction = (
            1, 0, JUNCTION_KIND_PARALLEL_DC_BRIDGE, 0, 0,
        )
        out.append((slots, edges, {}, {}, [junction]))

    return out


def _emit_parallel_tracks_with_crossover(
    catalog: TrackCatalog, inventory: Dict[str, int], dims: PortPairDimensions,
) -> List[Pattern]:
    """Two parallel ovals joined by a DOUBLE_CROSSOVER. Gated on inventory.

    Silently returns empty when ``DOUBLE_CROSSOVER`` is absent — emitter
    contract is "skip emit when prerequisite parts missing" so the seed
    pool stays usable on minimal inventories.
    """
    if inventory.get("DOUBLE_CROSSOVER", 0) < 1:
        return []
    if inventory.get("R40_CURVE", 0) < 32:
        return []
    if 33 > dims.N_max:
        return []

    slots: List[Tuple[int, str]] = [(0, "DOUBLE_CROSSOVER")]
    flips: Dict[int, int] = {}
    for k in range(1, 17):
        slots.append((k, "R40_CURVE"))
        flips[k] = 0
    for k in range(17, 33):
        slots.append((k, "R40_CURVE"))
        flips[k] = 0
    edges: List[Tuple[int, str, int, str]] = []
    edges.append((0, "A", 1, "A"))
    for k in range(1, 16):
        edges.append((k, "B", k + 1, "A"))
    edges.append((16, "B", 0, "B"))
    edges.append((0, "C", 17, "A"))
    for k in range(17, 32):
        edges.append((k, "B", k + 1, "A"))
    edges.append((32, "B", 0, "D"))
    return [(slots, edges, flips, {})]


# Phase 4 (§8.2.1): the simple_oval_with_siding pattern is re-enabled as a
# heuristic seed. Phase 3's mutate_grow_branch can rewrite the branch with
# an A*-found path post-seed if the geometry doesn't close exactly, so the
# closure-residual issue that originally disabled this seed no longer
# matters — bad branches get repaired into good ones.
_HEURISTIC_EMITTERS = (
    _emit_simple_loop,
    _emit_oval,
    _emit_sequential_ring_stadium,
    _emit_asymmetric_oval_with_siding,
    _emit_figure8_stadium,
    _emit_parallel_tracks_with_dc,
    _emit_racetrack,
    _emit_figure_8,
    _emit_multi_loop,
    _emit_dense_crossing_grid,
    _emit_simple_oval_with_siding,
    _emit_dog_bone,
    _emit_dumbbell,
    _emit_parallel_tracks_with_crossover,
)

# Emitters that take an extra ``boundary`` argument (Phase 2 stadium fit, Phase
# 5b siding-reservation, Phase 6b figure-8 stadium, Phase 7b parallel-DC).
# ``_build_patterns`` dispatches to them with the boundary; everything else
# uses the legacy 3-arg signature.
_BOUNDARY_AWARE_EMITTERS = frozenset({
    _emit_sequential_ring_stadium,
    _emit_asymmetric_oval_with_siding,
    _emit_figure8_stadium,
    _emit_parallel_tracks_with_dc,
})

=======
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
)

>>>>>>> Stashed changes

# =============================================================================
# Sampling
# =============================================================================


<<<<<<< Updated upstream
SWITCH_SEED_RATIO: float = 0.05
"""Fraction of the initial population seeded with a switch-pair structural
mutation already applied. Real LEGO sidings can't form via random sampling
or local mutation alone — closure constraints kill them within ~10
generations. Pre-seeding gives the GA a starting set of switch-bearing
candidates that crossover can then propagate through the population, so
the diversity bonus has feasible diverse layouts to choose from."""


class PortPairSampling(Sampling):
    """Initial population: heuristic emitters + random chromosomes + switch-pair seeds."""
=======
class PortPairSampling(Sampling):
    """Initial population: heuristic emitters + random chromosomes."""
>>>>>>> Stashed changes

    def __init__(
        self,
        dims: PortPairDimensions,
        catalog: TrackCatalog,
        config: OptimizationConfig,
        heuristic_ratio: float = 0.30,
<<<<<<< Updated upstream
        switch_seed_ratio: float = SWITCH_SEED_RATIO,
=======
>>>>>>> Stashed changes
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.dims = dims
        self.catalog = catalog
        self.config = config
        self.heuristic_ratio = heuristic_ratio
<<<<<<< Updated upstream
        self.switch_seed_ratio = switch_seed_ratio
=======
>>>>>>> Stashed changes
        self.rng = np.random.default_rng(seed)
        self.id_to_index = catalog.id_to_index
        self.index_to_id = catalog.index_to_id
        self.patterns = self._build_patterns()

    def _build_patterns(self) -> List[Pattern]:
        patterns: List[Pattern] = []
        for emitter in _HEURISTIC_EMITTERS:
<<<<<<< Updated upstream
            if emitter in _BOUNDARY_AWARE_EMITTERS:
                patterns.extend(emitter(
                    self.catalog, self.config.inventory, self.dims,
                    self.config.boundary,
                ))
            else:
                patterns.extend(emitter(self.catalog, self.config.inventory, self.dims))
=======
            patterns.extend(emitter(self.catalog, self.config.inventory, self.dims))
>>>>>>> Stashed changes
        return patterns

    def _do(self, problem, n_samples, **kwargs) -> NDArray:
        X = np.full((n_samples, self.dims.n_var), INACTIVE, dtype=DTYPE)
        n_heuristic = (
            max(1, int(n_samples * self.heuristic_ratio)) if self.patterns else 0
        )
<<<<<<< Updated upstream
        # Switch-seeded samples: applied AFTER heuristic patterns and only when
        # the inventory actually contains both LEFT and RIGHT switches; the
        # introduce_switch_pair primitive needs at least one of each.
        if (self.config.inventory.get("R40_SWITCH_LEFT", 0) >= 1
                and self.config.inventory.get("R40_SWITCH_RIGHT", 0) >= 1):
            n_switch_seeded = max(1, int(n_samples * self.switch_seed_ratio))
        else:
            n_switch_seeded = 0

        import random as _r
        seed_rng = _r.Random(int(self.rng.integers(0, 2**31)))
=======
>>>>>>> Stashed changes

        for i in range(n_samples):
            x = create_empty_chromosome(self.dims)
            if i < n_heuristic and self.patterns:
<<<<<<< Updated upstream
                # Heuristic seeds: pre-validated closed-loop patterns.
                pattern = self.patterns[i % len(self.patterns)]
                self._populate_from_pattern(x, pattern)
            elif i < n_heuristic + n_switch_seeded:
                # Switch-pair seeds: random base chromosome + one switch pair
                # inserted by the structural mutation. May or may not be
                # feasible — the GA's job is to mutate around it.
                self._populate_random(x)
                introduce_switch_pair(
                    x, self.dims, self.catalog, self.config.inventory, rng=seed_rng,
                )
=======
                pattern = self.patterns[i % len(self.patterns)]
                self._populate_from_pattern(x, pattern)
>>>>>>> Stashed changes
            else:
                self._populate_random(x)
            self._apply_random_anchor(x)
            X[i, :] = x

        return X

    def _populate_from_pattern(self, x: NDArray, pattern: Pattern) -> None:
<<<<<<< Updated upstream
        # Pattern is a 4-tuple (slots, edges, flips, rotates) for emitters
        # that don't seed junction descriptors, or a 5-tuple with a junction
        # list appended (Phase 5b+: ``_emit_asymmetric_oval_with_siding``).
        slots, edges, flips, rotates, *rest = pattern
        junctions: List[Tuple[int, int, int, int, int]] = rest[0] if rest else []
=======
        slots, edges = pattern
>>>>>>> Stashed changes
        slot_to_piece = {s: p for s, p in slots}

        for slot_idx, piece_id in slots:
            if slot_idx >= self.dims.N_max:
                continue
            piece_index = self.id_to_index.get(piece_id)
            if piece_index is None:
                continue
            set_piece_slot(x, self.dims, slot_idx, piece_index)
<<<<<<< Updated upstream
            # Apply flip / rotate only when the piece is the matching kind;
            # for others repair would zero them anyway.
            flip = flips.get(slot_idx, 0) if _is_symmetric_piece(self.catalog, piece_id) else 0
            set_slot_flip(x, self.dims, slot_idx, flip)
            rot = rotates.get(slot_idx, 0) if _is_rotatable_piece(self.catalog, piece_id) else 0
            set_slot_rotate(x, self.dims, slot_idx, rot)
=======
>>>>>>> Stashed changes

        for k, (sa, port_a_name, sb, port_b_name) in enumerate(edges):
            if k >= self.dims.E_max:
                break
            port_a_idx = self._port_idx_for_slot(slot_to_piece, sa, port_a_name)
            port_b_idx = self._port_idx_for_slot(slot_to_piece, sb, port_b_name)
            if port_a_idx is None or port_b_idx is None:
                continue
            set_port_pair(x, self.dims, k, sa, port_a_idx, sb, port_b_idx)

<<<<<<< Updated upstream
        # Phase 5b: write junction descriptors from the pattern (if any). Only
        # the first ``J_max`` junctions fit; excess are silently dropped.
        for j_idx, (active, anchor, kind, param_a, param_b) in enumerate(junctions):
            if j_idx >= self.dims.J_max:
                break
            set_junction(
                x, self.dims, j_idx,
                active=active, anchor=anchor, kind=kind,
                param_a=param_a, param_b=param_b,
            )

=======
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
            # Random flip for symmetric pieces, 0 for others.
            if _is_symmetric_piece(self.catalog, piece_id):
                set_slot_flip(x, self.dims, k, int(self.rng.integers(0, 2)))
            else:
                set_slot_flip(x, self.dims, k, 0)
            # Random rotate for rotatable pieces, 0 for others.
            if _is_rotatable_piece(self.catalog, piece_id):
                set_slot_rotate(x, self.dims, k, int(self.rng.integers(0, 2)))
            else:
                set_slot_rotate(x, self.dims, k, 0)
=======
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
        # Phase 3+: anchor xy is a small +/-ANCHOR_OFFSET_FRACTION offset from
        # the auto-centered layout, not an absolute world position.
        b = self.config.boundary
        off_x = max(1, int(b.width * ANCHOR_OFFSET_FRACTION))
        off_y = max(1, int(b.height * ANCHOR_OFFSET_FRACTION))
        ax = int(self.rng.integers(-off_x, off_x + 1))
        ay = int(self.rng.integers(-off_y, off_y + 1))
=======
        b = self.config.boundary
        margin_x = max(1, int((b.max_x - b.min_x) * 0.1))
        margin_y = max(1, int((b.max_y - b.min_y) * 0.1))
        ax = int(self.rng.integers(int(b.min_x) + margin_x, int(b.max_x) - margin_x + 1))
        ay = int(self.rng.integers(int(b.min_y) + margin_y, int(b.max_y) - margin_y + 1))
>>>>>>> Stashed changes
        atheta = int(self.rng.integers(0, 360))
        set_anchor(x, self.dims, ax, ay, atheta)


# =============================================================================
# Crossover
# =============================================================================


class PortPairCrossover(Crossover):
<<<<<<< Updated upstream
    """One-point crossover per region (slots, pair-rows, anchor uniform).

    Phase 4 (Coupling A): does NOT touch the junction segment. Junction
    descriptors swap via the separate :class:`JunctionCrossover` operator
    so the two ablation knobs (``crossover_prob``, ``junction_crossover_prob``)
    are decoupled per Rule 26 revised.
    """
=======
    """One-point crossover per region (slots, pair-rows, anchor uniform)."""
>>>>>>> Stashed changes

    def __init__(self, dims: PortPairDimensions, prob: float = 0.9) -> None:
        super().__init__(n_parents=2, n_offsprings=2, prob=prob)
        self.dims = dims
<<<<<<< Updated upstream
        # pymoo wraps ``prob`` as a ``Real`` variable; cache the raw float so
        # downstream composites can apply gating without unwrapping.
        self._prob_value: float = float(prob)
=======
>>>>>>> Stashed changes

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
<<<<<<< Updated upstream
                c1[dims.slot_start + cut:dims.slot_end] = p2[dims.slot_start + cut:dims.slot_end]
                c2[dims.slot_start + cut:dims.slot_end] = p1[dims.slot_start + cut:dims.slot_end]

            # Flip region: independent one-point crossover. Cut at a different
            # point so flips can recombine separately from piece identities.
            if dims.N_max > 1:
                cut_f = np.random.randint(1, dims.N_max)
                c1[dims.flip_start + cut_f:dims.flip_end] = p2[dims.flip_start + cut_f:dims.flip_end]
                c2[dims.flip_start + cut_f:dims.flip_end] = p1[dims.flip_start + cut_f:dims.flip_end]

            # Rotate region: same treatment, separate cut point.
            if dims.N_max > 1:
                cut_r = np.random.randint(1, dims.N_max)
                c1[dims.rotate_start + cut_r:dims.rotate_end] = p2[dims.rotate_start + cut_r:dims.rotate_end]
                c2[dims.rotate_start + cut_r:dims.rotate_end] = p1[dims.rotate_start + cut_r:dims.rotate_end]
=======
                c1[cut:dims.N_max] = p2[cut:dims.N_max]
                c2[cut:dims.N_max] = p1[cut:dims.N_max]
>>>>>>> Stashed changes

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


<<<<<<< Updated upstream
class JunctionCrossover(Crossover):
    """Phase 4 junction-segment crossover with semantic guard (Rule 4).

    For each junction slot ``j``, atomically swaps the 5-gene descriptor
    between offspring -- but only when the receiver chromosome's slot at
    ``j``'s anchor is currently active. If the receiver couldn't host the
    junction, the descriptor is *not* propagated; instead it lands inert
    (active=0) so :class:`PortPairRepairPipeline`'s junction-validity step
    has nothing to silently kill (which would erode heritability and
    replicate V2's existing crossover-destroys-feasibility failure mode).

    Outer ``prob`` is the standard pymoo per-mating gate (0.0 in Phase 4
    runs; ablated in Phase 5a+ per Rule 26 revised). Inside ``_do`` every
    junction is considered for swap deterministically -- the per-mating
    randomness already comes from the outer gate.
    """

    def __init__(
        self,
        dims: PortPairDimensions,
        catalog: TrackCatalog,
        prob: float = 0.0,
    ) -> None:
        super().__init__(n_parents=2, n_offsprings=2, prob=prob)
        self.dims = dims
        self.catalog = catalog
        self._prob_value: float = float(prob)

    def _do(self, problem, X, **kwargs) -> NDArray:
        _, n_matings, n_var = X.shape
        Y = np.empty((self.n_offsprings, n_matings, n_var), dtype=X.dtype)
        dims = self.dims
        if dims.J_max == 0:
            Y[0] = X[0]
            Y[1] = X[1]
            return Y

        for k in range(n_matings):
            p1, p2 = X[0, k], X[1, k]
            c1, c2 = p1.copy(), p2.copy()
            for j in range(dims.J_max):
                self._swap_junction_if_safe(c1, c2, p1, p2, j)
            Y[0, k] = c1
            Y[1, k] = c2
        return Y

    def _swap_junction_if_safe(
        self,
        c1: NDArray,
        c2: NDArray,
        p1: NDArray,
        p2: NDArray,
        j: int,
    ) -> None:
        dims = self.dims
        base = dims.junc_start + j * JUNCTION_GENES
        end = base + JUNCTION_GENES

        j1 = get_junction(p1, dims, j)
        j2 = get_junction(p2, dims, j)

        c1_can_receive_p2 = self._is_branch_capable_slot(c1, j2[1])
        c2_can_receive_p1 = self._is_branch_capable_slot(c2, j1[1])

        if c1_can_receive_p2 and c2_can_receive_p1:
            c1[base:end] = p2[base:end]
            c2[base:end] = p1[base:end]
            return

        # Receiver can't host one or both descriptors -- copy genes but
        # force ``active=0`` on the side that's invalid so repair has
        # nothing to silently kill.
        if not c1_can_receive_p2 and j2[0] == 1:
            c1[base:end] = p2[base:end]
            c1[base + JUNC_ACTIVE_OFFSET] = 0
        if not c2_can_receive_p1 and j1[0] == 1:
            c2[base:end] = p1[base:end]
            c2[base + JUNC_ACTIVE_OFFSET] = 0

    def _is_branch_capable_slot(self, x: NDArray, slot_idx: int) -> bool:
        """Conservative variant per Phase 4 spec: receiver hosts the descriptor
        iff its slot at ``slot_idx`` is active (any piece). Phase 5a+ tightens
        the predicate to "piece has a branch port"."""
        dims = self.dims
        if not 0 <= slot_idx < dims.N_max:
            return False
        return int(x[dims.slot_start + slot_idx]) != INACTIVE


class PortPairAndJunctionCrossover(Crossover):
    """Composite: port-pair crossover then junction crossover. Each component
    keeps its own ``prob`` so the two ablation knobs (``crossover_prob`` and
    ``junction_crossover_prob``, Coupling A) are decoupled. NSGA2 sees this as
    a single Crossover at ``prob=1.0`` while the per-mating gating is delegated
    to the components.
    """

    def __init__(
        self,
        port_cx: "PortPairCrossover",
        junc_cx: "JunctionCrossover",
    ) -> None:
        super().__init__(
            n_parents=port_cx.n_parents,
            n_offsprings=port_cx.n_offsprings,
            prob=1.0,
        )
        self.port_cx = port_cx
        self.junc_cx = junc_cx

    def _do(self, problem, X, **kwargs) -> NDArray:
        Y = self._gated_apply(self.port_cx, problem, X, **kwargs)
        Y = self._gated_apply(self.junc_cx, problem, Y, **kwargs)
        return Y

    @staticmethod
    def _gated_apply(cx: Crossover, problem, X: NDArray, **kwargs) -> NDArray:
        prob = getattr(cx, "_prob_value", 1.0)
        if prob <= 0.0:
            return X.copy()
        Y = cx._do(problem, X, **kwargs)
        if prob >= 1.0:
            return Y
        n_matings = X.shape[1]
        keep_offspring = np.random.random(n_matings) < prob
        for i in range(n_matings):
            if not keep_offspring[i]:
                Y[:, i, :] = X[:, i, :]
        return Y


=======
>>>>>>> Stashed changes
# =============================================================================
# Mutation
# =============================================================================


class PortPairMutation(Mutation):
<<<<<<< Updated upstream
    """Weighted sub-operators dispatched via cached CDF + ``searchsorted``.

    Sub-operator order (must match ``OP_WEIGHTS`` keys / ``_ops_tuple``):

    - ``piece_type``, ``activate``, ``deactivate``, ``add_edge``,
      ``remove_edge``, ``rewire_edge``, ``perturb_anchor``,
      ``toggle_flip``, ``toggle_rotate``,
      ``introduce_switch_pair``, ``grow_branch`` (Phase 3),
    - ``branch_extend``, ``branch_shrink``, ``swap_switch_hand``,
      ``reverse_switch_pairing``, ``split_loop_with_crossing``,
      ``closure_repair_lamarckian`` (Phase 4).

    ``introduce_crossing`` was promoted to the repair pipeline.
    """

    OP_WEIGHTS = {
        "piece_type":              0.10,
        "activate":                0.07,
        "deactivate":              0.06,
        "add_edge":                0.10,
        "remove_edge":             0.06,
        "rewire_edge":             0.08,
        "perturb_anchor":          0.05,
        "toggle_flip":             0.05,
        "toggle_rotate":           0.05,
        "introduce_switch_pair":   0.08,
        "grow_branch":             0.08,
        "branch_extend":           0.04,
        "branch_shrink":           0.04,
        "swap_switch_hand":        0.04,
        "reverse_switch_pairing":  0.04,
        "split_loop_with_crossing": 0.03,
        # Phase 5c (Rule 29 revised): four junction sub-ops collapsed under
        # a single ``tune_passing_siding`` ALNS slot. Cold-start weight =
        # ``max(other weights) = 0.10`` (Rule 29a -- supersedes Rule 14;
        # mean-init starves new ops, optimistic init mirrors MAB UCB1).
        "tune_passing_siding":     0.10,
        # ``closure_repair_lamarckian`` removed in Phase 1: the new
        # ``CycleClosureRepair`` (in repair.py) is the single source of
        # truth for cycle-angle repair. _build_cdf renormalizes the
        # remaining weights, so the dispatch distribution stays valid.
    }
=======
    """Weighted sub-operators: piece-type, activate, deactivate, add edge,
    remove edge, rewire edge, perturb anchor, introduce_crossing."""

    OP_WEIGHTS = np.array([0.20, 0.15, 0.10, 0.20, 0.10, 0.15, 0.10])
    # introduce_crossing was promoted from a mutation sub-op to the repair
    # pipeline (PortPairRepairPipeline), where it runs on every individual
    # rather than probabilistically — mirrors V1's repair-injection design.
>>>>>>> Stashed changes

    def __init__(
        self,
        dims: PortPairDimensions,
        catalog: TrackCatalog,
        config: OptimizationConfig,
        prob: float = 0.3,
<<<<<<< Updated upstream
        seed: Optional[int] = None,
=======
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
        self.rng = np.random.default_rng(seed)
        self._build_cdf()

    def _build_cdf(self) -> None:
        """Cache the cumulative distribution over ``OP_WEIGHTS`` for O(log N)
        operator dispatch via ``np.searchsorted``. Recompute when ALNS
        reweights (Phase 5)."""
        weights = np.array(list(self.OP_WEIGHTS.values()), dtype=np.float64)
        self._cdf = np.cumsum(weights / weights.sum())
        self._op_names = tuple(self.OP_WEIGHTS.keys())
        # Per-individual sub-operator picks for the most recent _do call —
        # ALNSCallback reads this to attribute reward back to operators.
        self._last_op_indices: tuple = ()

    def _ops_tuple(self) -> tuple:
        """Return the dispatch table aligned with ``self._op_names``."""
        return (
=======

    def _do(self, problem, X, **kwargs) -> NDArray:
        ops = (
>>>>>>> Stashed changes
            self._mutate_piece_type,
            self._activate_slot,
            self._deactivate_slot,
            self._add_edge,
            self._remove_edge,
            self._rewire_edge,
            self._perturb_anchor,
<<<<<<< Updated upstream
            self._toggle_flip,
            self._toggle_rotate,
            self._introduce_switch_pair,
            self._mutate_grow_branch,
            self._mutate_branch_extend,
            self._mutate_branch_shrink,
            self._mutate_swap_switch_hand,
            self._mutate_reverse_switch_pairing,
            self._mutate_split_loop_with_crossing,
            self._mutate_tune_passing_siding,
        )

    def _pick_op_idx(self) -> int:
        """O(log N) operator pick via cached CDF."""
        u = float(self.rng.random())
        return int(np.searchsorted(self._cdf, u, side="right"))

    def _do(self, problem, X, **kwargs) -> NDArray:
        ops = self._ops_tuple()
        picks: list = []
        for i in range(len(X)):
            idx = self._pick_op_idx()
            picks.append(idx)
            ops[idx](X[i])
        # ALNS reads this each generation to credit operators with the
        # offspring CV they produced (Phase 5).
        self._last_op_indices = tuple(picks)
=======
        )
        for i in range(len(X)):
            op_idx = int(np.random.choice(len(ops), p=self.OP_WEIGHTS))
            ops[op_idx](X[i])
>>>>>>> Stashed changes
        return X

    # ------------------------------------------------------------------
    # Sub-operators
    # ------------------------------------------------------------------

    def _mutate_piece_type(self, x: NDArray) -> None:
<<<<<<< Updated upstream
        switch_slots = self._switch_slot_indices(x)
        active = [
            (s, p) for s, p in iter_active_slots(x, self.dims)
            if s not in switch_slots
        ]
=======
        active = list(iter_active_slots(x, self.dims))
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
        self._reset_flip_for_piece(x, slot_idx, new_type)
=======
>>>>>>> Stashed changes

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
<<<<<<< Updated upstream
        new_type = available[np.random.randint(len(available))]
        set_piece_slot(x, self.dims, slot_idx, new_type)
        self._reset_flip_for_piece(x, slot_idx, new_type)

    def _reset_flip_for_piece(self, x: NDArray, slot_idx: int, piece_index: int) -> None:
        """Set a fresh random flip/rotate when the new piece supports them, else 0.

        Called whenever a slot's piece identity changes — the previous flip
        and rotate bits were bound to the old piece's symmetry / rotatability
        and need to be renegotiated for the new piece.
        """
        piece_id = self.index_to_id.get(piece_index)
        if piece_id and _is_symmetric_piece(self.catalog, piece_id):
            set_slot_flip(x, self.dims, slot_idx, int(np.random.randint(0, 2)))
        else:
            set_slot_flip(x, self.dims, slot_idx, 0)
        if piece_id and _is_rotatable_piece(self.catalog, piece_id):
            set_slot_rotate(x, self.dims, slot_idx, int(np.random.randint(0, 2)))
        else:
            set_slot_rotate(x, self.dims, slot_idx, 0)

    def _toggle_flip(self, x: NDArray) -> None:
        """Flip-bit toggle on a random symmetric-piece slot."""
        symmetric_slots: List[int] = []
        for slot_idx, piece_index in iter_active_slots(x, self.dims):
            piece_id = self.index_to_id.get(piece_index)
            if piece_id and _is_symmetric_piece(self.catalog, piece_id):
                symmetric_slots.append(slot_idx)
        if not symmetric_slots:
            return
        slot_idx = symmetric_slots[np.random.randint(len(symmetric_slots))]
        current = get_slot_flip(x, self.dims, slot_idx)
        set_slot_flip(x, self.dims, slot_idx, 1 - current)

    def _toggle_rotate(self, x: NDArray) -> None:
        """Rotate-bit toggle on a random rotatable-piece slot.

        Switch slots are excluded — flipping IN↔OUT on a placed switch
        breaks the carefully wired branch geometry (port C lands on the
        opposite side of the mainline).
        """
        switch_slots = self._switch_slot_indices(x)
        rotatable_slots = [
            slot_idx
            for slot_idx, piece_index in iter_active_slots(x, self.dims)
            if slot_idx not in switch_slots
            and (piece_id := self.index_to_id.get(piece_index)) is not None
            and _is_rotatable_piece(self.catalog, piece_id)
        ]
        if not rotatable_slots:
            return
        slot_idx = rotatable_slots[np.random.randint(len(rotatable_slots))]
        current = get_slot_rotate(x, self.dims, slot_idx)
        set_slot_rotate(x, self.dims, slot_idx, 1 - current)

    def _introduce_switch_pair(self, x: NDArray) -> None:
        """Insert a paired IN+OUT switch + A*-grown closing branch.

        Switches are always introduced as a pair — a single switch has no
        useful purpose in a layout. Phase 3 replaced the hardcoded 2-curve
        branch with an angular-budget A* search; on A* failure the operator
        rolls back the switch placement so x is unchanged.
        """
        import random as _r
        introduce_switch_pair(
            x, self.dims, self.catalog, self.config.inventory,
            rng=_r.Random(int(np.random.randint(0, 2**31))),
        )

    def _mutate_grow_branch(self, x: NDArray) -> None:
        """Phase 3 sub-operator: close incomplete switches' port C via A*.

        Picks a pair of switches with port C unpaired on the same through
        cycle and runs angular-budget A* over remaining inventory. On
        success, installs the closing branch pieces + edges; on failure,
        x is unchanged (no rollback needed because no chromosome state
        is modified before the A* call).
        """
        mutate_grow_branch(
            x, self.dims, self.catalog, self.decoder_config, self.config.inventory,
            rng=self._py_rng(),
        )

    def _py_rng(self):
        """Spawn a python ``random.Random`` seeded from ``self.rng`` for the
        ``mutate_*`` family (which expects ``random.Random`` not numpy)."""
        import random as _r
        return _r.Random(int(self.rng.integers(0, 2**31)))

    def _mutate_branch_extend(self, x: NDArray) -> None:
        """Phase 4: insert a STRAIGHT_16 in-line on a branch cycle edge."""
        mutate_branch_extend(
            x, self.dims, self.catalog, self.decoder_config, self.config.inventory,
            rng=self._py_rng(),
        )

    def _mutate_branch_shrink(self, x: NDArray) -> None:
        """Phase 4: drop a STRAIGHT_16 from a branch cycle and splice."""
        mutate_branch_shrink(
            x, self.dims, self.catalog, self.decoder_config, self.config.inventory,
            rng=self._py_rng(),
        )

    def _mutate_swap_switch_hand(self, x: NDArray) -> None:
        """Phase 4: toggle LEFT↔RIGHT on a switch + BFS-flip branch curves."""
        mutate_swap_switch_hand(
            x, self.dims, self.catalog, self.decoder_config, self.config.inventory,
            rng=self._py_rng(),
        )

    def _mutate_reverse_switch_pairing(self, x: NDArray) -> None:
        """Phase 4: swap a switch's mainline A↔B endpoints; re-grow branch."""
        mutate_reverse_switch_pairing(
            x, self.dims, self.catalog, self.decoder_config, self.config.inventory,
            rng=self._py_rng(),
        )

    def _mutate_split_loop_with_crossing(self, x: NDArray) -> None:
        """Phase 4: insert CROSS_90 between two non-adjacent perpendicular edges."""
        mutate_split_loop_with_crossing(
            x, self.dims, self.catalog, self.decoder_config, self.config.inventory,
            rng=self._py_rng(),
        )

    def _deactivate_slot(self, x: NDArray) -> None:
        switch_slots = self._switch_slot_indices(x)
        active = [
            (s, p) for s, p in iter_active_slots(x, self.dims)
            if s not in switch_slots
        ]
=======
        set_piece_slot(x, self.dims, slot_idx, available[np.random.randint(len(available))])

    def _deactivate_slot(self, x: NDArray) -> None:
        active = list(iter_active_slots(x, self.dims))
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
        switch_slots = self._switch_slot_indices(x)
        active_rows = [
            k for k, sa, _pa, sb, _pb in iter_active_pairs(x, self.dims)
            if sa not in switch_slots and sb not in switch_slots
        ]
=======
        active_rows = [k for k, *_ in iter_active_pairs(x, self.dims)]
>>>>>>> Stashed changes
        if not active_rows:
            return
        row = active_rows[np.random.randint(len(active_rows))]
        clear_port_pair(x, self.dims, row)

    def _rewire_edge(self, x: NDArray) -> None:
<<<<<<< Updated upstream
        switch_slots = self._switch_slot_indices(x)
        active_rows = [
            (k, sa, pa, sb, pb)
            for k, sa, pa, sb, pb in iter_active_pairs(x, self.dims)
            if sa not in switch_slots and sb not in switch_slots
        ]
        if not active_rows:
            return
        row, sa, pa, sb, pb = active_rows[np.random.randint(len(active_rows))]
        loose = [(s, p) for s, p in self._find_loose_ports(x) if s not in switch_slots]
=======
        active_rows = [(k, sa, pa, sb, pb) for k, sa, pa, sb, pb
                       in iter_active_pairs(x, self.dims)]
        if not active_rows:
            return
        row, sa, pa, sb, pb = active_rows[np.random.randint(len(active_rows))]
        loose = self._find_loose_ports(x)
>>>>>>> Stashed changes
        if not loose:
            return
        new_endpoint = loose[np.random.randint(len(loose))]
        # Replace one endpoint at random
        if np.random.random() < 0.5:
            set_port_pair(x, self.dims, row, new_endpoint[0], new_endpoint[1], sb, pb)
        else:
            set_port_pair(x, self.dims, row, sa, pa, new_endpoint[0], new_endpoint[1])

    def _perturb_anchor(self, x: NDArray) -> None:
<<<<<<< Updated upstream
        # Phase 3+: clip to the +/-ANCHOR_OFFSET_FRACTION offset window, NOT
        # the full boundary -- xy is now an offset from the auto-centered layout.
        b = self.config.boundary
        off_x = max(1, int(b.width * ANCHOR_OFFSET_FRACTION))
        off_y = max(1, int(b.height * ANCHOR_OFFSET_FRACTION))
        delta_x = np.random.randint(-5, 6)
        delta_y = np.random.randint(-5, 6)
        delta_theta = np.random.choice([-22, -11, 11, 22])  # degrees
        ax = int(np.clip(x[self.dims.anchor_start] + delta_x, -off_x, off_x))
        ay = int(np.clip(x[self.dims.anchor_start + 1] + delta_y, -off_y, off_y))
=======
        b = self.config.boundary
        delta_x = np.random.randint(-5, 6)
        delta_y = np.random.randint(-5, 6)
        delta_theta = np.random.choice([-22, -11, 11, 22])  # degrees
        ax = int(np.clip(x[self.dims.anchor_start] + delta_x, b.min_x, b.max_x))
        ay = int(np.clip(x[self.dims.anchor_start + 1] + delta_y, b.min_y, b.max_y))
>>>>>>> Stashed changes
        atheta = int((x[self.dims.anchor_start + 2] + delta_theta) % 360)
        set_anchor(x, self.dims, ax, ay, atheta)

    # ------------------------------------------------------------------
<<<<<<< Updated upstream
    # Phase 5c: junction sub-operators
    # ------------------------------------------------------------------
    #
    # Four operators that tune existing passing-siding junction descriptors.
    # The decoder (Phase 5a) materializes them; without these mutations the
    # GA has no way to relocate or resize a siding once it's seeded by the
    # Phase 5b emitter. All four are dispatched under one ALNS slot
    # (``tune_passing_siding``) per Rule 29 revised so the pool stays small.

    def _mutate_junction_toggle_active(self, x: NDArray) -> None:
        """Flip the ``active`` bit on a random junction descriptor."""
        if self.dims.J_max == 0:
            return
        j = int(self.rng.integers(self.dims.J_max))
        base = self.dims.junc_start + j * JUNCTION_GENES + JUNC_ACTIVE_OFFSET
        x[base] = 0 if int(x[base]) == 1 else 1

    def _mutate_junction_reposition(self, x: NDArray) -> None:
        """Shift a random junction's anchor by +/- 1..5 then snap to the
        nearest active slot (Rule 9: preserve locality, no random clamping)."""
        if self.dims.J_max == 0:
            return
        active_slots = [
            s for s in range(self.dims.N_max)
            if get_piece_slot(x, self.dims, s) != INACTIVE
        ]
        if not active_slots:
            return
        j = int(self.rng.integers(self.dims.J_max))
        base = self.dims.junc_start + j * JUNCTION_GENES + JUNC_ANCHOR_OFFSET
        cur = int(x[base])
        sign = 1 if self.rng.random() < 0.5 else -1
        shift = int(self.rng.integers(1, 6)) * sign
        target = max(0, min(self.dims.N_max - 1, cur + shift))
        nearest = min(active_slots, key=lambda s: (abs(s - target), s))
        x[base] = nearest

    def _mutate_junction_swap_handedness(self, x: NDArray) -> None:
        """Flip ``param_b`` between LEFT (0) and RIGHT (1) variants."""
        if self.dims.J_max == 0:
            return
        j = int(self.rng.integers(self.dims.J_max))
        base = self.dims.junc_start + j * JUNCTION_GENES + JUNC_PARAM_B_OFFSET
        cur = int(x[base])
        x[base] = 1 if cur == 0 else 0

    def _mutate_junction_adjust_straights(self, x: NDArray) -> None:
        """Bump ``param_a`` (n_branch_straights) by +/- 1..3 and clamp to
        ``min(JUNCTION_PARAM_MAX, inventory[STRAIGHT_16])``."""
        if self.dims.J_max == 0:
            return
        j = int(self.rng.integers(self.dims.J_max))
        base = self.dims.junc_start + j * JUNCTION_GENES + JUNC_PARAM_A_OFFSET
        cur = int(x[base])
        sign = 1 if self.rng.random() < 0.5 else -1
        delta = int(self.rng.integers(1, 4)) * sign
        cap = min(
            JUNCTION_PARAM_MAX,
            int(self.config.inventory.get("STRAIGHT_16", 0)),
        )
        x[base] = max(0, min(cap, cur + delta))

    def _mutate_tune_passing_siding(self, x: NDArray) -> None:
        """Meta-op (Rule 29 revised): uniformly dispatches to one of four
        junction sub-ops. Looking sub-ops up by name (not bound reference)
        keeps the dispatch test-friendly: ALNS reward attribution stays at
        the meta-op slot but instrumentation can monkey-patch sub-ops."""
        sub_op_names = (
            "_mutate_junction_toggle_active",
            "_mutate_junction_reposition",
            "_mutate_junction_swap_handedness",
            "_mutate_junction_adjust_straights",
        )
        chosen = sub_op_names[int(self.rng.integers(len(sub_op_names)))]
        getattr(self, chosen)(x)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _switch_slot_indices(self, x: NDArray) -> set:
        """Return slot indices currently holding a switch piece.

        Switch slots are protected from piece-changing, deactivation,
        rotate-bit, edge-removing and edge-rewiring sub-operators. The only
        way a switch can leave a chromosome is via a structural deletion
        operator (not currently implemented) or via crossover bringing in a
        switchless parent's region.
        """
        left_idx = self.id_to_index.get("R40_SWITCH_LEFT")
        right_idx = self.id_to_index.get("R40_SWITCH_RIGHT")
        return {
            slot for slot, piece_idx in iter_active_slots(x, self.dims)
            if piece_idx == left_idx or piece_idx == right_idx
        }

=======
    # Helpers
    # ------------------------------------------------------------------

>>>>>>> Stashed changes
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
