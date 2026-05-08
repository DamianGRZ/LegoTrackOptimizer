"""Hand-built chromosomes proving the encoding/decoder/renderer pipeline.

Five layouts, none from the GA — each constructed by direct chromosome
manipulation to demonstrate that the port-pair encoding can represent the
target topology, the decoder can compute valid SE(2) poses for it, and the
JSON serializer + canvas renderer can display it. If the GA later struggles
to evolve any of these topologies, that's a *search* problem, not an
*encoding* problem — these layouts prove the substrate works.

The five layouts:

- ``circle``           — 16 R40 curves; pure single cycle.
- ``oval``             — 8 R40 + 4 S16 + 8 R40 + 4 S16; mixed straights/curves.
- ``siding``           — passing siding: oval mainline with branch through
                         2 R40 curves between IN/OUT switches.
- ``figure8_cross90``  — two cycles meeting at one CROSS_90.
- ``double_crossover`` — two parallel cycles linked by one DOUBLE_CROSSOVER.
"""

from __future__ import annotations

from typing import Any, Dict, List

from numpy.typing import NDArray

from ..catalog import TrackCatalog
from ..config import OptimizationConfig
from ..decoder import DecoderConfig, decode_chromosome
from ..encoding import (
    PortPairDimensions,
    create_empty_chromosome,
    set_anchor,
    set_piece_slot,
    set_port_pair,
    set_slot_flip,
    set_slot_rotate,
)
from .serialize import port_graph_to_json


SHOWCASE_DIMS = PortPairDimensions(N_max=64, E_max=64)
"""Generous slot+edge capacity that fits every showcase topology."""


# Catalog port indices follow tuple(spec.ports) order: A=0, B=1, C=2, D=3.
PORT_A, PORT_B, PORT_C, PORT_D = 0, 1, 2, 3


# =============================================================================
# Builders — each returns a fully-populated chromosome
# =============================================================================


def _chain_edges(slots: List[int], chromosome: NDArray, *, edge_row: int = 0) -> int:
    """Wire ``slots[i].B → slots[(i+1) % n].A`` for every i; return next free row."""
    n = len(slots)
    for i in range(n):
        set_port_pair(
            chromosome, SHOWCASE_DIMS, edge_row + i,
            slots[i], PORT_B, slots[(i + 1) % n], PORT_A,
        )
    return edge_row + n


def _build_circle(catalog: TrackCatalog) -> NDArray:
    x = create_empty_chromosome(SHOWCASE_DIMS)
    curve = catalog.id_to_index["R40_CURVE"]
    slots = list(range(16))
    for slot in slots:
        set_piece_slot(x, SHOWCASE_DIMS, slot, curve)
        set_slot_flip(x, SHOWCASE_DIMS, slot, 0)
    _chain_edges(slots, x)
    set_anchor(x, SHOWCASE_DIMS, 0, 0, 0)
    return x


def _build_oval(catalog: TrackCatalog) -> NDArray:
    x = create_empty_chromosome(SHOWCASE_DIMS)
    curve = catalog.id_to_index["R40_CURVE"]
    straight = catalog.id_to_index["STRAIGHT_16"]
    pieces = (
        [(slot, curve) for slot in range(0, 8)]
        + [(slot, straight) for slot in range(8, 12)]
        + [(slot, curve) for slot in range(12, 20)]
        + [(slot, straight) for slot in range(20, 24)]
    )
    for slot, piece_idx in pieces:
        set_piece_slot(x, SHOWCASE_DIMS, slot, piece_idx)
        set_slot_flip(x, SHOWCASE_DIMS, slot, 0)
    _chain_edges([s for s, _ in pieces], x)
    set_anchor(x, SHOWCASE_DIMS, 0, 0, 0)
    return x


def _build_siding(catalog: TrackCatalog) -> NDArray:
    """Monty's-Trains-style passing siding: parallel branch OUTSIDE the oval.

    Both switches RIGHT canonical (rotate=0). After the first half-circle
    the mainline runs westward at the top (theta=π in world), so both
    RIGHT switches sit at theta=π. RIGHT switch's diverging port C is on
    the RIGHT of train motion — when train moves west, RIGHT-of-motion is
    +y world (north), which is OUTSIDE the oval. Branch runs parallel
    above the top mainline.

    Per Monty's Trains Part 1: "one curved track per switch (this allows
    our diverging route to line up parallel to the main route)". The
    branch is exactly that: 1 R40 curve at each end + n_branch_straights
    of S16 in between.

    Mainline: 8 R40 → IN (RIGHT, rotate=0) → n_a S16 → OUT (RIGHT, rotate=0)
              → 8 R40 → (n_a + 4) S16 → close
    Branch:   IN.C → R40(flip=0, left) → n_a S16 → R40(flip=1, right) → OUT.C
              (left + right curves cancel for net 0 angle change between
              IN.C and OUT.C, both facing the same direction)

    Closure: 6.25 stud — the inherent LEGO 9V passing-siding y-residual
    that physical track flex absorbs and Faza 2 branch_closure_tolerance
    accepts.
    """
    x = create_empty_chromosome(SHOWCASE_DIMS)
    curve = catalog.id_to_index["R40_CURVE"]
    straight = catalog.id_to_index["STRAIGHT_16"]
    sw_right = catalog.id_to_index["R40_SWITCH_RIGHT"]

    n_section_a = 3
    n_branch_straights = n_section_a
    n_section_b = n_section_a + 4

    mainline: List[int] = []
    slot = 0
    # First half-circle (8 R40 left curves → mainline goes counterclockwise)
    for _ in range(8):
        set_piece_slot(x, SHOWCASE_DIMS, slot, curve)
        mainline.append(slot)
        slot += 1
    # IN switch (RIGHT canonical). Mainline at top of oval flows west, so
    # train motion = -x. RIGHT switch's frog (port C) is on RIGHT of train
    # motion = +y world (north of mainline, OUTSIDE the oval).
    in_switch = slot
    set_piece_slot(x, SHOWCASE_DIMS, slot, sw_right)
    set_slot_rotate(x, SHOWCASE_DIMS, slot, 0)
    mainline.append(slot)
    slot += 1
    # Section A: between switches
    for _ in range(n_section_a):
        set_piece_slot(x, SHOWCASE_DIMS, slot, straight)
        mainline.append(slot)
        slot += 1
    # OUT switch (RIGHT canonical, same as IN). Both RIGHT canonical at
    # theta=π put their port C's on +y world side → branch parallel
    # OUTSIDE the oval (above the top mainline).
    out_switch = slot
    set_piece_slot(x, SHOWCASE_DIMS, slot, sw_right)
    set_slot_rotate(x, SHOWCASE_DIMS, slot, 0)
    mainline.append(slot)
    slot += 1
    # Second half-circle
    for _ in range(8):
        set_piece_slot(x, SHOWCASE_DIMS, slot, curve)
        mainline.append(slot)
        slot += 1
    # Section B: closing straights
    for _ in range(n_section_b):
        set_piece_slot(x, SHOWCASE_DIMS, slot, straight)
        mainline.append(slot)
        slot += 1
    # Branch: approach_curve flip=0 (left +π/8) + n straights + return_curve
    # flip=1 (right -π/8). Net angle change = 0, matching IN.C and OUT.C
    # both facing the same direction. Per Monty's Part 1: 1 curve per
    # switch lines the branch parallel to mainline.
    branch = [slot]
    set_piece_slot(x, SHOWCASE_DIMS, slot, curve)
    set_slot_flip(x, SHOWCASE_DIMS, slot, 0)
    slot += 1
    for _ in range(n_branch_straights):
        set_piece_slot(x, SHOWCASE_DIMS, slot, straight)
        branch.append(slot)
        slot += 1
    set_piece_slot(x, SHOWCASE_DIMS, slot, curve)
    set_slot_flip(x, SHOWCASE_DIMS, slot, 1)
    branch.append(slot)

    # Mainline cycle
    edge_row = _chain_edges(mainline, x)
    # Branch wiring: IN.C → branch[0].A, then chain B→A through branch,
    # then branch[-1].B → OUT.C
    set_port_pair(x, SHOWCASE_DIMS, edge_row, in_switch, PORT_C, branch[0], PORT_A)
    edge_row += 1
    for i in range(len(branch) - 1):
        set_port_pair(x, SHOWCASE_DIMS, edge_row, branch[i], PORT_B, branch[i + 1], PORT_A)
        edge_row += 1
    set_port_pair(x, SHOWCASE_DIMS, edge_row, branch[-1], PORT_B, out_switch, PORT_C)

    set_anchor(x, SHOWCASE_DIMS, 0, 0, 0)
    return x


def _build_figure8_cross90(catalog: TrackCatalog) -> NDArray:
    """Two cycles sharing one CROSS_90.

    Lobe A: 16 R40 left-curves traversed via cross A-B (chord, 0° rotation).
    Lobe B:  8 R40 left-curves traversed via cross C-D (180° rotation).

    Neither lobe closes perfectly with default R40 + CROSS_90 geometry —
    the 16-curve circle returns to its start but the cross's 16-stud chord
    leaves a position residual; the 8-curve half-circle paired with the
    180° C-D route similarly has a small offset. Residual is captured in
    ``stats.max_closure_pos_studs``; the topology (cycles=2, components=1,
    no loose ports) is what proves the encoding works.
    """
    x = create_empty_chromosome(SHOWCASE_DIMS)
    curve = catalog.id_to_index["R40_CURVE"]
    cross = catalog.id_to_index["CROSS_90"]

    set_piece_slot(x, SHOWCASE_DIMS, 0, cross)
    lobe_a = list(range(1, 17))    # 16 curves for full-circle lobe
    lobe_b = list(range(17, 25))   # 8 curves for half-circle lobe
    for slot in lobe_a + lobe_b:
        set_piece_slot(x, SHOWCASE_DIMS, slot, curve)
        set_slot_flip(x, SHOWCASE_DIMS, slot, 0)

    edge_row = 0
    # Lobe A
    for i in range(len(lobe_a) - 1):
        set_port_pair(x, SHOWCASE_DIMS, edge_row, lobe_a[i], PORT_B, lobe_a[i + 1], PORT_A)
        edge_row += 1
    set_port_pair(x, SHOWCASE_DIMS, edge_row, lobe_a[-1], PORT_B, 0, PORT_A)
    edge_row += 1
    set_port_pair(x, SHOWCASE_DIMS, edge_row, 0, PORT_B, lobe_a[0], PORT_A)
    edge_row += 1
    # Lobe B
    for i in range(len(lobe_b) - 1):
        set_port_pair(x, SHOWCASE_DIMS, edge_row, lobe_b[i], PORT_B, lobe_b[i + 1], PORT_A)
        edge_row += 1
    set_port_pair(x, SHOWCASE_DIMS, edge_row, lobe_b[-1], PORT_B, 0, PORT_D)
    edge_row += 1
    set_port_pair(x, SHOWCASE_DIMS, edge_row, 0, PORT_C, lobe_b[0], PORT_A)

    set_anchor(x, SHOWCASE_DIMS, 0, 0, 0)
    return x


def _build_double_crossover(catalog: TrackCatalog) -> NDArray:
    """Two parallel oval tracks linked by one DOUBLE_CROSSOVER.

    Each track is an oval: 8 R40 + DC + 8 R40 + 3 STRAIGHT_16. The DC body
    is 48 studs long, exactly matching 3× STRAIGHT_16 on the opposite side
    so the oval closes. Track 1 uses DC.A-B; track 2 uses DC.C-D.
    """
    x = create_empty_chromosome(SHOWCASE_DIMS)
    curve = catalog.id_to_index["R40_CURVE"]
    straight = catalog.id_to_index["STRAIGHT_16"]
    dc = catalog.id_to_index["DOUBLE_CROSSOVER"]

    set_piece_slot(x, SHOWCASE_DIMS, 0, dc)

    # Track 1: slots 1..8 (8 curves), DC, 9..16 (8 curves), 17..19 (3 straights)
    track1_curves_first = list(range(1, 9))
    track1_curves_second = list(range(9, 17))
    track1_straights = list(range(17, 20))
    # Track 2: slots 20..27 + DC + 28..35 + 36..38
    track2_curves_first = list(range(20, 28))
    track2_curves_second = list(range(28, 36))
    track2_straights = list(range(36, 39))

    for slot in track1_curves_first + track1_curves_second:
        set_piece_slot(x, SHOWCASE_DIMS, slot, curve)
        set_slot_flip(x, SHOWCASE_DIMS, slot, 0)
    for slot in track1_straights:
        set_piece_slot(x, SHOWCASE_DIMS, slot, straight)
    for slot in track2_curves_first + track2_curves_second:
        set_piece_slot(x, SHOWCASE_DIMS, slot, curve)
        set_slot_flip(x, SHOWCASE_DIMS, slot, 0)
    for slot in track2_straights:
        set_piece_slot(x, SHOWCASE_DIMS, slot, straight)

    def _track_chain(curves_first, curves_second, straights, dc_in_port, dc_out_port,
                     edge_row):
        # First half-circle
        for i in range(len(curves_first) - 1):
            set_port_pair(x, SHOWCASE_DIMS, edge_row,
                          curves_first[i], PORT_B, curves_first[i + 1], PORT_A)
            edge_row += 1
        # Last curve → DC.in
        set_port_pair(x, SHOWCASE_DIMS, edge_row,
                      curves_first[-1], PORT_B, 0, dc_in_port)
        edge_row += 1
        # DC.out → first curve of second half
        set_port_pair(x, SHOWCASE_DIMS, edge_row,
                      0, dc_out_port, curves_second[0], PORT_A)
        edge_row += 1
        # Second half-circle
        for i in range(len(curves_second) - 1):
            set_port_pair(x, SHOWCASE_DIMS, edge_row,
                          curves_second[i], PORT_B, curves_second[i + 1], PORT_A)
            edge_row += 1
        # Last curve → first straight
        set_port_pair(x, SHOWCASE_DIMS, edge_row,
                      curves_second[-1], PORT_B, straights[0], PORT_A)
        edge_row += 1
        # Straight chain
        for i in range(len(straights) - 1):
            set_port_pair(x, SHOWCASE_DIMS, edge_row,
                          straights[i], PORT_B, straights[i + 1], PORT_A)
            edge_row += 1
        # Closing: last straight → first curve
        set_port_pair(x, SHOWCASE_DIMS, edge_row,
                      straights[-1], PORT_B, curves_first[0], PORT_A)
        edge_row += 1
        return edge_row

    edge_row = _track_chain(track1_curves_first, track1_curves_second,
                            track1_straights, PORT_A, PORT_B, 0)
    _track_chain(track2_curves_first, track2_curves_second,
                 track2_straights, PORT_C, PORT_D, edge_row)

    set_anchor(x, SHOWCASE_DIMS, 0, 0, 0)
    return x


# =============================================================================
# Public API
# =============================================================================


_LAYOUTS: tuple = (
    ("circle", "16 R40 curves in a closed loop", _build_circle),
    ("oval", "8 curves + 4 straights + 8 curves + 4 straights", _build_oval),
    ("siding", "Oval with passing siding (2 switches + 2-curve branch)", _build_siding),
    ("figure8_cross90", "Two cycles sharing a perpendicular CROSS_90", _build_figure8_cross90),
    ("double_crossover", "Two parallel cycles via one DOUBLE_CROSSOVER", _build_double_crossover),
)


def build_topology_showcase(
    catalog: TrackCatalog, config: OptimizationConfig,
) -> List[Dict[str, Any]]:
    """Decode 5 hand-built chromosomes to layout-JSON entries."""
    decoder_config = DecoderConfig(
        boundary_min_x=config.boundary.min_x,
        boundary_max_x=config.boundary.max_x,
        boundary_min_y=config.boundary.min_y,
        boundary_max_y=config.boundary.max_y,
    )

    layouts: List[Dict[str, Any]] = []
    for name, description, builder in _LAYOUTS:
        chromosome = builder(catalog)
        graph = decode_chromosome(chromosome, SHOWCASE_DIMS, catalog, decoder_config)
        layout = port_graph_to_json(graph, catalog, config)
        layout["name"] = name
        layout["description"] = description
        layouts.append(layout)
    return layouts
