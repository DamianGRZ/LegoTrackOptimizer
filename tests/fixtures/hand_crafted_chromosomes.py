"""Hand-crafted chromosome library (Phase 17.C.2, PLAN Section 10.6).

Reference chromosomes used by Phase 1+ tests. Each builder takes
``(catalog, dims)`` and returns a deterministic ``int16`` array — no
global state, no side effects.

Phase-1-relevant entries authored now (per CLAUDE.md "no premature
abstraction"):

- ``perfect_oval_16_R40``       — closed-loop baseline
- ``deg250_deficit``            — open chain summing ~ 250°
- ``deg380_excess``             — closed loop summing ~ 380° (residual closure)
- ``broken_cycle_2_components`` — two disjoint 16-R40 ovals
- ``loose_port_chromosome``     — minimal open chain (2 R40, 1 edge)
- ``inventory_exhausted``       — 9 active R40 (paired with cap=8 in tests)
- ``isolated_active_slot``      — 1 active piece, 0 edges

Phase 2-7 entries (asymmetric_oval, oval_with_siding × 2, figure_8_with_cross,
parallel_dc_layout) authored when those phases land.
"""
from __future__ import annotations

from numpy.typing import NDArray

from src_v2.catalog import TrackCatalog
from src_v2.encoding import (
    PortPairDimensions,
    create_empty_chromosome,
    set_piece_slot,
    set_port_pair,
)


def perfect_oval_16_R40(catalog: TrackCatalog, dims: PortPairDimensions) -> NDArray:
    """16 R40 curves connected sequentially into a closed oval.

    Geometric closure exactly: 16 × 22.5° = 360°.
    """
    x = create_empty_chromosome(dims)
    c = catalog.id_to_index["R40_CURVE"]
    for k in range(16):
        set_piece_slot(x, dims, k, c)
    for k in range(16):
        set_port_pair(x, dims, k, k, 1, (k + 1) % 16, 0)
    return x


def deg250_deficit(catalog: TrackCatalog, dims: PortPairDimensions) -> NDArray:
    """11 R40 curves in a *closed* cycle — angular sum ~ 247.5° (113° deficit).

    Topologically a cycle (11 edges including the closer between slots 10 and
    0), but FK fails to close — accumulated rotation is short of 360° by
    ~ 113°, producing a large position-residual closure error. Phase 1's
    `CycleClosureRepair` should splice 5 more R40 into the cycle to reach
    ~ 360° (matches V1's `MainLoopClosureRepair._add_curves` reference).
    """
    x = create_empty_chromosome(dims)
    c = catalog.id_to_index["R40_CURVE"]
    for k in range(11):
        set_piece_slot(x, dims, k, c)
    for k in range(11):
        set_port_pair(x, dims, k, k, 1, (k + 1) % 11, 0)
    return x


def deg380_excess(catalog: TrackCatalog, dims: PortPairDimensions) -> NDArray:
    """17 R40 curves in a closed loop — sums to ~ 382.5°.

    All ports paired but the cycle's angular sum exceeds 360° by ~ 22.5°.
    Phase 1's repair should remove 1 R40 to reach 360°.
    """
    x = create_empty_chromosome(dims)
    c = catalog.id_to_index["R40_CURVE"]
    for k in range(17):
        set_piece_slot(x, dims, k, c)
    for k in range(17):
        set_port_pair(x, dims, k, k, 1, (k + 1) % 17, 0)
    return x


def broken_cycle_2_components(
    catalog: TrackCatalog, dims: PortPairDimensions,
) -> NDArray:
    """Two disjoint 16-R40 ovals at slots 0-15 and 16-31."""
    x = create_empty_chromosome(dims)
    c = catalog.id_to_index["R40_CURVE"]
    # Component 1: slots 0-15
    for k in range(16):
        set_piece_slot(x, dims, k, c)
    for k in range(16):
        set_port_pair(x, dims, k, k, 1, (k + 1) % 16, 0)
    # Component 2: slots 16-31
    for k in range(16):
        set_piece_slot(x, dims, 16 + k, c)
    for k in range(16):
        set_port_pair(x, dims, 16 + k, 16 + k, 1, 16 + ((k + 1) % 16), 0)
    return x


def loose_port_chromosome(
    catalog: TrackCatalog, dims: PortPairDimensions,
) -> NDArray:
    """Two R40 pieces joined by one edge — both outer ports unpaired.

    Minimal "needs-repair" chromosome: 2 active slots, 1 edge, 2 loose ports.
    """
    x = create_empty_chromosome(dims)
    c = catalog.id_to_index["R40_CURVE"]
    set_piece_slot(x, dims, 0, c)
    set_piece_slot(x, dims, 1, c)
    set_port_pair(x, dims, 0, 0, 1, 1, 0)
    return x


def inventory_exhausted(
    catalog: TrackCatalog, dims: PortPairDimensions,
) -> NDArray:
    """9 active R40 pieces (no edges) — exceeds an 8-piece R40 inventory cap.

    Builder is inventory-agnostic; pair this with an 8-R40 inventory at
    problem-construction time and assert the per-type inventory-excess
    constraint G[5+T_R40] > 0.
    """
    x = create_empty_chromosome(dims)
    c = catalog.id_to_index["R40_CURVE"]
    for k in range(9):
        set_piece_slot(x, dims, k, c)
    return x


def isolated_active_slot(
    catalog: TrackCatalog, dims: PortPairDimensions,
) -> NDArray:
    """1 active R40 with no edges — port-decoder edge case (no graph)."""
    x = create_empty_chromosome(dims)
    c = catalog.id_to_index["R40_CURVE"]
    set_piece_slot(x, dims, 0, c)
    return x
