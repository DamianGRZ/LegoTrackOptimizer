"""Tests for Phase 7b -- parallel-tracks-with-DC heuristic seed.

Phase 7b adds ``_emit_parallel_tracks_with_dc`` mirroring Phase 6b's
shape: two parallel stadium ovals (each ``[R40 x 8, STR x m, R40 x 8,
STR x m]``) joined at one slot by a ``DOUBLE_CROSSOVER``. The DC slot
sits in slot 0 with track 1 wired through ports A/B and track 2 wired
through ports C/D. A ``PARALLEL_DC_BRIDGE`` junction descriptor is
attached anchored at slot 0 so Phase 7c-onwards mutations can recognise
the structure even though Phase 7a's materializer is functionally a
no-op when the anchor is already a DC.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src_v2.catalog import TrackCatalog
from src_v2.config import BoundaryConfig, OptimizationConfig
from src_v2.encoding import (
    JUNCTION_KIND_PARALLEL_DC_BRIDGE,
    compute_port_pair_dimensions,
)
from src_v2.operators import _emit_parallel_tracks_with_dc


CATALOG_PATH = Path(__file__).parent.parent / "data" / "track_pieces_v2.yaml"
WITH_CROSSING_CFG = Path(__file__).parent.parent / "configs" / "with_crossing.yaml"


@pytest.fixture(scope="module")
def catalog() -> TrackCatalog:
    return TrackCatalog.load(CATALOG_PATH)


@pytest.fixture
def with_crossing(catalog):
    cfg = OptimizationConfig.load(WITH_CROSSING_CFG)
    # Phase 7b explicitly needs a DOUBLE_CROSSOVER plus 32 R40 + straights for
    # the dual-stadium layout. The bundled with_crossing config only ships a
    # CROSS_90, so synthesise a DC-aware inventory for these emitter tests.
    inv = dict(cfg.inventory)
    inv["DOUBLE_CROSSOVER"] = max(inv.get("DOUBLE_CROSSOVER", 0), 1)
    inv["R40_CURVE"] = max(inv.get("R40_CURVE", 0), 32)
    inv["STRAIGHT_16"] = max(inv.get("STRAIGHT_16", 0), 16)
    dims = compute_port_pair_dimensions(cfg.boundary, catalog, inv)
    return cfg, dims, inv


# ---------------------------------------------------------------- 7b.1
def test_7b_1_dc_slot_at_index_zero(catalog, with_crossing) -> None:
    """Slot 0 of every emitted pattern is a DOUBLE_CROSSOVER."""
    cfg, dims, inv = with_crossing
    patterns = _emit_parallel_tracks_with_dc(
        catalog, inv, dims, cfg.boundary,
    )
    assert patterns, "expected at least one pattern with DC + R40 inventory"
    for slots, _edges, _flips, _rotates, _junctions in patterns:
        slot0_piece = next((pid for s, pid in slots if s == 0), None)
        assert slot0_piece == "DOUBLE_CROSSOVER", (
            f"slot 0 should be DOUBLE_CROSSOVER, got {slot0_piece}"
        )


# ---------------------------------------------------------------- 7b.2
def test_7b_2_two_parallel_stadium_loops(catalog, with_crossing) -> None:
    """Each pattern has 32 R40 + balanced straight pairs (one stadium per
    track, two tracks). Total slot count = 1 (DC) + 32 (R40) + 4*m (STR)."""
    cfg, dims, inv = with_crossing
    patterns = _emit_parallel_tracks_with_dc(
        catalog, inv, dims, cfg.boundary,
    )
    assert patterns
    for slots, _edges, _flips, _rotates, _junctions in patterns:
        n_dc = sum(1 for _, pid in slots if pid == "DOUBLE_CROSSOVER")
        n_r40 = sum(1 for _, pid in slots if pid == "R40_CURVE")
        n_str = sum(1 for _, pid in slots if pid == "STRAIGHT_16")
        assert n_dc == 1, f"want 1 DC, got {n_dc}"
        assert n_r40 == 32, f"want 32 R40, got {n_r40}"
        assert n_str % 4 == 0, f"want straight count divisible by 4, got {n_str}"
        assert n_str >= 4, "want at least 4 straights (m=1 per stadium half)"
        assert len(slots) == 1 + 32 + n_str


# ---------------------------------------------------------------- 7b.3
def test_7b_3_no_dc_in_inventory_returns_empty(catalog, with_crossing) -> None:
    """Without DOUBLE_CROSSOVER the emitter falls back gracefully."""
    cfg, dims, inv = with_crossing
    inv_no_dc = {k: v for k, v in inv.items() if k != "DOUBLE_CROSSOVER"}
    patterns = _emit_parallel_tracks_with_dc(
        catalog, inv_no_dc, dims, cfg.boundary,
    )
    assert patterns == []


# ---------------------------------------------------------------- 7b.4
def test_7b_4_insufficient_r40_returns_empty(catalog, with_crossing) -> None:
    """Two parallel ovals need 32 R40_CURVE; less = empty."""
    cfg, dims, inv = with_crossing
    inv_short = dict(inv)
    inv_short["R40_CURVE"] = 31
    patterns = _emit_parallel_tracks_with_dc(
        catalog, inv_short, dims, cfg.boundary,
    )
    assert patterns == []


# ---------------------------------------------------------------- 7b.5
def test_7b_5_tiny_boundary_returns_empty(catalog, with_crossing) -> None:
    """Boundary too small to fit two stacked ovals -> empty list."""
    cfg, dims, inv = with_crossing
    tiny = BoundaryConfig(min_x=-30, max_x=30, min_y=-30, max_y=30)
    patterns = _emit_parallel_tracks_with_dc(
        catalog, inv, dims, tiny,
    )
    assert patterns == []


# ---------------------------------------------------------------- 7b.6
def test_7b_6_emitter_seeds_active_dc_bridge_junction(
    catalog, with_crossing,
) -> None:
    """Each pattern carries an active PARALLEL_DC_BRIDGE junction descriptor
    anchored at slot 0 (the DC slot)."""
    cfg, dims, inv = with_crossing
    patterns = _emit_parallel_tracks_with_dc(
        catalog, inv, dims, cfg.boundary,
    )
    assert patterns
    for slots, _edges, _flips, _rotates, junctions in patterns:
        assert junctions, "expected at least one junction descriptor per pattern"
        active = [j for j in junctions if j[0] == 1]
        assert active, "no active junctions"
        for _act, anchor, kind, _pa, _pb in active:
            assert kind == JUNCTION_KIND_PARALLEL_DC_BRIDGE
            assert anchor == 0, f"anchor should be the DC slot (0), got {anchor}"


# ---------------------------------------------------------------- 7b.7
def test_7b_7_track1_uses_ports_AB_track2_uses_ports_CD(
    catalog, with_crossing,
) -> None:
    """Slot 0's DC has both port A/B edges (track 1) and both port C/D
    edges (track 2). Each port appears exactly once in the edge list."""
    cfg, dims, inv = with_crossing
    patterns = _emit_parallel_tracks_with_dc(
        catalog, inv, dims, cfg.boundary,
    )
    assert patterns
    for _slots, edges, _flips, _rotates, _junctions in patterns:
        dc_ports = []
        for sa, port_a, sb, port_b in edges:
            if sa == 0:
                dc_ports.append(port_a)
            if sb == 0:
                dc_ports.append(port_b)
        assert sorted(dc_ports) == ["A", "B", "C", "D"], (
            f"DC ports not all wired exactly once: {dc_ports}"
        )
