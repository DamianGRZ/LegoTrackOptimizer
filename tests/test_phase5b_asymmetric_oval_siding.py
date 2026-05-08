"""Tests for Phase 5b -- asymmetric-oval-with-siding heuristic seed
(PLAN §10.2 5b.1, 5b.3, 5b.4).

Phase 5b adds a heuristic emitter that builds an asymmetric oval mainline
with `(m+2)` straights on the siding side and `m` on the other, plus an
active junction descriptor pointing into the longer straight section. The
+2 STR reserves 32-stud surplus for Phase 5a's switch injection so the
oval still closes after materialization (Coupling C).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src_v2.catalog import TrackCatalog
from src_v2.config import BoundaryConfig, OptimizationConfig
from src_v2.encoding import (
    JUNCTION_KIND_PASSING_SIDING,
    PortPairDimensions,
    compute_port_pair_dimensions,
)
from src_v2.operators import _emit_asymmetric_oval_with_siding


CATALOG_PATH = Path(__file__).parent.parent / "data" / "track_pieces_v2.yaml"
WITH_SWITCHES_CFG = Path(__file__).parent.parent / "configs" / "with_switches.yaml"


@pytest.fixture(scope="module")
def catalog() -> TrackCatalog:
    return TrackCatalog.load(CATALOG_PATH)


@pytest.fixture
def with_switches(catalog):
    cfg = OptimizationConfig.load(WITH_SWITCHES_CFG)
    dims = compute_port_pair_dimensions(cfg.boundary, catalog, cfg.inventory)
    return cfg, dims


# ---------------------------------------------------------------- 5b.1
def test_5b_1_two_corners_of_8_with_m_plus_2_asymmetry(catalog, with_switches) -> None:
    """Each pattern is [R40 x 8, STR x (m+2), R40 x 8, STR x m] -- two corners
    of 8 (Rule 10) with the +2 STR reserve on the siding side."""
    cfg, dims = with_switches
    patterns = _emit_asymmetric_oval_with_siding(
        catalog, cfg.inventory, dims, cfg.boundary,
    )
    assert patterns, "expected at least one pattern with switches in inventory"
    for slots, _edges, _flips, _rotates, _junctions in patterns:
        pieces = [pid for _, pid in slots]
        # First 8 are R40
        assert pieces[0:8] == ["R40_CURVE"] * 8
        # Find the boundary between section A straights and the second R40 bank
        section_a_end = next(
            (i for i in range(8, len(pieces)) if pieces[i] != "STRAIGHT_16"),
            None,
        )
        assert section_a_end is not None
        m_plus_2 = section_a_end - 8
        # Next 8 are R40
        assert pieces[section_a_end : section_a_end + 8] == ["R40_CURVE"] * 8
        # Then m STR
        section_b = pieces[section_a_end + 8 :]
        assert all(p == "STRAIGHT_16" for p in section_b)
        m = len(section_b)
        assert m_plus_2 == m + 2, (
            f"asymmetry violated: section A has {m_plus_2} STR, section B has {m}"
        )


# ---------------------------------------------------------------- 5b.3
def test_5b_3_no_switches_in_inventory_returns_empty(catalog, with_switches) -> None:
    """Without LEFT or RIGHT switches the emitter falls back gracefully."""
    cfg, dims = with_switches
    inv_no_switches = {
        k: v for k, v in cfg.inventory.items()
        if k not in {"R40_SWITCH_LEFT", "R40_SWITCH_RIGHT"}
    }
    patterns = _emit_asymmetric_oval_with_siding(
        catalog, inv_no_switches, dims, cfg.boundary,
    )
    assert patterns == []


# ---------------------------------------------------------------- 5b.4
def test_5b_4_tiny_boundary_returns_empty(catalog, with_switches) -> None:
    """Boundary that can't fit a siding (extra_axial=32 reservation) returns []."""
    cfg, dims = with_switches
    tiny_boundary = BoundaryConfig(min_x=-50, max_x=50, min_y=-50, max_y=50)
    patterns = _emit_asymmetric_oval_with_siding(
        catalog, cfg.inventory, dims, tiny_boundary,
    )
    assert patterns == []


# ---------------------------------------------------------------- Bonus
def test_emitter_seeds_active_junction_descriptor(catalog, with_switches) -> None:
    """Each emitted pattern includes an active junction descriptor pointing
    into the longer straight section (anchor in slots [8 .. 8 + m + 1])."""
    cfg, dims = with_switches
    patterns = _emit_asymmetric_oval_with_siding(
        catalog, cfg.inventory, dims, cfg.boundary,
    )
    assert patterns
    for slots, _edges, _flips, _rotates, junctions in patterns:
        assert junctions, "expected at least one junction descriptor per pattern"
        active_juncs = [j for j in junctions if j[0] == 1]
        assert active_juncs, "no active junction descriptors found"
        for active, anchor, kind, _param_a, _param_b in active_juncs:
            assert kind == JUNCTION_KIND_PASSING_SIDING
            # Anchor must land in the section A straights ([8 .. 8 + m + 1]).
            section_a_end = next(
                i for i, (_, pid) in enumerate(slots[8:], start=8)
                if pid != "STRAIGHT_16"
            )
            assert 8 <= anchor < section_a_end
