"""Tests for Phase 6b -- figure-8 stadium heuristic seed.

Phase 6b adds ``_emit_figure8_stadium`` mirroring Phase 5b's pattern: a
symmetric ``[R40 x 8, STR x m, R40 x 8, STR x m]`` oval mainline plus an
active ``FIGURE_8_CROSS`` junction descriptor anchored in one of the
straight sections. Phase 6a's :class:`JunctionMaterializer` is responsible
for expanding the descriptor into a full figure-8 -- whatever materializer
behaviour ships, the seed just needs to deliver a well-formed input.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src_v2.catalog import TrackCatalog
from src_v2.config import BoundaryConfig, OptimizationConfig
from src_v2.encoding import (
    JUNCTION_KIND_FIGURE_8_CROSS,
    PortPairDimensions,
    compute_port_pair_dimensions,
)
from src_v2.operators import _emit_figure8_stadium


CATALOG_PATH = Path(__file__).parent.parent / "data" / "track_pieces_v2.yaml"
WITH_CROSSING_CFG = Path(__file__).parent.parent / "configs" / "with_crossing.yaml"


@pytest.fixture(scope="module")
def catalog() -> TrackCatalog:
    return TrackCatalog.load(CATALOG_PATH)


@pytest.fixture
def with_crossing(catalog):
    cfg = OptimizationConfig.load(WITH_CROSSING_CFG)
    dims = compute_port_pair_dimensions(cfg.boundary, catalog, cfg.inventory)
    return cfg, dims


# ---------------------------------------------------------------- 6b.1
def test_6b_1_symmetric_oval_pattern(catalog, with_crossing) -> None:
    """Each emitted pattern's slot list is exactly
    ``[R40 x 8, STR x m, R40 x 8, STR x m]``."""
    cfg, dims = with_crossing
    patterns = _emit_figure8_stadium(catalog, cfg.inventory, dims, cfg.boundary)
    assert patterns, "expected at least one pattern with CROSS_90 + R40 inventory"
    for slots, _edges, _flips, _rotates, _junctions in patterns:
        pieces = [pid for _, pid in slots]
        n = len(pieces)
        assert n >= 18 and (n - 16) % 2 == 0, f"unexpected length {n}"
        m = (n - 16) // 2
        expected = (
            ["R40_CURVE"] * 8
            + ["STRAIGHT_16"] * m
            + ["R40_CURVE"] * 8
            + ["STRAIGHT_16"] * m
        )
        assert pieces == expected, "asymmetric / wrong piece order"


# ---------------------------------------------------------------- 6b.2
def test_6b_2_no_cross_in_inventory_returns_empty(catalog, with_crossing) -> None:
    """Without CROSS_90 the emitter falls back gracefully."""
    cfg, dims = with_crossing
    inv_no_cross = {
        k: v for k, v in cfg.inventory.items() if k != "CROSS_90"
    }
    patterns = _emit_figure8_stadium(catalog, inv_no_cross, dims, cfg.boundary)
    assert patterns == []


# ---------------------------------------------------------------- 6b.3
def test_6b_3_tiny_boundary_returns_empty(catalog, with_crossing) -> None:
    """Boundary that can't fit a 16-R40 oval -> empty list."""
    cfg, dims = with_crossing
    tiny = BoundaryConfig(min_x=-30, max_x=30, min_y=-30, max_y=30)
    patterns = _emit_figure8_stadium(catalog, cfg.inventory, dims, tiny)
    assert patterns == []


# ---------------------------------------------------------------- 6b.4
def test_6b_4_emitter_seeds_active_figure8_junction(catalog, with_crossing) -> None:
    """Each emitted pattern carries an active ``FIGURE_8_CROSS`` junction
    whose anchor is one of the straight-section slots in the mainline."""
    cfg, dims = with_crossing
    patterns = _emit_figure8_stadium(catalog, cfg.inventory, dims, cfg.boundary)
    assert patterns
    for slots, _edges, _flips, _rotates, junctions in patterns:
        assert junctions, "expected at least one junction descriptor per pattern"
        active_juncs = [j for j in junctions if j[0] == 1]
        assert active_juncs, "no active junctions found"
        for _active, anchor, kind, _pa, _pb in active_juncs:
            assert kind == JUNCTION_KIND_FIGURE_8_CROSS
            anchor_piece = next(
                (pid for s, pid in slots if s == anchor), None,
            )
            assert anchor_piece == "STRAIGHT_16", (
                f"anchor slot {anchor} expected STRAIGHT_16 host, "
                f"got {anchor_piece}"
            )
