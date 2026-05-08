"""Tests for Phase 2's sequential-ring stadium emitter (PLAN §10.2 2.1-2.4).

Phase 2 implements V1's "implicit chain -> closed walk by construction"
advantage on top of V2's port-pair encoding via a new heuristic emitter
``_emit_sequential_ring_stadium``. The pattern is two banks of 8 R40 curves
separated by symmetric straight runs of length m -- a stadium geometry the
GA's existing operators don't reliably grow into from random init.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src_v2.catalog import TrackCatalog
from src_v2.config import BoundaryConfig, OptimizationConfig
from src_v2.encoding import PortPairDimensions, compute_port_pair_dimensions
from src_v2.operators import _emit_sequential_ring_stadium


CATALOG_PATH = Path(__file__).parent.parent / "data" / "track_pieces_v2.yaml"
CONFIG_PATH = Path(__file__).parent.parent / "configs" / "default.yaml"


@pytest.fixture(scope="module")
def catalog() -> TrackCatalog:
    return TrackCatalog.load(CATALOG_PATH)


@pytest.fixture
def config() -> OptimizationConfig:
    return OptimizationConfig.load(CONFIG_PATH)


@pytest.fixture
def dims(config: OptimizationConfig, catalog: TrackCatalog) -> PortPairDimensions:
    return compute_port_pair_dimensions(
        config.boundary, catalog, config.inventory,
    )


# ---------------------------------------------------------------- 2.1
def test_2_1_two_corners_of_8_pattern(catalog, config, dims) -> None:
    """Each emitted pattern is [R40 x 8, STR x m, R40 x 8, STR x m]."""
    patterns = _emit_sequential_ring_stadium(
        catalog, config.inventory, dims, config.boundary,
    )

    assert patterns, "expected at least one pattern for default inventory"
    for slots, _edges, _flips, _rotates in patterns:
        pieces = [pid for _, pid in slots]
        n = len(pieces)
        assert n >= 18 and (n - 16) % 2 == 0, f"unexpected total length {n}"
        m = (n - 16) // 2
        expected = (
            ["R40_CURVE"] * 8
            + ["STRAIGHT_16"] * m
            + ["R40_CURVE"] * 8
            + ["STRAIGHT_16"] * m
        )
        assert pieces == expected, f"pattern mismatch: {pieces}"
        # Slot indices are 0..n-1 contiguous (sequential ring).
        assert [s for s, _ in slots] == list(range(n))


# ---------------------------------------------------------------- 2.2
def test_2_2_inventory_r40_lt_16_returns_empty(catalog, config, dims) -> None:
    """Insufficient R40 inventory -> emitter returns an empty list (graceful skip)."""
    inventory = {"R40_CURVE": 15, "STRAIGHT_16": 16}
    patterns = _emit_sequential_ring_stadium(
        catalog, inventory, dims, config.boundary,
    )
    assert patterns == []


# ---------------------------------------------------------------- 2.3
def test_2_3_boundary_too_small_returns_empty(catalog, config, dims) -> None:
    """Boundary smaller than the end-cap footprint -> emitter returns empty list."""
    tiny_boundary = BoundaryConfig(min_x=-30, max_x=30, min_y=-30, max_y=30)
    patterns = _emit_sequential_ring_stadium(
        catalog, config.inventory, dims, tiny_boundary,
    )
    assert patterns == []


# ---------------------------------------------------------------- 2.4
def test_2_4_sequential_edges_port_uniqueness(catalog, config, dims) -> None:
    """No (slot, port_name) pair is referenced twice across the edge list."""
    patterns = _emit_sequential_ring_stadium(
        catalog, config.inventory, dims, config.boundary,
    )
    assert patterns
    for slots, edges, _flips, _rotates in patterns:
        seen: set[tuple[int, str]] = set()
        for sa, port_a, sb, port_b in edges:
            for ref in ((sa, port_a), (sb, port_b)):
                assert ref not in seen, f"port {ref} reused across edges"
                seen.add(ref)
        n = len(slots)
        assert len(edges) == n, "sequential ring should emit one edge per slot"
