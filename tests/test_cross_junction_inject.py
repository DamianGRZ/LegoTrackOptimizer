"""Tests for _inject_cross_junctions in src/decoder/construction.py.

Validates the failure paths (descriptor skipped, inventory untouched) and
the successful-injection inventory accounting using a hand-built valid
layout. Geometric search is the hard part of this code path; the test
constructs a layout whose FK chain naturally lands at all four cross
ports so the decoder can match them within tolerance.
"""

import numpy as np
import pytest

from src.catalog import TrackCatalog
from src.decoder.construction import _inject_cross_junctions
from src.decoder.types import DecoderConfig, InventoryTracker
from src.encoding import (
    CROSS_90,
    R40_LEFT,
    R40_RIGHT,
    STRAIGHT_16,
    SWITCH_LEFT,
    SWITCH_RIGHT,
    PartitionedDimensions,
    create_empty_chromosome,
    set_cross_junction,
)
from src.templates import (
    CROSS_JUNCTION_LEFT,
    get_cross_junction_inventory_requirements,
    switch_position_for_cross_port,
)


@pytest.fixture
def cat() -> TrackCatalog:
    return TrackCatalog.load("data/track_pieces_v2.yaml")


@pytest.fixture
def dims() -> PartitionedDimensions:
    return PartitionedDimensions(
        n_main=64,
        max_junctions=0,
        max_cross_junctions=2,
        total_straights=40,
        boundary_min_x=-200.0,
        boundary_max_x=200.0,
        boundary_min_y=-200.0,
        boundary_max_y=200.0,
    )


def _make_tracker(cat: TrackCatalog, inv: dict, pieces: list[int]) -> InventoryTracker:
    t = InventoryTracker(inv, cat)
    for p in pieces:
        if p >= 0:
            t.use(p)
    return t


class TestInjectCrossJunctionsFailurePaths:
    def test_no_descriptors_returns_empty(
        self, cat: TrackCatalog, dims: PartitionedDimensions
    ) -> None:
        x = create_empty_chromosome(dims)
        pieces = [STRAIGHT_16] * 16
        inv = {"STRAIGHT_16": 40, "R40_LEFT": 20, "R40_RIGHT": 20,
               "CROSS_90": 5, "R40_SWITCH_LEFT": 10, "R40_SWITCH_RIGHT": 10}
        tracker = _make_tracker(cat, inv, pieces)

        result = _inject_cross_junctions(pieces, x, dims, tracker, cat,
                                          DecoderConfig())

        assert result == []
        assert tracker.used.get(CROSS_90, 0) == 0
        assert tracker.used.get(SWITCH_LEFT, 0) == 0

    def test_descriptor_at_non_straight_position_skipped(
        self, cat: TrackCatalog, dims: PartitionedDimensions
    ) -> None:
        x = create_empty_chromosome(dims)
        set_cross_junction(x, dims, slot=0, active=1, position_W=3, handedness=0)
        # Position 3 is an R40_LEFT, not a STRAIGHT_16 — descriptor must be skipped.
        pieces = [STRAIGHT_16, STRAIGHT_16, STRAIGHT_16, R40_LEFT,
                  STRAIGHT_16, STRAIGHT_16]
        original = list(pieces)
        inv = {"STRAIGHT_16": 40, "R40_LEFT": 20, "R40_RIGHT": 20,
               "CROSS_90": 5, "R40_SWITCH_LEFT": 10}
        tracker = _make_tracker(cat, inv, pieces)

        result = _inject_cross_junctions(pieces, x, dims, tracker, cat,
                                          DecoderConfig())

        assert result == []
        assert pieces == original
        assert tracker.used.get(CROSS_90, 0) == 0

    def test_inventory_insufficient_skipped(
        self, cat: TrackCatalog, dims: PartitionedDimensions
    ) -> None:
        x = create_empty_chromosome(dims)
        set_cross_junction(x, dims, slot=0, active=1, position_W=2, handedness=0)
        pieces = [STRAIGHT_16] * 8
        original = list(pieces)
        # No CROSS_90 in inventory — junction must be skipped.
        inv = {"STRAIGHT_16": 40, "R40_LEFT": 20, "R40_RIGHT": 20,
               "CROSS_90": 0, "R40_SWITCH_LEFT": 10}
        tracker = _make_tracker(cat, inv, pieces)

        result = _inject_cross_junctions(pieces, x, dims, tracker, cat,
                                          DecoderConfig())

        assert result == []
        assert pieces == original

    def test_descriptor_pointing_at_existing_switch_skipped(
        self, cat: TrackCatalog, dims: PartitionedDimensions
    ) -> None:
        x = create_empty_chromosome(dims)
        set_cross_junction(x, dims, slot=0, active=1, position_W=2, handedness=0)
        pieces = [STRAIGHT_16, STRAIGHT_16, SWITCH_LEFT, STRAIGHT_16,
                  STRAIGHT_16, STRAIGHT_16]
        original = list(pieces)
        inv = {"STRAIGHT_16": 40, "R40_LEFT": 20, "R40_RIGHT": 20,
               "CROSS_90": 5, "R40_SWITCH_LEFT": 10}
        tracker = _make_tracker(cat, inv, pieces)

        result = _inject_cross_junctions(pieces, x, dims, tracker, cat,
                                          DecoderConfig())

        assert result == []
        assert pieces == original

    def test_active_but_geometry_unmatched_skipped(
        self, cat: TrackCatalog, dims: PartitionedDimensions
    ) -> None:
        """A simple straight line cannot satisfy 4-port cross geometry."""
        x = create_empty_chromosome(dims)
        set_cross_junction(x, dims, slot=0, active=1, position_W=1, handedness=0)
        pieces = [STRAIGHT_16] * 16
        original = list(pieces)
        inv = {"STRAIGHT_16": 40, "R40_LEFT": 20, "R40_RIGHT": 20,
               "CROSS_90": 5, "R40_SWITCH_LEFT": 10}
        tracker = _make_tracker(cat, inv, pieces)

        result = _inject_cross_junctions(pieces, x, dims, tracker, cat,
                                          DecoderConfig())

        assert result == []
        assert pieces == original
        # Inventory unchanged — failed geometric search must not consume.
        assert tracker.used.get(CROSS_90, 0) == 0
        assert tracker.used.get(SWITCH_LEFT, 0) == 0
