"""Tests for _inject_cross_junctions (DC-style bare-crossing model).

A CROSS_90 is placed where the loop's FK chain passes through the same world
point twice on perpendicular STRAIGHT_16 segments. The descriptor is
(active, pos_1, pos_2); injection sets BOTH slots to CROSS_90 and consumes ONE
physical CROSS_90. Because CROSS_90 FK == STRAIGHT_16 FK, the rewrite preserves
the FK chain (closure cannot break).
"""

import numpy as np
import pytest

from src.catalog import TrackCatalog
from src.decoder.construction import _inject_cross_junctions, fk_array_with_flips
from src.decoder.types import InventoryTracker
from src.encoding import (
    CROSS_90,
    R40_CURVE,
    STRAIGHT_16,
    SWITCH_LEFT,
    PartitionedDimensions,
    create_empty_chromosome,
    set_cross_junction,
)
from src.geometry import compute_fk_chain
from src.intersection import count_dangling_cross_ports

# A bare perpendicular self-crossing: 4 straights east, 12 LEFT R40 curves (270
# deg), 4 straights heading south. The southbound run crosses the eastbound run
# at world (24, 0); slots 1 and 18 are the crossing STRAIGHT_16s (exactly 90 deg).
CROSSING_PIECES = [int(STRAIGHT_16)] * 4 + [int(R40_CURVE)] * 12 + [int(STRAIGHT_16)] * 4
CROSSING_FLIPS = [0] * len(CROSSING_PIECES)
CROSS_SLOT_1, CROSS_SLOT_2 = 1, 18


@pytest.fixture
def cat() -> TrackCatalog:
    return TrackCatalog.load("data/track_pieces_v2.yaml")


@pytest.fixture
def dims() -> PartitionedDimensions:
    return PartitionedDimensions(
        n_main=40,
        max_junctions=0,
        max_cross_junctions=2,
        max_double_crossovers=0,
        n_straights_16=120, n_straights_24=0,
        boundary_min_x=-200.0,
        boundary_max_x=200.0,
        boundary_min_y=-200.0,
        boundary_max_y=200.0,
    )


def _make_tracker(cat: TrackCatalog, inv: dict, pieces: list) -> InventoryTracker:
    t = InventoryTracker(inv, cat)
    for p in pieces:
        if p >= 0:
            t.use(p)
    return t


_INV = {"STRAIGHT_16": 120, "R40_CURVE": 80, "CROSS_90": 2, "R40_SWITCH_LEFT": 3}


class TestInjectCrossJunctionSuccess:
    def test_perpendicular_crossing_injects_one_cross(self, cat, dims) -> None:
        pieces = list(CROSSING_PIECES)
        flips = list(CROSSING_FLIPS)
        x = create_empty_chromosome(dims)
        set_cross_junction(x, dims, slot=0, active=1, pos_1=CROSS_SLOT_1, pos_2=CROSS_SLOT_2)
        tracker = _make_tracker(cat, _INV, pieces)

        result = _inject_cross_junctions(
            pieces, x, dims, tracker, cat, main_flips=flips,
        )

        assert len(result) == 1
        assert set(result[0].positions) == {CROSS_SLOT_1, CROSS_SLOT_2}
        assert pieces[CROSS_SLOT_1] == int(CROSS_90)
        assert pieces[CROSS_SLOT_2] == int(CROSS_90)

    def test_consumes_exactly_one_physical_cross(self, cat, dims) -> None:
        pieces = list(CROSSING_PIECES)
        x = create_empty_chromosome(dims)
        set_cross_junction(x, dims, slot=0, active=1, pos_1=CROSS_SLOT_1, pos_2=CROSS_SLOT_2)
        tracker = _make_tracker(cat, _INV, pieces)

        _inject_cross_junctions(pieces, x, dims, tracker, cat,
                                main_flips=list(CROSSING_FLIPS))

        assert tracker.used.get(int(CROSS_90), 0) == 1

    def test_fk_preserved_and_no_dangling(self, cat, dims) -> None:
        pieces = list(CROSSING_PIECES)
        flips = list(CROSSING_FLIPS)
        before = compute_fk_chain(fk_array_with_flips(cat, pieces, flips))
        x = create_empty_chromosome(dims)
        set_cross_junction(x, dims, slot=0, active=1, pos_1=CROSS_SLOT_1, pos_2=CROSS_SLOT_2)
        tracker = _make_tracker(cat, _INV, pieces)

        _inject_cross_junctions(pieces, x, dims, tracker, cat,
                                main_flips=flips)

        after = compute_fk_chain(fk_array_with_flips(cat, pieces, flips))
        np.testing.assert_allclose(after, before, atol=1e-9)
        assert count_dangling_cross_ports(after, pieces) == 0


class TestInjectCrossJunctionFailurePaths:
    def test_no_descriptors_returns_empty(self, cat, dims) -> None:
        pieces = list(CROSSING_PIECES)
        x = create_empty_chromosome(dims)  # all descriptors inactive
        tracker = _make_tracker(cat, _INV, pieces)
        result = _inject_cross_junctions(pieces, x, dims, tracker, cat,
                                         main_flips=list(CROSSING_FLIPS))
        assert result == []
        assert tracker.used.get(int(CROSS_90), 0) == 0

    def test_non_straight_slot_skipped(self, cat, dims) -> None:
        pieces = list(CROSSING_PIECES)
        original = list(pieces)
        x = create_empty_chromosome(dims)
        # slot 5 is an R40_CURVE, not STRAIGHT_16
        set_cross_junction(x, dims, slot=0, active=1, pos_1=5, pos_2=CROSS_SLOT_2)
        tracker = _make_tracker(cat, _INV, pieces)
        result = _inject_cross_junctions(pieces, x, dims, tracker, cat,
                                         main_flips=list(CROSSING_FLIPS))
        assert result == []
        assert pieces == original
        assert tracker.used.get(int(CROSS_90), 0) == 0

    def test_insufficient_inventory_skipped(self, cat, dims) -> None:
        pieces = list(CROSSING_PIECES)
        original = list(pieces)
        x = create_empty_chromosome(dims)
        set_cross_junction(x, dims, slot=0, active=1, pos_1=CROSS_SLOT_1, pos_2=CROSS_SLOT_2)
        inv = {"STRAIGHT_16": 120, "R40_CURVE": 80, "CROSS_90": 0}
        tracker = _make_tracker(cat, inv, pieces)
        result = _inject_cross_junctions(pieces, x, dims, tracker, cat,
                                         main_flips=list(CROSSING_FLIPS))
        assert result == []
        assert pieces == original

    def test_non_coincident_slots_skipped(self, cat, dims) -> None:
        """Two parallel, far-apart straights do not form a crossing."""
        pieces = [int(STRAIGHT_16)] * 16
        original = list(pieces)
        x = create_empty_chromosome(dims)
        set_cross_junction(x, dims, slot=0, active=1, pos_1=1, pos_2=8)
        tracker = _make_tracker(cat, _INV, pieces)
        result = _inject_cross_junctions(pieces, x, dims, tracker, cat,
                                         main_flips=[0] * 16)
        assert result == []
        assert pieces == original
        assert tracker.used.get(int(CROSS_90), 0) == 0

    def test_slot_occupied_by_switch_skipped(self, cat, dims) -> None:
        pieces = list(CROSSING_PIECES)
        pieces[CROSS_SLOT_1] = int(SWITCH_LEFT)
        original = list(pieces)
        x = create_empty_chromosome(dims)
        set_cross_junction(x, dims, slot=0, active=1, pos_1=CROSS_SLOT_1, pos_2=CROSS_SLOT_2)
        tracker = _make_tracker(cat, _INV, pieces)
        result = _inject_cross_junctions(pieces, x, dims, tracker, cat,
                                         main_flips=list(CROSSING_FLIPS))
        assert result == []
        assert pieces == original
