"""Tests for _apply_crossing_repair (FK-neutral by-chance CROSS_90 placement).

FK-neutral policy: a self-crossing is converted to CROSS_90 ONLY when both
crossing segments are STRAIGHT_16 (CROSS_90 FK == STRAIGHT_16 FK, so the rewrite
preserves the chain) AND the crossing is ~perpendicular. Curve-involved or
non-perpendicular crossings are left unconverted (a mild g_collisions penalty),
never rewritten into a closure break or a dangling cross.
"""

import numpy as np
import pytest

from src.catalog import TrackCatalog
from src.decoder.construction import _apply_crossing_repair, fk_array_with_flips
from src.decoder.types import InventoryTracker
from src.encoding import CROSS_90, R40_CURVE, STRAIGHT_16
from src.geometry import compute_fk_chain

# Verified perpendicular STR-on-STR self-crossing (slots 1 and 18 cross at 90 deg).
STR_CROSS = [int(STRAIGHT_16)] * 4 + [int(R40_CURVE)] * 12 + [int(STRAIGHT_16)] * 4
# A pure-curve spiral whose only self-crossings are curve-on-curve.
CURVE_CROSS = [int(R40_CURVE)] * 18


@pytest.fixture
def cat() -> TrackCatalog:
    return TrackCatalog.load("data/track_pieces_v2.yaml")


def _make_tracker(cat: TrackCatalog, inv: dict, pieces: list) -> InventoryTracker:
    t = InventoryTracker(inv, cat)
    for p in pieces:
        t.use(p)
    return t


class TestApplyCrossingRepair:
    def test_no_inventory_no_change(self, cat) -> None:
        pieces = list(STR_CROSS)
        tracker = _make_tracker(cat, {"STRAIGHT_16": 60, "R40_CURVE": 20, "CROSS_90": 0}, pieces)
        result, _flips, records = _apply_crossing_repair(pieces, tracker, cat)
        assert result == pieces
        assert records == []
        assert tracker.used.get(int(CROSS_90), 0) == 0

    def test_no_crossings_no_change(self, cat) -> None:
        pieces = [int(R40_CURVE)] * 16  # closed circle, no self-intersection
        tracker = _make_tracker(cat, {"R40_CURVE": 20, "CROSS_90": 20}, pieces)
        result, _flips, records = _apply_crossing_repair(pieces, tracker, cat)
        assert result == pieces
        assert records == []
        assert tracker.used.get(int(CROSS_90), 0) == 0

    def test_perpendicular_str_crossing_converted_fk_preserved(self, cat) -> None:
        pieces = list(STR_CROSS)
        flips = [0] * len(pieces)
        before = compute_fk_chain(fk_array_with_flips(cat, pieces, flips))
        tracker = _make_tracker(cat, {"STRAIGHT_16": 60, "R40_CURVE": 20, "CROSS_90": 20}, pieces)

        result, result_flips, records = _apply_crossing_repair(pieces, tracker, cat, flips=flips)

        # Both slots of the (1, 18) crossing become CROSS_90 — one physical piece
        # traversed twice — mirroring the descriptor commit exactly.
        assert result[1] == int(CROSS_90)
        assert result[18] == int(CROSS_90)
        assert tracker.used.get(int(CROSS_90), 0) == 1
        assert tracker.used.get(int(STRAIGHT_16), 0) == 6, "both straights released (8 - 2)"
        assert len(records) == 1
        assert records[0].positions == (1, 18)
        assert records[0].slot == -1, "emergent records carry no descriptor slot"
        # FK-neutral: STRAIGHT_16 and CROSS_90 share FK [16,0,0], so the chain is unchanged.
        after = compute_fk_chain(fk_array_with_flips(cat, result, result_flips))
        np.testing.assert_allclose(after, before, atol=1e-9)

    def test_curve_crossings_left_unconverted(self, cat) -> None:
        """Curve-on-curve crossings must NOT be rewritten (would break closure)."""
        pieces = list(CURVE_CROSS)
        tracker = _make_tracker(cat, {"R40_CURVE": 40, "CROSS_90": 20}, pieces)
        result, _flips, records = _apply_crossing_repair(pieces, tracker, cat)
        assert result == pieces
        assert records == []
        assert tracker.used.get(int(CROSS_90), 0) == 0
        assert all(p != int(CROSS_90) for p in result)
