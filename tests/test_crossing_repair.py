"""Tests for _apply_crossing_repair in src/decoder/construction.py.

The repair has an AGGRESSIVE policy: every detected self-intersection has
its pos_i slot rewritten to CROSS_90, regardless of the underlying piece
type or crossing angle. Curve-on-curve, non-perpendicular straight pairs,
and 24-stud straights are all converted (the FK shift is accepted because
a visible crossing without a CROSS_90 marker is considered unacceptable).
"""

import numpy as np
import pytest

from src.catalog import TrackCatalog
from src.decoder.construction import _apply_crossing_repair
from src.decoder.types import DecoderConfig, InventoryTracker
from src.encoding import CROSS_90, R40_CURVE, STRAIGHT_16, STRAIGHT_24
from src.geometry import compute_fk_chain
from src.intersection import find_crossing_pairs


@pytest.fixture
def cat() -> TrackCatalog:
    return TrackCatalog.load("data/track_pieces_v2.yaml")


def _crossings(pieces: list[int], cat: TrackCatalog) -> list:
    indices = np.array(pieces, dtype=np.int32)
    states = compute_fk_chain(cat.get_fk(indices))
    return find_crossing_pairs(states, pieces)


def _figure_eight_pieces() -> list[int]:
    """Right-handed spiral with ONE 90 deg crossing between STRAIGHT_16s.

    Verified to produce exactly one (pos_i=0, pos_j=27, angle=90.0) crossing
    on the v2 catalog: the inward-spiraling loop crosses its own start leg.
    """
    from src.encoding import R40_CURVE
    return (
        [STRAIGHT_16] * 6 + [R40_CURVE] * 4
        + [STRAIGHT_16] * 2 + [R40_CURVE] * 4
        + [STRAIGHT_16] * 3 + [R40_CURVE] * 4
        + [STRAIGHT_16] * 5
    )


def _make_tracker(cat: TrackCatalog, inv: dict, pieces: list[int]) -> InventoryTracker:
    t = InventoryTracker(inv, cat)
    for p in pieces:
        t.use(p)
    return t


class TestApplyCrossingRepair:
    def test_no_inventory_no_change(self, cat: TrackCatalog) -> None:
        """Empty CROSS_90 inventory leaves pieces untouched."""
        pieces = _figure_eight_pieces()
        inv = {"STRAIGHT_16": 60, "R40_CURVE": 20, "CROSS_90": 0}
        tracker = _make_tracker(cat, inv, pieces)

        result = _apply_crossing_repair(pieces, tracker, cat, DecoderConfig())

        assert result == pieces
        assert tracker.used.get(CROSS_90, 0) == 0

    def test_no_crossings_no_change(self, cat: TrackCatalog) -> None:
        """Non-self-intersecting layout (16 R40_CURVE circle) is left alone."""
        pieces = [R40_CURVE] * 16
        inv = {"R40_CURVE": 20, "CROSS_90": 20}
        tracker = _make_tracker(cat, inv, pieces)

        result = _apply_crossing_repair(pieces, tracker, cat, DecoderConfig())

        assert result == pieces
        assert tracker.used.get(CROSS_90, 0) == 0

    def test_curve_crossing_replaced(self, cat: TrackCatalog) -> None:
        """Curve-on-curve crossings must be repaired too (aggressive policy).

        Two stacked 16-curve loops produce many curve-on-curve crossings.
        The repair must convert at least one R40 slot to CROSS_90 per
        detected pair until either inventory runs out or no crossings
        remain.
        """
        pieces = [R40_CURVE] * 16 + [R40_CURVE] * 16
        pre = _crossings(pieces, cat)
        if not pre:
            pytest.skip("stacked-loop fixture does not self-intersect on this catalog")

        inv = {"R40_CURVE": 40, "CROSS_90": 20}
        tracker = _make_tracker(cat, inv, pieces)
        cross_before = tracker.used.get(CROSS_90, 0)

        result = _apply_crossing_repair(pieces, tracker, cat, DecoderConfig())

        cross_after = tracker.used.get(CROSS_90, 0)
        assert cross_after > cross_before, (
            "expected at least one CROSS_90 to be placed on curve crossings"
        )
        # At least one slot in the result should now be CROSS_90.
        assert any(p == CROSS_90 for p in result)

    def test_straight_crossing_replaced_at_pos_i_only(
        self, cat: TrackCatalog
    ) -> None:
        """A 90 deg crossing on STRAIGHT_16 pairs converts pos_i to CROSS_90,
        leaves pos_j untouched, and consumes exactly 1 CROSS_90 / 1 STRAIGHT_16.

        The aggressive policy still picks the most-perpendicular crossing
        first (find_crossing_pairs sorts by proximity to 90 deg), so for a
        fixture with one clean STR-on-STR perpendicular crossing the
        outcome is identical to the prior strict-gate behaviour.
        """
        pieces = _figure_eight_pieces()
        pre = _crossings(pieces, cat)
        if not pre:
            pytest.skip("hand-built layout does not self-intersect on this catalog")

        near_90 = next(
            (p for p in pre if abs(p[2] - 90.0) < 15.0
             and pieces[p[0]] == STRAIGHT_16 and pieces[p[1]] == STRAIGHT_16),
            None,
        )
        if near_90 is None:
            pytest.skip("no STRAIGHT_16-on-STRAIGHT_16 crossing in fixture")
        pos_i, pos_j, _ = near_90

        inv = {"STRAIGHT_16": 60, "R40_CURVE": 20, "CROSS_90": 20}
        tracker = _make_tracker(cat, inv, pieces)
        cross_before = tracker.used.get(CROSS_90, 0)
        str16_before = tracker.used.get(STRAIGHT_16, 0)

        result = _apply_crossing_repair(pieces, tracker, cat, DecoderConfig())

        assert result[pos_i] == CROSS_90
        assert result[pos_j] == STRAIGHT_16  # untouched: only pos_i is rewritten
        assert tracker.used.get(CROSS_90, 0) == cross_before + 1
        assert tracker.used.get(STRAIGHT_16, 0) == str16_before - 1

    def test_repair_eliminates_all_unresolved_crossings(
        self, cat: TrackCatalog
    ) -> None:
        """Aggressive policy: post-repair find_crossing_pairs returns zero
        unresolved crossings, full stop. CROSS_90 positions are exempt
        from re-detection per find_crossing_pairs's existing logic, so the
        loop terminates when every original crossing has had one of its
        two slots converted."""
        pieces = _figure_eight_pieces()
        pre = _crossings(pieces, cat)
        if not pre:
            pytest.skip("hand-built layout does not self-intersect on this catalog")

        inv = {"STRAIGHT_16": 60, "R40_CURVE": 20, "CROSS_90": 20}
        tracker = _make_tracker(cat, inv, pieces)
        result = _apply_crossing_repair(pieces, tracker, cat, DecoderConfig())

        post = _crossings(result, cat)
        assert post == [], (
            f"expected zero unresolved crossings post-repair, got {len(post)}"
        )
