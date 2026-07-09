"""Tests for count_dangling_cross_ports.

A CROSS_90 chromosome slot needs a perpendicular partner slot at the same
world midpoint for all 4 ports of the physical piece to be in use.
Without a partner, 2 ports dangle and the layout is unbuildable.
"""
import numpy as np
import pytest

from src.catalog import TrackCatalog
from src.encoding import CROSS_90, R40_CURVE, STRAIGHT_16
from src.geometry import compute_fk_chain
from src.intersection import count_dangling_cross_ports


@pytest.fixture
def cat() -> TrackCatalog:
    return TrackCatalog.load("data/track_pieces_v2.yaml")


def _states(pieces: list[int], cat: TrackCatalog) -> np.ndarray:
    indices = np.asarray(pieces, dtype=np.int32)
    return compute_fk_chain(cat.get_fk(indices))


class TestCountDanglingCrossPorts:
    def test_no_cross_no_dangling(self, cat: TrackCatalog) -> None:
        """A layout without any CROSS_90 returns 0 dangling."""
        pieces = [STRAIGHT_16] * 4 + [R40_CURVE] * 4
        s = _states(pieces, cat)
        assert count_dangling_cross_ports(s, pieces) == 0

    def test_lone_cross_is_dangling(self, cat: TrackCatalog) -> None:
        """A CROSS_90 with no perpendicular partner anywhere reports 1."""
        # Just place a CROSS_90 in the middle of straights — no other slot
        # is at the same world location with perpendicular heading.
        pieces = [STRAIGHT_16, STRAIGHT_16, CROSS_90, STRAIGHT_16, STRAIGHT_16]
        s = _states(pieces, cat)
        assert count_dangling_cross_ports(s, pieces) == 1

    def test_figure_eight_repaired_is_zero(self, cat: TrackCatalog) -> None:
        """The figure-8 spiral after repair: one CROSS_90 at slot 0 + a
        STRAIGHT_16 partner at slot 27 at the same world midpoint with
        perpendicular heading. count_dangling_cross_ports must report 0."""
        # Spiral known to produce a 90 deg STR-on-STR crossing at (0, 27).
        spiral = (
            [int(STRAIGHT_16)] * 6 + [int(R40_CURVE)] * 4
            + [int(STRAIGHT_16)] * 2 + [int(R40_CURVE)] * 4
            + [int(STRAIGHT_16)] * 3 + [int(R40_CURVE)] * 4
            + [int(STRAIGHT_16)] * 5
        )
        # Simulate post-repair: slot 0 was STR, becomes CROSS_90.
        # Slot 27 stays STR_16 — it's the perpendicular partner.
        repaired = list(spiral)
        repaired[0] = int(CROSS_90)
        s = _states(repaired, cat)
        # Slot 0 (CROSS_90, heading 0) and slot 27 (STR_16, heading 90)
        # should be at the same midpoint -> partner exists -> 0 dangling.
        assert count_dangling_cross_ports(s, repaired) == 0
