# tests/test_operators.py
"""Tests for partitioned chromosome genetic operators."""

import numpy as np
import pytest

from src.catalog import TrackCatalog
from src.encoding import (
    CROSS_90,
    GENES_PER_JUNCTION,
    R40_CURVE,
    R40_CURVE,
    STRAIGHT_16,
    PartitionedDimensions,
    create_empty_chromosome,
    set_junction,
)
from src.operators import (
    _change_handedness,
    _deactivate_position,
    _mutate_piece_type,
    _straighten_near_unresolved_crossing,
    _swap_positions,
)
from src.templates import TEMPLATES


@pytest.fixture
def dims() -> PartitionedDimensions:
    """Minimal partitioned dimensions with 2 junction slots."""
    return PartitionedDimensions(
        n_main=10,
        max_junctions=2,
        max_cross_junctions=0,
        max_double_crossovers=0,
        n_straights_16=10, n_straights_24=0,
        boundary_min_x=-100.0,
        boundary_max_x=100.0,
        boundary_min_y=-100.0,
        boundary_max_y=100.0,
    )


class TestChangeHandedness:
    def test_stays_within_template_bounds(self, dims):
        """_change_handedness must only produce values in [0, len(TEMPLATES) - 1]."""
        np.random.seed(42)
        x = create_empty_chromosome(dims)
        set_junction(x, dims, 0, active=1, position=0, handedness=0, n_straights=0)
        set_junction(x, dims, 1, active=1, position=0, handedness=0, n_straights=0)

        seen = set()
        for _ in range(1000):
            _change_handedness(x, dims)
            for slot in range(dims.max_junctions):
                base = dims.junc_start + slot * GENES_PER_JUNCTION
                seen.add(int(x[base + 2]))

        max_valid = len(TEMPLATES) - 1
        out_of_bounds = seen - set(range(len(TEMPLATES)))
        assert not out_of_bounds, (
            f"handedness values {out_of_bounds} exceed declared xu={max_valid}"
        )


@pytest.fixture
def cat() -> TrackCatalog:
    return TrackCatalog.load("data/track_pieces_v2.yaml")


@pytest.fixture
def big_dims() -> PartitionedDimensions:
    """Dims large enough to hold a self-intersecting hand-built layout."""
    return PartitionedDimensions(
        n_main=64,
        max_junctions=0,
        max_cross_junctions=0,
        max_double_crossovers=0,
        n_straights_16=40, n_straights_24=0,
        boundary_min_x=-200.0,
        boundary_max_x=200.0,
        boundary_min_y=-200.0,
        boundary_max_y=200.0,
    )


class TestStraightenNearUnresolvedCrossing:
    def test_no_catalog_falls_through_safely(self, big_dims):
        """Without a catalog the op cannot decode FK; it must not crash."""
        x = create_empty_chromosome(big_dims)
        x[0:10] = STRAIGHT_16
        np.random.seed(0)
        _straighten_near_unresolved_crossing(x, big_dims, catalog=None)
        # _mutate_piece_type may or may not have changed something — we just
        # require no exception and bounds preserved.
        assert all(int(v) in (-1, 0, 1, 2, 3) for v in x[: big_dims.n_main])

    def test_no_active_pieces_no_op(self, cat, big_dims):
        """Empty active main loop: must not crash and chromosome unchanged."""
        x = create_empty_chromosome(big_dims)
        before = x.copy()
        _straighten_near_unresolved_crossing(x, big_dims, catalog=cat)
        # _mutate_piece_type is the fallback; with all-INACTIVE it also no-ops.
        assert np.array_equal(x, before)

    def test_curve_crossing_gets_straightened(self, cat, big_dims):
        """A R40-on-STR_16 crossing must trigger a curve->STRAIGHT_16 swap.

        Builds a closed CW spiral [STR16*6, R40_R*4, STR16*2, R40_R*4,
        STR16*3, R40_R*4, STR16*5] -- known to produce a 90 deg STR-on-STR
        crossing. We then poison slot 27 to R40_CURVE so one of the crossing
        pieces becomes a curve; the operator must swap it back to STRAIGHT_16.
        """
        spiral = (
            [int(STRAIGHT_16)] * 6 + [int(R40_CURVE)] * 4
            + [int(STRAIGHT_16)] * 2 + [int(R40_CURVE)] * 4
            + [int(STRAIGHT_16)] * 3 + [int(R40_CURVE)] * 4
            + [int(STRAIGHT_16)] * 5
        )
        x = create_empty_chromosome(big_dims)
        x[: len(spiral)] = spiral
        # Poison slot 27 (one of the crossing pieces) with a curve.
        x[27] = int(R40_CURVE)
        # Without poisoning the other side, slot 0 is still STRAIGHT_16, so
        # the crossing pair (0, 27) is now STR_16 vs R40_CURVE -> unresolved.

        np.random.seed(0)
        _straighten_near_unresolved_crossing(x, big_dims, catalog=cat)

        # The op must replace the curve side with STRAIGHT_16, restoring
        # both crossing slots to STR_16 so the next decode pass can repair.
        assert int(x[27]) == int(STRAIGHT_16), (
            f"curve at slot 27 should have been straightened; got {int(x[27])}"
        )


class TestStickyCross90:
    """CROSS_90 slots placed by the decoder repair must not be churned away
    by mutation. Sticky behaviour means partial fixes accumulate across
    generations rather than each generation re-randomising the placements.
    """

    def test_mutate_piece_type_skips_cross90(self, big_dims):
        """_mutate_piece_type must never overwrite a CROSS_90 slot."""
        x = create_empty_chromosome(big_dims)
        # Layout: [STR16, STR16, CROSS_90, STR16, STR16].
        x[0:5] = [STRAIGHT_16, STRAIGHT_16, int(CROSS_90),
                  STRAIGHT_16, STRAIGHT_16]
        np.random.seed(0)
        for _ in range(200):
            _mutate_piece_type(x, big_dims)
            assert int(x[2]) == int(CROSS_90), (
                "CROSS_90 at slot 2 was overwritten by _mutate_piece_type"
            )

    def test_deactivate_skips_cross90(self, big_dims):
        """_deactivate_position must never deactivate a CROSS_90 slot."""
        x = create_empty_chromosome(big_dims)
        # Need >=5 active slots so the operator's "minimum 4" guard
        # doesn't no-op every call.
        x[0:8] = [STRAIGHT_16, STRAIGHT_16, int(CROSS_90), STRAIGHT_16,
                  STRAIGHT_16, STRAIGHT_16, STRAIGHT_16, STRAIGHT_16]
        np.random.seed(0)
        for _ in range(200):
            _deactivate_position(x, big_dims)
            assert int(x[2]) == int(CROSS_90), (
                "CROSS_90 at slot 2 was deactivated by _deactivate_position"
            )

    def test_swap_positions_does_not_move_cross90(self, big_dims):
        """_swap_positions must skip when either picked slot is CROSS_90,
        so the cross stays at its original world location (where its
        decoder-placed perpendicular partner lives)."""
        x = create_empty_chromosome(big_dims)
        x[0:5] = [STRAIGHT_16, STRAIGHT_16, int(CROSS_90),
                  STRAIGHT_16, STRAIGHT_16]
        np.random.seed(0)
        for _ in range(200):
            _swap_positions(x, big_dims)
            # CROSS_90 must remain at slot 2.
            cross_positions = [i for i in range(5) if int(x[i]) == int(CROSS_90)]
            assert cross_positions == [2], (
                f"CROSS_90 moved to {cross_positions}, expected [2]"
            )
