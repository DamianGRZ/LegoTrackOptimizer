"""Tests for the grow-around-DC mutation (_grow_dc_figure_eight).

The DOUBLE_CROSSOVER figure-8 main loop is FK-closure-tuned, so raw edits break
it (which is why every other operator leaves DC loops alone). _grow_dc_figure_eight
instead REGENERATES the figure-8 one size larger via the validated builder, so it
must (a) keep the layout a feasible 2-DBLX figure-8 and (b) raise piece count.
"""

import numpy as np
import pytest

from src.catalog import TrackCatalog
from src.decoder import DecoderConfig, decode_chromosome
from src.encoding import (
    DC_ROUTE_CROSS_1_TO_2,
    DC_ROUTE_CROSS_2_TO_1,
    DOUBLE_CROSSOVER,
    R40_CURVE,
    STRAIGHT_16,
    PartitionedDimensions,
    create_chromosome_from_pieces,
)
from src.operators import _grow_dc_figure_eight
from src.sampling import _figure_eight_main_loop


@pytest.fixture
def cat() -> TrackCatalog:
    return TrackCatalog.load("data/track_pieces_v2.yaml")


@pytest.fixture
def dc_dims() -> PartitionedDimensions:
    """Dims big enough for a figure-8 to grow several sizes (1 DC slot)."""
    return PartitionedDimensions(
        n_main=120,
        max_junctions=0,
        max_cross_junctions=0,
        max_double_crossovers=1,
        n_straights_16=120, n_straights_24=0,
        boundary_min_x=-250.0,
        boundary_max_x=250.0,
        boundary_min_y=-250.0,
        boundary_max_y=250.0,
    )


def _base_figure_eight(dims, k=0):
    pieces, flips = _figure_eight_main_loop(k)
    pos2 = len(pieces) // 2
    dcd = [(1, 0, DC_ROUTE_CROSS_1_TO_2, pos2, DC_ROUTE_CROSS_2_TO_1)]
    return create_chromosome_from_pieces(
        dims, pieces, main_loop_flips=flips, double_crossovers=dcd,
    )


def _decode(x, cat, dims):
    inv = {"STRAIGHT_16": 120, "R40_CURVE": 80, "DOUBLE_CROSSOVER": 1}
    cfg = DecoderConfig(
        position_tolerance=4.0, angle_tolerance=5.0,
        boundary_min_x=-250.0, boundary_max_x=250.0,
        boundary_min_y=-250.0, boundary_max_y=250.0,
    )
    return decode_chromosome(x, cat, inv, dims=dims, config=cfg)


def _dblx_in_loop(layout):
    return sum(1 for p in layout.main_loop_pieces if p == int(DOUBLE_CROSSOVER))


class TestGrowDcFigureEight:
    def test_base_figure_eight_decodes_to_two_dblx(self, cat, dc_dims):
        """Sanity: the k=0 seed is a feasible single-loop figure-8 with 2 DBLX slots."""
        x = _base_figure_eight(dc_dims, k=0)
        lay = _decode(x, cat, dc_dims)
        assert lay.n_dbl_crossovers == 1
        assert _dblx_in_loop(lay) == 2
        assert lay.max_closure_error < 4.0

    def test_grow_increases_pieces_and_keeps_dc(self, cat, dc_dims):
        """Growing regenerates the figure-8 one size up: still 2 DBLX, still closed, MORE pieces."""
        x = _base_figure_eight(dc_dims, k=0)
        before = _decode(x, cat, dc_dims)
        n0 = before.n_pieces

        _grow_dc_figure_eight(x, dc_dims)

        after = _decode(x, cat, dc_dims)
        assert after.n_dbl_crossovers == 1, "grow must preserve the double-crossover"
        assert _dblx_in_loop(after) == 2, "both DBLX traversals must remain"
        assert after.max_closure_error < 4.0, "grown figure-8 must still close"
        assert after.n_pieces > n0, f"grow must raise piece count ({n0} -> {after.n_pieces})"

    def test_grow_is_noop_without_dc(self, dc_dims):
        """No active DC -> the operator must not touch the chromosome."""
        pieces = [int(R40_CURVE)] * 8 + [int(STRAIGHT_16)] * 6 + [int(R40_CURVE)] * 8 + [int(STRAIGHT_16)] * 6
        x = create_chromosome_from_pieces(dc_dims, pieces)
        snapshot = x.copy()
        _grow_dc_figure_eight(x, dc_dims)
        assert np.array_equal(x, snapshot)

    def test_grow_stops_at_boundary(self, cat):
        """When the next size won't fit the boundary, grow is a safe no-op."""
        tight = PartitionedDimensions(
            n_main=120, max_junctions=0, max_cross_junctions=0, max_double_crossovers=1,
            n_straights_16=120, n_straights_24=0,
            boundary_min_x=-70.0, boundary_max_x=70.0,   # width 140: fits k=0 (128) not k=1 (160)
            boundary_min_y=-110.0, boundary_max_y=110.0,  # height 220 >= 200
        )
        x = _base_figure_eight(tight, k=0)
        snapshot = x.copy()
        _grow_dc_figure_eight(x, tight)
        assert np.array_equal(x, snapshot), "must not grow past the boundary"
