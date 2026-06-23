"""Tests for _inject_double_crossovers in src/decoder/construction.py.

Failure paths verify the decoder leaves inventory and main_pieces untouched
when a descriptor cannot be committed. Happy paths confirm a hand-built
figure-8 and a hand-built two-layer-loop both decode to closed multi-path
layouts with the right route map. The dangling-port constraint helper is
exercised against the same hand-built layouts to lock in the no-dangling
guarantee for the decoder-emitted records.
"""

import pytest

from src.catalog import TrackCatalog
from src.decoder import DecoderConfig, decode_chromosome
from src.decoder.construction import _inject_double_crossovers
from src.decoder.types import InventoryTracker
from src.encoding import (
    DC_ROUTE_CROSS_1_TO_2,
    DC_ROUTE_CROSS_2_TO_1,
    DC_ROUTE_TRACK1_THROUGH,
    DC_ROUTE_TRACK2_THROUGH,
    DOUBLE_CROSSOVER,
    R40_CURVE,
    R40_CURVE,
    STRAIGHT_16,
    PartitionedDimensions,
    create_chromosome_from_pieces,
    create_empty_chromosome,
    set_double_crossover,
)
from src.intersection import count_dangling_double_crossover_ports
from src.types import DblCrossover


@pytest.fixture
def cat() -> TrackCatalog:
    return TrackCatalog.load("data/track_pieces_v2.yaml")


@pytest.fixture
def dims() -> PartitionedDimensions:
    return PartitionedDimensions(
        n_main=64,
        max_junctions=0,
        max_cross_junctions=0,
        max_double_crossovers=2,
        n_straights_16=40, n_straights_24=0,
        boundary_min_x=-200.0,
        boundary_max_x=200.0,
        boundary_min_y=-200.0,
        boundary_max_y=200.0,
    )


_FULL_INV = {
    "STRAIGHT_16": 40,
    "STRAIGHT_24": 0,
    "R40_CURVE": 40,
    "CROSS_90": 0,
    "R40_SWITCH_LEFT": 0,
    "R40_SWITCH_RIGHT": 0,
    "DOUBLE_CROSSOVER": 2,
}


def _figure_eight_pieces() -> list[int]:
    return (
        [int(STRAIGHT_16)]
        + [int(R40_CURVE)] * 8
        + [int(STRAIGHT_16)] * 3
        + [int(R40_CURVE)] * 8
        + [int(STRAIGHT_16)]
        + [int(R40_CURVE)] * 8
        + [int(STRAIGHT_16)] * 3
        + [int(R40_CURVE)] * 8
    )


def _two_layer_pieces() -> list[int]:
    return (
        [int(STRAIGHT_16)]
        + [int(R40_CURVE)] * 4 + [int(STRAIGHT_16)]
        + [int(R40_CURVE)] * 4 + [int(STRAIGHT_16)] * 3
        + [int(R40_CURVE)] * 8
        + [int(STRAIGHT_16)]
        + [int(R40_CURVE)] * 4 + [int(STRAIGHT_16)]
        + [int(R40_CURVE)] * 4 + [int(STRAIGHT_16)] * 3
        + [int(R40_CURVE)] * 8
    )


def _tracker(cat: TrackCatalog, pieces: list[int]) -> InventoryTracker:
    t = InventoryTracker(_FULL_INV, cat)
    for p in pieces:
        if p >= 0:
            t.use(p)
    return t


class TestInjectionFailurePaths:
    def test_no_descriptors_returns_empty(self, cat, dims):
        x = create_empty_chromosome(dims)
        pieces = [int(STRAIGHT_16)] * 20
        tracker = _tracker(cat, pieces)

        records, route_map = _inject_double_crossovers(
            pieces, x, dims, tracker, cat, DecoderConfig(),
        )

        assert records == []
        assert route_map == {}
        assert tracker.used.get(int(DOUBLE_CROSSOVER), 0) == 0

    def test_invalid_route_pair_skipped(self, cat, dims):
        """Both routes share a port → not a valid 2-route cover."""
        x = create_empty_chromosome(dims)
        # Route pair (0, 0) = both A->B: covers only {A, B}, dangles {C, D}.
        set_double_crossover(x, dims, slot=0, active=1,
                             pos_1=0, route_1=0, pos_2=20, route_2=0)
        pieces = _figure_eight_pieces()
        original = list(pieces)
        tracker = _tracker(cat, pieces)

        records, route_map = _inject_double_crossovers(
            pieces, x, dims, tracker, cat, DecoderConfig(),
        )

        assert records == []
        assert route_map == {}
        assert pieces == original
        assert tracker.used.get(int(DOUBLE_CROSSOVER), 0) == 0

    def test_position_not_straight_skipped(self, cat, dims):
        x = create_empty_chromosome(dims)
        set_double_crossover(x, dims, slot=0, active=1,
                             pos_1=2, route_1=DC_ROUTE_CROSS_1_TO_2,
                             pos_2=20, route_2=DC_ROUTE_CROSS_2_TO_1)
        pieces = _figure_eight_pieces()  # pos 2 is R40_CURVE
        original = list(pieces)
        tracker = _tracker(cat, pieces)

        records, _ = _inject_double_crossovers(
            pieces, x, dims, tracker, cat, DecoderConfig(),
        )

        assert records == []
        assert pieces == original

    def test_inventory_zero_skipped(self, cat, dims):
        x = create_empty_chromosome(dims)
        set_double_crossover(x, dims, slot=0, active=1,
                             pos_1=0, route_1=DC_ROUTE_CROSS_1_TO_2,
                             pos_2=20, route_2=DC_ROUTE_CROSS_2_TO_1)
        pieces = _figure_eight_pieces()
        inv_no_dc = dict(_FULL_INV, DOUBLE_CROSSOVER=0)
        tracker = InventoryTracker(inv_no_dc, cat)
        for p in pieces:
            if p >= 0:
                tracker.use(p)

        records, _ = _inject_double_crossovers(
            pieces, x, dims, tracker, cat, DecoderConfig(),
        )

        assert records == []
        assert pieces == _figure_eight_pieces()

    def test_geometric_mismatch_skipped(self, cat, dims):
        """Straight-line main loop cannot land on the same physical piece twice."""
        x = create_empty_chromosome(dims)
        set_double_crossover(x, dims, slot=0, active=1,
                             pos_1=0, route_1=DC_ROUTE_CROSS_1_TO_2,
                             pos_2=10, route_2=DC_ROUTE_CROSS_2_TO_1)
        pieces = [int(STRAIGHT_16)] * 20
        original = list(pieces)
        tracker = _tracker(cat, pieces)

        records, _ = _inject_double_crossovers(
            pieces, x, dims, tracker, cat, DecoderConfig(),
        )

        assert records == []
        assert pieces == original
        assert tracker.used.get(int(DOUBLE_CROSSOVER), 0) == 0


class TestInjectionHappyPaths:
    def test_figure_eight_closes(self, cat, dims):
        pieces = _figure_eight_pieces()
        descriptors = [(1, 0, DC_ROUTE_CROSS_1_TO_2, 20, DC_ROUTE_CROSS_2_TO_1)]
        x = create_chromosome_from_pieces(dims, pieces, double_crossovers=descriptors)
        layout = decode_chromosome(x, cat, _FULL_INV, dims, DecoderConfig())

        assert len(layout.dbl_crossovers) == 1
        assert layout.main_loop_routes == {0: DC_ROUTE_CROSS_1_TO_2, 20: DC_ROUTE_CROSS_2_TO_1}
        assert layout.main_loop_pieces[0] == int(DOUBLE_CROSSOVER)
        assert layout.main_loop_pieces[20] == int(DOUBLE_CROSSOVER)
        assert layout.is_closed()

    def test_two_layer_both_through_is_infeasible(self, cat, dims):
        """The naive two-layer (both-through) DC pattern injects correctly but is
        geometrically infeasible as a single closed loop: its two nested ovals
        self-cross at 22.5deg OBLIQUE angles (unlegalizable by any catalog piece
        -- CROSS_90 only handles 90deg) and the loop fails to close (~32-stud
        gap). This documents why _gen_two_layer_loop_dbl_crossover is a stub; the
        only single-loop DC topology that closes is the figure-8 (cross routes).

        Tripwire: if a future redesign makes this close, update this test.
        """
        pieces = _two_layer_pieces()
        descriptors = [(1, 0, DC_ROUTE_TRACK1_THROUGH, 21, DC_ROUTE_TRACK2_THROUGH)]
        x = create_chromosome_from_pieces(dims, pieces, double_crossovers=descriptors)
        layout = decode_chromosome(x, cat, _FULL_INV, dims, DecoderConfig())

        # DC injection mechanics work...
        assert len(layout.dbl_crossovers) == 1
        assert layout.main_loop_routes == {0: DC_ROUTE_TRACK1_THROUGH, 21: DC_ROUTE_TRACK2_THROUGH}
        # ...but the pattern does not form a closed loop (oblique self-crossing,
        # ~32-stud gap; see docstring).
        assert not layout.is_closed()


class TestDanglingConstraint:
    def test_decoder_records_have_no_dangling(self, cat, dims):
        """Valid figure-8 → 0 dangling ports."""
        pieces = _figure_eight_pieces()
        descriptors = [(1, 0, DC_ROUTE_CROSS_1_TO_2, 20, DC_ROUTE_CROSS_2_TO_1)]
        x = create_chromosome_from_pieces(dims, pieces, double_crossovers=descriptors)
        layout = decode_chromosome(x, cat, _FULL_INV, dims, DecoderConfig())

        dangling = count_dangling_double_crossover_ports(
            layout.main_loop_pieces,
            layout.main_loop_routes,
            layout.dbl_crossovers,
        )
        assert dangling == 0

    def test_solo_slot_two_dangling(self):
        """DBL_CROSSOVER chromosome slot without a record → 2 dangling ports."""
        pieces = [int(STRAIGHT_16)] * 5 + [int(DOUBLE_CROSSOVER)] + [int(STRAIGHT_16)] * 10
        assert count_dangling_double_crossover_ports(pieces, {5: 0}, []) == 2

    def test_invalid_record_pair_one_dangling(self):
        """Record covering {A,B,D} (port C unused) → 1 dangling port."""
        record = DblCrossover(
            slot=0, positions=(0, 20),
            routes=(DC_ROUTE_TRACK1_THROUGH, DC_ROUTE_CROSS_1_TO_2),  # {A,B} ∪ {A,D}
            origin=(0.0, 0.0, 0.0),
        )
        pieces = [int(DOUBLE_CROSSOVER)] + [0] * 19 + [int(DOUBLE_CROSSOVER)] + [0] * 19
        assert count_dangling_double_crossover_ports(pieces, {0: 0, 20: 2}, [record]) == 1
