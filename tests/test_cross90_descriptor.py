"""Unit tests for the DC-style CROSS_90 descriptor model.

CROSS_90 is modelled as one physical piece traversed twice by the loop (both
passes straight, intersecting at ~90 deg). The descriptor is (active, pos_1,
pos_2) and the block is dimensioned directly from CROSS_90 inventory — NOT from
switch counts (the retired 4-switch cross-junction model).
"""

import numpy as np
import pytest

from src.catalog import TrackCatalog
from src.config import OptimizationConfig
from src.encoding import (
    PartitionedDimensions,
    compute_dimensions,
    create_empty_chromosome,
    generate_bounds,
    get_active_cross_junctions,
    get_cross_junction,
    set_cross_junction,
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
        max_double_crossovers=0,
        n_straights_16=40, n_straights_24=0,
        boundary_min_x=-200.0,
        boundary_max_x=200.0,
        boundary_min_y=-200.0,
        boundary_max_y=200.0,
    )


class TestCrossJunctionEncoding:
    def test_both_position_genes_bounded_to_main_range(self, dims) -> None:
        """gene 1 = pos_1, gene 2 = pos_2 (was handedness {0,1})."""
        xl, xu = generate_bounds(dims)
        base = dims.cross_junc_start  # slot 0
        assert (xl[base + 1], xu[base + 1]) == (0, dims.n_main - 1)
        assert (xl[base + 2], xu[base + 2]) == (0, dims.n_main - 1)

    def test_set_get_roundtrip_positions(self, dims) -> None:
        x = create_empty_chromosome(dims)
        set_cross_junction(x, dims, slot=0, active=1, pos_1=5, pos_2=12)
        assert get_cross_junction(x, dims, 0) == (1, 5, 12)

    def test_active_descriptors_sorted_by_pos_1(self, dims) -> None:
        x = create_empty_chromosome(dims)
        set_cross_junction(x, dims, slot=0, active=1, pos_1=20, pos_2=40)
        set_cross_junction(x, dims, slot=1, active=1, pos_1=4, pos_2=8)
        active = get_active_cross_junctions(x, dims)
        assert [a[2] for a in active] == [4, 20]


class TestCrossJunctionType:
    def test_fields_mirror_dbl_crossover(self) -> None:
        from src.types import CrossJunction

        cj = CrossJunction(slot=2, positions=(3, 9), origin=(10.0, 20.0, 90.0))
        assert cj.slot == 2
        assert cj.positions == (3, 9)
        assert cj.origin == (10.0, 20.0, 90.0)
        assert cj.is_valid()

    def test_invalid_when_positions_equal_or_negative(self) -> None:
        from src.types import CrossJunction

        assert not CrossJunction(0, (5, 5), (0.0, 0.0, 0.0)).is_valid()
        assert not CrossJunction(0, (-1, 4), (0.0, 0.0, 0.0)).is_valid()


class TestCrossJunctionDimensioning:
    def test_slots_equal_cross90_inventory(self, cat: TrackCatalog) -> None:
        """max_cross_junctions = inv[CROSS_90], independent of switch counts.

        all_pieces has CROSS_90:2 with 3 LEFT + 3 RIGHT switches. The old
        4-switch model gave (3//4)+(3//4)=0 slots; the descriptor model gives 2.
        """
        cfg = OptimizationConfig.load("configs/all_pieces.yaml")
        dims = compute_dimensions(cfg, cat)
        assert dims.max_cross_junctions == cfg.inventory["CROSS_90"]
        assert dims.max_cross_junctions == 2
