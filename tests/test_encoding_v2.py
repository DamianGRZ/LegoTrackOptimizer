"""Tests for src_v2/encoding.py — port-pair chromosome encoding."""

from pathlib import Path

import numpy as np
import pytest

from src_v2.catalog import TrackCatalog
from src_v2.config import BoundaryConfig
from src_v2.encoding import (
    ANCHOR_GENES,
    ANCHOR_THETA_OFFSET,
    ANCHOR_X_OFFSET,
    ANCHOR_Y_OFFSET,
    DTYPE,
    GENES_PER_PAIR,
    INACTIVE,
    MAX_PORT_IDX,
    PortPairDimensions,
    chromosome_stats,
    clear_port_pair,
    compute_port_pair_dimensions,
    create_empty_chromosome,
    generate_bounds,
    get_anchor,
    get_piece_slot,
    get_port_pair,
    iter_active_pairs,
    iter_active_slots,
    set_anchor,
    set_piece_slot,
    set_port_pair,
    validate_chromosome,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def boundary_200():
    return BoundaryConfig(min_x=-100.0, max_x=100.0, min_y=-100.0, max_y=100.0)


@pytest.fixture
def boundary_300():
    return BoundaryConfig(min_x=-150.0, max_x=150.0, min_y=-150.0, max_y=150.0)


@pytest.fixture
def boundary_100():
    return BoundaryConfig(min_x=-50.0, max_x=50.0, min_y=-50.0, max_y=50.0)


@pytest.fixture(scope="module")
def catalog():
    return TrackCatalog.load(Path("data/track_pieces.yaml"))


# =============================================================================
# PortPairDimensions
# =============================================================================

class TestPortPairDimensions:
    def test_offsets_consistent(self):
        dims = PortPairDimensions(N_max=10, E_max=5)
        assert dims.slot_start == 0
        assert dims.slot_end == 10
        assert dims.pair_start == 10
        assert dims.pair_end == 10 + GENES_PER_PAIR * 5
        assert dims.anchor_start == dims.pair_end
        assert dims.n_var == dims.anchor_start + ANCHOR_GENES

    def test_n_var_formula(self):
        dims = PortPairDimensions(N_max=80, E_max=96)
        assert dims.n_var == 80 + 4 * 96 + 3

    def test_minimal_dims(self):
        dims = PortPairDimensions(N_max=1, E_max=1)
        assert dims.n_var == 1 + 4 + 3


# =============================================================================
# compute_port_pair_dimensions
# =============================================================================

class TestComputeDimensions:
    def test_200x200_unconstrained(self, boundary_200, catalog):
        dims = compute_port_pair_dimensions(boundary_200, catalog)
        # Plan §4: N_max ≈ 62 for 200×200; allow ±10 ranged check
        assert 50 <= dims.N_max <= 80, f"N_max={dims.N_max} outside expected band"

    def test_inventory_binds_when_smaller(self, boundary_300, catalog):
        # Large box, small inventory → inventory cap is the binder
        inventory = {"R40_LEFT": 30, "STRAIGHT_16": 30}
        dims = compute_port_pair_dimensions(boundary_300, catalog, inventory)
        assert dims.N_max == 60

    def test_geometric_binds_when_smaller(self, boundary_200, catalog):
        # Tiny box would bind geometrically; 200×200 is just generous enough
        # that inventory=1000 still hits geometric cap
        inventory = {"R40_LEFT": 1000}
        dims = compute_port_pair_dimensions(boundary_200, catalog, inventory)
        assert dims.N_max < 100

    def test_n_var_in_expected_range_200(self, boundary_200, catalog):
        dims = compute_port_pair_dimensions(boundary_200, catalog)
        # Plan §4: 200×200 produces n_var in 350-470 range
        assert 200 <= dims.n_var <= 600

    def test_e_max_scales_with_n_max(self, boundary_200, catalog):
        small = compute_port_pair_dimensions(
            boundary_200, catalog, {"R40_LEFT": 16},
        )
        large = compute_port_pair_dimensions(
            boundary_200, catalog, {"R40_LEFT": 60},
        )
        assert large.E_max > small.E_max

    def test_tighter_margin_grows_n_max(self, boundary_200, catalog):
        tight_margin = compute_port_pair_dimensions(
            boundary_200, catalog, edge_margin_studs=10.0,
        )
        wide_margin = compute_port_pair_dimensions(
            boundary_200, catalog, edge_margin_studs=40.0,
        )
        assert tight_margin.N_max > wide_margin.N_max

    def test_higher_branch_factor_grows_n_max(self, boundary_200, catalog):
        sparse = compute_port_pair_dimensions(
            boundary_200, catalog, branch_overlay_factor=1.0,
        )
        dense = compute_port_pair_dimensions(
            boundary_200, catalog, branch_overlay_factor=2.0,
        )
        assert dense.N_max > sparse.N_max

    def test_smaller_box_reduces_n_max(self, boundary_100, boundary_200, catalog):
        small_box = compute_port_pair_dimensions(boundary_100, catalog)
        large_box = compute_port_pair_dimensions(boundary_200, catalog)
        assert small_box.N_max < large_box.N_max

    def test_e_max_at_least_one(self, boundary_100, catalog):
        # Even for the smallest reasonable box, E_max should be ≥ 1
        dims = compute_port_pair_dimensions(boundary_100, catalog)
        assert dims.E_max >= 1


# =============================================================================
# generate_bounds
# =============================================================================

class TestGenerateBounds:
    def test_shape_matches_n_var(self, boundary_200):
        dims = PortPairDimensions(N_max=10, E_max=5)
        xl, xu = generate_bounds(dims, boundary_200, max_piece_id=9)
        assert xl.shape == (dims.n_var,)
        assert xu.shape == (dims.n_var,)

    def test_dtype_is_int16(self, boundary_200):
        dims = PortPairDimensions(N_max=10, E_max=5)
        xl, xu = generate_bounds(dims, boundary_200, max_piece_id=9)
        assert xl.dtype == DTYPE
        assert xu.dtype == DTYPE

    def test_slot_region_bounds(self, boundary_200):
        dims = PortPairDimensions(N_max=10, E_max=5)
        xl, xu = generate_bounds(dims, boundary_200, max_piece_id=9)
        assert (xl[dims.slot_start:dims.slot_end] == INACTIVE).all()
        assert (xu[dims.slot_start:dims.slot_end] == 9).all()

    def test_pair_slot_bounds(self, boundary_200):
        dims = PortPairDimensions(N_max=10, E_max=5)
        xl, xu = generate_bounds(dims, boundary_200, max_piece_id=9)
        for k in range(dims.E_max):
            base = dims.pair_start + k * GENES_PER_PAIR
            # slot_a, slot_b genes
            assert xl[base + 0] == INACTIVE
            assert xu[base + 0] == dims.N_max - 1
            assert xl[base + 2] == INACTIVE
            assert xu[base + 2] == dims.N_max - 1

    def test_pair_port_bounds(self, boundary_200):
        dims = PortPairDimensions(N_max=10, E_max=5)
        xl, xu = generate_bounds(dims, boundary_200, max_piece_id=9)
        for k in range(dims.E_max):
            base = dims.pair_start + k * GENES_PER_PAIR
            # port_a, port_b genes
            assert xl[base + 1] == INACTIVE
            assert xu[base + 1] == MAX_PORT_IDX
            assert xl[base + 3] == INACTIVE
            assert xu[base + 3] == MAX_PORT_IDX

    def test_anchor_bounds_match_boundary(self, boundary_200):
        dims = PortPairDimensions(N_max=10, E_max=5)
        xl, xu = generate_bounds(dims, boundary_200, max_piece_id=9)
        assert xl[dims.anchor_start + ANCHOR_X_OFFSET] == int(boundary_200.min_x)
        assert xu[dims.anchor_start + ANCHOR_X_OFFSET] == int(boundary_200.max_x)
        assert xl[dims.anchor_start + ANCHOR_Y_OFFSET] == int(boundary_200.min_y)
        assert xu[dims.anchor_start + ANCHOR_Y_OFFSET] == int(boundary_200.max_y)
        assert xl[dims.anchor_start + ANCHOR_THETA_OFFSET] == 0
        assert xu[dims.anchor_start + ANCHOR_THETA_OFFSET] == 359

    def test_xl_le_xu_everywhere(self, boundary_200):
        dims = PortPairDimensions(N_max=10, E_max=5)
        xl, xu = generate_bounds(dims, boundary_200, max_piece_id=9)
        assert (xl <= xu).all()


# =============================================================================
# Accessor round-trips
# =============================================================================

class TestPieceSlotAccessors:
    def test_roundtrip(self):
        dims = PortPairDimensions(N_max=10, E_max=5)
        x = create_empty_chromosome(dims)
        set_piece_slot(x, dims, 3, 7)
        assert get_piece_slot(x, dims, 3) == 7

    def test_iter_active_only_yields_active(self):
        dims = PortPairDimensions(N_max=10, E_max=5)
        x = create_empty_chromosome(dims)
        set_piece_slot(x, dims, 0, 2)
        set_piece_slot(x, dims, 5, 7)
        set_piece_slot(x, dims, 9, 0)
        active = list(iter_active_slots(x, dims))
        assert active == [(0, 2), (5, 7), (9, 0)]

    def test_iter_active_skips_inactive(self):
        dims = PortPairDimensions(N_max=10, E_max=5)
        x = create_empty_chromosome(dims)
        # All start INACTIVE
        active = list(iter_active_slots(x, dims))
        assert active == []


class TestPortPairAccessors:
    def test_roundtrip(self):
        dims = PortPairDimensions(N_max=10, E_max=5)
        x = create_empty_chromosome(dims)
        set_port_pair(x, dims, 2, slot_a=3, port_a=1, slot_b=4, port_b=0)
        assert get_port_pair(x, dims, 2) == (3, 1, 4, 0)

    def test_iter_active_pairs(self):
        dims = PortPairDimensions(N_max=10, E_max=5)
        x = create_empty_chromosome(dims)
        set_port_pair(x, dims, 1, 2, 1, 3, 0)
        set_port_pair(x, dims, 4, 5, 1, 6, 0)
        active = list(iter_active_pairs(x, dims))
        assert active == [(1, 2, 1, 3, 0), (4, 5, 1, 6, 0)]

    def test_clear_port_pair(self):
        dims = PortPairDimensions(N_max=10, E_max=5)
        x = create_empty_chromosome(dims)
        set_port_pair(x, dims, 0, 1, 2, 3, 0)
        clear_port_pair(x, dims, 0)
        assert get_port_pair(x, dims, 0) == (INACTIVE, INACTIVE, INACTIVE, INACTIVE)

    def test_partial_inactive_row_treated_as_inactive(self):
        # If any of the four genes is INACTIVE, the whole row is inactive.
        dims = PortPairDimensions(N_max=10, E_max=5)
        x = create_empty_chromosome(dims)
        set_port_pair(x, dims, 0, 1, INACTIVE, 3, 0)  # port_a INACTIVE
        active = list(iter_active_pairs(x, dims))
        assert active == []


class TestAnchorAccessors:
    def test_roundtrip(self):
        dims = PortPairDimensions(N_max=10, E_max=5)
        x = create_empty_chromosome(dims)
        set_anchor(x, dims, ax=10, ay=-20, atheta=180)
        assert get_anchor(x, dims) == (10, -20, 180)

    def test_default_anchor_is_zero(self):
        dims = PortPairDimensions(N_max=10, E_max=5)
        x = create_empty_chromosome(dims)
        assert get_anchor(x, dims) == (0, 0, 0)


# =============================================================================
# Empty chromosome
# =============================================================================

class TestEmptyChromosome:
    def test_all_slots_inactive(self):
        dims = PortPairDimensions(N_max=10, E_max=5)
        x = create_empty_chromosome(dims)
        assert (x[dims.slot_start:dims.slot_end] == INACTIVE).all()

    def test_all_pair_genes_inactive(self):
        dims = PortPairDimensions(N_max=10, E_max=5)
        x = create_empty_chromosome(dims)
        assert (x[dims.pair_start:dims.pair_end] == INACTIVE).all()

    def test_zero_active_count(self):
        dims = PortPairDimensions(N_max=10, E_max=5)
        x = create_empty_chromosome(dims)
        stats = chromosome_stats(x, dims)
        assert stats["n_active_slots"] == 0
        assert stats["n_active_pairs"] == 0

    def test_dtype_is_int16(self):
        dims = PortPairDimensions(N_max=10, E_max=5)
        x = create_empty_chromosome(dims)
        assert x.dtype == DTYPE


# =============================================================================
# Validation
# =============================================================================

class TestValidation:
    def test_empty_is_valid(self):
        dims = PortPairDimensions(N_max=10, E_max=5)
        x = create_empty_chromosome(dims)
        assert validate_chromosome(x, dims) == []

    def test_well_formed_chromosome_is_valid(self):
        dims = PortPairDimensions(N_max=10, E_max=5)
        x = create_empty_chromosome(dims)
        set_piece_slot(x, dims, 0, 2)
        set_piece_slot(x, dims, 1, 3)
        set_port_pair(x, dims, 0, 0, 1, 1, 0)
        assert validate_chromosome(x, dims) == []

    def test_wrong_length_caught(self):
        dims = PortPairDimensions(N_max=10, E_max=5)
        x = np.zeros(dims.n_var - 1, dtype=DTYPE)
        errors = validate_chromosome(x, dims)
        assert any("Length" in e for e in errors)

    def test_out_of_range_slot_in_pair_caught(self):
        dims = PortPairDimensions(N_max=5, E_max=3)
        x = create_empty_chromosome(dims)
        set_port_pair(x, dims, 0, slot_a=99, port_a=0, slot_b=2, port_b=1)
        errors = validate_chromosome(x, dims)
        assert any("slot_a" in e and "99" in e for e in errors)

    def test_out_of_range_port_in_pair_caught(self):
        dims = PortPairDimensions(N_max=5, E_max=3)
        x = create_empty_chromosome(dims)
        set_port_pair(x, dims, 0, slot_a=1, port_a=99, slot_b=2, port_b=1)
        errors = validate_chromosome(x, dims)
        assert any("port_a" in e and "99" in e for e in errors)

    def test_below_inactive_sentinel_caught(self):
        dims = PortPairDimensions(N_max=5, E_max=3)
        x = create_empty_chromosome(dims)
        x[0] = -5
        errors = validate_chromosome(x, dims)
        assert any("Slot[0]" in e for e in errors)


# =============================================================================
# Statistics
# =============================================================================

class TestChromosomeStats:
    def test_counts_active_slots_and_pairs(self):
        dims = PortPairDimensions(N_max=10, E_max=5)
        x = create_empty_chromosome(dims)
        set_piece_slot(x, dims, 0, 2)
        set_piece_slot(x, dims, 1, 3)
        set_piece_slot(x, dims, 2, 2)
        set_port_pair(x, dims, 0, 0, 1, 1, 0)
        set_port_pair(x, dims, 1, 1, 1, 2, 0)
        stats = chromosome_stats(x, dims)
        assert stats["n_active_slots"] == 3
        assert stats["n_active_pairs"] == 2
        assert stats["piece_counts"] == {2: 2, 3: 1}

    def test_anchor_in_stats(self):
        dims = PortPairDimensions(N_max=10, E_max=5)
        x = create_empty_chromosome(dims)
        set_anchor(x, dims, 5, -3, 90)
        stats = chromosome_stats(x, dims)
        assert stats["anchor"] == (5, -3, 90)


# =============================================================================
# Integration: real catalog + boundary
# =============================================================================

class TestIntegrationWithCatalog:
    def test_full_roundtrip_with_real_catalog(self, boundary_200, catalog):
        dims = compute_port_pair_dimensions(
            boundary_200, catalog, {"R40_LEFT": 16, "STRAIGHT_16": 16},
        )
        xl, xu = generate_bounds(dims, boundary_200, max_piece_id=catalog.n_pieces - 1)
        x = create_empty_chromosome(dims)
        # Lay down a 16-piece R40 LEFT cycle (just slots, edges, anchor)
        for k in range(16):
            set_piece_slot(x, dims, k, 2)  # R40_LEFT index = 2
        for k in range(16):
            set_port_pair(x, dims, k, k, 1, (k + 1) % 16, 0)
        set_anchor(x, dims, 0, 0, 0)

        errors = validate_chromosome(x, dims)
        assert errors == [], f"Unexpected errors: {errors}"

        stats = chromosome_stats(x, dims)
        assert stats["n_active_slots"] == 16
        assert stats["n_active_pairs"] == 16
        assert stats["piece_counts"] == {2: 16}
