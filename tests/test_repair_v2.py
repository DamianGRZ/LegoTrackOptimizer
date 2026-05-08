"""Tests for src_v2/repair.py — port-pair repair pipeline."""

from pathlib import Path

import pytest

from src_v2.catalog import TrackCatalog
from src_v2.encoding import (
    INACTIVE,
    PortPairDimensions,
    create_empty_chromosome,
    get_port_pair,
    iter_active_pairs,
    iter_active_slots,
    set_piece_slot,
    set_port_pair,
)
from src_v2.repair import PortPairRepairPipeline


P_STRAIGHT_16 = 0
P_R40_LEFT = 2
P_SWITCH_LEFT_IN = 5
PA, PB, PC, PD = 0, 1, 2, 3


@pytest.fixture(scope="module")
def catalog():
    return TrackCatalog.load(Path("data/track_pieces_v2.yaml"))


@pytest.fixture
def small_dims():
    return PortPairDimensions(N_max=16, E_max=16)


@pytest.fixture
def repair(small_dims, catalog):
    inventory = {
        "STRAIGHT_16": 8,
        "R40_LEFT": 8,
        "R40_RIGHT": 0,
        "R40_SWITCH_LEFT_IN": 1,
        "R40_SWITCH_LEFT_OUT": 1,
        "R40_SWITCH_RIGHT_IN": 0,
        "R40_SWITCH_RIGHT_OUT": 0,
        "CROSS_90": 0,
        "STRAIGHT_24": 0,
        "DOUBLE_CROSSOVER": 0,
    }
    return PortPairRepairPipeline(small_dims, catalog, inventory)


# =============================================================================
# Identity on valid input
# =============================================================================

class TestRepairIdentity:
    def test_empty_chromosome_unchanged(self, repair, small_dims):
        import numpy as np

        x = create_empty_chromosome(small_dims)
        x_before = x.copy()
        repair._repair_one(x)
        assert np.array_equal(x, x_before)

    def test_valid_chromosome_unchanged(self, repair, small_dims):
        import numpy as np

        x = create_empty_chromosome(small_dims)
        for k in range(8):
            set_piece_slot(x, small_dims, k, P_R40_LEFT)
        for k in range(8):
            set_port_pair(x, small_dims, k, k, PB, (k + 1) % 8, PA)
        x_before = x.copy()
        repair._repair_one(x)
        assert np.array_equal(x, x_before)


# =============================================================================
# Edge sanitization
# =============================================================================

class TestEdgeSanitization:
    def test_self_loop_removed(self, repair, small_dims):
        x = create_empty_chromosome(small_dims)
        set_piece_slot(x, small_dims, 0, P_R40_LEFT)
        set_port_pair(x, small_dims, 0, 0, PA, 0, PB)
        repair._repair_one(x)
        assert get_port_pair(x, small_dims, 0) == (INACTIVE, INACTIVE, INACTIVE, INACTIVE)

    def test_double_booked_dropped_first_wins(self, repair, small_dims):
        x = create_empty_chromosome(small_dims)
        for k in range(3):
            set_piece_slot(x, small_dims, k, P_R40_LEFT)
        set_port_pair(x, small_dims, 0, 0, PB, 1, PA)
        set_port_pair(x, small_dims, 1, 0, PB, 2, PA)  # reuses (0, B)
        repair._repair_one(x)
        # First edge survives; second cleared
        assert get_port_pair(x, small_dims, 0) == (0, PB, 1, PA)
        assert get_port_pair(x, small_dims, 1) == (INACTIVE, INACTIVE, INACTIVE, INACTIVE)

    def test_edge_to_inactive_slot_dropped(self, repair, small_dims):
        x = create_empty_chromosome(small_dims)
        set_piece_slot(x, small_dims, 0, P_R40_LEFT)
        # Slot 5 is inactive; edge references it
        set_port_pair(x, small_dims, 0, 0, PB, 5, PA)
        repair._repair_one(x)
        assert get_port_pair(x, small_dims, 0) == (INACTIVE, INACTIVE, INACTIVE, INACTIVE)

    def test_out_of_range_port_dropped(self, repair, small_dims):
        x = create_empty_chromosome(small_dims)
        set_piece_slot(x, small_dims, 0, P_R40_LEFT)  # only ports A, B
        set_piece_slot(x, small_dims, 1, P_R40_LEFT)
        set_port_pair(x, small_dims, 0, 0, PC, 1, PA)  # port C invalid for R40
        repair._repair_one(x)
        assert get_port_pair(x, small_dims, 0) == (INACTIVE, INACTIVE, INACTIVE, INACTIVE)

    def test_partial_inactive_normalized(self, repair, small_dims):
        x = create_empty_chromosome(small_dims)
        set_piece_slot(x, small_dims, 0, P_R40_LEFT)
        set_piece_slot(x, small_dims, 1, P_R40_LEFT)
        # Partial-INACTIVE row: port_a is INACTIVE but others are valid
        set_port_pair(x, small_dims, 0, 0, INACTIVE, 1, PA)
        repair._repair_one(x)
        assert get_port_pair(x, small_dims, 0) == (INACTIVE, INACTIVE, INACTIVE, INACTIVE)

    def test_switch_port_c_allowed(self, repair, small_dims):
        x = create_empty_chromosome(small_dims)
        set_piece_slot(x, small_dims, 0, P_SWITCH_LEFT_IN)
        set_piece_slot(x, small_dims, 1, P_R40_LEFT)
        # Switch has 3 ports A/B/C, so port C is valid
        set_port_pair(x, small_dims, 0, 0, PC, 1, PA)
        repair._repair_one(x)
        assert get_port_pair(x, small_dims, 0) == (0, PC, 1, PA)


# =============================================================================
# Inventory enforcement
# =============================================================================

class TestInventoryEnforcement:
    def test_no_excess_unchanged(self, repair, small_dims):
        x = create_empty_chromosome(small_dims)
        for k in range(8):
            set_piece_slot(x, small_dims, k, P_R40_LEFT)  # exactly the limit
        repair._repair_one(x)
        active_count = sum(1 for _ in iter_active_slots(x, small_dims))
        assert active_count == 8

    def test_excess_deactivated_from_end(self, repair, small_dims):
        x = create_empty_chromosome(small_dims)
        # Place 12 R40 LEFT but inventory allows only 8
        for k in range(12):
            set_piece_slot(x, small_dims, k, P_R40_LEFT)
        repair._repair_one(x)
        active = list(iter_active_slots(x, small_dims))
        assert len(active) == 8
        # First 8 slots survived; last 4 deactivated
        active_slots = {idx for idx, _ in active}
        assert active_slots == {0, 1, 2, 3, 4, 5, 6, 7}

    def test_zero_inventory_piece_fully_dropped(self, repair, small_dims):
        x = create_empty_chromosome(small_dims)
        # R40_RIGHT inventory is 0; place one — should be dropped
        set_piece_slot(x, small_dims, 0, 3)  # R40_RIGHT index = 3
        repair._repair_one(x)
        assert int(x[0]) == INACTIVE


# =============================================================================
# Cascade — deactivating a slot drops edges referencing it
# =============================================================================

class TestCascade:
    def test_inventory_violation_cascades_to_edges(self, repair, small_dims):
        x = create_empty_chromosome(small_dims)
        # Place 12 R40 LEFT (violates inventory of 8) and edges between them
        for k in range(12):
            set_piece_slot(x, small_dims, k, P_R40_LEFT)
        for k in range(11):
            set_port_pair(x, small_dims, k, k, PB, k + 1, PA)
        repair._repair_one(x)
        # Slots 8-11 deactivated; edges referencing them must be dropped
        for k, sa, pa, sb, pb in iter_active_pairs(x, small_dims):
            assert sa < 8 and sb < 8, (
                f"Edge {k} survives but references deactivated slot: "
                f"({sa}, {pa}, {sb}, {pb})"
            )
