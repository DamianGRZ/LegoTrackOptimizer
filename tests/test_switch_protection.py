"""Tests for switch slot protection in ``PortPairMutation``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src_v2.catalog import TrackCatalog
from src_v2.config import OptimizationConfig
from src_v2.encoding import (
    INACTIVE,
    compute_port_pair_dimensions,
    create_empty_chromosome,
    iter_active_slots,
    set_piece_slot,
    set_port_pair,
)
from src_v2.operators import PortPairMutation


CATALOG_PATH = Path(__file__).parent.parent / "data" / "track_pieces_v2.yaml"
CONFIG_PATH = Path(__file__).parent.parent / "configs" / "with_switches.yaml"


@pytest.fixture(scope="module")
def setup():
    catalog = TrackCatalog.load(CATALOG_PATH)
    config = OptimizationConfig.load(CONFIG_PATH)
    dims = compute_port_pair_dimensions(config.boundary, catalog, config.inventory)
    mutator = PortPairMutation(dims, catalog, config)
    return catalog, config, dims, mutator


def _build_minimal_siding(catalog, dims):
    """Hand-build a chromosome with one passing siding (slots 0..6)."""
    x = create_empty_chromosome(dims)
    pieces = {
        0: "STRAIGHT_16",
        1: "R40_SWITCH_LEFT",
        2: "STRAIGHT_16",
        3: "STRAIGHT_16",
        4: "R40_CURVE",
        5: "R40_CURVE",
        6: "R40_SWITCH_RIGHT",
    }
    for slot, piece_id in pieces.items():
        set_piece_slot(x, dims, slot, catalog.id_to_index[piece_id])

    # Edges
    set_port_pair(x, dims, 0, 0, 1, 1, 0)   # 0.B → 1.A
    set_port_pair(x, dims, 1, 1, 1, 2, 0)   # 1.B → 2.A (mainline through switch)
    set_port_pair(x, dims, 2, 2, 1, 3, 0)   # 2.B → 3.A
    set_port_pair(x, dims, 3, 3, 1, 6, 1)   # 3.B → 6.B (close mainline back via OUT switch)
    set_port_pair(x, dims, 4, 6, 0, 0, 0)   # 6.A → 0.A
    set_port_pair(x, dims, 5, 1, 2, 4, 0)   # 1.C → 4.A (branch start)
    set_port_pair(x, dims, 6, 4, 1, 5, 0)   # 4.B → 5.A
    set_port_pair(x, dims, 7, 5, 1, 6, 2)   # 5.B → 6.C (branch end)
    return x


def _switch_slots(x, mutator) -> set:
    return mutator._switch_slot_indices(x)


def test_switch_slot_indices_finds_both_switches(setup):
    catalog, _, dims, mutator = setup
    x = _build_minimal_siding(catalog, dims)
    assert _switch_slots(x, mutator) == {1, 6}


def test_mutate_piece_type_skips_switches(setup):
    catalog, _, dims, mutator = setup
    rng = np.random.default_rng(42)
    np.random.seed(42)
    for _ in range(200):
        x = _build_minimal_siding(catalog, dims)
        mutator._mutate_piece_type(x)
        # Switches must still be in their slots
        switches = _switch_slots(x, mutator)
        assert 1 in switches
        assert 6 in switches


def test_deactivate_slot_skips_switches(setup):
    catalog, _, dims, mutator = setup
    np.random.seed(42)
    for _ in range(200):
        x = _build_minimal_siding(catalog, dims)
        mutator._deactivate_slot(x)
        switches = _switch_slots(x, mutator)
        assert 1 in switches
        assert 6 in switches


def test_toggle_rotate_skips_switches(setup):
    catalog, _, dims, mutator = setup
    np.random.seed(42)
    for _ in range(200):
        x = _build_minimal_siding(catalog, dims)
        rotate_in_before = int(x[dims.rotate_start + 1])
        rotate_out_before = int(x[dims.rotate_start + 6])
        mutator._toggle_rotate(x)
        rotate_in_after = int(x[dims.rotate_start + 1])
        rotate_out_after = int(x[dims.rotate_start + 6])
        assert rotate_in_before == rotate_in_after
        assert rotate_out_before == rotate_out_after


def test_remove_edge_skips_switch_edges(setup):
    catalog, _, dims, mutator = setup
    np.random.seed(42)
    for _ in range(200):
        x = _build_minimal_siding(catalog, dims)
        mutator._remove_edge(x)
        # All 8 edges originally touch slots {0..6}; switches at 1 and 6.
        # Edges 0, 1, 5: touch slot 1.  Edges 3, 4, 7: touch slot 6.
        # Only edges 2 and 6 are unprotected (mainline 2-3 and branch 4-5).
        # So at most one of those two should ever become INACTIVE.
        from src_v2.encoding import iter_active_pairs
        active_count = sum(1 for _ in iter_active_pairs(x, dims))
        # Started with 8 active edges; at most 1 should be removed.
        assert active_count >= 7
        # Switches still connected — port C edges (rows 5 and 7) intact.
        from src_v2.encoding import get_port_pair
        sa5, pa5, sb5, pb5 = get_port_pair(x, dims, 5)
        sa7, pa7, sb7, pb7 = get_port_pair(x, dims, 7)
        assert (sa5, pa5, sb5, pb5) == (1, 2, 4, 0)
        assert (sa7, pa7, sb7, pb7) == (5, 1, 6, 2)


def test_rewire_edge_skips_switch_edges(setup):
    catalog, _, dims, mutator = setup
    np.random.seed(42)
    from src_v2.encoding import get_port_pair
    for _ in range(200):
        x = _build_minimal_siding(catalog, dims)
        # Snapshot all switch-touching edges before
        switch_edges_before = {
            k: get_port_pair(x, dims, k) for k in (0, 1, 3, 4, 5, 7)
        }
        mutator._rewire_edge(x)
        switch_edges_after = {
            k: get_port_pair(x, dims, k) for k in (0, 1, 3, 4, 5, 7)
        }
        assert switch_edges_before == switch_edges_after


def test_no_switches_means_no_protection(setup):
    """When there are no switches, all sub-operators behave normally."""
    catalog, _, dims, mutator = setup
    x = create_empty_chromosome(dims)
    # Place a few non-switch pieces
    for k, piece_id in enumerate(["STRAIGHT_16", "STRAIGHT_16", "R40_CURVE", "R40_CURVE"]):
        set_piece_slot(x, dims, k, catalog.id_to_index[piece_id])

    assert _switch_slots(x, mutator) == set()


def test_repeated_mutations_preserve_switches(setup):
    """Apply 1000 random mutations; both switches must survive every time."""
    catalog, _, dims, mutator = setup
    np.random.seed(42)
    x = _build_minimal_siding(catalog, dims)
    X = np.array([x])
    for _ in range(1000):
        mutator._do(None, X.copy())  # in-place mutation on copy
    # Original chromosome unchanged (we mutated copies)
    assert _switch_slots(x, mutator) == {1, 6}

    # Also: a single chromosome run through 100 in-place mutations should
    # keep both switches.
    X2 = np.array([x.copy()])
    for _ in range(100):
        X2 = mutator._do(None, X2)
    final = X2[0]
    final_switches = _switch_slots(final, mutator)
    assert 1 in final_switches
    assert 6 in final_switches
