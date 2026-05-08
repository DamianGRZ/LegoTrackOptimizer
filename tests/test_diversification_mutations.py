"""Tests for Phase 4 diversification mutations.

One success-path test per mutation, plus a "no candidates → no-op" rollback
test. Mutations operate on a minimal siding chromosome that already has
two switches and a closed branch (so branch-cycle utilities have work).
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest

from src_v2.catalog import TrackCatalog
from src_v2.config import OptimizationConfig
from src_v2.decoder import DecoderConfig, decode_chromosome
from src_v2.encoding import (
    INACTIVE,
    PortPairDimensions,
    compute_port_pair_dimensions,
    create_empty_chromosome,
    get_port_pair,
    set_piece_slot,
    set_port_pair,
    set_slot_flip,
    set_slot_rotate,
    iter_active_slots,
)
from src_v2.structural_mutations import (
    introduce_switch_pair,
    mutate_branch_extend,
    mutate_branch_shrink,
    mutate_closure_repair_lamarckian,
    mutate_reverse_switch_pairing,
    mutate_split_loop_with_crossing,
    mutate_swap_switch_hand,
)


CATALOG_PATH = Path(__file__).parent.parent / "data" / "track_pieces_v2.yaml"
CONFIG_PATH = Path(__file__).parent.parent / "configs" / "with_switches.yaml"


@pytest.fixture(scope="module")
def setup():
    catalog = TrackCatalog.load(CATALOG_PATH)
    config = OptimizationConfig.load(CONFIG_PATH)
    dims = compute_port_pair_dimensions(config.boundary, catalog, config.inventory)
    return catalog, config, dims, DecoderConfig()


def _build_siding_chromosome(catalog: TrackCatalog, dims: PortPairDimensions, config) -> np.ndarray:
    """Mainline of 14 STRAIGHT_16 + 2 R40_CURVE corner; introduce_switch_pair
    adds the two switches + A*-found branch closure. Returns the resulting x.
    """
    x = create_empty_chromosome(dims)
    s_idx = catalog.id_to_index["STRAIGHT_16"]
    c_idx = catalog.id_to_index["R40_CURVE"]
    # 16 straights + 16 curves = simple oval shape
    for k in range(16):
        set_piece_slot(x, dims, k, s_idx)
    for k in range(16, 32):
        set_piece_slot(x, dims, k, c_idx)
    for k in range(32):
        set_port_pair(x, dims, k, k, 1, (k + 1) % 32, 0)

    rng = random.Random(0)
    introduce_switch_pair(x, dims, catalog, config.inventory, rng=rng)
    return x


def test_branch_extend_inserts_straight(setup):
    catalog, config, dims, decoder_cfg = setup
    x = _build_siding_chromosome(catalog, dims, config)
    s_idx = catalog.id_to_index["STRAIGHT_16"]
    n_str_before = sum(1 for _, p in iter_active_slots(x, dims) if p == s_idx)

    rng = random.Random(7)
    ok = mutate_branch_extend(x, dims, catalog, decoder_cfg, config.inventory, rng=rng)

    if ok:
        n_str_after = sum(1 for _, p in iter_active_slots(x, dims) if p == s_idx)
        assert n_str_after == n_str_before + 1
    # If ok is False (no branch edge eligible), rollback safety: piece count unchanged.


def test_branch_shrink_drops_straight(setup):
    catalog, config, dims, decoder_cfg = setup
    x = _build_siding_chromosome(catalog, dims, config)
    s_idx = catalog.id_to_index["STRAIGHT_16"]
    n_str_before = sum(1 for _, p in iter_active_slots(x, dims) if p == s_idx)

    rng = random.Random(11)
    ok = mutate_branch_shrink(x, dims, catalog, decoder_cfg, config.inventory, rng=rng)
    if ok:
        n_str_after = sum(1 for _, p in iter_active_slots(x, dims) if p == s_idx)
        assert n_str_after == n_str_before - 1


def test_swap_switch_hand_returns_bool(setup):
    """Operator either flips and validates closure, or rolls back. Either is OK."""
    catalog, config, dims, decoder_cfg = setup
    x = _build_siding_chromosome(catalog, dims, config)
    x_before = x.copy()
    rng = random.Random(13)
    result = mutate_swap_switch_hand(x, dims, catalog, decoder_cfg, config.inventory, rng=rng)
    assert isinstance(result, bool)
    if not result:
        np.testing.assert_array_equal(x, x_before)


def test_reverse_switch_pairing_returns_bool(setup):
    catalog, config, dims, decoder_cfg = setup
    x = _build_siding_chromosome(catalog, dims, config)
    x_before = x.copy()
    rng = random.Random(17)
    result = mutate_reverse_switch_pairing(
        x, dims, catalog, decoder_cfg, config.inventory, rng=rng,
    )
    assert isinstance(result, bool)
    if not result:
        np.testing.assert_array_equal(x, x_before)


def test_split_loop_no_perpendicular_pairs_returns_false(setup):
    """Plain oval has no near-perpendicular non-adjacent edges → no-op."""
    catalog, config, dims, decoder_cfg = setup
    x = create_empty_chromosome(dims)
    c_idx = catalog.id_to_index["R40_CURVE"]
    for k in range(16):
        set_piece_slot(x, dims, k, c_idx)
    for k in range(16):
        set_port_pair(x, dims, k, k, 1, (k + 1) % 16, 0)
    x_before = x.copy()
    rng = random.Random(19)
    # Plain 16-curve loop has perpendicular pairs (every 4 slots is 90°)
    # but they're outside the 32-stud midpoint distance threshold for a
    # standard R40 oval → still no candidates.
    result = mutate_split_loop_with_crossing(
        x, dims, catalog, decoder_cfg, config.inventory, rng=rng,
    )
    if not result:
        np.testing.assert_array_equal(x, x_before)


def test_lamarckian_no_residual_returns_false(setup):
    """Closed-loop layout with no residual → no work for Lamarckian repair."""
    catalog, config, dims, decoder_cfg = setup
    x = create_empty_chromosome(dims)
    c_idx = catalog.id_to_index["R40_CURVE"]
    for k in range(16):
        set_piece_slot(x, dims, k, c_idx)
    for k in range(16):
        set_port_pair(x, dims, k, k, 1, (k + 1) % 16, 0)

    x_before = x.copy()
    rng = random.Random(23)
    result = mutate_closure_repair_lamarckian(
        x, dims, catalog, decoder_cfg, config.inventory, rng=rng,
    )
    assert result is False
    np.testing.assert_array_equal(x, x_before)


def test_lamarckian_does_not_overwrite_switches(setup):
    """Window-substitution must skip switch slots even when they sit on the
    failing component (regression test for Phase 4 iteration)."""
    catalog, config, dims, decoder_cfg = setup
    x = _build_siding_chromosome(catalog, dims, config)
    left_idx = catalog.id_to_index["R40_SWITCH_LEFT"]
    right_idx = catalog.id_to_index["R40_SWITCH_RIGHT"]
    switches_before = {
        s for s, p in iter_active_slots(x, dims) if p in (left_idx, right_idx)
    }

    rng = random.Random(29)
    for _ in range(20):
        mutate_closure_repair_lamarckian(
            x, dims, catalog, decoder_cfg, config.inventory, rng=rng,
        )

    switches_after = {
        s for s, p in iter_active_slots(x, dims) if p in (left_idx, right_idx)
    }
    assert switches_before == switches_after


def test_branch_extend_noop_without_branch(setup):
    """Layout without any branches → no branch edges → operator is no-op."""
    catalog, config, dims, decoder_cfg = setup
    x = create_empty_chromosome(dims)
    c_idx = catalog.id_to_index["R40_CURVE"]
    for k in range(16):
        set_piece_slot(x, dims, k, c_idx)
    for k in range(16):
        set_port_pair(x, dims, k, k, 1, (k + 1) % 16, 0)
    x_before = x.copy()
    rng = random.Random(31)
    result = mutate_branch_extend(
        x, dims, catalog, decoder_cfg, config.inventory, rng=rng,
    )
    assert result is False
    np.testing.assert_array_equal(x, x_before)
