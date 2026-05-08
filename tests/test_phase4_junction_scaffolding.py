"""Tests for Phase 4 -- junction segment scaffolding (PLAN §10.2 4.1-4.8).

Phase 4 extends the chromosome with a J_max-sized junction segment whose
descriptors are read by every operator but materialized by no one yet
(decoder ignores them; templates ship in Phase 5a). Coupling A splits the
single ``PortPairCrossover`` into two operators so junction-cx can be
ablated independently of port-pair-cx (Rules 25, 26 revised).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src_v2.canonical import canonical_graph_hash
from src_v2.catalog import TrackCatalog
from src_v2.config import OptimizationConfig
from src_v2.decoder import DecoderConfig, decode_chromosome
from src_v2.encoding import (
    ENCODING_VERSION,
    GENES_PER_PAIR,
    INACTIVE,
    JUNCTION_GENES,
    PortPairDimensions,
    compute_port_pair_dimensions,
    create_empty_chromosome,
    generate_bounds,
    get_junction,
    set_junction,
    validate_chromosome,
)
from src_v2.operators import JunctionCrossover, PortPairCrossover


CATALOG_PATH = Path(__file__).parent.parent / "data" / "track_pieces_v2.yaml"
DEFAULT_CFG = Path(__file__).parent.parent / "configs" / "default.yaml"
WITH_SWITCHES_CFG = Path(__file__).parent.parent / "configs" / "with_switches.yaml"


@pytest.fixture(scope="module")
def catalog() -> TrackCatalog:
    return TrackCatalog.load(CATALOG_PATH)


@pytest.fixture
def with_switches(catalog):
    cfg = OptimizationConfig.load(WITH_SWITCHES_CFG)
    dims = compute_port_pair_dimensions(cfg.boundary, catalog, cfg.inventory)
    return cfg, dims


# ---------------------------------------------------------------- 4.1
def test_4_1_n_var_formula(catalog) -> None:
    """``n_var = 3*N_max + 4*E_max + 5*J_max + 3`` (slots/flips/rotates +
    pairs + junctions + anchor)."""
    cfg = OptimizationConfig.load(WITH_SWITCHES_CFG)
    dims = compute_port_pair_dimensions(cfg.boundary, catalog, cfg.inventory)
    expected = 3 * dims.N_max + GENES_PER_PAIR * dims.E_max + JUNCTION_GENES * dims.J_max + 3
    assert dims.n_var == expected
    assert dims.J_max > 0, "with_switches must have at least one junction slot"


# ---------------------------------------------------------------- 4.2
def test_4_2_int16_overflow_guard() -> None:
    """``__post_init__`` raises when n_var would exceed int16 range."""
    int16_max = int(np.iinfo(np.int16).max)
    bad_n_max = int16_max  # 32767 slots alone -> n_var well past int16
    with pytest.raises((AssertionError, ValueError, OverflowError)):
        PortPairDimensions(N_max=bad_n_max, E_max=bad_n_max, J_max=bad_n_max)


# ---------------------------------------------------------------- 4.3
def test_4_3_generate_bounds_covers_junction_range(with_switches, catalog) -> None:
    """``xl``/``xu`` arrays have length ``n_var`` and constrain the junction
    region: active in {0,1}; anchor slot in [0, N_max-1]; kind/params bounded."""
    cfg, dims = with_switches
    xl, xu = generate_bounds(dims, cfg.boundary, max_piece_id=catalog.n_pieces - 1)
    assert xl.shape == (dims.n_var,)
    assert xu.shape == (dims.n_var,)

    for j in range(dims.J_max):
        base = dims.junc_start + j * JUNCTION_GENES
        # active bit
        assert xl[base + 0] == 0 and xu[base + 0] == 1
        # anchor slot in [0, N_max-1]
        assert xl[base + 1] == 0
        assert xu[base + 1] == max(0, dims.N_max - 1)
        # kind, param_a, param_b: bounded non-negative
        assert xl[base + 2] >= 0
        assert xl[base + 3] >= 0
        assert xl[base + 4] >= 0
        assert xu[base + 2] >= xl[base + 2]
        assert xu[base + 3] >= xl[base + 3]
        assert xu[base + 4] >= xl[base + 4]


# ---------------------------------------------------------------- 4.4
def test_4_4_validate_chromosome_rejects_out_of_range_junction(with_switches) -> None:
    """``validate_chromosome`` flags junction values outside the active-bit
    domain or the slot-anchor range."""
    _cfg, dims = with_switches
    if dims.J_max == 0:
        pytest.skip("config has no junction capacity")
    x = create_empty_chromosome(dims)
    # Sane chromosome should validate
    assert validate_chromosome(x, dims) == []

    # Out-of-range active bit (must be 0 or 1)
    x_bad = x.copy()
    x_bad[dims.junc_start + 0] = 7
    errs = validate_chromosome(x_bad, dims)
    assert errs, "expected validation errors for out-of-range active bit"

    # Out-of-range anchor slot
    x_bad2 = x.copy()
    x_bad2[dims.junc_start + 1] = dims.N_max + 5
    errs2 = validate_chromosome(x_bad2, dims)
    assert errs2, "expected validation errors for out-of-range anchor slot"


# ---------------------------------------------------------------- 4.5
def test_4_5_junction_crossover_is_separate_operator(with_switches, catalog) -> None:
    """Coupling A: with port-pair crossover_prob=0 and junction_crossover_prob=1
    the port-pair regions stay byte-identical to the parents while junction
    descriptors swap."""
    _cfg, dims = with_switches
    if dims.J_max == 0:
        pytest.skip("config has no junction capacity")

    rng = np.random.default_rng(42)
    p1 = create_empty_chromosome(dims)
    p2 = create_empty_chromosome(dims)
    # Plant distinct junction descriptors so the swap is observable.
    set_junction(p1, dims, 0, active=1, anchor=3, kind=0, param_a=2, param_b=5)
    set_junction(p2, dims, 0, active=1, anchor=4, kind=0, param_a=7, param_b=1)
    # Plant distinct port-pair edges so we can confirm port-pair stays put.
    p1[dims.pair_start + 0] = 0   # slot_a
    p2[dims.pair_start + 0] = 1
    # Active anchor slots so the semantic guard accepts the swap.
    p1[dims.slot_start + 3] = 0  # piece index 0 active
    p1[dims.slot_start + 4] = 0
    p2[dims.slot_start + 3] = 0
    p2[dims.slot_start + 4] = 0

    junc_cx = JunctionCrossover(dims, catalog, prob=1.0)
    X = np.stack([p1[None, :], p2[None, :]])  # (n_parents=2, n_matings=1, n_var)
    Y = junc_cx._do(None, X)
    c1, c2 = Y[0, 0], Y[1, 0]

    # Port-pair, slot, flip, rotate, anchor regions: byte-identical to parents.
    assert np.array_equal(c1[: dims.junc_start], p1[: dims.junc_start])
    assert np.array_equal(c2[: dims.junc_start], p2[: dims.junc_start])
    assert np.array_equal(c1[dims.anchor_start :], p1[dims.anchor_start :])
    assert np.array_equal(c2[dims.anchor_start :], p2[dims.anchor_start :])

    # Junction descriptors differ from parents in at least one child.
    j1_after = get_junction(c1, dims, 0)
    j2_after = get_junction(c2, dims, 0)
    j1_before = get_junction(p1, dims, 0)
    j2_before = get_junction(p2, dims, 0)
    swapped = (j1_after == j2_before and j2_after == j1_before)
    unchanged = (j1_after == j1_before and j2_after == j2_before)
    assert swapped or not unchanged, (
        "junction descriptors must either swap or otherwise change vs parents"
    )


# ---------------------------------------------------------------- 4.6
def test_4_6_junction_crossover_semantic_guard(with_switches, catalog) -> None:
    """Rule 4: when the receiver chromosome's anchor_slot is INACTIVE, the
    swap deactivates the junction in the receiver instead of propagating it."""
    _cfg, dims = with_switches
    if dims.J_max == 0:
        pytest.skip("config has no junction capacity")

    p1 = create_empty_chromosome(dims)
    p2 = create_empty_chromosome(dims)
    # p1 has an active junction anchored at slot 99 (which is INACTIVE in p2).
    set_junction(p1, dims, 0, active=1, anchor=99, kind=0, param_a=0, param_b=0)
    # Ensure p2's slot 99 is inactive (default). Anchor slot is also inactive in p1.
    # Force semantic guard to fail in BOTH directions: junction can't land in receiver.
    set_junction(p2, dims, 0, active=0, anchor=0, kind=0, param_a=0, param_b=0)

    junc_cx = JunctionCrossover(dims, catalog, prob=1.0)
    X = np.stack([p1[None, :], p2[None, :]])
    Y = junc_cx._do(None, X)
    c1_j = get_junction(Y[0, 0], dims, 0)
    c2_j = get_junction(Y[1, 0], dims, 0)
    # After guard: receiver of an invalid-anchor junction has it deactivated.
    # c2 would have received p1's junction (anchor=99 inactive in c2 base) -> deactivated.
    assert c2_j[0] == 0, f"expected deactivation; got active={c2_j[0]} junction={c2_j}"


# ---------------------------------------------------------------- 4.7
def test_4_7_canonical_hash_invariant_to_junctions(with_switches, catalog) -> None:
    """Coupling D: ``canonical_graph_hash`` must NOT change when chromosomes
    differ only in their junction descriptors (Phase 4 decoder ignores them)."""
    cfg, dims = with_switches
    if dims.J_max == 0:
        pytest.skip("config has no junction capacity")
    decoder_cfg = DecoderConfig(
        boundary_min_x=cfg.boundary.min_x, boundary_max_x=cfg.boundary.max_x,
        boundary_min_y=cfg.boundary.min_y, boundary_max_y=cfg.boundary.max_y,
    )
    x1 = create_empty_chromosome(dims)
    x2 = create_empty_chromosome(dims)
    # Identical port-pair structure on both -- a tiny 16-R40 closed loop.
    r40_idx = catalog.id_to_index["R40_CURVE"]
    for k in range(16):
        x1[dims.slot_start + k] = r40_idx
        x2[dims.slot_start + k] = r40_idx
        # Each slot's port B connects to next slot's port A
        base = dims.pair_start + k * GENES_PER_PAIR
        x1[base : base + 4] = (k, 1, (k + 1) % 16, 0)
        x2[base : base + 4] = (k, 1, (k + 1) % 16, 0)
    # Different junction descriptors only.
    set_junction(x1, dims, 0, active=1, anchor=0, kind=0, param_a=2, param_b=3)
    set_junction(x2, dims, 0, active=0, anchor=5, kind=0, param_a=9, param_b=1)

    g1 = decode_chromosome(x1, dims, catalog, decoder_cfg)
    g2 = decode_chromosome(x2, dims, catalog, decoder_cfg)
    assert canonical_graph_hash(g1) == canonical_graph_hash(g2)


# ---------------------------------------------------------------- 4.8
def test_4_8_encoding_version_bumped_to_3() -> None:
    """Phase 4 changes ``n_var`` shape; encoding version must bump (Rule 13)."""
    assert ENCODING_VERSION == 3
