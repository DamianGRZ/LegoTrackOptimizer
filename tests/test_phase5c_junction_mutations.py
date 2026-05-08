"""Tests for Phase 5c -- junction mutations + meta-op consolidation
(PLAN §10.2 5c.2, 5c.3, 5c.4, 5c.5, 5c.6, 5c.7).

Phase 5c adds four junction sub-operators to ``PortPairMutation`` and
consolidates them under a single ``tune_passing_siding`` ALNS slot
(Rule 29 revised). Cold-start weight at ``max(existing weights)`` per
Rule 29a (supersedes Rule 14).
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from src_v2.catalog import TrackCatalog
from src_v2.config import OptimizationConfig
from src_v2.encoding import (
    JUNCTION_GENES,
    JUNCTION_KIND_PASSING_SIDING,
    compute_port_pair_dimensions,
    create_empty_chromosome,
    get_junction,
    set_junction,
    set_piece_slot,
    set_port_pair,
)
from src_v2.operators import PortPairMutation


CATALOG_PATH = Path(__file__).parent.parent / "data" / "track_pieces_v2.yaml"
WITH_SWITCHES_CFG = Path(__file__).parent.parent / "configs" / "with_switches.yaml"


@pytest.fixture(scope="module")
def setup():
    catalog = TrackCatalog.load(CATALOG_PATH)
    config = OptimizationConfig.load(WITH_SWITCHES_CFG)
    dims = compute_port_pair_dimensions(config.boundary, catalog, config.inventory)
    return catalog, config, dims


def _build_oval_with_active_junction(catalog, dims, *, anchor: int = 0,
                                     param_a: int = 2, param_b: int = 0):
    x = create_empty_chromosome(dims)
    r40 = catalog.id_to_index["R40_CURVE"]
    for k in range(16):
        set_piece_slot(x, dims, k, r40)
        set_port_pair(x, dims, k, k, 1, (k + 1) % 16, 0)
    set_junction(
        x, dims, 0,
        active=1, anchor=anchor,
        kind=JUNCTION_KIND_PASSING_SIDING,
        param_a=param_a, param_b=param_b,
    )
    return x


# ---------------------------------------------------------------- 5c.7
def test_5c_7_alns_pool_size_within_effective_cap(setup) -> None:
    """Phase 5c keeps the ALNS pool at <=17 effective slots: 16 prior +
    1 new ``tune_passing_siding`` meta-op."""
    catalog, config, _dims = setup
    mutation = PortPairMutation(_dims_for(catalog, config), catalog, config)
    assert len(mutation.OP_WEIGHTS) <= 17
    assert "tune_passing_siding" in mutation.OP_WEIGHTS


def _dims_for(catalog, config):
    return compute_port_pair_dimensions(config.boundary, catalog, config.inventory)


# ---------------------------------------------------------------- 5c.6
def test_5c_6_cold_start_at_max_weight(setup) -> None:
    """``tune_passing_siding`` initial weight equals the maximum of the
    pre-Phase-5c operator weights (Rule 29a, supersedes Rule 14)."""
    catalog, config, dims = setup
    mutation = PortPairMutation(dims, catalog, config)
    other_weights = [
        w for name, w in mutation.OP_WEIGHTS.items()
        if name != "tune_passing_siding"
    ]
    assert mutation.OP_WEIGHTS["tune_passing_siding"] == max(other_weights)


# ---------------------------------------------------------------- 5c.4
def test_5c_4_toggle_active_byte_equal_except_target(setup) -> None:
    """``_mutate_junction_toggle_active`` flips exactly one ``active`` bit
    and leaves every other gene byte-equal."""
    catalog, config, dims = setup
    mutation = PortPairMutation(dims, catalog, config, seed=42)
    x = _build_oval_with_active_junction(catalog, dims, anchor=3)
    x_before = x.copy()
    mutation._mutate_junction_toggle_active(x)

    diff_indices = np.where(x != x_before)[0]
    assert len(diff_indices) == 1, (
        f"toggle_active should flip exactly 1 gene; got {len(diff_indices)}"
    )
    flipped_idx = int(diff_indices[0])
    # Must land on a junction's active offset (multiple of JUNCTION_GENES from junc_start).
    rel = flipped_idx - dims.junc_start
    assert 0 <= rel < dims.J_max * JUNCTION_GENES
    assert rel % JUNCTION_GENES == 0, (
        f"flipped gene at offset {rel} is not a junction active bit"
    )
    # Bit must have actually flipped.
    assert int(x[flipped_idx]) != int(x_before[flipped_idx])


# ---------------------------------------------------------------- 5c.2
def test_5c_2_reposition_snaps_to_nearest_active_slot(setup) -> None:
    """``_mutate_junction_reposition`` shifts anchor then snaps to the
    nearest active slot (Rule 9 -- preserve locality, no random clamping)."""
    catalog, config, dims = setup
    mutation = PortPairMutation(dims, catalog, config, seed=42)
    # Build a chromosome where slot 3 is the junction anchor and slots
    # {3, 9} are the only active branch-capable slots in the chromosome.
    x = create_empty_chromosome(dims)
    r40 = catalog.id_to_index["R40_CURVE"]
    set_piece_slot(x, dims, 3, r40)
    set_piece_slot(x, dims, 9, r40)
    set_junction(
        x, dims, 0,
        active=1, anchor=3, kind=JUNCTION_KIND_PASSING_SIDING,
        param_a=2, param_b=0,
    )

    for _trial in range(40):
        x_trial = x.copy()
        mutation._mutate_junction_reposition(x_trial)
        new_anchor = get_junction(x_trial, dims, 0)[1]
        # New anchor must land on one of the two active slots, never on an
        # inactive slot or far from the originals (no random clamping).
        assert new_anchor in {3, 9}, (
            f"reposition produced anchor {new_anchor}; must snap to active slot"
        )


# ---------------------------------------------------------------- 5c.3
def test_5c_3_adjust_straights_respects_inventory(setup) -> None:
    """``_mutate_junction_adjust_straights`` clamps ``param_a`` to the
    available STRAIGHT_16 inventory."""
    catalog, config, dims = setup
    # Force a tiny straights inventory so we can verify clamping bites.
    cfg_small = OptimizationConfig.model_validate({
        **config.model_dump(),
        "inventory": {**config.inventory, "STRAIGHT_16": 3},
    })
    mutation = PortPairMutation(dims, catalog, cfg_small, seed=42)
    x = _build_oval_with_active_junction(catalog, dims, anchor=3, param_a=2)

    for _trial in range(40):
        x_trial = x.copy()
        mutation._mutate_junction_adjust_straights(x_trial)
        new_param_a = get_junction(x_trial, dims, 0)[3]
        assert 0 <= new_param_a <= 3, (
            f"adjust_straights produced param_a={new_param_a}; "
            f"must be in [0, inventory={cfg_small.inventory['STRAIGHT_16']}]"
        )


# ---------------------------------------------------------------- 5c.5
def test_5c_5_meta_op_dispatches_to_all_four_sub_ops(setup) -> None:
    """``_mutate_tune_passing_siding`` (the meta-op) reaches each of the
    four junction sub-ops over enough invocations -- sub-op picks are
    uniform inside the single ALNS slot (Rule 29 revised)."""
    catalog, config, dims = setup
    mutation = PortPairMutation(dims, catalog, config, seed=42)
    sub_op_names = [
        "_mutate_junction_toggle_active",
        "_mutate_junction_reposition",
        "_mutate_junction_swap_handedness",
        "_mutate_junction_adjust_straights",
    ]
    counts: Counter = Counter()
    original = {name: getattr(mutation, name) for name in sub_op_names}
    try:
        for name in sub_op_names:
            def make_spy(n=name):
                def spy(x):
                    counts[n] += 1
                    original[n](x)
                return spy
            setattr(mutation, name, make_spy())
        x = _build_oval_with_active_junction(catalog, dims, anchor=3)
        for _ in range(400):
            mutation._mutate_tune_passing_siding(x.copy())
    finally:
        for name, fn in original.items():
            setattr(mutation, name, fn)

    for name in sub_op_names:
        assert counts[name] > 0, f"sub-op {name} never dispatched"
    assert sum(counts.values()) == 400
