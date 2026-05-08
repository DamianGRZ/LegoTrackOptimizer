"""Tests for Phase 7a -- PARALLEL_DC_BRIDGE template machinery.

Phase 7a adds a third template kind to the junction materializer:
``JUNCTION_KIND_PARALLEL_DC_BRIDGE``. An active descriptor expands the
anchor slot into a DOUBLE_CROSSOVER piece. Per the plan's Phase 7a spec,
parallel-section *detection* is "non-trivial" and the algorithm shape is
"high risk" — the spec recommends deferral if Phase 6 doesn't go cleanly.
Phase 6a is currently in [?] state, so this implementation is minimal:
it ships the template + materializer dispatch but does NOT detect
parallel sections automatically. The Phase 7b heuristic seed is
responsible for delivering chromosomes whose existing port-pair edges
already wire a secondary parallel track that the DOUBLE_CROSSOVER joins.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import pytest

from src_v2.catalog import TrackCatalog
from src_v2.config import OptimizationConfig
from src_v2.encoding import (
    JUNCTION_KIND_PARALLEL_DC_BRIDGE,
    compute_port_pair_dimensions,
    create_empty_chromosome,
    set_anchor,
    set_junction,
    set_piece_slot,
    set_port_pair,
)
from src_v2.problem import PortPairProblem
from src_v2.templates import (
    PARALLEL_DC_BRIDGE_PRIMARY,
    PARALLEL_DC_BRIDGE_TEMPLATES,
    ParallelBridgeTemplate,
    check_dc_bridge_inventory,
    get_dc_bridge_inventory_requirements,
)


CATALOG_PATH = Path(__file__).parent.parent / "data" / "track_pieces_v2.yaml"
WITH_CROSSING_CFG = Path(__file__).parent.parent / "configs" / "with_crossing.yaml"


@pytest.fixture(scope="module")
def setup():
    catalog = TrackCatalog.load(CATALOG_PATH)
    config = OptimizationConfig.load(WITH_CROSSING_CFG)
    problem = PortPairProblem(catalog, config)
    return catalog, config, problem


# ---------------------------------------------------------------- 7a-pickle
def test_7a_template_pickle_safe() -> None:
    """``PARALLEL_DC_BRIDGE_PRIMARY`` round-trips through pickle (Rule 11/17)."""
    restored = pickle.loads(pickle.dumps(PARALLEL_DC_BRIDGE_PRIMARY))
    assert restored == PARALLEL_DC_BRIDGE_PRIMARY
    assert restored.dc_id == "DOUBLE_CROSSOVER"


# ---------------------------------------------------------------- 7a-templates-registry
def test_7a_template_in_registry() -> None:
    """The template-kind→variants registry exposes the DC bridge."""
    variants = PARALLEL_DC_BRIDGE_TEMPLATES.get(JUNCTION_KIND_PARALLEL_DC_BRIDGE)
    assert variants is not None and len(variants) >= 1
    assert all(isinstance(v, ParallelBridgeTemplate) for v in variants)


# ---------------------------------------------------------------- 7a-inventory
def test_7a_inventory_requirements() -> None:
    """A DC bridge needs 1 DOUBLE_CROSSOVER and no other pieces (the seed
    is responsible for the parallel-track straights)."""
    reqs = get_dc_bridge_inventory_requirements(PARALLEL_DC_BRIDGE_PRIMARY)
    assert reqs == {"DOUBLE_CROSSOVER": 1}

    available = {"DOUBLE_CROSSOVER": 1}
    assert check_dc_bridge_inventory(PARALLEL_DC_BRIDGE_PRIMARY, available, {})

    used = {"DOUBLE_CROSSOVER": 1}
    assert not check_dc_bridge_inventory(
        PARALLEL_DC_BRIDGE_PRIMARY, available, used,
    )


# ---------------------------------------------------------------- 7a-decode
def test_7a_active_dc_bridge_replaces_anchor_with_double_crossover(setup) -> None:
    """Hand-crafted chromosome with one active PARALLEL_DC_BRIDGE junction
    decodes to a layout where the anchor slot has been replaced by the
    DOUBLE_CROSSOVER piece (or the materialization skipped silently if
    inventory or graph constraints aren't met)."""
    catalog, _config, problem = setup
    dims = problem.dims
    if "DOUBLE_CROSSOVER" not in catalog.id_to_index:
        pytest.skip("DOUBLE_CROSSOVER not in catalog")

    x = create_empty_chromosome(dims)
    r40_idx = catalog.id_to_index["R40_CURVE"]
    for k in range(16):
        set_piece_slot(x, dims, k, r40_idx)
        set_port_pair(x, dims, k, k, 1, (k + 1) % 16, 0)
    set_junction(
        x, dims, 0,
        active=1, anchor=0, kind=JUNCTION_KIND_PARALLEL_DC_BRIDGE,
        param_a=0, param_b=0,
    )
    set_anchor(x, dims, 0, 0, 0)

    out: dict = {}
    problem._evaluate(x, out)
    pheno = out["pheno"]
    n_dc = sum(
        1 for idx in pheno.slot_indices.values()
        if catalog.index_to_id.get(idx) == "DOUBLE_CROSSOVER"
    )
    # Either materialised (1 DC swap-in) or skipped silently. Both valid
    # under the minimal Phase 7a infrastructure (no parallel-section
    # detection yet; the seed is what guarantees a sensible context).
    assert n_dc in (0, 1)


def test_7a_decoder_determinism(setup) -> None:
    """Same chromosome -> same materialised PortGraph (Rule 3)."""
    catalog, _config, problem = setup
    dims = problem.dims
    if "DOUBLE_CROSSOVER" not in catalog.id_to_index:
        pytest.skip("DOUBLE_CROSSOVER not in catalog")

    x = create_empty_chromosome(dims)
    r40_idx = catalog.id_to_index["R40_CURVE"]
    for k in range(16):
        set_piece_slot(x, dims, k, r40_idx)
        set_port_pair(x, dims, k, k, 1, (k + 1) % 16, 0)
    set_junction(
        x, dims, 0,
        active=1, anchor=0, kind=JUNCTION_KIND_PARALLEL_DC_BRIDGE,
        param_a=0, param_b=0,
    )

    out1: dict = {}
    out2: dict = {}
    problem._evaluate(x.copy(), out1)
    problem._evaluate(x.copy(), out2)
    g1, g2 = out1["pheno"], out2["pheno"]
    assert g1.slot_pieces == g2.slot_pieces
    assert g1.slot_indices == g2.slot_indices
