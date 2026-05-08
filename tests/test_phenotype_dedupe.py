"""Tests for the Phase 5 phenotype-dedupe callback.

Focused on the bucket-detection logic; full pymoo integration is covered
by the smoke run.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src_v2.catalog import TrackCatalog
from src_v2.config import OptimizationConfig
from src_v2.decoder import decode_chromosome
from src_v2.encoding import (
    compute_port_pair_dimensions,
    create_empty_chromosome,
    set_piece_slot,
    set_port_pair,
)
from src_v2.phenotype_dedupe import Phenotype, PhenotypeDedupeCallback
from src_v2.problem import PortPairProblem


CATALOG_PATH = Path(__file__).parent.parent / "data" / "track_pieces_v2.yaml"
CONFIG_PATH = Path(__file__).parent.parent / "configs" / "default.yaml"


@pytest.fixture(scope="module")
def setup():
    catalog = TrackCatalog.load(CATALOG_PATH)
    config = OptimizationConfig.load(CONFIG_PATH)
    problem = PortPairProblem(catalog, config)
    return catalog, config, problem


def _build_oval(catalog, dims):
    """16-curve closed oval — well-defined phenotype."""
    x = create_empty_chromosome(dims)
    c_idx = catalog.id_to_index["R40_CURVE"]
    for k in range(16):
        set_piece_slot(x, dims, k, c_idx)
    for k in range(16):
        set_port_pair(x, dims, k, k, 1, (k + 1) % 16, 0)
    return x


def test_phenotype_is_hashable(setup):
    catalog, _, problem = setup
    x = _build_oval(catalog, problem.dims)
    graph = decode_chromosome(x, problem.dims, catalog, problem.decoder_config)
    phen = problem.build_phenotype(graph)
    assert isinstance(phen, Phenotype)
    # Hashable → can live in a set / dict key.
    {phen}
    {phen: 1}


def test_identical_chromosomes_collide(setup):
    catalog, _, problem = setup
    x1 = _build_oval(catalog, problem.dims)
    x2 = _build_oval(catalog, problem.dims)
    g1 = decode_chromosome(x1, problem.dims, catalog, problem.decoder_config)
    g2 = decode_chromosome(x2, problem.dims, catalog, problem.decoder_config)
    assert problem.build_phenotype(g1) == problem.build_phenotype(g2)


def test_different_topology_distinct_phenotypes(setup):
    catalog, _, problem = setup
    # 16-curve oval
    x_oval = _build_oval(catalog, problem.dims)
    # 4-straight chain (distinct topology)
    x_chain = create_empty_chromosome(problem.dims)
    s_idx = catalog.id_to_index["STRAIGHT_16"]
    for k in range(4):
        set_piece_slot(x_chain, problem.dims, k, s_idx)
    for k in range(3):
        set_port_pair(x_chain, problem.dims, k, k, 1, k + 1, 0)

    g1 = decode_chromosome(x_oval, problem.dims, catalog, problem.decoder_config)
    g2 = decode_chromosome(x_chain, problem.dims, catalog, problem.decoder_config)
    assert problem.build_phenotype(g1) != problem.build_phenotype(g2)


def test_phenotype_n_cycles(setup):
    catalog, _, problem = setup
    x = _build_oval(catalog, problem.dims)
    graph = decode_chromosome(x, problem.dims, catalog, problem.decoder_config)
    phen = problem.build_phenotype(graph)
    assert phen.n_cycles == 1
    assert phen.n_switches == 0
    assert phen.n_crossings == 0
    assert phen.max_component_size == 16


def test_dedupe_callback_attribute_defaults(setup):
    """PhenotypeDedupeCallback initialises with the expected throttling defaults."""
    cb = PhenotypeDedupeCallback(sampling=None)
    assert cb._cadence == 20
    assert cb._gen == 0


def test_dedupe_no_problem_no_op(setup):
    """notify() with missing problem/pop must not raise."""
    from types import SimpleNamespace
    cb = PhenotypeDedupeCallback(sampling=None, cadence=1)
    cb.notify(SimpleNamespace(pop=None, problem=None))
    cb.notify(SimpleNamespace(pop=[], problem=None))


def test_dedupe_last_stats_populated_after_compute(setup):
    """After ``notify()`` with bucketing, ``last_*`` attrs reflect the latest counts.

    These attrs are read by ``DiagnosticsCallback`` to populate
    ``dedupe_rejection_rate`` in the per-gen CSV (Section 10.4).
    """
    from types import SimpleNamespace

    catalog, _, problem = setup
    cb = PhenotypeDedupeCallback(sampling=None, cadence=1)

    # Defaults are None until notify() actually computes buckets.
    assert (cb.last_gen, cb.last_pop_size, cb.last_n_phenotypes, cb.last_n_duplicates) == (
        None, None, None, None,
    )

    # Three individuals: two identical ovals (1 duplicate) + one chain.
    x_oval1 = _build_oval(catalog, problem.dims)
    x_oval2 = _build_oval(catalog, problem.dims)
    x_chain = create_empty_chromosome(problem.dims)
    s_idx = catalog.id_to_index["STRAIGHT_16"]
    for k in range(4):
        set_piece_slot(x_chain, problem.dims, k, s_idx)

    pop = [
        SimpleNamespace(get=lambda key, _x=x: _x if key == "X" else None)
        for x in (x_oval1, x_oval2, x_chain)
    ]
    cb.notify(SimpleNamespace(pop=pop, problem=problem))

    assert cb.last_gen == 1
    assert cb.last_pop_size == 3
    assert cb.last_n_phenotypes == 2
    assert cb.last_n_duplicates == 1
