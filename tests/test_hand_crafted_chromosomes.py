"""Tests for ``tests/fixtures/hand_crafted_chromosomes.py`` (Phase 17.C.2)."""
from __future__ import annotations

from pathlib import Path

import pytest

from src_v2.catalog import TrackCatalog
from src_v2.config import OptimizationConfig
from src_v2.decoder import decode_chromosome
from src_v2.problem import PortPairProblem
from tests.fixtures.hand_crafted_chromosomes import (
    broken_cycle_2_components,
    deg250_deficit,
    deg380_excess,
    inventory_exhausted,
    isolated_active_slot,
    loose_port_chromosome,
    perfect_oval_16_R40,
)


CATALOG_PATH = Path(__file__).parent.parent / "data" / "track_pieces_v2.yaml"
CONFIG_PATH = Path(__file__).parent.parent / "configs" / "default.yaml"


@pytest.fixture(scope="module")
def setup():
    catalog = TrackCatalog.load(CATALOG_PATH)
    config = OptimizationConfig.load(CONFIG_PATH)
    problem = PortPairProblem(catalog, config)
    return catalog, problem


def _decode(setup, chromosome):
    catalog, problem = setup
    return decode_chromosome(chromosome, problem.dims, catalog, problem.decoder_config)


# ---------------------------------------------------------------- HCC.1
def test_perfect_oval_16_R40_is_closed_single_cycle(setup):
    catalog, problem = setup
    g = _decode(setup, perfect_oval_16_R40(catalog, problem.dims))
    assert g.n_slots == 16
    assert g.n_components == 1
    assert g.n_cycles == 1
    assert g.n_loose_ports == 0


# ---------------------------------------------------------------- HCC.2
def test_deg250_deficit_is_closed_cycle_with_residual(setup):
    """11 R40 closed cycle; topologically valid but FK accumulates only
    247.5° of rotation. Phase 1's repair adds curves to drive sum → 360°."""
    catalog, problem = setup
    g = _decode(setup, deg250_deficit(catalog, problem.dims))
    assert g.n_slots == 11
    assert g.n_cycles == 1
    assert g.n_loose_ports == 0


# ---------------------------------------------------------------- HCC.3
def test_deg380_excess_closes_with_angular_residual(setup):
    catalog, problem = setup
    g = _decode(setup, deg380_excess(catalog, problem.dims))
    assert g.n_slots == 17
    assert g.n_cycles == 1
    assert g.n_loose_ports == 0


# ---------------------------------------------------------------- HCC.4
def test_broken_cycle_2_components(setup):
    catalog, problem = setup
    g = _decode(setup, broken_cycle_2_components(catalog, problem.dims))
    assert g.n_slots == 32
    assert g.n_components == 2
    assert g.n_cycles == 2
    assert g.n_loose_ports == 0


# ---------------------------------------------------------------- HCC.5
def test_loose_port_chromosome_has_unpaired_ports(setup):
    catalog, problem = setup
    g = _decode(setup, loose_port_chromosome(catalog, problem.dims))
    assert g.n_slots == 2
    assert g.n_loose_ports >= 1


# ---------------------------------------------------------------- HCC.6
def test_inventory_exhausted_has_9_active_R40_slots(setup):
    catalog, problem = setup
    g = _decode(setup, inventory_exhausted(catalog, problem.dims))
    # 9 active R40 pieces; a real Phase-1 test will pair this with an inventory
    # of 8 R40 and assert the inventory-excess constraint G[5+T_R40] > 0.
    assert g.n_slots == 9


# ---------------------------------------------------------------- HCC.7
def test_isolated_active_slot_no_edges(setup):
    catalog, problem = setup
    g = _decode(setup, isolated_active_slot(catalog, problem.dims))
    assert g.n_slots == 1
    assert g.n_edges == 0
