"""Tests for ``src_v2.repair.CycleClosureRepair`` (Phase 1, PLAN §10.2 1.1-1.7).

Phase 1.A: Lamarckian repair core (modifies x in place). Tests 1.1-1.5, 1.7
cover the algorithm correctness on isolated chromosomes.

Phase 1.B: Baldwinian wrapping (Rule 24 revised). Test 1.6 (Coupling B) checks
that the repaired phenotype reaches downstream consumers via ``out["pheno"]``,
including the pickle round-trip through ``StarmapParallelization`` workers,
while the chromosome in the population pool stays in its raw (unrepaired)
form so NSGA-II selection/crossover preserve genotypic diversity.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src_v2.catalog import TrackCatalog
from src_v2.config import OptimizationConfig
from src_v2.decoder import decode_chromosome
from src_v2.encoding import (
    create_empty_chromosome,
    set_piece_slot,
    set_port_pair,
)
from src_v2.problem import PortPairProblem
from src_v2.repair import CycleClosureRepair
from tests.fixtures.hand_crafted_chromosomes import (
    deg250_deficit,
    deg380_excess,
    isolated_active_slot,
)


CATALOG_PATH = Path(__file__).parent.parent / "data" / "track_pieces_v2.yaml"
CONFIG_PATH = Path(__file__).parent.parent / "configs" / "default.yaml"


@pytest.fixture(scope="module")
def setup():
    catalog = TrackCatalog.load(CATALOG_PATH)
    config = OptimizationConfig.load(CONFIG_PATH)
    problem = PortPairProblem(catalog, config)
    return catalog, config, problem


def _make_repair(setup, *, inventory_override=None):
    catalog, config, problem = setup
    return CycleClosureRepair(
        dims=problem.dims,
        catalog=catalog,
        decoder_config=problem.decoder_config,
        inventory=inventory_override or dict(config.inventory),
    )


def _decode(setup, x):
    catalog, _config, problem = setup
    return decode_chromosome(x, problem.dims, catalog, problem.decoder_config)


# ---------------------------------------------------------------- 1.1
def test_deg250_deficit_repair_grows_cycle_to_16_R40(setup):
    """11-R40 closed cycle (~247.5°) → repair splices 5 R40 → 16-piece cycle (~360°)."""
    catalog, _config, problem = setup
    repair = _make_repair(setup)
    x = deg250_deficit(catalog, problem.dims).copy()

    repair.repair_one(x)
    g = _decode(setup, x)

    assert g.n_slots == 16
    assert g.n_cycles == 1
    assert g.n_loose_ports == 0


# ---------------------------------------------------------------- 1.2
def test_deg380_excess_repair_shrinks_cycle_to_16_R40(setup):
    """17-R40 closed cycle (~382.5°) → repair removes 1 R40 → 16-piece cycle (~360°)."""
    catalog, _config, problem = setup
    repair = _make_repair(setup)
    x = deg380_excess(catalog, problem.dims).copy()

    repair.repair_one(x)
    g = _decode(setup, x)

    assert g.n_slots == 16
    assert g.n_cycles == 1
    assert g.n_loose_ports == 0


# ---------------------------------------------------------------- 1.3
def test_inventory_exhausted_graceful_stop(setup):
    """Cap R40 at 11 → repair on deg250_deficit cannot add → chromosome unchanged."""
    catalog, _config, problem = setup
    repair = _make_repair(setup, inventory_override={"R40_CURVE": 11, "STRAIGHT_16": 16})
    x = deg250_deficit(catalog, problem.dims).copy()
    x_before = x.copy()

    repair.repair_one(x)

    assert np.array_equal(x, x_before)


# ---------------------------------------------------------------- 1.4
def test_isolated_active_slot_no_cycle_unchanged(setup):
    """1 active R40, no edges → no cycle → repair is a no-op."""
    catalog, _config, problem = setup
    repair = _make_repair(setup)
    x = isolated_active_slot(catalog, problem.dims).copy()
    x_before = x.copy()

    repair.repair_one(x)

    assert np.array_equal(x, x_before)


# ---------------------------------------------------------------- 1.5
def test_two_cycles_only_deficit_one_modified(setup):
    """Two disjoint cycles: 16-R40 (closed at 360°) + 11-R40 (deficit). Repair
    modifies only the deficit cycle; the closed-at-360° one stays byte-equal.
    """
    catalog, _config, problem = setup
    repair = _make_repair(setup)
    dims = problem.dims
    c = catalog.id_to_index["R40_CURVE"]

    # Component A (slots 0-15): 16-R40 closed cycle, 360° — already valid.
    # Component B (slots 16-26): 11-R40 closed cycle, ~ 247.5° — needs +5 R40.
    x = create_empty_chromosome(dims)
    for k in range(16):
        set_piece_slot(x, dims, k, c)
    for k in range(16):
        set_port_pair(x, dims, k, k, 1, (k + 1) % 16, 0)
    for k in range(11):
        set_piece_slot(x, dims, 16 + k, c)
    for k in range(11):
        set_port_pair(x, dims, 16 + k, 16 + k, 1, 16 + ((k + 1) % 11), 0)

    x_before = x.copy()
    repair.repair_one(x)
    g = _decode(setup, x)

    # First cycle's slots (0-15) must be untouched.
    assert np.array_equal(x[:16], x_before[:16]), "slot 0-15 should be byte-equal"
    # Second cycle's piece-slot region (16-26) also untouched at the slot-id level
    # — repair adds NEW slots beyond 26, doesn't rewrite the existing 11.
    assert np.array_equal(x[16:27], x_before[16:27]), "slot 16-26 should be byte-equal"
    # Total active slots: 16 (cycle A) + 11 + 5 (cycle B grew) = 32.
    assert g.n_slots == 32
    assert g.n_cycles == 2
    assert g.n_components == 2


# ---------------------------------------------------------------- 1.7
def test_skip_anchor_slots_skips_containing_cycle(setup):
    """Cycle containing a slot in skip_anchor_slots is left unmodified
    (Coupling C placeholder for Phase 5+ junction-anchor protection).
    """
    catalog, _config, problem = setup
    repair = _make_repair(setup)
    x = deg250_deficit(catalog, problem.dims).copy()
    x_before = x.copy()

    # Slot 0 is part of the deg250 cycle. Placing it in skip_anchor_slots
    # should cause the entire cycle to be skipped.
    repair.repair_one(x, skip_anchor_slots={0})

    assert np.array_equal(x, x_before)


def test_skip_anchor_slots_default_is_empty_set(setup):
    """API contract: skip_anchor_slots parameter defaults to None (treated as empty)."""
    import inspect
    sig = inspect.signature(CycleClosureRepair.repair_one)
    assert "skip_anchor_slots" in sig.parameters
    assert sig.parameters["skip_anchor_slots"].default is None


# ---------------------------------------------------------------- 1.6
def test_baldwinian_pheno_roundtrip_via_starmap_pool(setup):
    """Coupling B (PLAN §10.2 1.6): out["pheno"] round-trips through
    ``StarmapParallelization(Pool(2).starmap)`` — the main process sees the
    SAME repaired ``PortGraph`` the worker computed, while the chromosome in
    the population pool keeps its raw (unrepaired) form (Rule 24 revised).

    This test bypasses ``PortPairRepairPipeline`` (calls ``Evaluator().eval``
    directly) so the only place a chromosome could be repaired is
    ``PortPairProblem._evaluate`` — which in Phase 1.B clones x, runs
    ``CycleClosureRepair.repair_one`` on the clone, decodes the clone, sets
    ``out["pheno"]`` to that graph, and computes F/G from it.

    Pre-Phase-1.B baseline: ``out["pheno"]`` is never set; this test fails
    on the ``pheno is not None`` assertion.
    """
    from multiprocessing import Pool

    from pymoo.core.evaluator import Evaluator
    from pymoo.core.individual import Individual
    from pymoo.core.population import Population
    from pymoo.parallelization import StarmapParallelization

    from src_v2.encoding import iter_active_slots
    from src_v2.types import PortGraph

    catalog, config, _ = setup
    raw_x = deg250_deficit(catalog, _make_repair(setup).dims)

    with Pool(2) as pool:
        runner = StarmapParallelization(pool.starmap)
        problem = PortPairProblem(catalog, config, elementwise_runner=runner)
        pop = Population.create(*[Individual(X=raw_x.copy()) for _ in range(4)])
        Evaluator().eval(problem, pop)

    raw_active = sum(
        1 for _ in iter_active_slots(raw_x, problem.dims)
    )
    assert raw_active == 11, "fixture sanity: deg250_deficit has 11 active slots"

    for ind in pop:
        pheno = ind.get("pheno")
        assert pheno is not None, (
            "out['pheno'] missing — Phase 1.B Baldwinian wiring not active"
        )
        assert isinstance(pheno, PortGraph), (
            f"out['pheno'] expected PortGraph, got {type(pheno).__name__}"
        )
        assert pheno.n_slots == 16, (
            f"repaired phenotype expected 16 slots (cycle grown to ~360°), "
            f"got {pheno.n_slots}"
        )

        x_pool = ind.get("X")
        pool_active = sum(
            1 for _ in iter_active_slots(x_pool, problem.dims)
        )
        assert pool_active == 11, (
            f"raw chromosome in pool should still have 11 active slots "
            f"(Baldwinian: surgery on clone, not X); got {pool_active}"
        )
