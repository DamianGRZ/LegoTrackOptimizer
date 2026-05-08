"""Tests for src_v2/problem.py — pymoo bi-objective port-pair problem."""

from pathlib import Path

import numpy as np
import pytest

from src_v2.catalog import TrackCatalog
from src_v2.config import BoundaryConfig, OptimizationConfig
from src_v2.encoding import (
    create_empty_chromosome,
    set_anchor,
    set_piece_slot,
    set_port_pair,
)
from src_v2.problem import PortPairProblem


P_R40_LEFT = 2
PA, PB = 0, 1


@pytest.fixture(scope="module")
def catalog():
    return TrackCatalog.load(Path("data/track_pieces_v2.yaml"))


@pytest.fixture
def config():
    return OptimizationConfig(
        inventory={
            "STRAIGHT_16": 16,
            "R40_LEFT": 32,
            "R40_RIGHT": 8,
            "R40_SWITCH_LEFT_IN": 1,
            "R40_SWITCH_LEFT_OUT": 1,
            "R40_SWITCH_RIGHT_IN": 0,
            "R40_SWITCH_RIGHT_OUT": 0,
            "CROSS_90": 1,
            "STRAIGHT_24": 0,
            "DOUBLE_CROSSOVER": 0,
        },
        boundary=BoundaryConfig(
            min_x=-100.0, max_x=100.0, min_y=-100.0, max_y=100.0,
        ),
        closure_tolerance=4.0,
        angle_tolerance=5.0,
    )


# =============================================================================
# Construction
# =============================================================================


class TestConstruction:
    def test_constructs_with_v2_catalog(self, catalog, config):
        problem = PortPairProblem(catalog, config)
        assert problem.n_obj == 2
        # 7 + n_pieces (10) = 17 constraints
        assert problem.n_ieq_constr == 7 + catalog.n_pieces

    def test_dimensions_match_dims(self, catalog, config):
        problem = PortPairProblem(catalog, config)
        assert problem.n_var == problem.dims.n_var

    def test_bounds_shape(self, catalog, config):
        problem = PortPairProblem(catalog, config)
        assert problem.xl.shape == (problem.n_var,)
        assert problem.xu.shape == (problem.n_var,)
        assert (problem.xl <= problem.xu).all()


# =============================================================================
# Empty chromosome → infeasibility sentinel
# =============================================================================


class TestEmpty:
    def test_empty_chromosome_returns_inf_and_violations(self, catalog, config):
        problem = PortPairProblem(catalog, config)
        x = create_empty_chromosome(problem.dims)
        out = {}
        problem._evaluate(x, out)
        assert np.isinf(out["F"]).all()
        # All G filled with sentinel violation
        assert (out["G"] >= 1e5).all()


# =============================================================================
# Hand-crafted closed cycle → feasible (or near-feasible)
# =============================================================================


class TestClosedCycle:
    def test_16_r40_circle_is_feasible(self, catalog, config):
        problem = PortPairProblem(catalog, config)
        x = create_empty_chromosome(problem.dims)
        for k in range(16):
            set_piece_slot(x, problem.dims, k, P_R40_LEFT)
        for k in range(16):
            set_port_pair(x, problem.dims, k, k, PB, (k + 1) % 16, PA)
        set_anchor(x, problem.dims, 0, 0, 0)

        out = {}
        problem._evaluate(x, out)

        # Closure constraints should all be < 0 (feasible)
        assert out["G"][0] < 0, f"closure_x violated: {out['G'][0]}"
        assert out["G"][1] < 0, f"closure_y violated: {out['G'][1]}"
        assert out["G"][2] < 0, f"closure_theta violated: {out['G'][2]}"

        # Boundary constraint: 16 R40 LEFT circle has diameter ~80,
        # comfortably within 200x200 box → boundary g should be ≤ 0
        assert out["G"][3] < 0, f"boundary violated: {out['G'][3]}"

        # Cycle count: 1 cycle → 1 - 1 = 0 ≤ 0
        cycle_g = out["G"][-1]
        assert cycle_g <= 0, f"cycle count constraint violated: {cycle_g}"

    def test_16_r40_circle_objectives_sane(self, catalog, config):
        problem = PortPairProblem(catalog, config)
        x = create_empty_chromosome(problem.dims)
        for k in range(16):
            set_piece_slot(x, problem.dims, k, P_R40_LEFT)
        for k in range(16):
            set_port_pair(x, problem.dims, k, k, PB, (k + 1) % 16, PA)
        set_anchor(x, problem.dims, 0, 0, 0)

        out = {}
        problem._evaluate(x, out)

        utilization = -out["F"][0]
        min_speed = -out["F"][1]
        # 16 pieces / 60 inventory ≈ 0.267 (post-tweak: useful slots only)
        assert 0.20 < utilization < 0.40, f"utilization odd: {utilization}"
        # All R40 pieces → min speed = 0.97 (R40 cap)
        assert min_speed == pytest.approx(0.97, abs=0.05), f"min_speed odd: {min_speed}"


# =============================================================================
# Open chain → cycle constraint violated
# =============================================================================


class TestOpenChain:
    def test_open_chain_fails_cycle_constraint(self, catalog, config):
        problem = PortPairProblem(catalog, config)
        x = create_empty_chromosome(problem.dims)
        for k in range(8):
            set_piece_slot(x, problem.dims, k, P_R40_LEFT)
        for k in range(7):  # 7 edges, no closing edge
            set_port_pair(x, problem.dims, k, k, PB, k + 1, PA)
        set_anchor(x, problem.dims, 0, 0, 0)

        out = {}
        problem._evaluate(x, out)

        # Cycle count constraint: 1 - 0 = 1 > 0 (infeasible)
        cycle_g = out["G"][-1]
        assert cycle_g > 0, f"open chain should fail cycle constraint, got {cycle_g}"


# =============================================================================
# Inventory excess violations
# =============================================================================


class TestInventoryExcess:
    def test_too_many_curves_triggers_inventory_constraint(self, catalog, config):
        problem = PortPairProblem(catalog, config)
        # Use R40_RIGHT but inventory only has 8
        x = create_empty_chromosome(problem.dims)
        for k in range(16):  # 16 active R40_RIGHT, exceeds inventory of 8
            set_piece_slot(x, problem.dims, k, 3)  # R40_RIGHT
        set_anchor(x, problem.dims, 0, 0, 0)

        out = {}
        problem._evaluate(x, out)

        # Per-type inventory excess for R40_RIGHT (idx 3) is at G[5 + 3] = G[8]
        # G layout: [closure_x, closure_y, closure_theta, boundary, collisions,
        #           inv_0, inv_1, ..., inv_9, loose_ports, cycle_count]
        inv_idx = 5 + 3  # R40_RIGHT is piece index 3
        assert out["G"][inv_idx] > 0, (
            f"inventory excess for R40_RIGHT not flagged: G[{inv_idx}]={out['G'][inv_idx]}"
        )
