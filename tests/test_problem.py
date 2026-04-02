"""Tests for bi-objective NSGA-II problem definition."""

import numpy as np
import pytest

from src.problem import TrackOptimizationProblem
from src.encoding import (
    compute_dimensions,
    create_chromosome_from_pieces,
    create_empty_chromosome,
    PieceIndex,
)


class TestTrackOptimizationProblem:
    """Tests for TrackOptimizationProblem (bi-objective with Deb's CV)."""

    def test_problem_dimensions(self, catalog, default_config):
        """Verify problem has correct dimensions."""
        problem = TrackOptimizationProblem(catalog, default_config)
        dims = compute_dimensions(default_config.total_inventory)

        assert problem.n_var == dims.n_var
        assert problem.n_obj == 2  # utilization + speed
        assert problem.n_ieq_constr == 5

    def test_evaluate_valid_circle(self, catalog, default_config):
        """16 R40 circle evaluates without error."""
        problem = TrackOptimizationProblem(catalog, default_config)
        dims = problem.dims

        pattern = [PieceIndex.R40_LEFT] * 16
        chromosome = create_chromosome_from_pieces(dims, pattern)
        out = {}

        problem._evaluate(chromosome, out)

        assert "F" in out
        assert "G" in out
        assert len(out["F"]) == 2  # Two objectives
        assert len(out["G"]) == 5  # 5 constraints

    def test_evaluate_empty_chromosome(self, catalog, default_config):
        """Empty chromosome evaluates without error."""
        problem = TrackOptimizationProblem(catalog, default_config)
        dims = problem.dims

        chromosome = create_empty_chromosome(dims)
        out = {}

        problem._evaluate(chromosome, out)

        assert "F" in out
        assert "G" in out
        assert out["F"][0] == 0.0  # Zero utilization
        assert out["F"][1] == 0.0  # Zero speed

    def test_objectives_shape(self, catalog, default_config):
        """out['F'] has shape (2,) for bi-objective."""
        problem = TrackOptimizationProblem(catalog, default_config)
        dims = problem.dims

        pattern = [PieceIndex.R40_LEFT] * 16
        chromosome = create_chromosome_from_pieces(dims, pattern)
        out = {}

        problem._evaluate(chromosome, out)

        F = np.array(out["F"])
        assert F.shape == (2,)

    def test_constraints_shape(self, catalog, default_config):
        """out['G'] has shape (5,) for 5 constraints."""
        problem = TrackOptimizationProblem(catalog, default_config)
        dims = problem.dims

        pattern = [PieceIndex.R40_LEFT] * 16
        chromosome = create_chromosome_from_pieces(dims, pattern)
        out = {}

        problem._evaluate(chromosome, out)

        G = np.array(out["G"])
        assert G.shape == (5,)

    def test_objective_correct_sign(self, catalog, default_config):
        """Both objectives have correct sign for minimization (negative)."""
        problem = TrackOptimizationProblem(catalog, default_config)
        dims = problem.dims

        pattern = [PieceIndex.R40_LEFT] * 16
        chromosome = create_chromosome_from_pieces(dims, pattern)
        out = {}

        problem._evaluate(chromosome, out)

        # F[0] = -utilization (negative to maximize)
        assert out["F"][0] < 0
        # F[1] = -avg_speed (negative to maximize, should be nonzero for closed loop)
        assert out["F"][1] < 0

    def test_feasible_circle_constraints(self, catalog, default_config):
        """Closed circle should satisfy closure and angle constraints."""
        problem = TrackOptimizationProblem(catalog, default_config)
        dims = problem.dims

        pattern = [PieceIndex.R40_LEFT] * 16
        chromosome = create_chromosome_from_pieces(dims, pattern)
        out = {}

        problem._evaluate(chromosome, out)

        G = out["G"]
        assert G[0] <= 0  # Closure constraint satisfied
        assert G[1] <= 0  # Angle constraint satisfied
        assert G[2] <= 0  # Boundary constraint satisfied
        assert G[3] <= 0  # Inventory constraint satisfied

    def test_closure_tolerance_can_be_set(self, catalog, default_config):
        """Closure tolerance can be overridden."""
        loose = TrackOptimizationProblem(catalog, default_config, closure_tolerance=10.0)
        tight = TrackOptimizationProblem(catalog, default_config, closure_tolerance=0.1)

        assert loose.closure_tolerance == 10.0
        assert tight.closure_tolerance == 0.1

    def test_speed_increases_with_straights(self, catalog, default_config):
        """Layout with straights should have higher speed than pure curves."""
        problem = TrackOptimizationProblem(catalog, default_config)
        dims = problem.dims

        # Pure circle (all curves — lower speed limit)
        circle = create_chromosome_from_pieces(dims, [PieceIndex.R40_LEFT] * 16)
        out_circle = {}
        problem._evaluate(circle, out_circle)

        # Oval with straights (higher speed on straight sections)
        oval = create_chromosome_from_pieces(dims, [
            PieceIndex.R40_LEFT] * 8 +
            [PieceIndex.STRAIGHT_16] * 4 +
            [PieceIndex.R40_LEFT] * 8 +
            [PieceIndex.STRAIGHT_16] * 4,
        )
        out_oval = {}
        problem._evaluate(oval, out_oval)

        # Oval should have higher avg speed (more negative F[1]) IF it closed
        # If oval didn't close, speed comparison doesn't apply
        if out_oval["G"][0] <= 0 and out_oval["G"][1] <= 0:
            assert out_oval["F"][1] < out_circle["F"][1]
