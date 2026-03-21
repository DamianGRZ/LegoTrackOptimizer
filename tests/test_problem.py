"""Tests for pymoo optimization problem.

Note: After migration to random-key encoding, chromosomes use [0,1] values.
Tests use create_chromosome_from_pattern() to encode known piece patterns.
"""

import numpy as np
import pytest

from src.problem import TrackOptimizationProblem, EpsilonTightening
from src.encoding import N_VAR, create_chromosome_from_pattern, create_empty_chromosome


def _get_available_pieces(catalog, inventory):
    """Get sorted list of available piece indices from inventory."""
    available = []
    for piece_id, count in inventory.items():
        idx = catalog._id_to_index.get(piece_id)
        if idx is not None and count > 0:
            available.append(idx)
    return sorted(available)


class TestTrackOptimizationProblem:
    """Tests for TrackOptimizationProblem (single-objective with Deb's CV)."""

    def test_problem_dimensions(self, catalog, default_config):
        """Verify problem has correct dimensions."""
        problem = TrackOptimizationProblem(catalog, default_config)

        assert problem.n_var == N_VAR  # Fixed chromosome length
        assert problem.n_obj == 1  # Single objective: -utilization
        assert problem.n_ieq_constr == 5  # closure, angle, boundary, inventory, loose_ports

    def test_evaluate_valid_circle(self, catalog, default_config):
        """16 R40 circle evaluates without error."""
        problem = TrackOptimizationProblem(catalog, default_config)
        available = _get_available_pieces(catalog, default_config.inventory)

        # Create chromosome with 16 R40_LEFT pieces (index 2)
        pattern = [2] * 16
        chromosome = create_chromosome_from_pattern(pattern, available)
        out = {}

        problem._evaluate(chromosome, out)

        assert "F" in out
        assert "G" in out
        assert len(out["F"]) == 1  # Single objective
        assert len(out["G"]) == 5  # 5 constraints (closure, angle, boundary, inventory, loose_ports)

    def test_evaluate_minimal_chromosome(self, catalog, default_config):
        """Minimal RK chromosome (NEAT-style) evaluates without error.

        With NEAT-style complexification, empty chromosomes have main loop
        genes at 0.0 (inactive), producing minimal layouts. Complexity
        grows through ComplexificationMutation.
        """
        problem = TrackOptimizationProblem(catalog, default_config)
        chromosome = create_empty_chromosome()  # Main loop inactive, other segments random
        out = {}

        problem._evaluate(chromosome, out)

        assert "F" in out
        assert "G" in out
        # Minimal chromosome produces minimal/empty layout with 0 utilization
        assert out["F"][0] == 0.0 or out["F"][0] < 0  # Either empty or has some pieces

    def test_objectives_shape(self, catalog, default_config):
        """out['F'] has shape (1,) for single objective."""
        problem = TrackOptimizationProblem(catalog, default_config)
        available = _get_available_pieces(catalog, default_config.inventory)

        pattern = [0, 0, 0, 0]  # Some straights
        chromosome = create_chromosome_from_pattern(pattern, available)
        out = {}

        problem._evaluate(chromosome, out)

        F = np.array(out["F"])
        assert F.shape == (1,)

    def test_constraints_shape(self, catalog, default_config):
        """out['G'] has shape (4,) for 4 constraints."""
        problem = TrackOptimizationProblem(catalog, default_config)
        available = _get_available_pieces(catalog, default_config.inventory)

        pattern = [0, 0, 0, 0]
        chromosome = create_chromosome_from_pattern(pattern, available)
        out = {}

        problem._evaluate(chromosome, out)

        G = np.array(out["G"])
        assert G.shape == (5,)

    def test_objective_correct_sign(self, catalog, default_config):
        """Objective has correct sign for minimization (negative utilization)."""
        problem = TrackOptimizationProblem(catalog, default_config)
        available = _get_available_pieces(catalog, default_config.inventory)

        # Create valid circle
        pattern = [2] * 16  # 16 R40_LEFT
        chromosome = create_chromosome_from_pattern(pattern, available)
        out = {}

        problem._evaluate(chromosome, out)

        # F[0] = -utilization (negative to maximize)
        assert out["F"][0] < 0

    def test_feasible_circle_constraints(self, catalog, default_config):
        """Closed circle should satisfy closure and angle constraints."""
        problem = TrackOptimizationProblem(catalog, default_config)
        available = _get_available_pieces(catalog, default_config.inventory)

        # Create valid 16-piece circle
        pattern = [2] * 16  # 16 R40_LEFT = 360 degrees
        chromosome = create_chromosome_from_pattern(pattern, available)
        out = {}

        problem._evaluate(chromosome, out)

        # Should be feasible (G <= 0)
        G = out["G"]
        assert G[0] <= 0  # Closure constraint satisfied
        assert G[1] <= 0  # Angle constraint satisfied
        assert G[2] <= 0  # Boundary constraint satisfied
        assert G[3] <= 0  # Inventory constraint satisfied (16 <= 20 available)

    def test_inventory_violation_computed(self, catalog, default_config):
        """Inventory violation is computed correctly."""
        problem = TrackOptimizationProblem(catalog, default_config)
        available = _get_available_pieces(catalog, default_config.inventory)

        # The decoder should prevent actual inventory violations,
        # but we can verify the constraint is being checked
        pattern = [2] * 16
        chromosome = create_chromosome_from_pattern(pattern, available)
        out = {}

        problem._evaluate(chromosome, out)

        # G[3] is inventory violation - should be <= 0 for valid layout
        assert out["G"][3] <= 0

    def test_closure_tolerance_can_be_set(self, catalog, default_config):
        """Closure tolerance can be overridden."""
        loose_problem = TrackOptimizationProblem(
            catalog, default_config,
            closure_tolerance=10.0
        )
        tight_problem = TrackOptimizationProblem(
            catalog, default_config,
            closure_tolerance=0.1
        )

        assert loose_problem.closure_tolerance == 10.0
        assert tight_problem.closure_tolerance == 0.1


class TestEpsilonTightening:
    """Tests for EpsilonTightening callback."""

    def test_callback_initialization(self):
        """Callback initializes with correct parameters."""
        callback = EpsilonTightening(
            initial_tol=8.0,
            final_tol=0.5,
            tighten_until=0.7
        )

        assert callback.initial_tol == 8.0
        assert callback.final_tol == 0.5
        assert callback.tighten_until == 0.7

    def test_tolerance_at_start(self, catalog, default_config):
        """At generation 0, tolerance should be initial_tol."""
        # Create a mock algorithm object
        class MockTermination:
            n_max_gen = 100

        class MockAlgorithm:
            n_gen = 1
            termination = MockTermination()
            problem = TrackOptimizationProblem(catalog, default_config)

        callback = EpsilonTightening(
            initial_tol=8.0,
            final_tol=0.5,
            tighten_until=0.7
        )

        algorithm = MockAlgorithm()
        callback.notify(algorithm)

        # At gen 1/100 = 1%, tolerance should be close to initial
        expected = 8.0 * (1 - 0.01/0.7) + 0.5 * (0.01/0.7)
        assert abs(algorithm.problem.closure_tolerance - expected) < 0.1

    def test_tolerance_at_end(self, catalog, default_config):
        """After tighten_until, tolerance should be final_tol."""
        class MockTermination:
            n_max_gen = 100

        class MockAlgorithm:
            n_gen = 80  # 80% > 70% (tighten_until)
            termination = MockTermination()
            problem = TrackOptimizationProblem(catalog, default_config)

        callback = EpsilonTightening(
            initial_tol=8.0,
            final_tol=0.5,
            tighten_until=0.7
        )

        algorithm = MockAlgorithm()
        callback.notify(algorithm)

        # Past tighten_until, should be at final tolerance
        assert algorithm.problem.closure_tolerance == 0.5


# Backward compatibility aliases should work
class TestBackwardCompatibility:
    """Test that backward compatibility aliases work."""

    def test_aliases_exist(self):
        """Backward compatibility aliases should import without error."""
        from src.problem import (
            TrackOptimizationProblem,
            MultiSegmentProblem,
            SingleObjectiveProblem,
            TrackLayoutProblem,
        )

        # All should be the same class
        assert MultiSegmentProblem is TrackOptimizationProblem
        assert SingleObjectiveProblem is TrackOptimizationProblem
        assert TrackLayoutProblem is TrackOptimizationProblem
