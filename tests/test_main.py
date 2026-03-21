"""Integration tests for main entry point and optimization pipeline."""

import numpy as np
import pytest

from main import run_optimization


class TestOptimizationRuns:
    """Tests for optimization execution."""

    def test_optimization_runs_short(self, catalog, default_config):
        """Can run 5 generations with pop_size=20."""
        # Override config for fast test
        default_config.algorithm.n_gen = 5
        default_config.algorithm.pop_size = 20

        res = run_optimization(default_config, catalog, verbose=False)

        # Verify results are valid
        assert res is not None

        # For single-objective optimization, res.X may be None if no feasible solution found
        # Check the final population instead
        assert res.pop is not None
        assert len(res.pop) > 0

        pop_F = res.pop.get("F")
        pop_X = res.pop.get("X")
        pop_G = res.pop.get("G")

        # Verify population has correct shapes
        assert pop_F is not None
        assert pop_X is not None
        assert len(pop_F) == len(pop_X)

        # Verify objectives have correct shape (1 objective: -utilization)
        assert pop_F.shape[1] == 1

        # Verify at least some solutions exist in the population
        # (feasibility is checked separately)
        assert len(pop_F) > 0

    def test_objectives_in_expected_ranges(self, catalog, default_config):
        """Verify objectives are in expected ranges."""
        # Quick test
        default_config.algorithm.n_gen = 10
        default_config.algorithm.pop_size = 30

        res = run_optimization(default_config, catalog, verbose=False)

        # Check population fitness values
        pop_F = res.pop.get("F")

        # Check objective is in expected range
        # F[0] = -utilization (should be negative for non-empty layouts)
        assert np.any(pop_F[:, 0] < 0), "Expected negative utilization (maximization)"
