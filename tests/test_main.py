"""Integration tests for main entry point and optimization pipeline."""

import numpy as np
import pytest

from main import run_optimization


class TestOptimizationRuns:
    """Tests for optimization execution."""

    def test_optimization_runs_short(self, catalog, default_config):
        """Can run 5 generations with pop_size=20."""
        default_config.algorithm.n_gen = 5
        default_config.algorithm.pop_size = 20

        res = run_optimization(default_config, catalog, verbose=False)

        assert res is not None
        assert res.pop is not None
        assert len(res.pop) > 0

        pop_F = res.pop.get("F")
        pop_X = res.pop.get("X")

        assert pop_F is not None
        assert pop_X is not None
        assert len(pop_F) == len(pop_X)

        # Bi-objective: 2 columns (utilization + speed)
        assert pop_F.shape[1] == 2

    def test_objectives_in_expected_ranges(self, catalog, default_config):
        """Verify objectives are in expected ranges."""
        default_config.algorithm.n_gen = 10
        default_config.algorithm.pop_size = 30

        res = run_optimization(default_config, catalog, verbose=False)

        pop_F = res.pop.get("F")

        # F[0] = -utilization (should be negative for non-empty layouts)
        assert np.any(pop_F[:, 0] < 0), "Expected negative utilization (maximization)"
        # F[1] = -avg_speed (should be negative for layouts with pieces)
        assert np.any(pop_F[:, 1] < 0), "Expected negative speed (maximization)"
