"""Tests for IntegerSampling with heuristic closed loop patterns."""

import numpy as np
import pytest

from src.encoding import compute_dimensions
from src.problem import TrackOptimizationProblem
from src.sampling import IntegerSampling


class TestIntegerSampling:
    """Tests for IntegerSampling operator."""

    def test_sampling_shape(self, catalog, default_config):
        """Returns (n_samples, n_var) array."""
        dims = compute_dimensions(default_config.total_inventory)
        sampling = IntegerSampling(catalog, default_config)
        problem = TrackOptimizationProblem(catalog, default_config)

        n_samples = 50
        X = sampling._do(problem, n_samples)

        assert X.shape == (n_samples, dims.n_var)

    def test_values_within_bounds(self, catalog, default_config):
        """All gene values within problem bounds."""
        sampling = IntegerSampling(catalog, default_config)
        problem = TrackOptimizationProblem(catalog, default_config)

        X = sampling._do(problem, 20)

        assert np.all(X >= problem.xl)
        assert np.all(X <= problem.xu)

    def test_heuristic_ratio_respected(self, catalog, default_config):
        """Some samples come from heuristic patterns."""
        sampling = IntegerSampling(
            catalog, default_config,
            heuristic_ratio=0.5,
        )
        problem = TrackOptimizationProblem(catalog, default_config)

        X = sampling._do(problem, 100)

        # With 50% heuristic ratio, at least some samples should be non-trivial
        # (not all inactive/-1)
        active_counts = np.sum(X[:, ::3] >= 0, axis=1)  # piece_type genes every 3
        has_pieces = np.sum(active_counts > 0)
        assert has_pieces > 0

    def test_population_diversity(self, catalog, default_config):
        """Population has diverse chromosomes."""
        sampling = IntegerSampling(catalog, default_config)
        problem = TrackOptimizationProblem(catalog, default_config)

        X = sampling._do(problem, 50)

        # Not all chromosomes should be identical
        unique_rows = len(set(tuple(row) for row in X))
        assert unique_rows > 1
