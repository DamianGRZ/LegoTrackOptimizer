"""Tests for IntegerSampling with heuristic closed loop patterns."""

import numpy as np
import pytest

from src.config import BoundaryConfig, OptimizationConfig
from src.encoding import compute_dimensions
from src.problem import TrackOptimizationProblem
from src.sampling import IntegerSampling


class TestIntegerSampling:
    """Tests for IntegerSampling operator."""

    def test_sampling_shape(self, catalog, default_config):
        """Returns (n_samples, n_var) array."""
        dims = compute_dimensions(default_config, catalog)
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

    def test_oval_size_scales_with_boundary(self, catalog):
        """Larger boundary -> larger seeded layouts. Proves hardcoded caps are gone."""
        inventory = {"STRAIGHT_16": 80, "R40_CURVE": 20, "R40_CURVE": 20}
        small = OptimizationConfig(
            inventory=inventory,
            boundary=BoundaryConfig(min_x=-100, max_x=100, min_y=-100, max_y=100),
        )
        large = OptimizationConfig(
            inventory=inventory,
            boundary=BoundaryConfig(min_x=-250, max_x=250, min_y=-250, max_y=250),
        )
        pats_small = IntegerSampling(catalog, small)._get_heuristic_patterns(
            np.random.default_rng()
        )
        pats_large = IntegerSampling(catalog, large)._get_heuristic_patterns(
            np.random.default_rng()
        )

        def max_pieces(pats):
            return max(len(p[0]) for p in pats)

        assert max_pieces(pats_large) > max_pieces(pats_small)

    def test_switch_seed_fraction_at_least_third(self, catalog, switches_config):
        """At least 33% of heuristic seeds carry an active junction when max_junctions >= 2."""
        patterns = IntegerSampling(catalog, switches_config)._get_heuristic_patterns(
            np.random.default_rng()
        )
        assert patterns, "expected non-empty heuristic patterns for switches_config"
        with_switch = sum(1 for p in patterns if p[1])
        assert with_switch / len(patterns) >= 0.33

    def test_two_siding_pattern_present_when_junctions_ge_2(
        self, catalog, switches_config,
    ):
        """When max_junctions >= 2, the pool includes at least one 2-siding seed."""
        patterns = IntegerSampling(catalog, switches_config)._get_heuristic_patterns(
            np.random.default_rng()
        )
        assert any(
            p[1] is not None
            and sum(1 for junc in p[1] if junc[0] == 1) == 2
            for p in patterns
        )

    def test_seeds_respect_inventory(self, catalog, switches_config):
        """No heuristic seed overuses any main-loop piece type."""
        sampling = IntegerSampling(catalog, switches_config)
        patterns = sampling._get_heuristic_patterns(np.random.default_rng())
        inv = sampling.inventory_by_index

        for pat in patterns:
            pieces = pat[0]
            counts: dict[int, int] = {}
            for p in pieces:
                counts[p] = counts.get(p, 0) + 1
            for idx, c in counts.items():
                assert c <= inv.get(idx, 0), (
                    f"variant overuses piece {idx}: {c} > {inv.get(idx, 0)}"
                )


class TestCrossJunctionSeeder:
    """Soft-seed cross-junction patterns: descriptor present, geometry not yet
    guaranteed to validate. The decoder skips invalid descriptors silently;
    the active flag enters the gene pool so crossover/mutation can carry it
    into individuals whose evolved geometry happens to fit."""

    def _config_with_cross(self, tmp_path) -> OptimizationConfig:
        """Build a config with switches + spurs + CROSS_90 inventory and a
        boundary large enough to host the seeded oval."""
        from src.config import AlgorithmConfig
        cfg = OptimizationConfig(
            train_config_path="trains/default.yaml",
            inventory={
                "STRAIGHT_16": 80,
                "R40_CURVE": 40,
                "R40_CURVE": 40,
                "R40_SWITCH_LEFT": 4,
                "R40_SWITCH_RIGHT": 4,
                "CROSS_90": 4,
            },
            boundary=BoundaryConfig(min_x=-200, max_x=200, min_y=-200, max_y=200),
            algorithm=AlgorithmConfig(name="NSGA2", pop_size=20, n_gen=5),
        )
        return cfg

    def test_cross_junction_pattern_emitted(self, catalog, tmp_path):
        cfg = self._config_with_cross(tmp_path)
        sampling = IntegerSampling(catalog, cfg)
        patterns = sampling._get_heuristic_patterns(np.random.default_rng())

        with_cross = [p for p in patterns if p[2]]
        assert with_cross, "expected at least one pattern with cross-junction descriptors"

        # Each cross-junction pattern carries an ACTIVE descriptor.
        for pat in with_cross:
            for desc in pat[2]:
                active, position_w, handedness = desc
                assert active == 1
                assert handedness in (0, 1)
                assert 0 <= position_w < len(pat[0])

    def test_cross_junction_not_emitted_without_inventory(self, catalog):
        """No cross-junction patterns when CROSS_90 (or switches) absent."""
        from src.config import AlgorithmConfig
        cfg = OptimizationConfig(
            train_config_path="trains/default.yaml",
            inventory={
                "STRAIGHT_16": 80,
                "R40_CURVE": 40,
                "R40_CURVE": 40,
                # No switches, no CROSS_90.
            },
            boundary=BoundaryConfig(min_x=-200, max_x=200, min_y=-200, max_y=200),
            algorithm=AlgorithmConfig(name="NSGA2", pop_size=20, n_gen=5),
        )
        sampling = IntegerSampling(catalog, cfg)
        patterns = sampling._get_heuristic_patterns(np.random.default_rng())
        with_cross = [p for p in patterns if p[2]]
        assert with_cross == []
