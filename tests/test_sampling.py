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

    def test_zero_heuristic_ratio_means_no_seeds(self, catalog, default_config):
        """ratio=0 must mean a PURE random population — with elite archives in
        play, even a single smuggled seed can win the whole run and confound
        any seeded-vs-unseeded experiment."""
        sampling = IntegerSampling(catalog, default_config, heuristic_ratio=0.0)
        assert sampling._n_heuristic(1000) == 0

    def test_positive_ratio_guarantees_at_least_one_seed(self, catalog, default_config):
        sampling = IntegerSampling(catalog, default_config, heuristic_ratio=0.001)
        assert sampling._n_heuristic(100) == 1

    def test_ratio_scales_seed_count(self, catalog, default_config):
        sampling = IntegerSampling(catalog, default_config, heuristic_ratio=0.2)
        assert sampling._n_heuristic(1000) == 200

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
        dims = compute_dimensions(default_config, catalog)
        active_counts = np.sum(X[:, :dims.n_main] >= 0, axis=1)  # main-loop type genes
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
        inventory = {"STRAIGHT_16": 80, "R40_CURVE": 20}
        small = OptimizationConfig(
            train_config_path="trains/measured_consist.yaml",
            inventory=inventory,
            boundary=BoundaryConfig(min_x=-100, max_x=100, min_y=-100, max_y=100),
        )
        large = OptimizationConfig(
            train_config_path="trains/measured_consist.yaml",
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
        with_switch = sum(
            1 for _pieces, _flips, junctions, _cross, _dc in patterns
            if junctions and any(active == 1 for active, _pos, _hand, _n_str in junctions)
        )
        assert with_switch / len(patterns) >= 0.33

    def test_two_siding_pattern_present_when_junctions_ge_2(
        self, catalog, switches_config,
    ):
        """When max_junctions >= 2, the pool includes at least one 2-siding seed."""
        patterns = IntegerSampling(catalog, switches_config)._get_heuristic_patterns(
            np.random.default_rng()
        )
        assert any(
            p[2] is not None
            and sum(1 for junc in p[2] if junc[0] == 1) == 2
            for p in patterns
        )

    def test_two_siding_seed_decodes_to_two_committed_sidings(
        self, catalog, switches_config,
    ):
        """The 2-siding seed must DECODE to a layout committing BOTH sidings —
        not merely exist as a pattern. Before the per-section walk-fit gate was
        added, the decoder dropped both junctions ("no OUT position at the
        required main distance")."""
        from src.decoder import decode_chromosome
        from src.encoding import create_chromosome_from_pieces
        from src.sampling import _gen_oval_two_sidings

        dims = compute_dimensions(switches_config, catalog)
        inv = catalog.inventory_by_index(switches_config.inventory)
        patterns = _gen_oval_two_sidings(inv, dims)
        assert patterns, "expected at least one 2-siding seed for switches_config"

        for pieces, flips, junctions, cross, dc in patterns:
            x = create_chromosome_from_pieces(
                dims, pieces, main_loop_flips=flips,
                junctions=junctions, cross_junctions=cross, double_crossovers=dc,
            )
            layout = decode_chromosome(x, catalog, switches_config.inventory, dims=dims)
            assert layout.drop_log == [], f"decoder dropped descriptors: {layout.drop_log}"
            assert len(layout.switch_pairs) == 2, (
                f"expected 2 committed sidings, got {len(layout.switch_pairs)}"
            )
            assert layout.max_closure_error < switches_config.closure_tolerance

    def test_seed_makes_initial_population_reproducible(self, catalog, switches_config):
        """config.algorithm.seed must control the initial population: same seed ->
        identical X, different seed -> different X. (seed=None stays random.)"""
        problem = TrackOptimizationProblem(catalog, switches_config)
        x_a = IntegerSampling(catalog, switches_config, seed=1234)._do(problem, 40)
        x_b = IntegerSampling(catalog, switches_config, seed=1234)._do(problem, 40)
        x_c = IntegerSampling(catalog, switches_config, seed=9999)._do(problem, 40)
        assert np.array_equal(x_a, x_b)
        assert not np.array_equal(x_a, x_c)

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
    """Figure-8 CROSS_90 seed: a closed self-crossing loop with one (active,
    pos_1, pos_2) cross descriptor at the perpendicular crossing slots. The
    geometry validates immediately, so the decoder places a real CROSS_90."""

    def _config_with_cross(self) -> OptimizationConfig:
        """Build a config with switches + spurs + CROSS_90 inventory and a
        boundary large enough to host the seeded oval."""
        from src.config import AlgorithmConfig
        cfg = OptimizationConfig(
            train_config_path="trains/measured_consist.yaml",
            inventory={
                "STRAIGHT_16": 80,
                "R40_CURVE": 40,
                "R40_SWITCH_LEFT": 4,
                "R40_SWITCH_RIGHT": 4,
                "CROSS_90": 4,
            },
            boundary=BoundaryConfig(min_x=-200, max_x=200, min_y=-200, max_y=200),
            algorithm=AlgorithmConfig(name="NSGA2", pop_size=20, n_gen=5),
        )
        return cfg

    def test_cross_junction_pattern_emitted(self, catalog):
        cfg = self._config_with_cross()
        sampling = IntegerSampling(catalog, cfg)
        patterns = sampling._get_heuristic_patterns(np.random.default_rng())

        with_cross = [p for p in patterns if p[3]]  # index 3 = cross descriptors
        assert with_cross, "expected at least one pattern with cross-junction descriptors"

        # Each cross pattern carries an ACTIVE (pos_1, pos_2) descriptor.
        for pat in with_cross:
            for desc in pat[3]:
                active, pos_1, pos_2 = desc
                assert active == 1
                assert pos_1 != pos_2
                assert 0 <= pos_1 < len(pat[0])
                assert 0 <= pos_2 < len(pat[0])

    def test_cross_junction_not_emitted_without_inventory(self, catalog):
        """No cross-junction patterns when CROSS_90 (or switches) absent."""
        from src.config import AlgorithmConfig
        cfg = OptimizationConfig(
            train_config_path="trains/measured_consist.yaml",
            inventory={
                "STRAIGHT_16": 80,
                "R40_CURVE": 40,
                # No switches, no CROSS_90.
            },
            boundary=BoundaryConfig(min_x=-200, max_x=200, min_y=-200, max_y=200),
            algorithm=AlgorithmConfig(name="NSGA2", pop_size=20, n_gen=5),
        )
        sampling = IntegerSampling(catalog, cfg)
        patterns = sampling._get_heuristic_patterns(np.random.default_rng())
        with_cross = [p for p in patterns if p[3]]
        assert with_cross == []
