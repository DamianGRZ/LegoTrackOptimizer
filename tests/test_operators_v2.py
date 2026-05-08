"""Tests for src_v2/operators.py — sampling, crossover, mutation."""

from pathlib import Path

import numpy as np
import pytest

from src_v2.catalog import TrackCatalog
from src_v2.config import BoundaryConfig, OptimizationConfig
from src_v2.decoder import DecoderConfig, decode_chromosome
from src_v2.encoding import (
    DTYPE,
    INACTIVE,
    PortPairDimensions,
    iter_active_pairs,
    iter_active_slots,
    validate_chromosome,
)
from src_v2.operators import (
    PortPairCrossover,
    PortPairMutation,
    PortPairSampling,
)
from src_v2.repair import PortPairRepairPipeline


@pytest.fixture(scope="module")
def catalog():
    return TrackCatalog.load(Path("data/track_pieces_v2.yaml"))


@pytest.fixture
def config():
    return OptimizationConfig(
        inventory={
            "STRAIGHT_16": 12,
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
    )


@pytest.fixture
def small_dims():
    return PortPairDimensions(N_max=64, E_max=64)


@pytest.fixture
def repair(small_dims, catalog, config):
    return PortPairRepairPipeline(small_dims, catalog, config.inventory)


# =============================================================================
# Sampling
# =============================================================================


class TestSampling:
    def test_produces_correct_shape(self, small_dims, catalog, config):
        sampler = PortPairSampling(small_dims, catalog, config, seed=42)
        X = sampler._do(None, n_samples=10)
        assert X.shape == (10, small_dims.n_var)
        assert X.dtype == DTYPE

    def test_validates_individually(self, small_dims, catalog, config):
        sampler = PortPairSampling(small_dims, catalog, config, seed=42)
        X = sampler._do(None, n_samples=20)
        for i in range(len(X)):
            errors = validate_chromosome(X[i], small_dims)
            assert errors == [], f"Individual {i} has errors: {errors}"

    def test_heuristic_individuals_decode_to_valid_topology(
        self, small_dims, catalog, config,
    ):
        """Heuristic seeds should decode without crashing and produce graphs
        with at least 1 connected component."""
        sampler = PortPairSampling(
            small_dims, catalog, config, heuristic_ratio=1.0, seed=42,
        )
        X = sampler._do(None, n_samples=10)
        decoder_config = DecoderConfig(
            boundary_min_x=-100, boundary_max_x=100,
            boundary_min_y=-100, boundary_max_y=100,
        )
        for i in range(len(X)):
            graph = decode_chromosome(X[i], small_dims, catalog, decoder_config)
            assert graph.n_components >= 1, f"Heuristic seed {i} produced empty graph"

    def test_heuristic_produces_at_least_one_closed_cycle_seed(
        self, small_dims, catalog, config,
    ):
        """Among heuristic seeds, at least one should decode to ≥ 1 cycle."""
        sampler = PortPairSampling(
            small_dims, catalog, config, heuristic_ratio=1.0, seed=42,
        )
        X = sampler._do(None, n_samples=20)
        decoder_config = DecoderConfig()
        any_with_cycles = False
        for i in range(len(X)):
            graph = decode_chromosome(X[i], small_dims, catalog, decoder_config)
            if graph.n_cycles >= 1:
                any_with_cycles = True
                break
        assert any_with_cycles, "No heuristic seed produced a closed cycle"

    def test_random_individuals_produce_some_active_slots(
        self, small_dims, catalog, config,
    ):
        sampler = PortPairSampling(
            small_dims, catalog, config, heuristic_ratio=0.0, seed=42,
        )
        X = sampler._do(None, n_samples=20)
        for i in range(len(X)):
            n_active = sum(1 for _ in iter_active_slots(X[i], small_dims))
            assert n_active >= 4, f"Random individual {i} has too few active slots"


# =============================================================================
# Crossover
# =============================================================================


class TestCrossover:
    def test_output_shape(self, small_dims, catalog, config):
        sampler = PortPairSampling(small_dims, catalog, config, seed=42)
        X_pop = sampler._do(None, n_samples=4)
        crossover = PortPairCrossover(small_dims)

        # pymoo expects shape (n_parents, n_matings, n_var)
        X = X_pop[:2].reshape(2, 1, small_dims.n_var)
        Y = crossover._do(None, X)
        assert Y.shape == (2, 1, small_dims.n_var)
        assert Y.dtype == DTYPE

    def test_offspring_validate(self, small_dims, catalog, config, repair):
        sampler = PortPairSampling(small_dims, catalog, config, seed=42)
        X_pop = sampler._do(None, n_samples=20)
        crossover = PortPairCrossover(small_dims)

        # 5 random pair offspring rounds
        for _ in range(5):
            i, j = np.random.choice(len(X_pop), size=2, replace=False)
            X = np.stack([X_pop[i], X_pop[j]]).reshape(2, 1, small_dims.n_var)
            Y = crossover._do(None, X)
            for k in range(2):
                offspring = Y[k, 0]
                # Crossover output may have invalid edges; repair should fix
                repair._repair_one(offspring)
                errors = validate_chromosome(offspring, small_dims)
                assert errors == [], f"Offspring has errors after repair: {errors}"


# =============================================================================
# Mutation
# =============================================================================


class TestMutation:
    @pytest.fixture
    def mutation(self, small_dims, catalog, config):
        return PortPairMutation(small_dims, catalog, config, prob=1.0)

    def test_output_shape_unchanged(self, small_dims, catalog, config, mutation):
        sampler = PortPairSampling(small_dims, catalog, config, seed=42)
        X = sampler._do(None, n_samples=10)
        Y = mutation._do(None, X.copy())
        assert Y.shape == X.shape
        assert Y.dtype == DTYPE

    def test_mutated_chromosomes_validate_after_repair(
        self, small_dims, catalog, config, repair, mutation,
    ):
        sampler = PortPairSampling(small_dims, catalog, config, seed=42)
        X = sampler._do(None, n_samples=20)
        Y = mutation._do(None, X.copy())
        for i in range(len(Y)):
            repair._repair_one(Y[i])
            errors = validate_chromosome(Y[i], small_dims)
            assert errors == [], f"Mutated individual {i} fails validation: {errors}"

    def test_each_subop_runs(self, small_dims, catalog, config):
        """Force each sub-operator individually; ensure no crashes."""
        from src_v2.operators import PortPairMutation
        m = PortPairMutation(small_dims, catalog, config, prob=1.0)
        sampler = PortPairSampling(small_dims, catalog, config, seed=42)
        X = sampler._do(None, n_samples=2)

        ops = [
            m._mutate_piece_type,
            m._activate_slot,
            m._deactivate_slot,
            m._add_edge,
            m._remove_edge,
            m._rewire_edge,
            m._perturb_anchor,
        ]
        for op in ops:
            op(X[0].copy())  # must not raise


# =============================================================================
# Combined operator pipeline (sample -> crossover -> mutate -> repair)
# =============================================================================


class TestFullOperatorPipeline:
    def test_sample_crossover_mutate_repair_decodes(
        self, small_dims, catalog, config, repair,
    ):
        sampler = PortPairSampling(small_dims, catalog, config, seed=42)
        crossover = PortPairCrossover(small_dims)
        mutation = PortPairMutation(small_dims, catalog, config, prob=1.0)

        X = sampler._do(None, n_samples=10)

        # Cross
        for _ in range(5):
            i, j = np.random.choice(len(X), size=2, replace=False)
            parents = np.stack([X[i], X[j]]).reshape(2, 1, small_dims.n_var)
            offspring = crossover._do(None, parents)
            X[i] = offspring[0, 0]
            X[j] = offspring[1, 0]

        # Mutate
        X = mutation._do(None, X)

        # Repair
        for i in range(len(X)):
            repair._repair_one(X[i])

        # Decode all
        decoder_config = DecoderConfig()
        for i in range(len(X)):
            graph = decode_chromosome(X[i], small_dims, catalog, decoder_config)
            # Just verify it doesn't crash and produces a coherent graph
            assert graph.n_slots >= 0
