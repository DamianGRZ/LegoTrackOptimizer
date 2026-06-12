"""Slack-aware compensated-pair GROW mutation.

The one genotype edit that preserves closure EXACTLY: insert two STRAIGHT_16
at loop gaps whose entering headings are anti-parallel — displacements cancel,
turning sum untouched. Grow-only by design: shrinking is BoundaryAwareRepair's
job (corrective, box-conditioned), so the two mechanisms never overlap.
Slack-aware: a pair is only inserted when the box has room for the growth
along that heading's axis, so no doomed over-the-box mutants are produced.
"""

import numpy as np
import pytest

from src.catalog import TrackCatalog
from src.config import OptimizationConfig
from src.decoder import decode_chromosome
from src.encoding import INACTIVE, compute_dimensions, create_chromosome_from_pieces
from src.operators import _compensated_pair_grow


@pytest.fixture(scope="module")
def cat():
    return TrackCatalog.load("data/track_pieces_v2.yaml")


@pytest.fixture(scope="module")
def cfg():
    return OptimizationConfig.load("configs/all_pieces.yaml")


@pytest.fixture(scope="module")
def dims(cfg, cat):
    return compute_dimensions(cfg, cat)


@pytest.fixture(scope="module")
def inv(cfg, cat):
    return {cat._id_to_index[k]: v for k, v in cfg.inventory.items()
            if k in cat._id_to_index}


def _n_active(x, dims):
    return int(np.sum(np.asarray(x[:dims.n_main]) != INACTIVE))


def _decode(x, cfg, cat, dims):
    return decode_chromosome(x, cat, cfg.inventory, dims=dims)


class TestGrowPlainLoop:

    def _racetrack(self, inv, dims):
        from src.sampling import _gen_racetrack
        pieces, flips, *_ = _gen_racetrack(inv, dims)[0]
        return create_chromosome_from_pieces(dims, pieces, main_loop_flips=flips)

    def test_repeated_growth_stays_closed_and_inside_box(self, cfg, cat, dims, inv):
        np.random.seed(7)
        x = self._racetrack(inv, dims)
        before = _n_active(x, dims)
        box_w = dims.boundary_max_x - dims.boundary_min_x
        box_h = dims.boundary_max_y - dims.boundary_min_y
        for _ in range(60):
            _compensated_pair_grow(x, dims, cat)
            lay = _decode(x, cfg, cat, dims)
            assert lay.paths[0].closure_error < cfg.closure_tolerance
            st = lay.paths[0].states
            assert st[:, 0].max() - st[:, 0].min() <= box_w + 1e-6, "grew past box width"
            assert st[:, 1].max() - st[:, 1].min() <= box_h + 1e-6, "grew past box height"
        assert _n_active(x, dims) > before, "60 attempts must grow the loop"

    def test_single_growth_adds_exactly_two_straights(self, cfg, cat, dims, inv):
        np.random.seed(3)
        x = self._racetrack(inv, dims)
        before = _n_active(x, dims)
        assert _compensated_pair_grow(x, dims, cat)
        assert _n_active(x, dims) == before + 2

    def test_tiny_genome_is_noop(self, cfg, cat, dims):
        x = create_chromosome_from_pieces(dims, [2, 2, 2])
        before = x.copy()
        assert not _compensated_pair_grow(x, dims, cat)
        np.testing.assert_array_equal(x, before)


class TestGrowDescriptorGenomes:

    def test_cross_genome_keeps_commit_and_closure(self, cfg, cat, dims, inv):
        from src.sampling import _gen_figure_eight_cross
        np.random.seed(11)
        pieces, flips, _, cjs, _ = _gen_figure_eight_cross(inv, dims)[-1]  # base 34 pcs
        x = create_chromosome_from_pieces(
            dims, pieces, main_loop_flips=flips, cross_junctions=cjs,
        )
        for _ in range(25):
            _compensated_pair_grow(x, dims, cat)
            lay = _decode(x, cfg, cat, dims)
            assert lay.paths[0].closure_error < cfg.closure_tolerance
            assert lay.n_cross_junctions == 1, "CROSS_90 must stay committed"

    def test_dc_genome_keeps_commit_and_closure(self, cfg, cat, dims, inv):
        from src.sampling import _gen_figure_eight_dbl_crossover
        np.random.seed(13)
        pieces, flips, _, _, dcs = _gen_figure_eight_dbl_crossover(inv, dims)[-1]
        x = create_chromosome_from_pieces(
            dims, pieces, main_loop_flips=flips, double_crossovers=dcs,
        )
        for _ in range(25):
            _compensated_pair_grow(x, dims, cat)
            lay = _decode(x, cfg, cat, dims)
            assert lay.paths[0].closure_error < cfg.closure_tolerance
            assert lay.n_dbl_crossovers == 1, "DC must stay committed"

    def test_mutation_dc_branch_can_grow(self, cfg, cat, dims, inv):
        """PartitionedMutation must let DC genomes grow via compensated pairs."""
        from src.operators import PartitionedMutation
        from src.sampling import _gen_figure_eight_dbl_crossover

        np.random.seed(5)
        pieces, flips, _, _, dcs = _gen_figure_eight_dbl_crossover(inv, dims)[-1]
        x = create_chromosome_from_pieces(
            dims, pieces, main_loop_flips=flips, double_crossovers=dcs,
        )
        before = _n_active(x, dims)

        class _P:
            catalog = cat

        mut = PartitionedMutation(dims, prob=1.0)
        X = np.array([x.copy() for _ in range(40)])
        mut._do(_P(), X)
        sizes = {int(np.sum(row[:dims.n_main] != INACTIVE)) for row in X}
        assert any(s > before for s in sizes), \
            "across 40 mutated DC genomes at least one must grow"
