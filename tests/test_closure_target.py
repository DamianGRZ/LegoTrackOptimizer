"""Genotype-dependent angular target in MainLoopClosureRepair.

A closed self-crossing loop (figure-8) has a turning sum of 0 deg, not 360.
Genomes carrying an active cross/DC descriptor must be repaired toward the
nearest of {0, 360, 720} so the repair stops mutilating crossing seeds;
plain genomes keep the 360-deg target unchanged.
"""

import numpy as np
import pytest

from src.catalog import TrackCatalog
from src.config import OptimizationConfig
from src.decoder import decode_chromosome
from src.encoding import INACTIVE, compute_dimensions, create_chromosome_from_pieces
from src.repair import TrackRepairPipeline


@pytest.fixture(scope="module")
def cat():
    return TrackCatalog.load("data/track_pieces_v2.yaml")


@pytest.fixture(scope="module")
def cfg(cat):
    return OptimizationConfig.load("configs/all_pieces.yaml")


@pytest.fixture(scope="module")
def dims(cfg, cat):
    return compute_dimensions(cfg, cat)


@pytest.fixture(scope="module")
def inv(cfg, cat):
    return {cat._id_to_index[k]: v for k, v in cfg.inventory.items()
            if k in cat._id_to_index}


def _pipeline(dims, inv, cat):
    return TrackRepairPipeline(
        dims=dims, inventory_by_index=inv, catalog_fk_table=cat._fk_table,
    )


def _n_active(x, dims):
    return int(np.sum(np.asarray(x[:dims.n_main]) != INACTIVE))


class TestCrossingGenomesKeepTurningZero:

    def test_figure_eight_cross_seed_survives_pipeline_intact(self, cfg, cat, dims, inv):
        from src.sampling import _gen_figure_eight_cross
        pieces, flips, _, cjs, _ = _gen_figure_eight_cross(inv, dims)[0]
        x = create_chromosome_from_pieces(
            dims, pieces, main_loop_flips=flips, cross_junctions=cjs,
        )
        before = _n_active(x, dims)

        X = _pipeline(dims, inv, cat)._do(None, np.array([x.copy()]))
        after = _n_active(X[0], dims)

        assert after == before, "repair must not add/remove pieces on a closed figure-8"
        layout = decode_chromosome(X[0], cat, cfg.inventory, dims=dims)
        assert layout.paths[0].closure_error < cfg.closure_tolerance
        assert layout.n_cross_junctions == 1

    def test_dc_figure_eight_seed_survives_pipeline_intact(self, cfg, cat, dims, inv):
        from src.sampling import _gen_figure_eight_dbl_crossover
        pieces, flips, _, _, dcs = _gen_figure_eight_dbl_crossover(inv, dims)[0]
        x = create_chromosome_from_pieces(
            dims, pieces, main_loop_flips=flips, double_crossovers=dcs,
        )
        before = _n_active(x, dims)

        X = _pipeline(dims, inv, cat)._do(None, np.array([x.copy()]))
        after = _n_active(X[0], dims)

        assert after == before
        layout = decode_chromosome(X[0], cat, cfg.inventory, dims=dims)
        assert layout.paths[0].closure_error < cfg.closure_tolerance
        assert layout.n_dbl_crossovers == 1


class TestPlainGenomesKeep360Target:

    def test_left_racetrack_missing_curve_still_repaired_toward_360(self, cfg, cat, dims, inv):
        from src.sampling import _gen_racetrack
        pieces, flips, *_ = _gen_racetrack(inv, dims)[0]
        x = create_chromosome_from_pieces(dims, pieces, main_loop_flips=flips)
        # Remove one corner curve: total angle 337.5 -> repair must add it back.
        curve_slots = [i for i, p in enumerate(pieces) if p == 2]
        x[curve_slots[0]] = INACTIVE
        before = _n_active(x, dims)

        X = _pipeline(dims, inv, cat)._do(None, np.array([x.copy()]))
        after = _n_active(X[0], dims)

        assert after == before + 1, "plain loop must still be repaired toward 360"
