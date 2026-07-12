"""Genotype-dependent angular target in MainLoopClosureRepair.

A closed loop's turning sum lands on a feasible closed target on its own side:
+/-360 for a single loop (either chirality) or 0 for a self-crossing figure-8.
Repair must reach that target without mutilating an already-closed loop, so a
right-handed loop (-360) is left intact rather than dragged toward +360.
"""

import numpy as np
import pytest

from src.catalog import TrackCatalog
from src.config import OptimizationConfig
from src.decoder import decode_chromosome
from src.encoding import (
    INACTIVE,
    R40_CURVE,
    compute_dimensions,
    create_chromosome_from_pieces,
)
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


class TestClosedLoopIsLeftIntact:
    """Repair must be a fixed point on already-closed loops of ANY chirality."""

    def _active_curve_flips(self, x, dims):
        types = np.asarray(x[:dims.n_main])
        flips = np.asarray(x[dims.main_flips_start:dims.main_flips_end])
        return set(flips[types == int(R40_CURVE)].tolist())

    @pytest.mark.parametrize("gen_name", ["_gen_simple_loop", "_gen_oval", "_gen_racetrack"])
    def test_closed_loop_seeds_survive_repair(self, gen_name, cfg, cat, dims, inv):
        # Generic sweep: every size/chirality variant of each pure-loop family
        # must pass through repair unchanged and still closed. The families emit
        # both handedness variants, so this covers left AND right without listing
        # cases; the final assert stops the test going vacuous (left-only).
        import src.sampling as sampling
        variants = getattr(sampling, gen_name)(inv, dims)
        assert variants, f"{gen_name} produced no seeds under all_pieces"

        seen_chirality = set()
        for pieces, flips, *_ in variants:
            x = create_chromosome_from_pieces(dims, pieces, main_loop_flips=flips)
            before = _n_active(x, dims)

            X = _pipeline(dims, inv, cat)._do(None, np.array([x.copy()]))

            assert _n_active(X[0], dims) == before, f"{gen_name} seed mutilated by repair"
            layout = decode_chromosome(X[0], cat, cfg.inventory, dims=dims)
            assert layout.max_closure_error < cfg.closure_tolerance
            seen_chirality |= {f for p, f in zip(pieces, flips) if p == int(R40_CURVE)}

        assert seen_chirality == {0, 1}, \
            f"{gen_name} must exercise both chiralities, saw {seen_chirality}"

    def test_right_handed_loop_is_idempotent(self, cat, dims, inv):
        # Minimal explicit anchor for the core bug: a right-handed circle (-360).
        x = create_chromosome_from_pieces(
            dims, [int(R40_CURVE)] * 16, main_loop_flips=[1] * 16,
        )
        before = _n_active(x, dims)

        X = _pipeline(dims, inv, cat)._do(None, np.array([x.copy()]))

        assert _n_active(X[0], dims) == before
        assert self._active_curve_flips(X[0], dims) == {1}

    def test_partial_right_loop_closes_toward_minus_360(self, cat, dims, inv):
        # 15 right curves = -337.5; repair adds ONE right curve, not a left one.
        x = create_chromosome_from_pieces(
            dims, [int(R40_CURVE)] * 15, main_loop_flips=[1] * 15,
        )

        X = _pipeline(dims, inv, cat)._do(None, np.array([x.copy()]))

        assert _n_active(X[0], dims) == 16
        assert self._active_curve_flips(X[0], dims) == {1}, \
            "curves must all stay right-handed (target -360)"

    def test_bare_figure_eight_survives_pipeline(self, cfg, cat, dims, inv):
        # Emergent-crossing figure-8 (no descriptor, turning 0): repair must
        # recognize the crossing and hold it at 0, not snap it to +/-360.
        from src.sampling import _gen_figure_eight
        variants = _gen_figure_eight(inv, dims)
        assert variants, "all_pieces config should enable the figure-8 seed"
        pieces, flips, *_ = variants[0]
        x = create_chromosome_from_pieces(dims, pieces, main_loop_flips=flips)
        before = _n_active(x, dims)

        X = _pipeline(dims, inv, cat)._do(None, np.array([x.copy()]))

        assert _n_active(X[0], dims) == before
        layout = decode_chromosome(X[0], cat, cfg.inventory, dims=dims)
        assert layout.paths[0].closure_error < cfg.closure_tolerance
        assert layout.n_cross_pieces == 1
