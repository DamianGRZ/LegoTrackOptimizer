"""Slack-aware compensated-pair GROW mutation.

The one genotype edit that preserves closure EXACTLY: insert two equal
straights at loop gaps whose entering headings are anti-parallel —
displacements cancel, turning sum untouched. STRAIGHT_16 preferred, a
STRAIGHT_24 pair once the 16s are exhausted (equal opposite displacement is
what compensates, not the piece type). Grow-only by design: shrinking is
BoundaryAwareRepair's job (corrective, box-conditioned), so the two
mechanisms never overlap.
Slack-aware: a pair is only inserted when the box has room for the growth
along that heading's axis, so no doomed over-the-box mutants are produced.
"""

import numpy as np
import pytest

from src.catalog import TrackCatalog
from src.config import OptimizationConfig
from src.decoder import decode_chromosome
from src.encoding import INACTIVE, compute_dimensions, create_chromosome_from_pieces
from src.operators import _active_view, _compensated_pair_grow


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


class TestPerTypeStraightBudget:
    """The straight budget is per type, not the combined S16+S24 total.

    Compensation needs equal opposite displacement, not identical pieces:
    an anti-parallel STRAIGHT_24 pair preserves closure exactly too. The
    operator prefers STRAIGHT_16 (finer step) and falls back to a
    STRAIGHT_24 pair when the 16s are exhausted; with no free pair of
    either type it declines instead of inserting pieces the inventory
    cannot supply.
    """

    BOUNDARY = {"min_x": -250.0, "max_x": 250.0,
                "min_y": -250.0, "max_y": 250.0}
    # Closed square racetrack: 4 corners of 4 R40 + runs of 2 S16.
    RACETRACK = ([2] * 4 + [0] * 2) * 4

    def _setup(self, cat, s16, s24):
        cfg = OptimizationConfig(
            train_config_path="trains/measured_consist.yaml",
            inventory={"STRAIGHT_16": s16, "STRAIGHT_24": s24,
                       "R40_CURVE": 16},
            boundary=dict(self.BOUNDARY),
        )
        dims = compute_dimensions(cfg, cat)
        x = create_chromosome_from_pieces(dims, list(self.RACETRACK))
        return cfg, dims, x

    def _straight_counts(self, x, dims):
        main = np.asarray(x[:dims.n_main])
        return int(np.sum(main == 0)), int(np.sum(main == 1))

    def test_exhausted_16s_fall_back_to_a_straight_24_pair(self, cat):
        np.random.seed(17)
        cfg, dims, x = self._setup(cat, s16=8, s24=4)  # all 8 S16 in use
        assert _compensated_pair_grow(x, dims, cat)
        n16, n24 = self._straight_counts(x, dims)
        assert (n16, n24) == (8, 2), "pair must be STRAIGHT_24 — no spare 16s"
        lay = _decode(x, cfg, cat, dims)
        assert lay.paths[0].closure_error < cfg.closure_tolerance

    def test_no_free_pair_of_either_type_is_a_noop(self, cat):
        np.random.seed(19)
        cfg, dims, x = self._setup(cat, s16=9, s24=1)  # 1 spare of each type
        before = x.copy()
        assert not _compensated_pair_grow(x, dims, cat)
        np.testing.assert_array_equal(x, before)

    def test_prefers_straight_16_when_both_types_have_room(self, cat):
        np.random.seed(23)
        cfg, dims, x = self._setup(cat, s16=12, s24=4)
        assert _compensated_pair_grow(x, dims, cat)
        n16, n24 = self._straight_counts(x, dims)
        assert (n16, n24) == (10, 0)


def _inject_holes(x, dims, hole_gaps=(1, 5, 9)):
    """Respace the active main-loop genes so INACTIVE holes sit inside the loop."""
    types, flips = _active_view(x, dims)
    assert len(types) + len(hole_gaps) <= dims.n_main
    x[:dims.n_main] = INACTIVE
    x[dims.main_flips_start:dims.main_flips_end] = 0
    slot = 0
    for i, (piece, flip) in enumerate(zip(types, flips)):
        if i in hole_gaps:
            slot += 1
        x[slot] = piece
        x[dims.main_flips_start + slot] = flip
        slot += 1


class TestNoCatalogFallback:
    def test_no_catalog_fallback_edit_reports_true(self, dims):
        """Without a catalog the grow degrades to a piece-type mutation — an
        edit — so it must report True, or fall-through callers double-edit."""
        np.random.seed(41)
        x = create_chromosome_from_pieces(dims, [2] * 16)
        assert _compensated_pair_grow(x, dims, None)


class TestGrowHoledGenomes:
    """Descriptor positions are active-order (the decoder skips INACTIVE
    slots), so a holed loop decodes and grows exactly like its compact twin —
    the operator may not decline on holes alone."""

    def test_holed_twin_decodes_identically(self, cfg, cat, dims, inv):
        from src.sampling import _gen_figure_eight_cross
        pieces, flips, _, cjs, _ = _gen_figure_eight_cross(inv, dims)[-1]
        compact = create_chromosome_from_pieces(
            dims, pieces, main_loop_flips=flips, cross_junctions=cjs,
        )
        holed = compact.copy()
        _inject_holes(holed, dims)
        lay_compact = _decode(compact, cfg, cat, dims)
        lay_holed = _decode(holed, cfg, cat, dims)
        assert lay_holed.main_loop_pieces == lay_compact.main_loop_pieces
        assert lay_holed.n_cross_junctions == lay_compact.n_cross_junctions == 1
        assert lay_holed.paths[0].closure_error == pytest.approx(
            lay_compact.paths[0].closure_error)

    def test_holed_racetrack_grows(self, cfg, cat, dims, inv):
        from src.sampling import _gen_racetrack
        np.random.seed(29)
        pieces, flips, *_ = _gen_racetrack(inv, dims)[0]
        x = create_chromosome_from_pieces(dims, pieces, main_loop_flips=flips)
        _inject_holes(x, dims)
        before = _n_active(x, dims)
        assert _compensated_pair_grow(x, dims, cat), "holes alone must not decline the grow"
        assert _n_active(x, dims) == before + 2
        lay = _decode(x, cfg, cat, dims)
        assert lay.paths[0].closure_error < cfg.closure_tolerance

    def test_holed_cross_genome_keeps_commit(self, cfg, cat, dims, inv):
        from src.sampling import _gen_figure_eight_cross
        np.random.seed(31)
        pieces, flips, _, cjs, _ = _gen_figure_eight_cross(inv, dims)[-1]
        x = create_chromosome_from_pieces(
            dims, pieces, main_loop_flips=flips, cross_junctions=cjs,
        )
        _inject_holes(x, dims)
        assert _compensated_pair_grow(x, dims, cat)
        lay = _decode(x, cfg, cat, dims)
        assert lay.n_cross_junctions == 1, "CROSS_90 must stay committed"
        assert lay.paths[0].closure_error < cfg.closure_tolerance


class TestSidingFallback:
    """A siding-bearing genome whose grow declines still receives an edit: the
    mutation falls back to a junction op instead of a silent no-op."""

    def test_declined_grow_falls_back_to_junction_op(self, cat):
        from src.operators import PartitionedMutation

        cfg = OptimizationConfig(
            train_config_path="trains/measured_consist.yaml",
            inventory={"STRAIGHT_16": 9, "R40_CURVE": 20,
                       "R40_SWITCH_LEFT": 1, "R40_SWITCH_RIGHT": 1},
            boundary={"min_x": -250.0, "max_x": 250.0,
                      "min_y": -250.0, "max_y": 250.0},
        )
        dims = compute_dimensions(cfg, cat)
        # Racetrack uses 8 S16; the active junction claims 1 more, so the grow
        # needs 11 > 9 S16 (and there are no S24) — it declines every time.
        x = create_chromosome_from_pieces(
            dims, ([2] * 4 + [0] * 2) * 4, junctions=[(1, 3, 0, 1)],
        )
        junc_block = slice(dims.junc_start, dims.junc_start + 4)
        before = x[junc_block].copy()

        class _P:
            catalog = cat

        np.random.seed(37)
        mut = PartitionedMutation(dims, prob=1.0)
        X = np.array([x.copy() for _ in range(40)])
        mut._do(_P(), X)

        untouched_main = sum(
            1 for row in X if np.array_equal(row[:dims.n_main], x[:dims.n_main]))
        assert untouched_main == 40, "a declined grow must leave the main loop alone"
        edited = sum(1 for row in X if not np.array_equal(row[junc_block], before))
        assert edited > 20, (
            f"only {edited}/40 rows got a junction edit; a declined grow must "
            f"fall back instead of no-opping"
        )
