"""Objective + constraint changes for the CROSS_90 / topology work.

- special_piece_weight (config) makes special pieces count for more toward the
  utilization objective, so they are never pure overhead.
- The per-type inventory census counts ONE physical CROSS_90/DC, not the two
  traversal slots it occupies (fixes a latent DC double-count too).
"""

import numpy as np
import pytest

from src.catalog import TrackCatalog
from src.config import OptimizationConfig
from src.decoder import decode_chromosome
from src.encoding import CROSS_90, STRAIGHT_16, create_chromosome_from_pieces
from src.problem import SPEED_SAFETY_MARGIN, TrackOptimizationProblem
from src.sampling import _gen_figure_eight_cross
from src.train import compute_speed_profile
from src.types import CrossJunction, MultiPathLayout


@pytest.fixture
def cat() -> TrackCatalog:
    return TrackCatalog.load("data/track_pieces_v2.yaml")


@pytest.fixture
def prob(cat) -> TrackOptimizationProblem:
    cfg = OptimizationConfig.load("configs/all_pieces.yaml")
    return TrackOptimizationProblem(catalog=cat, config=cfg)


class TestSpecialPieceWeightConfig:
    def test_default_is_three(self) -> None:
        cfg = OptimizationConfig.load("configs/all_pieces.yaml")
        assert cfg.special_piece_weight == 3.0


class TestPhysicalCensus:
    def test_crossing_counts_as_one_physical_piece(self, prob) -> None:
        """A CROSS_90 occupies two slots but is ONE physical piece."""
        mp = [int(STRAIGHT_16)] * 8 + [int(CROSS_90), int(CROSS_90)]
        layout = MultiPathLayout(
            main_loop_pieces=mp,
            cross_junctions=[CrossJunction(slot=0, positions=(8, 9), origin=(0.0, 0.0, 0.0))],
        )
        # Allow exactly ONE physical CROSS_90. If the census double-counted the two
        # traversal slots, excess would be (2-1)/1 = 1.0 (violation).
        prob.inventory_by_index[int(CROSS_90)] = 1
        g = prob._compute_per_type_inventory_violation(layout)
        assert g[int(CROSS_90)] == 0.0


class TestWeightedPieceScore:
    def test_crossing_layout_scores_higher_than_plain(self, prob) -> None:
        plain = MultiPathLayout(main_loop_pieces=[int(STRAIGHT_16)] * 10)
        crossed = MultiPathLayout(
            main_loop_pieces=[int(STRAIGHT_16)] * 8 + [int(CROSS_90), int(CROSS_90)],
            cross_junctions=[CrossJunction(slot=0, positions=(8, 9), origin=(0.0, 0.0, 0.0))],
        )
        # plain: physical=10, n_special=0
        # crossed: physical=9, n_special=1 -> (9 + (3-1)*1) = 11 effective > 10
        assert prob._weighted_piece_score(crossed) > prob._weighted_piece_score(plain)


class TestF1CrossingChargedOnce:
    """F[1] charges a descriptor CROSS_90 once: the crossing's two slots share
    one physical piece via the junction-record alias, so a closed figure-8's
    expected time lands strictly below its lap time (two passages averaged,
    not summed)."""

    def test_figure_eight_f1_below_lap_time(self, catalog, crossing_config) -> None:
        problem = TrackOptimizationProblem(catalog=catalog, config=crossing_config)
        inv = catalog.inventory_by_index(crossing_config.inventory)
        variants = _gen_figure_eight_cross(inv, problem.dims)
        assert variants, "seeder produced no figure-8-cross for this inventory"
        pieces, flips, _junctions, cross_descriptors, _dc = variants[-1]
        x = create_chromosome_from_pieces(
            problem.dims, pieces, main_loop_flips=flips, cross_junctions=cross_descriptors,
        )
        out = {}
        problem._evaluate(x, out)

        layout = decode_chromosome(
            x, catalog, crossing_config.inventory,
            dims=problem.dims, config=problem.decoder_config,
        )
        assert len(layout.cross_junctions) == 1, "descriptor crossing must commit"
        assert np.all(np.asarray(out["G"][:3]) <= 0), "figure-8 must close"

        profile = compute_speed_profile(
            layout, catalog, train_config=problem._train_config,
            safety_margin=SPEED_SAFETY_MARGIN,
        )
        assert out["F"][1] < profile.lap_time - 1e-3, (
            f"F[1]={out['F'][1]} must land below lap_time={profile.lap_time}: "
            f"the crossing is one physical piece passed twice, charged once"
        )


class TestEmergentDescriptorParity:
    """The same physical figure-8 must score identically whether its crossing
    was named by a descriptor or discovered by the emergent repair."""

    def test_emergent_crossing_scores_like_descriptor(self, catalog, crossing_config) -> None:
        problem = TrackOptimizationProblem(catalog=catalog, config=crossing_config)
        inv = catalog.inventory_by_index(crossing_config.inventory)
        variants = _gen_figure_eight_cross(inv, problem.dims)
        assert variants, "seeder produced no figure-8-cross for this inventory"
        pieces, flips, _junctions, cross_descriptors, _dc = variants[-1]

        x_descriptor = create_chromosome_from_pieces(
            problem.dims, pieces, main_loop_flips=flips, cross_junctions=cross_descriptors,
        )
        x_emergent = create_chromosome_from_pieces(problem.dims, pieces, main_loop_flips=flips)

        lay_descriptor = decode_chromosome(
            x_descriptor, catalog, crossing_config.inventory,
            dims=problem.dims, config=problem.decoder_config,
        )
        lay_emergent = decode_chromosome(
            x_emergent, catalog, crossing_config.inventory,
            dims=problem.dims, config=problem.decoder_config,
        )
        assert lay_emergent.n_cross_pieces == lay_descriptor.n_cross_pieces == 1
        assert lay_emergent.n_physical_pieces == lay_descriptor.n_physical_pieces

        out_descriptor, out_emergent = {}, {}
        problem._evaluate(x_descriptor, out_descriptor)
        problem._evaluate(x_emergent, out_emergent)
        assert out_emergent["F"][0] == pytest.approx(out_descriptor["F"][0]), (
            "emergent crossing must earn the same utilization credit"
        )
        assert out_emergent["F"][1] == pytest.approx(out_descriptor["F"][1]), (
            "emergent crossing must be charged once in F[1], like the descriptor"
        )


class TestF0Variant:
    """f0_objective selects what F[0] measures; the default stays piece_score."""

    def test_config_defaults(self) -> None:
        assert OptimizationConfig.load("configs/all_pieces.yaml").f0_objective == "piece_score"
        cfg = OptimizationConfig.load("configs/all_pieces_route_length.yaml")
        assert cfg.f0_objective == "route_length"

    def test_configured_variant_is_in_use(self, cat) -> None:
        """Same DC figure-8 genome, both variants: F[0] differs, F[1] does not,
        and the route_length value equals an independent sum over circuits."""
        from src.sampling import _gen_figure_eight_dbl_crossover

        by_variant = {}
        for path in ("configs/all_pieces.yaml", "configs/all_pieces_route_length.yaml"):
            cfg = OptimizationConfig.load(path)
            problem = TrackOptimizationProblem(catalog=cat, config=cfg)
            inv = cat.inventory_by_index(cfg.inventory)
            pieces, flips, _j, _c, dcs = _gen_figure_eight_dbl_crossover(inv, problem.dims)[0]
            x = create_chromosome_from_pieces(
                problem.dims, pieces, main_loop_flips=flips, double_crossovers=dcs,
            )
            out = {}
            problem._evaluate(x, out)
            by_variant[cfg.f0_objective] = (problem, x, out)

        _, _, out_score = by_variant["piece_score"]
        problem_len, x_len, out_len = by_variant["route_length"]
        assert out_len["F"][0] != pytest.approx(out_score["F"][0])
        assert out_len["F"][1] == pytest.approx(out_score["F"][1])
        assert problem_len.f0_csv_name == "neg_route_length_studs"

        layout = decode_chromosome(
            x_len, cat, problem_len.config.inventory,
            dims=problem_len.dims, config=problem_len.decoder_config,
        )
        assert layout.n_paths == 3
        expected = sum(
            float(cat.get_route_arc_lengths(
                np.asarray(p.piece_sequence, dtype=np.int32),
                np.asarray(p.route_indices, dtype=np.int32),
            ).sum())
            for p in layout.paths
        )
        assert -out_len["F"][0] == pytest.approx(expected)

    def test_route_length_premiums_the_crossover(self, cat) -> None:
        """A DC figure-8's summed circuits reach nearly twice its full lap."""
        from src.sampling import _gen_figure_eight_dbl_crossover

        cfg = OptimizationConfig.load("configs/all_pieces_route_length.yaml")
        problem = TrackOptimizationProblem(catalog=cat, config=cfg)
        inv = cat.inventory_by_index(cfg.inventory)
        pieces, flips, _j, _c, dcs = _gen_figure_eight_dbl_crossover(inv, problem.dims)[0]
        x = create_chromosome_from_pieces(
            problem.dims, pieces, main_loop_flips=flips, double_crossovers=dcs,
        )
        layout = decode_chromosome(
            x, cat, cfg.inventory, dims=problem.dims, config=problem.decoder_config,
        )
        full = layout.get_main_path()
        full_length = float(cat.get_route_arc_lengths(
            np.asarray(full.piece_sequence, dtype=np.int32),
            np.asarray(full.route_indices, dtype=np.int32),
        ).sum())
        assert problem._route_length_studs(layout) > 1.9 * full_length
