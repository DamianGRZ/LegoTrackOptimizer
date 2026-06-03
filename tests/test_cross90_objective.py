"""Objective + constraint changes for the CROSS_90 / topology work.

- special_piece_weight (config) makes special pieces count for more toward the
  utilization objective, so they are never pure overhead.
- The per-type inventory census counts ONE physical CROSS_90/DC, not the two
  traversal slots it occupies (fixes a latent DC double-count too).
"""

import pytest

from src.catalog import TrackCatalog
from src.config import OptimizationConfig
from src.encoding import CROSS_90, STRAIGHT_16
from src.problem import TrackOptimizationProblem
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


class TestWeightedUtilization:
    def test_crossing_layout_scores_higher_than_plain(self, prob) -> None:
        plain = MultiPathLayout(main_loop_pieces=[int(STRAIGHT_16)] * 10)
        crossed = MultiPathLayout(
            main_loop_pieces=[int(STRAIGHT_16)] * 8 + [int(CROSS_90), int(CROSS_90)],
            cross_junctions=[CrossJunction(slot=0, positions=(8, 9), origin=(0.0, 0.0, 0.0))],
        )
        # plain: physical=10, n_special=0
        # crossed: physical=9, n_special=1 -> (9 + (3-1)*1) = 11 effective > 10
        assert prob._weighted_utilization(crossed) > prob._weighted_utilization(plain)
