"""Tests for the run_info physical-piece census."""

from types import SimpleNamespace

from src.intersection import CROSS_90_INDEX, DOUBLE_CROSSOVER_INDEX
from src.run_info import _format_individual, count_pieces
from src.types import CrossJunction, DblCrossover, MultiPathLayout, SwitchPair

S16 = 0
R40 = 2


class TestCountPieces:
    """count_pieces reports PHYSICAL pieces, matching the evaluation census."""

    def test_plain_main_loop_counts_per_slot(self):
        layout = MultiPathLayout(main_loop_pieces=[S16, R40, R40, S16, -1])
        assert count_pieces(layout) == {S16: 2, R40: 2}

    def test_branch_pieces_are_added(self):
        layout = MultiPathLayout(
            main_loop_pieces=[S16, S16],
            switch_pairs=[SwitchPair(pair_id=0, in_position=0, out_position=1,
                                     branch_pieces=[R40, S16, R40])],
        )
        assert count_pieces(layout) == {S16: 3, R40: 2}

    def test_descriptor_double_crossover_counts_once(self):
        """One physical DC spans two traversal slots — must not be doubled."""
        layout = MultiPathLayout(
            main_loop_pieces=[S16, DOUBLE_CROSSOVER_INDEX, S16, DOUBLE_CROSSOVER_INDEX],
            dbl_crossovers=[DblCrossover(slot=0, positions=(1, 3), routes=(2, 3),
                                         origin=(0.0, 0.0, 0.0))],
        )
        assert count_pieces(layout)[DOUBLE_CROSSOVER_INDEX] == 1

    def test_descriptor_cross_counts_once(self):
        layout = MultiPathLayout(
            main_loop_pieces=[S16, CROSS_90_INDEX, S16, CROSS_90_INDEX],
            cross_junctions=[CrossJunction(slot=0, positions=(1, 3),
                                           origin=(0.0, 0.0, 0.0))],
        )
        assert count_pieces(layout)[CROSS_90_INDEX] == 1

    def test_emergent_cross_single_slot_counts_once(self):
        """An emergent CROSS_90 occupies ONE slot and carries no record."""
        layout = MultiPathLayout(main_loop_pieces=[S16, CROSS_90_INDEX, S16, S16])
        assert count_pieces(layout) == {S16: 3, CROSS_90_INDEX: 1}

    def test_legacy_layout_counts_indices_per_slot(self):
        legacy = SimpleNamespace(indices=[S16, R40, R40, -1])
        assert count_pieces(legacy) == {S16: 1, R40: 2}


class TestFormatIndividual:
    """The summary line must agree with the category report, not slot counts."""

    def test_headline_count_dedupes_descriptor_dc(self):
        layout = MultiPathLayout(
            main_loop_pieces=[S16, DOUBLE_CROSSOVER_INDEX, S16, DOUBLE_CROSSOVER_INDEX],
            dbl_crossovers=[DblCrossover(slot=0, positions=(1, 3), routes=(2, 3),
                                         origin=(0.0, 0.0, 0.0))],
        )
        assert layout.n_pieces == 4 and layout.n_physical_pieces == 3

        line, = _format_individual("Best feasible", layout, 0.5, 1.25, cv=None,
                                   total_inventory=10)

        assert "pieces=3/10" in line
        assert "pieces=4" not in line
        # Utilization is that physical count over the kit, never the F[0] score.
        assert "utilization=30.0%" in line
