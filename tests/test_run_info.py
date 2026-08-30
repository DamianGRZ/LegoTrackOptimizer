"""Tests for the run_info physical-piece census and provenance sections."""

from types import SimpleNamespace

from src.config import OptimizationConfig
from src.intersection import CROSS_90_INDEX, DOUBLE_CROSSOVER_INDEX
from src.run_info import _format_individual, count_pieces, write_run_info_header
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


class TestTrainPhysicsSection:
    """A finished run must record which locomotive produced its numbers."""

    def _header(self, tmp_path, config):
        write_run_info_header(tmp_path, "configs/default.yaml", config)
        return (tmp_path / "run_info.md").read_text(encoding="utf-8")

    def test_names_and_embeds_the_train_file(self, tmp_path, default_config):
        text = self._header(tmp_path, default_config)
        assert "## Train Physics" in text
        assert "measured_consist.yaml" in text
        # A value the file states, so the verbatim copy really is the file.
        assert "v_motor_max: 1.26" in text

    def test_reports_fields_the_file_does_not_state(self, tmp_path, default_config):
        """The train YAML is partial, so its text alone does not say what ran.
        Every field must appear with its effective value and its provenance."""
        text = self._header(tmp_path, default_config)
        effective = text[text.index("**Effective physics**"):]
        assert "`mu_roll`: 0.05" in effective
        assert "`mu_design`: 0.25" in effective
        assert effective.count("the file does not state it") == 6

    def test_unreadable_train_file_still_writes_the_run_info(self, tmp_path):
        """Provenance must never abort a run."""
        config = OptimizationConfig(train_config_path="trains/no_such_train.yaml",
                                    inventory={"STRAIGHT_16": 8})
        text = self._header(tmp_path, config)
        assert "## Code State" in text
        assert "Could not read" in text
        assert "could not be loaded" in text
