"""Tests for the CGP-inspired integer decoder."""

import pytest

from src.catalog import TrackCatalog
from src.decoder import DecoderConfig, decode_chromosome
from src.encoding import (
    compute_dimensions,
    create_chromosome_from_pieces,
    create_empty_chromosome,
    PieceIndex,
)
from src.types import MultiPathLayout


R40_CURVE = PieceIndex.R40_CURVE


class TestMainLoopDecoding:
    """Tests for main loop construction from CGP integer chromosome."""

    def test_empty_chromosome_produces_layout(self, catalog, default_config):
        """All-inactive chromosome produces empty layout."""
        dims = compute_dimensions(default_config, catalog)
        chromosome = create_empty_chromosome(dims)
        layout = decode_chromosome(chromosome, catalog, default_config.inventory, dims=dims)

        assert layout is not None
        assert isinstance(layout, MultiPathLayout)

    def test_simple_circle_decodes_correctly(self, catalog, default_config):
        """16 R40_CURVE curves should form a closed circle."""
        dims = compute_dimensions(default_config, catalog)
        pattern = [R40_CURVE] * 16
        chromosome = create_chromosome_from_pieces(dims, pattern)

        layout = decode_chromosome(chromosome, catalog, default_config.inventory, dims=dims)

        assert layout.n_pieces == 16
        assert layout.closure_error < 1.0
        assert layout.angle_error < 5.0

    def test_main_loop_respects_inventory(self, catalog, default_config):
        """Decoder should not exceed inventory limits."""
        dims = compute_dimensions(default_config, catalog)

        # Request more R40_CURVE than available in default inventory
        pattern = [R40_CURVE] * 50
        chromosome = create_chromosome_from_pieces(dims, pattern)

        layout = decode_chromosome(
            chromosome, catalog, default_config.inventory, dims=dims,
        )

        assert layout.n_pieces <= default_config.total_inventory


class TestMultiPathGeneration:
    """Tests for multi-path generation."""

    def test_no_switches_single_path(self, catalog, default_config):
        """Layout without switches should have exactly one path."""
        dims = compute_dimensions(default_config, catalog)
        pattern = [R40_CURVE] * 16
        chromosome = create_chromosome_from_pieces(dims, pattern)

        layout = decode_chromosome(chromosome, catalog, default_config.inventory, dims=dims)

        assert isinstance(layout, MultiPathLayout)
        assert layout.n_paths == 1
        assert layout.n_switch_pairs == 0


class TestMultiPathClosure:
    """Tests for path closure constraints."""

    def test_main_path_closure(self, catalog, default_config):
        """Main path (circle) should be closed."""
        dims = compute_dimensions(default_config, catalog)
        pattern = [R40_CURVE] * 16
        chromosome = create_chromosome_from_pieces(dims, pattern)

        layout = decode_chromosome(chromosome, catalog, default_config.inventory, dims=dims)

        assert layout.max_closure_error < 1.0
        assert layout.max_angle_error < 5.0


class TestPieceUids:
    """piece_uids: physical-piece identity parallel to piece_sequence.

    Per-piece aggregation across the 2^J routes (the whole-graph time
    objective) relies on paths naming shared physical pieces identically.
    """

    def test_switchless_uids_are_main_slots(self, catalog, default_config):
        """A plain loop's single path is uid-labeled by main-loop slot."""
        dims = compute_dimensions(default_config, catalog)
        chromosome = create_chromosome_from_pieces(dims, [R40_CURVE] * 16)

        layout = decode_chromosome(chromosome, catalog, default_config.inventory, dims=dims)

        path = layout.paths[0]
        assert path.piece_uids == [("main", i, 0) for i in range(16)]

    def test_switched_paths_align_and_share_main_uids(self, catalog, switches_config):
        """Through and diverge routes agree on shared uids, differ on exclusive ones."""
        from src.sampling import _gen_oval_with_siding

        dims = compute_dimensions(switches_config, catalog)
        inv = catalog.inventory_by_index(switches_config.inventory)
        variants = _gen_oval_with_siding(inv, dims)
        assert variants, "seeder produced no oval+siding variant for this inventory"
        pieces, flips, junctions, *_ = variants[0]
        chromosome = create_chromosome_from_pieces(
            dims, pieces, main_loop_flips=flips, junctions=junctions,
        )

        layout = decode_chromosome(
            chromosome, catalog, switches_config.inventory, dims=dims,
            config=DecoderConfig.from_optimization_config(switches_config),
        )
        assert layout.n_switch_pairs == 1 and layout.n_paths == 2

        for path in layout.paths:
            assert len(path.piece_uids) == len(path.piece_sequence)

        through = layout.get_path_by_choices((0,))
        diverge = layout.get_path_by_choices((1,))
        pair = layout.switch_pairs[0]

        branch_uids = [u for u in diverge.piece_uids if u[0] == "branch"]
        assert branch_uids == [("branch", 0, k) for k in range(len(pair.branch_pieces))]
        assert all(uid[0] == "main" for uid in through.piece_uids)

        # Both routes name the physical switches by their main-loop slots.
        switch_uids = {("main", pair.in_position, 0), ("main", pair.out_position, 0)}
        assert switch_uids <= set(through.piece_uids)
        assert switch_uids <= set(diverge.piece_uids)

        # Pieces bypassed by the siding exist only on the through route.
        bypassed = {("main", p, 0) for p in range(pair.in_position + 1, pair.out_position)}
        assert bypassed <= set(through.piece_uids)
        assert not bypassed & set(diverge.piece_uids)

        # divergent_ranges endpoints land exactly on the IN/OUT switch slots.
        start, end = diverge.divergent_ranges[0]
        assert diverge.piece_uids[start] == ("main", pair.in_position, 0)
        assert diverge.piece_uids[end] == ("main", pair.out_position, 0)


class TestBackwardCompatibility:
    """Tests for Layout interface compatibility."""

    def test_multi_path_has_layout_properties(self, catalog, default_config):
        """MultiPathLayout should have Layout-compatible properties."""
        dims = compute_dimensions(default_config, catalog)
        pattern = [R40_CURVE] * 16
        chromosome = create_chromosome_from_pieces(dims, pattern)

        layout = decode_chromosome(chromosome, catalog, default_config.inventory, dims=dims)

        assert hasattr(layout, 'n_pieces')
        assert hasattr(layout, 'indices')
        assert hasattr(layout, 'closure_error')
        assert hasattr(layout, 'angle_error')

        assert layout.n_pieces > 0
        assert len(layout.indices) == layout.n_pieces
