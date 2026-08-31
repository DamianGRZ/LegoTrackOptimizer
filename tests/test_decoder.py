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


class TestSidingRotationInvariance:
    """Cyclic shifts of the loop rotate the whole layout in world space, so the
    same physical siding must commit at every shift: the OUT walk measures in
    the IN switch's entry frame, not along the world X axis."""

    def test_siding_commits_at_every_rotation(self, catalog, switches_config):
        from src.sampling import _gen_oval_with_siding

        dims = compute_dimensions(switches_config, catalog)
        inv = catalog.inventory_by_index(switches_config.inventory)
        decoder_config = DecoderConfig.from_optimization_config(switches_config)

        variants = _gen_oval_with_siding(inv, dims)
        assert variants, "seeder produced no oval+siding variant for this inventory"
        pieces, flips, junctions, _cross, _dc = variants[0]
        n = len(pieces)
        active, pos, hand, n_str = junctions[0]

        # The walk cannot wrap past the array end, so keep the shifted junction
        # in the front half with room downstream.
        shifts = [k for k in range(0, n, 2) if 2 <= (pos - k) % n <= n // 2]
        assert len(shifts) >= 6, "test needs a spread of rotations"

        dropped = []
        for k in shifts:
            shifted_pieces = [int(p) for p in pieces[k:]] + [int(p) for p in pieces[:k]]
            shifted_flips = [int(f) for f in flips[k:]] + [int(f) for f in flips[:k]]
            junction = (active, (pos - k) % n, hand, n_str)
            x = create_chromosome_from_pieces(
                dims, shifted_pieces, main_loop_flips=shifted_flips, junctions=[junction],
            )
            layout = decode_chromosome(
                x, catalog, switches_config.inventory, dims=dims, config=decoder_config,
            )
            if layout.n_switch_pairs != 1:
                dropped.append((k, layout.drop_log))

        assert not dropped, (
            f"the same siding must commit at every rotation; dropped at {dropped}"
        )


class TestPortGraphCircuits:
    """Circuits beyond the 2^J switch choices: a DOUBLE_CROSSOVER's spare
    routes close sub-loops the choice enumeration cannot express."""

    def test_dc_figure_eight_has_full_loop_plus_both_lobes(self, catalog):
        import numpy as np

        from src.config import OptimizationConfig
        from src.sampling import _gen_figure_eight_dbl_crossover
        from src.templates import DC_R_CROSS_1_TO_2, DC_R_TRACK1_THROUGH

        config = OptimizationConfig.load("configs/all_pieces.yaml")
        dims = compute_dimensions(config, catalog)
        inv = catalog.inventory_by_index(config.inventory)
        patterns = _gen_figure_eight_dbl_crossover(inv, dims)
        assert patterns, "seeder produced no DC figure-8 pattern"
        pieces, flips, _junctions, _cross, dcs = patterns[0]
        x = create_chromosome_from_pieces(
            dims, pieces, main_loop_flips=flips, double_crossovers=dcs,
        )
        layout = decode_chromosome(
            x, catalog, config.inventory, dims=dims,
            config=DecoderConfig.from_optimization_config(config),
        )
        assert layout.n_dbl_crossovers == 1
        assert layout.n_paths == 3

        main, lobe_a, lobe_b = layout.paths
        assert main.path_id == 0 and layout.get_main_path() is main

        crossover = int(PieceIndex.DOUBLE_CROSSOVER)
        assert main.piece_sequence.count(crossover) == 2
        for lobe in (lobe_a, lobe_b):
            assert lobe.is_closed
            assert lobe.piece_sequence.count(crossover) == 1
        assert lobe_a.n_pieces + lobe_b.n_pieces == main.n_pieces

        def length(path):
            return float(catalog.get_route_arc_lengths(
                np.asarray(path.piece_sequence, dtype=np.int32),
                np.asarray(path.route_indices, dtype=np.int32),
            ).sum())

        through = catalog.get_arc_length_route(crossover, DC_R_TRACK1_THROUGH)
        diagonal = catalog.get_arc_length_route(crossover, DC_R_CROSS_1_TO_2)
        expected = length(lobe_a) + length(lobe_b) - 2 * through + 2 * diagonal
        assert length(main) == pytest.approx(expected)

    def test_dc_with_sidings_counts_sum_over_circuits(self, catalog):
        """1 DC + siding on one lobe -> 5 paths; a siding on each lobe -> 8.

        The count is a sum over circuits of 2^(sidings on that circuit), not
        a global 2^J product. Geometry is irrelevant to the count, so the
        layout is assembled record-by-record at the builder seam -- the
        genotype cannot express a siding on a DC figure-8 yet.
        """
        from src.decoder.construction import _build_multi_path_layout
        from src.templates import DC_R_CROSS_1_TO_2, DC_R_CROSS_2_TO_1
        from src.types import DblCrossover, SwitchPair

        straight = int(PieceIndex.STRAIGHT_16)
        sw_left = int(PieceIndex.R40_SWITCH_LEFT)
        sw_right = int(PieceIndex.R40_SWITCH_RIGHT)
        crossover = int(PieceIndex.DOUBLE_CROSSOVER)

        def build(pairs):
            pieces = [crossover] + [straight] * 5 + [crossover] + [straight] * 5
            for pair in pairs:
                pieces[pair.in_position] = sw_left
                pieces[pair.out_position] = sw_right
            record = DblCrossover(
                slot=0, positions=(0, 6),
                routes=(DC_R_CROSS_1_TO_2, DC_R_CROSS_2_TO_1),
                origin=(0.0, 0.0, 0.0),
            )
            return _build_multi_path_layout(
                pieces, [0] * len(pieces), pairs, catalog,
                main_loop_routes={0: DC_R_CROSS_1_TO_2, 6: DC_R_CROSS_2_TO_1},
                dbl_crossovers=[record],
            )

        siding_1 = SwitchPair(pair_id=0, in_position=2, out_position=4,
                              branch_pieces=[straight], branch_flips=[0])
        siding_2 = SwitchPair(pair_id=1, in_position=8, out_position=10,
                              branch_pieces=[straight], branch_flips=[0])

        assert build([siding_1]).n_paths == 5
        assert build([siding_1, siding_2]).n_paths == 8
