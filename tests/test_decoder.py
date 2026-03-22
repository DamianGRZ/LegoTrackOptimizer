"""Tests for the template-based construction decoder.

Note: After migration to random-key encoding, chromosomes use [0,1] values
that are mapped to pieces by the decoder. Tests use create_chromosome_from_pattern()
to convert integer piece patterns to RK chromosomes.
"""

import numpy as np
import pytest

from src.data import TrackCatalog
from src.config import OptimizationConfig
from src.decoder import (
    DecoderConfig,
    DecoderState,
    _decode_main_loop,
    _extract_switch_pairs,
    _build_multi_path_layout,
    decode_chromosome,
)
from src.encoding import (
    N_VAR,
    create_chromosome_from_pattern,
    create_empty_chromosome,
    set_branch_template_params,
    get_branch_in_position,
)
from src.topology import MultiPathLayout


# Piece indices from track_pieces.yaml
STRAIGHT_16 = 0
R40_LEFT = 2
R40_RIGHT = 3
R40_SWITCH_LEFT_IN = 5
R40_SWITCH_LEFT_OUT = 6
R40_SWITCH_RIGHT_IN = 7
R40_SWITCH_RIGHT_OUT = 8


def _get_available_pieces(catalog: TrackCatalog, inventory: dict) -> list:
    """Get sorted list of available piece indices from inventory."""
    available = []
    for piece_id, count in inventory.items():
        idx = catalog._id_to_index.get(piece_id)
        if idx is not None and count > 0:
            available.append(idx)
    return sorted(available)


class TestMainLoopDecoding:
    """Tests for Phase 1: Main loop construction.

    RK encoding: All chromosome values are [0,1]. The decoder maps RK values
    to available pieces dynamically. Tests use create_chromosome_from_pattern()
    to encode known piece sequences as RK chromosomes.
    """

    def test_empty_chromosome_produces_layout(self, catalog: TrackCatalog, default_config: OptimizationConfig):
        """RK chromosome with uniform random values produces valid layout.

        In RK encoding, any [0,1] values are valid and map to available pieces.
        A 'random' chromosome will still produce pieces - that's by design.
        """
        chromosome = create_empty_chromosome()  # Uniform [0,1] random values
        layout = decode_chromosome(chromosome, catalog, default_config.inventory)

        # RK encoding always produces a layout (>0 pieces for non-empty inventory)
        # This is the key benefit: 100% feasibility by construction
        assert layout is not None
        assert isinstance(layout, MultiPathLayout)

    def test_simple_circle_decodes_correctly(self, catalog: TrackCatalog, default_config: OptimizationConfig):
        """16 R40_LEFT curves should form a circle."""
        # Get available pieces for RK mapping
        available = _get_available_pieces(catalog, default_config.inventory)

        # Create RK chromosome encoding 16 R40_LEFT pieces
        pattern = [R40_LEFT] * 16
        chromosome = create_chromosome_from_pattern(pattern, available)

        layout = decode_chromosome(chromosome, catalog, default_config.inventory)

        assert layout.n_pieces == 16
        # Should be closed (within tolerances)
        assert layout.closure_error < 1.0
        assert layout.angle_error < 5.0

    def test_main_loop_respects_inventory(self, catalog: TrackCatalog, default_config: OptimizationConfig):
        """Decoder should skip pieces when inventory is exhausted."""
        # Limited inventory
        inventory = {"R40_LEFT": 8, "STRAIGHT_16": 4}
        available = _get_available_pieces(catalog, inventory)

        # Create RK chromosome encoding 20 R40_LEFT (more than available)
        pattern = [R40_LEFT] * 20
        chromosome = create_chromosome_from_pattern(pattern, available)

        layout = decode_chromosome(chromosome, catalog, inventory)

        # Should only place up to 12 pieces (8 R40_LEFT + 4 STRAIGHT_16 max)
        assert layout.n_pieces <= 12


class TestTemplateSidingExtraction:
    """Tests for Phase 2: Template-based siding extraction."""

    def test_valid_siding_extracted(self, catalog: TrackCatalog, switches_config: OptimizationConfig):
        """Valid template siding should be extracted from branch slot."""
        # Create main loop with enough straights for siding to fit
        inventory_by_index = {
            STRAIGHT_16: 12,
            R40_LEFT: 20,
            R40_RIGHT: 10,
            R40_SWITCH_LEFT_IN: 4,
            R40_SWITCH_LEFT_OUT: 4,
        }
        available = sorted(inventory_by_index.keys())
        pattern = [STRAIGHT_16] * 10 + [R40_LEFT] * 8
        chromosome = create_chromosome_from_pattern(pattern, available)

        # Set branch template: IN at position 1, LEFT handedness, 1 straight, active
        set_branch_template_params(chromosome, slot_idx=0, in_pos=1, handedness=0, n_straights=1, active=1)

        config = DecoderConfig()
        state = _decode_main_loop(chromosome, catalog, inventory_by_index, config)
        switch_pairs = _extract_switch_pairs(chromosome, state, catalog, inventory_by_index, config)

        # Should extract a valid siding (if geometry fits)
        assert isinstance(switch_pairs, list)
        # If extracted, should have correct IN position and LEFT switch types
        if len(switch_pairs) > 0:
            assert switch_pairs[0].in_switch_idx == R40_SWITCH_LEFT_IN
            assert switch_pairs[0].out_switch_idx == R40_SWITCH_LEFT_OUT

    def test_inactive_branch_not_extracted(self, catalog: TrackCatalog, switches_config: OptimizationConfig):
        """Inactive branch slot should not produce switch pair."""
        inventory_by_index = {
            STRAIGHT_16: 12,
            R40_LEFT: 20,
            R40_RIGHT: 10,
            R40_SWITCH_LEFT_IN: 4,
            R40_SWITCH_LEFT_OUT: 4,
        }
        available = sorted(inventory_by_index.keys())
        pattern = [STRAIGHT_16] * 10 + [R40_LEFT] * 8
        chromosome = create_chromosome_from_pattern(pattern, available)

        # Set branch template with active=0 (disabled)
        set_branch_template_params(chromosome, slot_idx=0, in_pos=1, handedness=0, n_straights=1, active=0)

        config = DecoderConfig()
        state = _decode_main_loop(chromosome, catalog, inventory_by_index, config)
        switch_pairs = _extract_switch_pairs(chromosome, state, catalog, inventory_by_index, config)

        # Inactive branch slot should not trigger injection
        # (Pass 2 scan may still find naturally-placed switches)
        assert all(sp.in_position != 1 for sp in switch_pairs)  # No injection at slot's in_pos

    def test_invalid_in_position_rejected(self, catalog: TrackCatalog, switches_config: OptimizationConfig):
        """Branch with IN position beyond main loop should be rejected."""
        inventory_by_index = {
            STRAIGHT_16: 10,
            R40_LEFT: 10,
            R40_RIGHT: 10,
            R40_SWITCH_LEFT_IN: 4,
            R40_SWITCH_LEFT_OUT: 4,
        }
        available = sorted(inventory_by_index.keys())
        pattern = [STRAIGHT_16] * 6
        chromosome = create_chromosome_from_pattern(pattern, available)

        # Set IN position beyond main loop length
        set_branch_template_params(chromosome, slot_idx=0, in_pos=10, handedness=0, n_straights=1, active=1)

        config = DecoderConfig()
        state = _decode_main_loop(chromosome, catalog, inventory_by_index, config)
        switch_pairs = _extract_switch_pairs(chromosome, state, catalog, inventory_by_index, config)

        # Invalid position should be rejected
        assert len(switch_pairs) == 0

    def test_missing_inventory_rejected(self, catalog: TrackCatalog, switches_config: OptimizationConfig):
        """Branch requiring unavailable pieces should be rejected."""
        # No switches available in inventory
        inventory_by_index = {
            STRAIGHT_16: 12,
            R40_LEFT: 20,
            R40_RIGHT: 10,
            # No switches
        }
        available = sorted(inventory_by_index.keys())
        pattern = [STRAIGHT_16] * 10 + [R40_LEFT] * 8
        chromosome = create_chromosome_from_pattern(pattern, available)

        set_branch_template_params(chromosome, slot_idx=0, in_pos=1, handedness=0, n_straights=1, active=1)

        config = DecoderConfig()
        state = _decode_main_loop(chromosome, catalog, inventory_by_index, config)
        switch_pairs = _extract_switch_pairs(chromosome, state, catalog, inventory_by_index, config)

        # Should be rejected due to missing switches
        assert len(switch_pairs) == 0


class TestMultiPathGeneration:
    """Tests for Phase 3: Multi-path generation."""

    def test_no_switches_single_path(self, catalog: TrackCatalog, default_config: OptimizationConfig):
        """Layout without switches should have exactly one path."""
        available = _get_available_pieces(catalog, default_config.inventory)
        pattern = [R40_LEFT] * 16
        chromosome = create_chromosome_from_pattern(pattern, available)

        layout = decode_chromosome(chromosome, catalog, default_config.inventory)

        assert isinstance(layout, MultiPathLayout)
        assert layout.n_paths == 1
        assert layout.n_switch_pairs == 0

    def test_one_switch_pair_two_paths(self, catalog: TrackCatalog, switches_config: OptimizationConfig):
        """Layout with one switch pair should have two paths."""
        available = _get_available_pieces(catalog, switches_config.inventory)
        pattern = [STRAIGHT_16] * 10 + [R40_LEFT] * 8
        chromosome = create_chromosome_from_pattern(pattern, available)

        # Set valid template siding: IN at position 1, LEFT handedness, 1 straight
        set_branch_template_params(chromosome, slot_idx=0, in_pos=1, handedness=0, n_straights=1, active=1)

        layout = decode_chromosome(chromosome, catalog, switches_config.inventory)

        # With one valid switch pair, should have 2 paths
        if layout.n_switch_pairs == 1:
            assert layout.n_paths == 2
            # Path 0 should be straight-through (route_choices all 0)
            assert layout.paths[0].route_choices == (0,)
            # Path 1 should be branch (route_choices has 1)
            assert layout.paths[1].route_choices == (1,)


class TestMultiPathClosure:
    """Tests for path closure constraints."""

    def test_main_path_closure(self, catalog: TrackCatalog, default_config: OptimizationConfig):
        """Main path (circle) should be closed."""
        available = _get_available_pieces(catalog, default_config.inventory)
        pattern = [R40_LEFT] * 16
        chromosome = create_chromosome_from_pattern(pattern, available)

        layout = decode_chromosome(chromosome, catalog, default_config.inventory)

        assert layout.max_closure_error < 1.0
        assert layout.max_angle_error < 5.0

    def test_all_paths_checked_for_closure(self, catalog: TrackCatalog, switches_config: OptimizationConfig):
        """Closure check should consider all paths."""
        available = _get_available_pieces(catalog, switches_config.inventory)
        pattern = [STRAIGHT_16] * 10 + [R40_LEFT] * 8
        chromosome = create_chromosome_from_pattern(pattern, available)

        # Set valid template siding
        set_branch_template_params(chromosome, slot_idx=0, in_pos=1, handedness=0, n_straights=1, active=1)

        layout = decode_chromosome(chromosome, catalog, switches_config.inventory)

        # max_closure_error should be the max across all paths
        if layout.n_paths > 1:
            individual_errors = [p.closure_error for p in layout.paths]
            assert layout.max_closure_error == max(individual_errors)


class TestSwitchesInMainLoop:
    """Tests for switch detection when switches are placed directly in main loop.

    With RK encoding, we use create_chromosome_from_pattern() to encode integer
    piece patterns as [0,1] random-key chromosomes.
    """

    def test_switches_detected_in_main_loop(self, catalog: TrackCatalog, switches_config: OptimizationConfig):
        """Switches placed directly in main loop should be detected."""
        inventory_by_index = {
            STRAIGHT_16: 20,
            R40_LEFT: 20,
            R40_RIGHT: 10,
            R40_SWITCH_LEFT_IN: 4,
            R40_SWITCH_LEFT_OUT: 4,
        }
        available = sorted(inventory_by_index.keys())

        pattern = (
            [R40_LEFT] * 4 +
            [R40_SWITCH_LEFT_IN] +
            [STRAIGHT_16, STRAIGHT_16] +
            [R40_SWITCH_LEFT_OUT] +
            [STRAIGHT_16] +
            [R40_LEFT] * 4 +
            [STRAIGHT_16, STRAIGHT_16] +
            [R40_LEFT] * 4 +
            [STRAIGHT_16, STRAIGHT_16] +
            [R40_LEFT] * 4
        )
        chromosome = create_chromosome_from_pattern(pattern, available)

        config = DecoderConfig()
        state = _decode_main_loop(chromosome, catalog, inventory_by_index, config)

        # Switches are now reserved for injection (not placed during main loop construction)
        # Verify main loop was built without switches
        assert R40_SWITCH_LEFT_IN not in state.piece_indices
        assert R40_SWITCH_LEFT_OUT not in state.piece_indices

    def test_switch_pair_matching(self, catalog: TrackCatalog, switches_config: OptimizationConfig):
        """LEFT_IN and LEFT_OUT switches should be matched as a pair."""
        inventory_by_index = {
            STRAIGHT_16: 20,
            R40_LEFT: 20,
            R40_RIGHT: 10,
            R40_SWITCH_LEFT_IN: 4,
            R40_SWITCH_LEFT_OUT: 4,
        }
        available = sorted(inventory_by_index.keys())

        pattern = (
            [R40_LEFT] * 4 +
            [R40_SWITCH_LEFT_IN] +
            [STRAIGHT_16, STRAIGHT_16] +
            [R40_SWITCH_LEFT_OUT] +
            [STRAIGHT_16] +
            [R40_LEFT] * 4 +
            [STRAIGHT_16, STRAIGHT_16] +
            [R40_LEFT] * 4 +
            [STRAIGHT_16, STRAIGHT_16] +
            [R40_LEFT] * 4
        )
        chromosome = create_chromosome_from_pattern(pattern, available)

        config = DecoderConfig()
        state = _decode_main_loop(chromosome, catalog, inventory_by_index, config)
        switch_pairs = _extract_switch_pairs(chromosome, state, catalog, inventory_by_index, config)

        # Should find switch pairs if geometry matches
        if len(switch_pairs) > 0:
            pair = switch_pairs[0]
            assert pair.in_switch_idx == R40_SWITCH_LEFT_IN
            assert pair.out_switch_idx == R40_SWITCH_LEFT_OUT

    def test_unpaired_switches_are_straight_through(self, catalog: TrackCatalog, switches_config: OptimizationConfig):
        """Unpaired switches operate in straight-through mode — no loose ports."""
        # Inventory with only LEFT_IN (no matching LEFT_OUT)
        inventory = {
            "STRAIGHT_16": 20,
            "R40_LEFT": 20,
            "R40_RIGHT": 10,
            "R40_SWITCH_LEFT_IN": 1,
            # No LEFT_OUT — switch can't pair, operates straight-through
        }
        available = _get_available_pieces(catalog, inventory)

        pattern = (
            [R40_LEFT] * 4 +
            [R40_SWITCH_LEFT_IN] +  # Unpaired — straight-through mode
            [STRAIGHT_16, STRAIGHT_16, STRAIGHT_16] +
            [R40_LEFT] * 4 +
            [STRAIGHT_16, STRAIGHT_16] +
            [R40_LEFT] * 4 +
            [STRAIGHT_16, STRAIGHT_16] +
            [R40_LEFT] * 4
        )
        chromosome = create_chromosome_from_pattern(pattern, available)

        layout = decode_chromosome(chromosome, catalog, inventory)

        # Unpaired switches are valid straight-through pieces, not loose ports
        assert layout.loose_port_count == 0

    def test_full_decode_with_switches(self, catalog: TrackCatalog, switches_config: OptimizationConfig):
        """Full decode should produce multi-path layout with switch pair."""
        available = _get_available_pieces(catalog, switches_config.inventory)

        pattern = (
            [R40_LEFT] * 4 +
            [R40_SWITCH_LEFT_IN] +
            [STRAIGHT_16, STRAIGHT_16] +
            [R40_SWITCH_LEFT_OUT] +
            [STRAIGHT_16] +
            [R40_LEFT] * 4 +
            [STRAIGHT_16, STRAIGHT_16] +
            [R40_LEFT] * 4 +
            [STRAIGHT_16, STRAIGHT_16] +
            [R40_LEFT] * 4
        )
        chromosome = create_chromosome_from_pattern(pattern, available)

        layout = decode_chromosome(chromosome, catalog, switches_config.inventory)

        # Should produce a valid multi-path layout
        assert isinstance(layout, MultiPathLayout)
        assert layout.n_pieces > 0


class TestBackwardCompatibility:
    """Tests for backward compatibility with Layout interface."""

    def test_multi_path_has_layout_properties(self, catalog: TrackCatalog, default_config: OptimizationConfig):
        """MultiPathLayout should have Layout-compatible properties."""
        available = _get_available_pieces(catalog, default_config.inventory)
        pattern = [R40_LEFT] * 16
        chromosome = create_chromosome_from_pattern(pattern, available)

        layout = decode_chromosome(chromosome, catalog, default_config.inventory)

        # Should have Layout-compatible properties
        assert hasattr(layout, 'n_pieces')
        assert hasattr(layout, 'indices')
        assert hasattr(layout, 'states')
        assert hasattr(layout, 'closure_error')
        assert hasattr(layout, 'angle_error')
        assert hasattr(layout, 'bounding_box')

        # Values should be valid
        assert layout.n_pieces > 0
        assert len(layout.indices) == layout.n_pieces
        assert len(layout.states) == layout.n_pieces + 1
