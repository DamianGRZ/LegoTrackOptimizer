"""Tests for the CGP-inspired integer decoder."""

import numpy as np
import pytest

from src.catalog import TrackCatalog
from src.config import OptimizationConfig
from src.decoder import DecoderConfig, decode_chromosome
from src.encoding import (
    PartitionedDimensions,
    compute_dimensions,
    create_chromosome_from_pieces,
    create_empty_chromosome,
    PieceIndex,
)
from src.types import MultiPathLayout


# Piece indices
STRAIGHT_16 = PieceIndex.STRAIGHT_16
R40_LEFT = PieceIndex.R40_LEFT
R40_RIGHT = PieceIndex.R40_RIGHT
R40_SWITCH_LEFT = PieceIndex.SWITCH_LEFT
R40_SWITCH_RIGHT = PieceIndex.SWITCH_RIGHT


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
        """16 R40_LEFT curves should form a closed circle."""
        dims = compute_dimensions(default_config, catalog)
        pattern = [R40_LEFT] * 16
        chromosome = create_chromosome_from_pieces(dims, pattern)

        layout = decode_chromosome(chromosome, catalog, default_config.inventory, dims=dims)

        assert layout.n_pieces == 16
        assert layout.closure_error < 1.0
        assert layout.angle_error < 5.0

    def test_main_loop_respects_inventory(self, catalog, default_config):
        """Decoder should not exceed inventory limits."""
        dims = compute_dimensions(default_config, catalog)

        # Request more R40_LEFT than available in default inventory
        pattern = [R40_LEFT] * 50
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
        pattern = [R40_LEFT] * 16
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
        pattern = [R40_LEFT] * 16
        chromosome = create_chromosome_from_pieces(dims, pattern)

        layout = decode_chromosome(chromosome, catalog, default_config.inventory, dims=dims)

        assert layout.max_closure_error < 1.0
        assert layout.max_angle_error < 5.0


class TestBackwardCompatibility:
    """Tests for Layout interface compatibility."""

    def test_multi_path_has_layout_properties(self, catalog, default_config):
        """MultiPathLayout should have Layout-compatible properties."""
        dims = compute_dimensions(default_config, catalog)
        pattern = [R40_LEFT] * 16
        chromosome = create_chromosome_from_pieces(dims, pattern)

        layout = decode_chromosome(chromosome, catalog, default_config.inventory, dims=dims)

        assert hasattr(layout, 'n_pieces')
        assert hasattr(layout, 'indices')
        assert hasattr(layout, 'closure_error')
        assert hasattr(layout, 'angle_error')

        assert layout.n_pieces > 0
        assert len(layout.indices) == layout.n_pieces
