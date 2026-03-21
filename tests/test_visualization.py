"""Tests for visualization functions."""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for tests

import numpy as np
import pytest
from matplotlib.figure import Figure

from src.geometry import build_layout
from src.visualization import plot_layout, plot_pareto_front


class TestPlotLayout:
    """Tests for plot_layout function."""

    def test_plot_layout_creates_figure(self, catalog, default_config):
        """plot_layout returns figure for R40 circle."""
        # Create R40 circle layout
        chromosome = np.full(16, 2, dtype=np.int32)  # 16x R40_LEFT
        layout = build_layout(chromosome, catalog)

        fig = plot_layout(layout, catalog, default_config.boundary, title="Test Circle")

        assert isinstance(fig, Figure)
        assert len(fig.axes) == 1

    def test_plot_layout_empty(self, catalog, default_config):
        """plot_layout handles empty layout."""
        # Create empty layout
        chromosome = np.array([-1, -1, -1], dtype=np.int32)
        layout = build_layout(chromosome, catalog)

        fig = plot_layout(layout, catalog, title="Empty Layout")

        assert isinstance(fig, Figure)

    def test_plot_layout_with_boundary(self, catalog, default_config):
        """plot_layout draws boundary when provided."""
        chromosome = np.array([0, 0, 0, 0], dtype=np.int32)  # 4 straights
        layout = build_layout(chromosome, catalog)

        fig = plot_layout(layout, catalog, boundary=default_config.boundary)

        assert isinstance(fig, Figure)

    def test_plot_layout_mixed_pieces(self, catalog):
        """plot_layout handles mixed piece types with colors."""
        # Mixed layout with different piece types
        chromosome = np.array([0, 2, 1, 3, 0], dtype=np.int32)
        layout = build_layout(chromosome, catalog)

        fig = plot_layout(layout, catalog, title="Mixed Pieces")

        assert isinstance(fig, Figure)
        # Should have different colored segments
        assert len(fig.axes) == 1


class TestPlotParetoFront:
    """Tests for plot_pareto_front function."""

    def test_plot_pareto_front_creates_figure(self):
        """plot_pareto_front creates 3D scatter."""
        # Create random Pareto front data
        n = 50
        F = np.random.randn(n, 3)
        F[:, 0] = -np.random.rand(n)  # Negative utilization
        F[:, 1] = np.random.rand(n) * 10000  # Positive area
        F[:, 2] = -np.random.rand(n)  # Negative speed

        fig = plot_pareto_front(F, title="Test Pareto Front")

        assert isinstance(fig, Figure)
        assert len(fig.axes) == 1

    def test_plot_pareto_front_with_constraints(self):
        """plot_pareto_front colors by feasibility."""
        n = 50
        F = np.random.randn(n, 3)

        # Create constraint array with some feasible, some infeasible
        G = np.random.randn(n, 5)
        G[:25] = -0.1  # First half feasible (all constraints <= 0)
        G[25:] = 0.1  # Second half infeasible

        fig = plot_pareto_front(F, G=G, title="Feasibility Coloring")

        assert isinstance(fig, Figure)

    def test_plot_pareto_front_without_constraints(self):
        """plot_pareto_front works without constraint array."""
        n = 30
        F = np.random.randn(n, 3)

        fig = plot_pareto_front(F)

        assert isinstance(fig, Figure)
