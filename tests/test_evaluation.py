"""Tests for track layout evaluation functions."""

import numpy as np
import pytest

from src.evaluation import compute_constraints, compute_objectives, compute_speed_profile
from src.geometry import build_layout


class TestSpeedProfile:
    """Tests for speed profile computation."""

    def test_straight_track_max_speed(self, catalog, physics):
        """4 straights should reach motor top speed (1.57 m/s)."""
        chromosome = np.array([0, 0, 0, 0], dtype=np.int32)  # 4x STRAIGHT_16
        layout = build_layout(chromosome, catalog)

        profile = compute_speed_profile(layout, catalog, physics)

        assert profile.max_speed == pytest.approx(1.57, abs=0.01)
        assert np.all(profile.speeds <= physics.motor_top_speed)

    def test_r40_circle_speed_limit(self, catalog, physics):
        """16 R40 curves limited to ~0.97 m/s (catalog limit)."""
        chromosome = np.full(16, 2, dtype=np.int32)  # 16x R40_LEFT
        layout = build_layout(chromosome, catalog)

        profile = compute_speed_profile(layout, catalog, physics)

        # R40 curves have catalog speed limit of 0.97 m/s
        assert profile.avg_speed < 1.0
        assert profile.max_speed <= 0.97 + 0.01  # Small tolerance

    def test_double_unroll_closure(self, catalog, physics):
        """Closed loop speeds consistent at wrap point."""
        chromosome = np.full(16, 2, dtype=np.int32)  # 16x R40_LEFT
        layout = build_layout(chromosome, catalog)

        profile = compute_speed_profile(layout, catalog, physics)

        # For closed loop, first and last speeds should be similar
        assert abs(profile.speeds[0] - profile.speeds[-1]) < 0.1

    def test_empty_layout(self, catalog, physics):
        """Empty layout returns valid SpeedProfile with zeros."""
        chromosome = np.array([-1, -1, -1], dtype=np.int32)
        layout = build_layout(chromosome, catalog)

        profile = compute_speed_profile(layout, catalog, physics)

        assert len(profile.speeds) == 0
        assert profile.avg_speed == 0.0
        assert profile.lap_time == 0.0
        assert profile.total_distance == 0.0


class TestObjectives:
    """Tests for objective function computation."""

    def test_objectives_correct_signs(self, catalog, physics, default_config):
        """Objectives have correct signs (both negative for maximization)."""
        chromosome = np.array([2] * 16, dtype=np.int32)  # 16x R40_LEFT
        layout = build_layout(chromosome, catalog)
        profile = compute_speed_profile(layout, catalog, physics)

        F = compute_objectives(layout, profile, catalog, default_config.total_inventory)

        # 2 objectives: utilization and speed
        assert len(F) == 2

        # F[0] should be negative utilization (to maximize)
        assert F[0] < 0  # Negative utilization

        # F[1] should be negative speed (to maximize)
        assert F[1] < 0  # Negative speed

    def test_utilization_range(self, catalog, default_config):
        """Utilization in [0, 1] range."""
        chromosome = np.array([0] * 10 + [-1] * 54, dtype=np.int32)  # 10 pieces
        layout = build_layout(chromosome, catalog)
        profile = compute_speed_profile(layout, catalog, default_config.physics)

        F = compute_objectives(layout, profile, catalog, default_config.total_inventory)

        utilization = -F[0]  # Flip sign to get actual utilization
        assert 0.0 <= utilization <= 1.0

    def test_speed_objective_negative(self, catalog, physics, default_config):
        """Speed objective is negative (for maximization)."""
        chromosome = np.array([2] * 16, dtype=np.int32)
        layout = build_layout(chromosome, catalog)
        profile = compute_speed_profile(layout, catalog, physics)

        F = compute_objectives(layout, profile, catalog, default_config.total_inventory)

        assert F[1] < 0  # Negative speed


class TestConstraints:
    """Tests for constraint function computation."""

    def test_closed_layout_feasible(self, catalog, default_config):
        """16 R40 circle satisfies closure constraints (G[0], G[1] <= 0)."""
        chromosome = np.full(16, 2, dtype=np.int32)  # 16x R40_LEFT
        layout = build_layout(chromosome, catalog)

        G = compute_constraints(
            layout,
            chromosome,
            default_config.inventory,
            catalog,
            default_config.closure_tolerance,
            default_config.angle_tolerance,
            default_config.boundary,
        )

        # Closure constraints should be satisfied
        assert G[0] <= 0  # Position closure
        assert G[1] <= 0  # Angle closure

    def test_open_layout_infeasible(self, catalog, default_config):
        """4 straights violate closure constraint (G[0] > 0)."""
        chromosome = np.array([0, 0, 0, 0], dtype=np.int32)  # 4x STRAIGHT_16
        layout = build_layout(chromosome, catalog)

        G = compute_constraints(
            layout,
            chromosome,
            default_config.inventory,
            catalog,
            default_config.closure_tolerance,
            default_config.angle_tolerance,
            default_config.boundary,
        )

        # Open layout violates closure
        assert G[0] > 0  # Position not closed

    def test_inventory_constraint(self, catalog, default_config):
        """Excess pieces violate inventory (G[3] > 0)."""
        # Create chromosome with 30 STRAIGHT_16 (inventory only has 16)
        chromosome = np.array([0] * 30, dtype=np.int32)
        layout = build_layout(chromosome, catalog)

        G = compute_constraints(
            layout,
            chromosome,
            default_config.inventory,
            catalog,
            default_config.closure_tolerance,
            default_config.angle_tolerance,
            default_config.boundary,
        )

        # Should violate inventory constraint
        assert G[3] > 0  # Inventory excess

    def test_orphan_switch_constraint(self, catalog, default_config):
        """Unpaired switches violate orphan constraint (G[4] > 0)."""
        # 3x LEFT_IN, 1x LEFT_OUT = 2 orphans
        chromosome = np.array([5, 5, 5, 6], dtype=np.int32)
        layout = build_layout(chromosome, catalog)

        G = compute_constraints(
            layout,
            chromosome,
            default_config.inventory,
            catalog,
            default_config.closure_tolerance,
            default_config.angle_tolerance,
            default_config.boundary,
        )

        # Should have orphan switches
        assert G[4] == 2.0  # 2 orphan switches
