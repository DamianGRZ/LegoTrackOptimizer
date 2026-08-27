"""Tests for speed profile computation."""

import numpy as np
import pytest

from src.train import compute_speed_profile
from src.geometry import build_layout


class TestSpeedProfile:
    """Tests for speed profile computation."""

    def test_straight_track_max_speed(self, catalog, train_config):
        """4 straights should reach motor top speed (1.10 m/s)."""
        chromosome = np.array([0, 0, 0, 0], dtype=np.int32)  # 4x STRAIGHT_16
        layout = build_layout(chromosome, catalog)

        profile = compute_speed_profile(layout, catalog, train_config=train_config)

        assert profile.max_speed == pytest.approx(1.10, abs=0.01)
        assert np.all(profile.speeds <= train_config.v_motor_max)

    def test_r40_circle_capped_by_lateral_slide(self, catalog, train_config):
        """16 R40 curves cap at the sliding-derailment speed, the binding mode.

        sqrt(mu_design * g * R) = sqrt(0.25 * 9.81 * 0.320) = 0.8859 m/s, well
        under tip-over (1.4007) and Nadal wheel-climb (1.5092).
        """
        chromosome = np.full(16, 2, dtype=np.int32)  # 16x R40_CURVE
        layout = build_layout(chromosome, catalog)

        profile = compute_speed_profile(layout, catalog, train_config=train_config)

        assert profile.max_speed == pytest.approx(0.8859, abs=1e-3)
        assert profile.max_speed < train_config.v_motor_max

    def test_double_unroll_closure(self, catalog, train_config):
        """Closed loop speeds consistent at wrap point."""
        chromosome = np.full(16, 2, dtype=np.int32)  # 16x R40_CURVE
        layout = build_layout(chromosome, catalog)

        profile = compute_speed_profile(layout, catalog, train_config=train_config)

        # For closed loop, first and last speeds should be similar
        assert abs(profile.speeds[0] - profile.speeds[-1]) < 0.1

    def test_empty_layout(self, catalog, train_config):
        """Empty layout returns valid SpeedProfile with zeros."""
        chromosome = np.array([-1, -1, -1], dtype=np.int32)
        layout = build_layout(chromosome, catalog)

        profile = compute_speed_profile(layout, catalog, train_config=train_config)

        assert len(profile.speeds) == 0
        assert profile.avg_speed == 0.0
        assert profile.lap_time == 0.0
        assert profile.total_distance == 0.0

    def test_friction_circle_limits_accel_on_curve(self, catalog, train_config):
        """On a mixed layout, the friction circle reduces avg_speed vs unconstrained model."""
        # Build an oval: 8 R40_CURVE curves + 8 straights
        chromosome = np.array([2] * 8 + [0] * 8, dtype=np.int32)
        layout = build_layout(chromosome, catalog)
        profile = compute_speed_profile(layout, catalog, train_config=train_config)

        assert profile.avg_speed > 0.5
        assert profile.avg_speed < train_config.v_motor_max
        assert profile.max_speed <= train_config.v_motor_max + 0.01


class TestRotationInvariance:
    """A closed loop's lap time must not depend on which piece is index 0.

    The triple unroll returns the middle copy, so the forward start transient
    and the backward pass's missing wrap-braking never reach the returned lap.
    Uses the measured consist (low max_accel = long transients) and a racetrack
    with straight runs long enough for braking to propagate several pieces."""

    def test_lap_time_is_rotation_invariant(self, catalog, measured_train_config):
        loop = [2] * 8 + [0] * 8 + [2] * 8 + [0] * 8
        times = set()
        for k in range(len(loop)):
            rotated = loop[k:] + loop[:k]
            layout = build_layout(np.array(rotated, dtype=np.int32), catalog)
            profile = compute_speed_profile(
                layout, catalog, train_config=measured_train_config, safety_margin=0.95,
            )
            times.add(round(profile.lap_time, 9))
        assert len(times) == 1, (
            f"lap time varies with rotation: {sorted(times)}"
        )
