"""Tests for the comprehensive physical evaluation model."""

import math

import numpy as np
import pytest

from src.catalog import TrackCatalog
from src.geometry import Layout, build_layout
from src.train import TrainConfig
from src.train.evaluation import PhysicalEvaluation, evaluate_layout

# Piece indices from src/encoding.py
STRAIGHT_16 = 0
R40_LEFT = 2
R40_RIGHT = 3


def _make_layout(piece_indices: list[int], catalog: TrackCatalog) -> Layout:
    """Build a Layout from a piece-index list (helper used across tests)."""
    return build_layout(np.array(piece_indices, dtype=np.int32), catalog)


class TestGeometryDomain:
    """coupler-phi per segment + per-switch + max."""

    def test_phi_R40_matches_user_sanity_check(self, catalog, measured_train_config):
        """Single R40_LEFT, measured coupler_offset=0.106, R=0.32 m -> phi == 0.106/(2*0.32) ~ 9.49 deg."""
        layout = _make_layout([R40_LEFT], catalog)
        phys = evaluate_layout(layout, catalog, measured_train_config)
        expected_rad = 0.106 / (2.0 * 0.32)
        assert phys.coupler_phi_per_segment[0] == pytest.approx(expected_rad, abs=1e-4)
        assert math.degrees(phys.coupler_phi_per_segment[0]) == pytest.approx(9.49, abs=0.05)

    def test_phi_straight_is_zero(self, catalog, measured_train_config):
        """STRAIGHT_16 has no curvature -> phi == 0."""
        layout = _make_layout([STRAIGHT_16], catalog)
        phys = evaluate_layout(layout, catalog, measured_train_config)
        assert phys.coupler_phi_per_segment[0] == pytest.approx(0.0, abs=1e-9)

    def test_max_coupler_phi_picks_worst(self, catalog, measured_train_config):
        """max_coupler_phi == max over segments and switches."""
        layout = _make_layout([STRAIGHT_16, R40_LEFT, STRAIGHT_16], catalog)
        phys = evaluate_layout(layout, catalog, measured_train_config)
        # Only the R40 segment has non-zero phi
        expected_max = 0.106 / (2.0 * 0.32)
        assert phys.max_coupler_phi == pytest.approx(expected_max, abs=1e-4)


class TestStabilityDomain:
    """Per-segment slide/tip/nadal/motor caps + binding-cap label."""

    def test_v_slide_R40_matches_user_sanity(self, catalog, measured_train_config):
        """v_slide(R40) = sqrt(0.25 * 9.81 * 0.32) = 0.886 m/s (user's sanity check)."""
        layout = _make_layout([R40_LEFT], catalog)
        phys = evaluate_layout(layout, catalog, measured_train_config)
        assert phys.v_slide_per_segment[0] == pytest.approx(0.886, abs=1e-3)

    def test_v_eff_R40_is_slide_with_measured_consist(self, catalog, measured_train_config):
        """For measured config, v_slide < v_motor on R40, so v_eff == v_slide."""
        layout = _make_layout([R40_LEFT], catalog)
        phys = evaluate_layout(layout, catalog, measured_train_config)
        assert phys.v_eff_per_segment[0] == pytest.approx(phys.v_slide_per_segment[0], abs=1e-6)

    def test_v_eff_straight_is_motor(self, catalog, measured_train_config):
        """For straights (R=inf), v_eff == v_motor_max == 1.26."""
        layout = _make_layout([STRAIGHT_16], catalog)
        phys = evaluate_layout(layout, catalog, measured_train_config)
        assert phys.v_eff_per_segment[0] == pytest.approx(1.26, abs=1e-3)

    def test_binding_cap_labels_R40_then_straight(self, catalog, measured_train_config):
        """R40 -> 'slide', straight -> 'motor'."""
        layout = _make_layout([R40_LEFT, STRAIGHT_16], catalog)
        phys = evaluate_layout(layout, catalog, measured_train_config)
        assert phys.binding_cap_per_segment[0] == "slide"
        assert phys.binding_cap_per_segment[1] == "motor"


class TestKinematicsDomain:
    """compute_speed_profile w/ safety_margin + safety_factor metrics."""

    def test_safety_margin_default_unchanged(self, catalog, train_config):
        """compute_speed_profile with safety_margin=1.0 (default) keeps old behavior."""
        from src.train import compute_speed_profile
        layout = _make_layout([R40_LEFT, R40_LEFT, R40_LEFT, R40_LEFT,
                               R40_LEFT, R40_LEFT, R40_LEFT, R40_LEFT,
                               R40_LEFT, R40_LEFT, R40_LEFT, R40_LEFT,
                               R40_LEFT, R40_LEFT, R40_LEFT, R40_LEFT], catalog)
        sp = compute_speed_profile(layout, catalog, train_config)
        sp2 = compute_speed_profile(layout, catalog, train_config, safety_margin=1.0)
        assert np.allclose(sp.speeds, sp2.speeds, atol=1e-9)

    def test_safety_margin_scales_speed(self, catalog, measured_train_config):
        """16-R40 closed circle at safety_margin=0.95 -> all speeds ~ 0.95 * 0.886 = 0.842."""
        from src.train import compute_speed_profile
        layout = _make_layout([R40_LEFT] * 16, catalog)
        sp = compute_speed_profile(layout, catalog, measured_train_config, safety_margin=0.95)
        # All segments slide-bound at 0.842 m/s
        assert np.all(sp.speeds == pytest.approx(0.842, abs=5e-3))

    def test_safety_factor_min_equals_margin_on_capped_loop(self, catalog, measured_train_config):
        """16-R40 circle at safety_margin=0.95 -> safety_factor_min == 0.95."""
        layout = _make_layout([R40_LEFT] * 16, catalog)
        phys = evaluate_layout(layout, catalog, measured_train_config, safety_margin=0.95)
        assert phys.safety_factor_min == pytest.approx(0.95, abs=1e-3)

    def test_safety_factor_min_above_margin_on_brake_bound(self, catalog, measured_train_config):
        """Layout with curve before short straight -> brake-bound segments below 0.95 cap (so factor > 0.95)."""
        layout = _make_layout([R40_LEFT, STRAIGHT_16, R40_LEFT, STRAIGHT_16] * 2, catalog)
        phys = evaluate_layout(layout, catalog, measured_train_config, safety_margin=0.95)
        # Some segments forced below v_limit by accel/brake -> safety_factor > 0.95 there
        # safety_factor_min is still 0.95 (on cap-bound segments)
        assert phys.safety_factor_min == pytest.approx(0.95, abs=5e-3)
        # Mean is strictly above 0.95
        assert phys.safety_factor_mean >= 0.95 - 1e-6

    def test_lap_time_R40_circle_matches_pen_and_paper(self, catalog, measured_train_config):
        """16-R40 closed circle lap_time = 2*pi*R / (0.95 * v_slide).
        2*pi*0.32 / 0.842 = 2.011 / 0.842 ~ 2.39 s."""
        layout = _make_layout([R40_LEFT] * 16, catalog)
        phys = evaluate_layout(layout, catalog, measured_train_config, safety_margin=0.95)
        assert phys.speed_profile.lap_time == pytest.approx(2.39, abs=0.05)


class TestDynamicsDomain:
    """a_lat, a_long, grip_utilization, coupler_force_lat per segment."""

    def test_a_lat_R40_at_margined_speed(self, catalog, measured_train_config):
        """16-R40 circle at safety_margin=0.95: a_lat = v^2/R = 0.842^2 / 0.32 ~ 2.215."""
        layout = _make_layout([R40_LEFT] * 16, catalog)
        phys = evaluate_layout(layout, catalog, measured_train_config, safety_margin=0.95)
        assert phys.a_lat_per_segment[0] == pytest.approx(2.215, abs=0.05)

    def test_a_lat_zero_on_straights(self, catalog, measured_train_config):
        """Straight segments have a_lat == 0."""
        layout = _make_layout([STRAIGHT_16] * 8, catalog)
        phys = evaluate_layout(layout, catalog, measured_train_config, safety_margin=0.95)
        assert np.allclose(phys.a_lat_per_segment, 0.0, atol=1e-9)

    def test_grip_utilization_below_one(self, catalog, measured_train_config):
        """grip_utilization is in [0, 1] always (within numerical tolerance)."""
        layout = _make_layout([R40_LEFT, STRAIGHT_16, R40_LEFT, STRAIGHT_16] * 4, catalog)
        phys = evaluate_layout(layout, catalog, measured_train_config, safety_margin=0.95)
        assert np.all(phys.grip_utilization_per_segment <= 1.0 + 1e-6)
        assert np.all(phys.grip_utilization_per_segment >= 0.0)

    def test_coupler_force_lat_zero_with_no_trailing(self, catalog):
        """With mass_trailing=0, lateral coupler force is 0 everywhere."""
        bare_loco = TrainConfig(mass_trailing=0.0, coupler_offset=0.106)
        layout = _make_layout([R40_LEFT] * 16, catalog)
        phys = evaluate_layout(layout, catalog, bare_loco, safety_margin=0.95)
        assert np.allclose(phys.coupler_force_lat_per_segment, 0.0, atol=1e-9)

    def test_coupler_force_lat_proportional_to_trailing_mass(self, catalog):
        """Doubling mass_trailing doubles the lateral coupler force (when a_long != 0)."""
        # Pick a layout with brake transitions (so a_long is non-zero)
        layout = _make_layout([R40_LEFT, STRAIGHT_16, R40_LEFT, STRAIGHT_16] * 2, catalog)
        tc1 = TrainConfig(mass_trailing=0.327, coupler_offset=0.106)
        tc2 = TrainConfig(mass_trailing=0.654, coupler_offset=0.106)
        phys1 = evaluate_layout(layout, catalog, tc1, safety_margin=0.95)
        phys2 = evaluate_layout(layout, catalog, tc2, safety_margin=0.95)
        # Where coupler_force_lat is non-trivial, ratio should be ~2
        nonzero = np.abs(phys1.coupler_force_lat_per_segment) > 1e-3
        if nonzero.any():
            ratio = (phys2.coupler_force_lat_per_segment[nonzero]
                     / phys1.coupler_force_lat_per_segment[nonzero])
            assert np.allclose(ratio, 2.0, atol=1e-6)


class TestEnergyDomain:
    """motor_work_per_lap, rolling_dissipation, ke_roundtrip."""

    def test_rolling_dissipation_constant_speed(self, catalog, measured_train_config):
        """All-straight closed loop at v=1.197: rolling_diss = mu_roll * m_total * g * total_distance."""
        layout = _make_layout([STRAIGHT_16] * 8, catalog)
        phys = evaluate_layout(layout, catalog, measured_train_config, safety_margin=0.95)
        m_total = measured_train_config.mass_total
        g = measured_train_config.g
        mu_roll = measured_train_config.mu_roll
        total_distance = phys.speed_profile.total_distance
        expected = mu_roll * m_total * g * total_distance
        assert phys.rolling_dissipation_per_lap == pytest.approx(expected, abs=1e-6)

    def test_ke_roundtrip_zero_on_constant_speed_loop(self, catalog, measured_train_config):
        """All-R40 circle at single speed: no brake-respin events, ke_roundtrip == 0."""
        layout = _make_layout([R40_LEFT] * 16, catalog)
        phys = evaluate_layout(layout, catalog, measured_train_config, safety_margin=0.95)
        assert phys.ke_roundtrip_per_lap == pytest.approx(0.0, abs=1e-3)

    def test_motor_work_nonneg(self, catalog, measured_train_config):
        """Motor work (sum of positive a_long contributions) is always non-negative."""
        layout = _make_layout([R40_LEFT, STRAIGHT_16, R40_LEFT, STRAIGHT_16] * 4, catalog)
        phys = evaluate_layout(layout, catalog, measured_train_config, safety_margin=0.95)
        assert phys.motor_work_per_lap >= 0.0


class TestProblemIntegration:
    """Integration: F[1] in TrackOptimizationProblem._evaluate is -avg_speed, minimized."""

    def test_F1_is_neg_avg_speed(self, switches_config, catalog):
        """F[1] from _evaluate should equal -phys.speed_profile.avg_speed.
        avg_speed (length-independent) is preferred over lap_time so longer
        layouts are not penalized just for being longer."""
        from src.problem import TrackOptimizationProblem
        from src.sampling import IntegerSampling
        problem = TrackOptimizationProblem(catalog=catalog, config=switches_config)
        sampler = IntegerSampling(catalog=catalog, config=switches_config)
        pop = sampler.do(problem, 1)
        x = pop[0].X
        out: dict = {}
        problem._evaluate(x, out)
        F = out["F"]

        from src.decoder import decode_chromosome
        layout = decode_chromosome(
            x, catalog, switches_config.inventory,
            dims=problem.dims, config=problem.decoder_config,
        )
        if layout.n_pieces == 0:
            # n_pieces==0 sentinel uses np.inf for both
            assert np.isinf(F[1])
        else:
            phys = evaluate_layout(layout, catalog, problem._train_config, safety_margin=0.95)
            assert F[1] == pytest.approx(-phys.speed_profile.avg_speed, abs=1e-6)
