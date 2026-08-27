"""Tests for the portable TrainConfig physics module."""

import math

import numpy as np
import pytest

from src.train import DEFAULT_TRAIN_CONFIG, TrainConfig, available_accel, v_eff_array


class TestTrainConfigFields:
    """TrainConfig expands with mu_nominal/mu_design/max_accel/brake_decel."""

    def test_default_mu_nominal(self):
        assert TrainConfig().mu_nominal == 0.30

    def test_default_mu_design(self):
        assert TrainConfig().mu_design == 0.25

    def test_default_max_accel(self):
        assert TrainConfig().max_accel == pytest.approx(3.92)

    def test_default_brake_decel(self):
        assert TrainConfig().brake_decel == pytest.approx(2.45)

    def test_legacy_mu_field_removed(self):
        assert not hasattr(TrainConfig(), "mu")

    def test_default_mu_roll(self):
        assert TrainConfig().mu_roll == pytest.approx(0.05)

    def test_mu_roll_can_be_overridden(self):
        assert TrainConfig(mu_roll=0.10).mu_roll == pytest.approx(0.10)


class TestSpeedCapsUseMuDesign:
    """Scalar and vectorised speed caps read mu_design, not mu_nominal."""

    def test_v_slide_r40_at_mu_design(self):
        # R40 = 0.320 m, mu_design = 0.25 -> sqrt(0.25*9.81*0.320) = 0.8862
        assert TrainConfig().v_slide(0.320) == pytest.approx(0.8862, abs=1e-3)

    def test_v_eff_r40_equals_v_slide(self):
        # Sliding binds below motor cap
        tc = TrainConfig()
        assert tc.v_eff(0.320) == pytest.approx(tc.v_slide(0.320), abs=1e-6)

    def test_v_eff_straight_equals_motor_cap(self):
        assert TrainConfig().v_eff(math.inf) == pytest.approx(1.10, abs=1e-6)

    def test_v_eff_array_vectorised(self):
        tc = TrainConfig()
        out = v_eff_array(tc, np.array([0.320, 0.448, np.inf]))
        assert np.allclose(out, [0.8862, 1.0488, 1.10], atol=1e-3)


class TestYamlRoundTrip:
    """YAML loader still tolerates empty files and ignores unknown keys."""

    def test_from_yaml_default_file(self):
        tc = TrainConfig.from_yaml("configs/trains/default.yaml")
        assert tc.mass_loco == pytest.approx(0.370)
        assert tc.mass_trailing == pytest.approx(0.600)
        assert tc.max_accel == pytest.approx(1.49)

    def test_default_singleton(self):
        assert DEFAULT_TRAIN_CONFIG == TrainConfig()


class TestConsistFields:
    """TrainConfig carries mass and coupler geometry for consist modeling."""

    def test_default_mass_loco(self):
        assert TrainConfig().mass_loco == pytest.approx(0.370)

    def test_default_mass_trailing(self):
        assert TrainConfig().mass_trailing == pytest.approx(0.0)

    def test_default_coupler_offset(self):
        assert TrainConfig().coupler_offset == pytest.approx(0.100)

    def test_mass_total_bare_loco(self):
        assert TrainConfig().mass_total == pytest.approx(0.370)

    def test_mass_total_with_cars(self):
        tc = TrainConfig(mass_loco=0.370, mass_trailing=0.600)
        assert tc.mass_total == pytest.approx(0.970)


class TestFrictionCircle:
    """available_accel implements the capped friction circle + coupler correction:
    a_long = min(cap, sqrt((mu*g)^2 - a_lat^2))."""

    def test_straight_accel_capped_by_circle(self):
        """On a straight (R=inf), the budget is min(max_accel, mu*g): the drive
        cap when torque-limited, the circle when the cap exceeds grip."""
        tc = TrainConfig()
        expected = min(tc.max_accel, tc.mu_design * tc.g)
        assert available_accel(tc, v=0.5, radius_m=math.inf) == pytest.approx(expected)

    def test_straight_full_brake(self):
        """On a straight (R=inf), full brake_decel is available."""
        tc = TrainConfig()
        assert available_accel(tc, v=0.5, radius_m=math.inf, is_braking=True) == pytest.approx(
            tc.brake_decel
        )

    def test_r40_at_vslide_zero_accel(self):
        """At v_slide on R40, lateral demand saturates friction — zero accel."""
        tc = TrainConfig()
        v_slide = tc.v_slide(0.320)
        assert available_accel(tc, v=v_slide, radius_m=0.320) == pytest.approx(0.0, abs=0.01)

    def test_r40_at_half_speed_partial_accel(self):
        """At half v_slide on R40, some accel budget remains."""
        tc = TrainConfig()
        v_half = tc.v_slide(0.320) / 2.0
        accel = available_accel(tc, v=v_half, radius_m=0.320)
        assert 0.0 < accel < tc.max_accel

    def test_r40_at_half_speed_partial_brake(self):
        """At half v_slide on R40, some brake budget remains."""
        tc = TrainConfig()
        v_half = tc.v_slide(0.320) / 2.0
        brake = available_accel(tc, v=v_half, radius_m=0.320, is_braking=True)
        assert 0.0 < brake < tc.brake_decel

    def test_coupler_reduces_accel_on_curve(self):
        """Trailing mass reduces available accel on curves via coupler force."""
        tc_bare = TrainConfig(mass_trailing=0.0)
        tc_consist = TrainConfig(mass_trailing=0.600)
        v = 0.5  # well below v_slide so there IS accel budget to reduce
        a_bare = available_accel(tc_bare, v=v, radius_m=0.320)
        a_consist = available_accel(tc_consist, v=v, radius_m=0.320)
        assert a_consist < a_bare

    def test_coupler_no_effect_on_straight(self):
        """Coupler has no effect on straights (coupler angle is zero)."""
        tc_bare = TrainConfig(mass_trailing=0.0)
        tc_consist = TrainConfig(mass_trailing=0.600)
        a_bare = available_accel(tc_bare, v=0.5, radius_m=math.inf)
        a_consist = available_accel(tc_consist, v=0.5, radius_m=math.inf)
        assert a_consist == pytest.approx(a_bare)

    def test_coupler_not_applied_during_braking(self):
        """Coupler correction skipped during braking (conservative)."""
        tc_bare = TrainConfig(mass_trailing=0.0)
        tc_consist = TrainConfig(mass_trailing=0.600)
        b_bare = available_accel(tc_bare, v=0.5, radius_m=0.320, is_braking=True)
        b_consist = available_accel(tc_consist, v=0.5, radius_m=0.320, is_braking=True)
        assert b_bare == pytest.approx(b_consist)

    def test_zero_speed_full_budget_on_curve(self):
        """At v=0 on any curve, no lateral demand — the full min(cap, mu*g)
        budget is available."""
        tc = TrainConfig()
        expected = min(tc.max_accel, tc.mu_design * tc.g)
        assert available_accel(tc, v=0.0, radius_m=0.320) == pytest.approx(expected)

    def test_torque_limited_cap_not_derated_on_curve(self):
        """A torque-limited consist (cap well under mu*g, like the measured
        0.68) keeps its FULL cap on a curve until the circle binds — lateral
        demand must not scale the motor torque down (the old ellipse did)."""
        tc = TrainConfig(max_accel=0.68, mass_trailing=0.0)
        v_half = tc.v_slide(0.320) / 2.0  # a_lat = mu*g/4, grip >> cap
        assert available_accel(tc, v=v_half, radius_m=0.320) == pytest.approx(0.68)

    def test_combined_demand_stays_inside_circle(self):
        """Over a speed sweep on R40: sqrt(a_lat^2 + a_long^2) <= mu*g and
        a_long <= cap — the invariant the old model violated both ways."""
        tc = TrainConfig(mass_trailing=0.0)
        a_lat_max = tc.mu_design * tc.g
        v_slide = tc.v_slide(0.320)
        for frac in (0.0, 0.3, 0.6, 0.9, 0.99):
            v = frac * v_slide
            a_long = available_accel(tc, v=v, radius_m=0.320)
            a_lat = v * v / 0.320
            assert math.hypot(a_lat, a_long) <= a_lat_max + 1e-9
            assert a_long <= tc.max_accel + 1e-9

    def test_grip_limited_equals_circle(self):
        """With a cap far above grip, the answer is the circle itself."""
        tc = TrainConfig(max_accel=100.0, mass_trailing=0.0)
        a_lat_max = tc.mu_design * tc.g
        v = 0.9 * tc.v_slide(0.320)
        a_lat = v * v / 0.320
        expected = math.sqrt(a_lat_max**2 - a_lat**2)
        assert available_accel(tc, v=v, radius_m=0.320) == pytest.approx(expected)
