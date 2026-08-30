"""Tests for the TrainConfig physics module."""

import math

import numpy as np
import pytest
from pydantic import ValidationError

from src.train import TrainConfig, TrainConfigError, available_accel, v_eff_array

# The five fields that describe one vehicle rather than shared physics. A train
# YAML must state every one of them; there is no locomotive in the code to fall
# back on. These values are the bare loco of configs/trains/only_loco.yaml.
VEHICLE = dict(v_motor_max=1.10, max_accel=3.92, mass_loco=0.370,
               mass_trailing=0.0, coupler_offset=0.100)


def _vehicle_only(**overrides) -> TrainConfig:
    """A config stating the vehicle alone, so shared fields fall back to defaults."""
    return TrainConfig(**{**VEHICLE, **overrides})


class TestVehicleFieldsAreRequired:
    """What differs between vehicles must be stated, never assumed."""

    @pytest.mark.parametrize("missing", sorted(VEHICLE))
    def test_omitting_a_vehicle_field_is_rejected(self, missing):
        stated = {k: v for k, v in VEHICLE.items() if k != missing}
        with pytest.raises(ValidationError):
            TrainConfig(**stated)

    def test_stating_the_vehicle_alone_is_enough(self):
        assert _vehicle_only().mass_total == pytest.approx(0.370)

    def test_unknown_field_is_rejected(self):
        with pytest.raises(ValidationError):
            _vehicle_only(mu_desing=0.40)

    def test_out_of_range_value_is_rejected(self):
        with pytest.raises(ValidationError):
            _vehicle_only(mass_loco=-0.5)


class TestSharedDefaults:
    """Friction, gravity and bogie geometry are shared assumptions, not per-vehicle
    measurements, so they keep defaults a file may restate."""

    def test_default_mu_nominal(self):
        assert _vehicle_only().mu_nominal == pytest.approx(0.30)

    def test_default_mu_design(self):
        assert _vehicle_only().mu_design == pytest.approx(0.25)

    def test_default_brake_decel(self):
        assert _vehicle_only().brake_decel == pytest.approx(2.45)

    def test_default_mu_roll(self):
        assert _vehicle_only().mu_roll == pytest.approx(0.05)

    def test_mu_roll_can_be_overridden(self):
        assert _vehicle_only(mu_roll=0.10).mu_roll == pytest.approx(0.10)

    def test_legacy_mu_field_removed(self):
        assert "mu" not in TrainConfig.model_fields


class TestSpeedCapsUseMuDesign:
    """Scalar and vectorised speed caps read mu_design, not mu_nominal."""

    def test_v_slide_r40_at_mu_design(self, train_config):
        # R40 = 0.320 m, mu_design = 0.25 -> sqrt(0.25*9.81*0.320) = 0.8862
        assert train_config.v_slide(0.320) == pytest.approx(0.8862, abs=1e-3)

    def test_v_eff_r40_equals_v_slide(self, train_config):
        # Sliding binds below motor cap
        assert train_config.v_eff(0.320) == pytest.approx(train_config.v_slide(0.320), abs=1e-6)

    def test_v_eff_straight_equals_motor_cap(self, train_config):
        assert train_config.v_eff(math.inf) == pytest.approx(1.10, abs=1e-6)

    def test_v_eff_array_vectorised(self, train_config):
        out = v_eff_array(train_config, np.array([0.320, 0.448, np.inf]))
        assert np.allclose(out, [0.8862, 1.0488, 1.10], atol=1e-3)


class TestStrictLoading:
    """A train YAML that does not say what it means is an error, not a default."""

    def _write(self, tmp_path, body):
        path = tmp_path / "train.yaml"
        path.write_text(body, encoding="utf-8")
        return path

    def test_unknown_key_is_rejected(self, tmp_path):
        body = "v_motor_maxx: 5.0\nmax_accel: 0.68\nmass_loco: 0.5\n" \
               "mass_trailing: 0.0\ncoupler_offset: 0.1\n"
        with pytest.raises(TrainConfigError, match="v_motor_maxx"):
            TrainConfig.from_yaml(self._write(tmp_path, body))

    def test_missing_vehicle_field_is_rejected(self, tmp_path):
        with pytest.raises(TrainConfigError, match="coupler_offset"):
            TrainConfig.from_yaml(self._write(tmp_path, "v_motor_max: 1.26\nmax_accel: 0.68\n"))

    def test_empty_file_is_rejected(self, tmp_path):
        with pytest.raises(TrainConfigError):
            TrainConfig.from_yaml(self._write(tmp_path, ""))

    def test_missing_file_names_the_path(self, tmp_path):
        with pytest.raises(TrainConfigError, match="cannot read"):
            TrainConfig.from_yaml(tmp_path / "no_such_train.yaml")

    def test_only_loco_file_restates_the_shared_defaults(self, train_config):
        """The bare-loco preset exists so that baseline lives in a file; it must stay
        field-for-field identical to the defaults it was lifted from."""
        assert train_config == _vehicle_only()

    def test_measured_consist_states_only_what_was_measured(self, measured_train_config):
        """Unmeasured fields are deliberately absent, so they fall back to defaults."""
        assert measured_train_config.v_motor_max == pytest.approx(1.26)
        assert measured_train_config.max_accel == pytest.approx(0.68)
        assert measured_train_config.mass_trailing == pytest.approx(0.327)
        # Absent from the file: the pull test was never run.
        assert measured_train_config.mu_design == pytest.approx(0.25)


class TestDerive:
    """Variants must go through validation; model_copy would not."""

    def test_derive_replaces_a_field(self, measured_train_config):
        assert measured_train_config.derive(mass_trailing=0.0).mass_trailing == 0.0

    def test_derive_leaves_the_original_untouched(self, measured_train_config):
        measured_train_config.derive(mass_trailing=0.0)
        assert measured_train_config.mass_trailing == pytest.approx(0.327)

    def test_derive_rejects_an_unknown_field(self, measured_train_config):
        with pytest.raises(TrainConfigError):
            measured_train_config.derive(mass_trailng=0.0)


class TestConsistFields:
    """TrainConfig carries mass and coupler geometry for consist modeling."""

    def test_mass_total_bare_loco(self):
        assert _vehicle_only().mass_total == pytest.approx(0.370)

    def test_mass_total_with_cars(self):
        assert _vehicle_only(mass_trailing=0.600).mass_total == pytest.approx(0.970)


class TestFrictionCircle:
    """available_accel implements the capped friction circle + coupler correction:
    a_long = min(cap, sqrt((mu*g)^2 - a_lat^2))."""

    def test_straight_accel_capped_by_circle(self, train_config):
        """On a straight (R=inf), the budget is min(max_accel, mu*g): the drive
        cap when torque-limited, the circle when the cap exceeds grip."""
        expected = min(train_config.max_accel, train_config.mu_design * train_config.g)
        assert available_accel(train_config, v=0.5, radius_m=math.inf) == pytest.approx(expected)

    def test_straight_full_brake(self, train_config):
        """On a straight (R=inf), full brake_decel is available."""
        actual = available_accel(train_config, v=0.5, radius_m=math.inf, is_braking=True)
        assert actual == pytest.approx(train_config.brake_decel)

    def test_r40_at_vslide_zero_accel(self, train_config):
        """At v_slide on R40, lateral demand saturates friction — zero accel."""
        v_slide = train_config.v_slide(0.320)
        actual = available_accel(train_config, v=v_slide, radius_m=0.320)
        assert actual == pytest.approx(0.0, abs=0.01)

    def test_r40_at_half_speed_partial_accel(self, train_config):
        """At half v_slide on R40, some accel budget remains."""
        v_half = train_config.v_slide(0.320) / 2.0
        accel = available_accel(train_config, v=v_half, radius_m=0.320)
        assert 0.0 < accel < train_config.max_accel

    def test_r40_at_half_speed_partial_brake(self, train_config):
        """At half v_slide on R40, some brake budget remains."""
        v_half = train_config.v_slide(0.320) / 2.0
        brake = available_accel(train_config, v=v_half, radius_m=0.320, is_braking=True)
        assert 0.0 < brake < train_config.brake_decel

    def test_coupler_reduces_accel_on_curve(self, train_config):
        """Trailing mass reduces available accel on curves via coupler force."""
        consist = train_config.derive(mass_trailing=0.600)
        v = 0.5  # well below v_slide so there IS accel budget to reduce
        a_bare = available_accel(train_config, v=v, radius_m=0.320)
        a_consist = available_accel(consist, v=v, radius_m=0.320)
        assert a_consist < a_bare

    def test_coupler_no_effect_on_straight(self, train_config):
        """Coupler has no effect on straights (coupler angle is zero)."""
        consist = train_config.derive(mass_trailing=0.600)
        a_bare = available_accel(train_config, v=0.5, radius_m=math.inf)
        a_consist = available_accel(consist, v=0.5, radius_m=math.inf)
        assert a_consist == pytest.approx(a_bare)

    def test_coupler_not_applied_during_braking(self, train_config):
        """Coupler correction skipped during braking (conservative)."""
        consist = train_config.derive(mass_trailing=0.600)
        b_bare = available_accel(train_config, v=0.5, radius_m=0.320, is_braking=True)
        b_consist = available_accel(consist, v=0.5, radius_m=0.320, is_braking=True)
        assert b_bare == pytest.approx(b_consist)

    def test_zero_speed_full_budget_on_curve(self, train_config):
        """At v=0 on any curve, no lateral demand — the full min(cap, mu*g)
        budget is available."""
        expected = min(train_config.max_accel, train_config.mu_design * train_config.g)
        assert available_accel(train_config, v=0.0, radius_m=0.320) == pytest.approx(expected)

    def test_torque_limited_cap_not_derated_on_curve(self, train_config):
        """A torque-limited consist (cap well under mu*g, like the measured
        0.68) keeps its FULL cap on a curve until the circle binds — lateral
        demand must not scale the motor torque down (the old ellipse did)."""
        tc = train_config.derive(max_accel=0.68)
        v_half = tc.v_slide(0.320) / 2.0  # a_lat = mu*g/4, grip >> cap
        assert available_accel(tc, v=v_half, radius_m=0.320) == pytest.approx(0.68)

    def test_combined_demand_stays_inside_circle(self, train_config):
        """Over a speed sweep on R40: sqrt(a_lat^2 + a_long^2) <= mu*g and
        a_long <= cap — the invariant the old model violated both ways."""
        a_lat_max = train_config.mu_design * train_config.g
        v_slide = train_config.v_slide(0.320)
        for frac in (0.0, 0.3, 0.6, 0.9, 0.99):
            v = frac * v_slide
            a_long = available_accel(train_config, v=v, radius_m=0.320)
            a_lat = v * v / 0.320
            assert math.hypot(a_lat, a_long) <= a_lat_max + 1e-9
            assert a_long <= train_config.max_accel + 1e-9

    def test_grip_limited_equals_circle(self, train_config):
        """With a cap far above grip, the answer is the circle itself."""
        tc = train_config.derive(max_accel=100.0)
        a_lat_max = tc.mu_design * tc.g
        v = 0.9 * tc.v_slide(0.320)
        a_lat = v * v / 0.320
        expected = math.sqrt(a_lat_max**2 - a_lat**2)
        assert available_accel(tc, v=v, radius_m=0.320) == pytest.approx(expected)
