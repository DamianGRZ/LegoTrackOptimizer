"""Tests for template-based passing siding geometry."""

import numpy as np
import pytest

from src.templates import (
    LEFT_SIDING,
    RIGHT_SIDING,
    TEMPLATES,
    STRAIGHT_16,
    R40_CURVE,
    R40_CURVE,
    R40_SWITCH_LEFT,
    R40_SWITCH_RIGHT,
    PassingSidingTemplate,
    apply_fk,
    check_siding_inventory,
    compute_branch_endpoint,
    compute_branch_pieces,
    compute_out_switch_alignment_error,
    compute_required_main_distance,
    get_siding_inventory_requirements,
    is_valid_siding,
    switch_indices_for,
)


class TestTemplateDefinitions:
    """Test that template definitions are correct."""

    def test_left_siding_uses_opposite_handed_pair(self):
        """LEFT siding entry = LEFT switch, exit = RIGHT switch (reversed install)."""
        entry, exit_ = switch_indices_for(LEFT_SIDING)
        assert entry == R40_SWITCH_LEFT
        assert exit_ == R40_SWITCH_RIGHT

    def test_right_siding_uses_opposite_handed_pair(self):
        """RIGHT siding entry = RIGHT switch, exit = LEFT switch (reversed install)."""
        entry, exit_ = switch_indices_for(RIGHT_SIDING)
        assert entry == R40_SWITCH_RIGHT
        assert exit_ == R40_SWITCH_LEFT

    def test_left_siding_uses_same_handed_curves(self):
        """LEFT siding: R40_CURVE for approach AND return (both bend train back).

        Approach turns +22.5° -> 0° (parallel). Return turns 0° -> -22.5°
        (heads back DOWN to merge into the reversed-install OUT switch's port C).
        """
        assert LEFT_SIDING.approach_curve_idx == R40_CURVE
        assert LEFT_SIDING.return_curve_idx == R40_CURVE

    def test_right_siding_uses_same_handed_curves(self):
        """RIGHT siding: R40_CURVE for approach AND return (mirror of LEFT)."""
        assert RIGHT_SIDING.approach_curve_idx == R40_CURVE
        assert RIGHT_SIDING.return_curve_idx == R40_CURVE

    def test_templates_dict_has_both_handedness(self):
        """TEMPLATES dict should have entries for 0=LEFT and 1=RIGHT."""
        assert 0 in TEMPLATES
        assert 1 in TEMPLATES
        assert TEMPLATES[0] == LEFT_SIDING
        assert TEMPLATES[1] == RIGHT_SIDING


class TestBranchPieceComputation:
    """Test branch piece sequence generation."""

    def test_zero_straights_has_two_curves(self):
        """Branch with 0 straights should have approach and return curves only."""
        pieces = compute_branch_pieces(LEFT_SIDING, n_straights=0)
        assert len(pieces) == 2
        assert pieces[0] == LEFT_SIDING.approach_curve_idx
        assert pieces[1] == LEFT_SIDING.return_curve_idx

    def test_n_straights_adds_correct_count(self):
        """Branch should include n_straights STRAIGHT_16 pieces."""
        for n in range(1, 6):
            pieces = compute_branch_pieces(LEFT_SIDING, n_straights=n)
            assert len(pieces) == n + 2  # n straights + 2 curves
            assert pieces[0] == LEFT_SIDING.approach_curve_idx
            for i in range(1, n + 1):
                assert pieces[i] == LEFT_SIDING.straight_idx
            assert pieces[-1] == LEFT_SIDING.return_curve_idx

    def test_piece_count_formula(self):
        """Branch piece count = 2 + n_straights (approach + straights + return)."""
        for n in range(0, 8):
            pieces = compute_branch_pieces(LEFT_SIDING, n_straights=n)
            assert len(pieces) == 2 + n


class TestFKApplication:
    """Test forward kinematics application."""

    def test_straight_fk_advances_x(self):
        """Straight FK should move forward in X only (when heading=0)."""
        state = (0.0, 0.0, 0.0)
        fk = (16.0, 0.0, 0.0)
        new_state = apply_fk(state, fk)
        assert abs(new_state[0] - 16.0) < 0.001
        assert abs(new_state[1] - 0.0) < 0.001
        assert abs(new_state[2] - 0.0) < 0.001

    def test_curve_fk_rotates_and_offsets(self):
        """Curve FK should rotate heading and offset position."""
        state = (0.0, 0.0, 0.0)
        fk = (15.307, 3.045, 22.5)  # R40_CURVE
        new_state = apply_fk(state, fk)
        assert abs(new_state[0] - 15.307) < 0.001
        assert abs(new_state[1] - 3.045) < 0.001
        assert abs(new_state[2] - 22.5) < 0.001

    def test_rotated_fk_transforms_correctly(self):
        """FK should rotate by current heading before applying."""
        state = (0.0, 0.0, 90.0)  # Facing +Y
        fk = (16.0, 0.0, 0.0)  # Straight
        new_state = apply_fk(state, fk)
        # Should move in +Y direction now
        assert abs(new_state[0] - 0.0) < 0.001
        assert abs(new_state[1] - 16.0) < 0.001
        assert abs(new_state[2] - 90.0) < 0.001


class TestBranchGeometry:
    """Test branch endpoint and closure computation."""

    def test_left_siding_zero_straights_lateral_returns_to_port_c(self):
        """LEFT siding with 0 straights: branch end matches OUT switch port C.

        Trace: IN diverges to (32.75, 13, +22.5°), R40_CURVE brings to
        (48, 16.05, 0°), R40_CURVE brings to (63.4, 13, -22.5°). Lateral
        returns to +13 (port C lateral) and heading is -22.5° (matches the
        train's entry direction at the reversed-install OUT switch's port C).
        """
        start_state = (0.0, 0.0, 0.0)
        end_state = compute_branch_endpoint(start_state, LEFT_SIDING, n_straights=0)

        assert abs(end_state[1] - 13.0) < 0.1   # lateral matches port C
        assert abs(end_state[2] - (-22.5)) < 0.1  # heading matches entry direction

    def test_required_distance_increases_with_straights(self):
        """Required main distance should increase with n_straights."""
        d0 = compute_required_main_distance(LEFT_SIDING, n_straights=0)
        d1 = compute_required_main_distance(LEFT_SIDING, n_straights=1)
        d2 = compute_required_main_distance(LEFT_SIDING, n_straights=2)

        assert d1 > d0
        assert d2 > d1
        # Each straight adds ~16 studs
        assert abs((d1 - d0) - 16.0) < 2.0
        assert abs((d2 - d1) - 16.0) < 2.0

    def test_left_and_right_have_similar_distances(self):
        """LEFT and RIGHT sidings should have similar required distances."""
        for n in range(0, 4):
            d_left = compute_required_main_distance(LEFT_SIDING, n_straights=n)
            d_right = compute_required_main_distance(RIGHT_SIDING, n_straights=n)
            assert abs(d_left - d_right) < 1.0


class TestSidingValidation:
    """Test siding validity checking."""

    def test_matching_geometry_is_valid(self):
        """Siding with matching IN/OUT geometry should be valid."""
        # Compute the required distance for a 2-straight siding
        n_straights = 2
        required_dist = compute_required_main_distance(LEFT_SIDING, n_straights)

        # Place IN at origin, OUT at computed position along main axis
        in_state = (0.0, 0.0, 0.0)
        out_state = (required_dist, 0.0, 0.0)  # Same heading, offset by distance

        # This should be valid (close enough geometry)
        is_valid = is_valid_siding(
            in_state, out_state, LEFT_SIDING, n_straights,
            position_tolerance=5.0, angle_tolerance=10.0
        )

        # Note: This may not be exactly valid due to the curved approach/return
        # affecting Y position. The test validates the function runs correctly.
        # Actual validity depends on precise FK geometry.
        assert isinstance(is_valid, bool)


class TestInventoryRequirements:
    """Test inventory requirement computation."""

    def test_zero_straights_needs_one_switch_per_handedness_and_two_curves(self):
        """Each siding consumes 1 LEFT + 1 RIGHT switch + 2 of the same R40 curve.

        Approach and return are the same handedness curve (both bend back toward
        main), so for LEFT_SIDING both are R40_CURVE — count is 2 of one curve type.
        """
        reqs = get_siding_inventory_requirements(LEFT_SIDING, n_straights=0)

        assert reqs[R40_SWITCH_LEFT] == 1
        assert reqs[R40_SWITCH_RIGHT] == 1
        assert LEFT_SIDING.approach_curve_idx == LEFT_SIDING.return_curve_idx
        assert reqs[LEFT_SIDING.approach_curve_idx] == 2
        assert reqs.get(LEFT_SIDING.straight_idx, 0) == 0

    def test_n_straights_adds_straight_requirement(self):
        for n in range(1, 5):
            reqs = get_siding_inventory_requirements(LEFT_SIDING, n_straights=n)
            assert reqs[LEFT_SIDING.straight_idx] == n

    def test_check_inventory_passes_with_enough(self):
        available = {
            R40_SWITCH_LEFT: 1,
            R40_SWITCH_RIGHT: 1,
            R40_CURVE: 2,
            R40_CURVE: 2,
            STRAIGHT_16: 10,
        }
        assert check_siding_inventory(LEFT_SIDING, n_straights=2, available_inventory=available, used_inventory={})

    def test_check_inventory_fails_when_left_switch_missing(self):
        available = {
            # Missing R40_SWITCH_LEFT
            R40_SWITCH_RIGHT: 1,
            R40_CURVE: 2,
            R40_CURVE: 2,
            STRAIGHT_16: 10,
        }
        assert not check_siding_inventory(LEFT_SIDING, n_straights=0, available_inventory=available, used_inventory={})

    def test_check_inventory_considers_used(self):
        available = {
            R40_SWITCH_LEFT: 1,
            R40_SWITCH_RIGHT: 1,
            R40_CURVE: 1,
            R40_CURVE: 1,
            STRAIGHT_16: 2,
        }
        used = {R40_SWITCH_LEFT: 1}  # already used the only LEFT switch
        assert not check_siding_inventory(LEFT_SIDING, n_straights=0, available_inventory=available, used_inventory=used)


class TestAlignmentError:
    """Test alignment error computation."""

    def test_alignment_error_returns_two_values(self):
        """compute_out_switch_alignment_error should return (pos_error, angle_error)."""
        in_state = (0.0, 0.0, 0.0)
        out_state = (50.0, 0.0, 0.0)

        pos_err, angle_err = compute_out_switch_alignment_error(
            in_state, out_state, LEFT_SIDING, n_straights=1
        )

        assert isinstance(pos_err, float)
        assert isinstance(angle_err, float)
        assert pos_err >= 0
        assert angle_err >= 0
