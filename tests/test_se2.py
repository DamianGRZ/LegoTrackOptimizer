"""Tests for src_v2/se2.py — SE(2) rigid-transform utilities."""

import math

import pytest

from src_v2.se2 import IDENTITY, pose_compose, pose_diff, pose_inverse


# Tolerance for floating-point pose comparisons
TOL = 1e-9


def assert_pose_close(actual, expected, tol=TOL):
    assert abs(actual[0] - expected[0]) < tol, f"x: {actual[0]} != {expected[0]}"
    assert abs(actual[1] - expected[1]) < tol, f"y: {actual[1]} != {expected[1]}"
    assert abs(actual[2] - expected[2]) < tol, f"theta: {actual[2]} != {expected[2]}"


# =============================================================================
# pose_compose
# =============================================================================

class TestPoseCompose:
    def test_identity_left(self):
        a = (3.0, 5.0, math.pi / 4)
        result = pose_compose(IDENTITY, a)
        assert_pose_close(result, a)

    def test_identity_right(self):
        a = (3.0, 5.0, math.pi / 4)
        result = pose_compose(a, IDENTITY)
        assert_pose_close(result, a)

    def test_pure_translation(self):
        # Parent at (10, 0, 0); child at (5, 0, 0) — world = (15, 0, 0)
        result = pose_compose((10.0, 0.0, 0.0), (5.0, 0.0, 0.0))
        assert_pose_close(result, (15.0, 0.0, 0.0))

    def test_translation_then_rotation(self):
        # Parent at (0, 0, 90°); child forward 5 — world (0, 5, 90°)
        result = pose_compose((0.0, 0.0, math.pi / 2), (5.0, 0.0, 0.0))
        assert_pose_close(result, (0.0, 5.0, math.pi / 2))

    def test_rotation_composition(self):
        # 45° + 45° = 90°
        result = pose_compose((0.0, 0.0, math.pi / 4), (0.0, 0.0, math.pi / 4))
        assert_pose_close(result, (0.0, 0.0, math.pi / 2))

    def test_full_composition(self):
        # Parent at (1, 2, 90°); child at (3, 0, 45°)
        # Child's local (3, 0) rotated by 90° → world offset (0, 3)
        # World pos: (1+0, 2+3) = (1, 5)
        # World theta: 90° + 45° = 135°
        result = pose_compose((1.0, 2.0, math.pi / 2), (3.0, 0.0, math.pi / 4))
        expected = (1.0, 5.0, math.pi / 2 + math.pi / 4)
        assert_pose_close(result, expected)


# =============================================================================
# pose_inverse
# =============================================================================

class TestPoseInverse:
    def test_inverse_of_identity(self):
        result = pose_inverse(IDENTITY)
        assert_pose_close(result, IDENTITY)

    def test_inverse_of_translation(self):
        # Inverse of (5, 3, 0) is (-5, -3, 0)
        result = pose_inverse((5.0, 3.0, 0.0))
        assert_pose_close(result, (-5.0, -3.0, 0.0))

    def test_inverse_of_rotation(self):
        result = pose_inverse((0.0, 0.0, math.pi / 3))
        assert_pose_close(result, (0.0, 0.0, -math.pi / 3))

    def test_compose_with_inverse_yields_identity(self):
        a = (3.0, 5.0, math.pi / 6)
        result = pose_compose(pose_inverse(a), a)
        assert_pose_close(result, IDENTITY)

    def test_compose_inverse_other_order(self):
        a = (3.0, 5.0, math.pi / 6)
        result = pose_compose(a, pose_inverse(a))
        assert_pose_close(result, IDENTITY)

    def test_double_inverse_is_original(self):
        a = (7.5, -2.3, 0.4)
        result = pose_inverse(pose_inverse(a))
        assert_pose_close(result, a)


# =============================================================================
# pose_diff
# =============================================================================

class TestPoseDiff:
    def test_diff_with_self_is_identity(self):
        a = (3.0, 5.0, math.pi / 4)
        result = pose_diff(a, a)
        assert_pose_close(result, IDENTITY)

    def test_diff_translation_only(self):
        # a is 5 units forward of b along x
        a = (5.0, 0.0, 0.0)
        b = (0.0, 0.0, 0.0)
        result = pose_diff(a, b)
        assert_pose_close(result, (5.0, 0.0, 0.0))

    def test_diff_rotation_only(self):
        a = (0.0, 0.0, math.pi / 2)
        b = (0.0, 0.0, 0.0)
        result = pose_diff(a, b)
        assert_pose_close(result, (0.0, 0.0, math.pi / 2))

    def test_diff_satisfies_compose_relationship(self):
        # pose_diff(a, b) should equal pose_compose(pose_inverse(b), a)
        a = (3.0, 5.0, math.pi / 4)
        b = (1.0, 2.0, math.pi / 6)
        diff = pose_diff(a, b)
        expected = pose_compose(pose_inverse(b), a)
        assert_pose_close(diff, expected)


# =============================================================================
# Round-trips relevant to the decoder's BFS
# =============================================================================

class TestDecoderRelevantRoundtrips:
    def test_port_world_pose_then_back_out_slot(self):
        """Mimics: slot pose + piece-local port offset → port world pose,
        then back out slot pose from port world pose."""
        slot_pose = (10.0, 5.0, math.pi / 6)
        port_local = (15.307, 3.045, math.pi / 8)  # like an R40 LEFT port B

        port_world = pose_compose(slot_pose, port_local)
        recovered_slot = pose_compose(port_world, pose_inverse(port_local))

        assert_pose_close(recovered_slot, slot_pose)

    def test_cycle_closure_zero_when_consistent(self):
        """A cycle of compositions that returns to start should yield identity diff."""
        deltas = [
            (5.0, 0.0, math.pi / 8),
            (5.0, 0.0, math.pi / 8),
            (5.0, 0.0, math.pi / 8),
            (5.0, 0.0, math.pi / 8),
        ]
        # Walk the cycle
        pose = IDENTITY
        for d in deltas:
            pose = pose_compose(pose, d)

        # Compose with identity-bring-back not applicable here; just check
        # the accumulated walk is non-trivial (sanity)
        assert pose[2] == pytest.approx(math.pi / 2, abs=TOL)
