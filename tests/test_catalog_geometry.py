"""Derivation tests: V2 spec's Pythagorean-triple switch geometry.

The shipped YAML carries the canonical PHYSICAL MEASUREMENTS of the kit
(port C at 32.75, ±13.0); the two-arc derivation verified here lands within
~0.05 stud of them, corroborating the measurements independently.
"""

import math
import numpy as np


def compute_lego_r40_switch_port_c() -> tuple[float, float, float]:
    """
    Two-arc compound path for LEGO R40 switch's diverging leg.

    Arc 1: +36.87° (arctan(3/4)) at R=40, left turn (CCW)
    Arc 2: -14.37° at R=40, right turn (CW)
    Net heading change: 22.5° = π/8

    Returns (dx, dy, dtheta) in studs/radians.
    """
    R = 40.0
    theta1 = math.atan2(3, 4)        # 36.87°, cos=4/5, sin=3/5
    cos1, sin1 = math.cos(theta1), math.sin(theta1)

    C1 = np.array([0.0, R])           # arc 1 center
    rot1 = np.array([[cos1, -sin1], [sin1, cos1]])
    end_vec1 = rot1 @ np.array([0.0, -R])  # (24, -32) expected
    p1 = C1 + end_vec1                # (24, 8)
    heading1 = theta1

    theta2 = math.pi / 8 - theta1     # ~-14.37° (negative = right turn)
    right_perp = np.array([math.sin(heading1), -math.cos(heading1)])
    C2 = p1 + R * right_perp          # (48, -24) expected
    cos2, sin2 = math.cos(theta2), math.sin(theta2)
    rot2 = np.array([[cos2, -sin2], [sin2, cos2]])
    end_vec2 = rot2 @ (p1 - C2)
    p_C = C2 + end_vec2
    return float(p_C[0]), float(p_C[1]), math.pi / 8


class TestSwitchGeometryReference:
    def test_arc_1_chord_is_integer_stud(self):
        """Arc 1 sweep produces the 8·(3,4,5) integer-stud chord."""
        R = 40.0
        theta1 = math.atan2(3, 4)
        cos1, sin1 = math.cos(theta1), math.sin(theta1)
        rot1 = np.array([[cos1, -sin1], [sin1, cos1]])
        end_vec1 = rot1 @ np.array([0.0, -R])
        assert abs(end_vec1[0] - 24.0) < 1e-9
        assert abs(end_vec1[1] - (-32.0)) < 1e-9

    def test_arc_1_end_at_integer_stud(self):
        """End of arc 1 is at piece-local (24, 8)."""
        R = 40.0
        theta1 = math.atan2(3, 4)
        cos1, sin1 = math.cos(theta1), math.sin(theta1)
        p1 = np.array([0, R]) + np.array([[cos1, -sin1], [sin1, cos1]]) @ np.array([0, -R])
        assert abs(p1[0] - 24.0) < 1e-9
        assert abs(p1[1] - 8.0) < 1e-9

    def test_arc_2_center_at_integer_stud(self):
        """Arc 2 center is at piece-local (48, -24)."""
        theta1 = math.atan2(3, 4)
        right_perp = np.array([math.sin(theta1), -math.cos(theta1)])
        p1 = np.array([24.0, 8.0])
        C2 = p1 + 40.0 * right_perp
        assert abs(C2[0] - 48.0) < 1e-9
        assert abs(C2[1] - (-24.0)) < 1e-9

    def test_port_c_derivation_matches_v2_spec(self):
        """V2's derivation yields port C ≈ (32.69, 12.96, π/8).

        Note: the V2 catalog report quotes (32.71, 12.96) to 4 s.f., but the
        exact two-arc composition using atan2(3, 4) for theta1 yields
        dx ≈ 32.693, dy ≈ 12.955. The 32.71 figure is a rounding artifact.
        """
        dx, dy, dtheta = compute_lego_r40_switch_port_c()
        assert abs(dx - 32.693) < 0.01, f"dx={dx:.4f}, expected ~32.693"
        assert abs(dy - 12.955) < 0.01, f"dy={dy:.4f}, expected ~12.955"
        assert abs(dtheta - math.pi / 8) < 1e-9

    def test_shipped_yaml_matches_canonical_measurements(self):
        """REGRESSION GUARD: port C ships the measured kit values (32.75, 13.0);
        the two-arc derivation corroborates them to within 0.1 stud."""
        from pathlib import Path
        from src.catalog.loader import load_catalog_spec
        cat = load_catalog_spec(Path("data") / "track_pieces_v2.yaml")
        port_c = cat.by_id["R40_SWITCH_LEFT"].ports["C"]
        assert port_c.dx == 32.75, f"port C dx {port_c.dx} drifted from the measured 32.75"
        assert port_c.dy == 13.0, f"port C dy {port_c.dy} drifted from the measured 13.0"

        v2_dx, v2_dy, _ = compute_lego_r40_switch_port_c()
        assert abs(port_c.dx - v2_dx) < 0.1
        assert abs(port_c.dy - v2_dy) < 0.1
