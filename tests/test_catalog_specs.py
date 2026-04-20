"""Tests for Pydantic v2 catalog specs (V2 schema)."""

import math
import pytest
from pydantic import ValidationError

from src.catalog.specs import PortDef


class TestPortDef:
    def test_valid_construction(self):
        """PortDef accepts three floats: dx, dy, dtheta."""
        port = PortDef(dx=0.0, dy=0.0, dtheta=0.0)
        assert port.dx == 0.0
        assert port.dy == 0.0
        assert port.dtheta == 0.0

    def test_frozen(self):
        """PortDef is immutable."""
        port = PortDef(dx=1.0, dy=2.0, dtheta=math.pi)
        with pytest.raises(ValidationError):
            port.dx = 99.0

    def test_extra_field_rejected(self):
        """Unknown fields are rejected by extra='forbid'."""
        with pytest.raises(ValidationError) as exc:
            PortDef(dx=0.0, dy=0.0, dtheta=0.0, color="red")
        assert "extra_forbidden" in str(exc.value) or "Extra inputs" in str(exc.value)

    def test_missing_field_rejected(self):
        """Missing required fields raise ValidationError."""
        with pytest.raises(ValidationError):
            PortDef(dx=0.0, dy=0.0)  # missing dtheta


class TestTrackPieceSpec:
    def _base_straight(self, **overrides):
        from src.catalog.specs import TrackPieceSpec
        payload = dict(
            piece_id="straight_16",
            kind="straight",
            manufacturer="lego",
            part_numbers=("53401",),
            length_studs=16.0,
            ports={
                "A": {"dx": 0.0, "dy": 0.0, "dtheta": 0.0},
                "B": {"dx": 16.0, "dy": 0.0, "dtheta": 0.0},
            },
            routes={"main": ("A", "B")},
        )
        payload.update(overrides)
        return TrackPieceSpec.model_validate(payload)

    def test_valid_straight_constructs(self):
        spec = self._base_straight()
        assert spec.piece_id == "straight_16"
        assert spec.kind == "straight"
        assert spec.manufacturer == "lego"

    def test_port_a_must_be_at_origin(self):
        with pytest.raises(ValidationError) as exc:
            self._base_straight(ports={
                "A": {"dx": 1.0, "dy": 0.0, "dtheta": 0.0},
                "B": {"dx": 16.0, "dy": 0.0, "dtheta": 0.0},
            })
        assert "Port 'A'" in str(exc.value)

    def test_port_a_must_exist(self):
        with pytest.raises(ValidationError) as exc:
            self._base_straight(ports={
                "X": {"dx": 0.0, "dy": 0.0, "dtheta": 0.0},
                "B": {"dx": 16.0, "dy": 0.0, "dtheta": 0.0},
            })
        assert "Port 'A'" in str(exc.value)

    def test_route_references_real_port(self):
        with pytest.raises(ValidationError) as exc:
            self._base_straight(routes={"main": ("A", "Z")})
        assert "route" in str(exc.value).lower() or "undefined" in str(exc.value).lower()

    def test_curve_requires_radius(self):
        from src.catalog.specs import TrackPieceSpec
        with pytest.raises(ValidationError) as exc:
            TrackPieceSpec.model_validate(dict(
                piece_id="curve_missing",
                kind="curve",
                manufacturer="lego",
                ports={
                    "A": {"dx": 0.0, "dy": 0.0, "dtheta": 0.0},
                    "B": {"dx": 15.307, "dy": 3.045, "dtheta": 0.3927},
                },
                routes={"main": ("A", "B")},
                # radius_studs and sector_angle_rad deliberately missing
            ))
        assert "radius_studs" in str(exc.value) or "curve" in str(exc.value)

    def test_switch_requires_diverging_radius(self):
        from src.catalog.specs import TrackPieceSpec
        with pytest.raises(ValidationError):
            TrackPieceSpec.model_validate(dict(
                piece_id="switch_missing",
                kind="switch",
                manufacturer="lego",
                ports={
                    "A": {"dx": 0.0, "dy": 0.0, "dtheta": 0.0},
                    "B": {"dx": 32.0, "dy": 0.0, "dtheta": 0.0},
                    "C": {"dx": 31.0, "dy": 6.2, "dtheta": 0.3927},
                },
                routes={"through": ("A", "B"), "diverging": ("A", "C")},
                body_length_studs=32.0,
                # diverging_radius_studs deliberately missing
            ))

    def test_on_angle_lattice_for_lego_r40(self):
        """R40 curve at 22.5° is on the π/32 lattice (k=4)."""
        from src.catalog.specs import TrackPieceSpec
        spec = TrackPieceSpec.model_validate(dict(
            piece_id="r40_left",
            kind="curve",
            manufacturer="lego",
            radius_studs=40.0,
            sector_angle_rad=math.pi / 8,
            hand="left",
            ports={
                "A": {"dx": 0.0, "dy": 0.0, "dtheta": 0.0},
                "B": {"dx": 15.307, "dy": 3.045, "dtheta": math.pi / 8},
            },
            routes={"main": ("A", "B")},
        ))
        assert spec.on_angle_lattice is True

    def test_on_angle_lattice_false_for_off_lattice_piece(self):
        """A piece whose port dtheta is NOT a multiple of π/32 should return False."""
        from src.catalog.specs import TrackPieceSpec, ATOMIC_ANGLE_RAD
        # Deliberately off-lattice: π/8 + a clearly-non-lattice offset (> tolerance)
        off_lattice_theta = math.pi / 8 + 1e-3
        spec = TrackPieceSpec.model_validate(dict(
            piece_id="off_lattice_curve",
            kind="curve",
            manufacturer="fxbricks",
            radius_studs=64.0,
            sector_angle_rad=off_lattice_theta,
            hand="left",
            ports={
                "A": {"dx": 0.0, "dy": 0.0, "dtheta": 0.0},
                "B": {"dx": 15.0, "dy": 3.0, "dtheta": off_lattice_theta},
            },
            routes={"main": ("A", "B")},
        ))
        assert spec.on_angle_lattice is False


class TestTrackCatalogSpec:
    def _minimal(self, pieces_override=None):
        from src.catalog.specs import TrackCatalogSpec
        if pieces_override is None:
            pieces_override = [
                dict(piece_id="straight_16", kind="straight", manufacturer="lego",
                     length_studs=16.0,
                     ports={"A": {"dx": 0, "dy": 0, "dtheta": 0},
                            "B": {"dx": 16, "dy": 0, "dtheta": 0}},
                     routes={"main": ("A", "B")}),
            ]
        return TrackCatalogSpec.model_validate(dict(
            meta={"schema_version": "1.0.0"},
            pieces=pieces_override,
        ))

    def test_valid_minimal_catalog(self):
        cat = self._minimal()
        assert cat.n_types == 1
        assert cat.piece_ids == ("straight_16",)

    def test_duplicate_piece_id_rejected(self):
        with pytest.raises(ValidationError) as exc:
            self._minimal(pieces_override=[
                dict(piece_id="dup", kind="straight", manufacturer="lego",
                     length_studs=16.0,
                     ports={"A": {"dx": 0, "dy": 0, "dtheta": 0},
                            "B": {"dx": 16, "dy": 0, "dtheta": 0}},
                     routes={"main": ("A", "B")}),
                dict(piece_id="dup", kind="straight", manufacturer="lego",
                     length_studs=24.0,
                     ports={"A": {"dx": 0, "dy": 0, "dtheta": 0},
                            "B": {"dx": 24, "dy": 0, "dtheta": 0}},
                     routes={"main": ("A", "B")}),
            ])
        assert "Duplicate" in str(exc.value) or "unique" in str(exc.value).lower()

    def test_by_id_lookup(self):
        cat = self._minimal()
        assert cat.by_id["straight_16"].length_studs == 16.0

    def test_by_kind_filter(self):
        cat = self._minimal()
        straights = cat.by_kind("straight")
        assert len(straights) == 1
        assert straights[0].piece_id == "straight_16"

    def test_by_manufacturer_filter(self):
        cat = self._minimal()
        assert len(cat.by_manufacturer("lego")) == 1
        assert len(cat.by_manufacturer("4dbrix")) == 0
