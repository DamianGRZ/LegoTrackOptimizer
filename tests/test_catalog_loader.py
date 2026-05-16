"""Tests for the ruamel.yaml + Pydantic catalog loader."""

from pathlib import Path
import pytest

FIXTURE_TINY = Path(__file__).parent / "fixtures" / "catalog_tiny.yaml"


class TestHappyPath:
    def test_loads_minimal_catalog(self):
        from src.catalog.loader import load_catalog_spec
        cat = load_catalog_spec(FIXTURE_TINY)
        assert cat.n_types == 2
        assert "lego_straight_16" in cat.by_id
        assert "lego_curve_R40_left" in cat.by_id

    def test_returns_track_catalog_spec_instance(self):
        from src.catalog.loader import load_catalog_spec
        from src.catalog.specs import TrackCatalogSpec
        cat = load_catalog_spec(FIXTURE_TINY)
        assert isinstance(cat, TrackCatalogSpec)


FIXTURES = Path(__file__).parent / "fixtures"


class TestErrorUX:
    def test_extra_field_reports_file_line_field(self):
        from src.catalog.loader import load_catalog_spec, CatalogLoadError
        with pytest.raises(CatalogLoadError) as exc:
            load_catalog_spec(FIXTURES / "catalog_bad_extra_field.yaml")
        msg = str(exc.value)
        assert "catalog_bad_extra_field.yaml:" in msg
        assert "pieces.0" in msg
        assert "extra_forbidden" in msg or "Extra" in msg

    def test_missing_port_a(self):
        from src.catalog.loader import load_catalog_spec, CatalogLoadError
        with pytest.raises(CatalogLoadError) as exc:
            load_catalog_spec(FIXTURES / "catalog_bad_missing_port_a.yaml")
        assert "Port 'A'" in str(exc.value)
        assert "catalog_bad_missing_port_a.yaml:" in str(exc.value)

    def test_duplicate_piece_id(self):
        from src.catalog.loader import load_catalog_spec, CatalogLoadError
        with pytest.raises(CatalogLoadError) as exc:
            load_catalog_spec(FIXTURES / "catalog_bad_duplicate_id.yaml")
        assert "Duplicate" in str(exc.value)
        assert "dup" in str(exc.value)

    def test_route_undefined_port(self):
        from src.catalog.loader import load_catalog_spec, CatalogLoadError
        with pytest.raises(CatalogLoadError) as exc:
            load_catalog_spec(FIXTURES / "catalog_bad_route_undefined_port.yaml")
        assert "undefined" in str(exc.value).lower() or "Z" in str(exc.value)

    def test_major_version_rejected(self):
        from src.catalog.loader import load_catalog_spec
        from src.catalog.specs import SchemaVersionError
        with pytest.raises(SchemaVersionError):
            load_catalog_spec(FIXTURES / "catalog_bad_major_version.yaml")


class TestV2YamlStraights:
    def test_straight_16_matches_legacy_fk(self):
        from src.catalog.loader import load_catalog_spec
        cat = load_catalog_spec(Path("data") / "track_pieces_v2.yaml")
        straight = cat.by_id["STRAIGHT_16"]
        assert straight.kind == "straight"
        assert straight.length_studs == 16.0
        assert straight.ports["B"].dx == 16.0
        assert straight.ports["B"].dy == 0.0
        assert straight.ports["B"].dtheta == 0.0


class TestV2YamlCurves:
    def test_r40_left_matches_legacy_fk(self):
        from src.catalog.loader import load_catalog_spec
        cat = load_catalog_spec(Path("data") / "track_pieces_v2.yaml")
        curve = cat.by_id["R40_CURVE"]
        assert curve.kind == "curve"
        assert curve.radius_studs == 40.0
        assert abs(curve.sector_angle_rad - 0.39269908) < 1e-7
        b = curve.ports["B"]
        assert abs(b.dx - 15.307) < 1e-3
        assert abs(b.dy - 3.045) < 1e-3
        assert abs(b.dtheta - 0.39269908) < 1e-7

    def test_r40_right_has_negative_dy_and_dtheta(self):
        from src.catalog.loader import load_catalog_spec
        cat = load_catalog_spec(Path("data") / "track_pieces_v2.yaml")
        curve = cat.by_id["R40_CURVE"]
        b = curve.ports["B"]
        assert b.dy < 0
        assert b.dtheta < 0


class TestV2YamlSwitches:
    @pytest.mark.parametrize("piece_id,port_c_dx,port_c_dy,port_c_dtheta", [
        # Post-refactor: 2 switch types, port C at canonical (dx=32.75, dy=±13)
        ("R40_SWITCH_LEFT",  32.75,  13.0,  0.39269908),
        ("R40_SWITCH_RIGHT", 32.75, -13.0, -0.39269908),
    ])
    def test_switch_port_c_matches_canonical_geometry(self, piece_id, port_c_dx, port_c_dy, port_c_dtheta):
        from src.catalog.loader import load_catalog_spec
        cat = load_catalog_spec(Path("data") / "track_pieces_v2.yaml")
        sw = cat.by_id[piece_id]
        assert sw.kind == "switch"
        assert sw.body_length_studs == 32.0
        c = sw.ports["C"]
        assert abs(c.dx - port_c_dx) < 1e-3
        assert abs(c.dy - port_c_dy) < 1e-3
        assert abs(c.dtheta - port_c_dtheta) < 1e-7
        assert set(sw.routes.keys()) == {"through", "diverging"}


class TestV2YamlCrossings:
    def test_cross_90_has_4_ports_and_2_routes(self):
        from src.catalog.loader import load_catalog_spec
        cat = load_catalog_spec(Path("data") / "track_pieces_v2.yaml")
        x = cat.by_id["CROSS_90"]
        assert x.kind == "crossing"
        assert set(x.ports.keys()) == {"A", "B", "C", "D"}
        assert set(x.routes.keys()) == {"horizontal", "vertical"}

    def test_double_crossover_has_4_routes(self):
        from src.catalog.loader import load_catalog_spec
        cat = load_catalog_spec(Path("data") / "track_pieces_v2.yaml")
        x = cat.by_id["DOUBLE_CROSSOVER"]
        assert x.kind == "crossing"
        assert x.length_studs == 48.0
        assert set(x.routes.keys()) == {"track1_through", "track2_through",
                                         "cross_1_to_2", "cross_2_to_1"}

    def test_v2_catalog_has_all_8_pieces(self):
        """Post-refactor: 8 piece types (4 switch entries collapsed to 2)."""
        from src.catalog.loader import load_catalog_spec
        cat = load_catalog_spec(Path("data") / "track_pieces_v2.yaml")
        expected = {
            "STRAIGHT_16", "STRAIGHT_24",
            "R40_CURVE", "R40_CURVE",
            "R40_SWITCH_LEFT", "R40_SWITCH_RIGHT",
            "CROSS_90", "DOUBLE_CROSSOVER",
        }
        assert set(cat.by_id.keys()) == expected
