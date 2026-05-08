"""Tests for track catalog loading and vectorized access."""

import numpy as np
import pytest

from src.catalog import FKDeltas, TrackCatalog, TrackPiece
from src.types import PieceClass


class TestTrackCatalog:
    """Tests for TrackCatalog loading and basic operations."""

    def test_catalog_loads(self, catalog: TrackCatalog):
        """Catalog loads and has pieces."""
        assert catalog.n_pieces > 0

    def test_fk_table_shape(self, catalog: TrackCatalog):
        """FK table has correct shape (n_pieces, 3)."""
        assert catalog.fk_table.ndim == 2
        assert catalog.fk_table.shape[1] == 3

    def test_piece_lookup_by_index(self, catalog: TrackCatalog):
        """Can lookup pieces by index."""
        # STRAIGHT_16 should be at index 0
        piece = catalog[0]
        assert piece is not None
        assert piece.id == "STRAIGHT_16"

    def test_piece_lookup_by_id(self, catalog: TrackCatalog):
        """ID to index mapping works."""
        assert "STRAIGHT_16" in catalog.id_to_index
        assert catalog.id_to_index["STRAIGHT_16"] == 0

    def test_straights_have_zero_dtheta(self, catalog: TrackCatalog):
        """Straight pieces have dtheta=0."""
        straight_16 = catalog[0]
        straight_24 = catalog[1]

        assert straight_16 is not None
        assert straight_24 is not None
        assert straight_16.fk.dtheta == 0.0
        assert straight_24.fk.dtheta == 0.0

    def test_curves_have_nonzero_dtheta(self, catalog: TrackCatalog):
        """Curves have correct angle deltas."""
        r40_left = catalog[2]
        r40_right = catalog[3]

        assert r40_left is not None
        assert r40_right is not None
        assert r40_left.fk.dtheta == pytest.approx(22.5, abs=0.1)
        assert r40_right.fk.dtheta == pytest.approx(-22.5, abs=0.1)

    def test_all_pieces_have_index(self, catalog: TrackCatalog):
        """All registered pieces have valid indices."""
        for piece_id, idx in catalog.id_to_index.items():
            piece = catalog[idx]
            assert piece is not None, f"No piece at index {idx} for {piece_id}"
            assert piece.id == piece_id


class TestVectorizedLookup:
    """Tests for vectorized piece property access."""

    def test_get_fk_valid_indices(self, catalog: TrackCatalog):
        """Vectorized FK retrieval works."""
        indices = np.array([0, 1, 2, 3])  # STRAIGHT_16, STRAIGHT_24, R40_LEFT, R40_RIGHT
        fk = catalog.get_fk(indices)

        assert fk.shape == (4, 3)
        assert fk[0, 0] == pytest.approx(16.0)  # STRAIGHT_16 dx
        assert fk[1, 0] == pytest.approx(24.0)  # STRAIGHT_24 dx

    def test_get_fk_handles_negative_indices(self, catalog: TrackCatalog):
        """Negative indices (empty slots) return zeros."""
        indices = np.array([-1, 0, -1])
        fk = catalog.get_fk(indices)

        assert fk.shape == (3, 3)
        np.testing.assert_array_equal(fk[0], [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(fk[2], [0.0, 0.0, 0.0])
        assert fk[1, 0] == pytest.approx(16.0)  # Valid index still works

    def test_get_radii(self, catalog: TrackCatalog):
        """Radius lookup works."""
        indices = np.array([0, 2])  # STRAIGHT_16, R40_LEFT
        radii = catalog.get_radii(indices)

        assert radii[0] == np.inf  # Straight has infinite radius
        assert radii[1] == pytest.approx(320.0)  # R40 = 320mm

    def test_get_speed_limits(self, catalog: TrackCatalog):
        """Speed limit lookup works."""
        indices = np.array([0, 2])  # STRAIGHT_16, R40_LEFT
        speeds = catalog.get_speed_limits(indices)

        assert speeds[0] == pytest.approx(1.57)  # Straight = motor top speed
        assert speeds[1] == pytest.approx(0.97)  # R40 curve limit

    def test_get_arc_lengths(self, catalog: TrackCatalog):
        """Arc length lookup works."""
        indices = np.array([0, 1])  # STRAIGHT_16, STRAIGHT_24
        lengths = catalog.get_arc_lengths(indices)

        assert lengths[0] == pytest.approx(16.0)
        assert lengths[1] == pytest.approx(24.0)


class TestFKDeltas:
    """Tests for FKDeltas dataclass."""

    def test_to_array(self):
        """FKDeltas converts to numpy array."""
        fk = FKDeltas(dx=16.0, dy=0.0, dtheta=0.0)
        arr = fk.to_array()

        assert arr.shape == (3,)
        np.testing.assert_array_equal(arr, [16.0, 0.0, 0.0])


class TestTrackPiece:
    """Tests for TrackPiece dataclass."""

    def test_arc_length_straight(self, catalog: TrackCatalog):
        """Straight arc_length equals length."""
        straight = catalog[0]
        assert straight is not None
        assert straight.arc_length == straight.length

    def test_arc_length_curve(self, catalog: TrackCatalog):
        """Curve arc_length = radius * angle_rad."""
        curve = catalog[2]  # R40_LEFT
        assert curve is not None
        assert curve.is_curve
        # R40: radius=40 studs, angle=22.5 deg
        expected = 40.0 * np.radians(22.5)
        assert curve.arc_length == pytest.approx(expected, rel=0.01)


class TestTopologyAndRoutes:
    """Tests for topology metadata and multi-route pieces."""

    def test_simple_piece_has_topology(self, catalog: TrackCatalog):
        """Simple pieces have topology metadata."""
        topo = catalog.get_topology(0)  # STRAIGHT_16
        assert topo is not None
        assert topo.piece_class == PieceClass.SIMPLE_2PORT
        assert topo.num_ports == 2

    def test_switch_has_multiple_routes(self, catalog: TrackCatalog):
        """Switch pieces have multiple routes."""
        topo = catalog.get_topology(5)  # R40_SWITCH_LEFT_IN
        assert topo is not None
        assert len(topo.routes) >= 2  # Straight and diverge

    def test_crossing_has_multiple_routes(self, catalog: TrackCatalog):
        """Crossing pieces have multiple routes."""
        topo = catalog.get_topology(4)  # CROSS_90
        assert topo is not None
        assert topo.piece_class == PieceClass.CROSSING_4PORT
        assert len(topo.routes) >= 2  # West-east and south-north

    def test_double_crossover_has_four_routes(self, catalog: TrackCatalog):
        """DOUBLE_CROSSOVER has 4 routes."""
        topo = catalog.get_topology(9)  # DOUBLE_CROSSOVER
        assert topo is not None
        assert len(topo.routes) == 4

    def test_get_fk_route(self, catalog: TrackCatalog):
        """Can get FK for specific route."""
        # CROSS_90 route 0 should be west-east (dx=16)
        fk = catalog.get_fk_route(4, 0)
        assert fk[0] == pytest.approx(16.0)

    def test_get_fk_with_routes(self, catalog: TrackCatalog):
        """Can get FK with route selection for each piece."""
        piece_indices = np.array([0, 4, 4])  # STRAIGHT_16, CROSS_90, CROSS_90
        route_indices = np.array([0, 0, 1])  # Default routes, then alternate

        fk = catalog.get_fk_with_routes(piece_indices, route_indices)

        assert fk.shape == (3, 3)
        assert fk[0, 0] == pytest.approx(16.0)  # STRAIGHT_16


class TestPieceClassification:
    """Tests for piece classification methods."""

    def test_classify_pieces(self, catalog: TrackCatalog):
        """Can classify all pieces by type."""
        classified = catalog.classify_pieces()

        assert PieceClass.SIMPLE_2PORT in classified
        assert len(classified[PieceClass.SIMPLE_2PORT]) >= 4  # Straights + curves

    def test_get_simple_pieces(self, catalog: TrackCatalog):
        """Can get simple (2-port) pieces."""
        simple = catalog.get_simple_pieces()
        assert 0 in simple  # STRAIGHT_16
        assert 2 in simple  # R40_LEFT

    def test_get_switch_pieces(self, catalog: TrackCatalog):
        """Can get switch pieces."""
        switches = catalog.get_switch_pieces()
        assert 5 in switches  # R40_SWITCH_LEFT_IN
        assert 6 in switches  # R40_SWITCH_LEFT_OUT


class TestStudMm:
    """TrackCatalog exposes stud_mm read from track_pieces.yaml metadata."""

    def test_catalog_has_stud_mm_attribute(self, catalog):
        assert hasattr(catalog, "stud_mm")

    def test_stud_mm_value_from_yaml(self, catalog):
        # data/track_pieces.yaml -> metadata.stud_mm = 8.0
        assert catalog.stud_mm == pytest.approx(8.0)


class TestV1Deprecation:
    def test_loading_v1_yaml_warns(self):
        import warnings
        from src.catalog import TrackCatalog
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            TrackCatalog.load("data/track_pieces.yaml")
            deprecations = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert deprecations, "loading v1 YAML should emit DeprecationWarning"
            assert "v1" in str(deprecations[0].message).lower() or \
                   "legacy" in str(deprecations[0].message).lower()
