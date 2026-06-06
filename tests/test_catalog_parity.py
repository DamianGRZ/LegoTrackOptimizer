"""Parity tests: v1 vs v2 catalogs produce bit-identical numpy tables."""

from pathlib import Path

import numpy as np
import pytest

from src.catalog import TrackCatalog


V1_PATH = Path("data") / "track_pieces.yaml"
V2_PATH = Path("data") / "track_pieces_v2.yaml"

# The v1 catalog was removed (v1 deprecated; only track_pieces_v2.yaml ships).
# v1<->v2 parity is obsolete, so skip the whole module when v1 is absent.
pytestmark = pytest.mark.skipif(
    not V1_PATH.exists(),
    reason="v1 catalog (data/track_pieces.yaml) removed; v1<->v2 parity obsolete",
)


@pytest.fixture
def v1_catalog():
    return TrackCatalog.load(V1_PATH)


@pytest.fixture
def v2_catalog():
    return TrackCatalog.load(V2_PATH)


class TestFKTableParity:
    def test_both_catalogs_have_same_piece_count(self, v1_catalog, v2_catalog):
        assert v1_catalog.n_pieces == v2_catalog.n_pieces

    def test_fk_table_shape_matches(self, v1_catalog, v2_catalog):
        assert v1_catalog.fk_table.shape == v2_catalog.fk_table.shape

    def test_radius_table_matches(self, v1_catalog, v2_catalog):
        np.testing.assert_array_almost_equal(
            v1_catalog.radius_table, v2_catalog.radius_table, decimal=3)

    def test_speed_table_matches(self, v1_catalog, v2_catalog):
        np.testing.assert_array_almost_equal(
            v1_catalog.speed_table, v2_catalog.speed_table, decimal=3)

    def test_id_to_index_matches(self, v1_catalog, v2_catalog):
        assert v1_catalog._id_to_index == v2_catalog._id_to_index

    @pytest.mark.parametrize("piece_id", [
        "STRAIGHT_16", "STRAIGHT_24",
        "R40_CURVE", "R40_CURVE",
        "CROSS_90", "DOUBLE_CROSSOVER",
        "R40_SWITCH_LEFT_IN", "R40_SWITCH_LEFT_OUT",
        "R40_SWITCH_RIGHT_IN", "R40_SWITCH_RIGHT_OUT",
    ])
    def test_per_piece_fk_matches(self, v1_catalog, v2_catalog, piece_id):
        idx1 = v1_catalog._id_to_index[piece_id]
        idx2 = v2_catalog._id_to_index[piece_id]
        np.testing.assert_array_almost_equal(
            v1_catalog.fk_table[idx1], v2_catalog.fk_table[idx2], decimal=3)
