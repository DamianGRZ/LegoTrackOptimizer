"""Tests for forward kinematics and layout geometry computation."""

import numpy as np
import pytest

from src.catalog import TrackCatalog
from src.geometry import (
    Layout,
    build_layout,
    compute_closure_metrics,
    compute_fk_chain,
)


class TestComputeFKChain:
    """Tests for FK chain computation."""

    def test_empty_chain(self):
        """Empty chain returns origin state."""
        fk_deltas = np.zeros((0, 3))
        states = compute_fk_chain(fk_deltas)

        assert states.shape == (1, 3)
        np.testing.assert_array_equal(states[0], [0.0, 0.0, 0.0])

    def test_single_straight(self):
        """Single straight moves x=16."""
        fk_deltas = np.array([[16.0, 0.0, 0.0]])
        states = compute_fk_chain(fk_deltas)

        assert states.shape == (2, 3)
        np.testing.assert_array_equal(states[0], [0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(states[1], [16.0, 0.0, 0.0])

    def test_two_straights(self):
        """Two straights move x=32."""
        fk_deltas = np.array([[16.0, 0.0, 0.0], [16.0, 0.0, 0.0]])
        states = compute_fk_chain(fk_deltas)

        assert states.shape == (3, 3)
        np.testing.assert_array_almost_equal(states[2], [32.0, 0.0, 0.0])

    def test_turn_left_90(self):
        """Four 22.5 deg curves = 90 deg total."""
        fk_deltas = np.array(
            [
                [15.307, 3.045, 22.5],
                [15.307, 3.045, 22.5],
                [15.307, 3.045, 22.5],
                [15.307, 3.045, 22.5],
            ]
        )
        states = compute_fk_chain(fk_deltas)

        assert states.shape == (5, 3)
        assert states[4, 2] == pytest.approx(90.0)

    def test_turn_right_90(self):
        """Four -22.5 deg curves = -90 deg total."""
        fk_deltas = np.array(
            [
                [15.307, -3.045, -22.5],
                [15.307, -3.045, -22.5],
                [15.307, -3.045, -22.5],
                [15.307, -3.045, -22.5],
            ]
        )
        states = compute_fk_chain(fk_deltas)

        assert states[4, 2] == pytest.approx(-90.0)


class TestBuildLayout:
    """Tests for Layout building from chromosome."""

    def test_build_from_chromosome(self, catalog: TrackCatalog):
        """Builds layout from valid chromosome."""
        chromosome = np.array([0, 0, 0, 0])
        layout = build_layout(chromosome, catalog)

        assert layout.n_pieces == 4
        assert layout.states.shape == (5, 3)

    def test_empty_chromosome(self, catalog: TrackCatalog):
        """Empty chromosome produces empty layout."""
        chromosome = np.array([-1, -1, -1])
        layout = build_layout(chromosome, catalog)

        assert layout.n_pieces == 0
        assert layout.states.shape == (1, 3)

    def test_mixed_valid_and_empty(self, catalog: TrackCatalog):
        """Mixed chromosome filters out empty slots."""
        chromosome = np.array([0, -1, 1, -1])
        layout = build_layout(chromosome, catalog)

        assert layout.n_pieces == 2


class TestLayout:
    """Tests for Layout properties."""

    def test_r40_circle_closure(self, catalog: TrackCatalog):
        """16 R40_LEFT pieces form closed circle."""
        chromosome = np.full(16, 2, dtype=np.int32)
        layout = build_layout(chromosome, catalog)

        assert layout.closure_error < 0.5
        assert layout.angle_error < 5.0
        assert layout.is_closed()

    def test_bounding_box(self, catalog: TrackCatalog):
        """Bounding box computed correctly."""
        chromosome = np.array([0, 0, 0, 0])
        layout = build_layout(chromosome, catalog)

        min_x, min_y, max_x, max_y = layout.bounding_box
        assert max_x - min_x == pytest.approx(64.0)

    def test_area(self, catalog: TrackCatalog):
        """Area equals bbox width * height."""
        chromosome = np.array([0, 0])
        layout = build_layout(chromosome, catalog)

        min_x, min_y, max_x, max_y = layout.bounding_box
        expected_area = (max_x - min_x) * (max_y - min_y)
        assert layout.area == pytest.approx(expected_area)

    def test_final_state(self, catalog: TrackCatalog):
        """final_state returns last row of states."""
        chromosome = np.array([0, 0])
        layout = build_layout(chromosome, catalog)

        np.testing.assert_array_equal(layout.final_state, layout.states[-1])

    def test_total_angle(self, catalog: TrackCatalog):
        """total_angle sums piece angles."""
        chromosome = np.array([2, 2, 2, 2])
        layout = build_layout(chromosome, catalog)

        assert layout.total_angle == pytest.approx(90.0, abs=0.1)


class TestMixedLayout:
    """Tests for layouts with mixed piece types."""

    def test_straight_and_curve(self, catalog: TrackCatalog):
        """Mixed pieces compute correctly."""
        chromosome = np.array([0, 2, 0])
        layout = build_layout(chromosome, catalog)

        assert layout.n_pieces == 3
        assert layout.total_angle == pytest.approx(22.5, abs=0.1)


class TestClosureMetrics:
    """Tests for closure metric computation."""

    def test_origin_is_closed(self):
        """Starting at origin has zero closure error."""
        states = np.array([[0.0, 0.0, 0.0]])
        closure, angle = compute_closure_metrics(states)

        assert closure == 0.0

    def test_displaced_not_closed(self):
        """Displaced state has closure error."""
        states = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        closure, angle = compute_closure_metrics(states)

        assert closure == pytest.approx(10.0)
