"""Objective-space normalization used by reporting (HV, Pareto axes, ASF pick)."""

import numpy as np
import pytest

from src.normalization import (
    compromise_index,
    has_extent,
    ideal_nadir,
    normalize,
)


class TestIdealNadir:
    def test_per_objective_min_and_max(self):
        F = np.array([[-0.8, -0.9], [-0.6, -1.05], [-0.7, -1.0]])
        ideal, nadir = ideal_nadir(F)
        assert ideal == pytest.approx([-0.8, -1.05])
        assert nadir == pytest.approx([-0.6, -0.9])


class TestHasExtent:
    def test_true_for_a_spread_front(self):
        assert has_extent(np.array([-0.8, -1.05]), np.array([-0.6, -0.9]))

    def test_false_when_one_objective_is_flat(self):
        assert not has_extent(np.array([-0.8, -1.0]), np.array([-0.6, -1.0]))

    def test_false_for_non_finite_bounds(self):
        assert not has_extent(np.array([-0.8, -np.inf]), np.array([-0.6, -0.9]))


class TestNormalize:
    def test_ideal_maps_to_zero_and_nadir_to_one(self):
        """0 is best and 1 worst for every objective, whatever sign it carries:
        a negated maximization target and a plain minimized one both land the
        same way."""
        ideal, nadir = np.array([-0.8, 2.0]), np.array([-0.6, 10.0])
        nF = normalize(np.array([[-0.8, 2.0], [-0.6, 10.0]]), ideal, nadir)
        assert nF[0] == pytest.approx([0.0, 0.0])
        assert nF[1] == pytest.approx([1.0, 1.0])

    def test_points_beyond_the_nadir_stay_outside_the_unit_box(self):
        ideal, nadir = np.array([0.0, 0.0]), np.array([1.0, 1.0])
        nF = normalize(np.array([[2.0, -1.0]]), ideal, nadir)
        assert nF[0] == pytest.approx([2.0, -1.0])

    def test_flat_objective_gets_a_unit_span_instead_of_nan(self):
        """pymoo's own fallback for a zero range is an unscaled shift; a
        defined mapping keeps plots drawable — has_extent is what guards the
        cases where the degenerate axis would make the number meaningless."""
        nF = normalize(np.array([[-0.7, -1.0], [-0.5, -1.0]]),
                       np.array([-0.7, -1.0]), np.array([-0.5, -1.0]))
        assert np.isfinite(nF).all()
        assert nF[:, 1] == pytest.approx([0.0, 0.0])


class TestCompromiseIndex:
    def test_picks_the_balanced_point_over_the_extremes(self):
        nF = np.array([[0.0, 1.0], [1.0, 0.0], [0.4, 0.35]])
        assert compromise_index(nF) == 2

    def test_weights_shift_the_pick_toward_the_favoured_objective(self):
        nF = np.array([[0.05, 0.9], [0.9, 0.05]])
        assert compromise_index(nF, np.array([0.9, 0.1])) == 0
        assert compromise_index(nF, np.array([0.1, 0.9])) == 1
