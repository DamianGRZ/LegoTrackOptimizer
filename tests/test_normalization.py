"""Objective-space normalization used by reporting (HV, Pareto axes, ASF pick)."""

import numpy as np
import pytest
from pymoo.decomposition.asf import ASF
from pymoo.indicators.hv import HV

from src.normalization import (
    balance_ranking,
    champion_ranking,
    compromise_index,
    first_objective_ranking,
    has_extent,
    hv_ref_point,
    ideal_nadir,
    normalize,
)


def _sample_front(n: int = 200) -> np.ndarray:
    """Deterministic F in the real objective scale: -utilization and seconds."""
    rng = np.random.default_rng(0)
    return np.column_stack([-rng.uniform(0.05, 0.62, n), rng.uniform(2.3, 15.8, n)])


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


class TestBalanceRanking:
    """The elite callbacks pick by this ranking, so it decides what survives."""

    def test_row_agrees_with_the_equal_weight_compromise(self):
        F = _sample_front(50)
        expected = compromise_index(normalize(F, *ideal_nadir(F)))
        assert balance_ranking(F)[0] == expected

    def test_orders_from_balanced_to_extreme(self):
        """Two extremes and one middle point: the middle wins, an extreme loses."""
        F = np.array([[-0.60, 15.0], [-0.05, 2.5], [-0.35, 8.0]])
        ranking = balance_ranking(F)
        assert ranking[0] == 2
        assert ranking[-1] in (0, 1)

    def test_infinite_rows_are_skipped_not_ranked(self):
        """Degenerate individuals carry the +inf sentinel; including them would
        collapse the normalization onto a single finite point."""
        F = np.array([[np.inf, np.inf], [-0.4, 9.0], [-0.2, 4.0]])
        assert set(balance_ranking(F).tolist()) == {1, 2}

    def test_no_finite_row_leaves_an_empty_ranking(self):
        assert len(balance_ranking(np.full((3, 2), np.inf))) == 0

    def test_ranks_three_objectives(self):
        """The pick must not assume two objectives — 3-objective runs use it too."""
        F = np.array([[-0.6, 15.0, -900.0], [-0.05, 2.5, -80.0], [-0.3, 8.0, -400.0]])
        assert balance_ranking(F)[0] == 2

    def test_single_row_is_its_own_champion(self):
        """pymoo's ASF keeps a (1, 1) shape for one row; the ranking must not
        trip over it."""
        assert balance_ranking(np.array([[-0.6, 1.0]])).tolist() == [0]


class TestChampionRankings:
    """``config.champion_selection`` chooses between these; they must be
    distinguishable on the same set or the knob selects nothing."""

    def test_first_objective_ranks_by_f0_alone(self):
        F = np.array([[-0.60, 15.0], [-0.05, 2.5], [-0.35, 8.0]])
        assert first_objective_ranking(F).tolist() == [0, 2, 1]

    def test_infinite_rows_are_dropped(self):
        F = np.array([[np.inf, np.inf], [-0.4, 9.0], [-0.2, 4.0]])
        assert first_objective_ranking(F).tolist() == [1, 2]
        assert len(first_objective_ranking(np.full((3, 2), np.inf))) == 0

    def test_rules_disagree_on_an_unbalanced_front(self):
        """The F0 extreme pays badly on F1: rule one still takes it, the
        balanced rule walks past it — the configured choice is observable."""
        F = np.array([[-0.60, 15.0], [-0.05, 2.5], [-0.35, 8.0]])
        assert first_objective_ranking(F)[0] == 0
        assert balance_ranking(F)[0] == 2

    def test_mapping_matches_the_config_literals(self):
        assert champion_ranking("first_objective") is first_objective_ranking
        assert champion_ranking("balanced") is balance_ranking


class TestPymooReferenceEquivalence:
    """Pin the implementation to pymoo's Getting-Started Part 3 reference.

    Each test recomputes the documented formula inline and demands a match,
    so a change in pymoo's normalization or decomposition internals surfaces
    here instead of silently shifting every reported front.
    """

    def test_normalize_matches_the_documented_formula(self):
        """Part 3: ``nF = (F - approx_ideal) / (approx_nadir - approx_ideal)``."""
        F = _sample_front()
        approx_ideal, approx_nadir = F.min(axis=0), F.max(axis=0)
        expected = (F - approx_ideal) / (approx_nadir - approx_ideal)

        assert normalize(F, *ideal_nadir(F)) == pytest.approx(expected, abs=1e-12)

    def test_compromise_matches_the_documented_asf_call(self):
        """Part 3: ``i = ASF().do(nF, 1/weights).argmin()`` — weights inverted."""
        F = _sample_front()
        nF = normalize(F, *ideal_nadir(F))

        for weights in (np.array([0.5, 0.5]), np.array([0.2, 0.8])):
            assert compromise_index(nF, weights) == ASF().do(nF, 1 / weights).argmin()

    def test_hypervolume_is_scale_free(self):
        """The zero_to_one path equals manual normalization, and rescaling an
        objective must not move the indicator — otherwise HV would be
        incomparable between runs whose objectives span different ranges."""
        F = _sample_front(40)
        ref_point = hv_ref_point(F.shape[1])

        def hv_normalized(objectives):
            ideal, nadir = ideal_nadir(objectives)
            return HV(ref_point=ref_point, norm_ref_point=False, zero_to_one=True,
                      ideal=ideal, nadir=nadir).do(objectives)

        hv = hv_normalized(F)
        manual = HV(ref_point=ref_point).do(normalize(F, *ideal_nadir(F)))

        assert hv == pytest.approx(manual)
        assert hv > 0.0, "ref point 1.1 must be dominated by the normalized front"
        assert hv == pytest.approx(hv_normalized(F * np.array([1.0, 1000.0])))
