"""Epsilon_0 calibration and recovery ratchet of LegoAdaptiveEpsilon."""

import types

import numpy as np
import pytest
from pymoo.algorithms.moo.nsga2 import NSGA2

from src.algorithm.runner import DEGENERATE_CV_FLOOR, LegoAdaptiveEpsilon


def _handler(**kwargs):
    return LegoAdaptiveEpsilon(NSGA2(pop_size=10), n_ieq_constr=12, **kwargs)


def _epsilon0(cv):
    return _handler()._calibrate_epsilon0(np.asarray(cv, dtype=float))


class TestEpsilon0Calibration:
    def test_degenerate_sentinels_are_excluded(self):
        cv = [0.2] * 50 + [1.5] * 30 + [4e6] * 20
        # order statistic (theta=0.2 of 80 real values) = 1.5, floor p10 = 0.2
        assert _epsilon0(cv) == pytest.approx(1.5)

    def test_cap_derives_from_soft_constraint_count(self):
        cv = [50.0] * 100  # real (below sentinel floor) but far above repairable
        h = _handler()
        assert h._calibrate_epsilon0(np.asarray(cv)) == pytest.approx(float(h.n_soft))

    def test_zero_when_population_fully_feasible(self):
        assert _epsilon0([0.0] * 100) == 0.0

    def test_zero_when_only_feasibles_and_degenerates(self):
        assert _epsilon0([0.0] * 80 + [5e6] * 20) == 0.0

    def test_floor_shelters_nearest_infeasible_decile(self):
        # 90% feasible: the order statistic lands on a feasible (CV=0), but the
        # p10-of-infeasibles floor keeps a nonzero band open.
        cv = [0.0] * 90 + [0.4] * 10
        assert _epsilon0(cv) == pytest.approx(0.4)

    def test_sentinel_floor_is_far_above_real_cvs(self):
        assert DEGENERATE_CV_FLOOR > 1e3


class TestRecoveryRatchet:
    def _collapse(self, n_feas):
        return np.array([0.0] * n_feas + [0.5] * (100 - n_feas))

    def test_halves_on_collapse_with_cooldown(self):
        h = _handler(ratchet_cooldown=5)
        h.max_cv = 2.0
        h._ratchet_check(1.0, self._collapse(90), gen=10)
        assert h.max_cv == 2.0  # healthy: sets the peak, no ratchet
        h._ratchet_check(1.0, self._collapse(5), gen=20)
        assert h.max_cv == 1.0
        h._ratchet_check(1.0, self._collapse(5), gen=22)
        assert h.max_cv == 1.0  # inside cooldown window
        h._ratchet_check(1.0, self._collapse(5), gen=30)
        assert h.max_cv == 0.5

    def test_never_widens_after_recovery(self):
        h = _handler()
        h.max_cv = 2.0
        h._ratchet_check(1.0, self._collapse(90), gen=10)
        h._ratchet_check(1.0, self._collapse(5), gen=20)
        h._ratchet_check(1.0, self._collapse(95), gen=40)
        assert h.max_cv == 1.0

    def test_inactive_in_strict_phase(self):
        h = _handler()
        h.max_cv = 2.0
        h._ratchet_check(1.0, self._collapse(90), gen=10)
        h._ratchet_check(0.0, self._collapse(5), gen=20)
        assert h.max_cv == 2.0


class TestScheduleTerminal:
    def test_strict_phase_zero_regardless_of_epsilon0(self):
        h = _handler(hold_until=0.2, perc_eps_until=0.9)
        h.max_cv = 3.0
        h.termination = types.SimpleNamespace(perc=0.95)
        cfg = {}
        h._adapt_constraint_handling(cfg)
        assert cfg["cv_eps"] == 0.0
        assert h.last_cv_eps == 0.0

    def test_hold_phase_publishes_live_epsilon(self):
        h = _handler(hold_until=0.2, perc_eps_until=0.9)
        h.max_cv = 3.0
        h.termination = types.SimpleNamespace(perc=0.0)
        cfg = {}
        h._adapt_constraint_handling(cfg)
        assert cfg["cv_eps"] == 3.0
        assert h.last_cv_eps == 3.0