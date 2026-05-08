"""Tests for ConvergenceMonitorCallback: HV, IGD, feasibility rate."""

import numpy as np
import pytest


class FakePop:
    def __init__(self, F, CV, G=None):
        self._F, self._CV, self._G = F, CV, G
    def get(self, key):
        return {"F": self._F, "CV": self._CV, "G": self._G}.get(key)
    def __len__(self):
        return len(self._F)


class FakeEvaluator:
    def __init__(self, n_eval=100):
        self.n_eval = n_eval


class FakeAlgo:
    def __init__(self, n_gen, F, CV, G=None, n_eval=100):
        self.n_gen = n_gen
        self.pop = FakePop(F, CV, G)
        self.evaluator = FakeEvaluator(n_eval)


class TestConvergenceMonitorCallback:
    def test_callback_initializes_data_keys(self):
        from src.algorithm.monitoring import ConvergenceMonitorCallback
        cb = ConvergenceMonitorCallback()
        expected_keys = {"n_gen", "n_eval", "hv", "igd", "n_feas", "feas_rate",
                         "mean_closure_x", "mean_closure_y", "mean_closure_theta"}
        assert expected_keys.issubset(cb.data.keys()), (
            f"Missing keys: {expected_keys - cb.data.keys()}"
        )
        for k in expected_keys:
            assert cb.data[k] == [], f"{k} should init as empty list, got {cb.data[k]}"

    def test_hv_filters_infeasibles_before_computing(self):
        """The +inf sentinel must never reach HV.__call__ — filter to feasible-only first."""
        from src.algorithm.monitoring import ConvergenceMonitorCallback
        cb = ConvergenceMonitorCallback(ref_point=(0.10, -0.55))

        algo = FakeAlgo(
            n_gen=5,
            F=np.array([[-0.8, -1.05], [np.inf, np.inf], [-0.6, -0.9]]),
            CV=np.array([[0.0], [5e6], [0.0]]),
        )
        cb.notify(algo)
        assert len(cb.data["hv"]) == 1
        assert np.isfinite(cb.data["hv"][0]), "HV must not be inf/nan"
        assert cb.data["hv"][0] > 0, f"HV should be positive, got {cb.data['hv'][0]}"
        assert cb.data["n_feas"][0] == 2
        assert cb.data["feas_rate"][0] == pytest.approx(2 / 3, rel=1e-6)

    def test_hv_zero_when_all_infeasible(self):
        from src.algorithm.monitoring import ConvergenceMonitorCallback
        cb = ConvergenceMonitorCallback()
        algo = FakeAlgo(
            n_gen=1,
            F=np.array([[np.inf, np.inf], [np.inf, np.inf]]),
            CV=np.array([[1e6], [2e6]]),
            n_eval=50,
        )
        cb.notify(algo)
        assert cb.data["hv"][0] == 0.0
        assert cb.data["n_feas"][0] == 0
        assert cb.data["feas_rate"][0] == 0.0

    def test_igd_self_improving_across_generations(self):
        """Without an external pareto_ref, IGD is computed against a rolling best-known front."""
        from src.algorithm.monitoring import ConvergenceMonitorCallback
        cb = ConvergenceMonitorCallback(pareto_ref=None)

        algo1 = FakeAlgo(1, np.array([[-0.5, -1.0]]), np.array([[0.0]]))
        cb.notify(algo1)
        assert np.isfinite(cb.data["igd"][0])

        algo2 = FakeAlgo(2, np.array([[-0.8, -1.05]]), np.array([[0.0]]))
        cb.notify(algo2)
        assert np.isfinite(cb.data["igd"][1])

    def test_mean_closure_populates_when_g_has_three_entries(self):
        """Pre-Phase-B, G has 6 entries — first three are legacy closure scalars.
        This de-normalization block executes when G is available and has ≥ 3 entries.
        The test just verifies the code path runs without crashing.
        """
        from src.algorithm.monitoring import ConvergenceMonitorCallback
        cb = ConvergenceMonitorCallback()
        G = np.array([
            [-0.5, -0.3, -0.8, 0.0, 0.0, 0.0],
            [-0.6, -0.4, -0.9, 0.0, 0.0, 0.0],
        ])
        algo = FakeAlgo(
            n_gen=3,
            F=np.array([[-0.5, -1.0], [-0.6, -0.9]]),
            CV=np.array([[0.0], [0.0]]),
            G=G,
        )
        cb.notify(algo)
        # Values are populated but pre-Phase-B they may not be meaningful closure;
        # we just require finiteness when G has enough columns.
        assert np.isfinite(cb.data["mean_closure_x"][0])
        assert np.isfinite(cb.data["mean_closure_y"][0])
        assert np.isfinite(cb.data["mean_closure_theta"][0])
