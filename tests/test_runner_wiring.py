"""Wiring tests for runner-level configuration plumbing.

Covers the two config blocks that used to be parsed-but-ignored:
- ``TerminationConfig`` -> ``DefaultMultiObjectiveTermination`` (early stop on
  no improvement instead of always grinding the full ``n_gen``),
- the epsilon schedule's progress driver (generation fraction, decoupled from
  ``termination.perc`` whose semantics change with the new termination).
"""

import pytest
from pymoo.parallelization.starmap import StarmapParallelization
from pymoo.termination.default import DefaultMultiObjectiveTermination

from src.algorithm.runner import (
    _build_elementwise_runner,
    _build_termination,
    _epsilon_alpha,
)
from src.config import OptimizationConfig


def _config(n_gen=200, n_max_gen=1000, ftol=0.005, period=100, xtol=1e-6):
    return OptimizationConfig(
        inventory={"STRAIGHT_16": 8, "R40_CURVE": 16},
        algorithm={
            "n_gen": n_gen,
            "termination": {
                "n_max_gen": n_max_gen,
                "ftol": ftol,
                "period": period,
                "xtol": xtol,
            },
        },
    )


# =============================================================================
# Epsilon schedule (pure function)
# =============================================================================

class TestEpsilonAlpha:
    """Three-phase schedule: hold -> linear decay -> strict."""

    def test_full_epsilon_during_hold_phase(self):
        assert _epsilon_alpha(0.0, 0.2, 0.9) == 1.0
        assert _epsilon_alpha(0.19, 0.2, 0.9) == 1.0

    def test_linear_decay_midpoint(self):
        # Halfway between hold_until=0.2 and perc_eps_until=0.9 is t=0.55.
        assert _epsilon_alpha(0.55, 0.2, 0.9) == pytest.approx(0.5)

    def test_strict_phase_is_zero(self):
        assert _epsilon_alpha(0.9, 0.2, 0.9) == 0.0
        assert _epsilon_alpha(1.0, 0.2, 0.9) == 0.0

    def test_decay_endpoints(self):
        assert _epsilon_alpha(0.2, 0.2, 0.9) == pytest.approx(1.0)
        assert _epsilon_alpha(0.89, 0.2, 0.9) == pytest.approx(
            1.0 - 0.69 / 0.7, abs=1e-9,
        )


# =============================================================================
# Termination wiring
# =============================================================================

class TestBuildTermination:

    def test_returns_default_multi_objective_termination(self):
        term = _build_termination(_config())
        assert isinstance(term, DefaultMultiObjectiveTermination)

    def test_n_gen_caps_when_smaller_than_termination_max(self):
        term = _build_termination(_config(n_gen=200, n_max_gen=1000))
        assert term.max_gen.n_max_gen == 200

    def test_termination_max_caps_when_smaller_than_n_gen(self):
        term = _build_termination(_config(n_gen=500, n_max_gen=300))
        assert term.max_gen.n_max_gen == 300

    def test_ftol_reaches_objective_space_criterion(self):
        term = _build_termination(_config(ftol=0.0125))
        assert term.f.termination.tol == pytest.approx(0.0125)


# =============================================================================
# Parallel evaluation wiring (n_workers — previously an orphan config field)
# =============================================================================

class TestHeadlessMatplotlibBackend:
    """PNG-only pipeline must never touch Tk.

    With a multiprocessing Pool in the parent process, Tk figure objects
    garbage-collected from the pool's result-handler threads crash Tcl
    (``Tcl_AsyncDelete: async handler deleted by the wrong thread``).
    Importing any plotting consumer must force the non-interactive Agg
    backend.
    """

    def test_runner_forces_agg_backend(self):
        import src.algorithm.runner  # noqa: F401
        import matplotlib
        assert matplotlib.get_backend().lower() == "agg"

    def test_visualization_forces_agg_backend(self):
        import src.visualization  # noqa: F401
        import matplotlib
        assert matplotlib.get_backend().lower() == "agg"


class TestBestFrontArchive:
    """The monitor's best-front archive must not grow with duplicate F rows.

    Duplicate points are mutually non-dominating, so without deduplication
    every generation's (mostly repeated) F values accumulate forever and the
    per-generation NonDominatedSorting cost grows quadratically — the runaway
    s/gen slowdown observed in run20/run21 (Ctrl+C traceback landed inside
    _update_best_front's NDS call).
    """

    def test_archive_bounded_under_repeated_identical_fronts(self):
        import numpy as np
        from src.algorithm.monitoring import ConvergenceMonitorCallback

        cb = ConvergenceMonitorCallback()
        F = np.array([[-0.5, -1.0], [-0.6, -0.9]])
        for _ in range(50):
            cb._best_F = cb._update_best_front(F)
        assert len(cb._best_F) == 2

    def test_archive_keeps_non_dominated_union(self):
        import numpy as np
        from src.algorithm.monitoring import ConvergenceMonitorCallback

        cb = ConvergenceMonitorCallback()
        cb._best_F = cb._update_best_front(np.array([[-0.5, -1.0]]))
        cb._best_F = cb._update_best_front(np.array([[-0.6, -0.9], [-0.4, -0.5]]))
        # (-0.4, -0.5) is dominated by (-0.5, -1.0); the other two trade off.
        kept = {tuple(row) for row in cb._best_F}
        assert kept == {(-0.5, -1.0), (-0.6, -0.9)}


class TestBuildElementwiseRunner:

    def test_single_worker_runs_sequentially(self):
        runner, pool = _build_elementwise_runner(1)
        assert runner is None
        assert pool is None

    def test_multi_worker_returns_starmap_runner_and_pool(self):
        runner, pool = _build_elementwise_runner(2)
        try:
            assert isinstance(runner, StarmapParallelization)
            assert pool is not None
        finally:
            if pool is not None:
                pool.close()
                pool.join()
