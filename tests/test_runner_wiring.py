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
from pymoo.termination.max_gen import MaximumGenerationTermination

from pydantic import ValidationError
from pymoo.operators.survival.rank_and_crowding import ConstrRankAndCrowding, RankAndCrowding
from pymoo.operators.survival.rank_and_crowding.metrics import calc_crowding_distance

from src.algorithm.runner import (
    _build_elementwise_runner,
    _build_survival,
    _build_termination,
    _epsilon_alpha,
    _pruning_crowding,
)
from src.config import OptimizationConfig

# Every shipped config names this file; these hand-built ones never resolve it
# (nothing here loads physics), but the field is required, so state it honestly.
TRAIN_CONFIG_PATH = "trains/measured_consist.yaml"


def _config(n_gen=200, n_max_gen=1000, ftol=0.005, period=100, xtol=1e-6):
    return OptimizationConfig(
        train_config_path=TRAIN_CONFIG_PATH,
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
    """period > 0: improvement-window early stop. period = 0 (default): run
    the full generation budget — no early stop."""

    def test_period_zero_runs_full_budget_no_early_stop(self):
        term = _build_termination(_config(period=0))
        assert isinstance(term, MaximumGenerationTermination)
        assert not isinstance(term, DefaultMultiObjectiveTermination)
        assert term.n_max_gen == 200

    def test_period_zero_is_the_config_default(self):
        cfg = OptimizationConfig(train_config_path=TRAIN_CONFIG_PATH, inventory={"STRAIGHT_16": 8})
        assert cfg.algorithm.termination.period == 0

    def test_period_zero_still_capped_by_termination_n_max_gen(self):
        term = _build_termination(_config(n_gen=500, n_max_gen=300, period=0))
        assert term.n_max_gen == 300

    def test_returns_default_multi_objective_termination(self):
        term = _build_termination(_config())
        assert isinstance(term, DefaultMultiObjectiveTermination)

    def test_eval_cap_never_binds(self):
        """DefaultTermination's hidden n_max_evals=100000 default (generation
        100 at pop 1000) must be disarmed: this project budgets by generations."""
        term = _build_termination(_config())
        assert term.max_evals.n_max_evals == float("inf")

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


# =============================================================================
# Survival wiring — which crowding metric sorts the splitting front
# =============================================================================

def _survival_config(crowding_func="cd", constr_survival=True):
    return OptimizationConfig(
        train_config_path=TRAIN_CONFIG_PATH,
        inventory={"STRAIGHT_16": 8, "R40_CURVE": 16},
        algorithm={"crowding_func": crowding_func,
                   "components": {"constr_survival": constr_survival}},
    )


class TestCrowdingFuncConfig:
    """An unknown metric must fail at load time — pymoo raises KeyError mid-run."""

    def test_defaults_to_cd(self):
        config = OptimizationConfig(train_config_path=TRAIN_CONFIG_PATH,
                                    inventory={"STRAIGHT_16": 8})
        assert config.algorithm.crowding_func == "cd"

    def test_unknown_metric_rejected(self):
        # Every other field must be valid, or this passes on the wrong error.
        with pytest.raises(ValidationError, match="crowding_func"):
            OptimizationConfig(train_config_path=TRAIN_CONFIG_PATH,
                               inventory={"STRAIGHT_16": 8},
                               algorithm={"crowding_func": "nope"})


class TestMisspelledKeysAreRejected:
    """A dropped key leaves its field on the default, so the run would not be the
    one the file describes. Every nesting level must refuse unknown names."""

    BASE = {"train_config_path": TRAIN_CONFIG_PATH, "inventory": {"STRAIGHT_16": 8}}

    @pytest.mark.parametrize("typo, patch", [
        ("train_confg_path", {"train_confg_path": "x.yaml"}),
        ("bogus", {"boundary": {"min_x": -1.0, "bogus": 2}}),
        ("pop_sizee", {"algorithm": {"pop_sizee": 10}}),
        ("perido", {"algorithm": {"termination": {"perido": 5}}}),
        ("repare", {"algorithm": {"components": {"repare": False}}}),
    ])
    def test_unknown_key_is_rejected(self, typo, patch):
        with pytest.raises(ValidationError, match=typo):
            OptimizationConfig.model_validate({**self.BASE, **patch})


class TestBuildSurvival:
    """``constr_survival`` picks the operator, ``crowding_func`` its metric."""

    def test_constr_rank_and_crowding_by_default(self):
        assert isinstance(_build_survival(_survival_config()), ConstrRankAndCrowding)

    def test_falls_back_to_stock_rank_and_crowding(self):
        survival = _build_survival(_survival_config(constr_survival=False))
        assert isinstance(survival, RankAndCrowding)
        assert not isinstance(survival, ConstrRankAndCrowding)

    def test_cd_keeps_duplicates_in_the_comparison(self):
        metric = _build_survival(_survival_config("cd")).ranking.crowding_func
        assert metric.function is calc_crowding_distance
        assert metric.filter_out_duplicates is False

    def test_pcd_reaches_the_operator_through_the_guard(self):
        metric = _build_survival(_survival_config("pcd")).ranking.crowding_func
        assert metric.function is _pruning_crowding
        assert metric.filter_out_duplicates is True

    def test_metric_survives_the_stock_fallback_arm(self):
        metric = _build_survival(_survival_config("pcd", False)).crowding_func
        assert metric.function is _pruning_crowding


class TestPruningCrowdingGuard:
    """Survival measures ``n_remove`` on the full front while the metric sees only
    the distinct rows, so on this problem it routinely exceeds what the compiled
    kernel tolerates. Unguarded, that corrupts memory: these tests fail by killing
    the interpreter rather than by asserting.
    """

    @staticmethod
    def _front(n_points):
        import numpy as np
        util = np.linspace(0.05, 0.95, n_points)
        return np.column_stack([-util, 200.0 + 300.0 * util])

    def test_clamps_n_remove_to_the_kernel_bound(self):
        import numpy as np
        F = self._front(40)
        bound = _pruning_crowding(F, n_remove=len(F) - F.shape[1] - 1)
        assert np.array_equal(_pruning_crowding(F, n_remove=len(F)), bound)
        assert np.array_equal(_pruning_crowding(F, n_remove=10 ** 6), bound)

    def test_negative_n_remove_floors_at_zero(self):
        import numpy as np
        F = self._front(40)
        assert np.array_equal(_pruning_crowding(F, n_remove=-5),
                              _pruning_crowding(F, n_remove=0))

    def test_clone_heavy_front_selects_without_crashing(self):
        import numpy as np
        from pymoo.core.population import Population
        from pymoo.core.problem import Problem

        class TwoObjective(Problem):
            def __init__(self):
                super().__init__(n_var=1, n_obj=2, n_ieq_constr=0, xl=0, xu=1)

        # Measured regime: a merged pool of 2000 holding ~72 distinct F rows.
        n_pool, n_survive = 2000, 1000
        distinct = self._front(72)
        rng = np.random.default_rng(0)
        F = distinct[rng.integers(0, len(distinct), size=n_pool)]
        pop = Population.new(X=np.zeros((n_pool, 1)), F=F)

        survivors = _build_survival(_survival_config("pcd")).do(
            TwoObjective(), pop, n_survive=n_survive,
        )

        assert len(survivors) == n_survive
        crowding = np.array([ind.get("crowding") for ind in survivors], dtype=float)
        assert not np.isnan(crowding).any()
