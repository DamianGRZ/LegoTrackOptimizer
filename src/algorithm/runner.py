"""NSGA-II runner for the track optimization problem."""

from __future__ import annotations

import copy
import logging
import traceback
from multiprocessing import Pool
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.constraints.eps import AdaptiveEpsilonConstraintHandling
from pymoo.core.callback import Callback
from pymoo.operators.survival.rank_and_crowding import ConstrRankAndCrowding
from pymoo.optimize import minimize
from pymoo.parallelization.starmap import StarmapParallelization
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.termination.max_gen import MaximumGenerationTermination

from src.algorithm.monitoring import ConvergenceMonitorCallback
from src.catalog import TrackCatalog
from src.config import OptimizationConfig
from src.decoder import DecoderConfig, decode_chromosome
from src.encoding import (
    chromosome_csv_header,
    compute_dimensions,
    get_active_cross_junctions,
    get_active_double_crossovers,
    get_active_junctions,
)
from src.normalization import compromise_index, has_extent, ideal_nadir, normalize
from src.operators import PartitionedCrossover, PartitionedMutation
from src.problem import DEGENERATE_G, TrackOptimizationProblem
from src.repair import TrackRepairPipeline
from src.run_info import count_pieces
from src.sampling import IntegerSampling
from src.visualization import plot_layout, plot_multi_path_layout, plot_pareto_front

# Headless pipeline: PNGs only. The interactive Tk backend crashes when its
# objects are garbage-collected from a multiprocessing.Pool result-handler
# thread (Tcl_AsyncDelete: async handler deleted by the wrong thread).
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402


def log_piece_usage(layout, inventory: dict, catalog: TrackCatalog,
                    logger: logging.Logger) -> None:
    piece_counts = count_pieces(layout)
    logger.info("Piece usage:")
    for piece_id, count in sorted(inventory.items()):
        idx = catalog.id_to_index.get(piece_id)
        if idx is not None:
            used = piece_counts.get(idx, 0)
            logger.info(f"  {piece_id}: {used}/{count}")


def log_front_scale_and_compromise(F: np.ndarray, feasible_indices: np.ndarray,
                                   logger: logging.Logger) -> None:
    """Log the raw span of both objectives and the balanced ASF pick.

    The objectives sit on different scales, so the balanced solution is
    chosen in normalized space (pymoo's compromise programming). The raw
    spans are logged alongside it because the normalized axes of the Pareto
    plot carry no units of their own.
    """
    if len(feasible_indices) == 0:
        return

    F_feas = F[feasible_indices]
    ideal, nadir = ideal_nadir(F_feas)
    logger.info(f"Objective scale (raw, minimized): f0 [{ideal[0]:.4g}, {nadir[0]:.4g}], "
                f"f1 [{ideal[1]:.4g}, {nadir[1]:.4g}]")

    if not has_extent(ideal, nadir):
        logger.info("Feasible front is a single point - no compromise to pick")
        return

    idx = int(feasible_indices[compromise_index(normalize(F_feas, ideal, nadir))])
    logger.info(f"Compromise (equal-weight ASF): individual {idx}, "
                f"score={-F[idx, 0]:.1%}, speed={-F[idx, 1]:.2f} m/s")


# =============================================================================
# Progress Callback
# =============================================================================

class ProgressCallback(Callback):
    """Log NSGA-II progress every N generations."""

    def __init__(self, every_n_gen: int = 10, total_inventory: int = 0):
        super().__init__()
        self.every_n_gen = every_n_gen
        self.total_inventory = max(1, total_inventory)

    def notify(self, algorithm):
        gen = algorithm.n_gen
        if gen % self.every_n_gen != 0:
            return

        logger = logging.getLogger(__name__)
        pop = algorithm.pop
        F = pop.get("F")
        G = pop.get("G")

        if F is None:
            return

        utils = -F[:, 0]
        speeds = -F[:, 1]
        feasible_mask = np.all(G <= 0, axis=1) if G is not None else np.ones(len(F), dtype=bool)
        n_feasible = int(np.sum(feasible_mask))

        feas_util = float(np.max(utils[feasible_mask])) if n_feasible else 0.0
        feas_speed = float(np.max(speeds[feasible_mask])) if n_feasible else 0.0
        infeas_util = float(np.max(utils[~feasible_mask])) if n_feasible < len(F) else 0.0

        logger.info(
            f"Gen {gen:4d} | "
            f"best_feas={feas_util:.1%} ({int(feas_util * self.total_inventory)} pcs) "
            f"{feas_speed:.2f} m/s | "
            f"best_infeas={infeas_util:.1%} ({int(infeas_util * self.total_inventory)} pcs) | "
            f"feasible={n_feasible}/{len(pop)}"
        )


# =============================================================================
# Feasible-Elite Callback
# =============================================================================

class FeasibleEliteCallback(Callback):
    """Preserve the best-utilization feasible individual across generations.

    Once the adaptive-epsilon schedule fully engages, small-loop feasibles
    reproduce easily and dominate the population, pushing out hard-won
    high-utilization feasibles. This callback keeps a deep copy of the
    best-ever-seen feasible individual and re-injects it (replacing the
    worst-CV infeasible or lowest-util feasible) whenever a better-util
    feasible is no longer present in the current population.
    """

    def __init__(self):
        super().__init__()
        self._elite = None
        self._elite_util = -np.inf

    def notify(self, algorithm):
        pop = algorithm.pop
        if pop is None:
            return

        F = pop.get("F")
        G = pop.get("G")
        if F is None or G is None:
            return

        feasible_mask = np.all(G <= 0, axis=1)

        if np.any(feasible_mask):
            utils = -F[feasible_mask, 0]
            best_in_feas = int(np.argmax(utils))
            best_util_now = float(utils[best_in_feas])
            if best_util_now > self._elite_util:
                global_idx = int(np.where(feasible_mask)[0][best_in_feas])
                self._elite = copy.deepcopy(pop[global_idx])
                self._elite_util = best_util_now

        if self._elite is None:
            return

        current_best_util = (
            float(np.max(-F[feasible_mask, 0])) if np.any(feasible_mask) else -np.inf
        )
        if self._elite_util <= current_best_util:
            return

        if not np.all(feasible_mask):
            infeas_idx = np.where(~feasible_mask)[0]
            cv = np.maximum(G[infeas_idx], 0).sum(axis=1)
            slot = int(infeas_idx[int(np.argmax(cv))])
        else:
            slot = int(np.argmax(F[:, 0]))

        pop[slot] = copy.deepcopy(self._elite)


# =============================================================================
# Category Elite Archive
# =============================================================================

# Category name -> custom out-key written by TrackOptimizationProblem._evaluate.
CATEGORY_KEYS = {
    "switch": "n_sw_pairs",
    "cross": "n_cross_comm",
    "dc": "n_dc_comm",
}


class CategoryEliteArchive(Callback):
    """Preserve the best individual containing each special element.

    Generalizes :class:`FeasibleEliteCallback` to per-category elites: for each
    category (committed switch pair / CROSS_90 / DOUBLE_CROSSOVER) the archive
    keeps the best-utilization FEASIBLE individual ever seen and re-injects it
    (one per category, replacing the worst individual) whenever the population
    no longer holds a feasible member of that category at least as good.
    Best INFEASIBLE individuals are archived too, but only for reporting —
    they are never injected.

    Must run AFTER FeasibleEliteCallback in the chain: the global elite's slot
    stops being "worst" once injected, so it is never clobbered here.
    """

    def __init__(self, inject: bool = True):
        super().__init__()
        self.inject = inject
        self.feasible: dict = {}    # category -> {"util": float, "ind": Individual}
        self.infeasible: dict = {}  # category -> {"util": float, "ind": Individual}

    def notify(self, algorithm):
        pop = algorithm.pop
        if pop is None:
            return
        used_slots: set = set()

        for category, key in CATEGORY_KEYS.items():
            counts = pop.get(key)
            if counts is None:
                return  # population evaluated without category keys
            # Re-fetched per category: an earlier category's injection has
            # already replaced individuals, and bookkeeping against a stale
            # snapshot pins the wrong utilization to an archived individual.
            F = pop.get("F")
            G = pop.get("G")
            if F is None or G is None:
                return
            feas_mask = np.all(G <= 0, axis=1)
            has_element = np.asarray(counts, dtype=float) > 0

            self._update(self.feasible, category, pop, F, has_element & feas_mask)
            self._update(self.infeasible, category, pop, F, has_element & ~feas_mask)

            if self.inject:
                self._inject(category, pop, F, G, feas_mask, has_element, used_slots)

    def _update(self, store, category, pop, F, mask):
        if not np.any(mask):
            return
        idx = np.where(mask)[0]
        best = int(idx[np.argmax(-F[idx, 0])])
        util = float(-F[best, 0])
        current = store.get(category)
        if current is None or util > current["util"]:
            store[category] = {"util": util, "ind": copy.deepcopy(pop[best])}

    def _inject(self, category, pop, F, G, feas_mask, has_element, used_slots):
        elite = self.feasible.get(category)
        if elite is None:
            return
        member_mask = has_element & feas_mask
        if np.any(member_mask) and float(np.max(-F[member_mask, 0])) >= elite["util"]:
            return  # an equal-or-better feasible member is already present

        slot = self._worst_slot(F, G, feas_mask, used_slots)
        if slot is None:
            return
        used_slots.add(slot)
        pop[slot] = copy.deepcopy(elite["ind"])

    @staticmethod
    def _worst_slot(F, G, feas_mask, used_slots):
        """Worst-CV infeasible slot first, else lowest-utilization feasible."""
        infeas = [i for i in np.where(~feas_mask)[0] if i not in used_slots]
        if infeas:
            cv = np.maximum(G[infeas], 0).sum(axis=1)
            return int(np.asarray(infeas)[int(np.argmax(cv))])
        feas = [i for i in np.where(feas_mask)[0] if i not in used_slots]
        if feas:
            return int(np.asarray(feas)[int(np.argmax(F[feas, 0]))])
        return None


# =============================================================================
# Progression Snapshots
# =============================================================================

def _compute_snapshot_targets(n_gen: int) -> list[int]:
    """Return the 10 generation numbers at which to take snapshots.

    Positions are [stride, 2*stride, ..., 10*stride] with stride = n_gen // 10,
    clamped to n_gen and deduped so small runs still produce a valid (shorter)
    list.
    """
    stride = max(1, n_gen // 10)
    raw = [i * stride for i in range(1, 11)]
    return sorted({min(g, n_gen) for g in raw})


class SnapshotCallback(Callback):
    """Render best feasible / infeasible individual to disk at target generations.

    "Best" means lowest F[0] (highest utilization). Either side may be absent
    at a given generation, in which case that PNG is simply not written.

    Rendering happens inside ``notify`` so the user can watch the
    ``outputs/snapshots/`` directory update live as the optimization runs.
    The directory is wiped of prior ``snapshot_*.png`` at construction time
    (i.e., at the start of the run) so an in-progress run cannot be confused
    with a stale previous one.
    """

    def __init__(self, targets: list[int], output_dir: Path,
                 catalog: TrackCatalog, config: OptimizationConfig, dims):
        super().__init__()
        self.targets = list(targets)
        self._target_set = set(self.targets)
        self._taken: set[int] = set()
        self.snapshots: list[dict] = []
        self._snap_dir = Path(output_dir) / "snapshots"
        self._catalog = catalog
        self._config = config
        self._decoder_config = DecoderConfig.from_optimization_config(config)
        self._dims = dims
        self._clean_dir()

    def _clean_dir(self) -> None:
        self._snap_dir.mkdir(parents=True, exist_ok=True)
        for f in self._snap_dir.glob("snapshot_*.png"):
            try:
                f.unlink()
            except OSError:
                pass

    def notify(self, algorithm):
        gen = int(algorithm.n_gen)
        if gen not in self._target_set or gen in self._taken:
            return

        pop = algorithm.pop
        if pop is None:
            return
        F = pop.get("F")
        G = pop.get("G")
        X = pop.get("X")
        if F is None or X is None:
            return

        if G is not None:
            cv_per_row = np.maximum(G, 0).sum(axis=1)
            feasible_mask = np.all(G <= 0, axis=1)
        else:
            cv_per_row = np.zeros(len(X))
            feasible_mask = np.ones(len(X), dtype=bool)

        def _pick(mask: np.ndarray) -> dict | None:
            if not np.any(mask):
                return None
            idx_pool = np.where(mask)[0]
            best = int(idx_pool[int(np.argmin(F[idx_pool, 0]))])
            return {
                "X": np.asarray(X[best]).copy(),
                "F": np.asarray(F[best]).copy(),
                "G": np.asarray(G[best]).copy() if G is not None else None,
                "cv": float(cv_per_row[best]),
            }

        self._taken.add(gen)
        snap_idx = self.targets.index(gen) + 1
        total = len(self.targets)
        snapshot = {
            "snapshot_idx": snap_idx,
            "gen": gen,
            "total": total,
            "feasible": _pick(feasible_mask),
            "infeasible": _pick(~feasible_mask),
        }
        self.snapshots.append(snapshot)

        for category in ("feasible", "infeasible"):
            entry = snapshot[category]
            if entry is None:
                continue
            self._render_one(snap_idx, gen, total, category, entry)

    def _render_one(self, snap_idx: int, gen: int, total: int,
                    category: str, entry: dict) -> None:
        logger = logging.getLogger(__name__)
        try:
            layout = decode_chromosome(
                entry["X"], self._catalog, self._config.inventory,
                dims=self._dims, config=self._decoder_config,
            )
            util = float(-entry["F"][0])
            speed = float(-entry["F"][1])
            base = (f"Snapshot {snap_idx}/{total} | Gen {gen} | {category} | "
                    f"{layout.n_physical_pieces} pcs, score={util:.1%}, "
                    f"speed={speed:.2f} m/s")
            title = base if category == "feasible" else f"{base}, CV={entry['cv']:.2f}"
            save_path = self._snap_dir / f"snapshot_{snap_idx:02d}_{category}.png"
            if hasattr(layout, "n_switch_pairs") and layout.n_switch_pairs > 0:
                fig = plot_multi_path_layout(
                    layout, self._catalog, self._config.boundary, title, save_path,
                )
            else:
                fig = plot_layout(
                    layout, self._catalog, self._config.boundary, title, save_path,
                )
            plt.close(fig)
        except Exception as e:
            logger.warning(f"Could not render snapshot {snap_idx:02d} {category}: {e}")


class CallbackChain(Callback):
    """Dispatch `notify` to a sequence of child callbacks."""

    def __init__(self, *callbacks: Callback):
        super().__init__()
        self._callbacks = callbacks

    def notify(self, algorithm):
        for cb in self._callbacks:
            cb.notify(algorithm)


# =============================================================================
# Adaptive Epsilon-Constraint Handling (Takahama & Sakai 2006)
# =============================================================================

# Closure (G[0:3]) and boundary (G[3]) are SOFT: a near-closed / near-fitting loop
# can evolve toward satisfying them, so the epsilon schedule may relax them.
# Collisions (G[4]) and per-type inventory (G[5:]) are HARD: structurally
# unbuildable, never relaxed.
SOFT_CONSTRAINT_COUNT = 4

# Collisions + inventory are weighted this far above the epsilon cap (max_cv is
# capped at n_soft, i.e. <= 4 in normalized soft-G units) so the schedule can
# never relax them: >= 4 / 0.2 (smallest real hard violation) = 20; 1000 leaves
# ample margin. Weighting (not per-constraint eps) keeps CV independent of the
# schedule, so pymoo's tournament feasible/infeasible routing -- which reads raw
# CV -- stays stable generation to generation. Per-constraint eps would make a
# relaxed individual's CV flip to 0 and reach the crowding comparison with
# crowding=None -> tournament crash.
HARD_CONSTRAINT_WEIGHT = 1000.0

# CVs at or above this are degenerate sentinels (0-piece layouts get every G set
# to DEGENERATE_G), not gradations of real infeasibility. They must be excluded
# from any population statistic (epsilon_0 calibration) or they inflate it by
# orders of magnitude.
DEGENERATE_CV_FLOOR = DEGENERATE_G / 10.0


def _epsilon_alpha(t: float, hold_until: float, perc_eps_until: float) -> float:
    """Three-phase epsilon multiplier over run progress ``t`` in [0, 1].

    Phase A (t < hold_until): 1.0 — full epsilon, broad exploration.
    Phase B (hold_until <= t < perc_eps_until): linear decay 1.0 -> 0.0.
    Phase C (t >= perc_eps_until): 0.0 — strict feasibility.
    """
    if t < hold_until:
        return 1.0
    if t < perc_eps_until:
        return 1.0 - (t - hold_until) / (perc_eps_until - hold_until)
    return 0.0


def _build_elementwise_runner(n_workers: int):
    """(elementwise_runner, pool) for parallel evaluation, or (None, None).

    ``n_workers > 1`` builds a process pool (real parallelism — evaluation is
    CPU-bound Python) wrapped in pymoo's ``StarmapParallelization``. The caller
    owns the pool and must close it after ``minimize`` returns.
    """
    if n_workers <= 1:
        return None, None
    pool = Pool(n_workers)
    return StarmapParallelization(pool.starmap), pool


def _build_termination(config: OptimizationConfig):
    """Termination from ``TerminationConfig``.

    ``period = 0`` (default): no improvement-based early stop — the run uses
    the full generation budget. A stagnation window judged while the epsilon
    schedule still relaxes constraints would cut the run before its late
    strict phase, where feasible improvements typically land.

    ``period > 0``: stop at the generation cap OR earlier when the feasible
    objective space stops improving by ``ftol`` over a ``period``-generation
    window, exactly as configured.
    """
    t = config.algorithm.termination
    n_max_gen = min(config.algorithm.n_gen, t.n_max_gen)
    if t.period <= 0:
        return MaximumGenerationTermination(n_max_gen)
    return DefaultMultiObjectiveTermination(
        xtol=t.xtol,
        ftol=t.ftol,
        period=t.period,
        n_max_gen=n_max_gen,
    )


class LegoAdaptiveEpsilon(AdaptiveEpsilonConstraintHandling):
    """Three-phase epsilon schedule for constraint relaxation.

    Allows infeasible-but-promising solutions (e.g., sidings with closure error)
    to survive and evolve toward feasibility, rather than being immediately killed
    by Deb's feasibility-first rules.

    Schedule:
        Phase A (0 to hold_until): epsilon = epsilon_0 (broad exploration)
        Phase B (hold_until to perc_eps_until): linear decay to 0
        Phase C (perc_eps_until to end): strict feasibility

    Epsilon_0: Takahama & Sakai (2006) order statistic — CV of the theta-th
    individual sorted by CV descending (theta=0.2N) — over non-degenerate CVs,
    floored at the 10th percentile of real infeasible CVs and capped at
    n_soft (all soft constraints at ~2x tolerance simultaneously).

    Recovery ratchet: when the strictly-feasible fraction falls below
    ``ratchet_trigger`` of its running peak while epsilon is active, epsilon_0
    is halved. Tighten-only; the strict phase stays exactly 0.

    Note: the class-level default ``perc_eps_until=0.7`` is overridden by
    callers in :func:`run_optimization`, which passes ``perc_eps_until=0.9``.
    """

    def __init__(self, algorithm, n_ieq_constr, hold_until=0.2,
                 perc_eps_until=0.7, n_soft=SOFT_CONSTRAINT_COUNT,
                 n_gen_planned=None, theta=0.2, ratchet_trigger=0.25,
                 ratchet_cooldown=5):
        super().__init__(algorithm, perc_eps_until=perc_eps_until)
        self.hold_until = hold_until
        self.n_ieq_constr = int(n_ieq_constr)
        self.n_soft = int(n_soft)
        self.theta = float(theta)
        self.ratchet_trigger = float(ratchet_trigger)
        self.ratchet_cooldown = int(ratchet_cooldown)
        self._feas_frac_peak = 0.0
        self._last_ratchet_gen: int | None = None
        self.last_cv_eps = float("nan")  # read by the convergence monitor
        # Schedule driver: planned generation count. With the improvement-aware
        # termination, termination.perc is max(ftol-window, n_gen, ...) progress
        # and jumps nonlinearly; the epsilon schedule must keep tracking the
        # generation fraction (identical to the old behaviour under a pure
        # n_gen termination). None falls back to termination.perc.
        self.n_gen_planned = int(n_gen_planned) if n_gen_planned else None
        self._logger = logging.getLogger(__name__)

    def _progress(self) -> float:
        """Run progress in [0, 1] driving the epsilon schedule."""
        if self.n_gen_planned:
            return min(1.0, (self.n_gen or 0) / self.n_gen_planned)
        return self.termination.perc

    def _calibrate_epsilon0(self, cv: np.ndarray) -> float:
        """Epsilon_0 from initial-population CVs (degenerate sentinels excluded)."""
        real = cv[cv < DEGENERATE_CV_FLOOR]
        infeas = real[real > 0]
        if len(infeas) == 0:
            return 0.0
        order = np.sort(real)[::-1]
        order_stat = float(order[min(len(order) - 1, int(self.theta * len(order)))])
        floor = float(np.percentile(infeas, 10))
        return min(max(order_stat, floor), float(self.n_soft))

    def _initialize_advance(self, infills=None, **kwargs):
        cv = None
        if infills is not None and infills.has("cv"):
            cv = infills.get("cv").flatten()
        elif self.pop is not None and self.pop.has("cv"):
            cv = self.pop.get("cv").flatten()

        self.max_cv = self._calibrate_epsilon0(cv) if cv is not None and len(cv) else 1.0
        self._logger.info(f"Epsilon-constraint: epsilon_0={self.max_cv:.3f}")
        return super(AdaptiveEpsilonConstraintHandling, self)._initialize_advance(
            infills, **kwargs
        )

    def _ratchet_check(self, alpha: float, cv: np.ndarray | None, gen: int) -> None:
        """Halve epsilon_0 on strict-feasibility collapse (tighten-only)."""
        if alpha <= 0.0 or self.max_cv <= 0.0 or cv is None or len(cv) == 0:
            return
        feas_frac = float(np.mean(cv <= 0.0))
        self._feas_frac_peak = max(self._feas_frac_peak, feas_frac)
        if self._feas_frac_peak <= 0.0:
            return
        if (self._last_ratchet_gen is not None
                and gen - self._last_ratchet_gen < self.ratchet_cooldown):
            return
        if feas_frac < self.ratchet_trigger * self._feas_frac_peak:
            self.max_cv *= 0.5
            self._last_ratchet_gen = gen
            self._logger.info(
                f"Epsilon ratchet at gen {gen}: feasible {feas_frac:.1%} < "
                f"{self.ratchet_trigger:.0%} of peak {self._feas_frac_peak:.1%}; "
                f"epsilon_0 halved to {self.max_cv:.3f}"
            )

    def _adapt_constraint_handling(self, config, **kwargs):
        alpha = _epsilon_alpha(self._progress(), self.hold_until, self.perc_eps_until)

        pop = self.pop
        pop_cv = (pop.get("cv").flatten()
                  if pop is not None and pop.has("cv") else None)
        self._ratchet_check(alpha, pop_cv, int(getattr(self, "n_gen", 0) or 0))

        # Relax ONLY the soft constraints (closure + boundary): the scheduled
        # epsilon is the scalar FEAS threshold (CV <= cv_eps), as in the original
        # design. Used by the feasibility-first survival, not the tournament.
        config["cv_eps"] = alpha * self.max_cv
        self.last_cv_eps = float(config["cv_eps"])

        # Keep collisions + inventory HARD by weighting them >> the epsilon cap,
        # so the smallest real hard violation already exceeds any cv_eps and can
        # never be relaxed. Soft constraints keep weight 1. CV is therefore a
        # fixed (schedule-independent) function of G, so a closed self-crosser
        # has CV > cv_eps at every phase and is dropped by the survival, while
        # the tournament's raw-CV routing is unchanged. See HARD_CONSTRAINT_WEIGHT.
        scale = np.ones(self.n_ieq_constr, dtype=float)
        scale[self.n_soft:] = 1.0 / HARD_CONSTRAINT_WEIGHT
        config["cv_ieq"] = dict(scale=scale, eps=0.0, pow=None, func=np.sum)


# =============================================================================
# Optimization
# =============================================================================

def _salvage_failed_run(algorithm, monitor, output_dir, logger):
    """Write error.log and build a partial result from the live population.

    Returns None (caller re-raises) when there is no population to salvage.
    """
    tb = traceback.format_exc()
    gens = monitor.data.get("n_gen") or []
    reached = gens[-1] if gens else 0
    logger.error(f"Optimization crashed at generation {reached}:\n{tb}")
    if output_dir is not None:
        try:
            (Path(output_dir) / "error.log").write_text(
                f"Optimization crashed at generation {reached}.\n\n{tb}",
                encoding="utf-8",
            )
        except OSError:
            logger.warning("Could not write error.log")

    pop = getattr(algorithm, "pop", None)
    if pop is None or len(pop) == 0:
        return None
    res = SimpleNamespace(pop=pop, algorithm=algorithm, opt=None,
                          X=None, F=None, G=None, CV=None, exec_time=None)
    res.crashed = True
    return res


def run_optimization(
    config: OptimizationConfig,
    catalog: TrackCatalog,
    verbose: bool = False,
    output_dir: Path | None = None,
) -> object:
    """Run NSGA-II multi-objective track optimization.

    If ``output_dir`` is given, progression snapshots are rendered live into
    ``<output_dir>/snapshots/`` during the run. Otherwise snapshots are still
    captured in memory on ``res.snapshots`` but no files are written.
    """
    logger = logging.getLogger(__name__)

    # Parallel elementwise evaluation (config.n_workers). The pool is owned
    # here and released after minimize() — decode/eval is CPU-bound Python,
    # so a process pool gives real parallelism.
    eval_runner, eval_pool = _build_elementwise_runner(config.n_workers)
    problem_kwargs = {"elementwise_runner": eval_runner} if eval_runner is not None else {}
    problem = TrackOptimizationProblem(catalog, config, **problem_kwargs)
    dims = problem.dims

    # Seed the global numpy RNG so the custom operators (PartitionedCrossover /
    # PartitionedMutation in src/operators.py), which draw from the global RNG,
    # are reproducible. seed=null leaves the run non-deterministic.
    if config.algorithm.seed is not None:
        np.random.seed(config.algorithm.seed)

    sampler = IntegerSampling(
        catalog, config,
        heuristic_ratio=config.algorithm.heuristic_ratio,
        seed=config.algorithm.seed,
    )

    repair = TrackRepairPipeline(
        dims=dims,
        inventory_by_index=problem.inventory_by_index,
        catalog_fk_table=catalog._fk_table,
        enable_closure_repair=True,
        enable_boundary_repair=True,
    )

    crossover = PartitionedCrossover(dims, prob=config.algorithm.crossover_prob)
    mutation = PartitionedMutation(dims, prob=config.algorithm.mutation_prob)

    algo_name = config.algorithm.name
    base_algorithm = NSGA2(
        pop_size=config.algorithm.pop_size,
        sampling=sampler,
        crossover=crossover,
        mutation=mutation,
        repair=repair,
        survival=ConstrRankAndCrowding(),
        eliminate_duplicates=config.algorithm.eliminate_duplicates,
    )
    logger.info("NSGA-II with ConstrRankAndCrowding (Deb's feasibility-first)")

    algorithm = LegoAdaptiveEpsilon(
        base_algorithm,
        n_ieq_constr=problem.n_ieq_constr,
        hold_until=0.2,
        perc_eps_until=0.9,
        n_gen_planned=config.algorithm.n_gen,
    )

    termination = _build_termination(config)

    elite_callback = FeasibleEliteCallback()
    # After FeasibleEliteCallback: the global elite's slot is no longer
    # "worst", so category injection never clobbers it.
    category_archive = CategoryEliteArchive()
    monitor = ConvergenceMonitorCallback(output_dir=output_dir,
                                         closure_tolerance=config.closure_tolerance,
                                         angle_tolerance=config.angle_tolerance)
    monitor.epsilon_source = algorithm
    chain: list[Callback] = [elite_callback, category_archive, monitor]
    if output_dir is not None:
        snapshot_cb = SnapshotCallback(
            _compute_snapshot_targets(config.algorithm.n_gen),
            output_dir, catalog, config, dims,
        )
        chain.append(snapshot_cb)
    else:
        snapshot_cb = None
    if verbose:
        chain.append(ProgressCallback(every_n_gen=1, total_inventory=config.total_inventory))
    callback = CallbackChain(*chain)

    logger.info(f"Starting {algo_name} track optimization...")
    logger.info("Objectives: utilization + speed (bi-objective)")
    logger.info(f"Population: {config.algorithm.pop_size}")
    logger.info(f"Generations: {config.algorithm.n_gen}")
    logger.info(
        f"Chromosome: {dims.n_var} genes (main={dims.n_main}, junctions={dims.max_junctions})"
    )
    logger.info(f"Total inventory: {config.total_inventory} pieces")
    logger.info(f"Heuristic seeding: {config.algorithm.heuristic_ratio:.0%}")

    if eval_pool is not None:
        logger.info(f"Parallel evaluation: {config.n_workers} workers")

    # copy_algorithm=False: minimize's default deepcopy would detach the
    # running algorithm from monitor.epsilon_source and from the salvage path.
    minimize_kwargs = dict(verbose=False, save_history=False,
                           seed=config.algorithm.seed, copy_algorithm=False)
    if callback is not None:
        minimize_kwargs["callback"] = callback

    try:
        res = minimize(problem, algorithm, termination, **minimize_kwargs)
    except Exception:
        res = _salvage_failed_run(algorithm, monitor, output_dir, logger)
        if res is None:
            raise
    finally:
        if eval_pool is not None:
            # terminate(), not close()+join(): after a KeyboardInterrupt the
            # workers sit interrupted on the task-queue semaphore and join()
            # blocks the terminal forever. On normal completion no tasks are
            # in flight, so a hard stop is equivalent and instant.
            eval_pool.terminate()
            eval_pool.join()

    res.monitor = monitor
    res.monitor_data = monitor.data
    res.snapshots = snapshot_cb.snapshots if snapshot_cb is not None else []
    res.category_elites = category_archive

    logger.info("Optimization complete!")

    if res.pop is not None:
        G = res.pop.get("G")
        F = res.pop.get("F")
        X = res.pop.get("X")
        if G is not None:
            n_feasible = np.sum(np.all(G <= 0, axis=1))
            logger.info(f"Feasible solutions: {n_feasible}/{len(res.pop)}")

        if X is not None and len(X) > 0 and F is not None:
            feasible_mask = (
                np.all(G <= 0, axis=1) if G is not None else np.ones(len(X), dtype=bool)
            )
            if np.any(feasible_mask):
                feasible_F = F[feasible_mask]
                feasible_indices = np.where(feasible_mask)[0]
                best_idx = feasible_indices[np.argmin(feasible_F[:, 0])]
                best_layout = decode_chromosome(
                    X[best_idx], catalog, config.inventory,
                    dims=dims, config=problem.decoder_config,
                )
                logger.info(
                    f"Best feasible: {best_layout.n_physical_pieces}/"
                    f"{sum(config.inventory.values())} pieces, "
                    f"score={-F[best_idx, 0]:.1%}, speed={-F[best_idx, 1]:.2f} m/s, "
                    f"switch_pairs={best_layout.n_switch_pairs}"
                )
                log_piece_usage(best_layout, config.inventory, catalog, logger)

    return res


# =============================================================================
# Category Report
# =============================================================================

# General geometric context per category (rule-level facts, not run data).
_CATEGORY_CONTEXT = {
    "switch": "a passing siding adds a parallel track segment inside the "
              "loop's existing bounding box",
    "cross": "a CROSS_90 needs the loop to cross itself perpendicular; the "
             "figure-8 family spends >=24 R40 curves on two turning-circle lobes",
    "dc": "a DOUBLE_CROSSOVER joins two parallel tracks 16 studs apart and "
          "both traversals must jointly cover all 4 ports",
}

# Category -> (active-descriptor reader, drop_log entry prefix).
_CATEGORY_GENOTYPE = {
    "switch": (get_active_junctions, "junction["),
    "cross": (get_active_cross_junctions, "CROSS["),
    "dc": (get_active_double_crossovers, "DC["),
}


def _collect_drop_reasons(category, X, catalog, config, dims, decoder_cfg,
                          limit=5):
    """Decode up to ``limit`` final-population genomes whose descriptor block
    is active for ``category`` and aggregate the decoder's drop reasons."""
    reader, prefix = _CATEGORY_GENOTYPE[category]
    reasons: list[str] = []
    sampled = 0
    for row in X:
        if sampled >= limit:
            break
        x = np.asarray(row, dtype=np.int16)
        if not reader(x, dims):
            continue
        sampled += 1
        layout = decode_chromosome(x, catalog, config.inventory,
                                   dims=dims, config=decoder_cfg)
        reasons.extend(e for e in layout.drop_log if e.startswith(prefix))
    deduped: list[str] = []
    for r in reasons:
        if r not in deduped:
            deduped.append(r)
    return deduped[:10]


def _write_category_report(res, output_dir, catalog, config, dims, decoder_cfg,
                           F, X, feasible_mask, plot_fn) -> None:
    """Render best_with_<cat>.png + category_report.md from the elite archive."""
    archive = getattr(res, "category_elites", None)
    if archive is None:
        return

    best_util = (float(np.max(-F[feasible_mask][:, 0]))
                 if np.any(feasible_mask) else None)
    boundary = config.boundary
    total_inv = sum(config.inventory.values())
    lines = ["# Category report", ""]

    for category in CATEGORY_KEYS:
        lines.append(f"## {category}")
        elite = archive.feasible.get(category)
        if elite is not None:
            ind = elite["ind"]
            layout = decode_chromosome(
                np.asarray(ind.X, dtype=np.int16), catalog,
                config.inventory, dims=dims, config=decoder_cfg,
            )
            util = elite["util"]
            n_element = {
                "switch": layout.n_switch_pairs,
                "cross": layout.n_cross_pieces,
                "dc": layout.n_dbl_crossovers,
            }[category]
            gap = (f"{(best_util - util) * 100:.1f}pp below global best"
                   if best_util is not None else "n/a")
            n_phys = layout.n_physical_pieces
            plot_fn(
                layout,
                f"Best with {category} ({n_phys}/{total_inv} pcs, "
                f"score {util:.1%})",
                output_dir / f"best_with_{category}.png",
            )
            spans = [p.states for p in layout.paths if len(p.states) > 1]
            lines += [
                f"- utilization score: {util:.1%} ({gap})",
                f"- pieces: {n_phys}/{total_inv} ({n_phys / total_inv:.1%} of inventory), "
                f"speed: {-float(ind.F[1]):.2f} m/s, {category} count: {n_element}",
            ]
            if spans:  # all-degenerate paths must not abort the whole report
                xs = np.concatenate([s[:, 0] for s in spans])
                ys = np.concatenate([s[:, 1] for s in spans])
                lines.append(
                    f"- bbox: {xs.max() - xs.min():.0f} x {ys.max() - ys.min():.0f} studs "
                    f"in {boundary.width:.0f} x {boundary.height:.0f} box"
                )
        else:
            lines.append("- no feasible solution containing this element was seen")
            infeas = archive.infeasible.get(category)
            if infeas is not None:
                cv = float(np.sum(np.maximum(0, infeas["ind"].G)))
                lines.append(
                    f"- best infeasible: util {infeas['util']:.1%}, CV={cv:.2f}"
                )
            reasons = _collect_drop_reasons(category, X, catalog, config,
                                            dims, decoder_cfg)
            if reasons:
                lines.append("- decoder drop reasons (final-population sample):")
                lines.extend(f"  - {r}" for r in reasons)
        lines.append(f"- context: {_CATEGORY_CONTEXT[category]}")
        lines.append("")

    (output_dir / "category_report.md").write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# Result Saving
# =============================================================================

def save_results(res, output_dir: Path, catalog: TrackCatalog,
                 config: OptimizationConfig) -> None:
    """Save optimization results including Pareto front."""
    logger = logging.getLogger(__name__)
    output_dir.mkdir(parents=True, exist_ok=True)
    dims = compute_dimensions(config, catalog)
    decoder_cfg = DecoderConfig.from_optimization_config(config)

    if getattr(res, "crashed", False):
        logger.warning("Saving partial results from a crashed run")

    if res.pop is None:
        logger.warning("No population to save")
        return

    X = res.pop.get("X")
    F = res.pop.get("F")
    G = res.pop.get("G")

    np.savetxt(output_dir / "chromosomes.csv", X, delimiter=",", fmt="%d",
               header=chromosome_csv_header(dims), comments="")
    np.savetxt(output_dir / "fitness.csv", F, delimiter=",",
               header="neg_utilization,neg_slowest_route_speed", comments="")
    if G is not None:
        # G layout: 5 base + one per catalog piece index (inv_<t>).
        constraint_header = (
            "closure_x,closure_y,closure_theta,boundary,collisions,"
            + ",".join(f"inv_{i}" for i in range(catalog.n_pieces))
        )
        np.savetxt(output_dir / "constraints.csv", G, delimiter=",",
                   header=constraint_header, comments="")

    feasible_mask = np.all(G <= 0, axis=1) if G is not None else np.ones(len(X), dtype=bool)
    feasible_indices = np.where(feasible_mask)[0]

    # Run-cumulative feasible front from the monitor: the terminal population
    # is a converged monoculture, so this archive is what actually shows the
    # discovered trade-off curve. Persisted so the plot can be regenerated.
    monitor = getattr(res, "monitor", None)
    archive_F = getattr(monitor, "best_front", None) if monitor is not None else None
    if archive_F is not None and len(archive_F) > 0:
        np.savetxt(output_dir / "pareto_archive.csv", archive_F, delimiter=",",
                   header="f0_neg_utilization,f1_neg_speed", comments="")

    try:
        fig = plot_pareto_front(F, G, title="Pareto Front: Utilization vs Speed",
                                save_path=output_dir / "pareto_front.png",
                                archive_F=archive_F)
        plt.close(fig)
        logger.info("Pareto front saved to pareto_front.png")
    except Exception as e:
        logger.warning(f"Could not plot Pareto front: {e}")

    def _plot_layout(layout, title, path):
        if hasattr(layout, 'n_switch_pairs') and layout.n_switch_pairs > 0:
            fig = plot_multi_path_layout(layout, catalog, config.boundary, title, path)
        else:
            fig = plot_layout(layout, catalog, config.boundary, title, path)
        plt.close(fig)

    # Each artifact block is isolated: a decode/render failure in one must not
    # cost the remaining artifacts of the run.
    try:
        best_overall_idx = np.argmin(F[:, 0])
        best_overall_layout = decode_chromosome(
            X[best_overall_idx], catalog, config.inventory,
            dims=dims, config=decoder_cfg,
        )
        best_overall_util = -F[best_overall_idx, 0]
        best_overall_speed = -F[best_overall_idx, 1]
        best_overall_cv = (
            float(np.sum(np.maximum(0, G[best_overall_idx]))) if G is not None else 0.0
        )
        is_feasible = bool(feasible_mask[best_overall_idx])

        logger.info(
            f"Best overall: {best_overall_layout.n_physical_pieces}/"
            f"{sum(config.inventory.values())} pieces, "
            f"score={best_overall_util:.1%}, speed={best_overall_speed:.2f} m/s, "
            f"switch_pairs={best_overall_layout.n_switch_pairs}, CV={best_overall_cv:.2f}"
            f"{' (FEASIBLE)' if is_feasible else ' (infeasible)'}"
        )
    except Exception as e:
        logger.warning(f"Could not summarize best overall individual: {e}")

    try:
        if len(feasible_indices) > 0:
            best_feas_idx = feasible_indices[np.argmin(F[feasible_indices, 0])]
            best_feas_layout = decode_chromosome(
                X[best_feas_idx], catalog, config.inventory,
                dims=dims, config=decoder_cfg,
            )
            best_feas_util = -F[best_feas_idx, 0]
            best_feas_speed = -F[best_feas_idx, 1]
            _plot_layout(
                best_feas_layout,
                f"Best Feasible ({best_feas_layout.n_physical_pieces}/"
                f"{sum(config.inventory.values())} pcs, score {best_feas_util:.1%}, "
                f"{best_feas_speed:.2f} m/s)",
                output_dir / "best_layout.png",
            )
        else:
            logger.warning("No feasible solutions found")
    except Exception as e:
        logger.warning(f"Could not render best_layout.png: {e}")

    log_front_scale_and_compromise(F, feasible_indices, logger)

    try:
        infeasible_indices = (
            np.where(~feasible_mask)[0] if G is not None else np.array([], dtype=int)
        )
        if len(infeasible_indices) > 0:
            best_infeas_idx = infeasible_indices[np.argmin(F[infeasible_indices, 0])]
            best_infeas_layout = decode_chromosome(
                X[best_infeas_idx], catalog, config.inventory,
                dims=dims, config=decoder_cfg,
            )
            best_infeas_util = -F[best_infeas_idx, 0]
            best_infeas_speed = -F[best_infeas_idx, 1]
            best_infeas_cv = float(np.sum(np.maximum(0, G[best_infeas_idx])))
            _plot_layout(
                best_infeas_layout,
                f"Best Infeasible ({best_infeas_layout.n_physical_pieces}/"
                f"{sum(config.inventory.values())} pcs, score {best_infeas_util:.1%}, "
                f"{best_infeas_speed:.2f} m/s, CV={best_infeas_cv:.2f})",
                output_dir / "best_infeasible.png",
            )
    except Exception as e:
        logger.warning(f"Could not render best_infeasible.png: {e}")

    try:
        _write_category_report(res, output_dir, catalog, config, dims,
                               decoder_cfg, F, X, feasible_mask, _plot_layout)
    except Exception as e:
        logger.warning(f"Could not write category report: {e}")

    logger.info(f"Results saved to {output_dir}")
