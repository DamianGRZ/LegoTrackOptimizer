"""NSGA-II runner for the track optimization problem."""

from __future__ import annotations

import copy
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.constraints.eps import AdaptiveEpsilonConstraintHandling
from pymoo.core.callback import Callback
from pymoo.operators.survival.rank_and_crowding import ConstrRankAndCrowding
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from src.algorithm.monitoring import ConvergenceMonitorCallback
from src.catalog import TrackCatalog
from src.config import OptimizationConfig
from src.decoder import decode_chromosome
from src.encoding import compute_dimensions
from src.operators import PartitionedCrossover, PartitionedMutation
from src.problem import TrackOptimizationProblem
from src.repair import TrackRepairPipeline
from src.sampling import IntegerSampling
from src.visualization import plot_layout, plot_multi_path_layout, plot_pareto_front


def log_piece_usage(layout, inventory: dict, catalog: TrackCatalog,
                    logger: logging.Logger) -> None:
    """Log piece usage breakdown."""
    piece_counts: dict = {}

    if hasattr(layout, 'main_loop_pieces'):
        for p in layout.main_loop_pieces:
            if p >= 0:
                piece_counts[p] = piece_counts.get(p, 0) + 1
    elif hasattr(layout, 'indices'):
        for p in layout.indices:
            if p >= 0:
                piece_counts[p] = piece_counts.get(p, 0) + 1

    if hasattr(layout, 'switch_pairs'):
        for pair in layout.switch_pairs:
            for p in pair.branch_pieces:
                if p >= 0:
                    piece_counts[p] = piece_counts.get(p, 0) + 1

    logger.info("Piece usage:")
    for piece_id, count in sorted(inventory.items()):
        idx = catalog._id_to_index.get(piece_id)
        if idx is not None:
            used = piece_counts.get(idx, 0)
            logger.info(f"  {piece_id}: {used}/{count}")


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

        best_util = float(-np.min(F[:, 0]))
        best_speed = float(-np.min(F[:, 1]))
        pieces = int(best_util * self.total_inventory)

        n_feasible = 0
        if G is not None:
            n_feasible = int(np.sum(np.all(G <= 0, axis=1)))

        logger.info(
            f"Gen {gen:4d} | best_util={best_util:.1%} ({pieces} pcs) | "
            f"best_speed={best_speed:.2f} m/s | feasible={n_feasible}/{len(pop)}"
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

        current_best_util = float(np.max(-F[feasible_mask, 0])) if np.any(feasible_mask) else -np.inf
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
                entry["X"], self._catalog, self._config.inventory, dims=self._dims,
            )
            util = float(-entry["F"][0])
            speed = float(-entry["F"][1])
            base = (f"Snapshot {snap_idx}/{total} | Gen {gen} | {category} | "
                    f"{layout.n_pieces} pcs, util={util:.1%}, speed={speed:.2f} m/s")
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

class LegoAdaptiveEpsilon(AdaptiveEpsilonConstraintHandling):
    """Three-phase epsilon schedule for constraint relaxation.

    Allows infeasible-but-promising solutions (e.g., sidings with closure error)
    to survive and evolve toward feasibility, rather than being immediately killed
    by Deb's feasibility-first rules.

    Schedule:
        Phase A (0 to hold_until): epsilon = epsilon_0 (broad exploration)
        Phase B (hold_until to perc_eps_until): linear decay to 0
        Phase C (perc_eps_until to end): strict feasibility

    Epsilon_0 auto-initialized from 80th percentile of initial population CVs
    (Takahama & Sakai 2006).
    """

    def __init__(self, algorithm, hold_until=0.2, perc_eps_until=0.7):
        super().__init__(algorithm, perc_eps_until=perc_eps_until)
        self.hold_until = hold_until
        self._logger = logging.getLogger(__name__)

    def _initialize_advance(self, infills=None, **kwargs):
        # Target: cover siding closure errors (~15-30 CV) without letting
        # random garbage (CV > 100) compete with feasible circles.
        # Use 10th percentile of infeasible CVs, capped at 30.
        cv = None
        if infills is not None and infills.has("cv"):
            cv = infills.get("cv").flatten()
        elif self.pop is not None and self.pop.has("cv"):
            cv = self.pop.get("cv").flatten()

        if cv is not None:
            infeas_cv = cv[cv > 0]
            if len(infeas_cv) > 0:
                self.max_cv = float(np.percentile(infeas_cv, 10))
                self.max_cv = min(self.max_cv, 30.0)
            else:
                self.max_cv = 1.0
        else:
            self.max_cv = 15.0

        self._logger.info(f"Epsilon-constraint: epsilon_0={self.max_cv:.3f}")
        return super(AdaptiveEpsilonConstraintHandling, self)._initialize_advance(infills, **kwargs)

    def _adapt_constraint_handling(self, config, **kwargs):
        t = self.termination.perc
        if t < self.hold_until:
            alpha = 1.0
        elif t < self.perc_eps_until:
            alpha = 1.0 - (t - self.hold_until) / (self.perc_eps_until - self.hold_until)
        else:
            alpha = 0.0
        config["cv_eps"] = alpha * self.max_cv


# =============================================================================
# Optimization
# =============================================================================

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

    problem = TrackOptimizationProblem(catalog, config)
    dims = problem.dims

    sampler = IntegerSampling(
        catalog, config,
        heuristic_ratio=config.algorithm.heuristic_ratio,
    )

    inventory_by_index = problem._convert_inventory(config.inventory)
    repair = TrackRepairPipeline(
        dims=dims,
        inventory_by_index=inventory_by_index,
        catalog_fk_table=catalog._fk_table,
        enable_closure_repair=True,
    )

    crossover = PartitionedCrossover(dims, prob=config.algorithm.crossover_prob)
    mutation = PartitionedMutation(dims, prob=config.algorithm.mutation_prob)

    base_algorithm = NSGA2(
        pop_size=config.algorithm.pop_size,
        sampling=sampler,
        crossover=crossover,
        mutation=mutation,
        repair=repair,
        survival=ConstrRankAndCrowding(),
        eliminate_duplicates=config.algorithm.eliminate_duplicates,
    )

    # Wrap with adaptive epsilon-constraint handling
    # Allows infeasible sidings to survive and evolve toward feasibility
    algorithm = LegoAdaptiveEpsilon(
        base_algorithm,
        hold_until=0.2,
        perc_eps_until=0.9,
    )

    termination = get_termination("n_gen", config.algorithm.n_gen)

    elite_callback = FeasibleEliteCallback()
    monitor = ConvergenceMonitorCallback(ref_point=(0.10, -0.55))
    chain: list[Callback] = [elite_callback, monitor]
    if output_dir is not None:
        snapshot_cb = SnapshotCallback(
            _compute_snapshot_targets(config.algorithm.n_gen),
            output_dir, catalog, config, dims,
        )
        chain.append(snapshot_cb)
    else:
        snapshot_cb = None
    if verbose:
        chain.append(ProgressCallback(every_n_gen=10, total_inventory=config.total_inventory))
    callback = CallbackChain(*chain)

    logger.info("Starting NSGA-II track optimization...")
    logger.info(f"Objectives: utilization + speed (bi-objective)")
    logger.info(f"Population: {config.algorithm.pop_size}")
    logger.info(f"Generations: {config.algorithm.n_gen}")
    logger.info(f"Chromosome: {dims.n_var} genes (main={dims.n_main}, junctions={dims.max_junctions})")
    logger.info(f"Total inventory: {config.total_inventory} pieces")
    logger.info(f"Heuristic seeding: {config.algorithm.heuristic_ratio:.0%}")

    minimize_kwargs = dict(verbose=False, save_history=False)
    if callback is not None:
        minimize_kwargs["callback"] = callback

    res = minimize(problem, algorithm, termination, **minimize_kwargs)

    res.monitor = monitor
    res.monitor_data = monitor.data
    res.snapshots = snapshot_cb.snapshots if snapshot_cb is not None else []

    logger.info("Optimization complete!")

    if res.pop is not None:
        G = res.pop.get("G")
        F = res.pop.get("F")
        X = res.pop.get("X")
        if G is not None:
            n_feasible = np.sum(np.all(G <= 0, axis=1))
            logger.info(f"Feasible solutions: {n_feasible}/{len(res.pop)}")

        if X is not None and len(X) > 0 and F is not None:
            feasible_mask = np.all(G <= 0, axis=1) if G is not None else np.ones(len(X), dtype=bool)
            if np.any(feasible_mask):
                feasible_F = F[feasible_mask]
                feasible_indices = np.where(feasible_mask)[0]
                best_idx = feasible_indices[np.argmin(feasible_F[:, 0])]
                best_layout = decode_chromosome(
                    X[best_idx], catalog, config.inventory, dims=dims,
                )
                logger.info(
                    f"Best feasible: {best_layout.n_pieces} pieces, "
                    f"util={-F[best_idx, 0]:.1%}, speed={-F[best_idx, 1]:.2f} m/s, "
                    f"switches={best_layout.n_switch_pairs}"
                )
                log_piece_usage(best_layout, config.inventory, catalog, logger)

    return res


# =============================================================================
# Result Saving
# =============================================================================

def save_results(res, output_dir: Path, catalog: TrackCatalog,
                 config: OptimizationConfig) -> None:
    """Save optimization results including Pareto front."""
    logger = logging.getLogger(__name__)
    output_dir.mkdir(parents=True, exist_ok=True)
    dims = compute_dimensions(config, catalog)

    if res.pop is None:
        logger.warning("No population to save")
        return

    X = res.pop.get("X")
    F = res.pop.get("F")
    G = res.pop.get("G")

    np.savetxt(output_dir / "chromosomes.csv", X, delimiter=",", fmt="%d")
    np.savetxt(output_dir / "fitness.csv", F, delimiter=",",
               header="neg_utilization,neg_min_speed", comments="")
    if G is not None:
        # Stage B G layout: 5 base + one per catalog piece index (inv_<t>).
        constraint_header = (
            "closure_x,closure_y,closure_theta,boundary,collisions,"
            + ",".join(f"inv_{i}" for i in range(catalog.n_pieces))
        )
        np.savetxt(output_dir / "constraints.csv", G, delimiter=",",
                   header=constraint_header, comments="")

    feasible_mask = np.all(G <= 0, axis=1) if G is not None else np.ones(len(X), dtype=bool)
    feasible_indices = np.where(feasible_mask)[0]

    try:
        plot_pareto_front(F, G, title="Pareto Front: Utilization vs Speed",
                          save_path=output_dir / "pareto_front.png")
        logger.info("Pareto front saved to pareto_front.png")
    except Exception as e:
        logger.warning(f"Could not plot Pareto front: {e}")

    def _plot_layout(layout, title, path):
        if hasattr(layout, 'n_switch_pairs') and layout.n_switch_pairs > 0:
            plot_multi_path_layout(layout, catalog, config.boundary, title, path)
        else:
            plot_layout(layout, catalog, config.boundary, title, path)

    best_overall_idx = np.argmin(F[:, 0])
    best_overall_layout = decode_chromosome(
        X[best_overall_idx], catalog, config.inventory, dims=dims,
    )
    best_overall_util = -F[best_overall_idx, 0]
    best_overall_speed = -F[best_overall_idx, 1]
    best_overall_cv = float(np.sum(np.maximum(0, G[best_overall_idx]))) if G is not None else 0.0
    is_feasible = feasible_mask[best_overall_idx] if G is not None else True

    logger.info(
        f"Best overall: {best_overall_layout.n_pieces} pieces, "
        f"util={best_overall_util:.1%}, speed={best_overall_speed:.2f} m/s, "
        f"switches={best_overall_layout.n_switch_pairs}, CV={best_overall_cv:.2f}"
        f"{' (FEASIBLE)' if is_feasible else ' (infeasible)'}"
    )

    if len(feasible_indices) > 0:
        best_feas_idx = feasible_indices[np.argmin(F[feasible_indices, 0])]
        best_feas_layout = decode_chromosome(
            X[best_feas_idx], catalog, config.inventory, dims=dims,
        )
        best_feas_util = -F[best_feas_idx, 0]
        best_feas_speed = -F[best_feas_idx, 1]
        _plot_layout(
            best_feas_layout,
            f"Best Feasible ({best_feas_layout.n_pieces} pcs, "
            f"{best_feas_util:.1%} util, {best_feas_speed:.2f} m/s)",
            output_dir / "best_layout.png",
        )
    else:
        logger.warning("No feasible solutions found")

    infeasible_indices = np.where(~feasible_mask)[0] if G is not None else np.array([], dtype=int)
    if len(infeasible_indices) > 0:
        best_infeas_idx = infeasible_indices[np.argmin(F[infeasible_indices, 0])]
        best_infeas_layout = decode_chromosome(
            X[best_infeas_idx], catalog, config.inventory, dims=dims,
        )
        best_infeas_util = -F[best_infeas_idx, 0]
        best_infeas_speed = -F[best_infeas_idx, 1]
        best_infeas_cv = float(np.sum(np.maximum(0, G[best_infeas_idx])))
        _plot_layout(
            best_infeas_layout,
            f"Best Infeasible ({best_infeas_layout.n_pieces} pcs, "
            f"{best_infeas_util:.1%} util, {best_infeas_speed:.2f} m/s, "
            f"CV={best_infeas_cv:.2f})",
            output_dir / "best_infeasible.png",
        )

    logger.info(f"Results saved to {output_dir}")
