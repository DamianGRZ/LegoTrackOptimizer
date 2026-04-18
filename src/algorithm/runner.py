"""NSGA-II runner for the track optimization problem."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.constraints.eps import AdaptiveEpsilonConstraintHandling
from pymoo.core.callback import Callback
from pymoo.operators.survival.rank_and_crowding import ConstrRankAndCrowding
from pymoo.optimize import minimize
from pymoo.termination import get_termination

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
) -> object:
    """Run NSGA-II multi-objective track optimization."""
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
        perc_eps_until=0.7,
    )

    termination = get_termination("n_gen", config.algorithm.n_gen)

    callback = None
    if verbose:
        callback = ProgressCallback(
            every_n_gen=10,
            total_inventory=config.total_inventory,
        )

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
               header="neg_utilization,neg_avg_speed", comments="")
    if G is not None:
        np.savetxt(output_dir / "constraints.csv", G, delimiter=",")

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
        _plot_layout(
            best_overall_layout,
            f"Best Layout (infeasible, {best_overall_layout.n_pieces} pcs, "
            f"CV={best_overall_cv:.1f})",
            output_dir / "best_layout.png",
        )
        logger.warning("No feasible solutions found")

    logger.info(f"Results saved to {output_dir}")
