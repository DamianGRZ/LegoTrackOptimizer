"""NSGA-II runner for the port-pair track optimization problem.

Wires the V2 problem with V2 operators and reuses V1's encoding-agnostic
callbacks (progress logging, feasible-elite preservation, adaptive epsilon
constraint handling).

Visualization is intentionally minimal in v0 — best-feasible chromosome is
saved as CSV plus a slot-pose scatter plot. Full V1-style track rendering
will land alongside the PortGraph -> MultiPathLayout adapter in a follow-up.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.constraints.eps import AdaptiveEpsilonConstraintHandling
from pymoo.core.callback import Callback
from pymoo.operators.survival.rank_and_crowding import ConstrRankAndCrowding
from pymoo.core.duplicate import ElementwiseDuplicateElimination
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from .canonical_hash import canonical_graph_signature
from .niched_survival import TopologyNichedSurvival
from .catalog import TrackCatalog
from .config import OptimizationConfig
from .decoder import decode_chromosome, port_graph_to_layout
from .operators import PortPairCrossover, PortPairMutation, PortPairSampling
from .problem import PortPairProblem
from .repair import PortPairRepairPipeline
from .visualization import plot_layout


# =============================================================================
# Duplicate elimination — topology-aware
# =============================================================================


class CanonicalGraphDuplicates(ElementwiseDuplicateElimination):
    """Collapse anchor-shifted / slot-permuted clones into one bucket.

    Default pymoo ``eliminate_duplicates=True`` compares raw int16 chromosomes
    elementwise, so two layouts with identical port-graph topology but
    different anchor poses or slot indices count as distinct. The result was
    that the population filled with near-duplicate ovals.

    We compare ``canonical_graph_signature`` instead — a 16-byte hash that is
    invariant to anchor pose, slot relabeling, and edge ordering.
    """

    def __init__(self, dims, catalog) -> None:
        super().__init__()
        self._dims = dims
        self._catalog = catalog

    def is_equal(self, a, b) -> bool:
        sig_a = a.get("canon_sig")
        if sig_a is None:
            sig_a = canonical_graph_signature(a.X, self._dims, self._catalog)
            a.set("canon_sig", sig_a)
        sig_b = b.get("canon_sig")
        if sig_b is None:
            sig_b = canonical_graph_signature(b.X, self._dims, self._catalog)
            b.set("canon_sig", sig_b)
        return sig_a == sig_b


# =============================================================================
# Callbacks
# =============================================================================


class ProgressCallback(Callback):
    """Log NSGA-II progress every N generations."""

    def __init__(self, every_n_gen: int = 10, total_inventory: int = 0) -> None:
        super().__init__()
        self.every_n_gen = every_n_gen
        self.total_inventory = max(1, total_inventory)

    def notify(self, algorithm) -> None:
        gen = algorithm.n_gen
        if gen % self.every_n_gen != 0:
            return

        logger = logging.getLogger(__name__)
        pop = algorithm.pop
        F = pop.get("F")
        G = pop.get("G")
        if F is None:
            return

        # Filter +inf rows (sentinel for empty chromosomes)
        finite = ~np.isinf(F).any(axis=1)
        if not finite.any():
            logger.info(f"Gen {gen:4d} | all individuals are sentinel +inf")
            return

        F_finite = F[finite]
        best_util = float(-np.min(F_finite[:, 0]))
        best_speed = float(-np.min(F_finite[:, 1]))
        pieces = int(best_util * self.total_inventory)

        n_feasible = 0
        if G is not None:
            n_feasible = int(np.sum(np.all(G <= 0, axis=1)))

        logger.info(
            f"Gen {gen:4d} | best_util={best_util:.1%} ({pieces} pcs) | "
            f"best_min_speed={best_speed:.2f} m/s | feasible={n_feasible}/{len(pop)}"
        )


class FeasibleEliteCallback(Callback):
    """Preserve the best-utilization feasible individual across generations."""

    def __init__(self) -> None:
        super().__init__()
        self._elite = None
        self._elite_util = -np.inf

    def notify(self, algorithm) -> None:
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

        current_best = (
            float(np.max(-F[feasible_mask, 0])) if np.any(feasible_mask) else -np.inf
        )
        if self._elite_util <= current_best:
            return

        if not np.all(feasible_mask):
            infeas_idx = np.where(~feasible_mask)[0]
            cv = np.maximum(G[infeas_idx], 0).sum(axis=1)
            slot = int(infeas_idx[int(np.argmax(cv))])
        else:
            slot = int(np.argmax(F[:, 0]))

        pop[slot] = copy.deepcopy(self._elite)


class LegoAdaptiveEpsilon(AdaptiveEpsilonConstraintHandling):
    """Three-phase epsilon schedule for constraint relaxation.

    Reused verbatim from V1 — encoding-agnostic.
    """

    def __init__(self, algorithm, hold_until: float = 0.2,
                 perc_eps_until: float = 0.7) -> None:
        super().__init__(algorithm, perc_eps_until=perc_eps_until)
        self.hold_until = hold_until
        self._logger = logging.getLogger(__name__)

    def _initialize_advance(self, infills=None, **kwargs):
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
        return super(AdaptiveEpsilonConstraintHandling, self)._initialize_advance(
            infills, **kwargs,
        )

    def _adapt_constraint_handling(self, config, **kwargs):
        t = self.termination.perc
        if t < self.hold_until:
            alpha = 1.0
        elif t < self.perc_eps_until:
            alpha = 1.0 - (t - self.hold_until) / (self.perc_eps_until - self.hold_until)
        else:
            alpha = 0.0
        config["cv_eps"] = alpha * self.max_cv


class CallbackChain(Callback):
    """Dispatch ``notify`` to a sequence of child callbacks."""

    def __init__(self, *callbacks: Callback) -> None:
        super().__init__()
        self._callbacks = callbacks

    def notify(self, algorithm) -> None:
        for cb in self._callbacks:
            cb.notify(algorithm)


# =============================================================================
# Run
# =============================================================================


def run_optimization(
    config: OptimizationConfig,
    catalog: TrackCatalog,
    verbose: bool = False,
    output_dir: Optional[Path] = None,
) -> object:
    """Run NSGA-II port-pair optimization with adaptive epsilon constraint handling."""
    logger = logging.getLogger(__name__)

    problem = PortPairProblem(catalog, config)
    dims = problem.dims

    sampler = PortPairSampling(
        dims, catalog, config,
        heuristic_ratio=config.algorithm.heuristic_ratio,
    )
    crossover = PortPairCrossover(dims, prob=config.algorithm.crossover_prob)
    mutation = PortPairMutation(dims, catalog, config, prob=config.algorithm.mutation_prob)
    repair = PortPairRepairPipeline(dims, catalog, config.inventory)

    # Topology-aware duplicate elimination: hash the canonical port-graph
    # signature instead of comparing raw int16 chromosomes, so anchor-shifted
    # or slot-permuted clones of the same layout collapse to a single bucket.
    if config.algorithm.eliminate_duplicates:
        eliminator = CanonicalGraphDuplicates(dims, catalog)
    else:
        eliminator = False

    base_algorithm = NSGA2(
        pop_size=config.algorithm.pop_size,
        sampling=sampler,
        crossover=crossover,
        mutation=mutation,
        repair=repair,
        survival=TopologyNichedSurvival(dims, catalog),
        eliminate_duplicates=eliminator,
    )

    algorithm = LegoAdaptiveEpsilon(
        base_algorithm,
        hold_until=0.2,
        perc_eps_until=0.9,
    )

    termination = get_termination("n_gen", config.algorithm.n_gen)

    chain = [FeasibleEliteCallback()]
    if verbose:
        chain.append(ProgressCallback(every_n_gen=10, total_inventory=config.total_inventory))
    callback = CallbackChain(*chain)

    logger.info("Starting NSGA-II port-pair track optimization...")
    logger.info(f"Population: {config.algorithm.pop_size}")
    logger.info(f"Generations: {config.algorithm.n_gen}")
    logger.info(
        f"Chromosome: {dims.n_var} genes "
        f"(N_max={dims.N_max}, E_max={dims.E_max})"
    )
    logger.info(f"Total inventory: {config.total_inventory} pieces")
    logger.info(f"Heuristic seeding: {config.algorithm.heuristic_ratio:.0%}")

    res = minimize(
        problem, algorithm, termination,
        verbose=False, save_history=False, callback=callback,
    )

    logger.info("Optimization complete!")

    if res.pop is not None:
        G = res.pop.get("G")
        if G is not None:
            n_feasible = np.sum(np.all(G <= 0, axis=1))
            logger.info(f"Feasible solutions: {n_feasible}/{len(res.pop)}")

    return res


# =============================================================================
# Result Saving
# =============================================================================


def save_results(
    res, output_dir: Path, catalog: TrackCatalog, config: OptimizationConfig,
) -> None:
    """Save chromosomes / fitness / constraints to CSV plus a Pareto plot."""
    logger = logging.getLogger(__name__)
    output_dir.mkdir(parents=True, exist_ok=True)

    if res.pop is None:
        logger.warning("No population to save")
        return

    X = res.pop.get("X")
    F = res.pop.get("F")
    G = res.pop.get("G")

    np.savetxt(output_dir / "chromosomes.csv", X, delimiter=",", fmt="%d")
    np.savetxt(
        output_dir / "fitness.csv", F, delimiter=",",
        header="neg_utilization,neg_min_speed", comments="",
    )

    if G is not None:
        constraint_header = (
            "closure_x,closure_y,closure_theta,boundary,collisions,"
            + ",".join(f"inv_{i}" for i in range(catalog.n_pieces))
            + ",loose_ports,cycle_count"
        )
        np.savetxt(
            output_dir / "constraints.csv", G, delimiter=",",
            header=constraint_header, comments="",
        )

    # Pareto plot
    feasible_mask = np.all(G <= 0, axis=1) if G is not None else np.ones(len(X), dtype=bool)
    finite = ~np.isinf(F).any(axis=1)
    keep = feasible_mask & finite

    fig, ax = plt.subplots(figsize=(8, 6))
    if finite.any():
        ax.scatter(-F[finite & ~feasible_mask, 0], -F[finite & ~feasible_mask, 1],
                   c="red", s=8, alpha=0.4, label="infeasible")
    if keep.any():
        ax.scatter(-F[keep, 0], -F[keep, 1],
                   c="green", s=20, label="feasible")
    ax.set_xlabel("Utilization")
    ax.set_ylabel("Min piece speed (m/s)")
    ax.set_title("Pareto Front: Utilization vs Min Speed (V2 port-pair)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "pareto_front.png", dpi=120)
    plt.close(fig)

    problem = PortPairProblem(catalog, config)

    # Best-feasible render
    if keep.any():
        feasible_indices = np.where(keep)[0]
        best_idx = feasible_indices[np.argmin(F[feasible_indices, 0])]
        graph = decode_chromosome(X[best_idx], problem.dims, catalog, problem.decoder_config)
        layout = port_graph_to_layout(graph, catalog)
        title = (
            f"Best Feasible ({graph.n_slots} pcs, "
            f"{-F[best_idx, 0]:.1%} util, "
            f"{-F[best_idx, 1]:.2f} m/s, "
            f"{graph.n_components} comp, {graph.n_cycles} cycle)"
        )
        plot_layout(layout, catalog, config.boundary, title,
                    save_path=output_dir / "best_layout.png")
        logger.info(
            f"Best feasible: {graph.n_slots} pieces, "
            f"util={-F[best_idx, 0]:.1%}, "
            f"min_speed={-F[best_idx, 1]:.2f} m/s, "
            f"cycles={graph.n_cycles}, components={graph.n_components}"
        )
    else:
        logger.warning("No feasible solutions found")

    # Best-infeasible render — best (highest util) infeasible solution, with CV
    if G is not None:
        infeasible_finite = (~feasible_mask) & finite
        if infeasible_finite.any():
            infeas_idx = np.where(infeasible_finite)[0]
            best_inf = infeas_idx[np.argmin(F[infeas_idx, 0])]
            cv = float(np.sum(np.maximum(0, G[best_inf])))
            graph_inf = decode_chromosome(
                X[best_inf], problem.dims, catalog, problem.decoder_config,
            )
            layout_inf = port_graph_to_layout(graph_inf, catalog)
            title = (
                f"Best Infeasible ({graph_inf.n_slots} pcs, "
                f"{-F[best_inf, 0]:.1%} util, "
                f"{-F[best_inf, 1]:.2f} m/s, "
                f"{graph_inf.n_components} comp, {graph_inf.n_cycles} cycle, "
                f"CV={cv:.2f})"
            )
            plot_layout(layout_inf, catalog, config.boundary, title,
                        save_path=output_dir / "best_infeasible.png")
            logger.info(
                f"Best infeasible: {graph_inf.n_slots} pieces, "
                f"util={-F[best_inf, 0]:.1%}, CV={cv:.2f}"
            )

    logger.info(f"Results saved to {output_dir}")
