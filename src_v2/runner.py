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
from multiprocessing import Pool
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.constraints.eps import AdaptiveEpsilonConstraintHandling
from pymoo.core.callback import Callback
from pymoo.parallelization import StarmapParallelization
from pymoo.operators.survival.rank_and_crowding import ConstrRankAndCrowding
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultMultiObjectiveTermination

from .alns_callback import ALNSCallback
from .canonical import PortGraphDuplicateElimination
from .catalog import TrackCatalog
from .config import OptimizationConfig
from .decoder import decode_chromosome, port_graph_to_layout
from .epsilon_archive import EpsilonArchiveCallback
from .finalization_callback import FinalizationGatingCallback
from .instrumentation import DiagnosticsCallback
from .operators import (
    JunctionCrossover,
    PortPairAndJunctionCrossover,
    PortPairCrossover,
    PortPairMutation,
    PortPairSampling,
)
from .phenotype_dedupe import PhenotypeDedupeCallback
from .problem import PortPairProblem
from .repair import PortPairRepairPipeline
from .snapshot_callback import SnapshotCallback
from .types import PieceClass
from .visualization import plot_layout


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
    """[DEPRECATED — replaced by EpsilonArchiveCallback in Phase 5; the
    archive's well-spread Pareto set obviates the single best-util elite.]

    Preserve the best-utilization feasible individual across generations."""

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
    seed: Optional[int] = None,
) -> object:
    """Run NSGA-II port-pair optimization with adaptive epsilon constraint handling."""
    logger = logging.getLogger(__name__)

    train_config = config.load_train_config()
    logger.info(
        f"Train physics: v_motor_max={train_config.v_motor_max:.2f} m/s, "
        f"mass_total={train_config.mass_total:.3f} kg, "
        f"coupler_offset={train_config.coupler_offset:.3f} m, "
        f"max_accel={train_config.max_accel:.2f} m/s^2"
    )

    n_workers = max(1, int(getattr(config, "n_workers", 1)))
    if n_workers > 1:
        pool = Pool(n_workers)
        runner = StarmapParallelization(pool.starmap)
        problem = PortPairProblem(
            catalog, config, train_config=train_config, elementwise_runner=runner,
        )
        logger.info(f"Parallel evaluation: {n_workers} processes")
    else:
        pool = None
        problem = PortPairProblem(catalog, config, train_config=train_config)
    dims = problem.dims

    sampler = PortPairSampling(
        dims, catalog, config,
        heuristic_ratio=config.algorithm.heuristic_ratio,
        seed=getattr(config.algorithm, "seed", None),
    )
    # Phase 4 (Coupling A): port-pair crossover and junction crossover are
    # separate operators with independent probability knobs. The composite
    # passes a single Crossover to NSGA2 while delegating per-mating gating
    # to each component.
    crossover = PortPairAndJunctionCrossover(
        PortPairCrossover(dims, prob=config.algorithm.crossover_prob),
        JunctionCrossover(
            dims, catalog,
            prob=config.algorithm.junction_crossover_prob,
        ),
    )
    mutation = PortPairMutation(
        dims, catalog, config,
        prob=config.algorithm.mutation_prob,
        seed=getattr(config.algorithm, "seed", None),
    )
    repair = PortPairRepairPipeline(dims, catalog, config.inventory)

    if config.algorithm.eliminate_duplicates:
        eliminate_duplicates = PortGraphDuplicateElimination(
            problem.dims, catalog, problem.decoder_config,
            inventory=config.inventory,  # Phase 5a: materialize before hashing
        )
    else:
        eliminate_duplicates = False

    base_algorithm = NSGA2(
        pop_size=config.algorithm.pop_size,
        sampling=sampler,
        crossover=crossover,
        mutation=mutation,
        repair=repair,
        survival=ConstrRankAndCrowding(),
        eliminate_duplicates=eliminate_duplicates,
    )

    # Adaptive epsilon disabled for the strict-constraint regime: with
    # ``G[9+T]`` (single-component) and ``G[10+T]`` (loose-port ratio)
    # active, the relaxation lets multi-component "epsilon-feasible"
    # individuals dominate truly-feasible single-component layouts in
    # NSGA-II survival because both ride at effective CV=0 inside the
    # epsilon band but the multi-component side has more pieces wired
    # into side cycles. ``ConstrRankAndCrowding`` alone gives feasible
    # strict priority over infeasible.
    algorithm = base_algorithm

    termination = DefaultMultiObjectiveTermination(
        xtol=1.0,        # disable: integer chromosomes; xtol semantically meaningless
        cvtol=1e-4,      # widened from 1e-8 (Bezerra 2019); avoids false-trigger on
                         # numerical plateau noise during early infeas
        ftol=0.01,       # 1% rel-HV; loosened from 0.005 default for constrained MOPs
                         # (Wagner-Bringmann GECCO 2013)
        period=50,       # default; fine for n_max_gen=500
        n_skip=5,        # default
        n_max_gen=config.algorithm.n_gen,   # per-config; 200/500 typical
    )

    eps_archive_path = (
        Path(output_dir) / "epsilon_archive.json" if output_dir is not None
        else Path("outputs_v2/_default") / "epsilon_archive.json"
    )
    eps_eps = getattr(config, "epsilon_archive_eps", (0.005, 0.01))
    eps_max = int(getattr(config, "epsilon_archive_max", 200))
    epsilon_archive = EpsilonArchiveCallback(
        epsilon=tuple(eps_eps),
        max_size=eps_max,
        output_path=eps_archive_path,
        dims=dims,
        catalog=catalog,
        cv_admission_threshold=getattr(config, "cv_admission_threshold", 1.0),
    )
    alns = ALNSCallback()
    alns.attach_to(mutation)
    dedupe = PhenotypeDedupeCallback(sampler)
    piece_classes = catalog.classify_pieces()
    switch_piece_indices = piece_classes.get(PieceClass.SWITCH_3PORT, [])
    crossing_piece_indices = piece_classes.get(PieceClass.CROSSING_4PORT, [])
    diagnostics = DiagnosticsCallback(
        output_dir=eps_archive_path.parent,
        n_constraints=11 + catalog.n_pieces,
        n_max=dims.N_max,
        switch_piece_indices=switch_piece_indices,
        crossing_piece_indices=crossing_piece_indices,
        dedupe_callback=dedupe,
    )
    snapshot_cb = SnapshotCallback(
        n_gen=config.algorithm.n_gen,
        output_dir=eps_archive_path.parent,
        problem=problem,
        catalog=catalog,
        config=config,
        n_snapshots=10,
    )

    # FinalizationGatingCallback FIRST so repair.finalization_active is
    # synchronized before downstream callbacks consult it. ALNS / archive /
    # dedupe replace the legacy FeasibleEliteCallback; the epsilon archive
    # itself preserves a well-spread Pareto set across generations.
    chain = [
        FinalizationGatingCallback(repair, threshold=0.9),
        alns,
        epsilon_archive,
        dedupe,        # runs first so its ``last_*`` stats are fresh ...
        diagnostics,   # ... when DiagnosticsCallback reads dedupe_rejection_rate
        snapshot_cb,   # writes 10 ordered PNG + .npy at scheduled gens
    ]
    if verbose:
        chain.append(ProgressCallback(every_n_gen=5, total_inventory=config.total_inventory))
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

    try:
        minimize_kwargs = dict(
            verbose=False, save_history=False, callback=callback,
        )
        if seed is not None:
            minimize_kwargs["seed"] = seed
        res = minimize(problem, algorithm, termination, **minimize_kwargs)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    epsilon_archive.finalize()
    snapshot_cb.finalize()
    logger.info(
        f"Epsilon archive: {len(epsilon_archive.archive)} feasible "
        f"non-dominated entries written to {eps_archive_path.name}",
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

    # Stamp the encoding version on every artifact bundle (Rule 13).
    import json as _json
    from .encoding import ENCODING_VERSION
    (output_dir / "meta.json").write_text(_json.dumps({
        "encoding_version": ENCODING_VERSION,
        "config_inventory": dict(config.inventory),
        "n_pop": int(F.shape[0]) if F is not None else 0,
    }, indent=2), encoding="utf-8")

    np.savetxt(output_dir / "chromosomes.csv", X, delimiter=",", fmt="%d")
    np.savetxt(
        output_dir / "fitness.csv", F, delimiter=",",
        header="neg_utilization,neg_min_speed", comments="",
    )

    if G is not None:
        constraint_header = (
            "closure_x,closure_y,closure_theta,boundary,collisions,"
            + ",".join(f"inv_{i}" for i in range(catalog.n_pieces))
            + ",incomplete_switch_ratio,incomplete_crossing_ratio,cycle_count,"
            + "branch_cycle_deficit,multi_component,loose_port_ratio"
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
