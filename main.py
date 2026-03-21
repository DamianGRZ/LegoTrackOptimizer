"""Main entry point for LEGO Track Optimizer.

Single-objective optimization to maximize piece utilization with Deb's CV constraints.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.callback import Callback
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from src.config import OptimizationConfig
from src.data import TrackCatalog
from src.decoder import decode_chromosome
from src.encoding import N_VAR
from src.sampling import INDEX_TO_ID
from src.operators import create_crossover_operator, create_mutation_operator
from src.problem import TrackOptimizationProblem, EpsilonTightening, StagnationCallback
from src.repair import TrackRepairPipeline
from src.sampling import MultiSegmentSampling
from src.survival import StructuralNichingSurvival
from src.visualization import plot_layout, plot_multi_path_layout


def setup_logging(verbose: bool = False) -> None:
    """Configure logging level.

    Args:
        verbose: If True, set to DEBUG level, otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def log_piece_usage(
    layout,
    inventory: dict,
    catalog: TrackCatalog,
    logger: logging.Logger,
) -> None:
    """Log piece usage breakdown by type from decoded layout.

    Args:
        layout: Decoded MultiPathLayout with main_loop_pieces and switch_pairs.
        inventory: Available inventory {piece_id: count}.
        catalog: Track catalog for index-to-ID mapping.
        logger: Logger instance.
    """
    # Count pieces from decoded layout (not raw chromosome)
    used_by_index = {}

    # Count main loop pieces
    if hasattr(layout, 'main_loop_pieces'):
        for piece_idx in layout.main_loop_pieces:
            idx = int(piece_idx)
            if idx >= 0:
                used_by_index[idx] = used_by_index.get(idx, 0) + 1

    # Count branch pieces from switch pairs
    if hasattr(layout, 'switch_pairs'):
        for switch_pair in layout.switch_pairs:
            for piece_idx in switch_pair.branch_pieces:
                idx = int(piece_idx)
                if idx >= 0:
                    used_by_index[idx] = used_by_index.get(idx, 0) + 1

    # Convert inventory to index-based for comparison
    inventory_by_index = {}
    for piece_id, count in inventory.items():
        idx = catalog._id_to_index.get(piece_id)
        if idx is not None:
            inventory_by_index[idx] = count

    # Log header
    logger.info("Piece usage breakdown:")

    # Log each piece type that exists in inventory
    for idx in sorted(inventory_by_index.keys()):
        piece_id = INDEX_TO_ID.get(idx, f"PIECE_{idx}")
        available = inventory_by_index.get(idx, 0)
        used = used_by_index.get(idx, 0)
        # Shorten piece names for cleaner output
        short_name = piece_id.replace("R40_SWITCH_", "SW_").replace("STRAIGHT_", "STR_").replace("R40_", "")
        logger.info(f"  {short_name:16s}: {used:3d}/{available:3d}")


# =============================================================================
# Progress Callback
# =============================================================================

class ProgressCallback(Callback):
    """Callback for logging optimization progress every N generations."""

    def __init__(
        self,
        every_n_gen: int = 10,
        single_objective: bool = False,
        total_inventory: int = 0,
    ):
        """Initialize callback.

        Args:
            every_n_gen: Log progress every N generations.
            single_objective: Whether using single-objective mode.
            total_inventory: Total number of pieces available in inventory.
        """
        super().__init__()
        self.every_n_gen = every_n_gen
        self.single_objective = single_objective
        self.total_inventory = total_inventory
        self.logger = logging.getLogger(__name__)

    def notify(self, algorithm):
        """Called after each generation."""
        n_gen = algorithm.n_gen

        if n_gen == 1 or n_gen % self.every_n_gen == 0:
            pop = algorithm.pop
            F = pop.get("F")
            G = pop.get("G")

            if F is not None and len(F) > 0:
                # Count feasible solutions
                n_feasible = 0
                if G is not None:
                    feasible_mask = np.all(G <= 0, axis=1)
                    n_feasible = np.sum(feasible_mask)

                if self.single_objective:
                    # Single objective: F is (n, 1), -utilization (higher utilization = more negative)
                    best_utilization = -np.min(F[:, 0])
                    best_pieces = int(best_utilization * self.total_inventory) if self.total_inventory > 0 else 0
                    pieces_str = f"{best_pieces}/{self.total_inventory}" if self.total_inventory > 0 else f"{best_pieces}"
                    self.logger.info(
                        f"Gen {n_gen:4d} | "
                        f"Feasible: {n_feasible:4d}/{len(F)} | "
                        f"Pieces: {pieces_str} | "
                        f"Util: {best_utilization:.1%}"
                    )
                else:
                    # Bi-objective: F is (n, 2)
                    best_utilization = -np.min(F[:, 0])
                    best_speed = -np.min(F[:, 1])
                    # Compute best pieces from utilization
                    best_pieces = int(best_utilization * self.total_inventory) if self.total_inventory > 0 else 0
                    pieces_str = f"{best_pieces}/{self.total_inventory}" if self.total_inventory > 0 else "N/A"
                    self.logger.info(
                        f"Gen {n_gen:4d} | "
                        f"Feasible: {n_feasible:4d}/{len(F)} | "
                        f"Pieces: {pieces_str} | "
                        f"Best Speed: {best_speed:.3f} m/s"
                    )


# =============================================================================
# Optimization Runner
# =============================================================================

def run_optimization(
    config: OptimizationConfig,
    catalog: TrackCatalog,
    verbose: bool = False,
    use_epsilon_tightening: bool = True,
    initial_tolerance: float = 8.0,
    final_tolerance: float = 0.5,
    tighten_until: float = 0.7,
    complex_ratio: float = 0.33,
) -> object:
    """Run single-objective track optimization.

    Uses TrackOptimizationProblem with:
    - ONE objective: maximize piece utilization
    - 4 constraints via Deb's CV rules (closure, angle, boundary, inventory)
    - Optional epsilon-tightening for closure tolerance

    Args:
        config: Optimization configuration.
        catalog: Track catalog.
        verbose: If True, log progress every 10 generations.
        use_epsilon_tightening: Whether to use epsilon-tightening for closure.
        initial_tolerance: Starting closure tolerance in studs (loose).
        final_tolerance: Final closure tolerance in studs (tight).
        tighten_until: Fraction of generations at which tightening completes.
        complex_ratio: Minimum fraction of survivors with complex pieces.

    Returns:
        pymoo Result object.
    """
    logger = logging.getLogger(__name__)

    # Create problem with initial tolerance
    start_tolerance = initial_tolerance if use_epsilon_tightening else config.closure_tolerance
    problem = TrackOptimizationProblem(
        catalog, config,
        closure_tolerance=start_tolerance,
        angle_tolerance=config.angle_tolerance,
    )

    # Create sampler (needed for available_pieces list)
    sampler = MultiSegmentSampling(catalog, config)

    # Create structural niching survival to preserve complex solutions
    # Pass available_pieces for proper RK decoding
    survival = StructuralNichingSurvival(
        complex_ratio=complex_ratio,
        prefer_balanced_switches=True,
        available_pieces=sampler.valid_pieces,
    )

    # Create operators
    crossover = create_crossover_operator(
        catalog, config.inventory,
        prob=config.algorithm.crossover_prob,
    )
    mutation = create_mutation_operator(catalog, config.inventory)

    # Create repair pipeline (now RK-aware using decode-modify-re-encode)
    repair = TrackRepairPipeline(
        catalog, config.inventory,
        enable_closure_repair=True,
        enable_switch_repair=True,
        prefer_add_switches=True,
    )

    # Create GA algorithm
    algorithm = GA(
        pop_size=config.algorithm.pop_size,
        sampling=sampler,
        crossover=crossover,
        mutation=mutation,
        repair=repair,
        survival=survival,
        eliminate_duplicates=True,
    )

    termination = get_termination("n_gen", config.algorithm.n_gen)

    # Build callback list
    callbacks = []

    if verbose:
        callbacks.append(ProgressCallback(
            every_n_gen=10,
            single_objective=True,
            total_inventory=config.total_inventory,
        ))

    if use_epsilon_tightening:
        callbacks.append(EpsilonTightening(
            initial_tol=initial_tolerance,
            final_tol=final_tolerance,
            hold_until=0.4,
            tighten_until=tighten_until,
        ))

    # Always add stagnation detection
    callbacks.append(StagnationCallback(patience=50, inject_ratio=0.10))

    # Log optimization parameters
    logger.info("Starting track optimization...")
    logger.info(f"Population size: {config.algorithm.pop_size}")
    logger.info(f"Generations: {config.algorithm.n_gen}")
    logger.info(f"Chromosome length: {N_VAR} genes")
    logger.info(f"Total inventory: {config.total_inventory} pieces")
    logger.info("Objective: maximize piece utilization")
    logger.info("Constraints: closure, angle, boundary, inventory, loose_ports (Deb's CV)")
    logger.info(f"Structural niching: {complex_ratio*100:.0f}% complex pieces preserved")
    if use_epsilon_tightening:
        tighten_gen = int(config.algorithm.n_gen * tighten_until)
        logger.info(f"Epsilon-tightening: {initial_tolerance:.1f} -> {final_tolerance:.1f} studs (until gen {tighten_gen})")

    # Combine callbacks
    if len(callbacks) == 1:
        callback = callbacks[0]
    elif len(callbacks) > 1:
        # Create a composite callback
        class CompositeCallback(Callback):
            def __init__(self, cbs):
                super().__init__()
                self.callbacks = cbs

            def notify(self, algorithm):
                for cb in self.callbacks:
                    cb.notify(algorithm)

        callback = CompositeCallback(callbacks)
    else:
        callback = None

    # Run optimization
    res = minimize(
        problem, algorithm, termination,
        callback=callback, verbose=False, save_history=True,
    )

    logger.info("Optimization complete!")

    # Count feasible solutions and log piece usage
    if res.pop is not None:
        G = res.pop.get("G")
        F = res.pop.get("F")
        X = res.pop.get("X")
        if G is not None:
            n_feasible = np.sum(np.all(G <= 0, axis=1))
            logger.info(f"Feasible solutions: {n_feasible}/{len(res.pop)}")

        # Log piece usage for best feasible solution
        if X is not None and len(X) > 0:
            best_idx = 0
            if G is not None and F is not None:
                feasible_mask = np.all(G <= 0, axis=1)
                if np.any(feasible_mask):
                    feasible_F = F[feasible_mask]
                    feasible_indices = np.where(feasible_mask)[0]
                    best_idx = feasible_indices[np.argmin(feasible_F[:, 0])]
            # Decode chromosome to get actual layout for accurate piece counting
            best_layout = decode_chromosome(X[best_idx], catalog, config.inventory)
            log_piece_usage(best_layout, config.inventory, catalog, logger)

    return res


# =============================================================================
# Result Saving
# =============================================================================

def save_results(
    res: object,
    output_dir: Path,
    catalog: TrackCatalog,
    config: OptimizationConfig,
) -> None:
    """Save optimization results.

    Args:
        res: pymoo Result object.
        output_dir: Output directory path.
        catalog: Track catalog.
        config: Optimization configuration.
    """
    logger = logging.getLogger(__name__)
    output_dir.mkdir(parents=True, exist_ok=True)

    if res.pop is None:
        logger.warning("No population to save")
        return

    # Get population data
    X = res.pop.get("X")
    F = res.pop.get("F")
    G = res.pop.get("G")

    # Save all solutions
    np.savetxt(output_dir / "chromosomes.csv", X, delimiter=",", fmt="%.4f")
    np.savetxt(output_dir / "fitness.csv", F, delimiter=",")
    if G is not None:
        np.savetxt(output_dir / "constraints.csv", G, delimiter=",")

    # Find best feasible solution
    feasible_mask = np.all(G <= 0, axis=1) if G is not None else np.ones(len(X), dtype=bool)
    feasible_indices = np.where(feasible_mask)[0]

    if len(feasible_indices) > 0:
        # Best feasible (highest utilization = most negative F)
        best_idx = feasible_indices[np.argmin(F[feasible_indices, 0])]
        best_chromosome = X[best_idx]
        best_utilization = -F[best_idx, 0]

        logger.info(f"Best feasible utilization: {best_utilization:.2%}")

        # Decode and plot best layout
        layout = decode_chromosome(best_chromosome, catalog, config.inventory)
        if layout.n_switch_pairs > 0 and layout.n_paths > 1:
            # Use multi-path plot to show all traversal paths including branch
            plot_multi_path_layout(
                layout, catalog, config.boundary,
                f"Best Layout ({layout.n_pieces} pieces, {best_utilization:.1%} util)",
                output_dir / "best_layout.png"
            )
            logger.info(f"Best layout: {layout.n_pieces} pieces, {layout.n_switch_pairs} switch pairs, {layout.n_paths} paths")
        else:
            plot_layout(
                layout, catalog, config.boundary,
                f"Best Layout ({layout.n_pieces} pieces, {best_utilization:.1%} util)",
                output_dir / "best_layout.png"
            )
            logger.info(f"Best layout: {layout.n_pieces} pieces, closure={layout.closure_error:.2f}")
    else:
        # No feasible solutions - plot best infeasible
        best_idx = np.argmin(F[:, 0])
        best_chromosome = X[best_idx]
        layout = decode_chromosome(best_chromosome, catalog, config.inventory)
        plot_layout(layout, catalog, config.boundary, "Best Layout (infeasible)", output_dir / "best_layout.png")
        logger.warning("No feasible solutions - saved best infeasible")

    logger.info(f"Results saved to {output_dir}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="LEGO Track Optimizer - Single-objective track layout optimization"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs",
        help="Output directory for results (default: outputs)",
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run quick test (20 generations, pop_size=20)",
    )
    parser.add_argument(
        "--no-epsilon-tightening",
        action="store_true",
        help="Disable epsilon-tightening for closure tolerance",
    )
    parser.add_argument(
        "--initial-tolerance",
        type=float,
        default=8.0,
        help="Initial closure tolerance in studs (default: 8.0)",
    )
    parser.add_argument(
        "--final-tolerance",
        type=float,
        default=1.0,
        help="Final closure tolerance in studs (default: 1.0)",
    )
    parser.add_argument(
        "--tighten-until",
        type=float,
        default=0.7,
        help="Fraction of generations at which tightening completes (default: 0.7)",
    )
    parser.add_argument(
        "--complex-ratio",
        type=float,
        default=0.33,
        help="Minimum fraction of survivors with complex pieces (switches, crossings). Default: 0.33",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Load configuration and catalog
    logger.info(f"Loading configuration from {args.config}")
    config = OptimizationConfig.load(args.config)

    # Override for quick test
    if args.quick_test:
        config.algorithm.n_gen = 20
        config.algorithm.pop_size = 20
        logger.info("Quick test mode: 20 generations, pop_size=20")

    logger.info("Loading track catalog from data/track_pieces.yaml")
    catalog = TrackCatalog.load("data/track_pieces.yaml")

    # Run optimization
    output_dir = Path(args.output)

    res = run_optimization(
        config, catalog,
        verbose=args.verbose,
        use_epsilon_tightening=not args.no_epsilon_tightening,
        initial_tolerance=args.initial_tolerance,
        final_tolerance=args.final_tolerance,
        tighten_until=args.tighten_until,
        complex_ratio=args.complex_ratio,
    )
    save_results(res, output_dir, catalog, config)

    logger.info("Done!")


if __name__ == "__main__":
    main()
