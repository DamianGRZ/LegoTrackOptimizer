"""pymoo optimization problem definition for track layout optimization.

Single problem class with:
- ONE objective: maximize utilization (-utilization for minimization)
- Constraints via Deb's CV (normalized to same scale)
- Decoder as single source of truth for layout evaluation
"""

from typing import Dict, List

import numpy as np
from numpy.typing import NDArray
from pymoo.core.problem import ElementwiseProblem

from .config import OptimizationConfig
from .data import TrackCatalog
from .decoder import DecoderConfig, decode_chromosome
from .encoding import N_VAR, generate_bounds


class TrackOptimizationProblem(ElementwiseProblem):
    """Single-objective track layout optimization.

    Objective: Maximize piece utilization (-utilization for pymoo minimization)
    Constraints: 5 normalized constraints via Deb's CV rules
        G[0]: closure error (position)
        G[1]: angle error
        G[2]: boundary violation
        G[3]: inventory violation
        G[4]: loose port count (switches/crossings must have all ports connected)

    Deb's Rules (pymoo default):
    1. Feasible solutions always beat infeasible
    2. Among infeasible, lower total CV wins
    3. Among feasible, better objective wins
    """

    def __init__(
        self,
        catalog: TrackCatalog,
        config: OptimizationConfig,
        closure_tolerance: float = None,
        angle_tolerance: float = None,
        **kwargs,
    ):
        """Initialize track optimization problem.

        Args:
            catalog: Track catalog with piece properties
            config: Optimization configuration
            closure_tolerance: Position closure tolerance in studs (default from config)
            angle_tolerance: Angle closure tolerance in degrees (default from config)
            **kwargs: Additional arguments passed to ElementwiseProblem
        """
        # Use config tolerances or provided values
        self.closure_tolerance = closure_tolerance or config.closure_tolerance
        self.angle_tolerance = angle_tolerance or config.angle_tolerance

        # Generate segment-specific bounds
        bounds = generate_bounds(
            max_piece_index=catalog._max_index,
            boundary_min_x=config.boundary.min_x,
            boundary_max_x=config.boundary.max_x,
            boundary_min_y=config.boundary.min_y,
            boundary_max_y=config.boundary.max_y,
        )

        # Compute boundary diagonal for normalization
        self.diagonal = np.sqrt(
            (config.boundary.max_x - config.boundary.min_x) ** 2
            + (config.boundary.max_y - config.boundary.min_y) ** 2
        )

        super().__init__(
            n_var=N_VAR,
            n_obj=1,           # ONE objective: -utilization
            n_ieq_constr=5,    # 5 constraints via Deb's CV (includes loose ports)
            xl=bounds.xl,
            xu=bounds.xu,
            **kwargs,
        )

        self.catalog = catalog
        self.config = config
        self.total_inventory = sum(config.inventory.values())

        # Convert inventory to index-based for decoder
        self.inventory_by_index = self._convert_inventory(config.inventory)

        # Decoder configuration with boundary for RK position scaling
        self.decoder_config = DecoderConfig(
            position_tolerance=self.closure_tolerance,
            angle_tolerance=self.angle_tolerance,
            boundary_min_x=config.boundary.min_x,
            boundary_max_x=config.boundary.max_x,
            boundary_min_y=config.boundary.min_y,
            boundary_max_y=config.boundary.max_y,
        )

    def _convert_inventory(self, inventory: Dict[str, int]) -> Dict[int, int]:
        """Convert inventory from piece_id to piece_index."""
        result = {}
        for piece_id, count in inventory.items():
            idx = self.catalog._id_to_index.get(piece_id)
            if idx is not None:
                result[idx] = count
        return result

    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate single chromosome.

        Decoder is the single source of truth - if it produces a layout, evaluate it.

        Args:
            x: Chromosome array (1D, length N_VAR)
            out: Dictionary to populate with "F" and "G"
        """
        # Decoder is the single source of truth
        layout = decode_chromosome(
            x,
            self.catalog,
            self.config.inventory,
            self.decoder_config,
        )

        # Handle empty layouts
        if layout.n_pieces == 0:
            out["F"] = [0.0]
            out["G"] = [1.0, 1.0, 1.0, 1.0, 1.0]  # All 5 constraints violated
            return

        # ONE objective: maximize utilization
        utilization = layout.n_pieces / self.total_inventory
        out["F"] = [-utilization]  # Minimize negative = maximize

        # Constraints normalized to same scale (Deb's CV)
        # G[i] <= 0 means constraint satisfied
        g_closure = (layout.max_closure_error - self.closure_tolerance) / self.closure_tolerance
        g_angle = (layout.max_angle_error - self.angle_tolerance) / self.angle_tolerance
        g_boundary = self._compute_boundary_violation(layout) / max(self.diagonal, 1.0)
        g_inventory = float(self._compute_inventory_violation(layout))
        g_loose_ports = float(layout.loose_port_count)  # Must be 0 for feasibility

        out["G"] = [g_closure, g_angle, g_boundary, g_inventory, g_loose_ports]

    def _compute_boundary_violation(self, layout) -> float:
        """Compute max boundary violation across all paths.

        Args:
            layout: Decoded layout object.

        Returns:
            Maximum violation distance in studs.
        """
        max_violation = 0.0
        boundary = self.config.boundary

        for path in layout.paths:
            if len(path.states) == 0:
                continue

            x = path.states[:, 0]
            y = path.states[:, 1]

            path_max = max(
                np.max(np.maximum(0, boundary.min_x - x)),
                np.max(np.maximum(0, x - boundary.max_x)),
                np.max(np.maximum(0, boundary.min_y - y)),
                np.max(np.maximum(0, y - boundary.max_y)),
            )
            max_violation = max(max_violation, path_max)

        return float(max_violation)

    def _compute_inventory_violation(self, layout) -> int:
        """Compute inventory violation from layout pieces.

        Args:
            layout: Decoded layout object.

        Returns:
            Total count of pieces used beyond available inventory.
        """
        # Count pieces used in the layout
        piece_counts: Dict[int, int] = {}

        # Count main loop pieces
        for piece_idx in layout.main_loop_pieces:
            if piece_idx >= 0:
                piece_counts[piece_idx] = piece_counts.get(piece_idx, 0) + 1

        # Count branch pieces from switch pairs
        for switch_pair in layout.switch_pairs:
            for piece_idx in switch_pair.branch_pieces:
                if piece_idx >= 0:
                    piece_counts[piece_idx] = piece_counts.get(piece_idx, 0) + 1

        # Compute violation (excess usage)
        total_violation = 0
        for piece_idx, used in piece_counts.items():
            available = self.inventory_by_index.get(piece_idx, 0)
            if used > available:
                total_violation += used - available

        return total_violation


# =============================================================================
# Epsilon-Tightening Callback
# =============================================================================

from pymoo.core.callback import Callback


class EpsilonTightening(Callback):
    """Tighten closure tolerance over generations.

    Starts with loose tolerance to allow broad exploration, then tightens
    to force convergence to tight closure.

    Schedule:
    - Early: closure_tolerance = initial_tolerance (explore broadly)
    - Late: closure_tolerance = final_tolerance (converge tightly)
    - Linear interpolation between initial and final until tighten_until

    Example:
        callback = EpsilonTightening(initial_tol=8.0, final_tol=0.5, tighten_until=0.7)
        result = minimize(problem, algorithm, callback=callback)
    """

    def __init__(
        self,
        initial_tol: float = 8.0,
        final_tol: float = 0.5,
        tighten_until: float = 0.7,
    ):
        """Initialize epsilon-tightening callback.

        Args:
            initial_tol: Starting closure tolerance in studs (loose).
            final_tol: Final closure tolerance in studs (tight).
            tighten_until: Progress fraction (0-1) at which tightening completes.
        """
        super().__init__()
        self.initial_tol = initial_tol
        self.final_tol = final_tol
        self.tighten_until = tighten_until

    def notify(self, algorithm):
        """Called after each generation.

        Args:
            algorithm: The pymoo algorithm instance.
        """
        # Get progress through optimization
        n_gen = algorithm.n_gen
        n_max_gen = getattr(algorithm.termination, 'n_max_gen', None)

        if n_max_gen is None:
            # Termination doesn't have n_max_gen, skip tightening
            return

        progress = n_gen / n_max_gen

        # Only tighten until tighten_until fraction
        if progress < self.tighten_until:
            # Linear interpolation
            t = progress / self.tighten_until
            tolerance = self.initial_tol * (1 - t) + self.final_tol * t
        else:
            # Past tighten_until, keep at final tolerance
            tolerance = self.final_tol

        # Update problem tolerance
        if hasattr(algorithm, 'problem'):
            algorithm.problem.closure_tolerance = tolerance


# =============================================================================
# Convergence Tracker Callback
# =============================================================================


class ConvergenceTracker(Callback):
    """Track optimization convergence metrics per generation.

    Records best objective value and feasibility percentage at each generation.
    Based on research best practices for monitoring GA progress.

    Attributes:
        data: Dictionary with tracked metrics:
            - 'generation': List of generation numbers
            - 'best_f': List of best objective values (negated for maximization)
            - 'feasible_pct': List of feasibility percentages (0-1)
            - 'n_pieces': List of best piece counts
            - 'avg_cv': List of average constraint violations

    Example:
        tracker = ConvergenceTracker()
        result = minimize(problem, algorithm, callback=tracker)
        # Access data after optimization:
        print(tracker.data['best_f'])
        tracker.plot_convergence()  # Optional visualization
    """

    def __init__(self):
        """Initialize convergence tracker."""
        super().__init__()
        self.data: Dict[str, List] = {
            "generation": [],
            "best_f": [],
            "feasible_pct": [],
            "n_pieces": [],
            "avg_cv": [],
        }

    def notify(self, algorithm):
        """Called after each generation to record metrics.

        Args:
            algorithm: The pymoo algorithm instance.
        """
        pop = algorithm.pop
        n_gen = algorithm.n_gen

        # Get objective and constraint values
        F = pop.get("F")
        G = pop.get("G")

        if F is None or len(F) == 0:
            return

        # Best objective (F is negated utilization, so min is best)
        best_f = float(np.min(F[:, 0]))

        # Best piece count (F[0] = -utilization = -n_pieces/total)
        # Assuming we can access problem's total_inventory
        if hasattr(algorithm, 'problem') and hasattr(algorithm.problem, 'total_inventory'):
            total = algorithm.problem.total_inventory
            best_pieces = int(-best_f * total)
        else:
            best_pieces = 0

        # Feasibility percentage
        if G is not None and len(G) > 0:
            feasible = np.all(G <= 0, axis=1)
            feasible_pct = float(np.mean(feasible))
            avg_cv = float(np.mean(np.maximum(0, G).sum(axis=1)))
        else:
            feasible_pct = 1.0
            avg_cv = 0.0

        # Record metrics
        self.data["generation"].append(n_gen)
        self.data["best_f"].append(best_f)
        self.data["feasible_pct"].append(feasible_pct)
        self.data["n_pieces"].append(best_pieces)
        self.data["avg_cv"].append(avg_cv)

    def get_summary(self) -> Dict:
        """Get summary statistics from tracked data.

        Returns:
            Dictionary with summary metrics.
        """
        if not self.data["generation"]:
            return {}

        return {
            "final_generation": self.data["generation"][-1],
            "best_objective": min(self.data["best_f"]),
            "final_feasible_pct": self.data["feasible_pct"][-1],
            "max_feasible_pct": max(self.data["feasible_pct"]),
            "best_n_pieces": max(self.data["n_pieces"]),
            "convergence_generation": self._find_convergence_gen(),
        }

    def _find_convergence_gen(self, window: int = 10, threshold: float = 0.001) -> int:
        """Find generation where optimization converged.

        Convergence defined as when best_f stops improving significantly.

        Args:
            window: Number of generations to check for improvement.
            threshold: Minimum improvement to consider as progress.

        Returns:
            Generation number where convergence was detected, or -1 if not converged.
        """
        if len(self.data["best_f"]) < window:
            return -1

        for i in range(window, len(self.data["best_f"])):
            recent = self.data["best_f"][i - window:i]
            improvement = abs(max(recent) - min(recent))
            if improvement < threshold:
                return self.data["generation"][i - window]

        return -1


# =============================================================================
# Aliases for backward compatibility (will be removed in future)
# =============================================================================

# These are kept temporarily so existing code doesn't break immediately
MultiSegmentProblem = TrackOptimizationProblem
SingleObjectiveProblem = TrackOptimizationProblem
TrackLayoutProblem = TrackOptimizationProblem
