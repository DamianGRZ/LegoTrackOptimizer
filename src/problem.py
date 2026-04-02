"""pymoo optimization problem for multi-objective track layout optimization.

Bi-objective NSGA-II with Deb's constraint handling:
- F[0] = -utilization (maximize piece usage)
- F[1] = -avg_speed (maximize safe train speed)
- 5 inequality constraints via Deb's CV rules
"""

from typing import Dict

import numpy as np
from pymoo.core.problem import ElementwiseProblem

from .config import OptimizationConfig
from .data import TrackCatalog
from .decoder import DecoderConfig, decode_chromosome
from .encoding import compute_dimensions, generate_bounds
from .evaluation import compute_speed_profile
from .intersection import count_path_crossings, count_segment_crossings


class TrackOptimizationProblem(ElementwiseProblem):
    """Bi-objective track layout optimization with NSGA-II.

    Objectives (both minimized for pymoo):
        F[0] = -utilization  (maximize piece usage)
        F[1] = -avg_speed    (maximize safe train speed)

    Constraints (g <= 0 feasible, Deb's CV rules):
        G[0]: closure error (position)
        G[1]: angle error
        G[2]: boundary violation
        G[3]: inventory violation
        G[4]: secondary loop closure (crossings)
    """

    def __init__(
        self,
        catalog: TrackCatalog,
        config: OptimizationConfig,
        closure_tolerance: float = None,
        angle_tolerance: float = None,
        **kwargs,
    ):
        self.closure_tolerance = closure_tolerance or config.closure_tolerance
        self.angle_tolerance = angle_tolerance or config.angle_tolerance
        self.dims = compute_dimensions(config.total_inventory)

        xl, xu = generate_bounds(self.dims, max_piece_index=catalog._max_index)

        self.diagonal = np.sqrt(
            (config.boundary.max_x - config.boundary.min_x) ** 2
            + (config.boundary.max_y - config.boundary.min_y) ** 2
        )

        super().__init__(
            n_var=self.dims.n_var,
            n_obj=2,
            n_ieq_constr=5,
            xl=xl,
            xu=xu,
            **kwargs,
        )

        self.catalog = catalog
        self.config = config
        self.total_inventory = sum(config.inventory.values())
        self.inventory_by_index = self._convert_inventory(config.inventory)

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
        """Evaluate single chromosome for both objectives and constraints."""
        layout = decode_chromosome(
            x, self.catalog, self.config.inventory,
            self.decoder_config, dims=self.dims,
        )

        # Empty layouts: worst on both objectives, infeasible
        if layout.n_pieces == 0:
            out["F"] = [0.0, 0.0]
            out["G"] = [1.0, 1.0, 1.0, 1.0, -1.0]
            return

        # F[0]: -utilization (maximize piece usage)
        utilization = layout.n_pieces / self.total_inventory

        # F[1]: -avg_speed (maximize safe train speed via physics model)
        speed_profile = compute_speed_profile(
            layout, self.catalog, self.config.physics,
        )

        out["F"] = [-utilization, -speed_profile.avg_speed]

        # Constraints (g <= 0 feasible)
        main_path = layout.get_main_path() if hasattr(layout, 'get_main_path') else None
        closure_err = main_path.closure_error if main_path else layout.closure_error
        angle_err = main_path.angle_error if main_path else layout.angle_error

        g_closure = (closure_err - self.closure_tolerance) / self.closure_tolerance
        g_angle = (angle_err - self.angle_tolerance) / self.angle_tolerance
        g_boundary = self._compute_boundary_violation(layout) / max(self.diagonal, 1.0)
        g_inventory = float(self._compute_inventory_violation(layout))

        sec_err = getattr(layout, 'secondary_closure_error', 0.0)
        g_secondary = (sec_err - self.closure_tolerance) / self.closure_tolerance if sec_err > 0 else -1.0

        out["G"] = [g_closure, g_angle, g_boundary, g_inventory, g_secondary]

    def _compute_boundary_violation(self, layout) -> float:
        """Compute max boundary violation across all paths."""
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
        """Compute total count of pieces used beyond available inventory."""
        piece_counts: Dict[int, int] = {}

        for piece_idx in layout.main_loop_pieces:
            if piece_idx >= 0:
                piece_counts[piece_idx] = piece_counts.get(piece_idx, 0) + 1

        for switch_pair in layout.switch_pairs:
            for piece_idx in switch_pair.branch_pieces:
                if piece_idx >= 0:
                    piece_counts[piece_idx] = piece_counts.get(piece_idx, 0) + 1

        total_violation = 0
        for piece_idx, used in piece_counts.items():
            available = self.inventory_by_index.get(piece_idx, 0)
            if used > available:
                total_violation += used - available

        return total_violation
