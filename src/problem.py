"""pymoo optimization problem definition for track layout optimization.

Single problem class with:
- ONE objective: maximize utilization (-utilization for minimization)
- Constraints via Deb's CV (normalized to same scale)
- Decoder as single source of truth for layout evaluation
"""

from typing import Dict

import numpy as np
from pymoo.core.problem import ElementwiseProblem

from .config import OptimizationConfig
from .data import TrackCatalog
from .decoder import DecoderConfig, decode_chromosome
from .encoding import ChromosomeDimensions, compute_dimensions, generate_bounds
from .intersection import count_segment_crossings, count_path_crossings


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

        # Compute dynamic chromosome dimensions from inventory
        self.dims = compute_dimensions(config.total_inventory)

        # Generate integer bounds
        xl, xu = generate_bounds(self.dims, max_piece_index=catalog._max_index)

        # Compute boundary diagonal for normalization
        self.diagonal = np.sqrt(
            (config.boundary.max_x - config.boundary.min_x) ** 2
            + (config.boundary.max_y - config.boundary.min_y) ** 2
        )

        super().__init__(
            n_var=self.dims.n_var,
            n_obj=1,           # ONE objective: -(utilization - loose_port_penalty)
            n_ieq_constr=5,    # 5 constraints: closure, angle, boundary, inventory, secondary_closure
            xl=xl,
            xu=xu,
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
            dims=self.dims,
        )

        # Handle empty layouts
        if layout.n_pieces == 0:
            out["F"] = [0.0]
            out["G"] = [1.0, 1.0, 1.0, 1.0, -1.0]  # 5 constraints (secondary = feasible)
            return

        # ONE objective: maximize utilization scaled by closure quality.
        # Layouts that can't close (wrong angle OR far position) get diminished score.
        utilization = layout.n_pieces / self.total_inventory
        loose_port_penalty = layout.loose_port_count * (2.0 / self.total_inventory)

        # Closure quality from ANGLE: 0° error → 1.0, 360° error → 0.0
        angle_err = layout.angle_error if hasattr(layout, 'angle_error') else 360.0
        angle_quality = max(0.0, 1.0 - angle_err / 360.0)

        # Closure quality from POSITION: 0 studs → 1.0, diagonal → 0.0
        closure_err = layout.closure_error if hasattr(layout, 'closure_error') else self.diagonal
        position_quality = max(0.0, 1.0 - closure_err / max(self.diagonal, 1.0))

        # Boundary quality: 0 violation → 1.0, large violation → 0.0
        # Penalizes layouts that exceed boundary — pushes toward compact shapes
        boundary_violation = self._compute_boundary_violation(layout)
        half_boundary = min(
            self.config.boundary.max_x - self.config.boundary.min_x,
            self.config.boundary.max_y - self.config.boundary.min_y,
        ) * 0.5
        boundary_quality = max(0.0, 1.0 - boundary_violation / max(half_boundary, 1.0))

        # Combined closure scale: ALL qualities must be good for full credit
        closure_scale = 0.3 + 0.7 * min(angle_quality, position_quality, boundary_quality)

        # Self-intersection penalty: each crossing without CROSS_90 costs 3 pieces
        main_path = layout.get_main_path() if hasattr(layout, 'get_main_path') else None
        n_crossings = 0
        if main_path is not None and len(main_path.states) > 4:
            n_crossings += count_segment_crossings(
                main_path.states, main_path.piece_sequence,
            )
            # Also penalize branch-vs-main crossings
            for path in layout.paths[1:]:
                if path.path_id < 100 and len(path.states) > 4:
                    n_crossings += count_path_crossings(
                        main_path.states, path.states,
                    )
        crossing_penalty = n_crossings * (3.0 / self.total_inventory)

        out["F"] = [-(utilization * closure_scale - loose_port_penalty - crossing_penalty)]

        # Hard constraints via Deb's CV (g <= 0 feasible)
        closure_err = main_path.closure_error if main_path else layout.closure_error
        angle_err = main_path.angle_error if main_path else layout.angle_error
        g_closure = (closure_err - self.closure_tolerance) / self.closure_tolerance
        g_angle = (angle_err - self.angle_tolerance) / self.angle_tolerance
        g_boundary = self._compute_boundary_violation(layout) / max(self.diagonal, 1.0)
        g_inventory = float(self._compute_inventory_violation(layout))

        # 5th constraint: secondary loop closure (crossings)
        sec_err = getattr(layout, 'secondary_closure_error', 0.0)
        if sec_err > 0:
            g_secondary = (sec_err - self.closure_tolerance) / self.closure_tolerance
        else:
            g_secondary = -1.0  # Feasible when no crossing or no secondary loop

        out["G"] = [g_closure, g_angle, g_boundary, g_inventory, g_secondary]

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
