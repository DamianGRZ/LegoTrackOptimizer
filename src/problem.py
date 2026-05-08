"""pymoo optimization problem for multi-objective track layout optimization.

Bi-objective NSGA-II with Deb's constraint handling:
- F[0] = -utilization (maximize piece usage)
- F[1] = -min_speed (maximize bottleneck speed, V2 semantics)
- 5 inequality constraints via Deb's CV rules
"""

from typing import Dict

import numpy as np
from pymoo.core.problem import ElementwiseProblem

from .config import OptimizationConfig
from .catalog import TrackCatalog
from .decoder import DecoderConfig, decode_chromosome
from .encoding import compute_dimensions, generate_bounds
from .train import compute_speed_profile
from .intersection import count_segment_crossings


class TrackOptimizationProblem(ElementwiseProblem):
    """Bi-objective track layout optimization with NSGA-II.

    Objectives (both minimized for pymoo):
        F[0] = -utilization  (maximize piece usage)
        F[1] = -min_speed    (maximize bottleneck, V2 semantics)

    Constraints (V2 shape, g <= 0 feasible, Deb's CV rules):
        G[0]: closure_x    = abs(dx) / closure_tolerance - 1
        G[1]: closure_y    = abs(dy) / closure_tolerance - 1
        G[2]: closure_theta= abs(dtheta_deg) / angle_tolerance - 1  (wrapped to (-180, 180])
        G[3]: boundary     = (boundary_violation - boundary_tolerance) / diagonal
        G[4]: collisions   = count_segment_crossings / 5.0
        G[5..4+T]: per-type inventory excess, normalized by max_occ[t]
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
        self.boundary_tolerance = config.boundary_tolerance

        # Partitioned dimensions from inventory
        self.dims = compute_dimensions(config, catalog)

        xl, xu = generate_bounds(self.dims)

        self.diagonal = np.sqrt(
            (config.boundary.max_x - config.boundary.min_x) ** 2
            + (config.boundary.max_y - config.boundary.min_y) ** 2
        )

        super().__init__(
            n_var=self.dims.n_var,
            n_obj=2,
            n_ieq_constr=5 + catalog.n_pieces,  # Stage B: 3 closure + boundary + collisions + per-type inventory
            xl=xl,
            xu=xu,
            **kwargs,
        )

        self.catalog = catalog
        self.config = config
        self._train_config = config.load_train_config()
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
            dims=self.dims, config=self.decoder_config,
        )

        # V2 infeasibility sentinel: +inf F so feasibles dominate,
        # large finite G so CV orders infeasibles by total violation.
        # Never NaN — pymoo tolerates +inf in HV when filtered to feasible-only,
        # but NaN breaks dominance comparison and requires replace_nan_values_by.
        if layout.n_pieces == 0:
            out["F"] = np.array([np.inf, np.inf])
            out["G"] = np.full(self.n_ieq_constr, 1e6)
            return

        # F[0]: -utilization
        utilization = layout.n_pieces / self.total_inventory

        # F[1] = -min_speed (V2 bottleneck). avg_speed was a 3-pass harmonic-mean
        # profile that masks dangerous curves behind fast straights — a safety
        # failure mode. min_speed is the minimum over per-segment speed caps,
        # which is the V2 v_bottleneck semantics (strictly conservative).
        speed_profile = compute_speed_profile(
            layout, self.catalog, train_config=self._train_config,
        )

        out["F"] = [-utilization, -speed_profile.min_speed]

        # Constraints (V2 shape: 5 + n_piece_types inequalities, g <= 0 feasible).
        # G[0..2]: closure split into per-axis inequalities (shim: degrees for theta
        #   until Phase 5 decoder ships radian-native closure residuals).
        # G[3]: boundary violation (project extension; V2 spec drops boundary).
        # G[4]: collision count (unresolved self-intersections).
        # G[5..4+T]: per-type inventory excess for each catalog piece index.
        main_path = layout.get_main_path() if hasattr(layout, 'get_main_path') else None
        states = main_path.states if main_path is not None else layout.states

        dx = float(states[-1, 0] - states[0, 0])
        dy = float(states[-1, 1] - states[0, 1])
        dtheta_deg = float(states[-1, 2] - states[0, 2])
        # Wrap to (-180, +180]
        dtheta_deg = ((dtheta_deg + 180.0) % 360.0) - 180.0
        if dtheta_deg == -180.0:
            dtheta_deg = 180.0

        g_closure_x = abs(dx) / self.closure_tolerance - 1.0
        g_closure_y = abs(dy) / self.closure_tolerance - 1.0
        g_closure_theta = abs(dtheta_deg) / self.angle_tolerance - 1.0

        g_boundary = (
            self._compute_boundary_violation(layout) - self.boundary_tolerance
        ) / max(self.diagonal, 1.0)

        g_collisions = float(
            count_segment_crossings(layout.states, list(layout.main_loop_pieces))
        ) / 5.0

        g_inventory_per_type = self._compute_per_type_inventory_violation(layout)

        g_vec = np.concatenate([
            np.array([g_closure_x, g_closure_y, g_closure_theta,
                      g_boundary, g_collisions], dtype=np.float64),
            g_inventory_per_type,
        ])
        out["G"] = g_vec

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
        """Compute total count of pieces used beyond available inventory.

        Deprecated in Stage B: no longer called by _evaluate; kept only as a
        scalar summary helper for external callers. The per-type version
        below is the constraint-facing metric.
        """
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

    def _compute_per_type_inventory_violation(self, layout) -> np.ndarray:
        """Per-catalog-index inventory excess, normalized by max_occ[t].

        Returns array of length self.catalog.n_pieces. Entry t is:
            max(0, census[t] - max_occ[t]) / max(1, max_occ[t])
        where max_occ[t] comes from self.inventory_by_index (0 if absent).
        """
        n_types = self.catalog.n_pieces
        census = np.zeros(n_types, dtype=np.int64)

        for piece_idx in layout.main_loop_pieces:
            if 0 <= piece_idx < n_types:
                census[piece_idx] += 1
        for switch_pair in layout.switch_pairs:
            for piece_idx in switch_pair.branch_pieces:
                if 0 <= piece_idx < n_types:
                    census[piece_idx] += 1

        result = np.zeros(n_types, dtype=np.float64)
        for t in range(n_types):
            max_occ_t = self.inventory_by_index.get(t, 0)
            excess = max(0, int(census[t]) - int(max_occ_t))
            result[t] = excess / max(1, max_occ_t)
        return result
