"""pymoo optimization problem for multi-objective track layout optimization.

Bi-objective NSGA-II with Deb's constraint handling:
- F[0] = -utilization (maximize piece usage)
- F[1] = -(avg speed of the slowest traversal route) at safety_margin=0.95
         (maximize the worst route's pace; see _slowest_route_speed)
- 5 + per-piece-type inequality constraints via Deb's CV rules
"""

from __future__ import annotations

import numpy as np
from pymoo.core.problem import ElementwiseProblem

from .config import OptimizationConfig
from .catalog import TrackCatalog
from .decoder import DecoderConfig, decode_chromosome
from .encoding import compute_dimensions, generate_bounds
from .geometry import Layout
from .train import compute_speed_profile
from .intersection import (
    CROSS_90_INDEX,
    DOUBLE_CROSSOVER_INDEX,
    count_dangling_cross_ports,
    count_dangling_double_crossover_ports,
    count_segment_crossings,
)

# Operating safety derate applied to every Pass-1 speed cap: the train runs at
# 95% of each segment's derailment-limited speed. Exposed so tests recomputing
# a layout's bottleneck independently use the same margin the objective does.
SPEED_SAFETY_MARGIN = 0.95

# G value assigned to every constraint of a degenerate (0-piece) layout. Large
# finite (never NaN — NaN breaks dominance comparison), so CV orders degenerates
# below every real individual. Consumers (e.g. epsilon calibration) must exclude
# CVs derived from this sentinel before computing population statistics.
DEGENERATE_G = 1.0e6


def _slowest_route_speed(
    layout,
    catalog: TrackCatalog,
    train_config,
    safety_margin: float = SPEED_SAFETY_MARGIN,
) -> float:
    """Average speed of the layout's slowest traversal route.

    A layout with J switch pairs offers 2^J routes (take or skip each siding).
    Each route is scored independently — a per-route ``Layout`` view fed to the
    3-pass profiler at ``safety_margin``, which brakes for curves and
    accelerates on straights, never exceeding the per-segment derailment cap —
    yielding that route's overall lap pace (``avg_speed``). F[1] uses the
    SLOWEST route: the most curve-bound one, which the train covers at the
    lowest overall speed. Scoring every route (not just the through-line) makes
    branch geometry register; a switchless layout has one route, so this is just
    that loop's pace.

    Returns 0.0 only for a degenerate layout with no pieces on any route (the
    caller short-circuits the empty-layout case before reaching here).
    """
    route_speeds = [
        compute_speed_profile(
            Layout(
                indices=np.asarray(path.piece_sequence, dtype=np.int32),
                states=path.states,
                route_indices=np.asarray(path.route_indices, dtype=np.int32),
            ),
            catalog, train_config, safety_margin=safety_margin,
        ).avg_speed
        for path in layout.paths
        if len(path.piece_sequence) > 0
    ]
    return float(min(route_speeds)) if route_speeds else 0.0


class TrackOptimizationProblem(ElementwiseProblem):
    """Bi-objective track layout optimization with NSGA-II.

    Objectives (both minimized for pymoo):
        F[0] = -utilization         (maximize piece usage)
        F[1] = -slowest_route_speed (maximize the slowest route's avg speed
                                     at safety_margin=0.95)

    Constraints (g <= 0 feasible, Deb's CV rules):
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
        closure_tolerance: float | None = None,
        angle_tolerance: float | None = None,
        **kwargs,
    ):
        self.closure_tolerance = (
            closure_tolerance if closure_tolerance is not None
            else config.closure_tolerance
        )
        self.angle_tolerance = (
            angle_tolerance if angle_tolerance is not None
            else config.angle_tolerance
        )
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
            # 3 closure + boundary + collisions + per-type inventory
            n_ieq_constr=5 + catalog.n_pieces,
            xl=xl,
            xu=xu,
            **kwargs,
        )

        self.catalog = catalog
        self.config = config
        self._train_config = config.load_train_config()
        self.total_inventory = sum(config.inventory.values())
        self.special_piece_weight = config.special_piece_weight
        self.inventory_by_index = catalog.inventory_by_index(config.inventory)

        self.decoder_config = DecoderConfig(
            boundary_min_x=config.boundary.min_x,
            boundary_max_x=config.boundary.max_x,
            boundary_min_y=config.boundary.min_y,
            boundary_max_y=config.boundary.max_y,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate single chromosome for both objectives and constraints."""
        layout = decode_chromosome(
            x, self.catalog, self.config.inventory,
            dims=self.dims, config=self.decoder_config,
        )

        # Infeasibility sentinel: +inf F so feasibles dominate,
        # large finite G so CV orders infeasibles by total violation.
        # Never NaN — pymoo tolerates +inf in HV when filtered to feasible-only,
        # but NaN breaks dominance comparison and requires replace_nan_values_by.
        if layout.n_pieces == 0:
            out["F"] = np.array([np.inf, np.inf])
            out["G"] = np.full(self.n_ieq_constr, DEGENERATE_G)
            out["n_sw_pairs"] = 0
            out["n_cross_comm"] = 0
            out["n_dc_comm"] = 0
            return

        # F[0]: -utilization (special pieces weighted so topology is not overhead)
        utilization = self._weighted_utilization(layout)

        # F[1] = -(slowest of the 2^J take/skip-siding routes) at
        # SPEED_SAFETY_MARGIN; scoring every route lets curve-heavy branches
        # lower F[1]. See _slowest_route_speed.
        slowest_route_speed = _slowest_route_speed(
            layout, self.catalog, self._train_config,
        )

        out["F"] = [-utilization, -slowest_route_speed]

        # Committed-element census as custom out-keys: the Evaluator copies
        # them onto the Population, so the category-elite archive reads them
        # via pop.get() without an extra decode. n_cross_pieces counts
        # physical CROSS_90s, so emergent (self-intersection repair)
        # placements are included alongside descriptor commits.
        out["n_sw_pairs"] = len(layout.switch_pairs)
        out["n_cross_comm"] = layout.n_cross_pieces
        out["n_dc_comm"] = len(getattr(layout, "dbl_crossovers", []))

        # Constraints: 5 + n_piece_types inequalities, g <= 0 feasible.
        # G[0..2]: per-axis closure residuals (dx, dy, dtheta in degrees).
        # G[3]: boundary violation.
        # G[4]: collision count (unresolved self-intersections).
        # G[5..4+T]: per-type inventory excess for each catalog piece index.
        main_path = layout.get_main_path()
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

        # Two failure modes contribute to this constraint with different
        # weights:
        #  - segment crossings without any CROSS_90 placement: scaled by
        #    /5 (mild penalty per unresolved crossing)
        #  - CROSS_90 slots whose perpendicular partner doesn't exist
        #    (2 dangling ports): each one contributes 1.0 directly, so a
        #    single dangling-port cross is enough to keep g_collisions > 0
        #    with a margin much larger than mutation can shrink in one
        #    operation. A dangling cross is structurally unbuildable, so
        #    we want it 5x more "infeasible" than an unresolved crossing.
        main_pieces_list = list(layout.main_loop_pieces)
        unresolved = count_segment_crossings(layout.states, main_pieces_list)
        dangling = count_dangling_cross_ports(layout.states, main_pieces_list)
        dangling_dc = count_dangling_double_crossover_ports(
            main_pieces_list,
            getattr(layout, "dbl_crossovers", []),
        )
        # Each dangling DBL_CROSSOVER port is structurally unbuildable, same
        # weight as a dangling CROSS_90 port (1.0 per port).
        g_collisions = (unresolved / 5.0) + float(dangling) + float(dangling_dc)

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

    def _weighted_utilization(self, layout) -> float:
        """Utilization with special pieces weighted by ``special_piece_weight``.

        Each switch pair / crossing / double-crossover counts as W physical pieces
        toward the score (W>1), so folding multi-path topology into a layout raises
        utilization rather than being pure overhead the GA would strip.
        """
        n_special = (
            len(layout.switch_pairs)
            + len(layout.cross_junctions)
            + len(getattr(layout, "dbl_crossovers", []))
        )
        effective = layout.n_physical_pieces + (self.special_piece_weight - 1.0) * n_special
        return effective / self.total_inventory

    def _compute_per_type_inventory_violation(self, layout) -> np.ndarray:
        """Per-catalog-index inventory excess, normalized by max_occ[t].

        Returns array of length self.catalog.n_pieces. Entry t is:
            max(0, census[t] - max_occ[t]) / max(1, max_occ[t])
        where max_occ[t] comes from self.inventory_by_index (0 if absent).

        CROSS_90 / DOUBLE_CROSSOVER are charged as physical pieces (see
        n_cross_pieces), not per traversal slot.
        """
        n_types = self.catalog.n_pieces
        census = np.zeros(n_types, dtype=np.int64)
        paired = (CROSS_90_INDEX, DOUBLE_CROSSOVER_INDEX)

        for piece_idx in layout.main_loop_pieces:
            if piece_idx in paired:
                continue  # charged as physical pieces below
            if 0 <= piece_idx < n_types:
                census[piece_idx] += 1
        for switch_pair in layout.switch_pairs:
            for piece_idx in switch_pair.branch_pieces:
                if 0 <= piece_idx < n_types:
                    census[piece_idx] += 1
        census[CROSS_90_INDEX] += layout.n_cross_pieces
        census[DOUBLE_CROSSOVER_INDEX] += len(layout.dbl_crossovers)

        max_occ = np.array(
            [self.inventory_by_index.get(t, 0) for t in range(n_types)],
            dtype=np.float64,
        )
        excess = np.maximum(0.0, census.astype(np.float64) - max_occ)
        return excess / np.maximum(1.0, max_occ)
