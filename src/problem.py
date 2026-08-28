"""pymoo optimization problem for multi-objective track layout optimization.

Bi-objective NSGA-II with Deb's constraint handling:
- F[0] = -weighted piece score (maximize piece usage; special pieces carry a
         premium so topology is not stripped as overhead). This is a search
         score, NOT utilization — the honest count rides on the ``n_pieces``
         out-key.
- F[1] = expected time to cover every physical piece once, averaged over the
         2^J traversal routes at safety_margin=0.95 (minimize; see
         _expected_traversal_time)
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

# Speed charged to a segment the profiler stalled at v = 0, which would otherwise
# divide by zero. Crawling makes that segment cost ~100 s, so the layout ranks out
# of contention on F[1] instead of vanishing from the sum.
STALLED_SPEED_MS = 0.001


def _expected_traversal_time(
    layout,
    catalog: TrackCatalog,
    train_config,
    safety_margin: float = SPEED_SAFETY_MARGIN,
    closure_pos_tol: float = 4.0,
    closure_angle_tol: float = 5.0,
) -> float:
    """Expected time to cover every physical piece of the layout once.

    Each of the 2^J take/skip-siding routes is profiled independently (the
    3-pass profiler at ``safety_margin``), timing every segment it passes.
    A physical piece is charged the MEAN of its traversal times across all
    passages — the expected per-piece cost when the train picks routes
    uniformly at random — and the objective is the sum over distinct
    physical pieces. Routes agree on piece identity via ``piece_uids``; a
    descriptor CROSS_90 / DOUBLE_CROSSOVER spans two main slots but is one
    physical piece, unified here through the layout's junction records. A
    plain loop therefore reduces to its lap time; a self-crossing one does
    not — the crossing is one physical piece the lap passes twice, so it is
    charged once, at the mean of those two passages.

    Returns +inf when no route carries any piece: zero time would rank an
    unusable layout as the fastest possible one.
    """
    alias = {}
    for record in (*layout.cross_junctions, *getattr(layout, "dbl_crossovers", [])):
        first_slot, second_slot = record.positions
        alias[("main", second_slot, 0)] = ("main", first_slot, 0)

    stud_to_m = catalog.stud_mm / 1000.0
    per_piece: dict = {}
    for path in layout.paths:
        if len(path.piece_sequence) == 0:
            continue
        indices = np.asarray(path.piece_sequence, dtype=np.int32)
        route_indices = np.asarray(path.route_indices, dtype=np.int32)
        profile = compute_speed_profile(
            Layout(indices=indices, states=path.states, route_indices=route_indices),
            catalog, train_config, safety_margin=safety_margin,
            closure_pos_tol=closure_pos_tol, closure_angle_tol=closure_angle_tol,
        )
        arc_m = catalog.get_route_arc_lengths(indices, route_indices) * stud_to_m
        safe_speeds = np.where(profile.speeds > 0, profile.speeds, STALLED_SPEED_MS)
        for uid, seg_time in zip(path.piece_uids, arc_m / safe_speeds, strict=True):
            entry = per_piece.setdefault(alias.get(uid, uid), [0.0, 0])
            entry[0] += float(seg_time)
            entry[1] += 1

    if not per_piece:
        return float("inf")
    return sum(total / passages for total, passages in per_piece.values())


class TrackOptimizationProblem(ElementwiseProblem):
    """Bi-objective track layout optimization with NSGA-II.

    Objectives (both minimized for pymoo):
        F[0] = -weighted_piece_score   (maximize piece usage, with a premium
                                        per special piece; not a utilization
                                        ratio — report ``n_pieces`` instead)
        F[1] = expected_traversal_time (minimize the expected time to cover
                                        every physical piece once at
                                        safety_margin=0.95)

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

        self.decoder_config = DecoderConfig.from_optimization_config(config)

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
            out["n_pieces"] = 0
            out["n_sw_pairs"] = 0
            out["n_cross_comm"] = 0
            out["n_dc_comm"] = 0
            return

        # F[0]: -weighted piece score (special pieces carry a premium so
        # topology is not stripped as overhead). Deliberately not a ratio of
        # inventory — `n_pieces` below is what reporting must call utilization.
        piece_score = self._weighted_piece_score(layout)

        # F[1] = expected whole-network traversal time at SPEED_SAFETY_MARGIN:
        # every physical piece charged the mean of its traversal times across
        # the 2^J routes, so branch topology pays its real time cost. See
        # _expected_traversal_time.
        traversal_time = _expected_traversal_time(
            layout, self.catalog, self._train_config,
            closure_pos_tol=self.closure_tolerance,
            closure_angle_tol=self.angle_tolerance,
        )

        out["F"] = [-piece_score, traversal_time]

        # Committed-element census as custom out-keys: the Evaluator copies
        # them onto the Population, so the category-elite archive reads them
        # via pop.get() without an extra decode. n_cross_pieces counts
        # physical CROSS_90s, so emergent (self-intersection repair)
        # placements are included alongside descriptor commits.
        # n_pieces is the honest physical census — a switch, a CROSS_90 and a
        # DOUBLE_CROSSOVER each count once, however many slots they span — so
        # reporting never has to invert the weighted F[0] to guess a count.
        out["n_pieces"] = layout.n_physical_pieces
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

    def max_weighted_piece_score(self) -> float:
        """Ceiling on the F[0] score for this inventory, ignoring all geometry.

        The whole kit placed, plus every special element it can form: siding
        pairs are capped by the scarcer switch handedness, crossings and
        double-crossovers by their own stock — all three already derived from
        inventory in ``dims``. No closure, collision or boundary term, so this
        is the score a layout would reach if the terrain imposed nothing.
        """
        n_special = (self.dims.max_junctions
                     + self.dims.max_cross_junctions
                     + self.dims.max_double_crossovers)
        return (
            self.total_inventory + (self.special_piece_weight - 1.0) * n_special
        ) / self.total_inventory

    def _weighted_piece_score(self, layout) -> float:
        """Piece-usage score with special pieces weighted by ``special_piece_weight``.

        Each switch pair / crossing / double-crossover counts as W physical pieces
        toward the score (W>1), so folding multi-path topology into a layout raises
        the score rather than being pure overhead the GA would strip.

        The premium makes this a search score, not a utilization ratio: it can
        exceed 1.0. Utilization is ``n_physical_pieces / total_inventory``.
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
