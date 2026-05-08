"""pymoo bi-objective problem for port-pair encoded track layouts.

Objectives (both minimized for pymoo, so return negated values):

- ``F[0] = -utilization`` — fraction of inventory placed in useful components
  (component size >= MIN_USEFUL_COMPONENT_SIZE). This excludes the 2-/3-piece
  side-cycles the GA otherwise creates as a utilization-inflation loophole.
- ``F[1] = -min_speed`` — slowest piece speed limit across useful components,
  recovering V1's bottleneck semantics. Pieces in junk components do not
  contribute to either objective.

Constraints (g <= 0 feasible):

- ``G[0..2]`` — per-axis closure residual / tolerance - 1, max over all
  cycle-closing edges. If the layout has no cycles, all three are 0
  (the cycle-count constraint G[5+T] catches that case).
- ``G[3]`` — boundary violation in studs / boundary diagonal.
- ``G[4]`` — collisions placeholder (0 in v0).
- ``G[5..4+T]`` — per-type inventory excess, normalized (T = catalog n_pieces).
- ``G[5+T]`` — loose-port count normalized by total active ports.
- ``G[6+T]`` — ``1 - n_cycles``, requires at least one closed cycle.

Total: ``7 + T`` inequality constraints.
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np
from pymoo.core.problem import ElementwiseProblem

from .catalog import TrackCatalog
from .config import OptimizationConfig
from .decoder import DecoderConfig, decode_chromosome
from .encoding import PortPairDimensions, compute_port_pair_dimensions, generate_bounds
from .se2 import pose_compose
from .types import PortGraph


MIN_USEFUL_COMPONENT_SIZE: int = 4
"""Components smaller than this contribute neither to utilization nor speed.

Rules out the 2- and 3-piece self-cycles the GA otherwise spawns as a
utilization-inflation loophole, while still allowing genuine multi-loop
layouts (figure-8s, parallel tracks joined by a crossover).
"""


class PortPairProblem(ElementwiseProblem):
    """Bi-objective NSGA-II problem for port-pair track layouts."""

    def __init__(
        self,
        catalog: TrackCatalog,
        config: OptimizationConfig,
        closure_tolerance: float = None,
        angle_tolerance: float = None,
        **kwargs,
    ) -> None:
        if catalog.spec is None:
            raise ValueError(
                "PortPairProblem requires a catalog loaded from V2 yaml; "
                "catalog.spec is None."
            )

        self.catalog = catalog
        self.config = config
        self.spec = catalog.spec

        self.closure_tolerance = closure_tolerance or config.closure_tolerance
        self.angle_tolerance = angle_tolerance or config.angle_tolerance
        self.boundary_tolerance = config.boundary_tolerance

        self.dims = compute_port_pair_dimensions(
            config.boundary, catalog, config.inventory,
        )
        xl, xu = generate_bounds(
            self.dims, config.boundary, max_piece_id=catalog.n_pieces - 1,
        )

        self.diagonal = math.sqrt(
            (config.boundary.max_x - config.boundary.min_x) ** 2
            + (config.boundary.max_y - config.boundary.min_y) ** 2
        )

        self.total_inventory = max(1, sum(config.inventory.values()))
        self.inventory_by_index: Dict[int, int] = {}
        for piece_id, count in config.inventory.items():
            idx = catalog.id_to_index.get(piece_id)
            if idx is not None:
                self.inventory_by_index[idx] = count

        self.decoder_config = DecoderConfig(
            closure_position_tolerance=self.closure_tolerance,
            closure_angle_tolerance_deg=self.angle_tolerance,
            boundary_min_x=config.boundary.min_x,
            boundary_max_x=config.boundary.max_x,
            boundary_min_y=config.boundary.min_y,
            boundary_max_y=config.boundary.max_y,
        )

        # Constraint count: 3 closure + boundary + collisions
        # + n_pieces inventory + loose-port + cycle = 7 + T
        n_constr = 7 + catalog.n_pieces

        super().__init__(
            n_var=self.dims.n_var,
            n_obj=2,
            n_ieq_constr=n_constr,
            xl=xl,
            xu=xu,
            **kwargs,
        )

    def _evaluate(self, x, out, *args, **kwargs) -> None:
        graph = decode_chromosome(x, self.dims, self.catalog, self.decoder_config)

        if graph.n_slots == 0:
            out["F"] = np.array([np.inf, np.inf])
            out["G"] = np.full(self.n_ieq_constr, 1e6)
            return

        # ---- Objectives (restricted to useful components) ----
        useful_slots = {
            slot
            for component in graph.connected_components
            if len(component) >= MIN_USEFUL_COMPONENT_SIZE
            for slot in component
        }

        if not useful_slots:
            out["F"] = np.array([np.inf, np.inf])
            out["G"] = np.full(self.n_ieq_constr, 1e6)
            return

        n_useful = len(useful_slots)
        utilization = n_useful / self.total_inventory

        useful_speeds = [
            float(self.catalog.speed_table[graph.slot_indices[slot]])
            for slot in useful_slots
            if slot in graph.slot_indices
            and 0 <= graph.slot_indices[slot] < len(self.catalog.speed_table)
        ]
        min_speed = min(useful_speeds) if useful_speeds else 0.0

        out["F"] = np.array([-utilization, -min_speed])

        # ---- Constraints ----
        G = []

        # Closure (per-axis, normalized to tolerance)
        if graph.closure_residuals:
            max_dx = max(abs(r.dx) for r in graph.closure_residuals)
            max_dy = max(abs(r.dy) for r in graph.closure_residuals)
            max_dtheta_deg = max(
                abs(math.degrees(r.dtheta)) for r in graph.closure_residuals
            )
            G.append(max_dx / self.closure_tolerance - 1.0)
            G.append(max_dy / self.closure_tolerance - 1.0)
            G.append(max_dtheta_deg / self.angle_tolerance - 1.0)
        else:
            G.extend([0.0, 0.0, 0.0])

        # Boundary
        bv = self._compute_boundary_violation(graph)
        G.append((bv - self.boundary_tolerance) / max(self.diagonal, 1.0))

        # Collisions placeholder (v0)
        G.append(0.0)

        # Per-type inventory excess
        G.extend(self._compute_inventory_excess(graph))

        # Loose ports — STRICT: any unconnected port is infeasible.
        # Previously normalized by total_active_ports, which made layouts with
        # 1-2 dangling ports look "almost feasible" and the GA learned to
        # ignore them. The user's stated requirement is "no loose ports", so
        # we surface the raw count instead.
        G.append(float(graph.n_loose_ports))

        # Cycle count: require ≥ 1
        G.append(1.0 - graph.n_cycles)

        out["G"] = np.array(G, dtype=np.float64)

    # ------------------------------------------------------------------
    # Constraint helpers
    # ------------------------------------------------------------------

    def _compute_boundary_violation(self, graph: PortGraph) -> float:
        """Max boundary violation in studs across all slot poses + ports."""
        if not graph.slot_poses:
            return 0.0

        b = self.config.boundary
        max_violation = 0.0

        for slot_idx, pose in graph.slot_poses.items():
            piece_id = graph.slot_pieces.get(slot_idx)
            if piece_id is None:
                continue
            piece_spec = self.spec.by_id.get(piece_id)
            if piece_spec is None:
                continue
            for port in piece_spec.ports.values():
                pw = pose_compose(pose, (port.dx, port.dy, port.dtheta))
                ex = max(0.0, b.min_x - pw[0], pw[0] - b.max_x)
                ey = max(0.0, b.min_y - pw[1], pw[1] - b.max_y)
                max_violation = max(max_violation, ex, ey)

        return max_violation

    def _compute_inventory_excess(self, graph: PortGraph) -> list:
        """Per-catalog-index inventory excess, normalized by limit."""
        n_types = self.catalog.n_pieces
        census = np.zeros(n_types, dtype=np.int64)

        for piece_index in graph.slot_indices.values():
            if 0 <= piece_index < n_types:
                census[piece_index] += 1

        result = []
        for t in range(n_types):
            limit = self.inventory_by_index.get(t, 0)
            excess = max(0, int(census[t]) - int(limit))
            result.append(excess / max(1, limit))
        return result
