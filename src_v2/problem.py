"""pymoo bi-objective problem for port-pair encoded track layouts.

Objectives (both minimized for pymoo, so return negated values):

- ``F[0] = -utilization`` — fraction of inventory placed in useful components
  (component size >= MIN_USEFUL_COMPONENT_SIZE). Excludes 2-/3-piece side-cycles
  the GA otherwise spawns as a utilization-inflation loophole. Diversity is
  enforced via NSGA-II's native crowding distance plus the external ε-archive
  (Laumanns et al. 2002), not by shaping this objective with a kind-count
  bonus — see Goldberg & Richardson 1987 and Deb 2001 on keeping objectives
  clean and pushing diversity into selection.
- ``F[1] = -min_speed`` — slowest piece speed limit across useful components,
  recovering V1's bottleneck semantics. Pieces in junk components do not
  contribute to either objective.

Constraints (g <= 0 feasible):

- ``G[0..2]`` — per-axis closure residual / tolerance (main vs branch-cycle
  tolerance per :func:`_residual_uses_branch_tolerance`).
- ``G[3]`` — boundary violation in studs / boundary diagonal.
- ``G[4]`` — uncovered piece-pair intersections / COLLISION_NORMALIZER.
- ``G[5..4+T]`` — per-type inventory excess (T = catalog.n_pieces).
- ``G[5+T]`` — incomplete switches / total switches (route-completeness).
- ``G[6+T]`` — incomplete crossings / total crossings (route-aligned check).
- ``G[7+T]`` — ``1 - n_cycles`` (require ≥ 1 closed cycle).
- ``G[8+T]`` — ``(min_branch_count - n_branch_cycles) / max(1, min_branch_count)``
  (always emitted; YAML-absent → no pressure).
- ``G[9+T]`` — ``(n_components - 1) / max(1, n_components)`` (single
  connected component required — no disjoint loops or open chains; ratio
  form so a heavy multi-component layout can't sit inside the adaptive
  epsilon during early generations).
- ``G[10+T]`` — ``n_loose_ports / total_ports`` (every port must be paired —
  rejects layouts with any open ends, including straights and curves).

Total: ``11 + T`` inequality constraints."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
from pymoo.core.problem import ElementwiseProblem

from .catalog import TrackCatalog
from .config import OptimizationConfig
from .decoder import DecoderConfig, decode_chromosome
from .encoding import (
    JUNCTION_GENES,
    JUNC_ACTIVE_OFFSET,
    JUNC_ANCHOR_OFFSET,
    PortPairDimensions,
    compute_port_pair_dimensions,
    generate_bounds,
)
from .intersection import count_uncovered_intersections
from .phenotype_dedupe import Phenotype
from .junction_materializer import JunctionMaterializer
from .repair import CycleClosureRepair
from .se2 import pose_compose
from .train import DEFAULT_TRAIN_CONFIG, TrainConfig
from .types import PortGraph


MIN_USEFUL_COMPONENT_SIZE: int = 4
"""Components smaller than this contribute neither to utilization nor speed.

Rules out the 2- and 3-piece self-cycles the GA otherwise spawns as a
utilization-inflation loophole, while still allowing genuine multi-loop
layouts (figure-8s, parallel tracks joined by a crossover).
"""


# Catalog routes whose presence on a cycle marks that cycle as a "branch"
# (siding) rather than a main loop. Branch cycles use the loose
# branch_closure_tolerance because LEGO 9V passing-siding geometry has an
# inherent ~6-stud y-residual absorbed by physical track flex.
BRANCH_ROUTES: frozenset = frozenset({"diverging"})


# Normaliser for the collision constraint G[4]: layouts with this many
# uncovered geometric intersections are at constraint boundary. Picked so
# that 1 uncovered intersection yields G[4] = 0.25 (well into infeasible).
COLLISION_NORMALIZER: int = 4


def _residual_uses_branch_tolerance(
    residual,
    branch_labels: Dict[Tuple[int, str], int],
) -> bool:
    """True iff the cycle this residual closes is a switch-induced branch.

    Cycles containing a switch's ``diverging`` route are treated as
    branches. Cycles that are pure mainline / through routes — including
    figure-8 cycles via crossings — keep the tight main tolerance.
    """
    if not branch_labels:
        return False
    cycles_a = {cid for (s, _r), cid in branch_labels.items() if s == residual.slot_a}
    cycles_b = {cid for (s, _r), cid in branch_labels.items() if s == residual.slot_b}
    common = cycles_a & cycles_b
    branch_cycles = {
        cid for (_s, route), cid in branch_labels.items()
        if route in BRANCH_ROUTES
    }
    return bool(common & branch_cycles)


class PortPairProblem(ElementwiseProblem):
    """Bi-objective NSGA-II problem for port-pair track layouts."""

    def __init__(
        self,
        catalog: TrackCatalog,
        config: OptimizationConfig,
        train_config: TrainConfig | None = None,
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
        self.train_config = train_config or DEFAULT_TRAIN_CONFIG

        self.closure_tolerance = closure_tolerance or config.closure_tolerance
        self.branch_closure_tolerance = config.branch_closure_tolerance
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

        # Phase 1.B (Rule 24 revised): cycle-closure repair runs Baldwinian
        # inside _evaluate — operates on a per-individual clone so the pool
        # keeps the raw chromosome and selection/crossover work from the
        # genotype, while F/G and out["pheno"] reflect the repaired phenotype.
        self._cycle_closure = CycleClosureRepair(
            dims=self.dims,
            catalog=catalog,
            decoder_config=self.decoder_config,
            inventory=config.inventory,
        )

        # Phase 5a: junction materializer expands active passing-siding
        # descriptors into spliced switches + branch pieces on the cloned
        # chromosome, then the existing decoder reads the materialized layout.
        # Validation failures are silent (junction deactivated, no inventory
        # consumed) per the PLAN's "silent skip - no cost" contract.
        self._materializer = JunctionMaterializer(
            dims=self.dims,
            catalog=catalog,
            inventory=config.inventory,
            decoder_config=self.decoder_config,
            position_tolerance=self.closure_tolerance,
            angle_tolerance_deg=self.angle_tolerance,
        )

        # Constraint count: 3 closure + boundary + collisions
        # + n_pieces inventory + switch_completeness + crossing_completeness
        # + cycle_count + branch_cycle_deficit + single_component
        # + loose_port_ratio = 11 + T
        n_constr = 11 + catalog.n_pieces

        super().__init__(
            n_var=self.dims.n_var,
            n_obj=2,
            n_ieq_constr=n_constr,
            xl=xl,
            xu=xu,
            **kwargs,
        )

    def _evaluate(self, x, out, *args, **kwargs) -> None:
        # Baldwinian closure repair (Rule 24 revised, PLAN §9.7): clone x,
        # run the surgery on the clone, evaluate from the repaired phenotype.
        # The pool keeps the raw x so NSGA-II selection/crossover see the
        # unrepaired genotype (preserves diversity); the repaired graph is
        # exposed via out["pheno"] so callbacks/visualization read the same
        # phenotype that F/G derive from. Standard pymoo "pheno" key
        # round-trips through StarmapParallelization (test 1.6 / Coupling B).
        # Coupling C: harvest active junction anchors from the RAW chromosome
        # so closure repair leaves their cycles untouched (otherwise materialize()
        # below would silently fail and the layout loses its passing-siding
        # topology). Read against x, not x_repaired, since both are byte-equal
        # at this point and the repair operates on the clone.
        active_anchors: set[int] = set()
        for j in range(self.dims.J_max):
            base = self.dims.junc_start + j * JUNCTION_GENES
            if int(x[base + JUNC_ACTIVE_OFFSET]) == 1:
                active_anchors.add(int(x[base + JUNC_ANCHOR_OFFSET]))

        x_repaired = x.copy()
        self._cycle_closure.repair_one(
            x_repaired,
            skip_anchor_slots=active_anchors if active_anchors else None,
        )
        # Phase 5a: expand active junction descriptors before final decode.
        # Silent failures deactivate their descriptors so re-decode is idempotent.
        self._materializer.materialize(x_repaired)
        graph = decode_chromosome(
            x_repaired, self.dims, self.catalog, self.decoder_config,
        )
        out["pheno"] = graph

        if graph.n_slots == 0:
            out["F"] = np.array([np.inf, np.inf])
            out["G"] = np.full(self.n_ieq_constr, 1e6)
            return

        # ---- Objectives (cycle slots in the LARGEST component only) ----
        # Multi-component layouts must not gain F-credit for cycles tucked
        # inside side components: with the adaptive epsilon admitting CV>0
        # individuals during early gens, a multi-cycle multi-component
        # chromosome would beat a true single-component closed track on F
        # while still being epsilon-feasible on CV — and crowd the real
        # feasibles out of survival. Restricting useful slots to the
        # largest component ties the F gradient to the same goal as the
        # single-component constraint.
        largest_component = (
            max(graph.connected_components, key=len)
            if graph.connected_components else set()
        )
        slots_on_cycles = {
            slot for (slot, _route) in graph.branch_labels
            if slot in largest_component
        }
        n_useful = len(slots_on_cycles)
        utilization = n_useful / self.total_inventory if n_useful else 0.0

        # Rule 35 / Rule 21: aggregate F[1] over (slot, route_name) pairs, NOT
        # over slots. Switches and DOUBLE_CROSSOVER have route-dependent radii
        # (through routes: straight; diverging routes: R=320 mm); iterating
        # slots and using a per-piece default inflates min_speed on switched
        # layouts.
        #
        # Speed source is TrainConfig.v_eff(radius), NOT the catalog static
        # speed_table. v_eff combines the curvature derailment cap (slide /
        # tip / Nadal at the loaded train's mu_design / mass / geometry) with
        # the motor cap (v_motor_max). This makes F[1] honest against the
        # measured locomotive physics in configs/trains/measured_consist.yaml.
        useful_speeds = []
        for (slot, route_name), _cycle_id in graph.branch_labels.items():
            if slot not in largest_component:
                continue
            piece_idx = graph.slot_indices.get(slot)
            if piece_idx is None or piece_idx < 0:
                continue
            radius_m = self.catalog.get_radius_m_for_route(piece_idx, route_name)
            speed = self.train_config.v_eff(
                radius_m if radius_m is not None else math.inf,
            )
            useful_speeds.append(speed)
        min_speed = min(useful_speeds) if useful_speeds else 0.0

        out["F"] = np.array([-utilization, -min_speed])

        # ---- Constraints ----
        G = []

        # Closure (per-axis, normalized; tolerance is loosened for branch cycles
        # induced by switches because LEGO 9V siding geometry has an inherent
        # ~6-stud y-residual that physical track flex absorbs).
        if graph.closure_residuals:
            def pos_tol(r) -> float:
                return (
                    self.branch_closure_tolerance
                    if _residual_uses_branch_tolerance(r, graph.branch_labels)
                    else self.closure_tolerance
                )
            G.append(max(abs(r.dx) / pos_tol(r) - 1.0 for r in graph.closure_residuals))
            G.append(max(abs(r.dy) / pos_tol(r) - 1.0 for r in graph.closure_residuals))
            G.append(max(
                abs(math.degrees(r.dtheta)) / self.angle_tolerance - 1.0
                for r in graph.closure_residuals
            ))
        else:
            G.extend([0.0, 0.0, 0.0])

        # Boundary
        bv = self._compute_boundary_violation(graph)
        G.append((bv - self.boundary_tolerance) / max(self.diagonal, 1.0))

        # Collisions: piece-pair geometric intersections not mediated by a
        # crossing piece (CROSS_90 / DOUBLE_CROSSOVER). 1 uncovered → infeasible.
        n_uncovered = count_uncovered_intersections(graph, self.catalog)
        G.append(n_uncovered / COLLISION_NORMALIZER)

        # Per-type inventory excess
        G.extend(self._compute_inventory_excess(graph))

        # Route-completeness ratios — Phase 2 replaces generic loose-port
        # aggregate with per-kind ratios. Soft-penalized during exploration
        # (warmup); only fully active post-warmup when finalization_active
        # flips on (and Phase 3's branch growth lands).
        switch_kinds = ("switch",)
        total_switches = sum(
            1 for pid in graph.slot_pieces.values()
            if pid in self.spec.by_id and self.spec.by_id[pid].kind in switch_kinds
        )
        G.append(self._count_incomplete_switches(graph) / max(1, total_switches))

        total_crossings = sum(
            1 for pid in graph.slot_pieces.values()
            if pid in self.spec.by_id and self.spec.by_id[pid].kind == "crossing"
        )
        G.append(self._count_incomplete_crossings(graph) / max(1, total_crossings))

        # Cycle presence
        G.append(1.0 - graph.n_cycles)

        # Branch-cycle target — always emitted; YAML-absent → min_branch_count = 0
        # → constraint always 0.0 → no pressure on layouts that don't need branches.
        min_branch_count = getattr(self.config, "min_branch_count", 0) or 0
        if min_branch_count > 0:
            n_branch_cycles = self._count_branch_cycles(graph)
            G.append((min_branch_count - n_branch_cycles) / min_branch_count)
        else:
            G.append(0.0)

        # Single-component requirement — feasible layouts must be ONE
        # connected closed track. Normalised to a 0–1 ratio so a heavy
        # multi-component layout doesn't sit inside the adaptive
        # epsilon (which can be ~15 during the early hold phase).
        n_components_with_pieces = sum(
            1 for c in graph.connected_components if len(c) >= 1
        )
        G.append(
            (n_components_with_pieces - 1) / max(1, n_components_with_pieces)
        )

        # Loose-port ratio across the whole layout (NOT just switches/
        # crossings). Any active port not paired by an edge counts; the
        # ratio is normalised by total ports to keep the constraint scale
        # comparable to the others.
        total_ports = sum(
            len(self.spec.by_id[pid].ports)
            for pid in graph.slot_pieces.values()
            if pid in self.spec.by_id
        )
        n_loose = len(graph.loose_ports)
        G.append(n_loose / max(1, total_ports))

        out["G"] = np.array(G, dtype=np.float64)

    # ------------------------------------------------------------------
    # Constraint helpers
    # ------------------------------------------------------------------

    def _port_use_set(self, graph: PortGraph) -> set:
        """Set of (slot, port_name) pairs that appear in any active edge."""
        return {
            (e.slot_a, e.port_a) for e in graph.edges
        } | {
            (e.slot_b, e.port_b) for e in graph.edges
        }

    def _count_incomplete_switches(self, graph: PortGraph) -> int:
        """Switches whose paired-port set != {A, B, C}.

        A complete switch has all three ports terminated by edges (port C
        being the diverging stub which must connect to a branch). Incomplete
        switches are tolerated during exploration (Phase 2 soft-penalizes
        via this constraint), and Phase 3's branch growth in finalization
        attempts to close them.
        """
        used = self._port_use_set(graph)
        return sum(
            1
            for slot, pid in graph.slot_pieces.items()
            for ps in [self.spec.by_id.get(pid)]
            if ps is not None and ps.kind == "switch"
            and {p for (s, p) in used if s == slot} != set(ps.ports.keys())
        )

    def _count_incomplete_crossings(self, graph: PortGraph) -> int:
        """Crossings whose paired-port set != {A, B, C, D}.

        Both routes (horizontal A-B and vertical C-D for CROSS_90; parallel
        and crossover route bundles for DOUBLE_CROSSOVER) require all four
        ports terminated. The crossing pair-set validation in repair.py
        ensures pair-rows match catalog routes; this constraint catches
        crossings with missing pair-rows entirely.
        """
        used = self._port_use_set(graph)
        return sum(
            1
            for slot, pid in graph.slot_pieces.items()
            for ps in [self.spec.by_id.get(pid)]
            if ps is not None and ps.kind == "crossing"
            and {p for (s, p) in used if s == slot} != set(ps.ports.keys())
        )

    def build_phenotype(self, graph: PortGraph) -> Phenotype:
        """Compute the structural-summary tuple for phenotype dedupe (Phase 5).

        Public so the dedupe callback can call it after re-decoding each
        individual's chromosome — pymoo's elementwise evaluator can't pass
        non-array data through ``out``, so the phenotype must be computed
        callback-side.
        """
        kinds = {
            pid: ps.kind
            for pid, ps in self.spec.by_id.items()
        }
        n_switches = sum(
            1 for pid in graph.slot_pieces.values() if kinds.get(pid) == "switch"
        )
        n_crossings = sum(
            1 for pid in graph.slot_pieces.values() if kinds.get(pid) == "crossing"
        )
        max_component_size = (
            max((len(c) for c in graph.connected_components), default=0)
        )
        histogram_dict: dict = {}
        for pid in graph.slot_pieces.values():
            histogram_dict[pid] = histogram_dict.get(pid, 0) + 1
        return Phenotype(
            n_switches=n_switches,
            n_crossings=n_crossings,
            n_cycles=int(graph.n_cycles),
            n_branch_cycles=self._count_branch_cycles(graph),
            max_component_size=int(max_component_size),
            piece_histogram=tuple(sorted(histogram_dict.items())),
        )

    def _count_branch_cycles(self, graph: PortGraph) -> int:
        """Distinct cycle ids that contain at least one switch's diverging
        route. Used by the ``min_branch_count`` constraint to require a
        target number of passing-siding branches in the final layout."""
        return len({
            cid
            for (_slot, route), cid in graph.branch_labels.items()
            if route in BRANCH_ROUTES
        })

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
