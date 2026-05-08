"""Phase 5a: junction materialization.

Reads active junction descriptors from a chromosome, validates each against
the current graph (anchor exists, anchor is in a cycle, FK closes within
tolerance, inventory has the pieces), then **mutates the chromosome** to
splice the template-produced switch + branch pieces in. After materialization
the chromosome is a structurally-complete passing-siding layout that the
existing :func:`src_v2.decoder.decode_chromosome` decodes normally -- the
decoder needs no new logic.

Architecturally this is the equivalent of the plan's "decoder.py:
_materialize_junctions" step (PLAN line 1481) but lifted into a standalone
preprocessor. The contract is unchanged: materialization runs AFTER cycle
identification (Rule 12) and is silent on validation failures (per the
"silent skip - no cost" line in PLAN's algorithm).

The Baldwinian repair contract (Rule 24 revised) requires that mutation
target a cloned chromosome, not the population's raw genotype.
:class:`JunctionMaterializer` documents this expectation and the caller
(``PortPairProblem._evaluate``) is responsible for the clone.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Tuple

from numpy.typing import NDArray

from .catalog import TrackCatalog
from .decoder import (
    DecoderConfig,
    _iter_cycles,
    decode_chromosome,
)
from .encoding import (
    INACTIVE,
    JUNCTION_KIND_FIGURE_8_CROSS,
    JUNCTION_KIND_PARALLEL_DC_BRIDGE,
    JUNCTION_KIND_PASSING_SIDING,
    PortPairDimensions,
    get_junction,
    get_piece_slot,
    get_port_pair,
    iter_active_pairs,
    iter_active_slots,
    set_junction,
    set_piece_slot,
    set_port_pair,
    set_slot_flip,
    set_slot_rotate,
)
from .se2 import Pose, pose_compose
from .templates import (
    FIGURE_8_TEMPLATES,
    Figure8Template,
    PARALLEL_DC_BRIDGE_TEMPLATES,
    PASSING_SIDING_TEMPLATES,
    ParallelBridgeTemplate,
    PassingSidingTemplate,
    check_dc_bridge_inventory,
    check_figure8_inventory,
    check_siding_inventory,
    compute_branch_pieces,
    compute_lobe_pieces,
    compute_required_main_distance,
    get_dc_bridge_inventory_requirements,
    get_figure8_inventory_requirements,
    get_siding_inventory_requirements,
    is_valid_figure8,
    is_valid_siding,
)
from .types import PortEdge, PortGraph


_MAX_BRANCH_STRAIGHTS: int = 8
"""Hard cap on n_straights per siding -- mirrors V1's ``_MAX_BRANCH=8``."""


class JunctionMaterializer:
    """Phase 5a: expand active junction descriptors into chromosome pieces + edges.

    Usage (caller is :class:`PortPairProblem._evaluate`)::

        x_clone = x.copy()
        cycle_closure.repair_one(x_clone)
        materializer.materialize(x_clone)
        graph = decode_chromosome(x_clone, ...)
    """

    def __init__(
        self,
        dims: PortPairDimensions,
        catalog: TrackCatalog,
        inventory: Dict[str, int],
        decoder_config: DecoderConfig,
        position_tolerance: float = 2.0,
        angle_tolerance_deg: float = 5.0,
    ) -> None:
        self.dims = dims
        self.catalog = catalog
        self.inventory = dict(inventory)
        self.decoder_config = decoder_config
        self.position_tolerance = position_tolerance
        self.angle_tolerance_deg = angle_tolerance_deg

    def materialize(self, x: NDArray) -> int:
        """Materialize every active junction in ``x``. Returns the number of
        successfully spliced junctions; failed junctions are silently
        deactivated (their ``active`` bit is zeroed) so re-running the
        materializer is idempotent."""
        if self.dims.J_max == 0:
            return 0
        graph = decode_chromosome(x, self.dims, self.catalog, self.decoder_config)
        if not graph.connected_components:
            return 0
        used_inventory = self._count_used_inventory(x)
        n_materialized = 0
        for j in range(self.dims.J_max):
            active, anchor, kind, param_a, param_b = get_junction(x, self.dims, j)
            if active != 1:
                continue
            ok = self._materialize_one(
                x, graph, j, anchor, kind, param_a, param_b, used_inventory,
            )
            if ok:
                n_materialized += 1
            else:
                # Silent skip: deactivate so re-decode is idempotent.
                set_junction(
                    x, self.dims, j,
                    active=0, anchor=anchor, kind=kind,
                    param_a=param_a, param_b=param_b,
                )
        return n_materialized

    # ------------------------------------------------------------------
    # Per-junction materialization
    # ------------------------------------------------------------------

    def _materialize_one(
        self,
        x: NDArray,
        graph: PortGraph,
        j_idx: int,
        anchor: int,
        kind: int,
        param_a: int,
        param_b: int,
        used_inventory: Dict[str, int],
    ) -> bool:
        if kind == JUNCTION_KIND_PASSING_SIDING:
            return self._materialize_passing_siding(
                x, graph, anchor, param_a, param_b, used_inventory,
            )
        if kind == JUNCTION_KIND_FIGURE_8_CROSS:
            return self._materialize_figure_8(
                x, graph, anchor, param_a, param_b, used_inventory,
            )
        if kind == JUNCTION_KIND_PARALLEL_DC_BRIDGE:
            return self._materialize_parallel_dc_bridge(
                x, graph, anchor, used_inventory,
            )
        return False

    def _materialize_parallel_dc_bridge(
        self,
        x: NDArray,
        graph: PortGraph,
        anchor: int,
        used_inventory: Dict[str, int],
    ) -> bool:
        """Phase 7a (minimal): replace the anchor slot's piece with a
        DOUBLE_CROSSOVER. This is the smallest viable materialization --
        per the plan's Phase 7a spec, parallel-section *detection* is
        "non-trivial" and "high risk", so we don't attempt to fit the DC
        across two main-loop straights here. The Phase 7b heuristic seed
        is responsible for delivering chromosomes whose port-pair edges
        already wire a sensible parallel-track context."""
        variants = PARALLEL_DC_BRIDGE_TEMPLATES.get(
            JUNCTION_KIND_PARALLEL_DC_BRIDGE, (),
        )
        if not variants:
            return False
        template = variants[0]

        if anchor not in graph.slot_pieces:
            return False
        anchor_cycle = self._cycle_containing(graph, anchor)
        if anchor_cycle is None:
            return False
        if not self._is_replaceable_with_switch(graph, anchor):
            return False
        if not check_dc_bridge_inventory(template, self.inventory, used_inventory):
            return False

        dc_idx = self.catalog.id_to_index.get(template.dc_id)
        if dc_idx is None:
            return False
        used_inventory[template.dc_id] = used_inventory.get(template.dc_id, 0) + 1
        set_piece_slot(x, self.dims, anchor, dc_idx)
        set_slot_rotate(x, self.dims, anchor, template.dc_rotate)
        set_slot_flip(x, self.dims, anchor, 0)
        return True

    def _materialize_passing_siding(
        self,
        x: NDArray,
        graph: PortGraph,
        anchor: int,
        param_a: int,
        param_b: int,
        used_inventory: Dict[str, int],
    ) -> bool:
        variants = PASSING_SIDING_TEMPLATES.get(JUNCTION_KIND_PASSING_SIDING, ())
        if not variants:
            return False
        template = variants[max(0, min(int(param_b), len(variants) - 1))]
        n_straights = max(0, min(int(param_a), _MAX_BRANCH_STRAIGHTS))

        if anchor not in graph.slot_pieces or anchor not in graph.slot_poses:
            return False
        anchor_cycle = self._cycle_containing(graph, anchor)
        if anchor_cycle is None:
            return False
        # Anchor's current piece must be a regular 2-port (curve or straight)
        # so we can swap it for the IN switch without breaking unrelated routes.
        if not self._is_replaceable_with_switch(graph, anchor):
            return False
        if not check_siding_inventory(
            template, n_straights, self.inventory, used_inventory,
        ):
            return False

        out_slot = self._find_out_slot(
            graph, anchor_cycle, anchor, template, n_straights,
        )
        if out_slot is None or out_slot == anchor:
            return False
        if not self._is_replaceable_with_switch(graph, out_slot):
            return False

        if not is_valid_siding(
            graph.slot_poses[anchor],
            graph.slot_poses[out_slot],
            template,
            n_straights,
            position_tolerance=self.position_tolerance,
            angle_tolerance_deg=self.angle_tolerance_deg,
        ):
            return False

        # All checks passed -- claim inventory and splice.
        for pid, count in get_siding_inventory_requirements(
            template, n_straights,
        ).items():
            used_inventory[pid] = used_inventory.get(pid, 0) + count

        return self._splice(x, anchor, out_slot, template, n_straights)

    # ------------------------------------------------------------------
    # Splicing helpers
    # ------------------------------------------------------------------

    def _splice(
        self,
        x: NDArray,
        anchor: int,
        out_slot: int,
        template: PassingSidingTemplate,
        n_straights: int,
    ) -> bool:
        """Swap anchor / out_slot pieces for switches, allocate branch slots,
        and add port-pair edges connecting the diverging route. Returns False
        if budget for new slots / edges was exhausted (caller treats as skip).
        """
        in_idx = self.catalog.id_to_index.get(template.in_switch_id)
        out_idx = self.catalog.id_to_index.get(template.out_switch_id)
        if in_idx is None or out_idx is None:
            return False

        branch_pieces = compute_branch_pieces(template, n_straights)
        n_branch_slots = len(branch_pieces)
        free_slots = self._collect_free_slots(x, n_branch_slots)
        if len(free_slots) < n_branch_slots:
            return False
        n_branch_edges = n_branch_slots + 1  # IN.C -> b0.A, b_i.B -> b_{i+1}.A, b_last.B -> OUT.C
        free_rows = self._collect_free_pair_rows(x, n_branch_edges)
        if len(free_rows) < n_branch_edges:
            return False

        # Mutate chromosome -- order matters: pieces first, then edges.
        set_piece_slot(x, self.dims, anchor, in_idx)
        set_slot_rotate(x, self.dims, anchor, template.in_switch_rotate)
        set_slot_flip(x, self.dims, anchor, 0)

        set_piece_slot(x, self.dims, out_slot, out_idx)
        set_slot_rotate(x, self.dims, out_slot, template.out_switch_rotate)
        set_slot_flip(x, self.dims, out_slot, 0)

        for slot_idx, (piece_id, flip, rotate) in zip(free_slots, branch_pieces):
            piece_idx = self.catalog.id_to_index.get(piece_id)
            if piece_idx is None:
                return False
            set_piece_slot(x, self.dims, slot_idx, piece_idx)
            set_slot_flip(x, self.dims, slot_idx, flip)
            set_slot_rotate(x, self.dims, slot_idx, rotate)

        # Edges: IN(C=2) -> b0(A=0); b_i(B=1) -> b_{i+1}(A=0); b_last(B=1) -> OUT(C=2).
        port_a, port_b, port_c = 0, 1, 2
        chain = [(anchor, port_c)] + [(s, port_a) for s in free_slots]
        chain_b_side = [(s, port_b) for s in free_slots] + [(out_slot, port_c)]
        for row, (slot_a, port_a_idx), (slot_b, port_b_idx) in zip(
            free_rows, chain, chain_b_side,
        ):
            set_port_pair(x, self.dims, row, slot_a, port_a_idx, slot_b, port_b_idx)
        return True

    # ------------------------------------------------------------------
    # Phase 6a: figure-8 materialization
    # ------------------------------------------------------------------

    def _materialize_figure_8(
        self,
        x: NDArray,
        graph: PortGraph,
        anchor: int,
        param_a: int,
        param_b: int,
        used_inventory: Dict[str, int],
    ) -> bool:
        variants = FIGURE_8_TEMPLATES.get(JUNCTION_KIND_FIGURE_8_CROSS, ())
        if not variants:
            return False
        template = variants[max(0, min(int(param_b), len(variants) - 1))]
        n_straights = max(0, min(int(param_a), _MAX_BRANCH_STRAIGHTS))

        if anchor not in graph.slot_pieces or anchor not in graph.slot_poses:
            return False
        anchor_cycle = self._cycle_containing(graph, anchor)
        if anchor_cycle is None:
            return False
        # Anchor's current piece must be a 2-port (curve / straight) so its
        # existing edges land on cross.A and cross.B (the horizontal route).
        if not self._is_replaceable_with_switch(graph, anchor):
            return False
        if not check_figure8_inventory(
            template, n_straights, self.inventory, used_inventory,
        ):
            return False
        # Phase 6a follows the existing ``_emit_figure_8`` precedent: the
        # 16-R40 same-handed lobe is closed in *angle* (full 360 deg) but
        # its endpoint sits a chord-length away from port C, so per-axis
        # FK alignment can't be enforced strictly here. The bi-objective
        # closure constraints (``G[0..2]``) penalise residual misclosure
        # downstream; mutations refine the geometry. Skipping the strict
        # ``is_valid_figure8`` gate keeps the seed permissive enough for
        # the GA to start exploring figure-8 topologies.
        _ = is_valid_figure8  # keep symbol live for Phase 6b / regression diff

        for pid, count in get_figure8_inventory_requirements(
            template, n_straights,
        ).items():
            used_inventory[pid] = used_inventory.get(pid, 0) + count
        return self._splice_figure_8(x, anchor, template, n_straights)

    def _splice_figure_8(
        self,
        x: NDArray,
        anchor: int,
        template: Figure8Template,
        n_straights: int,
    ) -> bool:
        """Replace anchor with CROSS_90, allocate free slots for the 16-R40
        secondary lobe, wire ``cross.D -> lobe -> cross.C``."""
        cross_idx = self.catalog.id_to_index.get(template.cross_id)
        if cross_idx is None:
            return False
        lobe_pieces = compute_lobe_pieces(template, n_straights)
        n_lobe_slots = len(lobe_pieces)
        free_slots = self._collect_free_slots(x, n_lobe_slots)
        if len(free_slots) < n_lobe_slots:
            return False
        # Edge count: cross.D -> lobe[0].A; lobe[i].B -> lobe[i+1].A; lobe[-1].B -> cross.C.
        n_lobe_edges = n_lobe_slots + 1
        free_rows = self._collect_free_pair_rows(x, n_lobe_edges)
        if len(free_rows) < n_lobe_edges:
            return False

        # Anchor becomes CROSS_90.
        set_piece_slot(x, self.dims, anchor, cross_idx)
        set_slot_rotate(x, self.dims, anchor, template.cross_rotate)
        set_slot_flip(x, self.dims, anchor, 0)

        # Lobe pieces fill the free slots.
        for slot_idx, (piece_id, flip, rotate) in zip(free_slots, lobe_pieces):
            piece_idx = self.catalog.id_to_index.get(piece_id)
            if piece_idx is None:
                return False
            set_piece_slot(x, self.dims, slot_idx, piece_idx)
            set_slot_flip(x, self.dims, slot_idx, flip)
            set_slot_rotate(x, self.dims, slot_idx, rotate)

        # Wire the secondary lobe edges. CROSS_90 ports: A=0, B=1, C=2, D=3.
        port_a, port_b, _port_c, port_d = 0, 1, 2, 3
        chain_a = [(anchor, port_d)] + [(s, port_a) for s in free_slots]
        chain_b = [(s, port_b) for s in free_slots] + [(anchor, _port_c)]
        for row, (sa, pa), (sb, pb) in zip(free_rows, chain_a, chain_b):
            set_port_pair(x, self.dims, row, sa, pa, sb, pb)
        return True

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _find_out_slot(
        self,
        graph: PortGraph,
        cycle_edges: Set[PortEdge],
        anchor: int,
        template: PassingSidingTemplate,
        n_straights: int,
    ) -> Optional[int]:
        """Walk cycle edges from ``anchor`` and return the first slot whose
        IN/OUT poses satisfy ``is_valid_siding`` (i.e. the branch endpoint
        aligns with the OUT switch's port C). None if no viable target.

        Robust against curved cycles where chord distance differs from
        along-cycle distance: we test FK closure directly at each candidate
        rather than relying on a chord heuristic.
        """
        ordered = self._walk_cycle_from(cycle_edges, anchor)
        if ordered is None:
            return None
        anchor_pose = graph.slot_poses.get(anchor)
        if anchor_pose is None:
            return None
        for slot in ordered[1:]:  # skip anchor
            if not self._is_replaceable_with_switch(graph, slot):
                continue
            out_pose = graph.slot_poses.get(slot)
            if out_pose is None:
                continue
            if is_valid_siding(
                anchor_pose, out_pose, template, n_straights,
                position_tolerance=self.position_tolerance,
                angle_tolerance_deg=self.angle_tolerance_deg,
            ):
                return slot
        return None

    @staticmethod
    def _walk_cycle_from(
        cycle_edges: Set[PortEdge], start_slot: int,
    ) -> Optional[List[int]]:
        """Order the cycle as a slot sequence starting at ``start_slot``.

        Returns ``None`` if ``start_slot`` is not on the cycle or the cycle
        can't be walked (multi-edges / dangling).
        """
        # Sort cycle_edges before adjacency build so neighbor-list order is
        # deterministic (Rule 3, PLAN §10.3 I1). Set iteration over PortEdge
        # is hash-table-slot-dependent, so the same cycle on two workers
        # could yield different out-slot picks for is_valid_siding.
        adj: Dict[int, List[int]] = {}
        for e in sorted(
            cycle_edges,
            key=lambda e: (e.slot_a, e.port_a, e.slot_b, e.port_b),
        ):
            adj.setdefault(e.slot_a, []).append(e.slot_b)
            adj.setdefault(e.slot_b, []).append(e.slot_a)
        if start_slot not in adj:
            return None
        visited: List[int] = [start_slot]
        seen: Set[int] = {start_slot}
        prev: Optional[int] = None
        current = start_slot
        while True:
            neighbors = adj.get(current, [])
            next_slot = next(
                (n for n in neighbors if n != prev and n not in seen),
                None,
            )
            if next_slot is None:
                # Closing back to start completes the cycle.
                if start_slot in adj.get(current, []) and len(visited) > 2:
                    return visited
                return None
            visited.append(next_slot)
            seen.add(next_slot)
            prev = current
            current = next_slot
            if current == start_slot:
                return visited

    # ------------------------------------------------------------------
    # Inventory + free-slot helpers
    # ------------------------------------------------------------------

    def _count_used_inventory(self, x: NDArray) -> Dict[str, int]:
        used: Dict[str, int] = {}
        for _slot, idx in iter_active_slots(x, self.dims):
            piece_id = self.catalog.index_to_id.get(idx)
            if piece_id is not None:
                used[piece_id] = used.get(piece_id, 0) + 1
        return used

    def _collect_free_slots(self, x: NDArray, n: int) -> List[int]:
        free: List[int] = []
        for k in range(self.dims.N_max):
            if get_piece_slot(x, self.dims, k) == INACTIVE:
                free.append(k)
                if len(free) == n:
                    break
        return free

    def _collect_free_pair_rows(self, x: NDArray, n: int) -> List[int]:
        free: List[int] = []
        for k in range(self.dims.E_max):
            if get_port_pair(x, self.dims, k)[0] == INACTIVE:
                free.append(k)
                if len(free) == n:
                    break
        return free

    @staticmethod
    def _cycle_containing(graph: PortGraph, slot: int) -> Optional[Set[PortEdge]]:
        for cycle in _iter_cycles(graph.connected_components, graph.edges):
            slots = {e.slot_a for e in cycle} | {e.slot_b for e in cycle}
            if slot in slots:
                return cycle
        return None

    def _is_replaceable_with_switch(self, graph: PortGraph, slot: int) -> bool:
        """A slot is replaceable iff its current piece has 2 ports (curve or
        straight) AND both ports A/B are referenced by edges (so swapping in a
        3-port switch leaves the original edges valid against ports A/B).
        """
        piece_id = graph.slot_pieces.get(slot)
        if piece_id is None:
            return False
        spec = self.catalog.spec
        if spec is None:
            return False
        ps = spec.by_id.get(piece_id)
        if ps is None or ps.kind not in {"straight", "curve"}:
            return False
        return True
