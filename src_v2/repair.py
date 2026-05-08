"""Port-pair chromosome repair pipeline.

The decoder is forgiving by design — it silently drops invalid edges. Repair's
job is to canonicalize chromosomes BEFORE downstream operators see them, so
that mutation/crossover work from clean state and the GA does not waste
evaluations on chromosomes the decoder would have to clean up anyway.

Pipeline (iterated to fixed point):

1. **Edge sanitization** — drop self-loops, double-booked ports, edges to
   inactive slots, out-of-range port indices; normalize partial-INACTIVE
   rows to all-INACTIVE.
2. **Inventory enforcement** — count active piece types, deactivate excess
   slots from the end of the slot region.
3. **Repeat** — deactivating a slot can invalidate edges referencing it, so
   we iterate until one full pass produces no changes.

What this pipeline DOES NOT do (deferred):

- Closure-promoting repair (V1's curve-adjustment to drive angle deficit
  toward 360 deg) — port-graph closure is per-cycle and structurally
  different; planned for v1.
- Connectedness enforcement — disconnected components are allowed by
  design (penalized via loose-port count).
- Canonical graph hashing — eliminate_duplicates uses raw array equality
  for v0; canonical hashing planned for v1.
"""

from __future__ import annotations

<<<<<<< Updated upstream
from typing import Dict, Iterable, Optional, Set, Tuple
=======
from typing import Dict, Set, Tuple
>>>>>>> Stashed changes

from numpy.typing import NDArray
from pymoo.core.repair import Repair

from .catalog import TrackCatalog
<<<<<<< Updated upstream
from .decoder import DecoderConfig, _iter_cycles, decode_chromosome
from .encoding import (
    INACTIVE,
    JUNCTION_GENES,
    JUNCTION_KIND_MAX,
    JUNCTION_PARAM_MAX,
    JUNC_ACTIVE_OFFSET,
    JUNC_ANCHOR_OFFSET,
    JUNC_KIND_OFFSET,
    JUNC_PARAM_A_OFFSET,
    JUNC_PARAM_B_OFFSET,
    PortPairDimensions,
    clear_port_pair,
    get_piece_slot,
    get_port_pair,
    iter_active_slots,
    set_piece_slot,
    set_port_pair,
    set_slot_flip,
    set_slot_rotate,
)
from .structural_mutations import introduce_crossing, mutate_grow_branch
from .types import PortEdge


# R40-only closed cycle: 16 pieces × 22.5° = 360° exactly.
_R40_PIECES_PER_CLOSED_CYCLE: int = 16
=======
from .decoder import DecoderConfig
from .encoding import (
    INACTIVE,
    PortPairDimensions,
    clear_port_pair,
    get_port_pair,
    iter_active_slots,
    set_piece_slot,
)
from .structural_mutations import introduce_crossing
>>>>>>> Stashed changes


class PortPairRepairPipeline(Repair):
    """Composite repair: edge sanitization + inventory enforcement, iterated."""

    def __init__(
        self,
        dims: PortPairDimensions,
        catalog: TrackCatalog,
        inventory: Dict[str, int],
        max_iterations: int = 5,
        crossing_injection_max: int = 4,
        decoder_config: DecoderConfig | None = None,
    ) -> None:
        super().__init__()
        self.dims = dims
        self.catalog = catalog
        self.inventory = inventory
        self.inventory_by_index: Dict[int, int] = {
            catalog.id_to_index[pid]: count
            for pid, count in inventory.items()
            if pid in catalog.id_to_index
        }
        self.max_iterations = max_iterations
        self.crossing_injection_max = crossing_injection_max
        self.decoder_config = decoder_config or DecoderConfig()
<<<<<<< Updated upstream
        # Toggled by FinalizationGatingCallback at termination.perc > 0.9.
        # When True, _enforce_route_completeness attempts branch growth on
        # incomplete switches (Phase 3 ships the branch_grow.find_branch_path
        # implementation; meanwhile the attempt stub returns False = revert).
        self.finalization_active: bool = False
=======
>>>>>>> Stashed changes

    def _do(self, problem, X, **kwargs) -> NDArray:
        for i in range(len(X)):
            self._repair_one(X[i])
        return X

    def _repair_one(self, x: NDArray) -> None:
<<<<<<< Updated upstream
        # Orientation-bit normalization: zero flip / rotate on every slot
        # whose piece doesn't support them, or is inactive. Cheap and
        # idempotent; runs once per individual since later steps don't
        # reintroduce flip / rotate bits.
        self._normalize_orientations(x)
        for _ in range(self.max_iterations):
            changed_edges = self._sanitize_edges(x)
            changed_routes = self._enforce_route_completeness(x)
            changed_inv = self._enforce_inventory(x)
            if not (changed_edges or changed_routes or changed_inv):
                break
        # After inventory may have deactivated slots, normalize again to
        # zero flip/rotate on the now-inactive slots (cosmetic, but keeps
        # the chromosome in a canonical form for diff/eliminate_duplicates).
        self._normalize_orientations(x)
        # Phase 4: clamp junction descriptors to valid bounds, then deactivate
        # any whose anchor_slot points to an INACTIVE slot in the post-repair
        # context. Templates ship in Phase 5+; for now this just keeps the
        # chromosome canonical.
        self._normalize_junctions(x)
        # Cycle-closure repair lives in PortPairProblem._evaluate as of
        # Phase 1.B (Rule 24 revised — Baldwinian: surgery on a clone, raw
        # x stays in the pool so NSGA-II selection/crossover preserve
        # genotypic diversity). The repaired graph is exposed via
        # out["pheno"]; F/G derive from it.
=======
        for _ in range(self.max_iterations):
            changed_edges = self._sanitize_edges(x)
            changed_inv = self._enforce_inventory(x)
            if not (changed_edges or changed_inv):
                break
>>>>>>> Stashed changes
        # After structural repair settles, opportunistically convert
        # near-perpendicular self-intersections to real CROSS_90 pieces.
        # This mirrors V1's CROSS_90 repair injection but with proper
        # graph surgery (both routes wired) rather than a type swap.
        if self.crossing_injection_max > 0 and self.catalog.spec is not None:
            for _ in range(self.crossing_injection_max):
                injected = introduce_crossing(
                    x, self.dims, self.catalog, self.decoder_config,
                    self.inventory,
                )
                if not injected:
                    break
            # Re-sanitize in case injection introduced any partial rows
            self._sanitize_edges(x)
            self._enforce_inventory(x)

    # ------------------------------------------------------------------
<<<<<<< Updated upstream
    # Orientation-bit normalization (flip + rotate)
    # ------------------------------------------------------------------

    def _normalize_junctions(self, x: NDArray) -> None:
        """Phase 4: clamp every junction's 5 genes to canonical bounds and
        deactivate junctions whose anchor_slot is INACTIVE in the post-repair
        slot context (Rule 4 / Phase 4 spec). Param semantics belong to the
        Phase 5+ templates; here we only enforce the structural envelope."""
        dims = self.dims
        for j in range(dims.J_max):
            base = dims.junc_start + j * JUNCTION_GENES
            # Clamp each gene to its declared range.
            x[base + JUNC_ACTIVE_OFFSET] = max(
                0, min(1, int(x[base + JUNC_ACTIVE_OFFSET])),
            )
            x[base + JUNC_ANCHOR_OFFSET] = max(
                0, min(max(0, dims.N_max - 1), int(x[base + JUNC_ANCHOR_OFFSET])),
            )
            x[base + JUNC_KIND_OFFSET] = max(
                0, min(JUNCTION_KIND_MAX, int(x[base + JUNC_KIND_OFFSET])),
            )
            x[base + JUNC_PARAM_A_OFFSET] = max(
                0, min(JUNCTION_PARAM_MAX, int(x[base + JUNC_PARAM_A_OFFSET])),
            )
            x[base + JUNC_PARAM_B_OFFSET] = max(
                0, min(JUNCTION_PARAM_MAX, int(x[base + JUNC_PARAM_B_OFFSET])),
            )
            # Deactivate if the anchor_slot is INACTIVE in the current chromosome.
            if x[base + JUNC_ACTIVE_OFFSET] == 1:
                anchor_slot = int(x[base + JUNC_ANCHOR_OFFSET])
                if int(x[dims.slot_start + anchor_slot]) == INACTIVE:
                    x[base + JUNC_ACTIVE_OFFSET] = 0

    def _normalize_orientations(self, x: NDArray) -> None:
        """Zero flip and rotate bits on inactive slots, and on slots whose
        piece spec doesn't support the corresponding orientation. The
        decoder enforces this internally as a safety net; running it here
        keeps the canonical chromosome form clean (eliminate_duplicates,
        diffability, etc.)."""
        spec = self.catalog.spec
        active = dict(iter_active_slots(x, self.dims))
        for slot_idx in range(self.dims.N_max):
            piece_index = active.get(slot_idx, INACTIVE)
            if piece_index == INACTIVE:
                set_slot_flip(x, self.dims, slot_idx, 0)
                set_slot_rotate(x, self.dims, slot_idx, 0)
                continue
            piece_id = self.catalog.index_to_id.get(piece_index)
            ps = spec.by_id.get(piece_id) if (spec and piece_id) else None
            if ps is None or not ps.symmetric:
                set_slot_flip(x, self.dims, slot_idx, 0)
            if ps is None or not ps.rotatable:
                set_slot_rotate(x, self.dims, slot_idx, 0)

    # ------------------------------------------------------------------
=======
>>>>>>> Stashed changes
    # Edge sanitization
    # ------------------------------------------------------------------

    def _sanitize_edges(self, x: NDArray) -> bool:
        """Drop invalid port-pair rows; normalize partial-INACTIVE rows."""
        changed = False
        used_ports: Set[Tuple[int, int]] = set()
        active_slots = dict(iter_active_slots(x, self.dims))
        spec = self.catalog.spec

        for k in range(self.dims.E_max):
            sa, pa, sb, pb = get_port_pair(x, self.dims, k)

            # Canonical inactive row — leave alone
            if sa == INACTIVE and pa == INACTIVE and sb == INACTIVE and pb == INACTIVE:
                continue

            # Partial INACTIVE → normalize to fully INACTIVE
            if INACTIVE in (sa, pa, sb, pb):
                clear_port_pair(x, self.dims, k)
                changed = True
                continue

            # Self-loop
            if sa == sb:
                clear_port_pair(x, self.dims, k)
                changed = True
                continue

            # Edge to inactive slot
            if sa not in active_slots or sb not in active_slots:
                clear_port_pair(x, self.dims, k)
                changed = True
                continue

            # Out-of-range port for the piece kind in that slot (V2 spec only)
            if spec is not None:
                piece_a_id = self.catalog.index_to_id.get(active_slots[sa])
                piece_b_id = self.catalog.index_to_id.get(active_slots[sb])
                if piece_a_id is None or piece_b_id is None:
                    clear_port_pair(x, self.dims, k)
                    changed = True
                    continue
                spec_a = spec.by_id.get(piece_a_id)
                spec_b = spec.by_id.get(piece_b_id)
                if spec_a is None or spec_b is None:
                    clear_port_pair(x, self.dims, k)
                    changed = True
                    continue
                if pa >= len(spec_a.ports) or pb >= len(spec_b.ports):
                    clear_port_pair(x, self.dims, k)
                    changed = True
                    continue

            # Double-booked port (first occurrence wins)
            if (sa, pa) in used_ports or (sb, pb) in used_ports:
                clear_port_pair(x, self.dims, k)
                changed = True
                continue

            used_ports.add((sa, pa))
            used_ports.add((sb, pb))

        return changed

    # ------------------------------------------------------------------
    # Inventory enforcement
    # ------------------------------------------------------------------

    def _enforce_inventory(self, x: NDArray) -> bool:
        """Deactivate excess slots from end of slot region per piece type."""
        usage: Dict[int, int] = {}
        for _, piece_index in iter_active_slots(x, self.dims):
            usage[piece_index] = usage.get(piece_index, 0) + 1

        violations: Dict[int, int] = {}
        for piece_index, count in usage.items():
            limit = self.inventory_by_index.get(piece_index, 0)
            if count > limit:
                violations[piece_index] = count - limit

        if not violations:
            return False

        for slot_idx in range(self.dims.N_max - 1, -1, -1):
            if not violations:
                break
<<<<<<< Updated upstream
            piece_index = int(x[self.dims.slot_start + slot_idx])
=======
            piece_index = int(x[slot_idx])
>>>>>>> Stashed changes
            if piece_index == INACTIVE:
                continue
            if piece_index in violations:
                set_piece_slot(x, self.dims, slot_idx, INACTIVE)
                violations[piece_index] -= 1
                if violations[piece_index] <= 0:
                    del violations[piece_index]

        return True
<<<<<<< Updated upstream

    # ------------------------------------------------------------------
    # Route completeness (Phase 2)
    # ------------------------------------------------------------------

    def _enforce_route_completeness(self, x: NDArray) -> bool:
        """Bond-graph completeness pass.

        During exploration (``self.finalization_active == False``): only
        validates crossing pair-sets (drops pair-rows that don't form a
        valid catalog route — kills CROSS_90 ``{A↔C}`` and DOUBLE_CROSSOVER
        union case).

        During finalization (``self.finalization_active == True``): for
        each incomplete switch / crossing, *attempt* to grow a closing
        branch. Phase 3 ships the A* implementation; meanwhile attempts
        return False (revert to leave the switch incomplete; G[5+T] soft
        penalty applies).

        Returns True iff x changed.
        """
        changed = self._validate_crossing_pair_sets(x)

        if self.spec is None:
            return changed

        handlers = self._completeness_handlers()
        for slot_idx, piece_index in list(iter_active_slots(x, self.dims)):
            piece_id = self.catalog.index_to_id.get(piece_index)
            ps = self.spec.by_id.get(piece_id) if piece_id else None
            if ps is None:
                continue
            check, repair = handlers.get(ps.kind, (None, None))
            if check is None:
                continue
            if not check(x, slot_idx, ps):
                continue
            if self.finalization_active and repair is not None:
                if repair(x, slot_idx, ps):
                    changed = True
        return changed

    def _completeness_handlers(self):
        """Registry: kind → (is_incomplete_fn, attempt_repair_fn).

        Avoids if/elif chains. Functions accept ``(chromosome, slot_idx,
        piece_spec)`` and return bool. ``attempt_repair_fn`` returns True
        iff x mutated.
        """
        return {
            "switch": (_switch_is_incomplete_factory(self.dims),
                       self._attempt_switch_completion),
            "crossing": (_crossing_is_incomplete_factory(self.dims),
                         self._attempt_crossing_completion),
        }

    def _attempt_switch_completion(self, x: NDArray, slot_idx: int, piece_spec) -> bool:
        """Try to close port C of an incomplete switch via branch growth.

        Delegates to :func:`mutate_grow_branch`, which finds a partner
        incomplete switch on the same through-cycle and A*-searches a
        closing branch. ``mutate_grow_branch`` is rollback-safe: on A*
        failure it returns False without mutating ``x``, so the switch
        stays incomplete and the soft G[5+T] penalty applies. Idempotent
        across the iteration in ``_enforce_route_completeness`` — once a
        pair is closed, subsequent calls within the same repair pass
        either close another pair or no-op cleanly.
        """
        return mutate_grow_branch(
            x, self.dims, self.catalog, self.decoder_config, self.inventory,
            rng=None,
        )

    def _attempt_crossing_completion(self, x: NDArray, slot_idx: int, piece_spec) -> bool:
        """Phase 3 stub: try to close the second route through ports C↔D
        when only A↔B is paired (rare success; symmetric with switch path).
        See ``_attempt_switch_completion`` docstring for status.
        """
        return False

    def _validate_crossing_pair_sets(self, x: NDArray) -> bool:
        """Drop pair-rows on crossing slots whose port-pair isn't in
        ``piece_spec.routes``.

        Catches:
        - CROSS_90 ``{A↔C}`` or ``{A↔D}`` etc. (only A-B and C-D are valid)
        - DOUBLE_CROSSOVER union case where 4 pair-rows over-book ports
          (real 4DBrix part 210.1 picks ONE of the two route sets:
          parallel ``{A-B, C-D}`` or crossover ``{A-D, C-B}``, never both)

        Returns True iff any pair-row was cleared.
        """
        if self.spec is None:
            return False

        active_slots = dict(iter_active_slots(x, self.dims))
        changed = False

        for k in range(self.dims.E_max):
            sa, pa, sb, pb = get_port_pair(x, self.dims, k)
            if sa == INACTIVE:
                continue
            piece_a_id = self.catalog.index_to_id.get(active_slots.get(sa))
            piece_b_id = self.catalog.index_to_id.get(active_slots.get(sb))
            spec_a = self.spec.by_id.get(piece_a_id) if piece_a_id else None
            spec_b = self.spec.by_id.get(piece_b_id) if piece_b_id else None

            if not _is_invalid_crossing_pair(spec_a, pa, spec_b, pb):
                continue
            clear_port_pair(x, self.dims, k)
            changed = True

        return changed

    @property
    def spec(self):
        """Convenience accessor — repair frequently consults the V2 spec."""
        return self.catalog.spec


# =============================================================================
# Module-level helpers (factory closures keep dims local without globals)
# =============================================================================


def _switch_is_incomplete_factory(dims: PortPairDimensions):
    """Return a check fn: (x, slot_idx, piece_spec) -> bool.

    A switch is incomplete iff its set of paired ports != ``ps.ports.keys()``
    (i.e., at least one of A/B/C is loose).
    """
    def _is_incomplete(x, slot_idx: int, piece_spec) -> bool:
        paired = _paired_ports_for_slot(x, dims, slot_idx, piece_spec)
        return paired != set(piece_spec.ports.keys())
    return _is_incomplete


def _crossing_is_incomplete_factory(dims: PortPairDimensions):
    """A crossing is incomplete iff its paired-port set isn't one of the
    valid route unions. For CROSS_90 the only valid full set is {A,B,C,D}
    (both routes engaged). For DOUBLE_CROSSOVER, valid is {A,B,C,D} via
    EITHER the parallel or the crossover route bundle.
    """
    def _is_incomplete(x, slot_idx: int, piece_spec) -> bool:
        paired = _paired_ports_for_slot(x, dims, slot_idx, piece_spec)
        return paired != set(piece_spec.ports.keys())
    return _is_incomplete


def _paired_ports_for_slot(
    x, dims: PortPairDimensions, slot_idx: int, piece_spec,
) -> set:
    """Set of port-name strings that appear paired (in any active edge)
    for the given slot. Port indices map to names via ``tuple(piece_spec.ports)``
    (A=0, B=1, C=2, D=3 by catalog convention).
    """
    port_names = tuple(piece_spec.ports)
    paired: set = set()
    for k in range(dims.E_max):
        sa, pa, sb, pb = get_port_pair(x, dims, k)
        if sa == INACTIVE:
            continue
        if sa == slot_idx and 0 <= pa < len(port_names):
            paired.add(port_names[pa])
        if sb == slot_idx and 0 <= pb < len(port_names):
            paired.add(port_names[pb])
    return paired


def _is_invalid_crossing_pair(spec_a, pa: int, spec_b, pb: int) -> bool:
    """True iff at least one side is a crossing whose port index isn't in
    any of its catalog routes' port lists.

    Lightweight check: a route appears as an ordered list of port names in
    ``piece_spec.routes`` (e.g., ``{"horizontal": ["A", "B"]}`` for
    CROSS_90). We allow a port if it appears in *any* route list. A port
    not in any route is invalid by definition (the catalog declares no
    train traversal involving it).
    """
    return (
        _port_outside_all_routes(spec_a, pa)
        or _port_outside_all_routes(spec_b, pb)
    )


def _port_outside_all_routes(piece_spec, port_idx: int) -> bool:
    """True iff this is a crossing piece AND ``port_idx`` doesn't appear
    in any of its catalog route port-sequences."""
    if piece_spec is None or piece_spec.kind != "crossing":
        return False
    port_names = tuple(piece_spec.ports)
    if not 0 <= port_idx < len(port_names):
        return True
    name = port_names[port_idx]
    return not any(name in seq for seq in piece_spec.routes.values())


# =============================================================================
# Cycle-closure repair (Phase 1, PLAN §5 + Rule 24 revised)
# =============================================================================


class CycleClosureRepair:
    """Cycle-aware closure repair (Phase 1 of PLAN, §10.2 1.1-1.7).

    Adjusts the number of R40 curves in each cycle of the decoded
    :class:`PortGraph` so the cycle's piece count drives toward the closed-
    loop target (16 R40 = 360°). Splices new R40s into existing cycle edges
    (deficit case) or deactivates R40 slots and merges adjacent edges
    (excess case), using the canonical 5-step edge-surgery pattern (cf.
    :func:`introduce_crossing`).

    The ``repair_one(x)`` method mutates ``x`` in place — the Baldwinian
    contract (Rule 24 revised) is enforced one level up by
    :meth:`PortPairProblem._evaluate`, which clones the chromosome before
    invoking this method, so the population pool retains the raw genotype
    while F/G derive from the repaired phenotype published on ``out["pheno"]``.

    The ``skip_anchor_slots`` parameter is a Coupling-C placeholder for
    Phase 5+: when junctions exist, their anchor slots are passed in here
    so this repair won't undo asymmetric-oval seeds.
    """

    def __init__(
        self,
        dims: PortPairDimensions,
        catalog: TrackCatalog,
        decoder_config: DecoderConfig,
        inventory: Dict[str, int],
        max_corrections: int = 16,
    ) -> None:
        self.dims = dims
        self.catalog = catalog
        self.decoder_config = decoder_config
        self.inventory = dict(inventory)
        self.max_corrections = max_corrections
        self.r40_idx = catalog.id_to_index.get("R40_CURVE")

    def repair_one(
        self,
        x: NDArray,
        skip_anchor_slots: Optional[Set[int]] = None,
    ) -> NDArray:
        """Apply cycle-closure repair to ``x`` in place. Returns ``x``."""
        if self.r40_idx is None:
            return x
        skip: Set[int] = set(skip_anchor_slots) if skip_anchor_slots else set()

        graph = decode_chromosome(x, self.dims, self.catalog, self.decoder_config)
        if not graph.connected_components:
            return x

        used = self._count_active_pieces(x)
        r40_cap = self.inventory.get("R40_CURVE", 0)
        cycles = sorted(
            _iter_cycles(graph.connected_components, graph.edges),
            key=len, reverse=True,
        )

        corrections = 0
        for cycle_edges in cycles:
            if corrections >= self.max_corrections:
                break

            cycle_slots = _slots_in_cycle(cycle_edges)
            if cycle_slots & skip:
                continue

            cycle_r40 = sum(
                1 for s in cycle_slots
                if get_piece_slot(x, self.dims, s) == self.r40_idx
            )
            delta = _R40_PIECES_PER_CLOSED_CYCLE - cycle_r40
            if delta == 0:
                continue

            if delta > 0:
                available = r40_cap - used.get(self.r40_idx, 0)
                budget = self.max_corrections - corrections
                to_add = min(delta, available, budget)
                if to_add <= 0:
                    continue
                added = self._add_r40s_to_cycle(x, cycle_edges, to_add)
                used[self.r40_idx] = used.get(self.r40_idx, 0) + added
                corrections += added
            else:
                budget = self.max_corrections - corrections
                to_remove = min(-delta, budget)
                removed = self._remove_r40s_from_cycle(
                    x, cycle_slots, to_remove,
                )
                used[self.r40_idx] = used.get(self.r40_idx, 0) - removed
                corrections += removed

        return x

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _count_active_pieces(self, x: NDArray) -> Dict[int, int]:
        counts: Dict[int, int] = {}
        for _slot, idx in iter_active_slots(x, self.dims):
            counts[idx] = counts.get(idx, 0) + 1
        return counts

    def _port_idx(self, piece_idx: int, port_name: str) -> Optional[int]:
        piece_id = self.catalog.index_to_id.get(piece_idx)
        if piece_id is None:
            return None
        spec = self.catalog.spec.by_id.get(piece_id) if self.catalog.spec else None
        if spec is None or port_name not in spec.ports:
            return None
        return tuple(spec.ports).index(port_name)

    def _find_row_for_edge(self, x: NDArray, edge: PortEdge) -> Optional[int]:
        """Return the row index holding ``edge`` (matched as an unordered pair)."""
        target_a = (
            edge.slot_a,
            self._port_idx(get_piece_slot(x, self.dims, edge.slot_a), edge.port_a),
        )
        target_b = (
            edge.slot_b,
            self._port_idx(get_piece_slot(x, self.dims, edge.slot_b), edge.port_b),
        )
        if None in (target_a[1], target_b[1]):
            return None
        for k in range(self.dims.E_max):
            sa, pa, sb, pb = get_port_pair(x, self.dims, k)
            if sa == INACTIVE:
                continue
            row_a, row_b = (sa, pa), (sb, pb)
            if {row_a, row_b} == {target_a, target_b}:
                return k
        return None

    def _add_r40s_to_cycle(
        self,
        x: NDArray,
        cycle_edges: Set[PortEdge],
        n_to_add: int,
    ) -> int:
        """Splice up to ``n_to_add`` R40 pieces into distinct cycle edges."""
        free_slots = [
            k for k in range(self.dims.N_max)
            if get_piece_slot(x, self.dims, k) == INACTIVE
        ]
        free_rows = [
            k for k in range(self.dims.E_max)
            if get_port_pair(x, self.dims, k)[0] == INACTIVE
        ]
        edges_list = list(cycle_edges)

        n_actual = min(n_to_add, len(free_slots), len(free_rows), len(edges_list))
        added = 0
        for i in range(n_actual):
            edge = edges_list[i]
            new_slot = free_slots[i]
            new_row = free_rows[i]

            edge_row = self._find_row_for_edge(x, edge)
            if edge_row is None:
                continue
            pa_idx = self._port_idx(get_piece_slot(x, self.dims, edge.slot_a), edge.port_a)
            pb_idx = self._port_idx(get_piece_slot(x, self.dims, edge.slot_b), edge.port_b)
            if pa_idx is None or pb_idx is None:
                continue

            set_piece_slot(x, self.dims, new_slot, self.r40_idx)
            # Replace ``edge`` with two edges through new_slot:
            #   (slot_a, port_a) <-> (new_slot, A=0)
            #   (new_slot, B=1)  <-> (slot_b, port_b)
            set_port_pair(
                x, self.dims, edge_row,
                edge.slot_a, pa_idx, new_slot, 0,
            )
            set_port_pair(
                x, self.dims, new_row,
                new_slot, 1, edge.slot_b, pb_idx,
            )
            added += 1
        return added

    def _remove_r40s_from_cycle(
        self,
        x: NDArray,
        cycle_slots: Set[int],
        n_to_remove: int,
    ) -> int:
        """Deactivate up to ``n_to_remove`` R40 slots in the cycle and merge their edges."""
        removed = 0
        candidates = [
            s for s in cycle_slots
            if get_piece_slot(x, self.dims, s) == self.r40_idx
        ]
        for target_slot in candidates:
            if removed >= n_to_remove:
                break
            if not self._remove_one_r40(x, target_slot):
                continue
            removed += 1
        return removed

    def _remove_one_r40(self, x: NDArray, target_slot: int) -> bool:
        rows_with_slot = []
        for k in range(self.dims.E_max):
            sa, pa, sb, pb = get_port_pair(x, self.dims, k)
            if sa == INACTIVE:
                continue
            if sa == target_slot or sb == target_slot:
                rows_with_slot.append((k, sa, pa, sb, pb))
        if len(rows_with_slot) != 2:
            return False  # not a two-edge slot — refuse to remove

        (row1, sa1, pa1, sb1, pb1), (row2, sa2, pa2, sb2, pb2) = rows_with_slot
        n1, p_n1 = (sb1, pb1) if sa1 == target_slot else (sa1, pa1)
        n2, p_n2 = (sb2, pb2) if sa2 == target_slot else (sa2, pa2)

        set_piece_slot(x, self.dims, target_slot, INACTIVE)
        set_port_pair(x, self.dims, row1, n1, p_n1, n2, p_n2)
        clear_port_pair(x, self.dims, row2)
        return True


def _slots_in_cycle(cycle_edges: Iterable[PortEdge]) -> Set[int]:
    slots: Set[int] = set()
    for edge in cycle_edges:
        slots.add(edge.slot_a)
        slots.add(edge.slot_b)
    return slots
=======
>>>>>>> Stashed changes
