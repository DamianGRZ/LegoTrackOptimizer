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

from typing import Dict, Set, Tuple

from numpy.typing import NDArray
from pymoo.core.repair import Repair

from .catalog import TrackCatalog
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

    def _do(self, problem, X, **kwargs) -> NDArray:
        for i in range(len(X)):
            self._repair_one(X[i])
        return X

    def _repair_one(self, x: NDArray) -> None:
        for _ in range(self.max_iterations):
            changed_edges = self._sanitize_edges(x)
            changed_inv = self._enforce_inventory(x)
            if not (changed_edges or changed_inv):
                break
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
            piece_index = int(x[slot_idx])
            if piece_index == INACTIVE:
                continue
            if piece_index in violations:
                set_piece_slot(x, self.dims, slot_idx, INACTIVE)
                violations[piece_index] -= 1
                if violations[piece_index] <= 0:
                    del violations[piece_index]

        return True
