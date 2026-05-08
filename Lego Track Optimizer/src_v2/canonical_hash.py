"""Canonical graph hashing for port-pair chromosomes.

The default ``eliminate_duplicates=True`` in pymoo uses raw array equality on
the int16 chromosome, which means two structurally-identical layouts that
differ by anchor pose, slot-index permutation, or edge ordering count as
distinct. Result: the population fills with anchor-shifted clones of the
same oval and crowding distance can't tell them apart.

This module computes a hash that is invariant to:

- **Anchor pose** — the (x, y, theta) trio is excluded from the signature.
- **Slot-index permutation** — slots are relabeled in BFS order from a
  canonical root, so swapping the contents of slot 3 and slot 17 produces the
  same hash if they're isomorphic in the port graph.
- **Edge ordering inside the chromosome's pair region.**
- **Edge endpoint ordering** — ``(a, b) == (b, a)``.

The output is suitable for ``pymoo.core.duplicate.ElementwiseDuplicateElimination``.

Algorithm — standard graph canonicalization:

1. Decode the chromosome to a port graph (just slots + sanitized edges; we
   don't need pose propagation for the hash).
2. Pick a canonical root: lowest piece-type-id; tiebreak by lowest degree;
   tiebreak by lowest slot index.
3. BFS from root, assigning new labels 0, 1, 2, ... in visit order. When the
   queue offers multiple choices, expand them in deterministic order
   (sorted by (piece_type_id, port_a_name, port_b_name)).
4. Serialize: piece-id sequence + sorted relabeled edge tuples.
5. Hash with blake2b for speed and determinism.

Cost: O(V + E log E). Negligible vs decode + BFS pose propagation.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple

from numpy.typing import NDArray

from .encoding import (
    INACTIVE,
    PortPairDimensions,
    iter_active_pairs,
    iter_active_slots,
)


# =============================================================================
# Public entry point
# =============================================================================


def canonical_graph_signature(
    x: NDArray, dims: PortPairDimensions, catalog,
) -> bytes:
    """Return a 16-byte canonical signature for the chromosome's port graph.

    Structurally-identical layouts produce identical signatures regardless
    of slot ordering, edge ordering, edge endpoint ordering, or anchor pose.
    """
    slot_pieces, edges = _read_topology(x, dims, catalog)
    if not slot_pieces:
        return b"\x00" * 16  # empty graph collapses to one bucket

    components = _connected_components(slot_pieces, edges)

    # Per-component canonical strings, then sort across components so the
    # final signature is invariant to which component happens to be hit first.
    component_strings: List[bytes] = []
    for component in components:
        component_strings.append(
            _canonical_component_string(component, slot_pieces, edges, catalog),
        )
    component_strings.sort()
    payload = b"|".join(component_strings)
    return hashlib.blake2b(payload, digest_size=16).digest()


# =============================================================================
# Topology extraction (decode-lite — no pose propagation)
# =============================================================================


def _read_topology(
    x: NDArray, dims: PortPairDimensions, catalog,
) -> Tuple[Dict[int, str], List[Tuple[int, str, int, str]]]:
    """Read active slots and sanitized edges with port names.

    Drops self-loops and double-booked ports, matching ``decoder._sanitize_edges``.
    Returns (slot_pieces, edges) where each edge is (slot_a, port_a_name,
    slot_b, port_b_name).
    """
    spec = catalog.spec
    index_to_id = catalog.index_to_id

    slot_pieces: Dict[int, str] = {}
    for slot_idx, piece_index in iter_active_slots(x, dims):
        piece_id = index_to_id.get(piece_index)
        if piece_id is not None:
            slot_pieces[slot_idx] = piece_id

    used_ports: Set[Tuple[int, str]] = set()
    edges: List[Tuple[int, str, int, str]] = []
    for _, sa, pa, sb, pb in iter_active_pairs(x, dims):
        if sa == sb:
            continue
        if sa not in slot_pieces or sb not in slot_pieces:
            continue
        if INACTIVE in (sa, pa, sb, pb):
            continue
        if spec is None:
            continue
        spec_a = spec.by_id.get(slot_pieces[sa])
        spec_b = spec.by_id.get(slot_pieces[sb])
        if spec_a is None or spec_b is None:
            continue
        names_a = list(spec_a.ports)
        names_b = list(spec_b.ports)
        if pa < 0 or pa >= len(names_a):
            continue
        if pb < 0 or pb >= len(names_b):
            continue
        port_a_name = names_a[pa]
        port_b_name = names_b[pb]
        key_a = (sa, port_a_name)
        key_b = (sb, port_b_name)
        if key_a in used_ports or key_b in used_ports:
            continue
        used_ports.add(key_a)
        used_ports.add(key_b)
        edges.append((sa, port_a_name, sb, port_b_name))

    return slot_pieces, edges


def _connected_components(
    slot_pieces: Dict[int, str],
    edges: List[Tuple[int, str, int, str]],
) -> List[Set[int]]:
    """Union-find connected components on slots."""
    parent: Dict[int, int] = {s: s for s in slot_pieces}

    def find(v: int) -> int:
        root = v
        while parent[root] != root:
            root = parent[root]
        while parent[v] != root:
            parent[v], v = root, parent[v]
        return root

    for sa, _, sb, _ in edges:
        if sa in parent and sb in parent:
            ra, rb = find(sa), find(sb)
            if ra != rb:
                parent[ra] = rb

    groups: Dict[int, Set[int]] = {}
    for slot in slot_pieces:
        groups.setdefault(find(slot), set()).add(slot)
    return list(groups.values())


# =============================================================================
# Per-component canonicalization
# =============================================================================


def _canonical_component_string(
    component: Set[int],
    slot_pieces: Dict[int, str],
    edges: List[Tuple[int, str, int, str]],
    catalog,
) -> bytes:
    """Build a canonical byte string for a connected component.

    BFS-relabel strategy: start from the lowest-(piece_type_id, degree, slot)
    root, expand neighbours in deterministic order, emit piece sequence +
    sorted relabeled edge tuples.
    """
    id_to_index = catalog.id_to_index

    # Component-local incidence map
    incidence: Dict[int, List[Tuple[int, str, int, str]]] = defaultdict(list)
    for edge in edges:
        sa, pa, sb, pb = edge
        if sa in component and sb in component:
            incidence[sa].append(edge)
            incidence[sb].append(edge)

    # Pick canonical root: lowest piece_type_id, tiebreak lowest degree,
    # tiebreak lowest slot index.
    def root_key(slot: int) -> Tuple[int, int, int]:
        piece_id = slot_pieces[slot]
        return (
            id_to_index.get(piece_id, 0),
            len(incidence.get(slot, [])),
            slot,
        )

    root = min(component, key=root_key)

    # BFS with deterministic neighbour expansion
    relabel: Dict[int, int] = {root: 0}
    queue = deque([root])
    next_label = 1
    visited_edges: Set[int] = set()

    while queue:
        current = queue.popleft()
        # Order neighbours by (piece_type_id_of_neighbour, port_pair_names, slot)
        # so the BFS order is canonical.
        candidates: List[Tuple[Tuple, int, str, int, str]] = []
        for edge in incidence.get(current, []):
            if id(edge) in visited_edges:
                continue
            sa, pa, sb, pb = edge
            if sa == current:
                other, port_self, port_other = sb, pa, pb
            else:
                other, port_self, port_other = sa, pb, pa
            other_piece_idx = id_to_index.get(slot_pieces[other], 0)
            sort_key = (other_piece_idx, port_self, port_other, other)
            candidates.append((sort_key, other, port_self, port_other))
            visited_edges.add(id(edge))

        candidates.sort(key=lambda c: c[0])

        for _, other, _, _ in candidates:
            if other not in relabel:
                relabel[other] = next_label
                next_label += 1
                queue.append(other)

    # Slots that weren't reached (impossible if `component` is a real
    # connected component, but be defensive): assign trailing labels in
    # piece-id order.
    for slot in sorted(component - relabel.keys(),
                       key=lambda s: (id_to_index.get(slot_pieces[s], 0), s)):
        relabel[slot] = next_label
        next_label += 1

    # Build canonical piece-id sequence in new-label order
    piece_seq = [slot_pieces[s] for s, _ in sorted(relabel.items(), key=lambda kv: kv[1])]

    # Build canonical edge list: relabel both endpoints, canonicalize endpoint
    # ordering (smaller-(label, port) first), then sort the whole list.
    canon_edges: List[Tuple[int, str, int, str]] = []
    for sa, pa, sb, pb in edges:
        if sa not in relabel or sb not in relabel:
            continue
        end_a = (relabel[sa], pa)
        end_b = (relabel[sb], pb)
        if end_b < end_a:
            end_a, end_b = end_b, end_a
        canon_edges.append((end_a[0], end_a[1], end_b[0], end_b[1]))
    canon_edges.sort()

    # Serialize: pieces joined by '/', edges joined by ';'
    pieces_blob = "/".join(piece_seq).encode("utf-8")
    edges_blob = ";".join(
        f"{a}.{pa}-{b}.{pb}" for a, pa, b, pb in canon_edges
    ).encode("utf-8")
    return b"P:" + pieces_blob + b"|E:" + edges_blob
