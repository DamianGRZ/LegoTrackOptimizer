"""Canonical PortGraph hashing and phenotype-based duplicate elimination.

The genotype-to-phenotype mapping for the port-pair encoding is many-to-one
(slot index permutation, edge row order, anchor placement, junk components
all leave the useful topology invariant). pymoo's default
``eliminate_duplicates=True`` compares raw ``X`` arrays and so misses every
one of these redundancies. ``canonical_graph_hash`` collapses all such
redundant chromosomes to one byte string, which a custom
:class:`ElementwiseDuplicateElimination` can compare in O(1).

Algorithm — Poziom 1 BFS canonization with lex-min over candidate starts:

1. Drop components below ``MIN_USEFUL_COMPONENT_SIZE`` (matches the fitness
   filter in :mod:`src_v2.problem`).
2. For each remaining component, find the set of starting-slot candidates
   sharing the lex-min initial signature ``(piece_id, flip, rotate, degree)``.
3. BFS from each candidate, emitting nodes in visit order and edges with
   re-labeled endpoints. Neighbor expansion order is determined by a
   deterministic sort key derived only from local invariants.
4. Take the lex-min signature across all candidate starts — this is the
   canonical signature of the component.
5. Sort component signatures, serialize, and hash with blake2b.

Anchor (x, y, theta) is intentionally excluded: two layouts that differ
only by their placement in the boundary are fitness-equivalent for closed
cycles and should collapse under dup-elim.

Caveat: this BFS canonization is exact when start candidates are
distinguishable by the initial key. For graphs with non-trivial
automorphisms (e.g., perfectly symmetric ovals where all start candidates
share the same key), lex-min over multiple BFS attempts is the safety net.
For pathological cases where even that fails, two true isomorphs may hash
differently — this produces *false negatives* (missed duplicates) which
are harmless. False positives (different graphs hashing the same) cannot
occur because every node and edge contributes to the signature.
"""

from __future__ import annotations

import hashlib
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

from pymoo.core.duplicate import ElementwiseDuplicateElimination

from .catalog import TrackCatalog
from .decoder import DecoderConfig, decode_chromosome
from .encoding import PortPairDimensions
from .problem import MIN_USEFUL_COMPONENT_SIZE
from .types import PortEdge, PortGraph


NodeRecord = Tuple[str, int, int]
"""(piece_id, flip_bit, rotate_bit) at a canonically-labeled slot."""

EdgeRecord = Tuple[int, str, int, str]
"""(canonical_a, port_a, canonical_b, port_b) with canonical_a <= canonical_b.

When canonical_a == canonical_b (self-loop, never produced by the decoder),
ports are sorted lexically as well.
"""

ComponentSig = Tuple[Tuple[NodeRecord, ...], Tuple[EdgeRecord, ...]]
"""Canonical (nodes, edges) signature for one connected component."""


_HASH_DIGEST_SIZE: int = 8
_EMPTY_SENTINEL: bytes = b"empty-port-graph"


def _component_incidence(
    component: Set[int], edges: List[PortEdge],
) -> Dict[int, List[PortEdge]]:
    """Adjacency map restricted to edges within ``component``."""
    inc: Dict[int, List[PortEdge]] = {s: [] for s in component}
    for edge in edges:
        if edge.slot_a in component and edge.slot_b in component:
            inc[edge.slot_a].append(edge)
            inc[edge.slot_b].append(edge)
    return inc


def _bfs_canonical_from(
    start: int,
    component: Set[int],
    incidence: Dict[int, List[PortEdge]],
    graph: PortGraph,
) -> ComponentSig:
    """BFS from ``start``, returning the canonical (nodes, edges) signature.

    Each visited slot is assigned a fresh integer label in BFS order
    (start slot gets label 0). Neighbors are explored in a deterministic
    order keyed only on local piece/port labels — no original slot index
    leaks into the signature.
    """
    label: Dict[int, int] = {start: 0}
    nodes: List[Optional[NodeRecord]] = [None] * len(component)
    edges_out: List[EdgeRecord] = []
    queue: deque = deque([start])
    visited_edges: Set[int] = set()

    while queue:
        slot = queue.popleft()
        slot_label = label[slot]
        nodes[slot_label] = (
            graph.slot_pieces[slot],
            graph.slot_flips.get(slot, 0),
            graph.slot_rotates.get(slot, 0),
        )

        candidate_edges: List[Tuple[Tuple, PortEdge, int, int, str, str]] = []
        for edge in incidence[slot]:
            edge_id = id(edge)
            if edge_id in visited_edges:
                continue
            other_slot, other_port = edge.other_endpoint(slot)
            this_port = edge.port_a if edge.slot_a == slot else edge.port_b
            sort_key = (
                this_port,
                graph.slot_pieces.get(other_slot, ""),
                graph.slot_flips.get(other_slot, 0),
                graph.slot_rotates.get(other_slot, 0),
                other_port,
            )
            candidate_edges.append(
                (sort_key, edge, edge_id, other_slot, this_port, other_port)
            )

        candidate_edges.sort(key=lambda t: t[0])

        for _, _edge, edge_id, other_slot, this_port, other_port in candidate_edges:
            if edge_id in visited_edges:
                continue
            visited_edges.add(edge_id)

            if other_slot not in label:
                label[other_slot] = len(label)
                queue.append(other_slot)

            a_lab, a_port = slot_label, this_port
            b_lab, b_port = label[other_slot], other_port
            if (a_lab, a_port) > (b_lab, b_port):
                a_lab, a_port, b_lab, b_port = b_lab, b_port, a_lab, a_port
            edges_out.append((a_lab, a_port, b_lab, b_port))

    edges_out.sort()
    final_nodes: Tuple[NodeRecord, ...] = tuple(
        node if node is not None else ("", 0, 0) for node in nodes
    )
    return (final_nodes, tuple(edges_out))


def _component_signature(component: Set[int], graph: PortGraph) -> ComponentSig:
    """Lex-min BFS signature over all candidate starting slots."""
    incidence = _component_incidence(component, graph.edges)

    keyed: List[Tuple[Tuple, int]] = []
    for slot in component:
        key = (
            graph.slot_pieces[slot],
            graph.slot_flips.get(slot, 0),
            graph.slot_rotates.get(slot, 0),
            len(incidence[slot]),
        )
        keyed.append((key, slot))
    keyed.sort()
    min_key = keyed[0][0]
    starts = [s for k, s in keyed if k == min_key]

    best: Optional[ComponentSig] = None
    for start in starts:
        sig = _bfs_canonical_from(start, component, incidence, graph)
        if best is None or sig < best:
            best = sig
    assert best is not None
    return best


def canonical_graph_hash(
    graph: PortGraph,
    *,
    min_useful_component_size: int = MIN_USEFUL_COMPONENT_SIZE,
) -> bytes:
    """Return an 8-byte canonical hash of ``graph``'s useful topology.

    Invariant under: slot permutation, edge row order, anchor pose,
    component ordering, and junk components below
    ``min_useful_component_size``.

    Distinguishes: piece-kind multisets, edge connectivity, flip/rotate
    bits per slot, number of cycles, number of useful components.
    """
    useful = [
        c for c in graph.connected_components
        if len(c) >= min_useful_component_size
    ]
    if not useful:
        return hashlib.blake2b(_EMPTY_SENTINEL, digest_size=_HASH_DIGEST_SIZE).digest()

    sigs = sorted(_component_signature(c, graph) for c in useful)
    payload = repr(sigs).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=_HASH_DIGEST_SIZE).digest()


class PortGraphDuplicateElimination(ElementwiseDuplicateElimination):
    """pymoo duplicate elimination via canonical PortGraph hash.

    Decodes each chromosome to a :class:`PortGraph` (Phase 5a+: AFTER
    junction materialization so chromosomes whose active junctions
    materialize differently hash differently per Coupling D / Rule 15),
    computes :func:`canonical_graph_hash`, and compares 8-byte digests.

    The hash is cached on the :class:`pymoo.core.individual.Individual`
    via ``ind.set("graph_hash", h)`` so subsequent comparisons of the same
    individual are O(1).
    """

    def __init__(
        self,
        dims: PortPairDimensions,
        catalog: TrackCatalog,
        decoder_config: DecoderConfig,
        inventory: Optional[Dict[str, int]] = None,
        min_useful_component_size: int = MIN_USEFUL_COMPONENT_SIZE,
    ) -> None:
        super().__init__()
        self.dims = dims
        self.catalog = catalog
        self.decoder_config = decoder_config
        self.min_useful_component_size = min_useful_component_size
        # Phase 5a: lazy-import to avoid circular dependency with
        # ``junction_materializer`` (which imports ``decoder``, which doesn't
        # need ``canonical``). Materializer is constructed only when we
        # actually need to hash; ``inventory=None`` skips materialization
        # (Phase 4 behaviour, junction-invariant hash).
        self._materializer = None
        if inventory is not None and dims.J_max > 0:
            from .junction_materializer import JunctionMaterializer
            self._materializer = JunctionMaterializer(
                dims=dims,
                catalog=catalog,
                inventory=inventory,
                decoder_config=decoder_config,
            )

    def _hash_for(self, ind) -> bytes:
        cached = ind.get("graph_hash")
        if cached is not None:
            return cached
        if self._materializer is not None:
            x = ind.X.copy()
            self._materializer.materialize(x)
        else:
            x = ind.X
        graph = decode_chromosome(x, self.dims, self.catalog, self.decoder_config)
        h = canonical_graph_hash(
            graph, min_useful_component_size=self.min_useful_component_size,
        )
        ind.set("graph_hash", h)
        return h

    def is_equal(self, a, b) -> bool:
        return self._hash_for(a) == self._hash_for(b)
