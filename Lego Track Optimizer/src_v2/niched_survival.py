"""Topology-niched survival for the port-pair problem.

Wraps pymoo's ``ConstrRankAndCrowding`` with a bucketing pre-pass that
preserves topological diversity in the survivor set.

Problem this solves:
- ``ConstrRankAndCrowding`` ranks by Pareto front + crowding distance in
  *objective space*. Two solutions with very different topologies but similar
  (utilization, min_speed) values look identical to it.
- Combined with the simple-loop heuristic bias, this means the population
  converges to all-ovals: the Pareto front is dense with oval clones because
  oval-with-1-extra-piece variants dominate every alternative topology in
  objective space.

Fix:
1. Compute a coarse topology signature for each individual:
       (n_components, n_cycles, n_switches, n_crosses, n_crossovers).
   Cheap — readable straight from the chromosome and the decoded port graph
   (the canon_sig cache from ``CanonicalGraphDuplicates`` shares the decode
   work where possible).
2. Bucket the population by signature.
3. Within each bucket, run normal NSGA-II rank+crowding to pick survivors.
4. Distribute the survival quota *across* buckets first, so a bucket with one
   member gets a guaranteed slot before a 50-member oval bucket gets its 6th.

Allocation policy: round-robin across non-empty buckets, picking the
best-ranked unselected member of each bucket per round, until we hit
``n_survive``. This is the "lightweight" niching the user approved (option 1
in the analysis doc) — no NSGA-III reference-direction machinery.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from pymoo.core.population import Population
from pymoo.core.survival import Survival
from pymoo.operators.survival.rank_and_crowding import ConstrRankAndCrowding

from .canonical_hash import _read_topology  # decode-lite — slots + edges only


# =============================================================================
# Signature extraction
# =============================================================================


# Piece-id sets used to bucket "what kind of layout is this".
_SWITCH_IDS = frozenset({
    "R40_SWITCH_LEFT_IN", "R40_SWITCH_LEFT_OUT",
    "R40_SWITCH_RIGHT_IN", "R40_SWITCH_RIGHT_OUT",
})
_CROSS_IDS = frozenset({"CROSS_90"})
_CROSSOVER_IDS = frozenset({"DOUBLE_CROSSOVER"})


def topology_signature(x, dims, catalog) -> Tuple[int, int, int, int, int]:
    """Coarse topology signature: (n_components, n_cycles, n_switches,
    n_crosses, n_crossovers).

    Cheap to compute — uses the same _read_topology helper as
    ``canonical_graph_signature`` and standard graph counting from there.
    """
    slot_pieces, edges = _read_topology(x, dims, catalog)
    if not slot_pieces:
        return (0, 0, 0, 0, 0)

    n_switches = sum(1 for pid in slot_pieces.values() if pid in _SWITCH_IDS)
    n_crosses = sum(1 for pid in slot_pieces.values() if pid in _CROSS_IDS)
    n_crossovers = sum(
        1 for pid in slot_pieces.values() if pid in _CROSSOVER_IDS
    )

    # Connected components + cycle count via Euler:  n_cycles = E - V + C
    parent: Dict[int, int] = {s: s for s in slot_pieces}

    def find(v: int) -> int:
        root = v
        while parent[root] != root:
            root = parent[root]
        while parent[v] != root:
            parent[v], v = root, parent[v]
        return root

    edge_count = 0
    for sa, _, sb, _ in edges:
        if sa in parent and sb in parent:
            edge_count += 1
            ra, rb = find(sa), find(sb)
            if ra != rb:
                parent[ra] = rb

    components = {find(s) for s in slot_pieces}
    n_components = len(components)
    n_vertices = len(slot_pieces)
    # Cyclomatic number: E - V + C (clamped at 0; tree-shaped components have 0)
    n_cycles = max(0, edge_count - n_vertices + n_components)

    return (n_components, n_cycles, n_switches, n_crosses, n_crossovers)


# =============================================================================
# Niched survival
# =============================================================================


class TopologyNichedSurvival(Survival):
    """ConstrRankAndCrowding with topology-bucketing fairness pre-pass.

    Each individual gets a 5-int signature; survival quota is round-robin'd
    across distinct signatures so a single oval can never crowd out a
    crossing-rich layout that happens to dominate it in objective space.

    Fallback: when the entire population shares one signature (early
    generations, tiny populations) we degrade gracefully to plain
    ConstrRankAndCrowding.
    """

    def __init__(self, dims, catalog) -> None:
        super().__init__(filter_infeasible=False)
        self._dims = dims
        self._catalog = catalog
        self._inner = ConstrRankAndCrowding()

    def _do(
        self, problem, pop: Population, n_survive: Optional[int] = None,
        **kwargs,
    ) -> Population:
        if n_survive is None or n_survive >= len(pop):
            return self._inner.do(problem, pop, n_survive, **kwargs)

        # 1. Rank everybody once via the inner survival on the full pop with
        #    n_survive=len(pop) to obtain an objective-space ordering. This
        #    runs the standard NSGA-II machinery (constraint violation,
        #    Pareto rank, crowding distance) but does no truncation.
        ranked = self._inner.do(problem, pop, len(pop), **kwargs)

        # 2. Bucket the ranked individuals by topology signature; within each
        #    bucket they retain rank order.
        buckets: Dict[Tuple, List[int]] = defaultdict(list)
        for i, ind in enumerate(ranked):
            sig = ind.get("topo_sig")
            if sig is None:
                sig = topology_signature(ind.X, self._dims, self._catalog)
                ind.set("topo_sig", sig)
            buckets[sig].append(i)

        if len(buckets) <= 1:
            # No topology diversity to preserve — fall back to plain truncation.
            return ranked[:n_survive]

        # 3. Round-robin across buckets in priority order. Bucket priority for
        #    a given round = the bucket's best unselected member's rank
        #    position in `ranked` (lower is better). This guarantees the
        #    rank-1 oval gets picked first, then the rank-1 figure-8, etc.
        cursors: Dict[Tuple, int] = {sig: 0 for sig in buckets}
        chosen: List[int] = []
        chosen_set: set = set()

        while len(chosen) < n_survive:
            # Build round candidates: (next_global_rank, signature)
            round_picks = []
            for sig, members in buckets.items():
                cur = cursors[sig]
                if cur >= len(members):
                    continue
                next_global_rank = members[cur]
                round_picks.append((next_global_rank, sig))

            if not round_picks:
                break  # no candidates left in any bucket

            round_picks.sort()  # sort by global rank ascending
            for global_rank, sig in round_picks:
                if len(chosen) >= n_survive:
                    break
                idx = buckets[sig][cursors[sig]]
                cursors[sig] += 1
                if idx not in chosen_set:
                    chosen.append(idx)
                    chosen_set.add(idx)

        return ranked[chosen]
