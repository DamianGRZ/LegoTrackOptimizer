# Implementation Plan — LEGO Track Optimizer V2

> **Single source of truth for the thesis implementation.** Originally drafted in Claude Code plan mode (2026-05-06), iterated through 10 Parts, 35 Golden Rules, and three rounds of expert review (Python/pymoo/physics + research-backed Part 9 revisions + Part 10 testing/cohesiveness audit).

---

## Quick Index

**Current state**: Phase 0 measurements collected (mass_loco=0.493 kg, mass_trailing=0.327 kg, coupler_offset=0.106 m, v_motor_max=1.26 m/s, max_accel=0.68 m/s²). YAML write pending.

### Status

- [x] Phase 0 — Train measurements **collected** (data in §Part 8)
- [x] Phase 0 — Write `configs/trains/measured_consist.yaml` (~5 min)
- [x] §9.7 runtime knobs — set `crossover_prob: 0.0` / `mutation_prob: 1.0` via UI tweaks panel per Phase 1+ run (Rule 25 revised; no static YAML/Pydantic edit)
- [x] §9.7 termination edit — `runner.py:260` → `DefaultMultiObjectiveTermination` (Rule 34 revised)
- [x] **Pre-Phase-1 scaffolding (~8 days)** — `DiagnosticsCallback` + `SnapshotCallback` + 6 shared fixtures (§10.8)
- [x] **§9.7 sanity run (HARD BLOCKER)** — abort plan if feasibility <500/1000 at `cx=0.0`
- [x] Phase 1 — Cycle closure repair (Baldwinian per Rule 24 revised)
- [x] Phase 2 — Sequential-ring stadium emitter
- [x] Phase 3 — Auto-centering decoder + anchor shrink + ENCODING_VERSION=2
- [x] Phase 4 — Junction segment scaffolding (**Coupling A**: refactor `PortPairCrossover` into 2 ops)
- [x] Phase 5a — `PASSING_SIDING` template machinery (per-route F[1] refactor; **Coupling E** dual-eval)
- [x] Phase 5b — Asymmetric-oval-with-siding seed
- [x] Phase 5c — Junction mutations (meta-op consolidation per Rule 29 revised)
- [?] Phase 6a — `FIGURE_8_CROSS` template
- [x] Phase 6b — Figure-8 heuristic seed
- [x] Phase 7a — `PARALLEL_DC_BRIDGE` template
- [x] Phase 7b — Parallel-tracks-with-DC seed
- [x] Phase 8 — Topology-aware archive admission + AlphaUCB ALNS migration eval (Rule 29b)

### How to find a section

| Want | Where | Approx. line |
|---|---|---|
| Phase 0 measurements + YAML spec | Part 8 → "Phase 0 — Physical Measurement Protocol" | ~1620 |
| Phase 1 implementation spec | Part 5 → "Phase 1 — Constructive Closure Repair" | ~590 |
| Phase 2-8 specs | Part 5 → "Phase N — ..." | varies |
| Golden Rules 1-23 (initial set) | Part 7 | ~1410 |
| Golden Rules 24-35 (research-backed final) | Part 9 §9.1-9.5 | ~2100 |
| Five hidden cross-phase couplings | Part 10 §10.3 | ~2480 |
| Per-phase test catalog (~80 tests) | Part 10 §10.2 | ~2330 |
| Snapshot mechanism (10 PNGs/run) | Part 10 §10.5 | ~2530 |
| Shared test fixtures (6 reusables) | Part 10 §10.6 | ~2580 |
| Pymoo compatibility audit (Part 6 → Part 9 risks 1-11) | Part 6 + §9 | ~1170 |
| Stale-reference list (constraint count is 11+T) | §9.6 | ~2230 |

### How Claude Code loads slices efficiently

Don't load the whole file. Use line-range reads:

```
Read docs/PLAN.md offset=590 limit=200      # Phase 1 spec
Read docs/PLAN.md offset=2100 limit=400     # Part 9 (revised Rules)
Read docs/PLAN.md offset=2330 limit=200     # Part 10 §10.2 test catalog
Grep "\*\*Rule [0-9]+" docs/PLAN.md         # all Rule citations
Grep "^### Phase " docs/PLAN.md             # all phase headers + line numbers
```

Or `@docs/PLAN.md` with a specific Part/section name in the prompt; Claude can scope the read.

### Working with the plan

When asked "what's next" or "implement next phase":

1. Read the **Status** checkbox list above; first unchecked item is next.
2. Read that phase's section via line range (find via `Grep "^### Phase N"` or §10.2 catalog).
3. Cross-reference applicable Golden Rules — `Grep "^\*\*Rule "` for the master list, then read by line.
4. Cross-reference Coupling impact — Part 10 §10.3 lists 5 couplings labeled A-E.
5. Implement; mark this checkbox done.
6. Run the phase's DoD (Part 10 §10.2 test list) before declaring done.

(The user does not run git commits in this workflow — phases land via direct edits to disk, not via commit gates.)

### Hard blockers in execution order

These must be cleared in sequence — do not skip ahead:

1. **Phase 0 YAML write** — without `configs/trains/measured_consist.yaml`, all later runs use V2's wrong defaults.
2. **§9.7 runtime knobs** — without `cx_prob=0.0`, the schema-disruption fix is not applied.
3. **§9.7 sanity run** — if feasibility doesn't recover to ≥500/1000 at `cx=0.0`, **Part 9's whole cascade is wrong** and must be re-planned before Phase 1.
4. **Phase 4 crossover refactor (Coupling A)** — without splitting `PortPairCrossover` into two ops, the post-Phase-5a junction-cx ablation is unimplementable.
5. **Phase 5a per-route F[1]** — without this, all Phase 5+ `min_speed` results are systematically inflated by ~62% on switched layouts.

### Open decisions before execution

- **Phase 5a Baldwinian implementation pattern**: phenotype-passthrough via `out["pheno"]` (preferred), or repair clones X internally? Decide at Phase 1 start.
- **Configs file inventory tuning**: keep current `with_switches` 168-piece inventory or adjust? Recommend keep (matches V1 snapshots).
- **Whether to run Phase 9 (ERX)**: deferred unless Phase 1-8 fail to deliver feasibility at `cx=0.0`. Decide post-Phase-8.

---

## Document History

- **2026-05-06**: Initial plan written in Claude Code plan mode. Parts 1-8 drafted across multiple iterations.
- **2026-05-06**: Expert reviews applied (python-pymoo-reviewer, fact-checker, physics-expert).
- **2026-05-06**: Part 9 added — research-backed revisions to Rules 24-35 (citation-grounded).
- **2026-05-06**: Part 10 added — testing pyramid, cohesiveness audit, snapshot mechanism, shared fixtures.
- **2026-05-06**: Migrated from `~/.claude/plans/` to `docs/PLAN.md` (in git, IDE-readable).

---

# Spanning-Tree Edge-Set EA â€” Applicability to LEGO Track Optimizer

## Context

The user surfaced **Raidl & Julstrom 2002, "Edge-Sets: An Effective Evolutionary Coding of Spanning Trees"** (TR-186-1-01-01, TU Wien) and asked whether the approach could be used in this project. The paper proposes a direct edge-set representation for spanning trees in EAs, with random-spanning-tree operators (PrimRST, KruskalRST, RandWalkRST) for initialization, recombination, and mutation, demonstrated on the degree-constrained MST problem.

This document is a research/applicability analysis â€” **not an implementation plan**. The user wants a thorough verdict on whether the paper's ideas fit the closed-cycle LEGO track optimization problem, and where (if anywhere) they could be adapted.

## TL;DR Verdict

**Direct usage: âŒ No.** A spanning tree is acyclic by definition (Raidl Â§I, p.2: *"a spanning tree on G is a maximal, acyclic subgraph"*). The LEGO problem **requires** cycles â€” `G[7+T] = 1 - n_cycles â‰¤ 0` ([problem.py:25](src_v2/problem.py:25)) â€” and rewards multi-cycle topology (figure-8s, sidings). A pure spanning-tree representation would never satisfy the closure constraint.

**Indirect usage: ðŸŸ¡ Partially, with adaptation.** Three of the paper's deeper ideas â€” **connected random initialization, edge-union crossover, cycle-aware add/remove mutation** â€” map to known weak spots in the current operators. Adopting them would require lifting from spanning trees to **spanning tree + chord set** (cycle-basis encoding), which the paper does not address but is a standard graph-theoretic generalization.

**Best fit:** improving `PortPairSampling._populate_random`, `PortPairCrossover`, and the `add_edge`/`remove_edge` mutations â€” *not* replacing the encoding.

---

## What the Raidl Paper Actually Proposes

### Core idea (Â§III, p.10)
Represent a spanning tree T directly as the **set of its edges** (hash table or array of (u,v) pairs).
- **Space**: O(n).
- **Locality + Heritability**: highest of all surveyed encodings (Table I, p.6).
- Outperforms PrÃ¼fer numbers, Blob Code, network random keys, link-and-node biasing for the d-MST problem (Table III, pp. 32-33).

### Three random-spanning-tree algorithms (Â§III-A)
| Algorithm | Time | Bias | Constraints |
|---|---|---|---|
| **PrimRST** | O(n) | toward stars (low-diameter) | very flexible |
| **KruskalRST** | O(n) (with union-find) | mild toward stars | local constraints only |
| **RandWalkRST** | O(n log n) avg | uniform (Broder 1989) | local constraints |

### Operators (Â§III-C, Â§III-D)
1. **Initialization**: pick a RST algorithm; generate the population.
2. **Recombination** (Fig 5, p.20): union the parents' edge-sets, run a RST algorithm on the union; offspring contains only parental edges (or a few non-parental ones if constraints force it).
3. **Mutation** (Fig 6, p.21): two variants â€”
   - (a) **Insert-then-delete-cycle-edge**: add a non-tree edge â†’ DFS finds the unique cycle â†’ remove a random cycle edge.
   - (b) **Delete-then-reconnect**: drop a tree edge â†’ identify the two split components â†’ reconnect with a new edge.

### Constraints + heuristics (Â§V)
- **Degree constraint** for d-MST: reject edges that would push a node's degree past d during all three operators (Fig 9, p.27).
- **Heuristic biasing**: tournament-pick edges weighted by cost for low-cost-favoring construction (Â§V-B, p.29).
- **Empirical**: HES-EA found optimum on most "structured-hard" instances; competitors stalled at 0.5%-15% above optimum (Table III).

---

## Why Direct Application Fails for LEGO Track Optimizer

| Paper assumption | LEGO reality |
|---|---|
| Solutions are **trees** (acyclic) | Solutions must be **closed cycles** (G[7+T]) and ideally multi-cycle |
| Edges are simple-graph (u, v) | Edges are **port-pair** (slot_a, port_a, slot_b, port_b); each port used at most once |
| Vertex degree = variable (1 to n-1 in tree) | Vertex degree = **fixed by piece type** (2 for straight/curve, 3 for switch, 4 for crossing) |
| Underlying graph G is fixed/complete | Underlying graph is **inventory-derived multigraph** with geometric per-edge feasibility (FK closure) |
| Cost is sum of edge costs | Cost is **multi-objective** (utilization + min_speed) with closure/boundary/collision constraints |
| Constraints are abstract (degree, diameter, capacity) | Constraints include **SE(2) pose closure** with tolerance, boundary box, intersection coverage |

A spanning tree on the slot graph would be a layout with `m = n - 1` edges â†’ cyclomatic 0 â†’ no closed loops â†’ infeasible. The paper's machinery cannot be lifted whole.

---

## Where the Paper's IDEAS Adapt â€” Three Concrete Opportunities

The decoder already operates in the cycle space: [decoder.py:663](src_v2/decoder.py:663) uses `nx.cycle_basis` to enumerate fundamental cycles per connected component (a spanning tree T + chord set C â‡’ each chord defines a fundamental cycle). The encoding is implicitly cycle-basis already, but **the operators don't reason about that structure**.

### Opportunity 1 â€” Connected random initialization (PrimRST-style)

**Current weakness** ([operators.py:703-756](src_v2/operators.py:703)):
`_populate_random` draws random pieces and `int(active_count * 1.2)` random port-pairs. There is **no connectedness guarantee** â€” most random chromosomes have multiple disconnected components, hitting `G[9+T] = n_components - 1 > 0` ([problem.py:28](src_v2/problem.py:28)). With Îµ-decay, infeasibles dominate early generations.

**Adaptation** (PrimRST analogue, called e.g. `_populate_random_prim`):
1. Place a random starting piece in slot 0.
2. While target slot count not reached: pick the next slot's piece from inventory, connect it via a compatible port-pair to *some already-placed slot*. (PrimRST eligibility check = port not already used + piece-spec port compatibility.)
3. After the spanning structure is built (n - 1 edges, connected, acyclic), **add 1-2 chord edges** between existing slots that still have free ports. Each chord creates exactly one fundamental cycle (cyclomatic +1).

This guarantees `G[9+T] â‰¤ 0` and `G[7+T] â‰¤ 0` by construction. Cost: O(n) per chromosome, no slower than the current random init.

**Adoption cost**: medium. Existing `branch_grow.py:73 find_branch_path` already does compatible-port piece insertion; it could be wrapped.

---

### Opportunity 2 â€” Edge-union crossover (heritability-preserving)

**Current weakness** ([operators.py:773-825](src_v2/operators.py:773)):
`PortPairCrossover` does **one-point per region** (slot, flip, rotate, pair-row, anchor). Project memory note `project_crossover_destroys_feasibility.md` records: at crossover_prob=0.9, only **1/500** individuals stayed feasible; mutation-only at crossover_prob=0.0 reached **500/500** feasible. This means crossover is currently **off** in practice â€” half the GA's recombination engine is disabled.

**Adaptation** (Raidl Â§III-C union-then-re-derive):
1. Union parental port-pair sets `E_union = E1 âˆª E2`.
2. Resolve port conflicts (port already taken by another edge) by picking randomly from the contributing parent.
3. Run a PrimRST-like greedy extractor on `E_union`: walk edges in random order, accept if both endpoints' ports are still free and the slot piece-id is consistent.
4. If the extracted graph is short of the target edge count, fall back to one parent's remaining edges.

**Heritability property**: offspring contains *only parental edges* (and at most a few inventory-feasible repair edges). This is exactly what the d-MST experiments showed cuts evaluations by ~85% (Table II columns Î´ for âˆ—-variants, p.24).

**Why this would help here specifically**: the current one-point cut splits a closed cycle in half, then offspring inherits half-cycles from each parent, almost always destroying both. Edge-union avoids this â€” it never splits a cycle, only re-derives.

**Adoption cost**: medium-high. Has to respect port-uniqueness, inventory caps, and slot piece-id consistency. Could be prototyped as an opt-in operator alongside the existing crossover, gated by a config flag.

---

### Opportunity 3 â€” Cycle-aware insert/remove mutation

**Current weakness** (`OP_WEIGHTS` table at [operators.py:849-867](src_v2/operators.py:849); implementations at `_add_edge` line 1126 and `_remove_edge` line 1147):
`add_edge` (weight 0.10) and `remove_edge` (weight 0.06) act independently. `add_edge` may create a redundant chord (cyclomatic too high â†’ constraint pressure on closure); `remove_edge` may cut a bridge (split component â†’ G[9+T] violation).

**Adaptation** (Raidl Â§III-D Fig 6):
- **Variant (a) â€” insert-then-trim**: After `add_edge`, run BFS in the slot graph to find the cycle the new chord just created. If cyclomatic > target_chord_count, remove a random edge from that cycle (pre-existing chord, never a tree edge). Net effect: one cycle replaced with a different one â€” locality preserved.
- **Variant (b) â€” remove-then-reconnect**: After `remove_edge`, check if it was a bridge (the decoder's union-find at [decoder.py:294](src_v2/decoder.py:294) already gives this for free). If yes, find a port-feasible reconnecting edge (PrimRST-style, restricted to the now-split components).

**Adoption cost**: low-medium. The decoder already exposes `connected_components` and `nx.cycle_basis`. The structural mutation `closure_repair_lamarckian` ([operators.py:866](src_v2/operators.py:866)) already has the spirit; this would be a port-graph-level analogue, sharper than the geometry-level closure repair.

---

## What the Paper Doesn't Solve (and we still need)

1. **Multi-cycle reward** â€” the paper's framework optimizes a single tree. LEGO's `MIN_USEFUL_COMPONENT_SIZE` filter and branch-cycle bonus require multi-cycle reasoning that the paper's 1985-2002-era spanning-tree machinery doesn't address.
2. **Geometric feasibility** â€” Raidl's edges have fixed cost; LEGO edges' "cost" is the FK pose deviation given upstream piece chain, which is path-dependent.
3. **Inventory dynamics** â€” Raidl's G is fixed; this project's G is inventory-derived and changes per chromosome.

## Better-Fitting Literature

If the user wants to dig further, more directly applicable bodies of work:

- **Cycle-basis genetic algorithms** â€” Horton (1987) min-cycle basis; recent EA work on cycle-space encodings for ring/loop topologies in network design.
- **Hamiltonian-cycle EAs** â€” closer to single-loop layouts; classical TSP encoding survey (Larranaga et al. 1999) covers edge-recombination crossover (ERX) which is essentially the paper's edge-union idea, applied to cycles.
- **Edge-recombination crossover (ERX)** for closed walks (Whitley, Starkweather, Shaner 1989) â€” directly applicable to LEGO's port-pair edge structure since LEGO layouts are closed walks (every port paired = Eulerian-like).
- **Multi-graph chord-set EAs** for FACT (fault-tolerant network design) â€” explicitly handle (spanning tree + redundant edges) encodings.

**Recommendation**: ERX (Whitley 1989) is probably a closer fit than the Raidl spanning-tree paper for this project's crossover problem, because LEGO closed layouts ARE closed walks.

---

## Honest Risk Assessment

Adopting any of the three opportunities involves real cost:
- **PrimRST-init** is the safest experiment â€” it just generates better starts; fallback is the existing random init. Worst case: minor slowdown, marginal feasibility gain.
- **Edge-union crossover** is higher-risk but addresses a known dead branch (crossover currently off). Could be A/B-tested against mutation-only by toggling the operator.
- **Cycle-aware mutation** is low-risk locality improvement, but its effect may be subtle since structural mutations already cover similar territory.

**None of these are guaranteed wins.** The paper's d-MST results are on a different problem class. They earn their keep only on graph-shaped MOPs where heritability matters; the LEGO problem also has tight inventory + geometric constraints that may dominate the operator's locality gains.

**The user should not implement any of this without a controlled experiment** comparing baseline vs. modified operator on the same configs (default, with_switches, with_crossing) over identical seed budgets.

---

## Critical Files (read-only, for traceability)

- [src_v2/operators.py:594-756](src_v2/operators.py:594) â€” `PortPairSampling`, the random-init target
- [src_v2/operators.py:773-825](src_v2/operators.py:773) â€” `PortPairCrossover`, current one-point-per-region
- [src_v2/operators.py:833-1230](src_v2/operators.py:833) â€” `PortPairMutation` and the 17 sub-ops
- [src_v2/decoder.py:294-360](src_v2/decoder.py:294) â€” `_connected_components`, union-find already here
- [src_v2/decoder.py:692-693](src_v2/decoder.py:692) â€” `_iter_cycles`, `nx.cycle_basis` already used at decode time
- [src_v2/problem.py:25-30](src_v2/problem.py:25) â€” closure/cycle/component constraints
- [src_v2/branch_grow.py:73](src_v2/branch_grow.py:73) â€” `find_branch_path` (A*-style piece insertion that PrimRST-init could reuse)
- Memory: [project_crossover_destroys_feasibility.md](C:/Users/dgpro/.claude/projects/S--Programming-Repo-DamianGRZ-LegoTrackOptimizer/memory/project_crossover_destroys_feasibility.md) â€” documents the crossover-disabled state

---

## Verdict for the User

| Aspect | Answer |
|---|---|
| Use Raidl's representation **as-is**? | No â€” spanning trees are acyclic, LEGO needs cycles. |
| Use his three RST algorithms (Prim/Kruskal/RandWalk) for **initialization**? | Yes, after extending tree â†’ tree+chords. Concrete benefit: connected initial pop. |
| Use **edge-union crossover** to fix the disabled crossover problem? | **Most promising direction.** Aligns with the project memory note that current crossover destroys feasibility. |
| Use **insert-then-cycle-trim mutation**? | Modest gain; existing structural mutations partially cover this. |
| Is there better-fitting literature? | **Edge-recombination crossover (Whitley 1989)** for closed walks is a closer fit than this paper. |

**Decision point**: if the user wants to pursue any of these, the lowest-risk first step is a **prototype edge-union crossover** as an opt-in alternative, A/B-tested against the current crossover on `with_switches.yaml` over 5+ seeds. If it improves feasibility under crossover_prob=0.9, the approach has earned its keep; otherwise the project's mutation-only equilibrium is the right answer.

No code changes are proposed in this plan. The user decides whether to act on any opportunity.

---

---

# Part 2 â€” Web Research: Special Operators That Would Suit This Project

## Context

After the Raidl spanning-tree analysis, the user asked for broader **web research** on what kind of mutation, crossover, and repair operators would suit this project. This section is a literature scan of operator families that match the project's specific shape:

- **Encoding**: port-pair edges (direct edge-set already, not permutation, not random-keys)
- **Topology**: closed multigraph (cycles required, multi-cycle preferred)
- **Constraints**: port-uniqueness, inventory, FK closure, boundary, intersections, connectedness
- **Objectives**: bi-objective NSGA-II (utilization, min_speed)
- **Known failure mode**: current one-point-per-region crossover destroys feasibility (memory: 1/500 feasible at p=0.9)
- **Existing**: 17 mutation sub-ops with ALNS adaptive selection, Lamarckian repair pipeline

I scanned ~15 papers/surveys via web search across edge-recombination crossover, k-opt local search, ALNS, memetic NSGA-II, BRKGA, MST GAs, and adaptive operator selection. Below is what's worth pulling into this project, ranked.

---

## Crossover Operators

### Tier 1 â€” Strongest fit: Edge Recombination Crossover (ERX)

**Source**: Whitley, Starkweather, Fuquay 1989 ([Wikipedia overview](https://en.wikipedia.org/wiki/Edge_recombination_operator)).

**Why it fits**: ERX operates on **edge-set semantics with closed-walk parents** â€” exactly what `PortPairProblem` already encodes. It treats each parent as an undirected edge graph, builds adjacency lists by **set-union of both parents' edges**, then constructs an offspring by walking the merged adjacency graph greedily.

**Algorithm (Wikipedia + IEEE refs)**:
1. For each piece slot v, build N(v) = neighbors of v in P1 âˆª neighbors of v in P2 (set union â€” common edges appear once, distinct edges accumulate).
2. Pick a random starting slot.
3. Repeatedly: append current slot to offspring, remove it from all adjacency lists, then move to the neighbor with the **fewest remaining connections** (ties broken randomly). This biases toward edges that are more "constrained" (fewer alternatives) â€” avoiding dead-ends.
4. If stuck (no neighbors), pick a random unvisited slot.

**Known performance**: in Larranaga et al. 1999 ([UPM survey](https://cig.fi.upm.es/wp-content/uploads/2024/01/Genetic-algorithms-for-the-travelling-salesman-problem-A-review-of-representations-and-operators.pdf)), ERX outperformed PMX, OX, CX on TSP â€” ERX is consistently best for closed-walk problems.

**Adaptation needed for LEGO**:
- "Slots" with degree > 2 (switches=3, crossings=4): adjacency lists carry up to 4 neighbors per slot. The "fewest-remaining-connections" heuristic still applies.
- Port assignment is a separate layer: after picking the next slot via ERX, also need to pick which port-pair was the inherited edge. Use the parent's port assignment when copying the edge.
- Inventory checks: if offspring exceeds inventory of a piece type, fall back to alternate.

**Cost**: high implementation cost (a real new operator class), but addresses the documented crossover-destroys-feasibility failure mode head-on. **Most promising single change in the codebase.**

### Tier 2 â€” Subtour Exchange Crossover (SEX / Greedy Subtour)

**Source**: Sengoku & Yoshihara 1993; modern variants in [IEEE 2022 â€” Complete Subtour Order Crossover](https://ieeexplore.ieee.org/document/9895081/).

**Why it fits**: extracts **continuous closed subtours** (a connected sub-cycle) from each parent and splices them into a single offspring. Closed sub-cycles are a natural unit for LEGO layouts â€” a complete loop, a passing siding, a figure-8 lobe.

**Algorithm**:
1. Identify the cycle basis of each parent via the existing `nx.cycle_basis` machinery in [decoder.py:692](src_v2/decoder.py:692).
2. Pick one cycle from P1 and one from P2 that share at least one slot (or anchor slot).
3. Splice: keep P1's cycle, replace shared portion with P2's alternate route.

**Best fit for**: switch-pair / branch insertion â€” one parent contributes the main loop, the other contributes the siding.

**Cost**: medium. Uses existing cycle-basis machinery. Could be a sub-operator alongside ERX.

### Tier 3 â€” Subtree / segment-based crossover (Raidl Â§III-C)

Already covered in Part 1. Edge-union + RST extraction. Slightly less specific to closed walks than ERX but still solid.

### What to AVOID

- **PMX, OX, CX** ([Wikipedia: Edge recombination](https://en.wikipedia.org/wiki/Edge_recombination_operator)) â€” these are **position-based for permutations**, not edge sets. They do not preserve the closed-walk structure well. Current one-point-per-region crossover is morally similar (positional) and is likely the root cause of the documented feasibility collapse.
- **Uniform crossover** on port-pair rows â€” breaks cycle topology by mixing rows from incompatible chromosomes.

---

## Mutation Operators

### Tier 1 â€” k-opt local-search-as-mutation

**Source**: Lin-Kernighan 1973 ([Wikipedia: Linâ€“Kernighan](https://en.wikipedia.org/wiki/Lin%E2%80%93Kernighan_heuristic)); Helsgaun's [General k-opt submoves](http://akira.ruc.dk/~keld/research/LKH/KoptReport.pdf).

#### 2-opt move (**simplest, highest ROI**)
- Pick 2 edges (e1, e2) in the same cycle.
- Remove both â†’ cycle splits into two paths.
- Reconnect with the **other two edges** that re-form a single cycle (effectively reversing one of the two paths).
- Net change: 2 edges removed, 2 edges added. Cyclomatic preserved. Strong locality.

**Why it fits**: 2-opt is the most ubiquitous TSP/closed-walk local search. It directly addresses *"reorder pieces along a cycle without disturbing topology"*. The LEGO problem rewards different orderings (better closure, smaller boundary) without changing graph structure.

**Adaptation**: pieces have port asymmetry (port A vs port B vs C/D); reversing a segment may require also flipping each piece's `flip` bit. The existing `toggle_flip` mutation has the machinery.

#### 3-opt and Or-opt
- 3-opt: remove 3 edges, reconnect (8 ways). Bigger neighborhood, more expensive.
- Or-opt: move a segment of length 1â€“3 to a different cycle position. Often outperforms 2-opt empirically.

**Or-opt is probably more valuable than 3-opt** because it handles "moving a stretch of pieces to a different part of the loop" â€” a natural LEGO move.

#### Lin-Kernighan as a one-shot mega-mutation
- Sequential k-opt with closed alternating walks.
- "Apply LK from current chromosome until no improvement." Memetic-style.
- Expensive (O(nÂ²) per chromosome) but high-quality.

**Recommendation**: implement 2-opt first as a sub-operator with low weight (~0.05). Or-opt next. Skip full LK initially.

### Tier 2 â€” Permutation-style mutations adapted to cycles

From [Wikipedia: Mutation (genetic algorithm)](https://en.wikipedia.org/wiki/Mutation_(genetic_algorithm)) and [tutorialspoint summary](https://www.tutorialspoint.com/genetic_algorithms/genetic_algorithms_mutation.htm):

- **Inversion mutation** â€” reverse a segment along the cycle. **Equivalent to 2-opt in TSP**; for the port-pair encoding, it requires segment-reversal with port-flip handling.
- **Insertion mutation** â€” move one piece to another cycle position. Equivalent to a 1-piece Or-opt.
- **Displacement mutation** â€” move a multi-piece segment. Equivalent to Or-opt.
- **Scramble mutation** â€” randomize a subset's order. High disruption â€” useful as occasional escape from local optima but not for fine search.
- **Swap mutation** â€” exchange two pieces' positions. Changes 4 edges. Lower locality than Or-opt.

### Tier 3 â€” Graph-structure mutations (already partly present)

The project's `add_edge`, `remove_edge`, `rewire_edge`, `introduce_switch_pair`, `grow_branch`, etc. cover this territory. Two refinements from the Raidl paper / TSP literature:

- **Add-edge-with-cycle-trim** (Raidl Â§III-D Fig 6a): inserting a chord creates a cycle; if cyclomatic > target, remove a redundant edge from the new cycle.
- **Remove-edge-with-reconnect** (Raidl Â§III-D Fig 6b): if removing an edge cuts a bridge (component split), reconnect via a port-feasible alternate.

### What to AVOID

- **High-disruption scramble** as a primary operator â€” destroys cycle structure too aggressively.
- **Bit-flip on slots** â€” already implicitly covered by `piece_type` mutation.
- **Pure random reset of edges** â€” covered by `add_edge`/`remove_edge` already.

---

## Repair Operators

### Tier 1 â€” Lamarckian vs Baldwinian repair (the meta-question)

**Source**: Liepins-Vose 1991 onward; modern review at [Hybrid Genetic Algorithms](https://link.springer.com/article/10.1007/s12065-020-00425-5).

**Lamarckian** (current pipeline): repair writes back to the chromosome. Fast convergence, may cause premature convergence.

**Baldwinian**: repair only computes fitness, chromosome unchanged. Slower convergence, better diversity.

**Partial Lamarckian (20â€“40% writeback)**: empirically the best of both ([researchgate study](https://www.researchgate.net/publication/2335408_Utilizing_Lamarckian_Evolution_and_the_Baldwin_Effect_in_Hybrid_Genetic_Algorithms)). Cited result: 20â€“40% partial-Lamarckian gave best solution-quality + computational-efficiency mix.

**Recommendation for LEGO**: try **partial Lamarckian**. Currently `PortPairRepairPipeline` always writes back. A flag like `lamarckian_prob = 0.3` would let the GA explore infeasible regions while still occasionally locking in good repairs.

**Cost**: trivial config change + one branch in repair.py â€” but needs careful experiment design.

### Tier 2 â€” ALNS-style destroy/repair (memetic)

**Source**: Ropke-Pisinger 2006; modern Python lib [N-Wouda/ALNS](https://github.com/N-Wouda/ALNS).

**Concept**: each generation, pick a few elite individuals and apply destroy-then-repair to them as a memetic step.

**Destroy operators** for LEGO:
- **Random-piece removal**: deactivate k random slots + their pairs.
- **Worst-component removal**: drop the smallest disconnected component.
- **Closure-error removal**: drop the cycle with worst FK residual.

**Repair operators** for LEGO:
- **Greedy reconnect**: PrimRST-style (already discussed in Part 1).
- **A\* branch growth**: reuse `branch_grow.py:73 find_branch_path`.
- **Inventory-aware insertion**: prefer pieces with surplus inventory.

The project already has **ALNS for sub-operator weights** at `runner.py:271` â€” extending ALNS to **destroy/repair pairs** would be the next step.

### Tier 3 â€” Adaptive greedy repair

**Source**: 2025 [AIMS Mathematics paper on minimum vertex cover](https://www.aimspress.com/article/doi/10.3934/math.2025600).

**Concept**: greedy repair to feasibility, then prune unnecessary additions, with **exploration/exploitation balance adjusted by population convergence**. When the population is converged, repair is more aggressive (exploits); when diverse, repair is gentler (explores).

**Adaptation**: repair pipeline could check the population's CV variance â€” if everyone is near-feasible, apply aggressive Lamarckian repair; if widespread infeasibility, apply Baldwinian (no writeback) to preserve exploration.

**Cost**: medium â€” needs to sample population state at each generation.

### Tier 4 â€” Path relinking as repair

**Source**: [Memetic NSGA-II surveys](https://link.springer.com/article/10.1007/s12532-022-00231-3).

**Concept**: when an offspring is infeasible, find the nearest feasible solution in the current population (in chromosome space) and "relink" the offspring toward it by single-edge moves, stopping at first feasible point.

**Why interesting**: leverages the existing feasible Pareto front members as "guides". Each generation has a small set of feasible solutions; instead of random repair, we walk infeasible offspring toward them.

**Cost**: high. Requires distance metric on port-pair chromosomes.

---

## Adaptive Operator Selection (the meta-layer)

### Already in place: ALNS at sub-operator level

The project's `ALNSCallback` at [runner.py:271](src_v2/runner.py:271) reweights the 17 mutation sub-operators each generation by offspring CV reduction. This is **state-of-the-art** for adaptive operator selection â€” confirmed by the [Wouda ALNS framework](https://alns.readthedocs.io/en/latest/setup/introduction_to_alns.html) and recent reviews.

### Possible extensions

1. **Multi-armed bandit (MAB) reframing**: cast operator selection as MAB with UCB1 or Thompson sampling. Empirically equivalent to roulette-wheel ALNS for 10-20 operators but more theoretically grounded.

2. **Dual Actor-Critic RL** ([2026 Tristan paper](https://tristan2025.org/proceedings/TRISTAN2025_ExtendedAbstract_143.pdf), [arxiv 2601.11414](https://arxiv.org/html/2601.11414v1)): learn an operator-selection policy via deep RL. Overkill for this project but a research-direction note.

3. **Subpopulation-based operator differentiation**: split the population into elite/general/weak groups and apply different operator distributions per group ([MDPI Energies 2015 multi-pop paper](https://www.mdpi.com/1996-1073/8/12/12433)).

---

## Memetic / Hybrid Local Search Layer

**Source**: NSGA-II + local search for multi-criteria MST ([CEC 2017](https://ieeexplore.ieee.org/abstract/document/7969432/)).

The mc-MST literature (closest formal analogue to our problem â€” bi-objective spanning structure on a graph) found that:
- NSGA-II alone reaches **premature convergence** and gets stuck at local optima.
- **Pareto Local Search**, **Tabu Search**, and **Path Relinking** integrated into NSGA-II all improve performance.
- Best result: HMOST hybrid NSGA-II + Multi-VNS scaled to >200 nodes and outperformed prior methods.

**Adaptation for LEGO**:
- Apply **2-opt local search** to the top 5â€“10% of each generation. (Cheap, embarrassingly parallel.)
- Optionally integrate **Pareto Local Search** as a sub-step after main NSGA-II survival â€” but be wary of generation-time blow-up.

**Cost**: medium. The current operators are all O(n) per chromosome; 2-opt local search is also O(nÂ²) per individual â€” manageable for top-k only, not the full population.

---

## Cross-Cutting Recommendations

### Priority Order (highest expected ROI first)

| # | Change | Expected Impact | Cost |
|---|---|---|---|
| 1 | **ERX-style edge-recombination crossover** | Restores crossover usefulness (currently disabled at p=0.0) | High |
| 2 | **2-opt mutation sub-operator** | Strong cycle-aware locality move | Medium |
| 3 | **Or-opt mutation sub-operator** | Segment relocation, complements 2-opt | Medium |
| 4 | **Partial-Lamarckian repair** (writeback prob 0.3) | Diversity preservation, prevents premature convergence | Trivial |
| 5 | **PrimRST-style connected init** (Part 1, Opp.1) | Better starting feasibility | Medium |
| 6 | **Add-edge-with-cycle-trim mutation** (Raidl) | Cycle-preserving structural mutation | Low-Medium |
| 7 | **2-opt local search on top-5% per gen** | Memetic refinement, mc-MST literature supports | Medium |
| 8 | **Subtour-exchange crossover** | Alternative to ERX, complementary | High |

### What NOT to do

1. **Don't switch encoding** â€” port-pair edge-set is already correct. BRKGA (random keys) is a different paradigm and `project_cgp_encoding_migration.md` records it was already replaced.
2. **Don't add scramble mutation as a primary operator** â€” too disruptive for cycle structure.
3. **Don't add PMX/OX/CX-style crossover** â€” they are position-based, not edge-based, and would replicate the current crossover's failure mode.
4. **Don't replace ALNS** â€” it's already a state-of-the-art adaptive selection mechanism.

### Risk Assessment

The biggest risk is **adding operators without retiring or reweighting the existing 17**. The mutation pool is already fairly large; adding 4 new sub-operators (2-opt, Or-opt, edge-trim, ERX-driven crossover) means ALNS has to learn weights for 21+ operators, which slows convergence in the early generations.

**Mitigation**: introduce one operator at a time, each gated by a config flag. A/B test each addition against the baseline on `with_switches.yaml` over 5+ seeds. Keep only operators that statistically improve feasibility OR Pareto front quality.

---

## What the Operator Literature Tells Us About Our Specific Failure Modes

The project memory lists several recurring problems:

1. **`project_crossover_destroys_feasibility.md`** â€” crossover at p=0.9 dropped feasibility to 1/500. **Diagnosis from literature**: position-based crossover on edge-encoded chromosomes is known to have this failure (Whitley 1989, Larranaga 1999). **Fix**: ERX or subtour-exchange.

2. **`project_visualization_limitations.md`** â€” multi-component layouts only partially shown. **Tangentially relevant**: implies multi-component layouts are still being generated (so the connectedness constraint isn't dominating). **Fix**: connected init (Part 1 Opp.1) reduces multi-component starts.

3. **`feedback_dynamic_chromosome_size.md`** â€” N_VAR scales to inventory. **Implication**: variable chromosome length is a project invariant. ERX adapts naturally; permutation-based operators do not.

4. **Speed objective is single-piece-bottleneck** â€” the slowest piece bottlenecks the cycle's min_speed. **Operator implication**: 2-opt mutation that swaps the slowest piece's position can dramatically improve F[1] without affecting topology. **High-leverage operator.**

---

## Final Synthesis: A Practical Operator Roadmap

**Phase A** (one-shot, low-risk, no new operators):
- Switch to **partial-Lamarckian repair** (writeback prob 0.3). One config flag.
- Run 5 seeds Ã— 4 configs. Measure feasibility, Pareto hypervolume, runtime.

**Phase B** (one new mutation):
- Add **2-opt mutation sub-operator** with weight 0.05. Wire into ALNS.
- Run 5 seeds Ã— 4 configs. Measure F[1] (min_speed) improvement specifically.

**Phase C** (one new mutation, one structural):
- Add **Or-opt mutation** with weight 0.05.
- Add **add-edge-with-cycle-trim** (Raidl) replacing the bare `add_edge` op.
- Run 5 seeds Ã— 4 configs.

**Phase D** (the big one â€” crossover overhaul):
- Implement **ERX-style edge-recombination crossover** as a new operator class.
- A/B test against current `PortPairCrossover` at crossover_prob âˆˆ {0.3, 0.6, 0.9}.
- If feasibility holds at p=0.9 with ERX, this single change is the most consequential.

**Phase E** (memetic, optional):
- Apply **2-opt local search** to top-5% per generation.
- Cost-benefit measured against runtime increase.

---

## Sources (web research)

- [Edge recombination operator â€” Wikipedia](https://en.wikipedia.org/wiki/Edge_recombination_operator)
- [Larranaga et al. 1999 â€” Genetic algorithms for the TSP: A review of representations and operators (UPM)](https://cig.fi.upm.es/wp-content/uploads/2024/01/Genetic-algorithms-for-the-travelling-salesman-problem-A-review-of-representations-and-operators.pdf)
- [Modified edge recombination operators (IEEE)](https://ieeexplore.ieee.org/document/972444/)
- [Genetic Edge Recombination (Whitley, Starkweather, Fuquay 1989) â€” ResearchGate](https://www.researchgate.net/publication/2527549_The_Traveling_Salesman_and_Sequence_Scheduling_Quality_Solutions_Using_Genetic_Edge_Recombination)
- [Linâ€“Kernighan heuristic â€” Wikipedia](https://en.wikipedia.org/wiki/Lin%E2%80%93Kernighan_heuristic)
- [Helsgaun â€” General k-opt submoves for the Linâ€“Kernighan TSP heuristic](http://akira.ruc.dk/~keld/research/LKH/KoptReport.pdf)
- [Mutation (genetic algorithm) â€” Wikipedia](https://en.wikipedia.org/wiki/Mutation_(genetic_algorithm))
- [Adaptive repair method for constraint handling in MOGA â€” ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1568494620300831)
- [A New Repair Operator for MOEA â€” arxiv 1504.00154](https://arxiv.org/pdf/1504.00154)
- [Constraint Handling Techniques review â€” Springer 2022](https://link.springer.com/article/10.1007/s11831-022-09859-9)
- [GeneRepair â€” A Repair Operator for GAs (ResearchGate)](https://www.researchgate.net/publication/229149949_GeneRepair_-_A_Repair_Operator_for_Genetic_Algorithms)
- [Adaptive Greedy Repair for Min Vertex Cover â€” AIMS Mathematics 2025](https://www.aimspress.com/article/doi/10.3934/math.2025600)
- [Lamarckian vs Baldwinian on Multi-objective Knapsack â€” ResearchGate](https://www.researchgate.net/publication/221228551_Comparison_Between_Lamarckian_and_Baldwinian_Repair_on_Multiobjective_01_Knapsack_Problems)
- [Self-adaptive learning for hybrid GAs â€” Springer Evolutionary Intelligence](https://link.springer.com/article/10.1007/s12065-020-00425-5)
- [Memetic algorithm â€” Wikipedia](https://en.wikipedia.org/wiki/Memetic_algorithm)
- [Local search strategies for NSGA-II + mc-MST â€” IEEE CEC 2017](https://ieeexplore.ieee.org/abstract/document/7969432/)
- [Memetic procedure for global multi-objective optimization â€” Springer 2022](https://link.springer.com/article/10.1007/s12532-022-00231-3)
- [Hybrid NSGA-II + Multi-VNS for MOST](https://ojs.srce.hr/crorr/article/view/33510/18711)
- [Adaptive Large Neighborhood Search â€” N-Wouda/ALNS](https://github.com/N-Wouda/ALNS)
- [ALNS Introduction â€” Read the Docs](https://alns.readthedocs.io/en/latest/setup/introduction_to_alns.html)
- [DR-ALNS: Deep Reinforced Adaptive LNS (ICAPS 2024)](https://github.com/RobbertReijnen/DR-ALNS)
- [New Adaptive Mechanism for LNS using Dual Actor-Critic â€” arxiv 2601.11414](https://arxiv.org/html/2601.11414v1)
- [Biased Random-Key Genetic Algorithms: A Review â€” arxiv 2312.00961](https://arxiv.org/html/2312.00961v2)
- [Random-Key GA tutorial â€” Resende](http://mauricio.resende.info/talks/2012-09-CLAIO2012-brkga-tutorial-both-days.pdf)
- [Variable-length chromosome GA crossover â€” ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0957417416301634)
- [Subtour Order Crossover for TSP â€” IEEE 2022](https://ieeexplore.ieee.org/document/9895081/)
- [Eulerian Circuit / Hierholzer â€” NetworkX docs](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.euler.eulerian_circuit.html)
- [HP-MOCD: Topology-aware NSGA-II for community detection â€” arxiv 2506.01752](https://www.arxiv.org/pdf/2506.01752)
- [pymoo NSGA-II custom operators](https://pymoo.org/algorithms/moo/nsga2.html)
- [pymoo subset selection custom operators tutorial](https://github.com/anyoptimization/pymoo/blob/main/docs/source/customization/subset.md)

---

---

# Part 3 â€” Why V1 Produced More Interesting Shapes Than V2

> âš ï¸ **NAVIGATION NOTE**: The main body of Part 3 below describes V1 as a CGP-tuple `(type, port2_conn, port3_conn)` encoding from git commit `e4bf9ef`. **This is the WRONG V1 era** â€” that commit is in this repo's git history but is not the V1 the user actually meant. The real V1 lives in a sibling folder `S:\Programming\Repo\DamianGRZ\LegoTrackOptimizer - V1 kind of worked\` and uses a multi-segment partitioned encoding `[main_loop:100 | switch_mask:100 | branches:16 | start:2]` with parametric template branches. See **Correction #2** further down in Part 3 (line ~711) and **Part 4** (deep V1 analysis based on the correct source) for authoritative architecture. The main-body discussion below is preserved as a record of the analytical evolution but should NOT be cited as architecture truth.

## Context

V1 (the Python integer-encoded GA â€” at git commit `e4bf9ef`, before the `84b0a2d` NSGA-II migration) "kind of worked": it produced visually richer closed loops and stayed connected, even though it had no real switch support. V2 (current `src_v2/`) produces small ovals and stops there. The user wants to understand the structural difference.

I read V1 source from git history: `src/encoding.py`, `src/sampling.py`, `src/operators.py`, `src/problem.py`, `src/decoder.py`, `src/templates.py`, plus the JS prototype's `Lego Track Optimizer/search.js` and `analysis.md`. Comparison below.

> âš ï¸ The above paragraph references the wrong V1 era â€” see the navigation note at the top of Part 3.

## The One Difference That Explains Everything

**V1 used an implicit sequential backbone. V2 uses an explicit edge-set graph.**

That is the entire story. Every other gap follows from this.

### V1 chromosome layout (`encoding.py`, e4bf9ef)

```
[type_0, p2conn_0, p3conn_0,  type_1, p2conn_1, p3conn_1,  ...]
```

- Each "node" = (piece_type, port2_conn, port3_conn) tuple
- **Ports 0 and 1 are implicit**: node `i.port1` always connects to node `i+1.port0`
- Ports 2 and 3 are **only used for branches** (switches/crossings)
- `INACTIVE = -1` for unused

Closure constraint reduced to: **angular sum of FK deltas â‰¡ 360Â° (mod 360Â°)** â€” a 1-dimensional, almost-trivial check.

### V2 chromosome layout (`encoding.py`, current)

```
[piece_slots: N_max int16 | port_pairs: E_max Ã— 4 int16 | anchor: 3 int16]
```

- Slots hold piece indices, but **slot order means nothing** â€” no implicit chain
- Every connection between two pieces is a separate gene: `(slot_a, port_a, slot_b, port_b)`
- Closure = full SE(2) pose match across all connected components

Closure constraint: **3-dimensional residual per cycle, summed across all cycles, gated by per-axis tolerance**. Plus connectedness constraint, plus loose-port constraint, plus n_cycles â‰¥ 1, plus per-port-uniqueness.

## The Cascade of Consequences

| What | V1 | V2 |
|---|---|---|
| Search space size | **O(n_pieces Ã— types)** â€” vary piece types in a sequence | **O(n_pieces Ã— types Ã— edges Ã— portsÂ²)** â€” vary pieces AND connections |
| Closure check | 1D angular sum to 360Â° | 3D Ã— per-component SE(2) residual |
| Connectedness | **guaranteed by construction** (sequence is sequence) | constraint that must be enforced |
| Random init feasibility | Often closes â€” angles balance easily | **Almost never closes** â€” no edges or wrong edges |
| Repair burden | Tiny â€” sequence is always valid | Huge â€” port-uniqueness, dangling edges, fragmented components, half-cycles |
| Mutation effect on topology | **None** â€” mutations change shape, not connectivity | **Massive** â€” `add_edge`/`remove_edge` break or split cycles |
| Crossover effect on topology | Mild â€” splicing two valid sequences usually gives a valid sequence | **Catastrophic** â€” splicing edge sets fragments the graph (memory: 1/500 feasible at p=0.9) |

## Why V1's Shapes Looked Better

The user wrote: *"shapes were much more interesting than in this one, and were still connecting, there were no switches but it looked kind of good."*

This is exactly what the V1 design predicts. With the implicit sequential backbone:

1. **The closed loop is the default.** A random V1 chromosome is *probably* a closed loop â€” you just need ~16 R40 curves' worth of angle. `IntegerSampling._closure_aware_chromosome` (`sampling.py`, e4bf9ef) explicitly tracked angle and placed curves until 360Â° â€” that's all the geometric constraint solving you needed.

2. **Mutations vary shape, not structure.** Every V1 mutation in `TrackMutation` (`operators.py`) â€” `piece_type`, `connection`, `insert`, `delete`, `swap`, `shift` â€” operates *within* the sequential chain. A single piece-type mutation can swap a curve for a straight, completely changing the shape of the loop while keeping it closed. The GA can explore "lots of different ovals/stadiums/dogbones" trivially.

3. **Templates carried the hard topology.** Branches and switches were handled by `templates.py` (`PassingSidingTemplate`), which embedded the closure math (`compute_branch_pieces`, `compute_branch_endpoint`). The GA didn't have to *discover* a passing siding's geometry â€” the template provided guaranteed-closing branch sequences, and the GA only varied the parameter `n_straights`. **This is also why V1 "kind of worked but didn't really do switches"** â€” switches weren't evolved, they were inserted from a template lookup.

4. **`HEURISTIC_RATIO = 0.08`.** V1 needed only 8% heuristic seeds because random init was already close-to-feasible. V2 has `heuristic_ratio = 0.30` and *still* collapses to small ovals â€” a higher fraction can't compensate for the harder constraint landscape.

The JS prototype (`Lego Track Optimizer/search.js`) is the most extreme version of this idea: it doesn't even use a GA. It enumerates **Circle, Oval, Stadium templates** with parameterized straight counts, validates each against geometry/boundary/collision, scores them. Per its README:

> *"Encoding that domain knowledge as templates is more useful than a generic search that thrashes."*

That's the same insight â€” restrict the topology, vary the parameters. V1 Python lifted this idea slightly: implicit-chain GA = "the chromosome IS a parameterized closed walk."

## Why V2 Collapses to Small Ovals

V2 is doing exactly what its objective function asks of it. The user's earlier diagnosis was correct: the algorithm doesn't know switches are wanted â€” the fitness function never tells it. **But there's an even deeper problem**: V2 spends almost all its evaluation budget *just satisfying the topology constraints*. By the time it has a closed, connected, port-clean graph, it has used up its convergence budget.

Concretely:
- V2's `_populate_random` (`operators.py:703`) generates `int(active_count * 1.2)` random port-pairs. Most random pair sets fragment the graph or leave loose ports â†’ infeasible.
- V2's `PortPairCrossover` (one-point per region) splits closed cycles â†’ near-100% infeasible offspring.
- V2's repair pipeline mostly **deactivates** broken edges/pieces â€” it can't synthesize a new closed cycle from scratch.

So the GA selects the smallest viable closed structure (16 R40 â†’ small oval) and stays there. Every mutation that would add a piece risks breaking the graph; every crossover destroys cycles. The "useful component" filter (`MIN_USEFUL_COMPONENT_SIZE = 4`) further pushes toward "one big simple cycle" because small companion loops get filtered out.

The `analysis.md` document already in the repo (`Lego Track Optimizer/analysis.md`) reaches the same conclusion via different routes:
1. Heuristic seeds dominated by simple loops (94% of patterns)
2. No structural mutations toward complexity
3. Lenient duplicate elimination (`np.array_equal` on int16)
4. Crowding distance only in 2D objective space â€” can't distinguish topologies
5. Loose-port constraint normalized too softly

## What V1 Got Right (And Should Be Carried Forward)

1. **Implicit backbone for the main loop.** Every chromosome is a closed walk by construction; the GA never wastes evaluation budget proving connectivity.
2. **Angular pre-check.** `_closure_aware_chromosome` explicitly placed curves until angular budget closed â€” a 1D check that guarantees one of the closure axes before geometry is run.
3. **Templates as first-class structural elements.** `templates.py` provided guaranteed-closing branch geometry. The GA picked *whether* to use a template, not *how* to assemble one.
4. **Lower heuristic ratio because random init was nearly feasible.** Less seeding bias, more genuine search.

## What V1 Got Wrong (And Why V2 Was Designed)

1. **Single main loop only.** Figure-8s, parallel tracks with crossovers, multi-component yards â€” none of these fit cleanly in a sequential chain.
2. **Branches were second-class** via port2_conn/port3_conn â€” workable but limited.
3. **No native multigraph** â€” the encoding can't express the rail topologies the user actually wants for richer layouts.

V2's port-pair edge-set encoding is the right answer **for the problem V2 was scoped to solve** â€” arbitrary multigraphs. But it loses V1's free closure-by-construction for the easy case.

## The Actual Diagnosis

V2 isn't worse than V1 because of NSGA-II vs single-objective, or because of operator weights, or because crossover is broken (though that's a real issue from Part 2). V2 is worse than V1 **for the simple-loop case** because V2 made the chromosome express structure that V1 got for free. The interesting shapes V1 produced were not produced by clever evolution â€” they were produced because *every single chromosome was already a candidate closed loop*, and the GA only had to vary which pieces.

V2's encoding is more expressive but the expressiveness comes at the cost of constraint-satisfaction work that V1 didn't have to do. With current operators, V2 spends its budget on satisfying topology and never reaches shape diversity.

## What This Implies For Going Forward

Three honest options, ordered from cheapest to most ambitious:

### Option A â€” Hybrid encoding (high payoff, high risk)

Bring V1's implicit backbone back as a **first-class part** of the V2 chromosome:

```
[main_loop_sequence: int16 Ã— N_main]   â† V1-style, implicit sequential chain
[branch_attachments: int16 Ã— N_branches Ã— (anchor_idx, switch_kind, branch_seq)]
[chord_edges: int16 Ã— N_chords Ã— 4]    â† only for figure-8 / crossings / multi-component
[anchor: 3 int16]
```

Decoder:
1. Decode main loop sequentially (V1-style), guaranteed closed by construction.
2. Insert branches at their anchor points using V1-style template-based geometry.
3. Apply chord edges to create cycles between non-adjacent main-loop slots.

**This is V1 + V2 = best of both.** Closed loops are easy; branches via templates are easy; multi-cycle layouts are still expressible via chord edges.

Cost: **major encoder/decoder/operators rewrite**. The chromosome layout in `encoding.py` and `decoder.py` would need to change. All operators would need backbone-aware variants.

### Option B â€” Add a sequential-chain heuristic emitter (low effort, partial fix)

Add an emitter to `PortPairSampling._build_patterns` that generates port-pair edges directly from a V1-style sequence: pick a piece sequence (e.g. 16 R40 + 8 STR) and emit the implied port-pair edges (slot 0 port B â†’ slot 1 port A, slot 1 port B â†’ slot 2 port A, â€¦, slot N-1 port B â†’ slot 0 port A).

This gets V1's main strength â€” *structurally guaranteed closed loops in the initial population* â€” without changing the encoding. The current `_emit_oval` etc. emitters partially do this; making the sequential-chain emitter the dominant seed pattern (60-70% of heuristic share) and adding many parameter variations (curve type mix, straight type mix, asymmetric stadiums) would replicate V1's shape diversity.

Cost: **small** â€” one new emitter function plus rebalancing `heuristic_ratio` and emitter weights. Could be done in a day.

### Option C â€” Restore V1 outright as a parallel mode (escape hatch)

Keep V2's port-pair encoding for general topology, but reintroduce V1's integer-sequential encoding as a **separate mode** the user picks via config. For thesis evaluation, run both: V1 mode for "interesting shape diversity on simple closed loops", V2 mode for "novel multigraph topologies".

Cost: **medium** â€” V1 source still in git history, would need to be reanimated against current `pymoo` version and the new TrackCatalog. But all the design decisions are documented.

## Critical Files (read-only references)

V1 (git commit `e4bf9ef`):
- `src/encoding.py` â€” implicit-chain layout, `GENES_PER_NODE = 3`, ports 0/1 implicit
- `src/sampling.py` â€” `IntegerSampling`, `_closure_aware_chromosome`, `HEURISTIC_RATIO = 0.08`
- `src/operators.py` â€” `TrackMutation`, six local mutations on piece types and connections
- `src/templates.py` â€” `PassingSidingTemplate`, `compute_branch_pieces`, FK closure math
- `src/problem.py` â€” single-objective `n_obj=1`, 5 normalized constraints

V2 (current):
- `src_v2/encoding.py` â€” explicit edge-set, `compute_port_pair_dimensions`
- `src_v2/operators.py:594` â€” `PortPairSampling._populate_random` (no closure-awareness)
- `src_v2/operators.py:773` â€” `PortPairCrossover` (one-point per region)
- `src_v2/decoder.py` â€” full graph decoder with `nx.cycle_basis`, union-find components
- `src_v2/repair.py` â€” heavy port-uniqueness/inventory enforcement

JS prototype (most extreme template approach):
- `Lego Track Optimizer/search.js` â€” Circle/Oval/Stadium templates, no GA
- `Lego Track Optimizer/README.md` â€” explicit "templates over generic search" rationale
- `Lego Track Optimizer/analysis.md` â€” V2 self-diagnosis (also confirms collapse to ovals)

## Verdict

**V1 wasn't smarter â€” V1 had an easier problem.** Its implicit sequential backbone made closure trivial and connectedness automatic, leaving 100% of the search budget for shape variation. V2 promoted topology to a first-class chromosome dimension, which is correct for the long-term scope (multigraph layouts with switches/crossings), but the operators don't yet pay for that promotion. The GA exhausts its budget on topology constraints and never reaches shape diversity.

**The cheapest fix that captures V1's spirit** is Option B: add a sequential-chain heuristic emitter so a meaningful fraction of the initial population is V1-style closed-by-construction, then let mutations explore from there. Combined with the operator improvements from Part 2 (ERX crossover, 2-opt mutation), this would likely close most of the V1-V2 gap without rewriting the encoding.

**The deepest fix** is Option A: hybrid encoding. That's a thesis-level commitment, not a one-week job.

No code changes are proposed in this plan. The user decides which option (if any) to act on.

---

## Correction â€” what the V1 snapshots actually show

I went on about switches earlier. The V1 snapshots at `outputs/snapshots/` (no `outputs_v1/` exists; the user's path was wrong but the data is here) **do not contain switches.** The visible V1-vs-V2 difference is simpler and more concrete:

| | V1 snapshots (`outputs/`) | V2 `with_switches` |
|---|---|---|
| Inventory | 24 pieces | 168 pieces |
| Used | **24 / 24 (100%)** | 68 / 168 (40%) |
| Composition | 16 R40 + 8 STR16 | 16 R40 only |
| Shape | **stadium** (curves + straights) | **small oval** (curves only) |
| Closure | exact, 0.00 stud / 0.00Â° | exact |
| Switches | 0 | 0 |
| `best_layout.png` title | `Best Feasible (24 pcs, 100.0% util, 0.89 m/s)` | n/a (different config) |

**The visible difference is that V1 used the straights to grow the loop into a stadium; V2 doesn't grow past the 16-curve minimum.** Same encoding-level cause as above: a single V1 mutation (`insert_node` on an inactive slot with STR16) inserts a straight into the implicit chain without breaking topology, because the chain is implicit. V2's equivalent insertion is a 5-gene surgery (activate slot, splice an existing edge into two new edges through the new STR16 ports, delete the bypassed edge, preserve port-uniqueness) and no single V2 operator does all five atomically.

So V1's "interesting shape" was: it grew the loop to fill the inventory by interleaving straights with curves. V2 doesn't grow past the smallest closed shape because every growth step risks breaking the graph.

The earlier claims about switches/templates as the V1 advantage are not supported by these snapshots â€” V1 didn't actually use switches in this run either. The encoding-level explanation (implicit chain vs explicit edges) still holds; the visible payoff is "stadium with straights", not "layout with switches".

## What this means for the recommendations

The Option B recommendation (sequential-chain heuristic emitter) is even more directly supported by the snapshots than I claimed. Specifically:

- The emitter should generate `(16 R40 + N STR16)` stadium variants for various N up to inventory cap, with the straights distributed symmetrically (N/2 on top, N/2 on bottom, or N/4 on each side, etc.).
- This is exactly what V1's `IntegerSampling._closure_aware_chromosome` did naturally: place curves until 360Â° angle, then sprinkle straights.
- A V2 emitter could do the same but emit port-pair edges directly: build the slot list `[R40, R40, R40, R40, STR16, STR16, R40, R40, R40, R40, STR16, STR16, ...]`, then emit edges `(slot_0_B â†’ slot_1_A), (slot_1_B â†’ slot_2_A), â€¦, (slot_N-1_B â†’ slot_0_A)` â€” a sequential ring. This produces the V1 stadium directly, port-pair encoded.

If even 30-40% of the initial heuristic seeds are sequential-ring stadiums of varying length and straight distribution, V2 should be able to hold onto them under selection, because they sit at high utilization with the same min_speed as the small oval (R40 curves still bottleneck), and so dominate it on the Pareto front.

The reason V2 doesn't already produce stadiums is **not** that the GA can't evaluate them favorably â€” it's that the GA can't reach them from the current operator set. Seeding them directly bypasses the operator-reach problem.

---

## Correction #2 â€” I had the V1 architecture wrong

The user pointed out V1 lives in a **sibling repo folder**: `S:\Programming\Repo\DamianGRZ\LegoTrackOptimizer - V1 kind of worked\` (not subfolder, separate repo). That repo's `CLAUDE.md` documents V1's actual encoding, which I had wrong.

### V1 actual encoding (multi-segment partitioned, fixed `N_VAR = 218`)

| Segment | Range | Length | Purpose |
|---|---|---|---|
| Main Loop | [0, 100) | 100 | Piece indices in sequence (-1 inactive, 0-9 piece) |
| Switch Mask | [100, 200) | 100 | Switch type per position (-1 no switch) |
| Branch Slots | [200, 216) | 16 | **4 slots Ã— 4 genes**: `(IN_pos, handedness, n_straights, active)` |
| Start | [216, 218) | 2 | `start_x, start_y` |

Constraints: `G[0..3]` closure/angle/boundary/inventory + `G[4]` = orphan-switch count (switches with unconnected port 2).

Branches were **parametric template instances**, not free-form pieces. Each branch slot just picked a template (`LEFT_SIDING` / `RIGHT_SIDING` from `templates.py`), an attachment point on the main loop, a length parameter, and on/off. The decoder materialized the actual branch pieces using template FK math, which **guaranteed branch closure by construction**. The GA never had to discover branch geometry â€” it picked from a fixed template grammar.

This is **not** the CGP-tuple encoding at git commit `e4bf9ef` that I read earlier. That commit was a different (later? earlier? unrelated?) era. I picked the wrong commit because git log only showed `e4bf9ef` as the integer-encoded GA before NSGA-II migration, but V1 was a different repo entirely with its own history.

### V1 `with_switches` snapshots (all 10, Gen 20 â†’ Gen 200)

Identical layout across all snapshots:
- **96 pieces, util 58.5 %, speed 0.89 m/s, switches 0, crossings 0**
- Big square stadium, ~450 Ã— 450 stud envelope, area â‰ˆ 160 000 studsÂ²
- 16 R40_LEFT at corners + ~80 STR16 along sides
- Closure 0.00 stud, 0.00Â°
- Inventory ~164 pieces; V1 used 96, V2 uses 68 (40 %)

### Why V1 grows but V2 doesn't (corrected, same structural cause)

V1's main loop is a **100-slot sequential array**. Slot `i` implicitly connects to slot `i+1`. Adding a piece = flipping one `-1` gene to a piece index. No edge genes to maintain. Closure-aware sampler biases initial population toward angle-summing sequences. Mutations on the main loop change *shape* without breaking *topology* â€” topology is the array order itself.

V1's branches are **parametric template instances** with pre-baked closure FK. Mutating a branch slot adjusts (where, kind, length) but never produces invalid branch geometry â€” the template materializer handles it.

V2's port-pair edges have no sequential array and no template machinery. Every connection is an explicit gene. Adding a piece is the 5-step edge surgery: activate slot, splice an existing edge into two new edges through the new piece's ports, delete the bypassed edge, preserve port-uniqueness. No single V2 mutation does this atomically. Branches in V2 must be discovered as raw multi-edge graph topology rather than picked from a template.

V1 traded expressiveness (no figure-8s, no multi-component yards, no novel branch geometries) for trivially feasible mutations and parametric branch templates. V2 traded trivially feasible mutations for expressiveness. The visible result is V1 grows the loop and V2 doesn't.

### What this changes about the recommendations

- **The sequential-ring heuristic emitter (Option B from earlier) is still the right cheapest fix**, but now I can be more specific about the target shapes: the V1 stadium at 96 pieces in a ~450 Ã— 450 envelope is what to seed. Emit (16 R40 + N STR16) sequential rings for N spanning the inventory cap, port-pair edges generated directly from the slot order.
- **Templates as parametric chromosome regions (Option A, hybrid encoding) is what V1 actually did**, not what I sketched. The hybrid wasn't "implicit chain + chord edges" â€” it was "main-loop sequential array + switch mask + parametric-template branches + start". If V2 is to recover V1's strengths, the cleanest path is to add a parametric-branch segment to the V2 chromosome that decodes to template-materialized passing sidings, while keeping V2's port-pair edges for arbitrary topology the templates don't cover. That gets V1's branch-discovery-by-grammar back without losing V2's expressiveness.
- **V1's choice to make N_VAR fixed at 218 contradicts the project invariant** "Chromosome length scales with inventory dynamically" â€” and was probably one of the reasons V1 was abandoned. So a V2 hybrid would need to keep dynamic sizing while adopting V1's segment structure. Doable; the segment lengths just need to derive from inventory at construction time.

### Concrete files to read in V1 if implementing the hybrid

In `S:\Programming\Repo\DamianGRZ\LegoTrackOptimizer - V1 kind of worked\src\`:
- `encoding.py` â€” `PartitionedDimensions`, segment offsets
- `templates.py` â€” `LEFT_SIDING`, `RIGHT_SIDING`, `compute_branch_pieces`, `compute_branch_endpoint`
- `sampling.py` â€” `IntegerSampling` and the closure-aware random builder
- `operators.py` â€” `PartitionedCrossover`, `PartitionedMutation` (the operators that knew about the segment structure)
- `decoder.py` â€” how the multi-segment chromosome was decoded into a `MultiPathLayout` (main loop + branches + traversal-path enumeration)
- `problem.py` â€” `TrackOptimizationProblem` with the 5 normalized constraints

I should have looked at this folder first instead of trusting `CLAUDE.md` of the V2 repo that said "V1 removed".

---

## Correction #3 â€” V1 infeasibles tell the real story

The user pointed at the infeasible snapshots (`outputs_v1/with_switches/snapshots/snapshot_*_infeasible.png`). They are dramatically more expressive than the feasibles and reveal what V1 was actually capable of generating.

### Per-snapshot infeasible catalog (V1 with_switches)

| # | Gen | Pcs | Util | Switches | Pairs | Closure | Angle | CV |
|---|---|---|---|---|---|---|---|---|
| 1 | 20 | 132 | 80.5% | 0 | 0 | 81.14 stud | 0.00Â° | 27.49 |
| 2 | 40 | 136 | 82.9% | 0 | 0 | 63.72 stud | 0.00Â° | 15.72 |
| 3 | 60 | 146 | 89.0% | 2 | 1 | 400.7 (branch) | 45.00Â° | 13.56 |
| 4 | 80 | 146 | 89.0% | 2 | 1 | 400.7 (branch) | 45.00Â° | 13.56 |
| 5 | 100 | 145 | 88.4% | 2 | 1 | 400.84 (branch) | 22.50Â° | 6.09 |
| 6 | 120 | 145 | 88.4% | 2 | 1 | 400.84 (branch) | 22.50Â° | 6.09 |
| 7 | 140 | 145 | 88.4% | 2 | 1 | 400.84 (branch) | 22.50Â° | 6.09 |
| 8 | 160 | 144 | 87.8% | 2 | 1 | 417.99 (branch) | 45.00Â° | 0.50 |
| 9 | 180 | 144 | 87.8% | 2 | 1 | 419.12 (branch) | 45.00Â° | 0.50 |
| 10 | 200 | 144 | 87.8% | 2 | 1 | 419.12 (branch) | 45.00Â° | 0.50 |

The feasibles in the same run plateau at 96 pcs, 58.5 %, 0 switches.

### Reachable layouts if the infeasibles closed

- **Snapshots 1-2** (132-136 pcs, no switches): big multi-loop topology with internal figure-8 lobe / parallel-loop attempt. Closure errors 60-80 stud â€” 3-5 piece swaps from a single big complex-shaped feasible.
- **Snapshots 3-7** (145-146 pcs, 2 switches, 1 branch pair): closure error on the branch is 400 stud and 22-45Â° â€” branch routing is fundamentally wrong, would need substantial rebuilding.
- **Snapshots 8-10** (144 pcs, 2 switches, CV = 0.50): **main loop already closes** (closure 4.48 stud, 0Â°). Only the branch is off â€” by 45Â° angle. That is one R40_LEFT â†’ R40_RIGHT swap in the branch's return curve away from feasibility.

The reachable feasible layouts these infeasibles encode:

| Source | If "rightly placed" | Util | Switches |
|---|---|---|---|
| Snapshots 1-2 | ~130-pc complex stadium, no switches | ~80% | 0 |
| Snapshots 3-7 | 138-pc main loop with abandoned branch | 84% | 0 (branch dropped) |
| **Snapshots 8-10** | **~144-pc layout with 1 working passing siding (2 switches)** | **87.8%** | **2** |

The 144-piece switched layout is what the user actually wants, and V1 was 1-2 piece swaps away from it for 50 generations (snapshots 8-10 are gens 160, 180, 200 â€” same chromosome stuck at the gate of feasibility).

### Why V1 failed to close them

NSGA-II constraint dominance plus the repair pipeline's limits:

1. **Feasibility-first selection killed the switched infeasibles.** As soon as the 96-piece simple-stadium feasible appeared, it Pareto-dominated every infeasible regardless of how close to feasibility they were. The 144-pc CV=0.5 infeasibles got no protection from the survival operator.
2. **No repair move for "branch angle 45Â° off, swap one curve handedness"**. V1's repair pipeline could deactivate broken pieces or adjust counts but couldn't reach into a parametric branch and flip its return-curve handedness. The branch was frozen at the wrong geometry.
3. **n_straights parameter wasn't being mutated effectively.** V1's branch slot has `(IN_pos, handedness, n_straights, active)` â€” the GA could mutate n_straights, but each value change shifts the branch endpoint by ~16 stud, and a 400-stud branch closure error would need Â±25 straights' worth of correction, which exceeds the parameter range.
4. **The Îµ-archive (if present) didn't preserve "topologically rich, near-feasible" solutions.** V1 used Deb's CV alone â€” no archive of expressive infeasibles to revisit later.

### What V2 should learn from V1's infeasibles

V2's problem isn't just "doesn't grow loops" â€” it's also "doesn't generate switched topologies at all". V1's operators DID generate them; selection just abandoned them. So V2 needs **both**:

1. **Operators that can reach 140+ piece layouts with switches** â€” sequential-ring stadium emitter (Part 2 Option B) covers the no-switch case; for switches V2 needs a parametric template emitter or structural mutation that inserts a passing-siding template into an existing loop, akin to V1's `templates.py`.
2. **Selection / archive that preserves near-feasible expressive solutions** â€” when CV is low (say < 1.0) and topology is rich (â‰¥ 1 switch pair, or â‰¥ 2 cycles), the solution should be archived even if a simpler feasible dominates it. The Îµ-archive at [runner.py:268](src_v2/runner.py:268) is the right hook; it just needs a topology-aware admission rule.
3. **Repair moves that can bridge small closure gaps in template branches** â€” a "tune-template" repair operator that takes a passing-siding instance with closure error and tries the local parameter neighborhood (n_straights Â± 1, swap handedness, swap approach/return curves). For a 45Â° angle error on a passing siding, swapping return-curve handedness is a single-mutation fix.

### Concrete numbers for the recommendations

If V2 reached and closed the V1 snapshot 8-10 layout, the result would be:

- **144 pieces (87.8%) used in `with_switches`** vs current 68 (40%)
- **2 switches actively used** vs current 0
- **1 passing siding present** vs current 0
- **Same 0.89 m/s min_speed** (R40 still bottlenecks)

That is a 2.1Ã— utilization improvement and the first time switches actually appear in V2 output. Achievable without changing objectives, encoding semantics, or NSGA-II â€” just by adding the missing operators and an archive admission rule.

### What an ideal V2 run would look like

1. Heuristic seeds include sequential-ring stadiums **and** template-passing-siding-on-loop variants â†’ 30-40% of initial pop has switches by construction.
2. Îµ-archive admission rule: keep any solution with â‰¥1 switch pair and CV < 1.0 even if dominated.
3. Repair pipeline includes "tune-template" move that closes branches with small angle/position residuals via parameter swap.
4. By gen ~50, the archive holds both the simple stadium (high feasibility) and the switched layout (high topology richness). Crowding distance + archive give selection something to choose between.
5. Final report shows both as Pareto solutions; the user picks the layout they want.

The V1 infeasibles prove this is reachable â€” they were already at 144 pcs with 2 switches and CV=0.5, just stranded by a selection scheme that abandoned them. Fix the selection / repair so the GA can finish what V1 started.

---

# Part 4 â€” Deep V1 architecture analysis (what actually made it work)

I read the full V1 `src/` tree at `S:\Programming\Repo\DamianGRZ\LegoTrackOptimizer - V1 kind of worked\src\`: `encoding.py`, `sampling.py`, `operators.py`, `templates.py`, `repair.py`, `problem.py`, `decoder/construction.py`. Below is the synthesis â€” what V1 got right and what V2 lost when it switched encodings.

## Architecture at a glance

```
[Main Loop: n_main genes][Junction Descriptors: max_junctions Ã— 4 genes][Start: 2 genes]
```

- `n_main` = number of non-switch pieces in inventory (dynamic per config)
- `max_junctions` = `min(typeA_switches, typeB_switches)` (a junction needs both)
- Junction descriptor = 4 genes: `(active, position, handedness, n_straights)`
- Start = small `(off_x, off_y)` offset, Â±5 % of boundary, applied on top of decoder auto-centering

Branch geometry is **template-determined**. The chromosome doesn't encode branch pieces â€” it encodes a *recipe* (which template at which position with how many straights), and the decoder materializes the actual pieces with FK-validated closure.

## The ten design wins, ranked by impact on the user-visible difference

### 1. Junction descriptors are parametric, not topological

A junction is 4 genes. To insert a passing siding into a chromosome, the GA:
- Flips `active` from 0 to 1 (one gene)
- Sets `position` to the slot it should attach at (one gene)
- Picks handedness LEFT/RIGHT (one gene)
- Picks `n_straights` for the parallel section (one gene)

Four gene flips and the chromosome contains a fully-specified passing siding. The decoder handles materializing it.

**V2 has no equivalent.** V2 must spawn 2 switches + 2 curves + N straights as separate pieces, wire them via port-pair edges respecting port uniqueness, then verify closure. Roughly 8-12 gene operations, none atomic in the current operator set. Single biggest reason V2 doesn't produce switched layouts.

### 2. Templates with FK closure validation

`templates.py` defines `LEFT_SIDING` and `RIGHT_SIDING` with explicit FK math:
- `compute_branch_endpoint`: traces the branch through approach curve â†’ N straights â†’ return curve
- `compute_out_switch_alignment_error`: position + angle error between branch endpoint and OUT switch port 2
- `is_valid_siding`: True iff position error < 2.0 stud and angle error < 5Â°

The decoder **rejects** any junction whose siding doesn't validate (`construction.py:_inject_switches:250`). Critically â€” when it rejects, **it releases the junction's inventory back** (`_release_junction_inventory`). A bad junction doesn't cost the chromosome any pieces.

This means infeasible junction configurations are essentially free: silently dropped during decoding. The GA can experiment with branch parameters without paying for failures.

### 3. The asymmetric-oval seed insight (the cleverest line in V1)

`sampling.py:_gen_oval_with_siding`, lines 199-265:

```python
main_pieces = (
    [int(curve)] * 8 + [int(STRAIGHT_16)] * (m + 2)
    + [int(curve)] * 8 + [int(STRAIGHT_16)] * m
)
```

The oval is built **asymmetric** by 2 straights. Why? Each switch is 32 studs but each straight is 16 studs, so injecting 2 switches into the second section adds +32 studs (= 2 straights' worth) of axial length. The asymmetric pre-injection shape becomes balanced post-injection â€” the loop closes.

A symmetric `[STRAIGHT_16] * m + [STRAIGHT_16] * m` would NOT close after switch injection. V1 encoded this domain knowledge into the seed generator. This is the kind of detail that turns "GA struggles to discover sidings" into "GA seeds with valid sidings from generation 0".

V2 has no sequential-ring stadium emitter, let alone one that compensates for switch injection.

### 4. Junction-specific mutations: parameter tuning in isolation

`operators.py` has four junction ops (lines 140-178):
- `_toggle_active` â€” flip a junction on/off
- `_reposition_junction` â€” shift `position` by Â±5
- `_change_handedness` â€” swap LEFT â†” RIGHT
- `_adjust_straights` â€” bump `n_straights` by Â±1-3

**This is the missing operator that would have closed the V1 snapshot 8-10 layout.** The 144-piece infeasible-with-CV=0.5 needed exactly one `_change_handedness` flip on its branch's return curve to close. V1 had the operator. The selection scheme just didn't favor running it on that chromosome.

V2's mutation set has nothing analogous. There's no first-class concept of "tune the branch parameters of an existing siding" because there's no first-class concept of "siding". Branches are raw port-pair edges; no operator swaps two specific edges' identities while preserving the rest.

### 5. Closure-aware repair: constructive, not destructive

`repair.py:MainLoopClosureRepair`:

```
total_angle = sum dtheta over active main loop pieces
deficit = 360.0 - total_angle
if deficit > +R40_ANGLE: append R40_LEFT into empty slots
if deficit < -R40_ANGLE: remove R40 from end
```

Repair **adds** missing curves or **removes** excess ones to push the angle budget toward 360Â°. It doesn't "fix infeasibility by deactivating broken pieces" â€” it directly constructs the closure condition.

Combined with the segment-aware encoding (where `INACTIVE` slots are first-class and can be activated by repair), this means a chromosome with 250Â° of curves becomes a chromosome with ~338Â° (5 R40 added) in one repair pass. The GA can discover layouts at 250Â° and let repair lift them to feasibility.

V2's repair (`PortPairRepairPipeline`) doesn't construct missing pieces. It sanitizes edges, drops conflicts, deactivates excess. Destructive repair, not constructive. This is why V2 chromosomes can't grow â€” repair doesn't help them grow; only mutations do, and mutations are too brittle.

### 6. Decoder rolls back failed junctions

`decoder/construction.py:_inject_switches`:

```
if not is_valid_siding(in_state, out_state, template, n_straights, ...):
    _release_junction_inventory(junc, tracker)
    continue
```

If the siding doesn't close, the junction is dropped AND its switches/curves/straights go back to the inventory pool. The chromosome encoded a bad junction, but the *layout* doesn't pay â€” those pieces become available for the main loop or other junctions.

This means the chromosome's "intended" inventory cost is decoupled from its "actual" inventory cost. The GA can over-claim inventory in junction descriptors and the decoder will trim it down to what fits.

V2's decoder/repair doesn't have this â€” port-pair edges that break feasibility get the slots deactivated, but the design isn't framed as "junction failed, return its budget".

### 7. Auto-centering decoupled from chromosome

The `start_pos` segment in V1 is just `(off_x, off_y)` Â±5 % of the boundary. The decoder calls `_auto_center` which computes the layout's bounding box, places its center at the boundary center, then applies the small chromosome offset.

This means:
- The chromosome doesn't have to figure out where to place a 450-stud-diameter stadium inside a 500-stud boundary â€” auto-centering does it.
- `start_pos` mutations are fine-tuning, not gross-placement.
- Crossover doesn't have to combine compatible "where in the boundary" decisions from two parents.

V2's anchor is `(start_x, start_y, start_theta)` with full boundary range. So V2 has to *learn* where to place the loop, which is wasted search effort.

### 8. CROSS_90 injection in decoder

`decoder/construction.py:_apply_crossing_repair` calls `find_crossing_pairs` and injects CROSS_90 pieces at near-perpendicular self-intersections. Self-intersection becomes a feature (figure-8 with crossing) rather than a failure (collision violation). V2 has a partial version of this.

### 9. Per-type inventory constraints, normalized by max_occ

`problem.py:_compute_per_type_inventory_violation`:

```
for t in range(n_types):
    max_occ_t = inventory_by_index.get(t, 0)
    excess = max(0, census[t] - max_occ_t)
    result[t] = excess / max(1, max_occ_t)
```

Instead of one "inventory excess" scalar, V1 has **one constraint per piece type**, normalized by what's available. A run with 80 R40 inventory and 100 R40 used contributes 0.25; a run with 4 switches and 5 used contributes 0.25 too. Constraint pressure is shape-aware: small inventories (switches) get the same weight as big ones (straights), so the GA doesn't spam straights to inflate utilization.

V2 has per-type inventory excess, but the V1 normalizer is what makes "use one extra switch" hurt as much as "use 5 extra straights" â€” the right ratio when switches are precious.

### 10. min_speed bottleneck (not avg)

`problem.py:_evaluate`: `F[1] = -speed_profile.min_speed`. Comment in the code:

> *"avg_speed was a 3-pass harmonic-mean profile that masks dangerous curves behind fast straights â€” a safety failure mode. min_speed is the minimum over per-segment speed caps."*

V1 already used the conservative bottleneck. V2 inherited this. Not V1-specific but worth noting the V1 author thought about why average is wrong.

## What V2 lost when it switched to port-pair edges

| V1 design element | V2 status |
|---|---|
| Partitioned chromosome with first-class branch segment | Lost â€” everything is port-pair edges |
| Junction descriptor (4 genes per branch) | Lost â€” branches are graph topology |
| Template-validated branch closure | Lost â€” branches are discovered, not templated |
| Asymmetric-oval seed (with siding-injection compensation) | Lost â€” heuristic seeds don't anticipate switch injection |
| Junction-specific mutations (tune handedness, n_straights) | Lost â€” no per-branch parameter ops |
| Closure-aware constructive repair (add/remove R40 to hit 360Â°) | Lost â€” repair only sanitizes, doesn't construct |
| Decoder roll-back of failed junctions | Lost â€” bad edges get sanitized, no inventory recovery |
| Auto-centered layout with small chromosomal offset | Lost â€” anchor uses full boundary range |
| CROSS_90 injection at self-intersections | Partially present â€” V2 has crossing repair, but only opportunistic |
| Per-type inventory normalization | Present (V2 inherited it) |
| min_speed bottleneck | Present (V2 inherited it) |

## The two design philosophies, in one sentence each

- **V1**: *"The chromosome describes intent; the decoder + repair handle reality."* Genes are high-level recipes (loop pieces, branch parameters); the decoder validates closure, the repair constructs missing pieces, and the layout that comes out is feasible-by-construction.
- **V2**: *"The chromosome describes reality; the GA must discover validity."* Genes are low-level edges; the decoder is a literal reader; repair only sanitizes; the GA must do all the geometric reasoning itself via mutation/crossover/selection.

V1's philosophy works because the *language* of the chromosome already excludes most invalid layouts â€” branches can't be malformed (templates), inventory can't be over-used (per-type constraints + per-junction validation), closure can be repaired constructively (add curves to hit 360Â°). The GA gets to spend its budget on shape diversity and Pareto exploration, not on satisfying topology.

V2's philosophy is theoretically more expressive (any multigraph is representable) but the operators don't yet pay for that expressiveness. The GA spends its budget on the things V1 got for free.

## What porting V1's wins to V2 would look like

This is the architecture shape, not an implementation plan:

1. **Add a `JunctionsSegment` to V2's chromosome.** New segment with `max_junctions Ã— 4` genes. Encoding becomes `[port_pairs | junctions | anchor]`. Junction descriptor: `(active, anchor_slot, handedness, n_straights)`.

2. **Materialize junctions in the decoder.** Before decoding port-pair edges, walk the active junctions, find each anchor slot in the existing port-pair graph, splice in the template-materialized siding. Validate closure via the template's FK math. If invalid, deactivate the junction.

3. **Add junction mutations.** Port the four operators from V1: `toggle_active`, `reposition`, `change_handedness`, `adjust_straights`. Plug them into ALNS so weights adapt.

4. **Add the asymmetric-ring seed emitter.** Generate `(16 R40 + (m+2) STR16 + 16 R40 + m STR16)` sequential rings with one junction descriptor pre-set. Emit the port-pair edges for the main ring directly. Decoder splices the junction template at decode time.

5. **Add constructive closure repair.** Compute angular sum across all main-loop port-pair edges. If deficit, add R40 edges to inactive slots; if excess, remove R40 edges from end. Same algorithm as V1's `MainLoopClosureRepair`, applied to V2's port-pair representation.

6. **Decouple anchor from chromosome.** Make anchor a `Â±5%` offset on top of decoder auto-centering. Existing `anchor` segment shrinks from full-boundary range to small-offset range.

7. **Roll back failed junctions in decoder.** When a junction fails closure validation, mark its inventory as available again so the rest of the chromosome can use it.

The result would be V2 with V1's structural advantages: parametric branches, template closure, constructive repair, validated junctions, all on top of V2's port-pair edge graph for the cases templates can't cover (figure-8s, multi-component yards, novel topologies).

## What V1 didn't do well (so V2's existence is justified)

V1 cannot express:
- Figure-8s (single loop with internal CROSS_90 traversed twice â€” V1's CROSS_90 injection is opportunistic, not searchable)
- Multi-component yards (V1's main loop is one sequence)
- Novel branch geometries (V1's templates are LEFT_SIDING and RIGHT_SIDING; nothing else)
- Branches off branches (no nested junctions)
- DOUBLE_CROSSOVERs as first-class structural elements

V2's port-pair edge encoding can express all of these. V2's failure isn't that the encoding is wrong â€” it's that the operators don't reach across the expressiveness gap. Adding V1's design wins back into V2 (segments, templates, junctions, constructive repair) closes most of that gap without giving up the expressiveness.

## Conclusion: what made V1 work

V1 worked because **the chromosome was a vocabulary of valid layouts, not a description of arbitrary graphs**. Junctions are template instances with FK-validated closure. Heuristic seeds know about switch-injection arithmetic. Repair constructs missing curves to hit angular closure. The decoder rolls back bad junctions. Auto-centering removes a search dimension. The result is that random chromosomes are mostly feasible, mutations preserve feasibility, and the GA can spend its budget on shape exploration.

V2 abandoned this vocabulary in exchange for a more expressive low-level representation, and the operators haven't yet been built to make that expressiveness affordable.

V1 didn't win because of cleverness in the GA. It won because the encoding was a *grammar*. V2 doesn't have a grammar â€” it has an alphabet.

---

# Part 5 â€” Implementation Plan: V2 + V1's Grammar (Multi-Template, Phased)

## Context

User direction (confirmed):

1. **Chromosome must support multi-path solutions involving switches, CROSS_90, and DOUBLE_CROSSOVER.** Not just passing sidings.
2. **No reverse-template variants** â€” trains are bidirectional by construction; the decoder's 2^J traversal-path enumeration already handles direction.
3. **Goal: use CROSS_90 and DOUBLE_CROSSOVER while maximizing piece usage**, not just include them.
4. **Phased rollout with checkpoints** â€” verify feasibility/utilization/switch usage after each phase before moving on.

Plan preserves everything V2 does well and adds V1's grammar layer as an **additive segment** alongside the existing port-pair edges.

### Out of scope for this plan (deferred)

- **Edge Recombination Crossover (ERX)** â€” Part 2's web research ranked ERX as the highest-payoff single change to address V2's documented "1/500 feasible at p=0.9" `PortPairCrossover` failure mode. Part 5 deliberately does NOT include ERX because the junction segment + template grammar (Phases 4-7) are expected to deliver V1's structural advantage WITHOUT overhauling crossover; junctions are heritability-preserving by construction (Phase 4's semantic-guard swap). If post-Phase-8 measurements still show crossover destroying feasibility, ERX returns as a Phase 9. This is a deliberate scoping decision, not an oversight.
- **Memetic local-search layer** (2-opt, Or-opt as standalone mutations) â€” Part 2 ranked 2-opt as the second-highest priority, principally to improve `min_speed` by reordering pieces along an existing cycle. Phase 5c's junction mutations cover the high-value reordering cases (handedness swap, n_straights tune); standalone 2-opt is deferred to a Phase 9 if needed.
- **Reverse-template variants** â€” see user direction #2; trains are bidirectional, the decoder's 2^J path enumeration already handles direction.

## What V2 Already Does Well â€” Preserve, Don't Replace

| V2 element | Why keep |
|---|---|
| **Port-pair edge encoding** for arbitrary multigraph topology | Templates can't cover every topology; port-pair edges remain the fallback for novel layouts |
| **pymoo NSGA-II + ConstrRankAndCrowding + LegoAdaptiveEpsilon** | Working algorithm, tuning has cost-benefit data |
| **ALNS adaptive sub-operator weights** ([runner.py:271](src_v2/runner.py:271)) | State-of-the-art operator selection; new junction mutations slot in as additional ops |
| **EpsilonArchiveCallback** ([runner.py:268](src_v2/runner.py:268)) | Right hook for topology-aware admission rules; just needs new admission criterion |
| **PortGraphDuplicateElimination** | Phenotype-aware dedupe; will continue to work as junctions add to phenotype |
| **Decoder cycle basis** ([decoder.py:692](src_v2/decoder.py:692) `nx.cycle_basis`) | Used downstream for branch/cycle labeling; junction materialization plugs in BEFORE cycle decode |
| **Per-type inventory normalization** ([problem.py:208](src_v2/problem.py:208)) | Already inherited from V1's design |
| **min_speed bottleneck objective** | Already correct |
| **Train physics** (`compute_speed_profile`) | Untouched |
| **Repair pipeline architecture** ([repair.py:51](src_v2/repair.py:51)) | Add new repair stages, don't replace existing ones |
| **Dynamic chromosome dimensions from inventory** ([encoding.py:189](src_v2/encoding.py:189)) | Junction segment extends `PortPairDimensions` â€” same dynamic-sizing principle |

## What V1's Grammar Adds â€” Multi-Template Junction Segment

New chromosome layout (additive â€” port-pair edges remain):

```
[piece_slots: N_max | flips: N_max | rotates: N_max | port_pairs: E_max Ã— 4 | junctions: J_max Ã— 5 | anchor: 3]
```

`J_max` derives dynamically from inventory:
- `J_max_passing` = min(typeA_switches, typeB_switches) per V1
- `J_max_figure8` = floor(CROSS_90_inventory)
- `J_max_dc` = floor(DOUBLE_CROSSOVER_inventory)
- `J_max = J_max_passing + J_max_figure8 + J_max_dc`

**Junction descriptor (5 genes):** `(active, anchor_slot, template_kind, param_a, param_b)`

| Gene | Range | Meaning |
|---|---|---|
| `active` | 0..1 | Whether this junction is materialized |
| `anchor_slot` | 0..N_max-1 | Slot in piece array where template attaches |
| `template_kind` | 0..3 | 0 = PASSING_SIDING_LEFT, 1 = PASSING_SIDING_RIGHT, 2 = FIGURE_8_CROSS, 3 = PARALLEL_DC_BRIDGE |
| `param_a` | template-specific | e.g. n_straights for siding, lobe_size for figure-8, parallel_run_length for DC |
| `param_b` | template-specific | e.g. unused for siding, secondary_curve_handedness for figure-8, bridge_offset for DC |

**Why 5 genes per junction (not 4 like V1)**: V1 only had passing sidings, so 4 genes sufficed. We need a `template_kind` selector to discriminate between siding / figure-8 / DC, hence 5.

**No reverse variants.** Trains traverse the graph in either direction; the decoder's 2^J path enumeration already covers both. Reverse templates were a V1 artifact of an earlier era; V1's current `templates.py` only has LEFT_SIDING and RIGHT_SIDING.

## The Three Templates â€” Materialization Spec

### Template 0/1: PASSING_SIDING_LEFT, PASSING_SIDING_RIGHT

Direct port from V1's `LEFT_SIDING` / `RIGHT_SIDING` (`src/templates.py` in V1 repo).

**Materialization (decoder step before edge decode)**:
1. Take `anchor_slot` from main loop. Replace it with `IN_switch` (LEFT_IN or RIGHT_IN).
2. Walk forward along main loop FK to find `OUT_position` matching `compute_required_main_distance(template, n_straights)`.
3. Replace OUT_position with `OUT_switch`.
4. Materialize branch: approach_curve + N straights + return_curve. Add as port-pair edges between IN_switch.port_C and OUT_switch.port_C.
5. Validate: `is_valid_siding(in_state, out_state, template, n_straights, position_tolerance=2.0, angle_tolerance=5.0)`.
6. If invalid, deactivate junction AND release inventory back to pool.

`param_a` = n_straights, `param_b` = unused (reserved).

### Template 2: FIGURE_8_CROSS

A CROSS_90 inserted at anchor_slot, with a secondary lobe returning through it.

**Materialization**:
1. Replace `anchor_slot` with CROSS_90.
2. Materialize secondary lobe: `param_a` curves + return through CROSS_90's port C/D.
3. Validate that the secondary lobe closes back to CROSS_90 within tolerance.
4. If valid, the layout becomes a figure-8 with two cycles sharing the CROSS_90.

`param_a` = lobe_size (number of R40 curves in secondary lobe), `param_b` = secondary_curve_handedness (0 = LEFT, 1 = RIGHT).

A 16-curve secondary lobe gives a closed inner loop; smaller lobes need straights to close.

### Template 3: PARALLEL_DC_BRIDGE

A DOUBLE_CROSSOVER bridging two parallel main-loop sections.

**Materialization**:
1. Identify two roughly-parallel sections in the main loop (use existing `_find_parallel_segments` heuristic from V2 if available, or compute from FK).
2. Replace one piece in section A with DOUBLE_CROSSOVER's port AB; replace one piece in section B with DOUBLE_CROSSOVER's port CD.
3. Validate that the DC's geometry matches the inter-section spacing.
4. If invalid, deactivate.

`param_a` = bridge_offset_along_section (which slot in the parallel section), `param_b` = inter_section_distance (used to validate DC fit).

This is the hardest template; details may need refinement in implementation. Phase it last.

## Phase Roadmap (Eight Phases, Each with a Verification Checkpoint)

Each phase is **independently testable** and produces a measurable improvement (or no regression) before the next starts. Order is by risk and dependency.

### Phase 1 â€” Constructive Closure Repair (No encoding change)

**Goal**: Repair adds/removes R40 edges in cycles whose angular sum is off-target, instead of just sanitizing.

**Critical correction from expert review**: V2 has no sequential main loop, so V1's "splice R40 into the chain" pattern does not transfer directly. The repair must operate on the **largest cycle of the decoded `PortGraph`**, not on a flat angle sum across all active slots. Active slots not part of any cycle contribute angle budget that will never close â€” their angle is irrelevant.

**Files**:
- [src_v2/repair.py](src_v2/repair.py) â€” new class `CycleClosureRepair` (renamed from `MainLoopClosureRepair` to reflect cycle semantics)
- [src_v2/runner.py:231](src_v2/runner.py:231) â€” wire into `PortPairRepairPipeline` after edge sanitization + inventory enforcement, before crossing injection

**Algorithm**:
```
For each chromosome:
    1. Decode chromosome â†’ PortGraph (reuse existing decoder)
    2. Identify the largest connected component with â‰¥1 cycle
    3. For each cycle in that component (use existing nx.cycle_basis at decoder.py:692):
        a. Compute sum of FK dtheta over the cycle's edges
        b. deficit = 360 - cycle_angle_sum (mod 360, signed)
        c. If |deficit| <= R40_ANGLE: done; skip this cycle
        d. If deficit > 0:
           - Pick a port-pair edge in the cycle
           - Find an inactive slot; activate it with R40_LEFT (or R40_RIGHT matching deficit sign)
           - Splice the edge through the new slot (5-step edge surgery,
             same as V2's existing introduce_crossing pattern in
             structural_mutations.py): remove old edge, add two new edges
             through the new slot's two ports, update port-pair table
           - Verify port-uniqueness; rollback if violated
        e. If deficit < 0:
           - Find an R40 slot in the cycle whose handedness matches the excess sign
           - Deactivate it; merge the two adjacent edges into one (rejoining the cycle)
        f. Cap at max_corrections = 4 per repair pass
    4. Respect inventory: no curve added if its piece type's inventory is exhausted
    5. Cycle iteration order: largest cycle first (most likely the main loop)
```

**Files to study before implementation**:
- `src_v2/structural_mutations.py:introduce_crossing` â€” already does the 5-step edge surgery pattern this repair needs
- `src_v2/decoder.py:_iter_cycles` â€” already uses `nx.cycle_basis` to enumerate fundamental cycles
- V1 `repair.py:MainLoopClosureRepair._add_curves` â€” reference for the angle-budget arithmetic

**Interaction with Phase 5b (asymmetric-oval seed)**:
The asymmetric `(m+2) vs m` straight distribution exists deliberately â€” post-switch-injection adds +32 studs which balances the loop. **Phase 1 closure repair must skip cycles that contain an active junction's anchor slot** (or equivalently, must run before Phase 5a's junction materialization in the pipeline). Otherwise the repair will "heal" the asymmetry back to symmetric, undoing the seed's intent. Add a guard: `if any active junction.anchor_slot is in cycle: skip closure repair for this cycle`.

**Interaction with existing `closure_repair_lamarckian` mutation**:
V2 already has a `closure_repair_lamarckian` sub-operator in `PortPairMutation.OP_WEIGHTS` (operators.py:866, weight 0.03) that attempts similar work via the mutation route. Once `CycleClosureRepair` runs in the Repair pipeline, the mutation becomes redundant and competes with itself for ALNS weight. Decision required at Phase 1 implementation time: (a) **Remove** the `closure_repair_lamarckian` entry from `OP_WEIGHTS` and re-normalize weights; (b) **Refactor** the mutation to delegate to the new `CycleClosureRepair.repair_one` method (so the mutation becomes a stochastic in-line trigger of the same logic, useful when repair runs at end of generation and mutation runs per-individual); (c) **Keep both** with `closure_repair_lamarckian` weight reduced to 0.01 as a backup fallback in case repair is bypassed. **Recommend (a)** â€” single source of truth is cleaner; ALNS removes one operator from the budget which other ops then absorb. Document the decision in the Phase 1 phase notes.

**Verification**: run `default` config (24-piece inventory). Pre-fix, V2 stops at 16-piece small oval. Post-fix, expect chromosomes that started with ~250Â° cycle angle to grow toward 360Â° during repair, increasing population utilization.

**Risk**: medium (revised from low). Cycle-aware repair is more invasive than the V1-style flat angle repair. The 5-step edge surgery is exactly the operation V2's current operators don't do atomically â€” implementing it correctly inside Repair is the heart of the phase.

**Checkpoint**: feasibility rate non-decreasing; mean utilization +5 % or more on `default`; no regression on `with_switches` Phase 5b seeds (asymmetric-oval pieces preserved).

### Phase 2 â€” Sequential-Ring Stadium Heuristic Seed (No encoding change)

**Goal**: Seed initial population with V1-style stadiums of varying length.

**Implements Option B from Part 1** (sequential-chain heuristic emitter). Part 1 ranked this as the cheapest fix that captures V1's "implicit chain â†’ closed walk by construction" advantage; Phase 2 is the concrete realization of that recommendation, on top of V2's port-pair encoding (no encoding change required).

**Files**:
- [src_v2/operators.py:617](src_v2/operators.py:617) â€” new emitter `_emit_sequential_ring_stadium` in `_HEURISTIC_EMITTERS`

**Critical correction from expert review**: V1's stadium uses **two corners of 8 curves each** (16 R40 total = a full 360Â°), not four corners of 4 curves each (which would consume 16 R40 just for corners). The plan's earlier draft was wrong. The correct V1 pattern is at `S:\Programming\Repo\DamianGRZ\LegoTrackOptimizer - V1 kind of worked\src\sampling.py:_gen_oval` (line ~146): two banks of 8 curves with straight runs between them.

**Algorithm** (corrected):
```
Inventory check first:
    If inv[R40_LEFT] >= 16:
        max_straights_total = inv[STRAIGHT_16]
        max_straights_per_side = max_straights_total // 2
        m_max = min(max_straights_per_side, _fit_oval_straights_per_section(boundary, ...))
        For m in sorted({m_max, max(1, int(m_max * 0.75)), max(1, int(m_max * 0.5)), max(1, int(m_max * 0.25))}, reverse=True):
            pieces = [R40_LEFT Ã— 8, STRAIGHT_16 Ã— m, R40_LEFT Ã— 8, STRAIGHT_16 Ã— m]
            Emit port-pair edges for the sequential ring:
                for i in range(N-1): edge(slot_i.port_B, slot_{i+1}.port_A)
                close: edge(slot_{N-1}.port_B, slot_0.port_A)
            Yield (slots, edges) as a Pattern
    Repeat with R40_RIGHT (also requires >= 16 inventory check)
    If both LEFT and RIGHT >= 8 each, also emit symmetric mixed-handedness variants
```

Bound `m` by both inventory and boundary fit (V1's `_fit_oval_straights_per_section` algorithm with `extra_axial=0` for the symmetric stadium â€” 32-stud reservation is only for siding seeds in Phase 5b).

Port `_fit_oval_straights_per_section` directly from V1's `sampling.py:65-83`.

**Verification**: with `with_switches` config (168 inventory), expect heuristic seeds to produce 96-piece+ stadium variants in initial population. After 200 gens, mean utilization should land near V1's 58.5 % on `with_switches` even before junction segment exists.

**Risk**: low. New heuristic only, doesn't modify existing operators.

**Checkpoint**: best-feasible piece count on `with_switches` â‰¥ 80 pieces.

### Phase 3 â€” Auto-Centering (Decoder refactor, anchor range shrunk)

**Goal**: Decoder auto-centers layout in boundary; chromosome anchor becomes Â±5 % offset.

**Critical correction from expert review**: shrinking anchor range invalidates pre-Phase-3 chromosomes AND silently corrupts pre-Phase-3 Îµ-archive JSONs (existing `EpsilonArchive.to_json()` at [epsilon_archive.py:104-113](src_v2/epsilon_archive.py:104) has no version stamp). Loading a pre-Phase-3 archive after Phase 3 deploys would pass anchor values outside the new bounds; repair would clamp them silently, producing garbage placements with no warning.

**Files**:
- [src_v2/decoder.py](src_v2/decoder.py) â€” new step `_auto_center_layout` after pose propagation, before boundary check
- [src_v2/encoding.py:189](src_v2/encoding.py:189) â€” anchor bounds shrink from full-boundary to Â±5 % of boundary; add `ENCODING_VERSION: int` constant
- [src_v2/epsilon_archive.py](src_v2/epsilon_archive.py) â€” `EpsilonArchive.to_json` writes `"encoding_version"` field; add `EpsilonArchive.from_json` (currently missing) that refuses mismatched versions
- Any other JSON-producing path (`runner.py:save_results`, etc.) â€” stamp `encoding_version` in output

**Algorithm**:
```
After FK propagation, before boundary constraint check:
    bbox = compute_layout_bbox(graph)
    layout_center = bbox.midpoint
    boundary_center = (boundary.min + boundary.max) / 2
    centering_offset = boundary_center - layout_center
    Apply chromosome's small (off_x, off_y) offset on top
    Translate all states by centering_offset + chromosome_offset
```

**Versioning** (new requirement, lifts the silent-corruption risk to a hard fail):
```python
# encoding.py
ENCODING_VERSION: int = 2  # Phase 3 bumps from 1 to 2

# epsilon_archive.py:to_json (additive)
data["encoding_version"] = ENCODING_VERSION

# epsilon_archive.py:from_json (NEW)
def from_json(path) -> EpsilonArchive:
    payload = json.loads(Path(path).read_text())
    if payload.get("encoding_version") != ENCODING_VERSION:
        raise EncodingVersionMismatch(
            f"archive at {path} has encoding_version={payload.get('encoding_version')!r}, "
            f"current is {ENCODING_VERSION}; archive cannot be safely loaded"
        )
    # ... reconstruct archive ...
```

Bump `ENCODING_VERSION` again in Phase 4 (junction segment changes n_var).

**Verification**: with `default` config, layouts should consistently land near boundary center regardless of anchor mutation. Boundary violations should drop substantially. Loading a pre-Phase-3 archive raises `EncodingVersionMismatch` cleanly.

**Risk**: medium. Changes how boundary constraint is computed. Existing chromosomes interpret anchor differently. Mitigate by re-running same configs and confirming no regression. Mid-thesis archive loads from before this phase fail loudly instead of silently corrupting.

**Checkpoint**: boundary-violation count on `default` drops; feasibility rate non-decreasing; `from_json` rejects pre-Phase-3 archives with explicit error.

### Phase 4 â€” Junction Segment Scaffolding (Chromosome layout change, all ops no-op on it)

**Goal**: Add `junctions` segment to chromosome. All existing operators see it but don't touch it. Decoder ignores it. Pure structural extension.

**Critical correction from expert review**: Junction crossover (per-slot 50% swap, mirroring V1's `PartitionedCrossover`) has the same failure mode as V2's existing `PortPairCrossover`: the inherited `anchor_slot` from parent A may point into parent B's slot context where the slot is INACTIVE or holds a piece that lacks a branch port. `JunctionValidityRepair` then deactivates the junction â†’ silent junction-kill, eroding heritability. The crossover must **guard at swap time**, not "fix later in repair".

Also: the canonical hash update for `PortGraphDuplicateElimination` should **NOT** happen in Phase 4. Phase 4's decoder ignores junctions, so two chromosomes with identical port-pair edges but different junction descriptors still produce identical layouts. Hashing junctions in Phase 4 would cause phenotypic duplicates to be treated as distinct, wasting population diversity. Defer the canonical hash update to Phase 5a (when junctions actually affect decoded output).

**Files**:
- [src_v2/encoding.py](src_v2/encoding.py) â€” new `JunctionSegmentDimensions` integrated into `PortPairDimensions`. Genes-per-junction = 5. `J_max` computed from inventory. Bump `ENCODING_VERSION` from 2 (Phase 3) to 3.
- [src_v2/encoding.py](src_v2/encoding.py) â€” `generate_bounds` extended for junction gene ranges
- [src_v2/encoding.py](src_v2/encoding.py) â€” `validate_chromosome` extended for junction gene ranges (must update atomically with `generate_bounds` â€” see Golden Rule 8 in Part 7)
- [src_v2/encoding.py](src_v2/encoding.py) â€” `__post_init__` on `PortPairDimensions` asserts `n_var < np.iinfo(np.int16).max` (Golden Rule 2 in Part 7)
- [src_v2/encoding.py](src_v2/encoding.py) â€” `create_empty_chromosome` initializes junctions to `(active=0, anchor=0, kind=0, param_a=0, param_b=0)`
- [src_v2/operators.py:773](src_v2/operators.py:773) â€” `PortPairCrossover` semantically-guarded per-slot junction swap (see algorithm below)
- [src_v2/operators.py:833](src_v2/operators.py:833) â€” `PortPairMutation` skips junction segment (no junction ops yet)
- [src_v2/repair.py](src_v2/repair.py) â€” `PortPairRepairPipeline` clamps junction genes to bounds AND deactivates junctions whose `anchor_slot` is INACTIVE or non-branch-capable in the post-repair slot context
- [src_v2/decoder.py](src_v2/decoder.py) â€” reads junctions but does nothing yet (returns same layout)
- [src_v2/canonical.py](src_v2/canonical.py) â€” **NO change in Phase 4** (junction descriptors NOT included in canonical hash until Phase 5a)

**Junction crossover algorithm (semantic guard)**:
```python
# For each junction slot j in dims.J_max:
def _swap_junction_if_safe(c1, c2, p1, p2, j, dims):
    # Read junction descriptors
    j1_active, j1_anchor, j1_kind, _, _ = read_junction(p1, dims, j)
    j2_active, j2_anchor, j2_kind, _, _ = read_junction(p2, dims, j)
    # Identify whether each parent's anchor_slot is currently active
    # in the OTHER parent's slot context (where the descriptor would land)
    p1_can_receive_p2 = is_branch_capable_slot(c1, p2_anchor=j2_anchor)
    p2_can_receive_p1 = is_branch_capable_slot(c2, p1_anchor=j1_anchor)
    if np.random.random() >= 0.5:
        return  # no swap
    if not (p1_can_receive_p2 and p2_can_receive_p1):
        # Either side would receive a junction with invalid anchor context.
        # Deactivate that junction in the receiver instead of swapping blindly.
        if not p1_can_receive_p2 and j2_active:
            write_junction(c1, dims, j, active=0, anchor=j2_anchor, kind=j2_kind, ...)
        if not p2_can_receive_p1 and j1_active:
            write_junction(c2, dims, j, active=0, anchor=j1_anchor, kind=j1_kind, ...)
        return
    # Both sides receive valid context â€” swap atomically
    base = dims.junc_start + j * 5
    end = base + 5
    c1[base:end], c2[base:end] = p2[base:end].copy(), p1[base:end].copy()
```

`is_branch_capable_slot(chromosome, anchor)` returns True iff the chromosome's slot at `anchor` is active AND its piece spec has a branch port (CROSS_90, switches, DOUBLE_CROSSOVER) OR is a regular piece that can be replaced by a switch (any straight or curve, since template materialization replaces the slot). Conservative variant: only require the slot is active.

**Verification**: existing configs run identically (junctions ignored by decoder). Population characteristics unchanged. Chromosome length increases per inventory. Junction crossover with all-inactive junctions degenerates to no-op.

**Risk**: medium-high (revised from medium). Old saved chromosomes incompatible (encoding version bump). Encoding-touching code spread across many files; need to update consistently. Junction crossover guard is non-trivial but essential â€” implementing only the unguarded swap will replicate V2's existing crossover-destroys-feasibility failure mode for the junction segment.

**Checkpoint**: with no junction templates active (all chromosomes have all-inactive junction descriptors after Phase 4 init), all metrics on `default` / `with_switches` / `with_crossing` match Phase 3 baseline within noise.

### Phase 5a â€” Template Machinery: PASSING_SIDING

**Goal**: Decoder materializes active passing-siding junctions with FK validation and rollback.

**Critical correction from expert review**: V2 has no sequential main loop pre-decode. The plan's earlier draft said "walk forward along main loop FK" before edge decode â€” there is no main loop to walk until edges are decoded into a `PortGraph` and cycles are identified. The materialization order must be **after** edge decode, not before. Specifically:

1. Read main loop active pieces (existing `_read_main_loop`)
2. **Decode port-pair edges into `PortGraph`** (existing `_decode_edges`)
3. Compute connected components and cycles (existing union-find + `nx.cycle_basis`)
4. **Materialize template junctions** into the largest cycle of each component (NEW â€” this is what Phase 5a adds)
5. Self-intersection / CROSS_90 injection (existing crossing repair)
6. Compute FK chain across the augmented graph
7. Enumerate 2^J traversal paths
8. Auto-center within boundary

**Files**:
- New `src_v2/templates.py` â€” port `PassingSidingTemplate`, `LEFT_SIDING`, `RIGHT_SIDING`, `compute_branch_pieces`, `compute_branch_endpoint`, `compute_out_switch_alignment_error`, `is_valid_siding`, `get_siding_inventory_requirements`, `check_siding_inventory` from V1's `src/templates.py` (use `@dataclass(frozen=True)` for pickling)
- [src_v2/decoder.py](src_v2/decoder.py) â€” new step `_materialize_junctions` AFTER port-pair edge decode, AFTER cycle identification, BEFORE crossing repair
  - For each active junction: validate template inventory, locate anchor_slot in cycle space, splice in template-materialized siding edges, validate closure via FK math, or roll back and release inventory
- [src_v2/repair.py](src_v2/repair.py) â€” `JunctionValidityRepair` clamps junction genes (port from V1) â€” runs in the repair pipeline before evaluation, so junction descriptors arriving at the decoder are already in-bounds

**Junction materialization algorithm** (revised, post-edge-decode):
```
def _materialize_junctions(graph, junctions, inventory_tracker):
    active = sorted_by_anchor_slot([j for j in junctions if j.active])
    for j in active:
        # 1. Anchor slot must exist as an active vertex in graph
        if j.anchor_slot not in graph.active_slots: continue

        # 2. Anchor must be in a cycle (else there's no main loop to attach to)
        cycle = graph.cycle_containing(j.anchor_slot)
        if cycle is None: continue

        # 3. Inventory check
        template = TEMPLATES[j.template_kind]
        reqs = template.inventory_requirements(j.param_a, j.param_b)
        if not inventory_tracker.can_use_batch(reqs): continue  # silent skip â€” no cost

        # 4. FK validation: walk from anchor along cycle to find OUT position
        #    (V1 logic, but operating on cycle edges, not a flat sequence)
        in_state = graph.fk_state_at(j.anchor_slot)
        out_state, out_slot = template.find_out_position(graph, cycle, j.anchor_slot, j.param_a)
        if out_slot is None: continue

        # 5. Closure validation: branch endpoint must align with OUT switch port_C
        if not template.is_valid(in_state, out_state, j.param_a, j.param_b): continue

        # 6. Commit: splice template into graph, claim inventory
        inventory_tracker.use_batch(reqs)
        template.materialize_into(graph, j.anchor_slot, out_slot, j.param_a, j.param_b)
```

The "find OUT position" step is V1's `_find_out_position` (`decoder/construction.py:_find_out_position`) â€” walk cycle edges accumulating x-distance, return the slot whose cumulative distance best matches `compute_required_main_distance(template, n_straights)`. Port that algorithm directly, adapted to walk cycle edges instead of a sequential array.

**Verification**: chromosomes with `template_kind = PASSING_SIDING_*` and a valid `n_straights` decode to layouts containing 2 switches and a closed branch. CV constraint metric `incomplete_switch_ratio` ([problem.py:23](src_v2/problem.py:23)) drops to 0 for those chromosomes.

**Risk**: high (revised from medium). The materialization-after-edge-decode order is more complex than V1's pre-decode pattern; cycle identification + anchor location in cycle space must work correctly for the splice to land in the right place. Bugs here invalidate Phase 5b (seeds depend on materialization) and Phase 5c (mutations rely on existing materialized junctions).

**Checkpoint**: hand-crafted test chromosome with one active siding (anchor_slot in main cycle, valid n_straights) decodes to a valid passing-siding layout (CV from incomplete_switch_ratio = 0, n_switch_pairs = 1).

### Phase 5b â€” Asymmetric-Oval Siding Heuristic Seed

**Goal**: Seed initial population with V1's asymmetric-oval-with-siding pattern.

**Files**:
- [src_v2/operators.py](src_v2/operators.py) â€” new emitter `_emit_asymmetric_oval_with_siding`
- Direct port from V1's `_gen_oval_with_siding` (`sampling.py:199-265`), adapted to emit junction descriptors + port-pair edges

**Algorithm**:
```
m = _fit_oval_straights_per_section(dims, n_straights, extra_axial=32.0)  # Reserve 32 stud for siding injection
For each handedness in (LEFT, RIGHT):
    main_pieces = [R40 Ã— 8, STRAIGHT_16 Ã— (m+2), R40 Ã— 8, STRAIGHT_16 Ã— m]
        # Two corner banks of 8 R40 each (= 16 total = full 360Â°), NOT four banks of 4.
        # Asymmetric (m+2 vs m) reserves 32 studs for siding's switch-injection +32 stud surplus.
        # Direct port from V1's _gen_oval_with_siding (sampling.py:230-233 in V1 repo).
    Emit port-pair edges for the sequential ring
    Emit junction descriptor: (active=1, anchor_slot=section_start+offset, kind=PASSING_SIDING_handedness, param_a=n_branch_straights)
For multiple offsets, multiple n_branch_straights
```

**Verification**: with `with_switches` config, initial population contains chromosomes with 1 active passing-siding junction. After Phase 5a's decoder materializes them, ~30 % of feasible solutions in early generations have switches.

**Risk**: low (depends on Phase 5a working). Worst case: seeds get rejected by validation, fall back to no-junction stadium.

**Checkpoint**: feasible-with-switches solutions present in population by gen 50.

### Phase 5c â€” Junction Mutations for Sidings

**Goal**: Add junction-specific mutation operators so the GA can tune existing sidings.

**Files**:
- [src_v2/operators.py:833](src_v2/operators.py:833) â€” extend `PortPairMutation` with four junction sub-ops:
  - `_mutate_junction_toggle_active` â€” flip `active`
  - `_mutate_junction_reposition` â€” shift `anchor_slot` by Â±5 (clamped to N_max), prefer nearest active branch-capable slot to current value (Golden Rule 9 in Part 7 â€” preserve locality)
  - `_mutate_junction_swap_handedness` â€” flip `template_kind` between PASSING_SIDING_LEFT and PASSING_SIDING_RIGHT
  - `_mutate_junction_adjust_straights` â€” bump `param_a` by Â±1-3 (clamped to inventory)
- ALNS picks up the new sub-ops via `OP_WEIGHTS` table; **initial weight = mean of existing operator weights**, not a low sentinel like 0.05 (Golden Rule 14 in Part 7 â€” avoid cold-start starvation)

**Verification**: take the V1 snapshot 8-10 layout (144 pcs, CV=0.5, branch return curve angle off by 45Â°). Apply `_mutate_junction_swap_handedness` to flip the branch's return-curve handedness. **Expected outcome**: branch angle error drops from 45Â° to either 0Â° (if 1 swap suffices â€” single curve is the cause) or 22.5Â° (if 2 swaps needed â€” two curves at fault). The plan's earlier draft claimed "one swap closes it" â€” the V1 FK math in `templates.py:R40_LEFT_FK = (15.307, 3.045, 22.5)` makes this credible for a single mis-handed return curve, but a 45Â° error could also be caused by two curves of 22.5Â° each. Implementation should test both: if 1 swap closes the branch, we get 144-piece feasible directly; otherwise, 2 sequential `swap_handedness` mutations close it (one mutation per generation, archive admission preserves the intermediate). Either way is achievable in a few generations.

**Risk**: medium. Wrong mutation ranges (e.g. n_straights mutation range too aggressive) can destroy valid sidings. ALNS cold-start starvation if new ops start with low weights.

**Checkpoint**: switch-using feasible solutions appear in Pareto archive on `with_switches` by gen 100. The 144-piece V1-snapshot-8-10 chromosome (or its equivalent re-derived from a Phase 5b seed) becomes feasible within â‰¤2 generations once it's in the population.

### Phase 6a â€” Template Machinery: FIGURE_8_CROSS

**Goal**: Decoder materializes figure-8 templates: a CROSS_90 at anchor with a secondary lobe.

**Files**:
- [src_v2/templates.py](src_v2/templates.py) â€” new `Figure8Template`, `compute_lobe_pieces`, `compute_lobe_closure`, `is_valid_figure8`
- [src_v2/decoder.py](src_v2/decoder.py) â€” extend `_materialize_junctions` to handle `FIGURE_8_CROSS` template_kind

**Algorithm**:
```
For an active FIGURE_8_CROSS junction:
    Replace anchor_slot piece with CROSS_90
    Materialize secondary lobe: param_a curves of param_b handedness + closing pieces
    Lobe enters CROSS_90 port C, exits port D (or vice versa)
    Validate lobe closure within tolerance
    If invalid, deactivate AND release CROSS_90 + lobe pieces back to inventory
```

**Verification**: hand-crafted test chromosome with one FIGURE_8_CROSS decodes to a closed figure-8 with two cycles sharing the CROSS_90.

**Risk**: medium. Lobe geometry is more complex than passing-siding (two cycles, more closure constraints).

**Checkpoint**: figure-8 layouts appear in population on `with_crossing` config.

#### Decision log (2026-05-07)

**Architectural decision: FULL_TEMPLATE_AS_PLANNED.**

Closure search ([tools/figure8_closure_search.py](tools/figure8_closure_search.py))
went through three iterations before producing a definitive answer:

1. **v1 (perpendicular-only)** — single-handedness 4-segment patterns,
   `n_R40 ∈ {4..16}`, `n_STR ∈ [0,16)`, target = port C/D pose with
   *inverted heading*. 8 false positives because the inverted-heading
   target doesn't match the chromosome decoder's literal port-pose
   matching contract.

2. **v1 corrected** — same search space, target = literal port-D pose.
   0 closures across all 5 lobe topology variants (4 diagonal pairings
   + 1 perpendicular control).

3. **v2 (diagonal + outer-port-extension straights)** — wider search:
   `n_R40 ∈ [2, 12]` with full 2^n_R40 handedness enumeration up to
   `n_R40=8`, M1/M2 outer STRs ∈ [0, 16] anchored at start/end ports,
   inner per-gap STRs ∈ [0, 4]. **20 closing tuples found** across the
   four diagonal-quadrant pairings (B→C, B→D, D→A, C→A); the
   perpendicular C→D control still found 0.

**The closing parametric family**: each lobe is `M1 STR_16 + 12 R40
same-handed + M2 STR_16` (no inner straights), with `(M1, M2)` ∈
`{(2, 2), (2, 3)}` depending on which port pair the lobe spans.

| Lobe | Handedness | (M1, M2) | Total STR | Residual |
|---|---|---|---|---|
| B→C (lower-right) | LEFT | (2, 3) | 5 | dx=+0.0001, dy=-0.0014, dθ=0° |
| B→D (upper-right) | RIGHT | (2, 3) | 5 | dx=+0.0001, dy=+0.0014, dθ=0° |
| D→A (upper-left) | LEFT | (2, 2) | 4 | dx=+0.0014, dy=+0.0001, dθ=0° |
| C→A (lower-left) | RIGHT | (2, 2) | 4 | dx=+0.0014, dy=-0.0001, dθ=0° |

The 12 R40 same-handed contribute 270° net rotation, which combined
with the cross's port-pair heading offset produces the chain heading
needed to align with the next port. The `(M1, M2)` outer straights
handle the residual position offset.

**Two viable diagonal-pair figure-8 topologies** (each forms a complete
figure-8 with two lobes that don't physically intersect):

1. **"Top-left + bottom-right"** — lobes D→A + B→C, both LEFT-handed.
   24 R40 + 9 STR + 1 cross = 34 pieces.
2. **"Top-right + bottom-left"** — lobes B→D + C→A, both RIGHT-handed.
   Mirror image, same piece counts.

**The "perpendicular" topology** (lobes B↔A horizontal + C↔D vertical
— the pre-fix `_emit_figure_8` design) is **geometrically impossible**:
the closure search confirmed 0 tuples close in any (n_R40, n_STR,
handedness, interleaving) configuration tested. The two perpendicular
lobes would have to physically intersect outside the cross on a flat
plane.

**Implications:**

- `_emit_figure_8` (existing direct port-pair emitter) updated to emit
  the two diagonal-pair variants only. See
  [src_v2/operators.py](src_v2/operators.py).
- A regression test asserts the emitted layout's closure residuals
  satisfy ≤2 stud / ≤2 deg via direct decode
  ([tests/test_phase6a_figure_8_closure.py](tests/test_phase6a_figure_8_closure.py)).
- **The Phase 6a junction materializer's parameter domain** is the
  closing family above: `param_a` selects the diagonal pair (0 = top-left
  + bottom-right LEFT, 1 = top-right + bottom-left RIGHT). `M1`/`M2` are
  fixed at the smallest closing values per lobe but could become tunable
  parameters in a future refinement.
- **Known integration issue (follow-up):** `CycleClosureRepair` (Phase 1,
  in [src_v2/repair.py](src_v2/repair.py)) targets a fixed 16-R40 cycle
  budget. Figure-8 lobes use 12 R40 — repair currently tries to splice
  4 extra R40s into each lobe and structurally breaks the closure. The
  regression test bypasses `PortPairProblem._evaluate` and uses
  `decode_chromosome` directly. Production runs that route figure-8
  chromosomes through `_evaluate` will see closure errors until repair
  is taught either to (a) skip cycles whose residuals are already small
  or (b) recognize figure-8 lobes via a `skip_anchor_slots`-style
  Coupling-C extension. Decide concurrently with Phase 6b's seed
  emitter which mechanism to wire.
- **Phase 6c (junction mutations for figure-8) is on hold** pending a
  separate decision. The diagonal-pair parameter space is small (2
  discrete configurations per cross slot). Decide post-Phase-8 when
  topology archive's effect on figure-8 survival is empirically known.

### Phase 6b â€” Figure-8 Heuristic Seed

**Goal**: Seed with figure-8 stadiums.

**Files**: [src_v2/operators.py](src_v2/operators.py) â€” new `_emit_figure8_stadium` emitter

**Algorithm**:
```
main_pieces = [R40 Ã— 8, STR16 Ã— m, R40 Ã— 8, STR16 Ã— m]  (oval)
Junction: (active=1, anchor_slot=midpoint, kind=FIGURE_8_CROSS, param_a=lobe_curve_count, param_b=handedness)
```

**Verification**: figure-8 layouts in initial pop on `with_crossing`.

**Checkpoint**: feasible figure-8 (CROSS_90 used, two cycles) appears by gen 100.

### Phase 7a â€” Template Machinery: PARALLEL_DC_BRIDGE

**Goal**: Decoder materializes DOUBLE_CROSSOVER bridges between parallel sections.

**Files**:
- [src_v2/templates.py](src_v2/templates.py) â€” new `ParallelBridgeTemplate`
- [src_v2/decoder.py](src_v2/decoder.py) â€” extend `_materialize_junctions` for `PARALLEL_DC_BRIDGE`

**Algorithm**:
```
For an active PARALLEL_DC_BRIDGE junction:
    Find two parallel sections in main loop (FK-based parallel detection)
    Insert DOUBLE_CROSSOVER spanning between them
    Validate DC's 4-port geometry matches the inter-section distance
    If invalid, deactivate AND release DC piece
```

**Risk**: high. DC is the trickiest piece. Parallel-section detection is non-trivial. Defer if Phase 6 doesn't go cleanly.

**Verification**: hand-crafted test chromosome with one PARALLEL_DC_BRIDGE decodes to two parallel runs joined by a DC.

**Checkpoint**: DC layouts in population on a custom DC-heavy inventory config.

### Phase 7b â€” Parallel-Tracks Heuristic Seed

**Goal**: Seed with parallel-track-with-DC layouts.

**Files**: [src_v2/operators.py](src_v2/operators.py) â€” new `_emit_parallel_tracks_with_dc`

**Checkpoint**: DC-using feasible solutions in archive.

### Phase 8 â€” Topology-Aware Archive Admission

**Goal**: Îµ-archive preserves topologically rich solutions even when dominated.

**Critical correction from expert review**: my earlier draft proposed populating `out["topology_sig"]` in `_evaluate` and reading it via `ind.get("topology_sig")` from the callback. **This does not work under `StarmapParallelization`.** Custom keys written to `out` inside a worker process are not propagated back to the main process â€” pymoo's `Evaluator._eval_elementwise` only round-trips the standard keys (`F`, `G`, `dF`, `dG`, `pheno`, `feasible`). V2's `canonical.py` does NOT contradict this: `ind.set("graph_hash", h)` runs in the **main process during dedupe**, not in a worker. The mechanism must be lazy (compute in the callback's main process), not eager (compute in a worker).

**Files**:
- [src_v2/epsilon_archive.py](src_v2/epsilon_archive.py) â€” extend `EpsilonArchiveCallback.notify` with topology-aware admission rule and lazy topology-sig caching

**Lazy topology-sig pattern** (mirrors `canonical.py:228-236`):

Note on attribute access: `decode_chromosome` returns a `PortGraph` (per [decoder.py:74,119](src_v2/decoder.py:74)). `PortGraph` exposes `n_components`, `n_cycles`, `n_loose_ports`, `n_slots`, `n_edges` ([types.py:401-429](src_v2/types.py:401)) but does NOT expose `n_switch_pairs`, `n_crossings`, `n_dc_bridges`. Those have to be computed from `graph.slot_pieces` + the catalog `kind` lookup. Phase 5a's `MultiPathLayout` (returned by the V1 decoder pattern) DOES have `n_switch_pairs` ([types.py:189](src_v2/types.py:189)); if Phase 5a refactors `decode_chromosome` to return `MultiPathLayout`, the count can come from there directly. Until then, count from `slot_pieces`:

```python
def _topology_sig_for(self, ind) -> tuple:
    cached = ind.get("topology_sig")
    if cached is not None:
        return cached
    graph = decode_chromosome(ind.get("X"), self.dims, self.catalog, self.decoder_config)
    spec_by_id = self.catalog.spec.by_id

    n_switch_slots = 0   # individual IN+OUT switches, not pairs
    n_cross_90 = 0
    n_dc = 0
    for piece_id in graph.slot_pieces.values():
        ps = spec_by_id.get(piece_id)
        if ps is None:
            continue
        if ps.kind == "switch":
            n_switch_slots += 1
        elif piece_id == "CROSS_90":
            n_cross_90 += 1
        elif piece_id == "DOUBLE_CROSSOVER":
            n_dc += 1
    n_switch_pairs = n_switch_slots // 2  # IN + OUT make one passing siding

    sig = (
        n_switch_pairs,
        n_cross_90,
        n_dc,
        graph.n_components,
        graph.n_cycles,
    )
    ind.set("topology_sig", sig)
    return sig
```

**Admission rule** (in `EpsilonArchiveCallback.notify`):
```python
def notify(self, algorithm) -> None:
    # Standard feasible admission (existing behavior â€” read algorithm.opt only)
    for ind in algorithm.opt:
        if float(ind.get("CV")) <= 0.0:
            self._archive.admit(ind.get("X"), ind.get("F"))

    # Topology-aware near-feasible admission (Phase 8 NEW â€” read algorithm.pop)
    # Cap the scan to top-K-by-CV individuals to avoid 500 redundant decodes/gen
    pop = algorithm.pop
    cv_values = pop.get("CV").flatten() if pop.get("CV") is not None else np.zeros(len(pop))
    near_feasible_mask = (cv_values > 0) & (cv_values < self._cv_admission_threshold)
    near_feasible_indices = np.where(near_feasible_mask)[0]
    # Sort by CV ascending; cap at K = max_size // 4 to bound runtime
    k = min(len(near_feasible_indices), self._archive._max_size // 4)
    sorted_idx = near_feasible_indices[np.argsort(cv_values[near_feasible_indices])[:k]]

    for i in sorted_idx:
        ind = pop[i]
        sig = self._topology_sig_for(ind)  # Lazy decode, cached on ind
        if sig[0] >= 1 or sig[1] >= 1 or sig[2] >= 1:  # has switches/crossings/DC
            self._archive.admit_topology_aware(ind.get("X"), ind.get("F"), sig, cv=float(cv_values[i]))
```

`_archive.admit_topology_aware` is a new method that bypasses Îµ-non-dominance for topology-rich solutions but still respects `max_size` and prefers lower-CV entries within the topology-rich subset.

**Performance bound**: capping at `K = max_size // 4` (= 50 individuals for default max_size=200) means at most 50 extra decodes per generation in the main process. Combined with the existing `algorithm.opt` admission scan, this is bounded above by 50 + |opt| â‰ˆ 100 decodes/gen â€” within the DoD's "â‰¤ 2Ã— current V2 wall clock" budget.

**Verification**: Pareto archive on `with_switches` contains both simple stadium AND switched layouts (archive's `to_json()` output has entries with non-zero `(n_switches, n_crossings)` topology signatures).

**Risk**: medium. Wrong admission threshold could bloat archive with infeasibles. Topology-sig cache invalidation is not yet handled â€” see Golden Rule 6 in Part 7.

**Checkpoint**: final Pareto archive on `with_switches` has â‰¥ 1 solution with `n_switch_pairs >= 1`; per-generation wall clock â‰¤ 2Ã— current V2.

## Critical Files Touched

Across all phases:

- [src_v2/encoding.py](src_v2/encoding.py) â€” Phase 3 (anchor range + ENCODING_VERSION = 2), Phase 4 (junction segment scaffolding + ENCODING_VERSION = 3, int16 overflow guard, atomic update of `generate_bounds` / `validate_chromosome` / `create_empty_chromosome`)
- [src_v2/operators.py](src_v2/operators.py) â€” Phase 1 (decision on `closure_repair_lamarckian` mutation removal/refactor), Phases 2, 4, 5b, 5c, 6b, 7b (sampling + crossover semantic guard + mutation extensions)
- [src_v2/decoder.py](src_v2/decoder.py) â€” Phases 3, 4, 5a, 6a, 7a (auto-center + post-edge-decode template materialization)
- [src_v2/repair.py](src_v2/repair.py) â€” Phases 1, 4, 5a (cycle-based constructive closure + junction validity + post-edge-decode rollback)
- [src_v2/templates.py](src_v2/templates.py) â€” Phases 5a, 6a, 7a (NEW FILE â€” `@dataclass(frozen=True)` module-level port from V1)
- [src_v2/problem.py](src_v2/problem.py) â€” minimal changes; per-type inventory already correct (Phase 8 does NOT need to populate `out["topology_sig"]` â€” that is computed lazily in the callback, see Risk 11 / Rule 1)
- [src_v2/runner.py](src_v2/runner.py) â€” Phase 8 (passes `dims, catalog, decoder_config, cv_threshold` to `EpsilonArchiveCallback` constructor)
- [src_v2/types.py](src_v2/types.py) â€” add `ValidatedJunction`, `Figure8Pair`, `DCBridge` types
- [src_v2/canonical.py](src_v2/canonical.py) â€” Phase 5a (extend canonical hash to include active junction descriptors `(active, anchor_slot, template_kind, param_a, param_b)` sorted by anchor_slot; **DO NOT update in Phase 4** â€” see Risk 8 / Rule 15)
- [src_v2/epsilon_archive.py](src_v2/epsilon_archive.py) â€” Phase 3 (add `ENCODING_VERSION` field to `to_json`, add `from_json` that rejects mismatched versions, define `EncodingVersionMismatch` exception); Phase 8 (extend `EpsilonArchiveCallback` constructor to accept `dims`/`catalog`/`decoder_config`/`cv_threshold`; add `_topology_sig_for` lazy method; rewrite `notify` for topology-aware admission with bounded scan; add `EpsilonArchive.admit_topology_aware`)
- [src_v2/alns_callback.py](src_v2/alns_callback.py) â€” Phase 5c, 6, 7 (new-operator initial weight = mean of existing weights per Risk 9 / Rule 14, NOT a low sentinel like 0.05)
- [src_v2/structural_mutations.py](src_v2/structural_mutations.py) â€” Phase 1 references `introduce_crossing` as the existing 5-step edge-surgery pattern that `CycleClosureRepair` should mirror; Phase 5c junction mutations (`_mutate_junction_*`) live here OR in `operators.py` depending on file-organization preference
- [configs/*.yaml](configs/) â€” Phase 8 (introduce `cv_admission_threshold` config key); Phase 3 (optional `auto_center: true` flag if backwards-compat for ad-hoc loads is desired)
- **`configs/trains/measured_consist.yaml`** (NEW FILE â€” Phase 0) â€” measured loco + 2 cars consist; produced by physical measurement before Phase 1 starts. All thesis runs reference this via `OptimizationConfig.train_config_path`. The V2 default `coupler_offset = 0.100 m` is wrong for any real LEGO car â€” measured value is ~0.256 m for a 32-stud car. See Phase 0 in Part 8 for the measurement protocol.

## Verification Strategy (Per Phase)

After each phase, run **all three configs** (`default`, `with_switches`, `with_crossing`) for 200 generations Ã— 5 seeds. Capture:

1. **Feasibility rate** (% of final population feasible)
2. **Mean utilization** (across feasibles)
3. **Switch usage rate** (% of feasibles with â‰¥1 switch)
4. **Crossing usage rate** (% of feasibles with â‰¥1 CROSS_90 or DOUBLE_CROSSOVER)
5. **Pareto front size and topology diversity** (number of distinct topology signatures)
6. **Best-feasible piece count** per config
7. **Wall-clock per generation**

Compare each phase against the previous phase's baseline. Regression on any of (1)â€“(6) means roll back the phase or fix before moving on.

## Targets

| Metric | Current V2 | Phase 5c target | Phase 8 target |
|---|---|---|---|
| `default` best feasible | 50 / 64 (78 %) | â‰¥ 60 / 64 (94 %) | â‰¥ 60 / 64 (94 %) |
| `with_switches` best feasible | 68 / 168 (40.5 %) | â‰¥ 96 / 168 (57.1 %) | â‰¥ 144 / 168 (85.7 %) |
| `with_switches` switches used | 0 | â‰¥ 1 in best | â‰¥ 2 in best |
| `with_crossing` CROSS_90 used | 0 | n/a (Phase 6) | â‰¥ 1 in best |
| Pareto front topology diversity | 1-2 | 3+ | 5+ |

The Phase 8 `with_switches` target (144/168, â‰¥2 switches) is exactly the V1 snapshot 8-10 layout that V1 generated but couldn't close.

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Phase 4 chromosome layout change breaks existing chromosomes / saved state | Bump encoding version; runs from before Phase 4 are obsoleted (acceptable for thesis) |
| Junction materialization in decoder slows generations significantly | Profile each template's materialization cost; defer Phase 7 if DC is too expensive |
| Figure-8 / DC templates have edge cases the test suite doesn't cover | Each template gets its own test suite with hand-crafted positive + negative cases |
| Junction mutations destroy more than they tune | Phase 5c starts with conservative mutation ranges; tune via ALNS feedback |
| Archive admission rule (Phase 8) bloats archive | Cap archive size at 200; admission is OR'd with default Îµ-non-dominance check |
| Phases land out of dependency order | Strict gating: don't merge Phase N+1 until Phase N's checkpoint passes 5 seeds Ã— 3 configs |

## Estimated Scope

- Phase 1 (constructive closure repair): **1-2 days**
- Phase 2 (stadium seed): **1 day**
- Phase 3 (auto-centering): **1-2 days**
- Phase 4 (junction segment scaffolding): **2-3 days** (touches many files)
- Phase 5 (passing-siding template + seed + mutations): **3-5 days**
- Phase 6 (figure-8 template + seed): **3-5 days**
- Phase 7 (DC bridge template + seed): **5-7 days** (hardest)
- Phase 8 (archive admission): **1-2 days**

**Total**: 17-27 days of focused work. Suitable for a thesis sprint with weekly checkpoints.

## Open Questions Before Phase 1 Starts

1. **Test infrastructure**: V2's `tests/` is empty per CLAUDE.md. Do we re-create it before Phase 1, or in parallel? Recommend: write a minimal test harness for `MainLoopClosureRepair` (Phase 1) and grow tests phase-by-phase.

2. **Inventory config tuning**: do we keep current `with_switches` inventory (168 pieces, 4 LEFT switches + 4 RIGHT) or adjust? Recommend: keep, since matching V1's snapshot inventory makes comparison meaningful.

3. **Figure-8 template parameter `param_b` (handedness)**: should the secondary lobe always match main loop handedness, or be free? Recommend: free, but heuristic seeds prefer same-handedness for early feasibility.

4. **DC bridge parallel-section detection algorithm**: greedy FK-based or geometric search? Recommend: greedy FK-based, fall back to "any two slots whose poses are within tolerance" if too restrictive.

5. **Should the junction segment crossover be uniform (V1 style) or one-point**? Recommend: uniform per-slot, like V1, since each junction is logically independent.

These can be answered during Phase 4 / 5 design; not blocking for Phase 1.

## Definition of Done

The implementation is "done" when:

1. All three configs (`default`, `with_switches`, `with_crossing`) produce feasible Pareto fronts with topology diversity â‰¥ 3.
2. `with_switches` produces â‰¥ 1 final-archive solution with â‰¥ 2 switches and â‰¥ 80 % utilization.
3. `with_crossing` produces â‰¥ 1 final-archive solution with â‰¥ 1 CROSS_90 and a closed multi-cycle topology.
4. A custom DC-inventory config produces â‰¥ 1 solution with a DOUBLE_CROSSOVER.
5. No regression on `default`'s best-feasible vs current V2.
6. Wall-clock per generation â‰¤ 2Ã— current V2 (added decoder steps must not blow up runtime).
7. Tests cover each template's materialization, each junction mutation, and constructive closure repair (â‰¥ 80 % line coverage on the new modules).

---

# Part 6 â€” Pymoo 0.6.1.6 Compatibility Audit

Before any code lands, verify each phase against pymoo's actual API contracts and known limitations. Researched via `mcp__context7__query-docs` against `/anyoptimization/pymoo` and pymoo 0.6.1.6 documentation. V2's existing imports confirm which APIs are in active use.

## V2's Current Pymoo Imports (Baseline â€” All Already Working)

| Import | Used in | Compatibility |
|---|---|---|
| `from pymoo.algorithms.moo.nsga2 import NSGA2` | `runner.py:22` | âœ“ stable in 0.6.x |
| `from pymoo.constraints.eps import AdaptiveEpsilonConstraintHandling` | `runner.py:23` | âœ“ stable; V2's `LegoAdaptiveEpsilon` subclasses it |
| `from pymoo.core.callback import Callback` | 5 callback files | âœ“ stable; `notify(algorithm)` API documented |
| `from pymoo.core.crossover import Crossover` | `operators.py:21` | âœ“ stable; `_do(problem, X, **kw)` API |
| `from pymoo.core.duplicate import ElementwiseDuplicateElimination` | `canonical.py:44` | âœ“ stable; V2's `PortGraphDuplicateElimination._hash_for` at [canonical.py:228-236](src_v2/canonical.py:228) already uses `ind.get("graph_hash")` / `ind.set("graph_hash", h)` under NSGA-II â€” the `Individual.get`/`set` API is `pymoo.core.individual.Individual` core, algorithm-agnostic |
| `from pymoo.core.mutation import Mutation` | `operators.py:22` | âœ“ stable |
| `from pymoo.core.problem import ElementwiseProblem` | `problem.py:43` | âœ“ stable; `_evaluate(x, out, *args, **kw)` API |
| `from pymoo.core.repair import Repair` | `repair.py:34` | âœ“ stable; `_do(problem, X, **kw)` API |
| `from pymoo.core.sampling import Sampling` | `operators.py:23` | âœ“ stable |
| `from pymoo.operators.survival.rank_and_crowding import ConstrRankAndCrowding` | `runner.py:26` | âœ“ part of pymoo core 0.6.x (NOT pymoode, despite some docs hits) |
| `from pymoo.parallelization import StarmapParallelization` | `runner.py:25` | âœ“ stable; works with `multiprocessing.Pool` |
| `from pymoo.optimize import minimize` | `runner.py:27` | âœ“ stable |
| `from pymoo.termination import get_termination` | `runner.py:28` | âœ“ stable |

**Verdict**: every pymoo API V2 already uses is stable in 0.6.1.6. None of the phases require new imports beyond what's already proven working.

## Per-Phase Compatibility Verification

### Phase 1: Constructive Closure Repair

**Required pymoo contract**:
- `Repair._do(problem, X, **kwargs) -> X` â€” modify chromosomes in place, return X
- Multi-stage pipeline supported (V2 already chains sanitize â†’ inventory â†’ orientation)
- Pipeline can iterate to fixed point (V2 uses `max_iterations=5`)

**API check** (Context7 docs):
> *"Defines a custom Repair operator 'MyRepair' that enforces an equality constraint by modifying the solution. ... `_do(self, problem, X, **kwargs): X[:, 0] = 1/3 * X[:, 1]; return X`"*

**Compatibility**: âœ“ direct-modify pattern is the canonical pymoo idiom. Adding a new pipeline stage (`MainLoopClosureRepair`) that activates inactive slots and adds port-pair edges is exactly within the contract.

**Gotcha**: must terminate. V1 caps at `max_corrections=4` per pass; V2's pipeline already has `max_iterations=5`. Compose them; total bounded.

**Phase 1: âœ“ COMPATIBLE.**

### Phase 2: Sequential-Ring Stadium Heuristic Seed

**Required pymoo contract**:
- `Sampling._do(problem, n_samples, **kwargs) -> X` of shape `(n_samples, problem.n_var)`
- Custom emitter functions producing Pattern tuples â†’ fed into existing `_populate_from_pattern`

**API check**: V2's `PortPairSampling._do` ([operators.py:623](src_v2/operators.py:623)) already uses `_HEURISTIC_EMITTERS` list. Adding one emitter is additive.

**Compatibility**: âœ“ matches V2's existing pattern.

**Phase 2: âœ“ COMPATIBLE.**

### Phase 3: Auto-Centering + Anchor Range Shrink

**Required pymoo contract**:
- `xl/xu` arrays of length `n_var` define per-gene bounds. Modifying bounds for the anchor genes is a `generate_bounds` change.
- Decoder is internal to V2's `Problem._evaluate`; pymoo doesn't constrain it.

**API check**: pymoo allows per-gene bounds via `xl=np.array([...])`, `xu=np.array([...])`. âœ“ already used in V2.

**Compatibility**: âœ“ pure decoder/encoding refactor.

**Gotcha**: existing saved chromosomes have anchor genes in full-boundary range. After Phase 3, those values are out of bounds. Must regenerate populations or clamp on load.

**Phase 3: âœ“ COMPATIBLE** with caveat that old saved chromosomes are invalidated.

### Phase 4: Junction Segment Scaffolding

**Required pymoo contract**:
- `Problem.n_var` is fixed PER INSTANCE. Different problem instances can have different `n_var`. âœ“
- Chromosome shape `(n_var,)` per individual; `(n_samples, n_var)` per population. Must be consistent across operators.
- `xl/xu` arrays of length `n_var` extend naturally.

**API check** (Context7):
> *"super().__init__(n_var=10, n_obj=1, n_ieq_constr=2, xl=xl, xu=xu)"*

V2's `compute_port_pair_dimensions` already returns dynamic `n_var`. Extending it to include junction segment is a well-trodden pattern.

**Compatibility**: âœ“ pymoo doesn't care what's IN the chromosome; only that operators preserve shape. All custom operators (PortPairCrossover, PortPairMutation, PortPairRepairPipeline, PortPairSampling) need to be updated to handle the new segment, but the pymoo contract is stable.

**Gotchas**:
- **Crossover must not produce shape-mismatched offspring.** `_do` returns `(n_offsprings, n_matings, n_var)`. Adding junction segment crossover adds genes; output shape must equal input shape. V1's PartitionedCrossover does per-slot uniform swap on junctions â€” no shape change. Port directly **with the semantic guard** described in Phase 4 of Part 5 (don't swap when anchor_slot is invalid in the receiver chromosome).
- **`PortGraphDuplicateElimination` canonical hash update is deferred to Phase 5a.** In Phase 4 the decoder ignores junctions, so two chromosomes with identical port-pair edges but different junction genes still produce identical layouts; hashing junctions in Phase 4 would treat phenotypic duplicates as distinct, wasting diversity budget. Update the canonical form when junctions actually affect decoded output (Phase 5a). [Earlier draft of this gotcha had the timing wrong â€” corrected per expert review.]
- **`eliminate_duplicates=False`** is also a valid option (V2 has both paths). Phase 4 doesn't break either.

**Phase 4: âœ“ COMPATIBLE** with required updates to: `encoding.py` (compute n_var with int16 overflow guard, generate_bounds, validate_chromosome, segment helpers, ENCODING_VERSION bump to 3), `operators.py` (Crossover with semantic guard, Mutation skipping junction segment, Sampling extended), `repair.py` (clamp junction genes + deactivate junctions with invalid anchor context), `decoder.py` (read but don't materialize until Phase 5a). **Do NOT update `canonical.py` in Phase 4** â€” defer to Phase 5a.

### Phase 5a: Template Decoder Machinery (PASSING_SIDING)

**Required pymoo contract**:
- `Problem._evaluate(x, out, *args, **kwargs)` populates `out["F"]`, `out["G"]`. Custom keys CAN be set in `out`, BUT see critical caveat below regarding multiprocessing.
- Decoder is internal to `_evaluate`; pymoo treats it as a black box.

**Critical correction from expert review (REPLACES the earlier draft of this section)**:

My earlier draft claimed `out["topology_sig"]` set in `_evaluate` could be retrieved via `ind.get("topology_sig")` in a callback, citing V2's `canonical.py:228-236` as proof. **The analogy is wrong.** `canonical.py`'s `ind.set("graph_hash", h)` runs in the **main process during dedupe**, not in a worker. Custom keys written to the `out` dict inside a `multiprocessing.Pool` worker (via `StarmapParallelization`) are **not propagated back** to the main process â€” pymoo's `Evaluator._eval_elementwise` only round-trips the standard keys (`F`, `G`, `dF`, `dG`, `pheno`, `feasible`).

Therefore:
- âœ“ **Lazy pattern (works)**: compute on first access from the main-process side, cache via `ind.set()`. V2 already uses this for `graph_hash`.
- âœ— **Eager pattern (does NOT work under StarmapParallelization)**: populate custom keys in `_evaluate` and read from a callback. Silently returns `None`.

For Phase 8's `topology_sig`, use the lazy pattern: callback decodes from `ind.get("X")` and caches via `ind.set("topology_sig", sig)`. See Phase 8 in Part 5 for the corrected algorithm.

For Phase 5a itself, the decoder runs inside `_evaluate` and populates `out["F"]`, `out["G"]` as usual; no custom out-keys are required for Phase 5a's correctness. Junction materialization rollback happens internally during decode and doesn't need to escape the worker process.

**Compatibility**: âœ“ pymoo doesn't impose decoder structure. Template materialization with rollback is internal `_evaluate` logic.

**Gotcha**: parallelization. V2 uses `StarmapParallelization` with `multiprocessing.Pool`. Templates module must be importable cleanly with **`@dataclass(frozen=True)` module-level constants** (V1's pattern) so each worker pickles consistent templates. Static module-level data is fine; mutable state is not. Lazy template registries are not â€” define all templates at import time.

**Phase 5a: âœ“ COMPATIBLE** assuming templates are pure module-level frozen dataclasses (no lazy construction, no per-instance state, no `np.random` calls inside template materialization).

### Phase 5b: Asymmetric-Oval Siding Heuristic Seed

Same as Phase 2 â€” additive emitter. âœ“ COMPATIBLE.

### Phase 5c: Junction Mutations

**Required pymoo contract**:
- `Mutation._do(problem, X, **kwargs) -> X` â€” sub-operator dispatch is just Python control flow inside `_do`
- ALNS callback adapts weights â€” V2 already does this

**API check** (Context7):
> *"for i in range(len(X)): r = np.random.random(); if r < 0.4: ... elif r < 0.8: ..."*

Sub-operator dispatch with weighted random selection is the documented pattern.

**Compatibility**: âœ“ direct extension of V2's `PortPairMutation`.

**Gotcha**: ALNS cold-start. New sub-ops start with uniform weights; first 10-20 generations adapt. Don't expect immediate impact.

**Phase 5c: âœ“ COMPATIBLE.**

### Phase 6a, 6b: FIGURE_8_CROSS Template + Seed

Same compatibility as Phase 5a, 5b. New template kind in `template_kind` gene; new materialization branch in decoder. âœ“ COMPATIBLE.

**Risk-only note**: lobe closure validation is more complex than passing-siding (two cycles). Test extensively before Phase 6b builds on it.

### Phase 7a, 7b: PARALLEL_DC_BRIDGE Template + Seed

Same compatibility as 5a, 6a. âœ“ COMPATIBLE in API contract.

**Risk-only note**: parallel-section detection algorithm is open. Falls back to opportunistic placement if exact parallel detection fails. Defer if Phase 6 doesn't go cleanly.

### Phase 8: Topology-Aware Archive Admission

**Required pymoo contract**:
- `Callback.notify(algorithm)` â€” called per generation; `algorithm.pop`, `algorithm.opt`, `algorithm.n_gen`, `algorithm.evaluator.n_eval` accessible.
- `ind.get("CV")`, `ind.get("F")`, `ind.get("X")` â€” and crucially `ind.get("topology_sig")` if `_evaluate` populated `out["topology_sig"]`.

**API check** (Context7 callback docs):
> *"`algorithm.pop` can be used to access the population, with methods like `get('F')` to retrieve objective values"*
> *"Accessing `algorithm.n_gen` for the current generation number"*

V2's `EpsilonArchiveCallback.notify` already iterates `algorithm.opt` per gen and admits feasibles. Phase 8 extends this to also admit "topologically rich + CV < 1.0" individuals from `algorithm.pop` (broader than `opt`).

**Compatibility**: âœ“ extension via lazy main-process decode + `ind.set` cache (NOT eager `out["topology_sig"]` from worker â€” see Risk 11 below).

**Implementation steps (corrected)**:
1. `epsilon_archive.py:EpsilonArchiveCallback.notify` runs in the main process. For each candidate individual it reads `ind.get("X")`, decodes via the existing `decode_chromosome` (same code path `canonical.py:228-236` uses), and computes `topology_sig` from the decoded `PortGraph`. Caches via `ind.set("topology_sig", sig)`.
2. Admission rule: standard feasible admission via `algorithm.opt` (unchanged); topology-aware near-feasible admission via top-K of `algorithm.pop` (`K = max_size // 4`) sorted by ascending CV â€” bounded scan, not full-pop iteration.

**Gotchas**:
- Reading the entire `algorithm.pop` per generation (1000 individuals per `with_switches.yaml:pop_size: 1000`) and decoding all of them is too expensive. Cap at top-K-by-CV.
- Lazy cache invalidation: a chromosome that was mutated since last admission may have a stale `topology_sig`. Invalidate via Golden Rule 6 in Part 7 (clear cached metadata when `ind.X` changes).
- Don't mutate `algorithm.pop` from the callback. Read-only is safe; modifying is risky.

**Phase 8: âœ“ COMPATIBLE** with the lazy-cache pattern; would NOT be compatible with the eager `out["topology_sig"]` pattern from the earlier draft.

## Cross-Phase Risks

### Risk 1: NumPy Version

- **Issue** ([github #606](https://github.com/anyoptimization/pymoo/issues/606)): NumPy 2.0.0 broke pymoo 0.6.1.1 with AttributeError on ElementwiseProblem import. Fixed in 0.6.1.3+.
- **Project status**: pymoo == 0.6.1.6 (per [CLAUDE.md](CLAUDE.md), `requirements.txt`). Safe.

### Risk 2: Module Organization

- **Issue**: pymoo 0.5.0 reorganized modules; 0.6.0 deprecated factory methods. Must use direct class imports.
- **Project status**: V2 already uses direct imports (verified in `runner.py`). Safe.

### Risk 3: Decoder Determinism

- pymoo assumes `_evaluate` is a pure function: same chromosome â†’ same F, G. Random behavior inside the decoder breaks this and corrupts NSGA-II selection.
- V1 / V2 decoders are deterministic. Templates must follow the same rule â€” no `np.random` calls inside decoder; all randomness must come from the chromosome itself.
- **Templates must be deterministic given (template_kind, param_a, param_b)**.

### Risk 4: Parallelization Side Effects

- `StarmapParallelization` uses `multiprocessing.Pool` â€” each worker gets a pickled copy of the problem. Module-level mutable state (e.g., a template registry that grows) doesn't propagate.
- **Templates must be defined at module-import time, not lazily**. Direct port from V1 satisfies this.

### Risk 5: Chromosome Layout Migration

- Phase 4 changes `n_var`. Old saved populations become incompatible. No automatic migration.
- **Mitigation**: bump an `encoding_version` constant; refuse to load populations from older versions. Acceptable for thesis.

### Risk 6: Junction Crossover Producing Invalid Descriptors

- One-point or uniform-per-slot crossover can mix junction descriptors from two parents with different anchor positions, producing junctions whose anchor_slot points to a now-INACTIVE main loop slot in the receiving chromosome.
- **Mitigation (corrected)**: BOTH a swap-time semantic guard (Phase 4 algorithm in Part 5) AND a `JunctionValidityRepair` post-pass. The guard prevents the bad descriptor from being inherited blindly; the repair handles edge cases (e.g., when subsequent mutations later invalidate the anchor). "Clamp to valid slot" alone is insufficient because it teleports the junction to an unrelated location, destroying heritability â€” the repair should prefer deactivation OR nearest-active-slot relocation, not random clamping (Golden Rule 9 in Part 7).

### Risk 7: Pymoo's Default Selection Compatibility

- pymoo NSGA2 uses binary tournament with crowded comparison by default. Custom Survival (`ConstrRankAndCrowding`) and custom Repair don't affect Selection.
- All phases are compatible with default Selection. âœ“

### Risk 8: Eliminate_duplicates Granularity (TIMING CORRECTED)

- `PortGraphDuplicateElimination` compares decoded port graphs. **In Phase 4**, the decoder ignores junctions, so two chromosomes with identical port-pair edges but different junction descriptors produce identical layouts â€” they SHOULD be treated as duplicates. Hashing junctions in Phase 4 corrupts dedupe.
- **In Phase 5a**, the decoder materializes junctions into the layout, so the canonical form must reflect them.
- **Mitigation**: extend `canonical.py` canonical form to include `(active, anchor_slot, template_kind, param_a, param_b)` tuples for active junctions, sorted by anchor_slot â€” **but defer this update to Phase 5a, not Phase 4**. (Earlier draft had the timing wrong; corrected per expert review.)

### Risk 9: ALNS Weight Adaptation Time (MITIGATION CORRECTED)

- New sub-operators added in Phases 5c, 6, 7 need to be sampled enough times early-on to accumulate fitness signal. With â‰¥21 operators in the pool and offspring counts of 200-500 per generation, an operator sampled at uniform 1/21 weight gets ~10-25 trials/gen â€” barely enough.
- **Mitigation (corrected)**: initial weight = mean of existing operator weights (so new ops compete on equal footing from gen 1). The earlier draft suggested `initial_weight = 0.05` which makes the problem worse â€” lower weight means fewer samples means slower learning. Set new-op initial weight to `np.mean(current_weights)`. (Golden Rule 14 in Part 7.)

### Risk 10: Pymoo Version Drift

- Project pins `pymoo==0.6.1.6` per `requirements.txt`. Future updates may change APIs.
- **Mitigation**: implementation should not rely on undocumented internals. All extension points used (Repair, Mutation, Sampling, Crossover, Callback, ElementwiseDuplicateElimination, ConstrRankAndCrowding, AdaptiveEpsilonConstraintHandling) are public API.

### Risk 11: Custom `out` keys NOT propagated under StarmapParallelization (NEW)

- Pymoo's `Evaluator._eval_elementwise` only round-trips standard `out` keys (`F`, `G`, `dF`, `dG`, `pheno`, `feasible`) from worker processes back to the main process. Custom keys written to `out` inside a `multiprocessing.Pool` worker silently disappear.
- The earlier plan proposed `out["topology_sig"]` set in `_evaluate` and read via `ind.get("topology_sig")` in a callback. **This does not work** under V2's `StarmapParallelization`. V2's `canonical.py:228-236` works for `graph_hash` only because `ind.set()` runs in the main-process side of dedupe, not in a worker.
- **Mitigation**: use the lazy pattern â€” compute custom metadata on first access in the main-process side (callback / dedupe / survival) and cache via `ind.set()`. NEVER rely on custom `out` keys round-tripping from workers. (Golden Rule 1 in Part 7.)

## Specific API Patterns to Use (Copy-Pasteable, all corrected post-review)

### Lazy topology signature in callback (CORRECTED â€” does not use `out["topology_sig"]`)

```python
# In epsilon_archive.py:EpsilonArchiveCallback (V2's existing class â€” extended):

class EpsilonArchiveCallback(Callback):
    def __init__(self, ..., dims, catalog, decoder_config, cv_threshold=1.0):
        super().__init__()
        # ... existing init ...
        self._dims = dims
        self._catalog = catalog
        self._decoder_config = decoder_config
        self._cv_threshold = cv_threshold

    def _topology_sig_for(self, ind) -> tuple:
        """Lazy decode + cache, mirroring canonical.py:228-236.

        PortGraph exposes n_components / n_cycles / n_loose_ports natively
        (types.py:401-429); switch-pair / CROSS_90 / DC counts are derived
        from slot_pieces + catalog spec lookups, since those properties
        are NOT on PortGraph itself.
        """
        cached = ind.get("topology_sig")
        if cached is not None:
            return cached
        graph = decode_chromosome(ind.get("X"), self._dims, self._catalog, self._decoder_config)
        spec_by_id = self._catalog.spec.by_id
        n_switch_slots = n_cross_90 = n_dc = 0
        for piece_id in graph.slot_pieces.values():
            ps = spec_by_id.get(piece_id)
            if ps is None:
                continue
            if ps.kind == "switch":
                n_switch_slots += 1
            elif piece_id == "CROSS_90":
                n_cross_90 += 1
            elif piece_id == "DOUBLE_CROSSOVER":
                n_dc += 1
        sig = (
            n_switch_slots // 2,  # IN+OUT pairs = one passing siding
            n_cross_90,
            n_dc,
            graph.n_components,
            graph.n_cycles,
        )
        ind.set("topology_sig", sig)
        return sig

    def notify(self, algorithm) -> None:
        # Standard feasible admission (existing)
        for ind in algorithm.opt:
            if float(ind.get("CV")) <= 0.0:
                self._archive.admit(ind.get("X"), ind.get("F"))
        # Topology-aware near-feasible admission (Phase 8 NEW, bounded scan)
        pop = algorithm.pop
        cv = pop.get("CV").flatten()
        near_mask = (cv > 0) & (cv < self._cv_threshold)
        idxs = np.where(near_mask)[0]
        k = min(len(idxs), self._archive._max_size // 4)
        sorted_idxs = idxs[np.argsort(cv[idxs])[:k]]
        for i in sorted_idxs:
            ind = pop[i]
            sig = self._topology_sig_for(ind)
            if sig[0] >= 1 or sig[1] >= 1 or sig[2] >= 1:
                self._archive.admit_topology_aware(ind.get("X"), ind.get("F"), sig, cv=float(cv[i]))
```

### Junction segment in encoding.py (with int16 overflow guard)

```python
@dataclass(frozen=True)
class PortPairDimensions:
    N_max: int
    E_max: int
    J_max: int  # NEW
    # ... other fields ...

    def __post_init__(self):
        # Golden Rule 2: catch int16 overflow at construction time
        if self.n_var >= np.iinfo(np.int16).max:
            raise ValueError(
                f"PortPairDimensions.n_var={self.n_var} exceeds int16 max "
                f"({np.iinfo(np.int16).max}); inventory too large for current encoding."
            )

    @property
    def junc_start(self) -> int:
        return self.pair_end
    @property
    def junc_end(self) -> int:
        return self.junc_start + self.J_max * 5  # 5 genes per junction
    @property
    def anchor_start(self) -> int:
        return self.junc_end  # CHANGED: was pair_end
    @property
    def n_var(self) -> int:
        return self.anchor_start + 3

ENCODING_VERSION: int = 3  # bumped at Phase 4 (was 2 at Phase 3)
```

### Junction crossover (semantically guarded, NOT blind per-slot uniform)

```python
# In PortPairCrossover._do, after pair region crossover:
for j in range(dims.J_max):
    j1_active, j1_anchor, j1_kind, j1_a, j1_b = read_junction(p1, dims, j)
    j2_active, j2_anchor, j2_kind, j2_a, j2_b = read_junction(p2, dims, j)

    # Guard: each side's anchor_slot must be branch-capable in the receiver's slot context
    p1_can_receive_p2 = (j2_anchor < dims.N_max) and is_branch_capable(c1, dims, j2_anchor)
    p2_can_receive_p1 = (j1_anchor < dims.N_max) and is_branch_capable(c2, dims, j1_anchor)

    if np.random.random() >= 0.5:
        continue  # no swap chosen
    if p1_can_receive_p2 and p2_can_receive_p1:
        # Atomic swap when both anchors land in valid context
        base = dims.junc_start + j * 5
        end = base + 5
        c1[base:end], c2[base:end] = p2[base:end].copy(), p1[base:end].copy()
    else:
        # Deactivate the would-be invalid junction in the receiver, don't propagate broken descriptor
        if not p1_can_receive_p2 and j2_active:
            write_junction(c1, dims, j, active=0, anchor=j2_anchor, kind=j2_kind, param_a=j2_a, param_b=j2_b)
        if not p2_can_receive_p1 and j1_active:
            write_junction(c2, dims, j, active=0, anchor=j1_anchor, kind=j1_kind, param_a=j1_a, param_b=j1_b)
```

### Template materialization in decoder (rollback pattern, AFTER edge decode)

```python
# In decoder.py, AFTER port-pair edges have been decoded into PortGraph
# and after _connected_components / nx.cycle_basis have run:

def _materialize_junctions(graph: PortGraph, junctions, tracker, dims, decoder_cfg):
    # Sort by anchor_slot for deterministic order under multiprocessing
    active = sorted(
        (j for j in junctions if j.active),
        key=lambda j: j.anchor_slot,
    )
    for j in active:
        # 1. Anchor must be an active vertex
        if j.anchor_slot not in graph.active_slots:
            continue
        # 2. Anchor must be in a cycle
        cycle = graph.cycle_containing(j.anchor_slot)
        if cycle is None:
            continue
        template = TEMPLATES[j.template_kind]
        # 3. Inventory check (no claim yet)
        reqs = template.inventory_requirements(j.param_a, j.param_b)
        if not tracker.can_use_batch(reqs):
            continue
        # 4. Find OUT position by walking cycle (V1's _find_out_position adapted)
        in_state = graph.fk_state_at(j.anchor_slot)
        out_state, out_slot = template.find_out_position(graph, cycle, j.anchor_slot, j.param_a)
        if out_slot is None:
            continue
        # 5. Closure validation
        if not template.is_valid(in_state, out_state, j.param_a, j.param_b):
            continue
        # 6. Commit: claim inventory, splice template into graph
        tracker.use_batch(reqs)
        template.materialize_into(graph, j.anchor_slot, out_slot, j.param_a, j.param_b)
```

### Îµ-archive JSON versioning

```python
# epsilon_archive.py:to_json (additive)
def to_json(self, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "encoding_version": ENCODING_VERSION,  # NEW
        "epsilon": self._eps.tolist(),
        "max_size": self._max_size,
        "F": self._F.tolist(),
        "X": [np.asarray(x).astype(int).tolist() for x in self._X],
    }, indent=2))

# epsilon_archive.py:from_json (NEW)
def from_json(path: Path) -> EpsilonArchive:
    payload = json.loads(Path(path).read_text())
    if payload.get("encoding_version") != ENCODING_VERSION:
        raise EncodingVersionMismatch(
            f"archive at {path} has encoding_version={payload.get('encoding_version')!r}, "
            f"current is {ENCODING_VERSION}; archive cannot be safely loaded"
        )
    archive = EpsilonArchive(tuple(payload["epsilon"]), max_size=payload["max_size"])
    archive._F = np.array(payload["F"], dtype=np.float64)
    archive._X = [np.array(x, dtype=np.int16) for x in payload["X"]]
    return archive
```
```

## Summary: Every Phase Is Pymoo-Compatible

All eight phases use only pymoo public APIs that V2 already exercises in production. No new pymoo features required. No version-specific workarounds beyond what V2 already does (NumPy 2.0 already mitigated by 0.6.1.6 pin).

The compatibility-blocking risks are entirely about chromosome layout migration (Phase 4 obsoletes saved populations) and deterministic decoder requirements (templates must not use `np.random`). Both are well-understood and have documented mitigations.

**No phase requires rewriting how V2 talks to pymoo.** All extensions slot into the existing extension points (Repair, Mutation, Sampling, Crossover, Callback, Survival).

---

# Part 7 â€” Golden Python Rules to Follow During Implementation

These rules came out of the python-pymoo-reviewer expert review of the plan. Each is one actionable sentence + one-line rationale citing where it applies. Read this list before starting any phase, and check it again before merging each phase.

## Rule 1 â€” Custom `out` keys do not survive `StarmapParallelization`

Never write a custom key to `out` in `_evaluate` and try to read it from a Callback under `StarmapParallelization` â€” the worker's `out` dict is not propagated; pymoo only round-trips standard keys (`F`, `G`, `dF`, `dG`, `pheno`, `feasible`). For Phase 8 `topology_sig`, compute lazily in the main-process callback from `ind.get("X")` and cache via `ind.set()`, mirroring V2's `canonical.py:228-236` pattern for `graph_hash`.

## Rule 2 â€” Guard int16 overflow at PortPairDimensions construction

When extending `PortPairDimensions` with a `J_max` field, assert `self.n_var < np.iinfo(np.int16).max` in `__post_init__`. Phase 4 adds up to `J_max * 5` new genes; for very large inventories this can push `n_var` toward int16's 32767 ceiling, where chromosome values silently truncate.

## Rule 3 â€” Decoder and templates must be pure functions of the chromosome

Never call `np.random` or any RNG inside the decoder or templates â€” pymoo NSGA-II selection assumes `_evaluate` is deterministic. Two evaluations of the same chromosome must produce the same `F` and `G`. Templates' geometric materialization is deterministic given `(template_kind, param_a, param_b)`; enforce this with a comment at the decoder entry point and a regression test that runs `_evaluate` twice on the same chromosome and asserts identical output.

## Rule 4 â€” Junction crossover swaps require a semantic guard, not blind uniform exchange

When implementing junction crossover, only swap junction descriptor `j` if both parents have an active branch-capable piece at `anchor_slot` in the receiver chromosome's slot context; otherwise deactivate the junction in the receiver rather than swapping blindly. Blind swaps produce junctions whose anchor refers to incompatible slots â€” `JunctionValidityRepair` then deactivates them silently, and that is V2's existing crossover-destroys-feasibility failure mode (memory note: 1/500 feasible at p=0.9) replicated for the junction segment.

## Rule 5 â€” Closure repair operates on cycles, not flat angle sums

In the V2 `CycleClosureRepair`, operate on the largest cycle of the decoded `PortGraph` (use `nx.cycle_basis` already in `decoder.py:692`), not on a flat angle sum across all active slots. Active slots not part of any cycle contribute angle budget that will never close. The 5-step edge surgery (remove edge â†’ activate slot with R40 â†’ add two new edges through new slot's ports â†’ verify port-uniqueness) is the operation V2's current operators don't do atomically; that is what Phase 1 must implement.

## Rule 6 â€” Invalidate cached metadata when `ind.X` mutates

When caching decoded results on `Individual` objects (V2 does this with `graph_hash`, Phase 8 will add `topology_sig`), always check for stale caches when the chromosome has been mutated â€” pymoo does not invalidate `ind.data` when `ind.X` changes. A mutated individual with a parent-cached hash can falsely pass dup-elim. Mitigate by either (a) clearing all cached metadata in `PortPairMutation._do` after mutating, or (b) keying the cache by `id(ind.X.data.tobytes())` so a different array invalidates automatically.

## Rule 7 â€” Lazy-cache the topology signature; bound the population scan

When the Phase 8 archive iterates `algorithm.pop` for topology-aware admission, cap the scan at `K = max_size // 4` individuals (sorted ascending by CV) and cache the decoded topology tuple via `ind.set("topology_sig", sig)`. Decoding all 500 population members per generation in the serial main-process path will exceed the DoD's "â‰¤2Ã— current V2 wall clock" budget.

## Rule 8 â€” Update `generate_bounds`, `validate_chromosome`, AND `create_empty_chromosome` together

Whenever you change `PortPairDimensions` (Phase 4 most importantly), update `generate_bounds`, `validate_chromosome`, and `create_empty_chromosome` atomically â€” pymoo's polynomial mutation (if ever used) and the existing repair pipeline will operate with stale bounds for the new segment, producing out-of-range genes that silently corrupt chromosomes. Phase 4's file list explicitly includes `validate_chromosome` for this reason.

## Rule 9 â€” Junction repair: prefer nearest valid slot over random clamping

In `JunctionValidityRepair`, when an `anchor_slot` value points to an INACTIVE or non-branch-capable slot, prefer (a) deactivating the junction OR (b) relocating to the nearest active branch-capable slot â€” never clamp to a random valid slot. Random clamping teleports junctions arbitrarily and destroys any heritability the crossover preserved.

## Rule 10 â€” Use V1's two-corners-of-8 stadium pattern, not four-corners-of-4

Phase 2's `_emit_sequential_ring_stadium` must use `[R40 Ã— 8, STR Ã— m, R40 Ã— 8, STR Ã— m]` with `inv.get(curve_idx, 0) >= 16` guard, not `[R40 Ã— 4, STR Ã— m] Ã— 4` which would consume 16 R40s on corners alone and leave no straights for body. Direct port from V1 `sampling.py:_gen_oval` (line ~146).

## Rule 11 â€” Templates as `@dataclass(frozen=True)` module-level constants

Define every template (`PassingSidingTemplate`, `Figure8Template`, `ParallelBridgeTemplate`) as a `@dataclass(frozen=True)` module-level constant, never as a class attribute or lazily-constructed object. `multiprocessing.Pool` workers pickle the problem on startup; lazy registries don't propagate, frozen dataclasses do. V1's `templates.py` is the reference implementation.

## Rule 12 â€” Junction materialization runs AFTER port-pair edge decode, not before

Phase 5a's `_materialize_junctions` step in the decoder must run AFTER port-pair edges have been decoded into a `PortGraph` and AFTER `_connected_components`/`nx.cycle_basis` have identified cycles â€” there is no main loop to walk before edge decode. The plan's earlier draft had this order reversed; do not regress to it during implementation.

## Rule 13 â€” Stamp `ENCODING_VERSION` in every persisted artifact

Define `ENCODING_VERSION: int` in `encoding.py` and write it into every persisted file (Îµ-archive JSON, run results, saved populations); refuse to load a file whose `encoding_version` does not match the current constant. The Îµ-archive currently has no version stamp ([epsilon_archive.py:104-113](src_v2/epsilon_archive.py:104)) â€” Phase 3's anchor-range change would silently corrupt loaded pre-Phase-3 archives if `from_json` is implemented without the version check.

## Rule 14 â€” ALNS new-operator initial weight = mean of existing weights

When adding new ALNS sub-operators in Phases 5c, 6, 7, set their initial weight to `np.mean(current_weights)`, not a low sentinel like `0.05`. Lower initial weight starves the operator of trials in early generations and prevents ALNS from ever learning that it's effective. Equal-mean initial weight lets ALNS adapt up or down based on actual fitness signal from generation 1.

## Rule 15 â€” Defer canonical-hash junction inclusion until junctions affect decoded output (Phase 5a, not Phase 4)

Update `canonical.py`'s canonical form to include junction descriptors only when the decoder actually materializes them (Phase 5a). Updating the hash in Phase 4 â€” when the decoder ignores junctions â€” causes phenotypic duplicates (chromosomes that decode to identical layouts) to be treated as distinct, wasting population diversity budget.

## Rule 16 â€” Profile decoder cost before merging Phase 5a

V2's per-generation wall clock is dominated by `_evaluate` with `decode_chromosome`. Phase 5a adds `_materialize_junctions` to that hot path. Before merging, profile a 200-gen run with a cprofile sampler and confirm the added cost is < 25 % of baseline `_evaluate` time. If higher, optimize the cycle-search step in `find_out_position` (currently O(cycle_length) per junction; can be cached per cycle).

## Rule 17 â€” All IPC-crossing data must be pickle-safe

Anything that lives in `Problem` or its operators is pickled to multiprocessing workers. `TrackCatalog`, `OptimizationConfig`, and templates must all be pickle-safe (no lambdas, no local closures, no logger references). V2's existing `runner.py:213` already uses this pattern; Phase 5a templates module must respect it. Check by: `import pickle; pickle.dumps(problem)` before any optimizer run starts.

## Rule 18 â€” Use `numpy.typing.NDArray` and explicit dtypes everywhere

Chromosome operations are `int16`; pose computations are `float64`. Mixing types silently downcasts on assignment â€” V2 already uses `np.full((n_samples, n_var), INACTIVE, dtype=DTYPE)` consistently. New code in Phase 4+ must follow the same pattern. Use `from numpy.typing import NDArray` for type hints (V2's convention) â€” `NDArray` is parametrized by dtype in NumPy 1.21+ and improves IDE support.

## Rule 19 â€” Test multiprocessing-end-to-end at every phase merge

Phase boundaries are integration points. After each phase merge, run with `n_workers >= 2` (config flag) for at least 10 generations and confirm: (a) no pickling errors, (b) deterministic results given the same seed, (c) per-generation wall clock comparable to the single-worker run. Pickling failures in templates and side effects in decoder caches are the two most common multiprocessing bugs in this codebase pattern.

## Rule 20 â€” Run a regression test suite per phase, not per change

Each phase's checkpoint specifies a verification (e.g. "â‰¥1 switch in best-feasible on `with_switches`"). Implement these as pytest cases that load a known-feasible chromosome, run the relevant operator/decoder/repair, and assert the property. Add to `tests/` per phase. Do not move to phase N+1 until phase N's regression test set is green on 5 seeds Ã— 3 configs.

---

These rules are living guidance â€” if implementation surfaces additional gotchas, append them here with `Rule N` numbering. The expert review found 5 critical issues that became Rules 1, 5, 10, 12, and 14; the remainder are preventive measures derived from V2's existing patterns + pymoo 0.6.1.6 idioms.

---

# Part 8 â€” Train & Cars Physics Model Audit (For The Implementation Plan)

The user asked whether V2's current train physics model is sufficient for the planned implementation (Phases 1â€“8) or needs improvement. Below is the audit.

## What V2 has today

Read [src_v2/train/physics.py](src_v2/train/physics.py) (173 lines) and [src_v2/train/scoring.py](src_v2/train/scoring.py) (167 lines). The model is split into:

### Per-piece speed caps (`physics.py`)

`TrainConfig` is a frozen dataclass with **15 physical parameters**:

| Group | Field | Default | Meaning |
|---|---|---|---|
| Friction | `mu_nominal` | 0.30 | Reference friction coefficient (diagnostics only) |
| | `mu_design` | 0.25 | **Pessimistic** friction used by all speed-cap formulas |
| Environment | `g` | 9.81 m/sÂ² | Gravity |
| Motor | `v_motor_max` | 1.10 m/s | Powered Up motor cap |
| Bogie geometry | `gauge_b` | 0.0375 m | Inner rail gauge |
| | `cog_height_h` | 0.030 m | CoG above rail head |
| | `flange_angle_deg` | 50Â° | Effective flange contact angle |
| Profile dynamics | `max_accel` | 3.92 m/sÂ² | Forward accel limit |
| | `brake_decel` | 2.45 m/sÂ² | Brake decel limit |
| Consist | `mass_loco` | 0.370 kg | Locomotive mass |
| | `mass_trailing` | **0.0 kg** | Total trailing-vehicle mass (bare loco by default) |
| | `coupler_offset` | 0.100 m | Coupler-angle calc length |

Three derailment-mode caps (lateral stability):
- `v_slide(R) = sqrt(mu_design Â· g Â· R)` â€” lateral sliding limit (the binding cap on R40 at 0.886 m/s)
- `v_tip(R) = sqrt(g Â· R Â· b/2 / h)` â€” tip-over from CoG height (= 1.40 m/s on R40)
- `v_nadal(R) = sqrt(g Â· R Â· (tan(d) âˆ’ Î¼) / (1 + Î¼Â·tan(d)))` â€” Nadal wheel-climb criterion (= 1.51 m/s on R40)
- `v_max(R) = min(slide, tip, nadal)` â€” derailment-mode minimum
- `v_eff(R) = min(v_max, v_motor_max)` â€” effective cap including motor

`available_accel(v, R)` implements a **friction ellipse** (Kapania-Gerdes 2015 pattern): combined lateral + longitudinal grip, with a one-iteration coupler-correction adding lateral force from trailing wagons during acceleration.

### Speed profile (`scoring.py`)

`compute_speed_profile(layout, catalog, train_config)` â†’ `SpeedProfile` (speeds per segment, avg_speed, lap_time, total_distance, max_speed, **min_speed**).

Three-pass algorithm:
1. **Pass 1** â€” per-piece curvature limit `min(v_eff(R_i), catalog.speed_table[piece_i])`
2. **Pass 2** â€” forward accel respecting friction-ellipse-reduced `max_accel`
3. **Pass 3** â€” backward brake respecting friction-ellipse-reduced `brake_decel`

For closed loops, the pass operates on a **double-unrolled** sequence (length 2N) and slices out the central N for the steady-state profile â€” handles wrap-around correctly.

### Per-piece data (`catalog/catalog.py`)

`_speed_table[piece_idx]` is a single-valued NumPy array. `_V2_DEFAULT_PHYSICS` and `_V2_ROUTE_PHYSICS` ([catalog.py:46-76](src_v2/catalog/catalog.py:46)) hold:

| Piece | Default route (used by `speed_table`) | Diverging routes (NOT in `speed_table`) |
|---|---|---|
| `R40_CURVE` | (320 mm, 0.97 m/s) | n/a |
| `R40_SWITCH_LEFT` | `through` (âˆž, 1.57 m/s) | `diverging` (320 mm, 0.97 m/s) |
| `R40_SWITCH_RIGHT` | `through` (âˆž, 1.57 m/s) | `diverging` (320 mm, 0.97 m/s) |
| `CROSS_90` | `horizontal` (âˆž, 1.57 m/s) | `vertical` (âˆž, 1.57 m/s) â€” same |
| `DOUBLE_CROSSOVER` | `track1_through` (âˆž, 1.57 m/s) | `cross_*` (320 mm, 0.97 m/s) |

For `R40_CURVE` at radius 320 mm and `mu_design = 0.25`:
`v_slide = sqrt(0.25 Â· 9.81 Â· 0.32) = 0.886 m/s` â‡’ this is the 0.89 m/s figure visible in every snapshot. The model is internally consistent with the V1 snapshot values.

## How the plan currently consumes the physics

[problem.py:196-202](src_v2/problem.py:196):

```python
useful_speeds = [
    float(self.catalog.speed_table[graph.slot_indices[slot]])
    for slot in slots_on_cycles
    if slot in graph.slot_indices
    and 0 <= graph.slot_indices[slot] < len(self.catalog.speed_table)
]
min_speed = min(useful_speeds) if useful_speeds else 0.0

out["F"] = np.array([-utilization, -min_speed])
```

`F[1] = -min(speed_table[piece_idx] for slot in slots_on_cycles)`. **Per-piece**, NOT per-route.

## The verdict â€” what's adequate, what isn't

### âœ… Adequate for the implementation plan

1. **Lateral stability physics** â€” slide / tip / Nadal triple-min is the textbook approach for narrow-gauge models. R40's binding limit is `v_slide` at 0.886 m/s, which matches every observed bottleneck in V1 snapshots and current V2.
2. **Friction ellipse + 3-pass speed profile** â€” Kapania-Gerdes pattern is the right framework for combined-grip motion with closed-loop steady state. Double-unroll handles closure correctly.
3. **Pessimistic `mu_design = 0.25`** â€” defensive; thesis-defensible against examiner pushback ("but real LEGO friction is closer to 0.30!"). Tunable via `TrainConfig.from_yaml`.
4. **Motor cap** â€” `v_motor_max = 1.10 m/s` for Powered Up matches published spec; cleanly separated from derailment cap so future motor swaps are a one-line change.
5. **Stoichiometric speed for R40 diverging route on switches** â€” already 0.97 m/s in `_V2_ROUTE_PHYSICS`, same as `R40_CURVE`. The data is right; the consumer just doesn't use it (see issue #1 below).

### ðŸŸ¡ Adequate but with caveats

6. **Mass / consist model** â€” `mass_trailing = 0.0` default means the coupler correction is inactive. Adequate for "bare locomotive" optimization (the V1 snapshots show this configuration). If the thesis wants to optimize for actual trains pulling cars (e.g., 4-wagon Crocodile or Maersk 60052), `mass_trailing` and `coupler_offset` need realistic values per consist. **Action**: document the bare-loco assumption explicitly in the thesis chapter; provide an example `train_config.yaml` for a multi-car consist.
7. **Crossing physics** â€” `CROSS_90` has `(âˆž, 1.57)` on both routes (no special speed cap for the rail gap at the X). In real LEGO this is fine â€” the gap is small enough that a Powered Up loco crosses without flange-to-rail-end issues at 1.57 m/s. Acknowledge this is an idealization.
8. **No grade / incline support** â€” LEGO 4DBrix tracks are flat by default; this is a non-issue unless someone introduces ramps. None of the planned templates (passing siding, figure-8, DC bridge) are vertical, so safe to ignore.

### âŒ Gaps that the implementation plan must address

9. **Per-route speed not used in `F[1]`** ([problem.py:196-202](src_v2/problem.py:196)): `speed_table[piece_idx]` returns the **default route** speed for switches and DOUBLE_CROSSOVER. So a passing siding's diverging branch â€” which actually traverses the slow 0.97 m/s diverging route â€” is reported as 1.57 m/s for that switch's contribution to `min_speed`. This silently overstates min_speed by ~62% on switched layouts. **The plan's headline target ("â‰¥0.89 m/s preserved on Phase 8 switched layouts") is currently unverifiable** because the metric being measured ignores the slow route.

   **Fix scope**: Phase 5a should refactor problem.py:_evaluate to look up speeds **per (slot, route_name)** using the same `branch_labels` map already used to identify cycle membership. The `_V2_ROUTE_PHYSICS` data is already structured for this; the catalog's per-route physics is just not exposed in `speed_table` form. Either:
   - **(a)** Extend the catalog with `get_speed_for_route(piece_idx, route_name) -> float`, and rewrite the F[1] aggregation to use `(slot, route)` pairs from `branch_labels`.
   - **(b)** (Cleaner): refactor `speed_table` to a dict keyed by `(piece_idx, route_name)` with default fall-through, expose `get_speed_for_route` accessor.

   **Add this to Phase 5a's file list** ([catalog.py](src_v2/catalog/catalog.py) + [problem.py](src_v2/problem.py)). Without it, the F[1] objective doesn't truly reward "fast layouts" once switches are introduced â€” it just maintains the false-high default-route speed.

10. **Multi-path enumeration is missing in V2** â€” V1's `MultiPathLayout` enumerated 2^J traversal paths (each switch can be entered as through or diverging), and per-path speed profiles informed F[1]. V2's `PortGraph` has `branch_labels: {(slot, route_name) â†’ cycle_id}` ([types.py:383](src_v2/types.py:383)) but no mechanism to enumerate "which subset of switches' branches the train traverses on this lap". With Phase 5aâ€“7a active, a layout with 2 passing sidings + 1 figure-8 has 2Â³ = 8 distinct lap paths; the conservative F[1] should be `min over paths of min over pieces in path of speed`.

   **Fix scope**: either accept the conservative interpretation that **min_speed = slowest piece in any cycle on the largest component** (current implementation, route-aware) â€” which is what V1's `min` over per-segment speed caps already gives â€” OR introduce a `MultiPathLayout`-equivalent in V2 that enumerates paths and reports per-path lap times. The plan currently assumes the former; this is a defensible simplification (worst-case conservative) but **must be documented in the thesis** as the chosen semantic. V1's commentary at [problem.py:113-116](src_v2/problem.py:113) already acknowledges "min_speed is the V2 v_bottleneck semantics (strictly conservative)".

11. **`compute_speed_profile` operates on a sequential `Layout`, not a graph** â€” for current single-cycle layouts this is fine (the layout IS the sequence). For Phase 5a+ with branches, the function would need to be called **per-cycle** in the graph, with each cycle's speed profile reported separately, then min'd. This is mostly mechanical to wire up but currently unwired.

   **Fix scope**: in problem.py, after Phase 5a, iterate `graph.connected_components â†’ cycles`, build a `Layout` per cycle, call `compute_speed_profile` per cycle, take `min over (per-cycle min_speed)`. Or simpler: aggregate the `v_eff_array` directly over cycle edges and skip the 3-pass algorithm if `min_speed` is the only metric needed. The 3-pass is only needed if `lap_time` or `avg_speed` matters â€” for `F[1] = -min_speed`, the per-piece curvature cap suffices.

   **Recommendation**: for `min_speed` objective, keep the per-piece curvature cap path (Pass 1 only) since accel/brake passes (2 and 3) only **lower** speeds, never raise them. The Pass-1 minimum IS the correct conservative `min_speed`. This means problem.py F[1] doesn't need to call `compute_speed_profile`; it needs to call `v_eff_array` directly with per-route radii. Faster AND simpler.

### ðŸ”µ Nice-to-have but not blocking

12. **Train length / multi-wagon coupler chain** â€” current `coupler_offset = 0.100 m` is a single value. A 4-car train has 4 different coupler positions, each contributing slightly different lateral demand on curves. Probably negligible for thesis but worth a sentence in the assumptions chapter.
13. **Tractive effort vs grade**, **rail joint impact**, **rail-head wear** â€” all real but absent. None matter for LEGO at this scale.

## Specific changes to fold into the implementation plan

The audit surfaces three concrete additions. Adding them now while plan is still in plan mode:

### Addition to Phase 5a (PASSING_SIDING template machinery)

Add to Phase 5a's file list:
- [src_v2/catalog/catalog.py](src_v2/catalog/catalog.py) â€” add `get_speed_for_route(piece_idx: int, route_name: str) -> float` method that returns the per-route speed from `_V2_ROUTE_PHYSICS` (with fallback to `speed_table[piece_idx]` for unknown routes)
- [src_v2/problem.py](src_v2/problem.py) â€” refactor F[1] computation to iterate `(slot, route_name)` pairs from `graph.branch_labels` and call `catalog.get_speed_for_route(piece_idx, route_name)`. Take `min over all (slot, route)` pairs in useful cycles.

This is essential for Phase 5a/b/c to produce **honest** F[1] values once switches are introduced. Without it, Phase 5c's "switch-using feasibles in Pareto archive" checkpoint reports inflated `min_speed` values that aren't physically realizable.

### Addition to Phase 6a (FIGURE_8_CROSS template)

`CROSS_90` already has identical 1.57 m/s on both routes â€” no per-route change needed for figure-8. âœ“ no action.

### Addition to Phase 7a (PARALLEL_DC_BRIDGE template)

DOUBLE_CROSSOVER has `cross_1_to_2` / `cross_2_to_1` at 0.97 m/s (320 mm radius). When DC is used in **bridge** mode (parallel-tracks-with-crossover), the train traverses one of the cross routes when switching tracks, hitting the 0.97 m/s cap. Phase 7a must use the per-route speed lookup added in Phase 5a â€” already covered by the Phase 5a addition above.

### Addition to Part 7 (Golden Python Rules)

**Rule 21** â€” *Always look up speed per (slot, route_name), not per piece_idx*: The `speed_table[piece_idx]` API returns the default route speed only. For switches and DOUBLE_CROSSOVER, the diverging/cross routes are slower. F[1] computation must use `catalog.get_speed_for_route(piece_idx, route_name)` keyed by the `(slot, route_name)` pairs from `graph.branch_labels`. Otherwise switched layouts report inflated min_speed values and the GA optimizes for a false target.

**Rule 22** â€” *For min_speed, skip the 3-pass profile and use Pass 1 directly*: `compute_speed_profile`'s passes 2 and 3 only lower speeds (accel/brake constraints); they cannot raise the per-piece curvature cap. For `F[1] = -min_speed`, calling `v_eff_array(train_config, radii_m)` directly is faster AND identical in result to running all three passes and taking the min. Reserve the full 3-pass for `lap_time` / `avg_speed` reporting.

**Rule 23** â€” *Use the measured-consist YAML as the single source of train physics*: Phase 0 produces `configs/trains/measured_consist.yaml` from physical measurements (loco mass, trailing-cars mass, coupler offset, CoG height). All thesis runs reference this file via `OptimizationConfig.train_config_path`. Never hardcode physics values in Python; never run main results against V2 defaults without the measured override (V2's default `coupler_offset = 0.100 m` is wrong for any real LEGO car â€” measured value is ~0.256 m for a 32-stud car). The optional sensitivity-sweep appendix runs at `mass_trailing âˆˆ {0.0, measured/2, measured}` to defend against examiner challenges, but the headline results use the measured consist throughout.

## Bottom line

**The lateral stability physics (slide/tip/Nadal/friction-ellipse/3-pass) is solid and thesis-grade.** No structural changes needed to that core.

**The consumption layer (problem.py F[1] computation) is broken for switched layouts** â€” it ignores per-route speed differences. Phase 5a must fix this; otherwise the optimization target is wrong once switches are involved. Cost: 1 method on `TrackCatalog`, 5-line refactor in `problem.py:_evaluate`. Sub-day.

**The 3-pass profile is unnecessary for `F[1]` computation** â€” Pass 1 alone gives the correct min_speed. Using only Pass 1 simplifies Phase 5a and avoids re-running the full profile per chromosome. The 3-pass remains valid for any future `lap_time` objective.

**Mass / coupler / multi-wagon** infrastructure is already in `TrainConfig`. Phase 0 activates it by producing `configs/trains/measured_consist.yaml` from physical measurements. No code changes needed beyond pointing `OptimizationConfig.train_config_path` at the measured YAML. F[1] = `min_speed` results will be effectively identical to bare-loco runs (per-piece curvature cap is loco-only at LEGO scale), but the measurement defends the thesis against "what about cars" challenges and sets up infrastructure for any future Phase 9 `lap_time` objective.

---

# Resume tomorrow â€” entry point

**Where we are**: plan complete (Parts 1-8 + Golden Rules), expert review applied, fact-check applied, consist decision locked (loco + 2 measured cars).

**Tomorrow's two paths**:

### Path A â€” Phase 0 (measurements first, recommended)
Take the kitchen scale + tape measure + the actual locomotive + 2 cars. Fill in `configs/trains/measured_consist.yaml` per the protocol in Part 8 â†’ "Phase 0 â€” Physical Measurement Protocol". 2-4 hours. No code yet. Validation gate: `v_eff(R40) â‰ˆ 0.886 m/s` after measurements. Then move to Phase 1.

### Path B â€” Phase 1 (code first, measurements later)
If hardware isn't accessible in the morning: start Phase 1 (`CycleClosureRepair` in `src_v2/repair.py`). Phase 1 is independent of Phase 0 measurements â€” it operates on chromosome geometry, not train physics. Phase 0 can be done any time before Phase 5a (when per-route speeds become a constraint).

**Recommended**: Path A, because:
- Phase 0 is short (2-4 hours) and unblocks every later phase
- Doing measurements first means thesis methodology chapter has data to cite
- Phase 1 implementation will go faster with clear physics setup confirmed first

### Open questions for the promotor (raise before code lands)

The advisor-prep questions written earlier in chat (Tier 1 first):
1. V1â†’V2 regression narrative â€” how to frame in thesis chapter
2. Scientific contribution â€” what's the one-sentence novelty
3. Physical verification feasibility â€” is 2-stud closure achievable in real LEGO?
4. Timeline realism â€” 17-27 day implementation alongside thesis writing

Decide whether to raise these BEFORE Phase 0 measurements (quick) or AFTER (gives advisor concrete numbers to react to).

### File state (tomorrow's first reads)

- This plan file: `C:\Users\dgpro\.claude\plans\c-users-dgpro-downloads-s10462-023-1052-resilient-harbor.md`
- V2 source: `S:\Programming\Repo\DamianGRZ\LegoTrackOptimizer\src_v2\`
- V1 reference (read-only): `S:\Programming\Repo\DamianGRZ\LegoTrackOptimizer - V1 kind of worked\src\`
- Train physics: `src_v2/train/physics.py` (no changes needed)
- Measurement target file (new): `configs/trains/measured_consist.yaml` (Phase 0 produces it)

### Tomorrow's first command (Path A)

After measurements:
```bash
python -c "from src_v2.train.physics import TrainConfig; t = TrainConfig.from_yaml('configs/trains/measured_consist.yaml'); print('mass_total=', t.mass_total, 'v_eff(R40)=', t.v_eff(0.32))"
```

Expected output: `mass_total=` 0.7-1.0 kg, `v_eff(R40)=` ~0.886 m/s.

If those numbers come out, you're cleared to start Phase 1.

The plan as written is implementable on top of the current physics model with the Phase 5a additions noted above. Three new Golden Python Rules (21, 22, 23) capture the operational implications. **No major physics rewrite is required.**

## Consist decision (locked + executed)

**Hardware identified**: AFM SL+æ©Ÿé–¢è»Š + ã‚«ãƒ¼ã‚´ãƒˆãƒ¬ã‚¤ãƒ³ set (model M0015TW, ~841 blocks). Private label of Militarybase Co., Ltd. (Japanese airsoft retailer); OEM China KAZI/WUEJIN/Sluban-class supplier. Sold on Amazon Japan / Yahoo / Yodobashi. **Promotor selected and provided this set personally**, so 3rd-party hardware is approved for the thesis.

**Consist composition** (loco â†’ short car â†’ long car):
- Locomotive (cargo train style, 33-stud / 265 mm body, with sound module)
- Car 1 (directly behind loco): short cargo wagon, 13-stud / 106 mm long
- Car 2 (rear): long cargo wagon, 28-stud / 220 mm long

**Power source**: 5Ã—AAA alkaline (= 7.5 V OCV fresh, ~1.0 Î© pack internal resistance, ~6 Wh deliverable energy). **Note**: AFM at 7.5 V is electrically equivalent to LEGO Powered Up Hub @ 7.5 V, **NOT** to LEGO Power Functions @ 9 V. Motor characteristics are extrapolated from Philohome's PUP train motor measurements (FA-130/RE-140 class can motor, ~20:1 gear, ~25-30 % efficiency, ~0.4 A continuous draw, ~0.77 W mechanical output @ 7.1 V motor terminal voltage).

**Track compatibility**: LEGO L-gauge (37.5 mm rail spacing) â€” physically verified by user. Track piece geometry assumed compatible with `data/track_pieces_v2.yaml` 4DBrix catalog. Electrical bus is plastic battery-train style (no track-fed power).

This sets the **baseline consist for the entire thesis**:
- All main results run with measured `mass_loco`, `mass_trailing`, `coupler_offset`, and measured `v_motor_max`.
- No bare-loco runs in the main thesis chapter (single-consist baseline, not a sweep).
- An optional brief sensitivity appendix (`mass_trailing âˆˆ {0.0, measured/2, measured}`) defends against examiner pushback at low cost.

**Implication**: at LEGO scale with R40-only inventory, the binding `min_speed` constraint is the per-piece curvature cap (loco-only, `v_slide(R) = 0.886 m/s`). With measured `v_motor_max = 1.26 m/s` (> `v_slide`), the motor does NOT bind on R40 â†’ bi-objective NSGA-II remains meaningful, F[1] = -min_speed varies by topology, the thesis narrative survives. (If `v_motor_max < v_slide`, F[1] would degenerate to constant â€” checked and ruled out by measurement.)

The value of measuring real hardware is:
- Defending the thesis against "but you didn't model the cars" without scrambling for data after-the-fact
- Setting up infrastructure for any future Phase 9 `lap_time` objective (which IS sensitive to consist mass)
- Catching V2's wrong default `coupler_offset = 0.100 m` (real measured value is 0.106 m for the short car)
- Catching V2's optimistic `v_motor_max = 1.10` default â€” actual is 1.26 m/s for AFM (above default; AFM motor is stronger than research-extrapolated 0.91 m/s for PUP-class @ 7.5 V).

## Phase 0 â€” Physical Measurement Protocol (Before Phase 1)

**Goal**: produce a single `configs/trains/measured_consist.yaml` that all subsequent phases use as the train physics input. One day of work, no coding.

### Equipment needed
- Kitchen scale (1 g resolution preferred; 5 g acceptable)
- Tape measure or caliper (1 mm resolution)
- The actual locomotive
- 2 actual large cars (e.g. tank cars, cargo wagons, passenger coaches â€” pick two from your collection that match what you'd actually run)
- Printed `configs/trains/measured_consist.yaml.template`

### Measurement table

Record values directly into `configs/trains/measured_consist.yaml`. Every field has a V2 default; values that match defaults can be omitted from YAML (TrainConfig fills them in via `from_yaml`).

| Field | Procedure | Expected range | V2 default | Priority | **Measured value** |
|---|---|---|---|---|---|
| `mass_loco` (kg) | Place loco on scale alone, **with batteries inside** (5Ã—AAA fresh = 1.5 V/cell) | 0.30â€“0.50 kg | 0.370 | **must measure** | **0.493 kg** |
| `mass_trailing` (kg) | Place both cars on scale together; record sum | 0.20â€“0.60 kg total | 0.0 | **must measure** | **0.327 kg** (94 g short + 233 g long) |
| `coupler_offset` (m) | Length of the **first car** (the one directly behind loco) â€” coupler-to-coupler measurement on flat track | 0.10â€“0.30 m | 0.100 | **must measure** | **0.106 m** (short car first) |
| `cog_height_h` (m) | Tilt-test: raise board until loco tips, record Î¸; `h = (gauge_b/2) / tan(Î¸)` | 0.025â€“0.045 m | 0.030 | should measure | **0.030 m (default)** â€” tilt-test deferred to optional appendix |
| `gauge_b` (m) | Caliper measurement between inner rail edges | 0.0375 m (4DBrix standard) | 0.0375 | optional | **0.0375 m** (verified, AFM track matches LEGO L-gauge) |
| `flange_angle_deg` (Â°) | Hard to measure; use catalog spec | 45â€“55Â° | 50 | use default | **50.0Â°** (default) |
| `mu_design` | Pull-test: incline ramp until loco slides; mu = tan(angle) | 0.20â€“0.35 | 0.25 | optional (default conservative) | **0.25** (default) |
| `mu_nominal` | Same procedure but record central estimate | 0.25â€“0.35 | 0.30 | optional | **0.30** (default) |
| `v_motor_max` (m/s) | Video at 60 fps, full consist, steady-state on 1+ m straight, frame-step in playback to compute v | 0.8â€“1.5 m/s | 1.10 | **MEASURED** (motor binds at R40 if v_motor_max < 0.886) | **1.26 m/s** (full consist, steady-state from 1-sec window in 47f52d36.mp4) |
| `max_accel`, `brake_decel` | 0-to-v_max distance test on straight (s/tÂ² â†’ uniform-accel-then-coast model) | 0.5â€“2.0 m/sÂ² LEGO scale | 3.92, 2.45 | **MEASURED** (`max_accel`) / use default (`brake_decel`) | **`max_accel = 0.68 m/sÂ²`** (measured: 1.35 m / 2.0 s starting from 0; uniform-accel-then-coast model with v_max = 1.26 cap; 95% CI [0.61, 0.77]); `brake_decel = 2.45` (default; passive coasting unmeasured). **Note**: this is "kinematic mean acceleration equivalent" under simplifying assumption of constant tractive force â€” NOT a motor characteristic; real DC motor dynamics are exponential `v(t) = v_âˆž(1âˆ’e^(-t/Ï„))`. Rolling resistance (a_rr â‰ˆ 0.03â€“0.10 m/sÂ²) not deducted; raw motor-only a is 4-15% higher. **Train is motor-torque-limited, NOT friction-limited** (a_friction at Î¼=0.25 would be 1.47 m/sÂ² â€” never reached). |

### CoG tilt-test procedure

1. Place loco on a flat board.
2. Slowly raise one end of the board until the loco tips over.
3. Record the tilt angle Î¸ at which tipping begins.
4. Compute: `cog_height_h = (gauge_b / 2) / tan(Î¸)`.
5. Validate: typical LEGO loco tips at Î¸ â‰ˆ 30â€“40Â°, giving `cog_height_h â‰ˆ 0.025â€“0.035 m`.

(If tilt-test isn't accessible: estimate `cog_height_h` as the height above rail head of the loco's mass centroid. For Powered Up locos this is approximately the height of the battery box's center, typically 25â€“35 mm above the rail head.)

### Coupler offset clarification (and what `coupler_offset = 0.106` means in this thesis)

The friction-ellipse coupler correction in `available_accel` uses `phi = coupler_offset / (2*R)` â€” the chord half-angle across the coupler length on a curve of radius R. For this thesis:

- The first car directly behind the loco is the **short cargo wagon (13 stud / 0.106 m, 94 g)**.
- The second car (rear) is the **long cargo wagon (28 stud / 0.220 m, 233 g)**.
- `coupler_offset = 0.106 m` reflects the FIRST car (short wagon) â€” the model uses one value, and the front-of-consist coupler dominates the lateral-force calculation.
- This particular ordering (short-then-long) was chosen mechanically: smaller coupler offset â†’ smaller phi on R40 (9.5Â° vs 19.7Â° if long was first) â†’ less lateral demand on the loco's grip envelope on curves â†’ more available longitudinal acceleration.
- The long car's coupler force is NOT modeled directly in V2 â€” it's lumped into `mass_trailing` as inertia.

### Speed measurement procedure that was actually executed

**Method**: 60 fps phone video, two markers on a 3.776 m straight track, full consist (loco + short + long), full motor power, fresh batteries (each AAA verified at 1.5 V on multimeter).

**Bare-loco trial** (`b990558a-9a71-45ca-83df-9f2f37b5de56.mp4`):
| Trial | Time | Speed |
|---|---|---|
| 1 | 3.30 s | 1.144 m/s |
| 2 | 2.62 s | 1.441 m/s |
| 3 | 2.84 s | 1.330 m/s |
| **Average** | | **~1.30 m/s** |
| Variance | 23 % | (high â€” startup-ramp effects on bare loco) |

**Full-consist trial** (`47f52d36-8e5d-4dfb-bc64-67fb703a1f97.mp4`):
| Trial | Time | Speed |
|---|---|---|
| 1 | 3.22 s | 1.173 m/s |
| 2 | 3.19 s | 1.184 m/s |
| 3 | 3.31 s | 1.141 m/s |
| **Trial average** | | 1.166 m/s |
| **Steady-state window** | 1.00 s for 1.26 m | **1.26 m/s** â† used as `v_motor_max` |
| Variance | 4 % | (low â€” load brings motor close to torque-balance) |

**Why steady-state takes priority over trial average**: the 1-sec window between 2-3 s of the consist video captures pure steady-state (well past the acceleration ramp, well before any deceleration). Trial averages over ~3.2 s include some startup phase, which depresses the average below the true sustained capability. **Authoritative `v_motor_max = 1.26 m/s`.**

### Sanity check at LEGO scale

```
v_slide(R40)    = sqrt(0.25 Ã— 9.81 Ã— 0.32)               = 0.886 m/s  (curvature limit, loco)
v_tip(R40)      = sqrt(9.81 Ã— 0.32 Ã— 0.01875 / 0.030)    = 1.401 m/s  (loco tip-over)
v_nadal(R40)    = sqrt(9.81 Ã— 0.32 Ã— 0.7256)             = 1.509 m/s  (Nadal wheel-climb)
v_max(R40)      = min(slide, tip, nadal)                  = 0.886 m/s  (slide-bound)
v_motor_max     = 1.26 m/s                                              (measured, full consist)
v_eff(R40)      = min(0.886, 1.26)                        = 0.886 m/s  â† slide binds, motor doesn't
v_eff(STR)      = 1.26 m/s                                              (motor on straights)
phi(R40)        = 0.106 / (2 Ã— 0.32)                      = 9.5Â°       (coupler angle)
mass_total      = 0.493 + 0.327                           = 0.820 kg
max_accel       = MEASURED 0.68 m/sÂ²                       (kinematic mean from 0â†’1.26 in ~1.86s)
                                                            friction-limited theoretical: 1.47 m/sÂ² (at Î¼=0.25)
                                                            measured << friction limit â†’ motor-torque-limited
                                                            NOT a motor characteristic; consist-level kinematic
                                                            equivalent under uniform-accel-then-coast model
```

`v_motor_max > v_slide(R40)` â†’ motor does NOT bind on R40 â†’ bi-objective NSGA-II `F[1] = -min_speed` remains meaningful â†’ thesis narrative survives.

### Output: `configs/trains/measured_consist.yaml` (final, with measured values)

```yaml
# configs/trains/measured_consist.yaml
# Pomiary: 2026-05-06
# SprzÄ™t: AFM SL+Cargo Train M0015TW (Militarybase house brand, OEM China KAZI/WUEJIN class)
# Klocki: 841 Å‚Ä…cznie
# Zasilanie: 5Ã—AAA alkaline fresh @ 1.5 V/cell measured = 7.5 V OCV
#   (AFM @ 7.5V â‰ˆ LEGO Powered Up Hub electrically; NOT LEGO Power Functions @ 9V)
# SkÅ‚ad: loco â†’ krÃ³tki wagon (1, 94 g, 0.106 m) â†’ dÅ‚ugi wagon (2, 233 g, 0.220 m)
# Track: LEGO L-gauge 37.5 mm (verified â€” AFM track is mechanically compatible)
# v_motor_max method: 60-fps phone video, full consist, steady-state from 1-s window
# Measured by: Damian GRZ

# --- Friction (V2 conservative defaults; pull-test deferred) ---
mu_design: 0.25
mu_nominal: 0.30

# --- Environment ---
g: 9.81

# --- Motor (measured) ---
v_motor_max: 1.26                # full consist steady-state on 1.26 m / 1.00 s window
                                 # bare loco gives ~1.30 m/s but with high variance
                                 # consist drag delta ~0.04 m/s (~3%)
                                 # NOTE: at AFM 5Ã—AAA = 7.5V, this exceeds research extrapolation
                                 # of 0.91 m/s (PUP-class @ 7.5V). Possible explanations:
                                 # (a) AFM motor is over-spec for the platform,
                                 # (b) gear ratio is lower (more speed, less torque),
                                 # (c) Mabuchi RE-260-class instead of FA-130-class.
                                 # Recommendation: open loco at end of thesis, read motor
                                 # part number, document for reviewer.

# --- Bogie / wheel geometry (LEGO L-gauge confirmed) ---
gauge_b: 0.0375
cog_height_h: 0.030              # default â€” tilt-test optional, deferred to appendix
flange_angle_deg: 50.0

# --- Speed-profile dynamics ---
brake_decel: 2.45                # default
max_accel: 0.68                  # MEASURED: 1.35 m / 2.0 s starting from 0 (uniform-accel-then-coast)
                                 # 95% CI [0.61, 0.77] m/sÂ² for Â±5cm position, Â±0.1s time uncertainty
                                 # Train is motor-torque-limited, NOT friction-limited at this consist+voltage:
                                 # friction-theoretical (Î¼=0.25) would be 1.47 m/sÂ², never reached.
                                 # This value is a "kinematic mean equivalent" â€” real DC motor dynamics are
                                 # exponential, not linear. Adequate for Pass 1 (F[1]=min_speed) which doesn't
                                 # use max_accel anyway (Golden Rule 22). Becomes binding for Phase 9 lap_time.
                                 # Rolling resistance ~0.03-0.10 m/sÂ² not deducted (would raise raw motor a
                                 # by 4-15% if Phase 9 needs it). Pre-thesis: redo with Tracker software for
                                 # tighter bounds.
                                 #   = 0.4 Ã— 9.81 Ã— 0.493 / 0.820
                                 # friction-limited; for AFM motor power output
                                 # (~0.77 W @ 7.1 V), this becomes torque-binding
                                 # at v > ~0.40 m/s but Pass-1 min_speed (Golden
                                 # Rule 22) is unaffected â€” only matters if Phase 9
                                 # adds lap_time objective.

# --- Consist (measured) ---
mass_loco: 0.493                 # 493 g WITH 5Ã—AAA inside
                                 # â‰ˆ 378 g loco + 115 g batteries
mass_trailing: 0.327             # 94 g (short, first) + 233 g (long, second)
coupler_offset: 0.106            # short car length (first wagon)
                                 # rationale: smaller phi on R40 = lower lateral
                                 # demand on loco grip envelope = better dynamics
```

### Validation gate before Phase 1

After the YAML is created, run:

```bash
python -c "from src_v2.train.physics import TrainConfig; t = TrainConfig.from_yaml('configs/trains/measured_consist.yaml'); print('mass_total=', t.mass_total, 'v_eff(R40)=', t.v_eff(0.32))"
```

Expected output: `mass_total=` â‰ˆ 0.7â€“1.0 kg, `v_eff(R40)=` 0.886 m/s (unchanged from defaults â€” confirms the speed cap is loco-only, as expected).

If `v_eff(R40)` is anything other than â‰ˆ0.886 m/s, re-check `mu_design`, `gauge_b`, `cog_height_h` â€” one of them was set to a non-default that should not have been.

### Updated reference in Phase 1

[Phase 1 implementation onwards](#phase-1) reads `train_config_path` from `OptimizationConfig`. Set this to `configs/trains/measured_consist.yaml` for all main thesis runs. The `runner.py` already resolves it via `config.load_train_config()` ([problem.py:71](src_v2/problem.py:71)). No code changes; YAML-only.

### Phase 0 deliverable (completed)

âœ… All required measurements collected:
- `mass_loco = 0.493 kg` (kitchen scale, with 5Ã—AAA fresh batteries)
- `mass_trailing = 0.327 kg` (94 g short + 233 g long)
- `coupler_offset = 0.106 m` (short car first, by design)
- `gauge_b = 0.0375 m` (verified LEGO-compatible)
- `v_motor_max = 1.26 m/s` (60-fps video, full consist, steady-state, 1-s window)
- `max_accel = 0.68 m/sÂ²` (measured: 1.35 m in 2 s starting from rest; CI [0.61, 0.77])
- Loco length: 0.265 m (informational; not in TrainConfig)

ðŸŸ¡ Optional (deferred to appendix):
- Tilt-test for `cog_height_h` (5 min experiment â€” currently using default 0.030 m)
- Pull-test for `mu_design` (15 min experiment â€” currently using V2 default 0.25; literature lower bound, not "conservative" without proof)
- Bare-loco vs full-consist drag delta (~3% measured; documented but not used in main runs)
- Worst-case battery V_cell = 1.3 V threshold v_motor_max retest (sensitivity check)
- **Tracker video reanalysis** (FREE, AAPT standard tool at https://physlets.org/tracker) â€” subpixel position tracking, automated v(t) and a(t) extraction, fits exponential DC motor dynamics. **Strongly recommended pre-thesis** to replace single-trial "uniform accel" approximation with proper Ï„_m mechanical time constant.

ðŸ“‹ Pending raw-data archive for thesis appendix:
- `b990558a-9a71-45ca-83df-9f2f37b5de56.mp4` (bare loco, 3 trials)
- `47f52d36-8e5d-4dfb-bc64-67fb703a1f97.mp4` (full consist, 3 trials)

ðŸ“Œ Pending action item (Stage 4 from compass research artifact):
- Open the locomotive body, photograph the can motor, read the part-number stamping. Likely Mabuchi FA-130/RE-140/RE-260-class running at over-voltage (rated 3-6V driven on 7.5V). **Doing this BEFORE thesis defense is recommended** â€” measured `v_motor_max = 1.26 m/s` is 1.83Ã— faster than published Philohome PUP train motor at 7.5V, confirming AFM uses a different (more powerful) motor class, but identity is unknown without tear-down.

### Methodology gaps to address in thesis chapter (per expert physics review)

The current measurements are sufficient for the optimizer (`F[1] = -min_speed` only uses Pass 1, which depends only on `v_motor_max` and `Î¼`). But for thesis-defense rigor, the following gaps in the acceleration measurement should be documented or closed before submission:

1. **Single trial only**: `max_accel = 0.68 m/sÂ²` comes from one observed run, no variance estimate. Master's thesis convention requires â‰¥3 trials with mean Â± std deviation. **Action**: re-run the 0-to-v_max test 3 more times under identical conditions; report mean and Ïƒ.
2. **Position precision unspecified**: "approximately 135 cm" lacks Â±X cm. **Action**: state measurement method (ruler? track-piece-count? video frame?) and resolution. Â±5 cm propagates to Â±4% in `a`; Â±10 cm propagates to Â±8%.
3. **Time precision unspecified**: "2 seconds" â€” stopwatch (~Â±0.2 s, terrible) or video frames (60 fps â†’ Â±0.017 s, good)? Use video frames.
4. **Battery state consistency**: was the acceleration test on the same battery state as the v_motor_max test? Document.
5. **Track orientation**: was this measured on a straight track section (no centripetal load)? Should be explicit.
6. **Uniform-accel assumption**: physically wrong for a DC motor (exponential dynamics with Ï„_m = 50-500 ms typical for this motor class). **Document explicitly** that 0.68 m/sÂ² is "kinematic mean acceleration equivalent" not a motor characteristic. **Action**: use Tracker software to fit v(t) curve and report Ï„_m as the physical parameter.
7. **Rolling resistance correction omitted**: a_rr â‰ˆ 0.03-0.10 m/sÂ² (LEGO ABS wheel on ABS rail, by analogy with model railway literature). Raw motor-driven a would be 0.71-0.78 m/sÂ². **Action**: include `a_rr` correction in thesis methodology, even as a literature estimate.
8. **No force validation**: thesis claims F_tractive = 0.557 N average from kinematics. Cross-check with luggage-scale stall test (5 minutes, would also bound the rolling resistance).
9. **Force/power consistency check**: at 0.4 A current draw and 6.7 V terminal (after pack sag), electrical input = 2.68 W. Mean mechanical output = 0.351 W. Implied efficiency = 13% â€” **lower than typical for a budget DC motor** (25-30% expected). Either current draw is lower than 0.4 A (motor running at lower load) or there are significant gearbox + drivetrain losses (50-75% gearbox efficiency for budget OEM is documented). **Action**: measure motor current draw with shunt + multimeter to validate.

### Implications for Phase 9 (potential lap_time objective)

If a `lap_time` objective is added later, the measured `max_accel = 0.68 m/sÂ²` becomes binding for compact layouts: `s_accel = v_maxÂ² / (2 Ã— max_accel) = 1.26Â² / 1.36 = **1.17 m**`. Any straight section shorter than 1.17 m means the train cannot reach `v_motor_max` before the next curve forces deceleration. For LEGO with R40 curves (â‰ˆ125 mm arc length per piece) and typical straights of 16-stud (128 mm) or 32-stud (256 mm) per piece, **most layouts will not let the train reach v_motor_max anywhere**. Effective speed everywhere will be below 1.26 m/s.

This is a strong **argument for** including `lap_time` in a Phase 9 future-work section: it reveals dynamic constraints (acceleration phase length) that pure steady-state `min_speed` misses. Compact layouts, which the optimizer currently can produce favorably, may actually under-perform on lap-time relative to layouts with longer straights.

### Why Phase 0 exists as a numbered phase

Inserting measurement as Phase 0 (before Phase 1) makes the thesis defensible: any examiner can ask "what consist did you optimize for?" and the answer is "the measured AFM consist documented in `configs/trains/measured_consist.yaml`, dated 2026-05-06, raw video archived in two .mp4 files." Without Phase 0, the answer would be "the V2 default values which I never verified" â€” much weaker. The phase produced no Python code; just a config file, video archive, and a thesis methodology paragraph.

### Why Phase 0 exists as a numbered phase

Inserting measurement as Phase 0 (before Phase 1) makes the thesis defensible: any examiner can ask "what consist did you optimize for?" and the answer is "the measured AFM consist documented in `configs/trains/measured_consist.yaml`, dated 2026-05-06, raw video archived in two .mp4 files." Without Phase 0, the answer would be "the V2 default values which I never verified" â€” much weaker. The phase produced no Python code; just a config file, video archive, and a thesis methodology paragraph.

---

# Part 9 Ã¢â‚¬â€ Research-Backed Final Revisions to Parts 5/6/7

This Part supersedes the speculative Golden Rules 24-35 sketched at the end of the final review with **citation-backed final versions** based on three parallel literature investigations conducted 2026-05-06. Where Part 5/6/7 conflicts with Part 9, **Part 9 wins**.

The three research questions and headline findings:

| Question | Headline | Citation |
|---|---|---|
| Why does `cx_prob=0.9` destroy feasibility for edge-set encodings? | Schema disruption rate Ã¢â€°Ë† `cx_prob Ãƒâ€” defining_length / (nÃ¢Ë†â€™1)`. For port-pair, defining length Ã¢â€°Ë† chromosome length Ã¢â€ â€™ Ã¢â€°Ë†90 % of feasibility schemata destroyed per crossover. **Not a tuning issue, a representation mismatch.** | Holland 1975; White 2018 (Grad Journal of Math 3:37-59); Lukasiewycz et al. PPSN 2008 |
| Mutation-only vs cx+mut for combinatorial problems | Mutation-only EAs win **94.16 % (113/120)** across 4 NP-hard combinatorial problems with statistical significance | Barrero, Camacho, D'Angelo, *Sci. World J.* PMC4137700 (2014) |
| Lamarckian vs Baldwinian repair for constrained MOEAs | **Baldwinian dominates** in 3 independent studies on combinatorial MOEAs (knapsack, scheduling, graph). Lamarckian collapses Pareto diversity. | Ishibuchi et al. EMO 2005; Ishibuchi et al. 2009 (Memetic Algorithms book ch.); Springer LNCS 14173 (2023) |
| ALNS pool size for 17+ ops | VRP literature uses **4-12 ops** typical. Turkes 2021 meta-analysis: adaptive layer over equal-weight = **0.14 % improvement, statistically marginal**. | Ropke-Pisinger 2006; Pisinger-Ropke 2010; Turkes-Sorensen-Hvattum, *EJOR* 2021; Mara et al. 2022 (251-paper survey) |
| Pymoo `DefaultMultiObjectiveTermination` defaults | `cvtol=1e-8` is dangerously tight; should be widened to `1e-4`. `xtol` is meaningless on integer chromosomes. | pymoo 0.6 docs + DeepHyper source mirror; Bezerra et al. 2019 *Evol. Comput.*; Wagner-Bringmann GECCO 2013 |

These findings rewrite five of the speculative Rules. The remaining six (27, 28, 30, 32, 33, 35) survive review and are kept as originally drafted.

---

## Ã‚Â§9.1 Ã¢â‚¬â€ Headline Decision: Crossover Probability (REPLACES Rules 25, 26, 31)

The single most consequential change to the plan.

### What the research actually says

**Schema theorem under one-point crossover** (Holland 1975, formalized in White 2018):

$$P(\text{schema disruption}) \approx p_{\text{xover}} \times \frac{\text{defining\_length}}{n_{\text{chromosome}} - 1}$$

For V2's port-pair encoding, every feasible schema spans a full connected component (every port-pair in a cycle must co-occur) Ã¢â€ â€™ defining length Ã¢â€°Ë† chromosome length Ã¢â€ â€™ disruption probability Ã¢â€°Ë† `p_xover Ãƒâ€” 1.0`. **At `p_xover = 0.9`, ~90 % of feasibility schemata are destroyed every crossover application.** This is not a performance degradation; it is a **representation correctness failure**. Building blocks cannot accumulate because they are destroyed at the rate they are created.

The V2 memory note (`project_crossover_destroys_feasibility.md`: 1/500 feasible at p=0.9 vs 500/500 at p=0.0) is the empirical signature of exactly this failure mode.

**Empirical sweep results** (Hassanat et al. 2019, MDPI *Information* 10(12):390): for *discrete/combinatorial* GAs the reported optimum is `cx_prob Ã¢Ë†Ë† [0.6, 0.8]`, **not 0.9**. The `cx_prob=0.9` default was calibrated for *continuous SBX* (Deb 2001) and has been incorrectly cargo-culted into discrete encodings. Pymoo's `SBX(prob=0.9)` is the proximate source Ã¢â‚¬â€ confirmed by direct inspection of pymoo 0.6.1.6 docs.

**Mutation-only as a principled design** (Barrero, Camacho, D'Angelo 2014, Sci. World J. PMC4137700): six GAs vs three mutation-only EAs across TSP, CVRP, N-Queens, Bin Packing Ã¢â€ â€™ mutation-only wins in **94.16 % (113/120) of cases**, significant in 81.25 %. This is the standard Evolution Strategy paradigm (Schwefel 1977; Rechenberg 1973), re-validated for modern combinatorial problems. **Mutation-only is not a workaround Ã¢â‚¬â€ it is the principled design when crossover cannot preserve structural invariants.**

**NSGA-II crossover-speedup theorem caveat** (Doerr & Qu, AAAI 2023, arXiv:2208.08759): the formal proof that crossover provides super-constant speedup for NSGA-II on OneJumpZeroJump **requires offspring be feasible** (CV = 0) to enter non-dominated sorting. For constrained problems where crossover produces infeasible offspring, the speedup mechanism is blocked. **Theory does not support `cx_prob > 0` for problems where crossover destroys feasibility.**

### Rule 25 (REVISED) Ã¢â‚¬â€ Set `crossover_prob = 0.0` via UI tweaks panel for every Phase 1+ run

**Action**: do **not** edit `configs/*.yaml` or the Pydantic defaults at `src_v2/config.py:55-56`. Per project workflow (memory note `feedback_dont_edit_yaml_for_runtime_params`), runtime hyperparameters stay dynamic and are set per run via the web UI tweaks panel ([web/tweaks-panel.jsx](web/tweaks-panel.jsx)). For each Phase 1+ run, the panel must set:

```yaml
algorithm:
  crossover_prob: 0.0    # was 0.9 Ã¢â‚¬â€ destroys port-pair schemata per Holland 1975 + Barrero 2014
  mutation_prob: 1.0     # was 0.1 Ã¢â‚¬â€ compensate for absent crossover diversity
```

**Rationale**: Barrero et al. 2014 mutation-only formulation; restores V2 memory note's 500/500 feasible regime as the GA's default operating point. Phase 4's junction-segment crossover (semantic-guarded per Rule 4) is the **only** crossover survivor; even it remains gated behind `crossover_prob`, so setting `crossover_prob = 0.0` disables both port-pair AND junction crossover for Phases 1-8.

**Mutation rate increase rationale**: with crossover absent, mutation must carry the full diversity-injection burden. `mutation_prob=1.0` (apply mutation to every individual every generation) restores the (ÃŽÂ¼+ÃŽÂ»)-EA Schwefel 1977 design.

### Rule 26 (REVISED) Ã¢â‚¬â€ Reserve a *post-Phase-8* thesis chapter for cx-prob ablation, not per-phase

The earlier Rule 25 (run `cx_prob Ã¢Ë†Ë† {0.0, 0.5, 0.9}` per phase checkpoint) was speculative. Research says the ablation point is now elsewhere: **after Phase 4-5a junction crossover lands, ablate junction-only `cx_prob` while port-pair crossover stays disabled**:

| Sweep | `cx_prob` (port-pair) | `cx_prob` (junction segment) | When |
|---|---|---|---|
| Baseline | 0.0 | 0.0 (disabled too) | Phase 1-3 |
| Junction-cx test | 0.0 | $\{0.3, 0.6, 0.9\}$ | After Phase 5a checkpoint |
| Final ablation chapter | 0.0 | best from above | Post-Phase-8 |

Junction-segment crossover is **structurally different** from port-pair crossover: junction descriptors are short (5 genes) and semantically-guarded swaps (Rule 4). The schema-disruption argument does NOT trivially apply to them. Empirical ablation needed; theory permissive.

### Rule 31 (REVISED) Ã¢â‚¬â€ ERX deferral remains correct; do NOT reintroduce port-pair crossover in any phase 1-8

**Without** Lukasiewycz-style feasibility-preserving crossover (PPSN 2008) or ERX (Whitley 1989), port-pair crossover is structurally broken. Phase 9 (if reached) is the only home for port-pair crossover. Rules 4 (junction crossover semantic guard) and 25 (cx_prob = 0.0) together fully replace the speculative "trust junctions + repair" Option B from Ã‚Â§C of the review.

---

## Ã‚Â§9.2 Ã¢â‚¬â€ Headline Decision: Repair Mode (REPLACES Rule 24)

### What the research actually says

**Schema theorem under repair** (Liepins-Vose 1991, *Complex Systems* 5; Whitley-Gordon-Mathias 1994, *Complex Systems* 8): Lamarckian repair (writeback to chromosome) does not break Holland's theorem outright; the building-block amplification inequality survives **but for schemata defined over the post-repair representation**, not the raw chromosome. The risk is empirical: if repair is highly non-injective (many chromosomes map to the same repaired solution Ã¢â‚¬â€ exactly the case for constructive cycle-closure repair, which adds R40 to whichever inactive slot is available), Lamarckian writeback **collapses the chromosome pool to the repair attractor** and silently reduces effective population diversity.

**Three independent comparisons, same direction**:

1. **Ishibuchi, Yoshida, Murata 2005** (EMO, Springer LNCS 3410): NSGA-II + SPEA2 on multi-objective 0/1 knapsack (250- and 500-item). **Baldwinian wins on hypervolume across all configurations.** Lamarckian collapses to a smaller region of the Pareto front.
2. **Ishibuchi, Hitotsuyanagi, Nojima 2009** (book chapter, *Memetic Algorithms*): "for combinatorial MOEAs, Baldwinian dominates when the repair operator is highly constructive (large fraction of chromosome changed)."
3. **Springer LNCS 14173, 2023**: discrete combinatorial head-to-head confirms Baldwinian statistical superiority in early/middle phases; Lamarckian only catches up under very heavy convergence pressure.

**Partial Lamarckian sweet spot Ã¢â‚¬â€ primary source**: Orvosh & Davis 1993 (ICGA Proc.) found 5 % Lamarckian probability beat full Lamarckian and pure Baldwinian on graph-coloring + survival-network design. Houck, Joines, Kay & Wilson 1997 (*Evolutionary Computation* 5(1):31-60) systematically swept writeback probability 0-100 % across six problems Ã¢â€ â€™ broad optimum in **10-30 %**, with no sharp peak. **The popularly cited "20-40 %" figure is folk wisdom**; the primary literature supports **5-20 %**.

**Constructive vs destructive repair tradeoff** (Salcedo-Sanz 2009 survey, ScienceDirect; ScienceDirect 2020 adaptive-repair-MOGA paper):

| Mode | Convergence behavior | Diversity impact |
|---|---|---|
| **Destructive** (current V2: deactivate broken pieces) | Faster feasibility convergence | Biases toward sparse low-utilization solutions Ã¢â‚¬â€ **exactly the V2 small-oval pathology** |
| **Constructive** (Phase 1: add R40 to close cycles) | Slower convergence; introduces larger perturbation | Increases diversity short-term; broadens feasible region exploration; risks attractor collapse if Lamarckian |

The 2020 ScienceDirect adaptive-repair-MOGA paper found constructive repair reduced constraint violations faster and improved Pareto spread **when the repair heuristic was problem-aware (not random filling)** Ã¢â‚¬â€ Phase 1's cycle-aware closure repair qualifies.

### Rule 24 (REVISED) Ã¢â‚¬â€ Phase 1 closure repair must be **Baldwinian**, not Lamarckian

**Action**: Phase 1's `CycleClosureRepair` writes the repaired chromosome to a **separate buffer**, not back to `X[i]`. Pymoo Repair operator should:

```python
class CycleClosureRepair(Repair):
    def _do(self, problem, X, **kwargs) -> NDArray:
        for i in range(len(X)):
            X_repaired = self._repair_one(X[i].copy())  # work on copy
            # Lamarckian (default in V2):    X[i] = X_repaired
            # Baldwinian (Rule 24):          fitness uses X_repaired,
            #                                 chromosome stays at X[i]
            ...
        return X    # NOT modified Ã¢â‚¬â€ Baldwinian
```

But pymoo's `Repair._do` contract returns the modified `X` for evaluation. **The Baldwinian pattern in pymoo** is implemented via **dual evaluation**: store repaired phenotype as a custom field, but pymoo's `Evaluator` only round-trips F/G/CV. So:

**Concrete Baldwinian pattern for V2**:
1. `Repair._do` clones X internally, computes `X_repaired`, calls decoder on `X_repaired`, writes F/G/CV from `X_repaired`'s phenotype, but returns the **original X** so the chromosome pool retains its raw form.
2. The decoder is invoked **inside** Repair (against pymoo convention Ã¢â‚¬â€ but the alternative is intractable: pymoo's `Evaluator` runs on the chromosome it sees, which would mean re-decoding the same X twice).

**Equivalent simpler implementation using pymoo's pheno passthrough**: stash the repaired graph on `out["pheno"]` from `_evaluate` (pymoo standard key, round-trips through Starmap workers per Rule 1) and have `_evaluate` skip repair-aware decoding by reading `pheno` if present. This is the cleanest pattern; the repaired phenotype is the evaluation source of truth, the raw chromosome remains in the pool for crossover/selection inheritance.

**Trade-off acknowledged**: full Baldwinian costs ~5-10 % more compute (decoder runs twice if not cached). Acceptable for the ~17-fold diversity gain reported in the Ishibuchi 2005/2009 results.

### Rule 24a (NEW) Ã¢â‚¬â€ If partial Lamarckian is needed, use 5-20 % writeback (Orvosh-Davis 1993; Houck 1997), NOT 20-40 %

If Phase 1 testing shows Baldwinian-only is too slow (feasibility rate plateau below target), the fallback is **partial Lamarckian with `writeback_prob Ã¢Ë†Ë† [0.05, 0.20]`**, explicitly cited from primary sources, NOT the secondary-source folk wisdom range. Concrete config flag: `repair.writeback_prob: 0.10` (default 0.0 = pure Baldwinian).

### Rule 24b (NEW) Ã¢â‚¬â€ Report pre-repair AND post-repair CV in thesis methodology

The thesis must distinguish:
- **Raw CV** (pre-repair): what NSGA-II's selection saw if Baldwinian
- **Repaired CV** (post-repair): the phenotype's actual constraint state

Both metrics belong in the experiment chapter. Examiner question "did NSGA-II see the repaired or unrepaired chromosome?" must have a clear citation-backed answer. Under Baldwinian: NSGA-II sees raw chromosome with *repaired phenotype's* fitness. The thesis must describe this hybrid explicitly.

---

## Ã‚Â§9.3 Ã¢â‚¬â€ Operator Pool Size & Selection Mechanism (REPLACES Rule 29)

### What the research actually says

**Empirical operator pool sizes in published ALNS implementations**:

| Implementation | Destroy ops | Repair ops | Total |
|---|---|---|---|
| Ropke-Pisinger 2006 (PDPTW) | 6 | 4 | 10 |
| Crainic et al. 2013 (2E-VRP) | 8 | 4 | 12 |
| Voigt 2024 review (211 papers, *EJOR*) | 57 distinct | 42 distinct | (catalog total; per-paper cluster: 3-12) |

V2's current 17-mutation-op pool **already exceeds the empirical typical**. Phase 5c-7c additions push toward 27 ops, **2-4x beyond the practical range**.

**Adaptive layer effectiveness**: Turkes, Sorensen, Hvattum 2021 (*EJOR*) Ã¢â‚¬â€ meta-analysis of 25 ALNS implementations using random-effects model. Adaptive weighting yields **0.14 % mean improvement** over equal-weight selection. **Statistically marginal and problem-dependent.** This calls into question whether maintaining reliable ALNS weights across 17+ ops is worth the implementation complexity at all.

**Cold-start best practice for new operators**: MAB literature canonical answer is **optimistic initialization** Ã¢â‚¬â€ assign reward equal to current best observed reward. For roulette-wheel: **initialize new operator's weight at `max(current_weights)`, not the mean.** Equal-mean initialization (my speculative Rule 14) under-explores. **UCB1 handles this automatically** via the `sqrt(ln(N)/n_i)` exploration term Ã¢â‚¬â€ `n_i = 0` for a new operator gives infinite UCB bonus.

**Wouda ALNS framework v7.0.0** (the Python library V2's ALNSCallback structurally resembles): supports `RouletteWheel`, `SegmentedRouletteWheel`, `AlphaUCB(alpha=0.05)`, `MABSelector` via MABWiser. AlphaUCB with ÃŽÂ±Ã¢â€°Â¤0.1 explicitly recommended in framework docs. Wouda & Lan 2023 (*JOSS*).

**Meta-operator pattern precedent**: Derbel et al. PPSN 2012 (LNCS 7492) "Adaptive Operator Selection at the Hyper-level" Ã¢â‚¬â€ two-level AOS hierarchy. Melo et al. 2024 (*EJOR* 312:70-91) MAB-based hyper-heuristics for combinatorial optimization. **Wrapping similar sub-operators under one ALNS slot with internal uniform sub-selection is published practice**, not invented heuristic.

### Rule 29 (REVISED) Ã¢â‚¬â€ Cap effective ALNS pool at Ã¢â€°Â¤ 12-15 via meta-operators

**Action**: when Phase 5c-7c land, consolidate semantically similar new ops under meta-slots:

| Meta-op | Sub-ops dispatched internally |
|---|---|
| `tune_passing_siding` | toggle / reposition / swap_handedness / adjust_n_straights |
| `tune_figure8` | toggle / move_anchor / change_lobe_size / swap_handedness |
| `tune_dc_bridge` | toggle / move_anchor / adjust_offset / swap_routes |

This keeps **effective ALNS pool at ~14**: 17 current ops minus the 4 already-similar siding ops collapsed under `tune_passing_siding` = 14, plus 2 new meta-ops for figure-8 and DC = **16 effective ALNS slots**, just above the empirical upper bound but below the original 27-op concern.

### Rule 29a (REVISED Rule 14, was "initial weight = mean") Ã¢â‚¬â€ Cold-start at `max(current_weights)`, NOT mean

When adding meta-ops in Phases 5c/6/7, the new entry's initial weight is `max(self._weights)`, not `np.mean`. Cite: MAB optimistic-initialization (Sutton & Barto 2018, ch. 2.6) and Auer-Cesa-Bianchi-Fischer 2002 *Mach. Learn.* on UCB1 cold-arm handling.

### Rule 29b (NEW) Ã¢â‚¬â€ Consider migrating from RouletteWheel to AlphaUCB(ÃŽÂ±=0.05) at Phase 5c

**Trigger condition**: if Phase 5c's checkpoint shows new junction operators have <5 trials/gen due to the now-16-op pool (with `pop_size=1000, mutation_prob=1.0` per Rule 25), switch ALNS selector from weighted roulette to AlphaUCB. AlphaUCB's exploration bonus structurally mitigates the trial-starvation problem.

**Implementation cost**: ~30 lines in `alns_callback.py` to compute UCB scores instead of normalize-and-sample weights. Wouda ALNS framework provides reference implementation.

**Honest caveat**: Turkes 2021's 0.14 % effect-size finding means this migration's expected ROI is small. **Do not migrate without empirical justification** from the Phase 5c checkpoint.

---

## Ã‚Â§9.4 Ã¢â‚¬â€ Termination Criteria (REPLACES Rule 34)

### What the research actually says

**Pymoo 0.6.1.6 `DefaultMultiObjectiveTermination` defaults** (verified via pymoo docs + DeepHyper source mirror):

| Param | Default | Concern for LEGO |
|---|---|---|
| `xtol` | `0.0005` | **Meaningless on int chromosomes** Ã¢â‚¬â€ should be disabled |
| `ftol` | `0.005` | Relative HV improvement; uses `only_feas=True` so safe during infeasible plateau |
| `cvtol` | `1e-8` | **Dangerously tight** Ã¢â‚¬â€ will trigger on numerical-noise plateaus |
| `period` | `50` | Fine for `n_max_gen=500` |
| `n_skip` | `5` | Fine |
| `n_max_gen` | `1000` | Override per-config |

**HV-based stopping thresholds** (Bezerra-Lopez-Ibanez-Stutzle 2019 *Evol. Comput.* 28(2):195; Wagner-Bringmann GECCO 2013): 0.1 % relative HV improvement per period is a tight convergence threshold for clean bi-objective problems. **For constrained problems with infeasible plateaus, threshold should be loosened to 1 %** Ã¢â‚¬â€ tight thresholds cause premature termination during the early infeasible-dominated phase common in our problem.

### Rule 34 (REVISED) Ã¢â‚¬â€ Termination params calibrated for constrained integer NSGA-II

```python
from pymoo.termination.default import DefaultMultiObjectiveTermination

termination = DefaultMultiObjectiveTermination(
    xtol=1.0,        # disable: integer chromosomes; xtol semantically meaningless
    cvtol=1e-4,      # widened from 1e-8 (Bezerra 2019 recommendation); avoids
                     # false-trigger on numerical plateau noise during early infeas
    ftol=0.01,       # 1% rel-HV; loosened from 0.005 default for constrained MOPs
                     # (Wagner-Bringmann GECCO 2013)
    period=50,       # default; fine for n_max_gen=500
    n_skip=5,        # default
    n_max_gen=config.algorithm.n_gen,   # per-config; 200/500 typical
)
```

Replace the current `get_termination("n_gen", config.algorithm.n_gen)` in `runner.py:260`. Add a Phase-8 deliverable.

**Expected wall-clock saving**: 20-40 % across the experiment suite, conservative estimate based on 2025 Lsdyna/Ansys NSGA-II convergence study reporting 15-20 % early termination on ZDT/DTLZ benchmarks. LEGO's constrained landscape may save more.

---

## Ã‚Â§9.5 Ã¢â‚¬â€ Surviving Speculative Rules (Unchanged from Review)

Rules **27, 28, 30, 32, 33, 35** survive research review unchanged. Brief restatement:

- **Rule 27** Ã¢â‚¬â€ Heuristic emitter weights are first-class config keys (`algorithm.heuristic_emitter_weights` dict per config).
- **Rule 28** Ã¢â‚¬â€ Phase 8 archive admission recomputes raw CV from `G`, not `ind.get("CV")`, defensive against future ÃŽÂµ-handling re-enablement.
- **Rule 30** Ã¢â‚¬â€ Every template ships Ã¢â€°Â¥3 dedicated mutations (consolidated under meta-ops per Rule 29).
- **Rule 32** Ã¢â‚¬â€ Phase 4 must temporarily disable `eliminate_duplicates` OR be merged within 24h of Phase 5a (avoid junction-diversity collapse during the dedupe-misaligned interval).
- **Rule 33** Ã¢â‚¬â€ Stay bi-objective; topology stays in archive only, never $F_2$.
- **Rule 35** Ã¢â‚¬â€ F[1] aggregates over `(slot, route)` pairs from `branch_labels`, not slots, post-Phase-5a.

---

## Ã‚Â§9.6 Ã¢â‚¬â€ Stale References in Earlier Plan Parts (Document Now, Don't Edit Earlier)

| Location | Stale claim | Truth (per [problem.py:25-35](src_v2/problem.py:25)) |
|---|---|---|
| Part 1 | `G[7+T] = 1 - n_cycles Ã¢â€°Â¤ 0` | Index correct (7+T is cycles); rest of Part 1's constraint enumeration omits `G[5+T]` (incomplete switches), `G[6+T]` (incomplete crossings), `G[8+T]` (branch-cycle deficit), `G[10+T]` (loose ports). Total is **11+T**, not 7+T. |
| Part 1 | "G[9+T] = n_components - 1 > 0" | Index correct; constraint formula simplified (real form is `(n_comp - 1) / max(1, n_comp)`). |
| Part 4 | "constraint count is 10+T" | Wrong Ã¢â‚¬â€ actual is **11+T** (added `loose_port_ratio` post-V1-snapshot). |
| Part 5 Phase 8 admission rule | "`cv_admission_threshold = 1.0`" | Threshold is fine for the rule, but the rule must read **raw CV** (sum of G's clipped at 0), not pymoo's adapted CV. See Rule 28. |
| Part 5 Phase 1 file list | implicit Lamarckian | **Now Baldwinian** per Rule 24 Ã¢â‚¬â€ pheno-passthrough pattern. |
| Part 7 Rule 14 | "initial weight = mean of existing weights" | **Superseded by Rule 29a**: cold-start at `max(current_weights)` per MAB optimistic-initialization. |
| Part 7 Rule 25/26/31 (speculative) | "trust junctions + repair, keep cx_prob=0.9" | **Superseded by Rule 25 revision**: `crossover_prob = 0.0` in all configs; junction-cx ablation deferred to post-Phase-5a. |

These corrections are **documentation deltas**, not file edits to earlier Parts. The plan is now a layered document where Part 9 is the canonical source for any Rule numbered 24-35. Earlier Parts retain their historical analysis value but defer to Part 9 for actionable rules.

---

## Ã‚Â§9.7 Ã¢â‚¬â€ Action Items Before Phase 1 Implementation Starts

In strict order:

1. **Set runtime knobs via UI tweaks panel** for every Phase 1+ run (Rule 25 revised). Do **not** edit `configs/*.yaml` or `src_v2/config.py:55-56` Pydantic defaults — per project workflow (memory note `feedback_dont_edit_yaml_for_runtime_params`), runtime hyperparameters stay dynamic and per-run:
   - `crossover_prob: 0.0` (Barrero et al. 2014 mutation-only paradigm)
   - `mutation_prob: 1.0`
   - For thesis-baseline reproducibility, document the chosen values in the run's exported metadata or phase notes.

2. **Edit `runner.py:260`** to use `DefaultMultiObjectiveTermination` per Rule 34 (5-line change).

3. **Decide Baldwinian implementation pattern** for Phase 1: either (a) phenotype-passthrough via `out["pheno"]`, or (b) repair clones X internally and pymoo gets the original. Phase 1's PR description must state which.

4. **Optional pre-Phase-1 sanity run**: run `with_switches.yaml` baseline at `crossover_prob=0.0, mutation_prob=1.0` (no Phase 1 code yet) for 1 seed Ãƒâ€” 200 gen. Expected: feasibility rate goes from 0/1000 (current memory note) to 500/1000+. **If feasibility doesn't recover, the schema-disruption diagnosis is wrong** and Phase 1 needs re-planning. Cheap (~30 min) and decisive.

5. Then proceed with Phase 1 implementation per Part 5, augmented by Part 9 corrections.

---

## Ã‚Â§9.8 Ã¢â‚¬â€ Citations (full bibliography for the thesis)

### Crossover / mutation / schema theorem
- Holland, J.H. (1975). *Adaptation in Natural and Artificial Systems*. University of Michigan Press.
- Whitley, D., Starkweather, T., Fuquay, D. (1989). "Scheduling Problems and Traveling Salesmen: The Genetic Edge Recombination Operator." ICGA Proc.
- Larranaga, P., Kuijpers, C.M.H., Murga, R.H., Inza, I., Dizdarevic, S. (1999). "Genetic Algorithms for the TSP: A Review of Representations and Operators." *Artificial Intelligence Review* 13:129-170.
- Lukasiewycz, M., GlaÃƒÅ¸, M., Teich, J. (2008). "A Feasibility-Preserving Crossover and Mutation Operator for Constrained Combinatorial Problems." PPSN X, LNCS 5199.
- Barrero, D.F., Camacho, D., D'Angelo, M.R. (2014). "Crossover versus Mutation: A Comparative Analysis of the Evolutionary Strategy of Genetic Algorithms Applied to Combinatorial Optimization Problems." *Scientific World Journal* 2014:154676. PMC4137700.
- White, J. (2018). "An Overview of Schema Theory." *Graduate Journal of Mathematics* 3:37-59.
- Hassanat, A. et al. (2019). "Choosing Mutation and Crossover Ratios for Genetic Algorithms Ã¢â‚¬â€ A Review with a New Dynamic Approach." *Information* 10(12):390.
- Doerr, B., Qu, Z. (2023). "Runtime Analysis for the NSGA-II: Provable Speed-Ups From Crossover." AAAI 2023. arXiv:2208.08759.

### Lamarckian / Baldwinian repair
- Liepins, G.E., Vose, M.D. (1991). "Deceptiveness and Genetic Algorithm Dynamics." *Complex Systems* 5(1):31-46.
- Whitley, D., Gordon, V.S., Mathias, K. (1994). "Lamarckian Evolution, the Baldwin Effect and Function Optimization." *Complex Systems* 8(1):31-45.
- Orvosh, D., Davis, L. (1993). "Shall We Repair? Genetic Algorithms, Combinatorial Optimization, and Feasibility Constraints." ICGA Proc.
- Houck, C.R., Joines, J.A., Kay, M.G., Wilson, J.R. (1997). "Empirical Investigation of the Benefits of Partial Lamarckianism." *Evolutionary Computation* 5(1):31-60.
- Ishibuchi, H., Yoshida, T., Murata, T. (2005). "Comparison Between Lamarckian and Baldwinian Repair on Multiobjective 0/1 Knapsack Problems." EMO 2005, LNCS 3410.
- Ishibuchi, H., Hitotsuyanagi, Y., Nojima, Y. (2009). "Effects of Repair Procedures on the Performance of EMO Algorithms for Multiobjective 0/1 Knapsack Problems." In *Memetic Algorithms* book.
- Salcedo-Sanz, S. (2009). "A Survey of Repair Methods Used as Constraint Handling Techniques in Evolutionary Algorithms." *Computer Science Review* 3(3):175-192.
- Mezura-Montes, E., Coello Coello, C.A. (2011). "Constraint-Handling in Nature-Inspired Numerical Optimization: Past, Present and Future." *Swarm and Evolutionary Computation* 1(4):173-194.
- Springer LNCS 14173 (2023). "Comparing Lamarckian and Baldwinian Approaches in Memetic Optimization."

### ALNS / adaptive operator selection
- Ropke, S., Pisinger, D. (2006). "An Adaptive Large Neighborhood Search Heuristic for the Pickup and Delivery Problem with Time Windows." *Transportation Science* 40(4):455-472.
- Pisinger, D., Ropke, S. (2010). "Large Neighborhood Search." In *Handbook of Metaheuristics* (Springer), pp. 399-420.
- Karafotias, G., Hoogendoorn, M., Eiben, A.E. (2015). "Parameter Control in Evolutionary Algorithms: Trends and Challenges." *IEEE TEC* 19(2):167-187.
- Turkes, R., Sorensen, K., Hvattum, L.M. (2021). "Meta-Analysis of Metaheuristics: Quantifying the Effect of Adaptiveness in Adaptive Large Neighborhood Search." *European Journal of Operational Research* 292(2):423-442.
- Mara, S.T. et al. (2022). "A Survey of Adaptive Large Neighborhood Search Algorithms and Applications." *Computers & Operations Research* 146:105903.
- Wouda, N., Lan, L. (2023). "ALNS: A Python Implementation of the Adaptive Large Neighbourhood Search Metaheuristic." *Journal of Open Source Software* 8(81):5028.
- Reijnen, R. et al. (2024). "Online Control of Adaptive Large Neighborhood Search Using Deep Reinforcement Learning." ICAPS 2024. arXiv:2211.00759.
- Voigt, S. (2025). "A Review and Ranking of Operators in Adaptive Large Neighborhood Search for Vehicle Routing Problems." *European Journal of Operational Research* 322(2):357-375.

### Termination / hypervolume
- Wagner, T., Bringmann, K. (2013). "Indicator-Based Selection in Multiobjective Search." GECCO 2013.
- Bezerra, L.C.T., Lopez-Ibanez, M., Stutzle, T. (2019). "Automatically Designing State-of-the-Art Multi- and Many-Objective Evolutionary Algorithms." *Evolutionary Computation* 28(2):195-226.
- pymoo 0.6 official termination docs: https://pymoo.org/interface/termination.html

### Evolution Strategy foundations (mutation-only paradigm)
- Rechenberg, I. (1973). *Evolutionsstrategie*. Frommann-Holzboog.
- Schwefel, H.-P. (1977). *Numerische Optimierung von Computer-Modellen mittels der Evolutionsstrategie*. BirkhÃƒÂ¤user.

### Optimistic initialization (cold-start)
- Auer, P., Cesa-Bianchi, N., Fischer, P. (2002). "Finite-time Analysis of the Multiarmed Bandit Problem." *Machine Learning* 47:235-256.
- Sutton, R.S., Barto, A.G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.), MIT Press, ch. 2.6.

---

# Part 10 Ã¢â‚¬â€ Testing, Cohesiveness, Diagnostic Instrumentation

This Part formalizes the testing strategy that Parts 5/6/7/9 left implicit, audits five hidden cross-phase couplings, specifies a diagnostic instrumentation harness (per-gen CSV + 10-snapshot mechanism) that lands **before** Phase 1, and lists shared test fixtures used by every later phase.

The Plan as written through Part 9 is implementable but its testing scaffolding is informal: "module test" and "full optimization run" are conflated; cross-phase couplings are not named; infeasibility behavior has no instrumentation. Part 10 closes those gaps.

**Where Part 10 conflicts with Part 5's "Verification Strategy" or any per-phase "Checkpoint" stanza, Part 10 wins.** Earlier checkpoints become single-line summaries; the formal definition-of-done lives here.

---

## Ã‚Â§10.1 Ã¢â‚¬â€ The five-tier testing pyramid

Replace the plan's single "checkpoint" concept with five distinct testing tiers:

```
                         Ã¢â€Å’Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Â
                         Ã¢â€â€š T5: Full thesis runs   Ã¢â€â€š   30-180 min each
                         Ã¢â€â€š 5 seeds Ãƒâ€” 3 configs    Ã¢â€â€š   end-of-Phase-8 ONLY
                         Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Ëœ
                       Ã¢â€Å’Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Â
                       Ã¢â€â€š T4: Per-phase full runs    Ã¢â€â€š  10-30 min each
                       Ã¢â€â€š 1 config Ãƒâ€” 3 seeds Ãƒâ€” 200genÃ¢â€â€š  end of every phase
                       Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Ëœ
                     Ã¢â€Å’Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Â
                     Ã¢â€â€š T3: Mini-optimization tests    Ã¢â€â€š  30-60 sec each
                     Ã¢â€â€š tiny problem Ãƒâ€” 50gen Ãƒâ€” 50pop   Ã¢â€â€š  in CI per commit
                     Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Ëœ
                   Ã¢â€Å’Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Â
                   Ã¢â€â€š T2: Module integration tests       Ã¢â€â€š  1-5 sec each
                   Ã¢â€â€š 2-3 components composed            Ã¢â€â€š  in CI per commit
                   Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Ëœ
                 Ã¢â€Å’Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Â
                 Ã¢â€â€š T1: Pure unit tests                    Ã¢â€â€š  <1 sec each
                 Ã¢â€â€š single function, hand-crafted fixtures Ã¢â€â€š  in CI per commit
                 Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Ëœ
```

| Tier | Trigger | Duration | What it catches | What it misses |
|---|---|---|---|---|
| T1 unit | on change | <1s/test | logic bugs, off-by-one, edge cases | composition issues |
| T2 module integration | on change | 1-5s/test | composition bugs, contract mismatches | population dynamics |
| T3 mini-opt | on change (CI) | 30-60s | does the GA actually MOVE | scale effects |
| T4 per-phase full | end of phase | 10-30 min | phase delivers its checkpoint metrics | cross-seed variance |
| T5 thesis run | end of Phase 8 | 30-180 min | reproducible thesis numbers | (this IS the thesis) |

**The currently-missing tier is T3.** Plan jumps from T2 (module test) to T4 (200-gen full run); T3 (30-second full GA loop on a tiny problem) is the missing CI gate that catches GA-loop regressions on each change without consuming hours.

### T3 fixture spec Ã¢â‚¬â€ `tests/fixtures/mini_problem.py`

Authored at start of Phase 1, used by every subsequent phase:

```python
def mini_optimization_run(seed: int = 42, n_gen: int = 50, pop_size: int = 50,
                           inventory: dict = None, **algorithm_overrides):
    """30-second full GA loop on a 16-piece-default inventory.

    Returns: (final_pop, final_archive, gen_metrics_dataframe, snapshot_paths)

    Used as integration smoke test for every phase. If a phase change
    breaks mini_optimization_run, the bug is at the GA-loop level, not in
    isolated component logic.
    """
```

Default inventory: 16 R40_LEFT + 16 STR16 + 4 R40_SWITCH_LEFT + 4 R40_SWITCH_RIGHT Ã¢â‚¬â€ minimal but switch-bearing.

---

## Ã‚Â§10.2 Ã¢â‚¬â€ Per-phase test catalog with formal definition-of-done

Each phase's tests are numbered `<phase>.<n>`. Tier in `[T#]`. Acceptance criterion is the precise thing that must be true to mark the test passing.

### Phase 0 Ã¢â‚¬â€ Measurements

| # | Test | Tier | Acceptance |
|---|---|---|---|
| 0.1 | YAML loads via `TrainConfig.from_yaml` | T1 | `mass_total > 0.7` |
| 0.2 | Sanity invariants | T1 | `v_eff(0.32) Ã¢â€°Ë† 0.886` (Ã‚Â±0.001 m/s) |
| 0.3 | Worker pickle of TrainConfig | T2 | `pickle.dumps(train_cfg)` succeeds |

### Phase 1 Ã¢â‚¬â€ `CycleClosureRepair` (Baldwinian per Rule 24 revised)

| # | Test | Tier | Acceptance |
|---|---|---|---|
| 1.1 | 250Ã‚Â° deficit chromosome Ã¢â€ â€™ repair adds curves | T1 | post sum Ã¢Ë†Ë† [357Ã‚Â°, 363Ã‚Â°] |
| 1.2 | 380Ã‚Â° excess chromosome Ã¢â€ â€™ repair removes curves | T1 | post sum Ã¢Ë†Ë† [357Ã‚Â°, 363Ã‚Â°] |
| 1.3 | Inventory exhausted Ã¢â€ â€™ graceful stop | T1 | no exception, CV unchanged |
| 1.4 | Active slot not in any cycle Ã¢â€ â€™ ignored | T1 | unchanged |
| 1.5 | Two cycles, only one needs closure Ã¢â€ â€™ only that one modified | T1 | other cycle byte-equal |
| 1.6 | **Coupling B**: Baldwinian `out["pheno"]` round-trip via `Pool(2)` | T2 | callback sees same `PortGraph` as worker computed |
| 1.7 | API contract for `skip_anchor_slots` parameter (Coupling C placeholder) | T1 | parameter exists; default empty set |
| 1.8 | Mini-opt before/after Phase 1 on default inventory | T3 | feasibility +5pp; mean util +5pp; runtime <1.5Ãƒâ€” |
| 1.9 | **Pre-Phase-1 sanity** (Rule Ã‚Â§9.7 step 4) at `cx_prob=0.0` | T4 | feasibility Ã¢â€°Â¥500/1000 OR **abort Part 9** |
| 1.10 | Full `default` 200-gen Ãƒâ€” 3-seed | T4 | best feasible piece count Ã¢â€°Â¥58 (was 50) |

**DoD for Phase 1**: 1.1-1.10 green AND 1.9 passes the abort criterion. If 1.9 fails (feasibility <100/1000 at `cx=0.0`), the schema-disruption diagnosis is wrong and Part 9 needs revisiting.

### Phase 2 Ã¢â‚¬â€ Sequential-ring stadium emitter

| # | Test | Tier | Acceptance |
|---|---|---|---|
| 2.1 | Two-corners-of-8 pattern (Rule 10) | T1 | exact pattern: `[R40 Ãƒâ€” 8, STR Ãƒâ€” m, R40 Ãƒâ€” 8, STR Ãƒâ€” m]` |
| 2.2 | Inventory R40 < 16 Ã¢â€ â€™ emitter returns None | T1 | graceful skip |
| 2.3 | Boundary too small Ã¢â€ â€™ no stadium emitted | T1 | falls back without exception |
| 2.4 | Sequential edges port-uniqueness | T1 | no port reused across edges |
| 2.5 | Mini-opt with `with_switches` heuristic_ratioÃ¢â€°Â¥0.20 | T3 | Ã¢â€°Â¥20% of seeds are stadiums |
| 2.6 | Full `with_switches` 200-gen Ãƒâ€” 3-seed | T4 | best feasible piece count Ã¢â€°Â¥80 |

### Phase 3 Ã¢â‚¬â€ Auto-centering + ENCODING_VERSION=2

| # | Test | Tier | Acceptance |
|---|---|---|---|
| 3.1 | Bounded layout post-decode bbox.center Ã¢â€°Ë† boundary.center | T1 | within 1 stud |
| 3.2 | Anchor offset Ã‚Â±5% applied additively | T1 | matches fixture |
| 3.3 | `from_json` rejects pre-Phase-3 archive | T1 | `EncodingVersionMismatch` raised |
| 3.4 | `from_json` accepts current-version archive | T1 | round-trip identity |
| 3.5 | Mini-opt: deliberately off-center initial Ã¢â€ â€™ converges centered | T3 | bbox center within 5 stud by gen 50 |
| 3.6 | Full `default` 3-seed | T4 | boundary-violation count drops Ã¢â€°Â¥30%; feasibility non-decreasing |

### Phase 4 Ã¢â‚¬â€ Junction segment scaffolding (decoder ignores junctions)

| # | Test | Tier | Acceptance |
|---|---|---|---|
| 4.1 | `n_var = N_max + 4Ã‚Â·E_max + 5Ã‚Â·J_max + 3` formula | T1 | matches per-config calc |
| 4.2 | int16 overflow guard (Rule 2) | T1 | raises on inventory > threshold |
| 4.3 | `generate_bounds` covers junction range | T1 | xl/xu shapes correct |
| 4.4 | `validate_chromosome` on junction values | T1 | rejects out-of-range |
| 4.5 | **Coupling A Ã¢â‚¬â€ refactor delivered**: `JunctionCrossover` is its own operator | T2 | `crossover_prob=0.0, junction_crossover_prob=0.5` Ã¢â€ â€™ port-pair unchanged, junctions swap |
| 4.6 | Junction crossover semantic guard (Rule 4) | T2 | invalid anchor Ã¢â€ â€™ deactivate, don't propagate |
| 4.7 | **Coupling D contract**: canonical hash unchanged in Phase 4 | T1 | `hash(c1) == hash(c2)` when only junction descriptors differ |
| 4.8 | Phase 4 dedupe behavior (Rule 32) | T2 | with dedupe disabled, junction descriptor diversity preserved |
| 4.9 | Mini-opt: identical metrics to Phase 3 within 5% | T3 | regression sanity |
| 4.10 | Full `with_switches` 3-seed | T4 | metrics within noise of Phase 3 baseline |

### Phase 5a Ã¢â‚¬â€ `PASSING_SIDING` template materialization

| # | Test | Tier | Acceptance |
|---|---|---|---|
| 5a.1 | Hand-crafted 1-active-siding chromosome decodes valid layout | T1 | `incomplete_switch_ratio = 0`, n_switch_pairs = 1 |
| 5a.2 | Invalid siding (wrong n_straights) Ã¢â€ â€™ deactivated, inventory released | T1 | inventory restored |
| 5a.3 | Anchor not in cycle Ã¢â€ â€™ siding skipped | T1 | unchanged graph |
| 5a.4 | Inventory exhausted Ã¢â€ â€™ siding skipped | T1 | unchanged graph |
| 5a.5 | `is_valid_siding` boundary cases (2 stud, 5Ã‚Â°) | T1 | accepts at threshold, rejects beyond |
| 5a.6 | **Determinism (Rule 3)**: same chromosome Ã¢â€ â€™ same materialized graph | T1 | repeated decode byte-equal |
| 5a.7 | **Coupling C**: closure repair skips junction-anchor cycles | T2 | asymmetric oval seed survives Phase 1 repair |
| 5a.8 | **Coupling D**: canonical hash now distinguishes junctions | T1 | hash differs on `param_b` change |
| 5a.9 | Per-route F[1] aggregation over (slot, route) pairs (Rule 35) | T2 | switched layout F[1] = 0.886 m/s, NOT 1.57 |
| 5a.10 | **Coupling E**: dual-evaluation mode | T2 | both `old_F1` (per-piece) and `new_F1` (per-route) reported in `fitness.csv` |
| 5a.11 | TEMPLATES module pickle-safe (Rule 11) | T2 | `pickle.dumps(PASSING_SIDING_LEFT)` succeeds |
| 5a.12 | V1-snapshot-8-10 chromosome Ã¢â€ â€™ feasible after Ã¢â€°Â¤2 mutations | T3 | reproducibility on named benchmark |
| 5a.13 | Full `with_switches` 5-seed Ãƒâ€” 500-gen | T4 | Ã¢â€°Â¥1 feasible with switches by gen 100 |

### Phase 5b Ã¢â‚¬â€ Asymmetric-oval-with-siding seed

| # | Test | Tier | Acceptance |
|---|---|---|---|
| 5b.1 | Two-corners-of-8 pattern with `m+2` vs `m` STR asymmetry (Rule 10) | T1 | exact pattern asserted |
| 5b.2 | Seed asymmetry preserved through Phase 1 repair (Coupling C end-to-end) | T2 | +2 STR survives |
| 5b.3 | Inventory < 2 switches Ã¢â€ â€™ emitter falls back | T1 | graceful skip |
| 5b.4 | Boundary fit with `extra_axial=32` reservation | T1 | siding-injected layout fits |
| 5b.5 | Mini-opt: Ã¢â€°Â¥30% of feasibles have switches by gen 50 | T3 | seed Ã¢â€ â€™ selection survival |
| 5b.6 | Full `with_switches` 3-seed | T4 | switch-using feasible by gen 50 |

### Phase 5c Ã¢â‚¬â€ Junction mutations + meta-op consolidation

| # | Test | Tier | Acceptance |
|---|---|---|---|
| 5c.1 | `swap_handedness` on V1-snapshot-8-10 closes branch | T2 | CV: 0.5 Ã¢â€ â€™ 0.0 in 1-2 mutations |
| 5c.2 | `reposition` prefers nearest valid slot (Rule 9) | T1 | not random clamping |
| 5c.3 | `adjust_n_straights` respects inventory | T1 | doesn't exceed cap |
| 5c.4 | `toggle_active` doesn't corrupt other genes | T1 | byte-equality except target |
| 5c.5 | Meta-op pattern (Rule 29 revised) | T2 | 4 sub-ops dispatched under 1 ALNS slot |
| 5c.6 | ALNS cold-start at `max(weights)` (Rule 29a revised) | T1 | initial weight assertion |
| 5c.7 | ALNS pool size Ã¢â€°Â¤ 16 effective slots | T1 | invariant assertion |
| 5c.8 | Full `with_switches` 5-seed Ãƒâ€” 500-gen | T4 | switch-using feasible in archive by gen 100 |

### Phase 6, 7 Ã¢â‚¬â€ abbreviated (same template as 5a/5b/5c)

Defer test-level specifics until Phase 5 fully lands and the test scaffolding is established. Mirror 5a/5b/5c structure for `FIGURE_8_CROSS` and `PARALLEL_DC_BRIDGE` respectively.

### Phase 8 Ã¢â‚¬â€ Topology-aware archive admission + termination

| # | Test | Tier | Acceptance |
|---|---|---|---|
| 8.1 | `_topology_sig_for` lazy cache hit/miss | T1 | one decode per individual lifetime |
| 8.2 | Cache invalidated on chromosome mutation (Rule 6) | T2 | mutated individual re-decodes |
| 8.3 | Bounded scan: only top-K admitted | T1 | k = max_size//4 strictly |
| 8.4 | Topology-rich infeasible (CV<1, 1 switch) admitted | T2 | archive grows by 1 |
| 8.5 | Topology-poor near-feasible NOT admitted | T2 | archive unchanged |
| 8.6 | Raw CV recompute, NOT `ind.get("CV")` (Rule 28) | T1 | defensive against future ÃŽÂµ re-enable |
| 8.7 | Termination params from Rule 34 revised | T1 | `DefaultMultiObjectiveTermination` instantiated correctly |
| 8.8 | RobustTermination doesn't fire on infeasible plateau | T2 | 50-gen all-infeasible run continues |
| 8.9 | Full thesis runs: 5 seeds Ãƒâ€” 3 configs Ãƒâ€” 500 gen | T5 | meets DoD targets in Ã‚Â§Part 5 |

---

## Ã‚Â§10.3 Ã¢â‚¬â€ Five hidden cross-phase couplings (each with a dedicated test)

The plan presents Phases 0-8 as sequential checkpoints. They aren't Ã¢â‚¬â€ five non-trivial couplings cross phase boundaries. Each is testable; until Part 10, none had a dedicated test.

### Coupling A Ã¢â‚¬â€ Junction crossover probability is unspecified (highest risk)

`PortPairCrossover._do` currently gates on a single `crossover_prob`. Rule 25 sets it to 0.0 Ã¢â€ â€™ junction crossover is **also disabled**. But Rule 26 revised requires a junction-cx ablation `Ã¢Ë†Ë† {0.3, 0.6, 0.9}`. **The two rules contradict architecturally.**

**Resolution (NEW)**: Phase 4 must split `PortPairCrossover` into two operators:
- `PortPairCrossover` Ã¢â‚¬â€ port-pair edges only, gated by `crossover_prob` (= 0.0 in configs per Rule 25)
- `JunctionCrossover` Ã¢â‚¬â€ junction descriptors only, gated by new `junction_crossover_prob` (= 0.0 in Phase 4, ablated in Phase 5a+)

Pymoo composite via `OperatorFromGenerator` or override of `Crossover._do` to dispatch both internally.

**Test 4.5** validates the refactor.

### Coupling B Ã¢â‚¬â€ Baldwinian `out["pheno"]` round-trip is asserted, not tested (high risk)

Rule 24 routes Baldwinian repair through `out["pheno"]`. Rule 1 says custom `out` keys don't round-trip under `StarmapParallelization`. Rule 24 implies `out["pheno"]` is a "standard" pymoo key that DOES round-trip. **This is unverified for `PortGraph` objects.** A `PortGraph` containing `nx.cycle_basis` results may not pickle cleanly.

**Failure mode if untested**: Baldwinian silently degrades to "decode raw chromosome twice with no carryover" Ã¢â€ â€™ 5-10% slowdown with no diversity gain Ã¢â€ â€™ looks like Baldwinian failed empirically when it was never running.

**Test 1.6** validates round-trip via `Pool(2)` fixture.

### Coupling C Ã¢â‚¬â€ Phase 1 closure repair vs Phase 5b asymmetric-oval seed (medium risk)

Phase 1's `CycleClosureRepair` adds R40 to push cycle angle to 360Ã‚Â°. Phase 5b's seed is asymmetric by 2 STR16 (+32 stud) to compensate for switch injection. **Implementation order matters**: repair must run AFTER junction materialization, OR repair must read junction descriptors and skip preemptively.

**Test 1.7** specifies the API contract (`skip_anchor_slots: set[int]` parameter exists). **Test 5b.2 / 5a.7** validate end-to-end behavior.

### Coupling D Ã¢â‚¬â€ Phase 4 dedupe disabling vs Phase 5a re-enable (medium risk)

Rule 32: Phase 4 disables `eliminate_duplicates` (canonical hash hasn't been updated). Phase 5a updates the hash and re-enables. **No test in the original plan validates the re-enable point.** If the canonical hash extension is incomplete (forgets `param_b`), two chromosomes with different `param_b` hash identically Ã¢â€ â€™ silent dedupe bug.

**Test 4.7** validates Phase 4's hash invariance. **Test 5a.8** validates Phase 5a's hash distinguishability.

### Coupling E Ã¢â‚¬â€ Phase 5a F[1] refactor breaks regression-suite continuity (high methodology risk)

Phase 5a changes F[1] from per-piece-default-route to per-(slot, route)-pair. **Pre/post Phase 5a Pareto fronts are not directly comparable.** The standard regression check ("compare against previous phase baseline") will FALSELY flag a regression on F[1] when nothing actually got worse Ã¢â‚¬â€ only the metric got more honest.

**Resolution**: Phase 5a runs **dual-evaluation mode** for one regression cycle. Both old and new F[1] in `fitness.csv`. Compare old vs old (regression discipline) and report new alongside (going-forward truth). Document in thesis methodology that pre-Phase-5a F[1] was systematically inflated by ~62% on switched layouts.

**Test 5a.10** validates dual-evaluation mode.

---

## Ã‚Â§10.4 Ã¢â‚¬â€ Diagnostic Instrumentation: per-generation CSV harness

The plan tracks **feasible** solutions well (utilization, switches, archive). It under-instruments **infeasible** behavior. Yet 80%+ of the population is infeasible most of the run, and the thesis methodology chapter must describe what infeasibles do.

### What's currently invisible

| Question the thesis must answer | Current observability |
|---|---|
| Are infeasibles concentrated near feasibility (CV ~ 0.1-1.0) or scattered widely (CV ~ 10+)? | None |
| Does the GA push infeasibles toward feasibility (CV trending down)? | None |
| Are there CV plateaus where many chromosomes get stuck? | None |
| Which constraint dominates at each generation? | None |
| Does Baldwinian repair actually reduce CV vs Lamarckian? | None |
| Does dedupe rejection rate spike? | None |

### Spec Ã¢â‚¬â€ `src_v2/instrumentation.py`, `DiagnosticsCallback`

Single new file. Pymoo `Callback` subclass that writes `outputs_v2/{config}/diagnostics.csv` per generation:

```
gen, n_feasible, n_infeasible, mean_cv, median_cv, p90_cv, p99_cv,
constraint_0_mean, constraint_0_p90,    # closure_x
constraint_1_mean, constraint_1_p90,    # closure_y
constraint_2_mean, constraint_2_p90,    # closure_theta
constraint_3_mean, constraint_3_p90,    # boundary
constraint_4_mean, constraint_4_p90,    # collisions
... (per-type inventory T cols, summarized)
constraint_5+T_mean, ...                # incomplete switches
... etc up to G[10+T]
n_chromosomes_with_switches, n_chromosomes_with_crossings,
mean_pre_repair_cv, mean_post_repair_cv,
dedupe_rejection_rate,
mean_util_feasible, max_util_feasible,
mean_min_speed_feasible, max_min_speed_feasible
```

Cost: ~120 lines, ~3 hours. Lands **once before Phase 1**, used by every later phase. Without this, Phase 1's "feasibility +5pp" target is the only quantitative signal Ã¢â‚¬â€ too coarse for thesis-grade analysis.

**Plot helpers** (`src_v2/visualization/diagnostics_plots.py`): from `diagnostics.csv` produce:
- CV trajectory (median + p90 over generations) Ã¢â‚¬â€ answers "is the GA converging on feasibility?"
- Per-constraint contribution stacked area Ã¢â‚¬â€ answers "which constraint dominates?"
- Topology-richness vs CV scatter (per-gen) Ã¢â‚¬â€ answers "does topology richness correlate with infeasibility?"

These plots are thesis-chapter material, not just debugging.

---

## Ã‚Â§10.5 Ã¢â‚¬â€ Snapshot mechanism (10 ordered snapshots per run)

V1 produced `outputs_v1/with_switches/snapshots/snapshot_*.png` files showing the trajectory of best feasible and best infeasible layouts across the run. V2 currently saves only the final best (`best_layout.png`, `best_infeasible.png`) Ã¢â‚¬â€ the trajectory is invisible.

**Restore V1's snapshot mechanism as part of the diagnostic instrumentation landing before Phase 1.** This is the single highest-leverage tool for thesis-quality reporting: a 10-PNG strip showing how layouts evolved is more informative than any summary statistic, and the user explicitly asked for its return.

### Schedule Ã¢â‚¬â€ exactly 10 snapshots, evenly spaced

For a run of `n_gen` generations, produce snapshots at:

```python
schedule = sorted(set(
    np.linspace(1, n_gen, 10).round().astype(int).tolist()
))
```

Snapshot 1 fires at gen 1, snapshot 10 at gen `n_gen`, the other 8 evenly spaced (NumPy `linspace` semantics, rounded to ints, deduped if `n_gen < 10`).

**Schedule examples**:

| `n_gen` | Schedule |
|---|---|
| 200 | `[1, 23, 45, 67, 89, 112, 134, 156, 178, 200]` |
| 500 | `[1, 56, 111, 167, 222, 278, 334, 389, 445, 500]` |
| 100 | `[1, 12, 23, 34, 45, 56, 67, 78, 89, 100]` |
| 50 | `[1, 6, 12, 17, 23, 28, 34, 39, 45, 50]` |
| 10 | `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]` |
| 9 | `[1, 2, 3, 4, 5, 6, 7, 8, 9]` (dedupe drops collision at endpoints) |
| 5 | `[1, 2, 3, 4, 5]` (5 snapshots only) |

### Capture spec per snapshot

At each scheduled generation `N`:

1. **Best feasible** Ã¢â‚¬â€ `argmax(util)` among `all(G Ã¢â€°Â¤ 0)` individuals in `algorithm.pop`. If none, write no feasible PNG; metadata records `feasible: null`.
2. **Best infeasible** Ã¢â‚¬â€ `argmax(util)` among `any(G > 0)` individuals (largest infeasible by piece-count). If none (entire pop feasible), write no infeasible PNG; metadata records `infeasible: null`.
3. Render via existing `plot_layout(layout, catalog, boundary, title, save_path=...)` from `src_v2/visualization/`.

### File naming

```
outputs_v2/{config_name}/snapshots/
    snapshot_01_gen001_feasible.png
    snapshot_01_gen001_feasible.npy           Ã¢â€ Â chromosome bytes (re-render later)
    snapshot_01_gen001_infeasible.png
    snapshot_01_gen001_infeasible.npy
    snapshot_02_gen023_feasible.png
    ...
    snapshot_10_gen200_feasible.png
    snapshots_metadata.json
```

Storing the chromosome `.npy` alongside each PNG enables re-rendering with different visual styles after the fact (thesis figure regeneration without re-running optimizer).

### PNG title spec

```
Snapshot 03 (gen 67) | Feasible | 96 pcs, 58.5% util, 0.89 m/s, 1 cycle, 0 switches
Snapshot 03 (gen 67) | Infeasible | 144 pcs, 87.8% util, 0.89 m/s, 2 switches, 1 cycle, CV=0.50
```

Match V1's title format for visual consistency with archived V1 snapshots.

### Metadata sidecar Ã¢â‚¬â€ `snapshots_metadata.json`

```json
{
  "config_name": "with_switches",
  "n_gen": 500,
  "schedule": [1, 56, 111, 167, 222, 278, 334, 389, 445, 500],
  "snapshots": [
    {
      "n": 1,
      "gen": 1,
      "feasible": null,
      "infeasible": {
        "util": 0.42,
        "min_speed": 0.886,
        "n_pieces": 70,
        "cv": 12.34,
        "n_switches": 0,
        "n_crossings": 0,
        "n_components": 3,
        "n_cycles": 0,
        "png_path": "snapshots/snapshot_01_gen001_infeasible.png",
        "chromosome_path": "snapshots/snapshot_01_gen001_infeasible.npy"
      }
    },
    {
      "n": 2,
      "gen": 56,
      "feasible": {
        "util": 0.10,
        "min_speed": 0.886,
        "n_pieces": 16,
        "cv": 0.0,
        "n_switches": 0,
        "n_crossings": 0,
        "n_components": 1,
        "n_cycles": 1,
        "png_path": "snapshots/snapshot_02_gen056_feasible.png",
        "chromosome_path": "snapshots/snapshot_02_gen056_feasible.npy"
      },
      "infeasible": {
        "util": 0.78,
        "min_speed": 0.886,
        "n_pieces": 130,
        "cv": 4.5,
        "n_switches": 1,
        "n_crossings": 0,
        "n_components": 1,
        "n_cycles": 0,
        "png_path": "snapshots/snapshot_02_gen056_infeasible.png",
        "chromosome_path": "snapshots/snapshot_02_gen056_infeasible.npy"
      }
    }
  ]
}
```

### Implementation Ã¢â‚¬â€ `src_v2/snapshot_callback.py`

```python
"""Per-generation snapshot writer.

Captures best feasible and best infeasible layouts at 10 evenly-spaced
generations, writing PNGs + chromosome .npy + JSON metadata to
outputs_v2/{config}/snapshots/. Restores V1's diagnostic visualization.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from pymoo.core.callback import Callback

from .catalog import TrackCatalog
from .config import OptimizationConfig
from .decoder import decode_chromosome, port_graph_to_layout
from .visualization import plot_layout


def compute_snapshot_schedule(n_gen: int, n_snapshots: int = 10) -> list[int]:
    """Schedule of `n_snapshots` evenly-spaced generations from 1 to n_gen.

    Snapshot 1 always at gen 1; snapshot N always at gen n_gen; remaining
    n_snapshots - 2 evenly spaced. Deduped when n_gen < n_snapshots.
    """
    raw = np.linspace(1, max(n_gen, 1), n_snapshots).round().astype(int).tolist()
    return sorted(set(raw))


class SnapshotCallback(Callback):
    """Writes 10 ordered snapshots of best feasible / best infeasible layouts.

    Pymoo callbacks run in the MAIN process only (not worker pool), so
    snapshot rendering does not need to be pickle-safe and can use heavy
    matplotlib calls without affecting StarmapParallelization workers.
    """

    def __init__(
        self,
        n_gen: int,
        output_dir: Path,
        problem,
        catalog: TrackCatalog,
        config: OptimizationConfig,
        n_snapshots: int = 10,
    ) -> None:
        super().__init__()
        self.schedule = compute_snapshot_schedule(n_gen, n_snapshots)
        self.snapshot_dir = Path(output_dir) / "snapshots"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.problem = problem
        self.catalog = catalog
        self.config = config
        self._metadata: list[dict] = []
        self._n: int = 0
        self._logger = logging.getLogger(__name__)

    def notify(self, algorithm) -> None:
        gen = int(algorithm.n_gen)
        if gen not in self.schedule:
            return
        self._n += 1
        meta = {"n": self._n, "gen": gen, "feasible": None, "infeasible": None}

        pop = algorithm.pop
        F = pop.get("F")
        G = pop.get("G")
        X = pop.get("X")
        if F is None or G is None or X is None:
            return

        finite = ~np.isinf(F).any(axis=1)
        feas_mask = np.all(G <= 0, axis=1) & finite
        infeas_mask = (~feas_mask) & finite

        if feas_mask.any():
            idx = int(np.where(feas_mask)[0][np.argmin(F[feas_mask, 0])])
            meta["feasible"] = self._render(X[idx], F[idx], G[idx], gen, "feasible")

        if infeas_mask.any():
            idx = int(np.where(infeas_mask)[0][np.argmin(F[infeas_mask, 0])])
            meta["infeasible"] = self._render(X[idx], F[idx], G[idx], gen, "infeasible")

        self._metadata.append(meta)
        self._logger.info(
            f"Snapshot {self._n:02d} written at gen {gen} "
            f"(feasible={meta['feasible'] is not None}, "
            f"infeasible={meta['infeasible'] is not None})"
        )

    def _render(self, x: NDArray, f: NDArray, g: NDArray,
                gen: int, kind: str) -> dict:
        graph = decode_chromosome(
            x, self.problem.dims, self.catalog, self.problem.decoder_config,
        )
        layout = port_graph_to_layout(graph, self.catalog)

        cv = float(np.sum(np.maximum(0, g))) if kind == "infeasible" else 0.0
        spec = self.catalog.spec.by_id
        n_switch_slots = sum(
            1 for pid in graph.slot_pieces.values()
            if pid in spec and spec[pid].kind == "switch"
        )
        n_switches = n_switch_slots // 2
        n_crossings = sum(
            1 for pid in graph.slot_pieces.values()
            if pid in spec and spec[pid].kind == "crossing"
        )

        title_parts = [
            f"Snapshot {self._n:02d} (gen {gen})",
            kind.capitalize(),
            f"{graph.n_slots} pcs",
            f"{-f[0]:.1%} util",
            f"{-f[1]:.2f} m/s",
            f"{n_switches} switches",
            f"{graph.n_cycles} cycle",
        ]
        if kind == "infeasible":
            title_parts.append(f"CV={cv:.2f}")
        title = " | ".join(title_parts)

        png_path = self.snapshot_dir / (
            f"snapshot_{self._n:02d}_gen{gen:03d}_{kind}.png"
        )
        chromo_path = self.snapshot_dir / (
            f"snapshot_{self._n:02d}_gen{gen:03d}_{kind}.npy"
        )

        plot_layout(
            layout, self.catalog, self.config.boundary,
            title, save_path=png_path,
        )
        np.save(chromo_path, x)

        return {
            "util": float(-f[0]),
            "min_speed": float(-f[1]),
            "n_pieces": int(graph.n_slots),
            "cv": cv,
            "n_switches": n_switches,
            "n_crossings": n_crossings,
            "n_components": int(graph.n_components),
            "n_cycles": int(graph.n_cycles),
            "png_path": str(png_path.relative_to(self.snapshot_dir.parent)),
            "chromosome_path": str(chromo_path.relative_to(self.snapshot_dir.parent)),
        }

    def finalize(self) -> None:
        """Write metadata JSON. Call after `minimize()` returns."""
        meta_path = self.snapshot_dir / "snapshots_metadata.json"
        config_name = getattr(self.config, "name", "unknown")
        meta_path.write_text(json.dumps({
            "config_name": config_name,
            "n_gen": int(self.config.algorithm.n_gen),
            "schedule": self.schedule,
            "snapshots": self._metadata,
        }, indent=2))
        self._logger.info(
            f"Snapshot metadata written: {meta_path} "
            f"({len(self._metadata)} snapshots)"
        )
```

### Wiring into `runner.py`

Insert into the existing `chain` list in `run_optimization` ([runner.py:279-284](src_v2/runner.py:279)):

```python
snapshot_cb = SnapshotCallback(
    n_gen=config.algorithm.n_gen,
    output_dir=output_dir,
    problem=problem,
    catalog=catalog,
    config=config,
    n_snapshots=10,
)
chain = [
    FinalizationGatingCallback(repair, threshold=0.9),
    alns,
    epsilon_archive,
    dedupe,
    snapshot_cb,                           # NEW Ã¢â‚¬â€ runs after dedupe
]
if verbose:
    chain.append(ProgressCallback(...))
```

After `minimize()` returns:

```python
epsilon_archive.finalize()
snapshot_cb.finalize()                     # NEW Ã¢â‚¬â€ writes JSON metadata
```

### Tests

| # | Test | Tier | Acceptance |
|---|---|---|---|
| S.1 | Schedule for n_gen=200 | T1 | exact list `[1, 23, 45, 67, 89, 112, 134, 156, 178, 200]` |
| S.2 | Schedule dedupe for n_gen=5 | T1 | `[1, 2, 3, 4, 5]` |
| S.3 | Schedule for n_gen=10 | T1 | `[1, 2, ..., 10]` |
| S.4 | Schedule for n_gen=1 | T1 | `[1]` (single snapshot) |
| S.5 | Snapshot fires only at scheduled gens | T1 | mock `notify`, count invocations |
| S.6 | Best feasible selection (highest util among feasibles) | T1 | hand-crafted pop |
| S.7 | Best infeasible selection (highest util among infeasibles) | T1 | hand-crafted pop |
| S.8 | All-feasible population: only feasible PNG written | T1 | `meta.infeasible == null` |
| S.9 | All-infeasible population: only infeasible PNG written | T1 | `meta.feasible == null` |
| S.10 | Empty pop edge case (sentinel `+inf`): no PNG | T1 | `notify` returns without exception |
| S.11 | Mini-opt run produces 10 PNG pairs + JSON metadata | T3 | filesystem assertions; JSON parse succeeds |
| S.12 | Multiprocessing safety: callback fires only in main process | T2 | with `Pool(2)`, `_n` reaches 10 not 20 |
| S.13 | Chromosome `.npy` round-trips back to same `PortGraph` | T2 | re-decode identity |

### Why Part 10 / before Phase 1, not Phase 1+

Snapshot mechanism is **pure observability** Ã¢â‚¬â€ it doesn't change algorithm behavior, only adds visibility. It must land **before** Phase 1 because every Phase 1 checkpoint metric ("feasibility +5pp on default", "best feasible piece count Ã¢â€°Â¥58") becomes much easier to interpret with visual snapshots showing how layouts evolved. Phase 1 reviewer asks "did the GA actually grow the loop?" Ã¢â‚¬â€ answer is the snapshot strip, not a CSV row.

### Cost

- `src_v2/snapshot_callback.py` Ã¢â‚¬â€ ~180 lines (above)
- 13 tests Ã¢â‚¬â€ ~1 day to write
- Wiring into `runner.py` Ã¢â‚¬â€ 8 lines
- **Total**: ~1 day implementation, lands once, used by every later phase.

---

## Ã‚Â§10.6 Ã¢â‚¬â€ Six shared test fixtures (re-usable across all phases)

Currently the plan has zero shared fixtures. Each phase's tests would re-author the same hand-crafted chromosomes and oracles. Six fixtures, authored once during Phase 0/1, used by every subsequent phase.

| Fixture | Path | Purpose | Cost |
|---|---|---|---|
| `mini_problem.py` | `tests/fixtures/` | Tier-3 minimal GA loop (Ã‚Â§10.1) | 1 day |
| `hand_crafted_chromosomes.py` | `tests/fixtures/` | Library of 20-30 reference chromosomes (perfect oval, asymmetric oval, oval-with-siding, V1-snapshot-8-10, broken-cycle, isolated-component, Ã¢â‚¬Â¦) | 2 days |
| `decoder_oracle.py` | `tests/fixtures/` | Hand-traced expected decoder output for hand-crafted chromosomes | 1 day |
| `multiprocessing_smoke.py` | `tests/fixtures/` | 2-worker `StarmapParallelization` sanity test, validates Rule 1 / Rule 24 round-trips | 0.5 day |
| `regression_baselines/` | `tests/baselines/` | Captured metrics from Phase 0 baseline runs, JSON per config | 0.5 day |
| `thesis_metrics.py` | `tests/fixtures/` | Standardized metric extraction from a `result` object Ã¢â‚¬â€ feasibility rate, hypervolume, topology diversity, switch-usage rate | 1 day |

**Total fixture investment**: ~6 days, front-loaded into Phase 0/1. Pays back across Phases 1-8 (every later phase reuses these Ã¢â‚¬â€ no per-phase fixture re-authoring).

### `hand_crafted_chromosomes.py` Ã¢â‚¬â€ required entries

Minimum set the fixture library must contain (Phase 1 onward depends on these):

| Name | Topology | Used by |
|---|---|---|
| `perfect_oval_16_R40` | 16 R40, single cycle, closes exactly | Phase 1 baseline |
| `asymmetric_oval_24_pcs` | 16 R40 + 8 STR, single cycle | Phase 2 |
| `oval_with_siding_active` | 16 R40 + 8 STR + 2 switches + branch | Phase 5a |
| `oval_with_siding_45deg_off` | V1 snapshot 8-10 reproducer | Phase 5c |
| `figure_8_with_cross` | 24 R40 + 1 CROSS_90 + secondary lobe | Phase 6a |
| `parallel_dc_layout` | 2 parallel runs joined by DOUBLE_CROSSOVER | Phase 7a |
| `250deg_deficit` | Open chain summing to 250Ã‚Â° angle | Phase 1 |
| `380deg_excess` | Closed loop summing to 380Ã‚Â° angle | Phase 1 |
| `broken_cycle_2_components` | 2 disjoint loops | repair tests |
| `loose_port_chromosome` | Active piece with 1 port unpaired | repair tests |
| `inventory_exhausted` | Uses 9 R40 when inventory cap is 8 | repair tests |
| `isolated_active_slot` | 1 active piece with no edges | decoder tests |

---

## Ã‚Â§10.7 Ã¢â‚¬â€ Definition-of-Done formalization (replaces per-phase checkpoint stanzas)

For each phase, DoD is the conjunction of:

1. **All T1 unit tests for the phase pass** (no skip, no xfail)
2. **All T2 module integration tests for the phase pass**
3. **T3 mini-opt regression**: phase changes don't break `mini_optimization_run` baseline
4. **T4 per-phase full run** meets the phase's quantitative target (per phase table in Ã‚Â§10.2)
5. **No regression on previous phases' T4 metrics** (within Ã‚Â±5% noise band)
6. **Snapshot strip + diagnostics CSV produced for each phase's T4 run** (validates instrumentation lands)
7. **All Golden Rules touched by the phase have explicit references in the phase notes**

DoD is a hard gate: a phase is "done" only when 1-7 are true. Without 5, Phase N could silently break Phase N-1's checkpoint and not be detected until much later. Without 6, the thesis chapter loses the trajectory data for that phase. Without 7, code review can't audit which Rules were applied.

---

## Ã‚Â§10.8 Ã¢â‚¬â€ Lands-before-Phase-1 deliverables (consolidated)

Three things must exist before Phase 1 begins:

1. **`src_v2/instrumentation.py` Ã¢â‚¬â€ `DiagnosticsCallback`** (Ã‚Â§10.4). Per-gen CSV output. ~120 lines, ~3 hours.
2. **`src_v2/snapshot_callback.py` Ã¢â‚¬â€ `SnapshotCallback`** (Ã‚Â§10.5). 10 ordered snapshot PNGs + chromosome `.npy` + JSON metadata. ~180 lines, ~1 day with tests.
3. **`tests/fixtures/` Ã¢â‚¬â€ six shared fixtures** (Ã‚Â§10.6). ~6 days.

Total pre-Phase-1 investment: **~8 days**. This is a real cost. The alternative Ã¢â‚¬â€ building this scaffolding lazily during Phases 1-8 Ã¢â‚¬â€ costs ~20 days because every later phase re-authors fragments of the same scaffolding.

After this 8-day investment lands, Part 9's Ã‚Â§9.7 action items 1-3 (config edits, runner termination edit, sanity run) take ~2 hours total. Phase 1 starts on day 9.

---

## Ã‚Â§10.9 Ã¢â‚¬â€ Headline recommendations (drop-in replacement for the Ã‚Â§7 review summary)

1. **Refactor `PortPairCrossover` into two operators in Phase 4** (Coupling A). Without this, Part 9's Ã‚Â§9.1 ablation cannot be performed. Highest-priority structural change.
2. **Add the T3 mini-opt tier** to the testing pyramid. CI gate that doesn't exist in the plan today.
3. **Author the 6 shared fixtures during Phase 0/1**, not lazily during later phases. 6 days upfront saves ~20 days across Phases 1-8.
4. **Land the `DiagnosticsCallback` and `SnapshotCallback` BEFORE Phase 1**. Without observability, "feasibility +5pp" is the only quantitative signal Ã¢â‚¬â€ too coarse for thesis-grade analysis.
5. **Phase 5a runs in dual-evaluation mode** (Coupling E). Both old and new F[1] for one regression cycle.
6. **Ã‚Â§9.7 sanity run is a hard blocker** with formal abort criterion: feasibility Ã¢â€°Â¥500/1000 at `cx_prob=0.0` or do not proceed.
7. **Test Baldwinian `out["pheno"]` round-trip explicitly at start of Phase 1** (Test 1.6 / Coupling B). If pheno doesn't round-trip cleanly under multiprocessing, Rule 24 redesign is at risk.

---

**End of Part 10.** Plan now spans Parts 0-10 + Golden Rules 1-35 (with Rules 24-29, 31, 34 revised in Part 9). Implementation order: Phase 0 Ã¢â€ â€™ Part 10 deliverables (instrumentation, snapshots, fixtures) Ã¢â€ â€™ Ã‚Â§9.7 action items Ã¢â€ â€™ Phase 1.
