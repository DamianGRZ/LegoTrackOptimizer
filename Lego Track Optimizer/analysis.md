# V2 Optimizer — Diversity & Complexity Plan

**Goal.** Push NSGA-II away from the simple-oval attractor and toward a wide,
topologically-diverse Pareto front in which switches, CROSS_90s, and
DOUBLE_CROSSOVERs do real work (route divergences, parallel tracks, branching
loops) instead of appearing as decoration.

**Scope.** Soft pressure (keep 2 objectives), moderate-sized change: new
heuristic emitters + new mutation sub-ops + canonical graph-hash diversity +
topology-niched survival. No new objectives, no per-route physics rewrite.

---

## 1. Diagnosis — why current behavior is what it is

Reading `operators.py`, `repair.py`, `runner.py`, and `problem.py` together,
five mechanisms are individually reasonable but combine to collapse the
population onto simple ovals:

### 1.1 Heuristic seeds are dominated by simple loops
`_HEURISTIC_EMITTERS` produces, in expected count for a typical inventory:

| Emitter                       | Variants emitted | Uses switches? | Uses CROSS_90? | Uses DOUBLE_CROSSOVER? |
|-------------------------------|------------------|----------------|----------------|------------------------|
| `_emit_simple_loop`           | 2                | no             | no             | no                     |
| `_emit_oval`                  | up to 6          | no             | no             | no                     |
| `_emit_racetrack`             | up to 4          | no             | no             | no                     |
| `_emit_simple_oval_with_siding` | 1              | yes (2)        | no             | no                     |
| `_emit_figure_8`              | up to 4          | no             | yes (1)        | no                     |
| `_emit_multi_loop`            | 1                | no             | no             | no                     |
| `_emit_dense_crossing_grid`   | 1                | no             | yes (2)        | no                     |

So out of ~19 patterns, **only 6 use any complex piece, and zero use
DOUBLE_CROSSOVER.** With `heuristic_ratio=0.30`, that's ~94% of seeds being
plain loops/ovals — a strong attractor that crowding distance can't escape.

### 1.2 Mutations have no structural moves toward complexity
`PortPairMutation.OP_WEIGHTS` is `[0.20, 0.15, 0.10, 0.20, 0.10, 0.15, 0.10]`
across `[mutate_piece_type, activate_slot, deactivate_slot, add_edge,
remove_edge, rewire_edge, perturb_anchor]`. Every one of these is **local**
(touches one slot or one edge). None of them can:

- Insert a switch pair to fork a loop
- Insert a CROSS_90 to join two components
- Insert a DOUBLE_CROSSOVER to bridge parallel tracks
- Excise a switch and rewire cleanly

The only structural mutation is `introduce_crossing`, **which lives in repair**
and only fires when the FK chain is *already* near-perpendicular at a
self-intersection. Empty-field, switch-rich seeds rarely make it that far.

### 1.3 `eliminate_duplicates` is too lenient
`NSGA2(..., eliminate_duplicates=True)` uses raw `np.array_equal` on int16
chromosomes. Two layouts that are **structurally identical** but differ by
even one stud of anchor or one slot index permutation count as distinct.
Result: the population fills with anchor-shifted clones of the same oval.

### 1.4 Crowding distance is computed in objective space only
`(utilization, min_speed)` is a 2D space. With only ~3 distinct (util, speed)
clusters reachable by simple loops, NSGA-II's secondary criterion (crowding
distance in F-space) cannot distinguish a figure-8 from an oval at the same
(util, speed). It picks one, drops the other.

### 1.5 Constraint formulation under-penalizes loose ports
`G[5+T] = n_loose_ports / total_active_ports`. A 30-piece simple loop has 0
loose ports. A 30-piece **figure-8 in progress** (one CROSS_90 + 16 R40 lobe
+ 13 R40 partial second lobe) might have 1–2 loose ports → constraint value
~0.03 → epsilon-handling treats it as "almost feasible" → it's pruned by the
oval that *is* feasible at the same utilization. Your stated requirement
("must be connected, no loose ports") matches this — we should make it harder.

### 1.6 The "useful component" gate cuts both ways
`MIN_USEFUL_COMPONENT_SIZE = 4` correctly rules out 2/3-piece junk loops, but
it also means a 4-piece component nestled next to a 16-piece loop now
counts toward utilization. The GA learns to spawn small companion loops to
inflate util — pretending to be "complex" without being interesting.

---

## 2. Plan

Five changes, ordered by expected impact. Each is independently testable.

### 2.1 Heuristic seed expansion (high impact, low risk)

**File:** `operators.py`

Add five emitters and rebalance:

| New emitter                       | What it produces                                                      | Uses                                  |
|-----------------------------------|------------------------------------------------------------------------|---------------------------------------|
| `_emit_dogbone`                   | Two-loop end-caps connected by a switch-bounded straight section      | 2 switches, R40s, straights           |
| `_emit_double_oval_crossover`     | Two parallel ovals joined by a DOUBLE_CROSSOVER                        | 1 DOUBLE_CROSSOVER, R40s, straights   |
| `_emit_ladder`                    | N parallel straights joined by repeated DOUBLE_CROSSOVERs              | k DOUBLE_CROSSOVERs, straights        |
| `_emit_yard`                      | Mainline with 2–3 sidings (switch ladder)                              | 4–6 switches                          |
| `_emit_tri_lobe_crossing`         | Three R40 lobes meeting at one CROSS_90                                | 1 CROSS_90, R40s                      |
| `_emit_complex_combo` (composite) | Random combination of two simpler emitters with components linked      | mix                                   |

Plus rebalance: bump `heuristic_ratio` default to 0.5 and add a
`complex_emitter_quota` so simple loops/ovals never exceed 30% of the
heuristic share even when more complex emitters fail their inventory checks.

### 2.2 Structural mutation operators (highest impact)

**File:** `structural_mutations.py` (extend) + `operators.py` (wire in)

Three new graph-surgery ops, each callable from `PortPairMutation`:

- **`split_with_switch`** — pick an active edge `(slot_a, port_a) ↔ (slot_b, port_b)`
  on a straight section; allocate two free slots for an `IN`+`OUT` switch pair;
  rewire the edge through them; allocate a small branch chain (1–4 straights or
  curves) on port C. Promotes any loop into a loop-with-siding.

- **`insert_crossover_bridge`** — find two roughly-parallel edges on different
  components (or the same component); allocate a DOUBLE_CROSSOVER; rewire so
  track1 carries one edge, track2 carries the other, and the two cross routes
  link the components. This is the only way the GA will discover
  parallel-tracks-with-crossover layouts without seeding them all.

- **`merge_components_via_cross`** — like `introduce_crossing`, but doesn't
  require a perpendicular self-intersection. Picks one edge from each of two
  components, allocates a CROSS_90 between them, rewires. Critical for
  joining heuristic-seeded sub-loops into one big multi-loop topology.

Add to `OP_WEIGHTS` with weights `[0.10, 0.05, 0.05]` and re-normalize. Keep
the local ops dominant — these are exploratory, not the workhorse.

### 2.3 Canonical graph hash for true duplicate elimination

**Files:** `repair.py` (new helper) + `runner.py` (custom duplicate handler)

Compute a hash that's invariant to:
- Slot-index permutation (the slot region is positional but topology isn't)
- Anchor pose
- Edge ordering within port-pair rows
- Edge endpoint ordering (`(a,b) == (b,a)`)

Algorithm: relabel slots in BFS order from a canonical root (lowest-degree,
lowest piece-type-id tiebreaker), serialize as
`(piece_id_sequence, sorted edge tuples)`, hash. This is the standard
graph canonicalization trick — runs in O(V + E log E).

Pass it to NSGA-II via a custom `pymoo.core.duplicate.ElementwiseDuplicateElimination`
subclass.

### 2.4 Topology-niched crowding (the core diversity lever)

**File:** `runner.py`

Define a per-individual **topology signature**:

```python
sig = (n_components, n_cycles, n_switches, n_crosses, n_crossovers)
```

Two options, in order of preference:

1. **Lightweight:** wrap `ConstrRankAndCrowding` with a survival operator that,
   within each non-dominated front, prefers individuals from
   under-represented signature buckets. ~80 lines, no new pymoo machinery.

2. **Heavier (only if option 1 doesn't work):** swap NSGA-II for NSGA-III with
   a reference-direction set built from the signature space. More invasive.

Start with option 1.

### 2.5 Constraint tightening for loose ports

**File:** `problem.py`

Replace the soft normalization with a step penalty: anything with
`n_loose_ports > 0` becomes meaningfully infeasible.

```python
# Before:
G.append(graph.n_loose_ports / max(1, total_active_ports))

# After:
G.append(graph.n_loose_ports)  # any loose port is infeasible
```

This is what the user actually wants ("there are no loose ports"). It will
hurt feasibility rate at first, which is fine — adaptive epsilon will absorb
the early-generation pain, and by the time epsilon clamps to 0 the operators
above will be producing properly-closed multi-component layouts.

---

## 3. What I am NOT changing

- **Objectives.** Stays at `(-utilization, -min_speed)`. No third objective.
- **Per-route physics.** Switches still report their through-route speed.
- **`MIN_USEFUL_COMPONENT_SIZE`.** Stays at 4. (Could revisit if 4-piece
  companion loops turn out to be a problem after the changes above.)
- **Adaptive epsilon schedule.** Stays as-is.
- **Encoding layout.** Stays as-is — int16 `[slots | pairs | anchor]`.
- **Decoder.** Stays as-is.

---

## 4. Test plan after edits

Each change has a clear before/after metric. After implementing all five:

| Metric                                    | Before (expected)   | After (target)      |
|-------------------------------------------|---------------------|---------------------|
| Final pop: distinct topology signatures   | 2–3                 | ≥ 8                 |
| Final pop: % using ≥1 switch              | < 10 %              | ≥ 50 %              |
| Final pop: % using ≥1 CROSS_90            | < 5 %               | ≥ 30 %              |
| Final pop: % using ≥1 DOUBLE_CROSSOVER    | 0 %                 | ≥ 15 %              |
| Final pop: max utilization                | should not regress  | within 5 %          |
| Final pop: max min-speed                  | should not regress  | within 5 %          |
| Loose-port count on feasible solutions    | typically 0–2       | strictly 0          |

Run `runner.run_optimization(...)` with the same `OptimizationConfig` you've
been using (whichever your validation case is) and pull the metrics from
`res.pop`.

---

## 5. Order of execution

1. **2.5** (loose-port tightening) — 3 lines, immediate effect.
2. **2.3** (canonical hash) — unblocks everything else by exposing real
   diversity loss.
3. **2.1** (heuristic seeds) — gives the GA something to crossover from.
4. **2.2** (structural mutations) — gives the GA something to evolve into.
5. **2.4** (topology niching) — preserves the diversity once it exists.

This ordering means each step is testable on its own.

---

## Open questions before I edit

1. **Is `heuristic_ratio = 0.5` acceptable**, or do you want to keep it lower
   (current default 0.20 in config, runner reads it from config)?
2. **For the niching survival op**, is it OK to bias selection by topology
   signature *within* fronts (option 1), or do you want the cleanest possible
   pymoo integration even at the cost of code volume (option 2)?
3. **Do you want me to keep `sampling.py`** (the dead V1 file) or delete it
   while I'm in there?

If those are all "decide for me," I will. Tell me to proceed and I'll execute
the plan top-to-bottom.
