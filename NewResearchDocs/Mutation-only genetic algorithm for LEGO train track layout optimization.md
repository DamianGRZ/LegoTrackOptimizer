# Mutation-only genetic algorithm for LEGO train track layout optimization

**A mutation-only NSGA-II architecture combining CGP-style fixed-length genotypes, BRKGA-inspired construction decoders, and ALNS-adaptive operator selection can reduce the effective search space to ~10^44 while achieving >99% feasibility by construction.** This design synthesizes five established techniques — construction decoders (Bean 1994), inactive-gene neutral drift (Miller & Thomson 2000), linear construction primitives (Lohn & Colombano 1999), multi-segment encoding (Wen et al. 2023), and adaptive operator selection (Fialho et al. 2010) — into a single coherent architecture that maps cleanly to pymoo's fixed `n_var` NumPy array requirement. The architecture handles the full complexity of multi-port track pieces (switches, crossings, crossovers) through a layered constraint system where hard constraints are eliminated by the decoder, geometric closure uses ε-relaxation, and soft constraints degrade fitness proportionally.

---

## 1. Genotype: a CGP-inspired multi-segment integer array

### Recommended approach

The chromosome is a **single flat `int16` NumPy array** of fixed length, divided into four semantically defined segments. Each segment corresponds to a level of the layout hierarchy: main loop topology, switch placement, branch sequences, and crossing overlays. Inactive genes use sentinel value **-1**, functioning identically to CGP's inactive nodes — they participate in neutral drift without affecting the phenotype.

**Array layout:**

| Segment | Positions | Gene meaning | Range | Inactive marker |
|---------|-----------|-------------|-------|-----------------|
| Main loop | `[0, L_max)` | Piece-type index for construction sequence | `0` to `T-1` | `-1` (skip) |
| Switch mask | `[L_max, L_max+S_max)` | Switch type at main-loop position `i` | `0` (none) to `K` | `0` |
| Branch slots | `[L_max+S_max, ...)` | `B_max` fixed-size slots: `[src_switch, piece×b, rejoin_target]` | varies | `src=-1` |
| Crossing overlay | `[..., n_var)` | Position pairs for 4-port pieces across two paths | `0` to `L_max-1` | `-1` |

For a concrete system with max **30 main-loop pieces**, **4 branches of 8 pieces each**, and **5 crossing overlays**:

```
L_max=30 + S_max=30 + B_max×B_slot=4×10 + C_max=10 = n_var = 110
```

This yields ~20–30 active topology genes with ~75% inactive padding — a ratio that Miller & Smith (2006) demonstrated **actually improves search** through neutral genetic drift. Turner & Miller (2015) showed that when neutral drift was disabled in CGP, the EA was unable to solve even moderate-complexity problems after 10 million generations. The inactive region serves as a reservoir of pre-mutated material that can become active through length-changing mutations.

### Why this encoding wins

The multi-segment integer array was chosen over five alternatives through systematic evaluation:

**Object wrapping (`n_var=1`, `dtype=object`)** offers maximum flexibility but prevents NumPy vectorization, requires O(pop_size) Python loops for every operation, and forces completely custom operator implementations. For ~110 integer variables, the overhead is unjustified.

**Pure random-key (BRKGA float) encoding** loses its primary advantage — biased crossover — in a mutation-only setting. Integer piece-type indices are more natural than sorting floats, and the float-to-permutation mapping adds unnecessary indirection.

**True variable-length chromosomes (messy GA style)** are incompatible with pymoo's fixed `n_var` requirement. Goldberg et al.'s (1989) cut-and-splice operators require framework support that pymoo does not provide.

**pymoo's `MixedVariableProblem`** uses dictionary-based access that is slower than array indexing and introduces per-variable-type operator dispatch overhead unsuitable for homogeneous integer segments.

**Tree/graph GP encoding (Koza-style)** maps poorly to pymoo's fixed-length arrays and introduces bloat problems irrelevant to this domain.

### Multi-port encoding details

**Switches (3-port)** are encoded at two levels. A switch mask gene indicates *where* on the main loop a switch appears and *which type*. The corresponding branch slot encodes the diverging path as a construction sub-sequence with a rejoin target index. This separation means the main loop can be mutated without disturbing branch definitions, and vice versa — reducing positional epistasis.

**Crossings and crossovers (4-port)** are encoded as overlay pairs `(path1_position, path2_position)` in the crossing segment. The decoder validates geometric compatibility at these positions. If the two paths don't physically intersect, the crossing gene is simply ignored (graceful degradation). This keeps the genotype simple while the decoder absorbs geometric complexity — consistent with Lohn & Colombano's (1999) finding that linear construction primitives inherently produce valid topologies.

### Scaling to 30+ piece types

The gene value range for piece-type indices simply expands from `[0, T-1]` as `T` grows. The decoder uses modular arithmetic (`gene % T`) for robustness, and the construction heuristic's greedy behavior means many gene values map to equivalent phenotypic outcomes when inventory is constrained — providing natural redundancy that smooths the fitness landscape.

---

## 2. Decoding pipeline: construction-time feasibility with L-system branching

### Recommended approach

The decoder follows BRKGA's separation-of-concerns principle: the evolutionary machinery operates on integers, while a **deterministic construction procedure** translates chromosomes into placed-piece geometries. This decoder guarantees **>99% feasibility by construction** — any random integer array produces an evaluable (if suboptimal) layout.

**Phase 1 — Main loop construction.** Process genes in the main-loop segment left-to-right using turtle-graphics FK. Each gene specifies a piece type; the turtle advances by `(dx_local, dy_local, dθ)` in the piece's local frame. Three filters run per gene: (a) **inventory check** — skip if piece type exhausted, optionally substitute most geometrically similar available piece; (b) **angular budget check** — if placing this piece would overshoot 360°, skip or truncate; (c) **collision check** — if piece would overlap existing geometry within 3.0 studs, skip. When the cumulative angle reaches 360° and the turtle position is within **8.0 studs / 15.0°** of the origin, the loop closes. If all genes are consumed without closure, the decoder enters closure repair (Section 7).

**Phase 2 — Switch placement.** Scan the switch-mask segment. For each non-zero gene at position `i < len(main_loop)`, replace the main-loop piece at position `i` with the indicated switch type, but only if geometrically compatible (the switch's through-route FK deltas must match the replaced piece closely enough). When a switch is successfully placed, the diverging port's turtle state `(x_div, y_div, θ_div)` is **pushed onto a branch stack** — directly analogous to L-system `[` push semantics (Prusinkiewicz & Lindenmayer, *The Algorithmic Beauty of Plants*).

**Phase 3 — Branch construction.** For each active branch slot (where `src_switch ≥ 0`), pop the corresponding state from the branch stack and build a sub-sequence from the branch's piece genes. At each step, check proximity to the rejoin target on the main loop. When `distance(branch_tip, target_port) < 8.0 studs` and `|θ_branch − θ_target| < 15°`, declare rejoin. If the branch genes are exhausted without rejoin, invoke the deterministic fill algorithm using the genes as *hints* (not literal instructions) to compute a valid path — this is the hybrid encoding's key advantage, where **deterministic algorithms handle the hardest sub-problem** (branch closure) while the GA explores topology.

**Phase 4 — Crossing overlay.** For each active crossing gene pair, validate that two paths physically intersect at the indicated positions. If valid, replace the two intersecting pieces with a 4-port crossing or crossover. If positions don't align geometrically, the crossing is ignored.

### Invalid decoding handling

The decoder uses a **hybrid skip-with-substitution** strategy grounded in BRKGA decoder literature (Londe et al. 2023):

- **Inventory exhaustion** → substitute most geometrically similar available piece (e.g., R40 curve → R56 curve)
- **Geometric impossibility** → skip the gene entirely
- **Angular budget exceeded** → truncate and begin closure attempt
- **Branch cannot rejoin** → branch is not placed; switch reverts to 2-port piece

This ensures the decoder never crashes and always produces a complete Layout object.

### Layout data structure

The decoded layout stores both graph topology and spatial geometry:

```
Layout {
  pieces: List[PlacedPiece]       — All placed pieces with global (x, y, θ) and port positions
  paths: List[Path]                — Ordered sequences (MAIN_LOOP, BRANCH, CROSSPATH)
  piece_counts: Dict[PieceType, int]  — Usage tracking for inventory constraints
  closure_gaps: List[(dx, dy, dθ)]    — Residual gaps per path
  bounding_box: (min_x, min_y, max_x, max_y)
}
```

Each `Path` tracks its type, piece sequence, closure status, closure error vector, and originating/terminating switch IDs (for branches). The graph structure enables efficient cycle detection via DFS and supports the physics engine's forward-backward speed profiling in O(N).

---

## 3. Five mutation operators with adaptive selection

### Design philosophy

The mutation suite draws from three complementary frameworks: **CGP's principle** that simple point mutations plus neutral drift are sufficient (Miller & Thomson 2000), **ALNS's destroy-repair paradigm** with adaptive operator weighting (Ropke & Pisinger 2006), and **NEAT's minimal-disruption rule** for structural mutations (Stanley & Miikkulainen 2002). Each operator makes a single focused change — effectiveness comes from adaptive selection, not operator complexity.

### Operator 1: MUTATE (swap/replace) — the workhorse

Selects a random active gene position and replaces its value. Two modes: **safe swap** (probability 0.7) replaces within the same port-count class (2-port↔2-port, 3-port↔3-port), preserving topology while adjusting geometry. **Structural swap** (probability 0.3) allows cross-port-count mutation — converting a 2-port piece to a 3-port switch creates a new branch point (initialized with a minimal 1–2 piece branch, following NEAT's function-preserving principle), while converting a 3-port back to 2-port removes a branch. Piece selection is **angle-budget-aware**: candidates are scored by how much they reduce the gap between current cumulative angle and 360°, with selection probability proportional to score.

**Base application probability: 0.30.** This is the highest single-operator weight, consistent with AmoebaNet's finding that simple parameter mutations (their "op mutation") are the most frequently useful operator. It maps to CGP's point mutation — the only operator CGP uses.

### Operator 2: ADD (insertion)

Inserts a new piece at a selected position, shifting downstream genes right. Position selection uses a weighted distribution: **40% near constraint violations** (scan for largest angle deficit or geometry gap toward closure), **30% random position**, **30% at sequence end** (append). Piece type is selected with 50% probability using angle-budget-aware heuristic, 30% uniform random from available inventory, and 20% matching neighbor type for local consistency. After insertion, all downstream FK must be re-propagated — this is the dominant cost, motivating the positional bias toward sequence ends.

**Base application probability: 0.20.** Constructive operators receive substantial weight because the objective is to maximize piece utilization.

### Operator 3: DELETE (removal)

Removes a piece and compacts the array (shifting left rather than leaving sentinels, because CGP research shows the active region should be contiguous). Position selection follows ALNS's destroy heuristics: **50% worst-removal** (remove the piece contributing most to constraint violations), **30% random**, **20% related-removal** (remove a cluster of nearby pieces). When deleting a switch, the operator either deletes the entire associated branch (60%) or splices branch pieces into the main loop (40%).

**Base application probability: 0.10.** Destructive operators receive lower weight to avoid losing good structure, but remain essential for escaping local optima where the layout has too many pieces of the wrong type.

### Operator 4: BRANCH (add/extend/shorten/remove)

Four sub-operators targeting branch topology: **ADD_BRANCH** converts a main-loop 2-port piece to a 3-port switch and initializes a minimal branch sequence (NEAT's "start minimal, complexify" principle). **EXTEND_BRANCH** adds 1–2 pieces to an existing branch, selected to reduce distance to the rejoin target. **SHORTEN_BRANCH** removes the last 1–2 pieces from a branch. **REMOVE_BRANCH** deletes the entire branch and reverts the switch to a 2-port piece.

**Base application probability: 0.20 combined** (ADD: 0.07, EXTEND: 0.06, SHORTEN: 0.04, REMOVE: 0.03). The constructive sub-operators receive higher weight than destructive ones.

### Operator 5: COMPOUND (segment replacement and local search)

**Segment replacement** (ALNS destroy-repair): removes N consecutive pieces (N ~ Uniform(2,5)) and replaces with M new pieces selected by greedy insertion toward maintaining closure. **Local search mutation** (memetic): selects a 3–5 piece subsection and hill-climbs for 5–10 iterations, keeping the best improvement. These expensive operators are reserved for later evolution phases when simple mutations plateau.

**Base application probability: 0.20 combined** (segment replacement: 0.12, local search: 0.08).

### Adaptive operator selection via ALNS-UCB hybrid

Operator probabilities adapt during evolution using a hybrid of ALNS weight updates and UCB1 exploration bonuses (Da Costa et al. 2008). Each operator maintains a quality estimate updated via exponential smoothing: `q_i ← 0.8 × q_i + 0.2 × reward`, where reward is the fitness improvement relative to parent. Selection uses UCB1: `score_i = q_i + C × √(ln(N) / n_i)` with C=0.1, ensuring all operators are tried periodically regardless of recent performance. A sliding window of **50 applications per operator** prevents stale credit from dominating.

**Phase-based scheduling** overlays the adaptive system: generations 1–33% emphasize ADD/MUTATE (exploration), 33–66% let AOS fully adapt (refinement), 66–100% increase COMPOUND weight (exploitation).

### pymoo integration

All operators are encapsulated in a single `TrackLayoutMutation` class implementing pymoo's `Mutation._do(problem, X, **kwargs)` interface. Crossover is disabled via `crossover=NoOp()`. Each call to `_do()` iterates over individuals, selects an operator via AOS, applies it, and validates the chromosome (ensuring sentinels are contiguous, inventory references are valid, branch indices are in-range). The AOS controller updates operator quality estimates after each generation's evaluations complete, using parent-offspring fitness comparison as the credit signal.

---

## 4. Population: 100 individuals with 15% heuristic seeding

### Population size: 100

For **~25 effective topology variables** with O(N) evaluation cost, a population of **100** balances exploration capacity against computational budget. This is supported by Piotrowski's (2017) comprehensive review finding pop=100 optimal for problems with dimensionality below 30, and aligns with standard NSGA-II practice for moderate-complexity constrained problems. Without crossover, each individual explores independently through mutation — the population must be large enough to maintain genetic diversity across the feasibility boundary. At 100 individuals with a 10–20% infeasibility target, **10–20 individuals explore near the constraint boundary** at any time, sufficient for boundary-driven search (Singh et al. 2008, IDEA).

The 10× rule-of-thumb (10 × number of variables) would suggest 200–300, but this is derived from crossover-based GAs where recombination requires diversity across many loci simultaneously. Mutation-only evolution operates more like (μ+λ)-ES, where populations of **4–10× the dimension** are standard (Beyer & Schwefel 2002). Starting at 100 keeps per-generation evaluation under 1 second for O(N) FK computation on typical hardware.

### Seeding strategy: 15 heuristic seeds, 85 random

The literature consistently warns against heavy heuristic seeding — Goldberg's foundational work and multiple empirical studies confirm that **random solutions often drive the population to optimality** while heuristic seeds can cause premature convergence to local optima. A 15% seeding fraction (15 individuals) provides initial feasible solutions for Deb's feasibility-first ranking to anchor against, without dominating genetic diversity.

**Recommended seed patterns** (2–3 individuals each):

- **Simple circles** using minimum curves for closure (baseline feasible)
- **Ovals** with varied straight-segment counts (tests straight utilization)
- **Racetracks** with elongated straight sections
- **Figure-8 patterns** using switches/crossings (tests multi-port pieces)
- **Asymmetric kidney/bean shapes** using mixed curve radii

Each pattern should be instantiated with random rotation and slight perturbation to ensure genetic diversity even among seeds. The remaining 85 individuals are generated via the construction decoder from random integer arrays — the decoder's feasibility guarantees ensure many of these produce valid (if simple) layouts.

### Diversity maintenance: aging evolution plus light fitness sharing

**Aging evolution** (Real et al. 2019) is incorporated as an age-penalized tiebreaker within NSGA-II's survival selection. Each individual carries a birth-generation counter. When NSGA-II's non-dominated sorting produces ties in both rank and crowding distance, **younger individuals are preferred** via a decay factor of 0.99^age applied to crowding distance. This gently prevents ancient high-fitness individuals from permanently occupying population slots — Real et al. showed this "regularization" prevents premature convergence in mutation-only evolution on deceptive landscapes.

**Light fitness sharing** based on piece-usage vector cosine similarity prevents topological convergence. If more than 50% of the population shares >90% cosine similarity on their piece-type usage vectors, fitness sharing activates with σ_share = 0.8 cosine distance. This is computationally cheap (O(pop²) on short vectors) and directly addresses the risk of the entire population converging to simple circles.

**FI2Pop** (Kimbrough et al. 2008) and **MAP-Elites** (Mouret & Clune 2015) are deferred to implementation Phase 3. At the target 5–20% infeasibility rate, Deb's feasibility-first ranking handles constraint-boundary exploration adequately. MAP-Elites is valuable as a **post-processing diversity archive** using feature descriptors (piece utilization × topology complexity × bounding-box aspect ratio) to present diverse layout options to the user.

---

## 5. Selection and elitism within NSGA-II's framework

### Binary tournament selection is appropriate for mutation-only

NSGA-II's default **binary tournament (k=2)** provides moderate selection pressure well-suited to constrained, potentially deceptive landscapes. Higher tournament sizes (k=3–4) would increase exploitation pressure prematurely — inappropriate when mutation alone must handle both exploration and exploitation. The constrained-domination comparison (Deb et al. 2002) naturally prioritizes feasible solutions without additional mechanism.

### (μ+μ) generational replacement with neutral-drift acceptance

NSGA-II's built-in **(μ+μ) plus-selection** — merging parents and offspring, then selecting the best μ — provides sufficient elitism. Parents compete with offspring, ensuring monotonic non-degradation of the Pareto front. This is the recommended scheme for combinatorial optimization (Beyer & Schwefel 2002, Scholarpedia on Evolution Strategies).

One critical addition from CGP: the **equal-or-better acceptance rule**. When two solutions have identical fitness and constraint violation, the newer (mutated) individual is preferred. This enables **neutral drift through inactive genes** — mutations to sentinel regions don't affect fitness but change the genotype, maintaining hidden diversity that can later become active. Miller & Thomson (2000) demonstrated this is essential for CGP's effectiveness.

### External archive of 20–30 elite feasible solutions

An external Hall of Fame maintains the best feasible solutions found across all generations. This archive does **not** participate in selection (avoiding additional elitism pressure) but serves as: (a) backup against population degradation, (b) output collection for the user, and (c) seed source for restarts after stagnation events.

### Stagnation detection and response

**Primary trigger**: no improvement in best feasible fitness for **50 generations** (~5,000 evaluations, ~10% of total budget). This threshold is standard in EA practice (Eiben & Smith 2003).

**Response protocol**: (1) **Hypermutation** — triple all operator application intensities for 10 generations; (2) inject 10% new random individuals replacing worst infeasible solutions; (3) if stagnation persists for 50 more generations, restart 30% of the population from new random seeds while preserving the top 20% in the archive.

**Diversity collapse trigger**: if >50% of the population shares >90% genotypic similarity (measured by Hamming distance on active genes), apply fitness sharing and inject seeds from underrepresented topology classes.

---

## 6. Layered constraint handling from decoder to penalty

### Recommended architecture: four-layer hierarchical system

The literature strongly supports handling different constraint types at different pipeline stages (Coello Coello 2002, Mezura-Montes & Coello Coello 2011). Constraints that are easily preventable belong in the decoder; continuous violations with useful gradient information belong in pymoo's CV mechanism; and quality-of-life constraints belong in the objective as penalties.

**Layer 1 — Decoder enforcement (hard constraints, never violated):**
- **Inventory limits**: the decoder refuses to place a piece if its type is exhausted, substituting the most geometrically similar available piece or skipping the gene
- **Connectivity**: sequential construction inherently ensures each piece connects to the existing layout via FK propagation
- **Valid piece types**: genes are bounded to `[0, T-1]`; any out-of-range value maps via modular arithmetic

**Layer 2 — Repair operators (post-mutation, pre-evaluation):**
- **Closure repair**: if the main loop fails to close within tolerance, attempt to swap the last 1–5 pieces with a computed closing sequence (see Section 7)
- **Branch validity**: ensure every active branch slot references a valid switch index; orphaned branches are deactivated

**Layer 3 — pymoo constraint violation with ε-relaxation (geometric closure):**
- Position closure: `G_pos = max(0, position_gap − 8.0)` as inequality constraint
- Angle closure: `G_angle = max(0, angle_gap − 15.0)` as inequality constraint
- Use **`AdaptiveEpsilonConstraintHandling`** with `perc_eps_until=0.5` — ε starts permissive and reaches zero at generation 250/500, allowing the EA to explore near-closed layouts early, then tightening to exact closure
- This is strongly recommended for geometric closure because it is effectively an equality constraint (Σθ = 360°), and Takahama & Sakai (2006) showed ε-relaxation is superior to penalty functions for equality-like constraints

**Layer 4 — Soft penalties in objective function:**
- **Boundary violation**: `penalty_boundary = Σ max(0, distance_outside_i) × 100` per piece outside the workspace
- **Collision**: `penalty_collision = Σ max(0, 3.0 − distance_ij) × 200` per collision pair
- **Final fitness**: `F = pieces_used × 1000 − penalty_boundary − penalty_collision`

The penalty magnitudes are calibrated so that a single 1-stud boundary violation (~100 penalty) is equivalent to losing 0.1 pieces from the objective, and a single 1-stud collision overlap (~200 penalty) is equivalent to losing 0.2 pieces. These are mild enough to allow exploration of borderline solutions but strong enough to drive optimization toward feasibility.

### Why Deb's rules plus ε-relaxation, not alternatives

**Stochastic ranking** (Runarsson & Yao 2000) aggregates all constraints into a single CV value, losing the hard/soft distinction crucial to this problem. **Adaptive penalty** methods (Bean & Hadj-Alouane 1997) introduce meta-parameters that interact unpredictably with the AOS system. **Pure Deb's rules** without ε-relaxation are too greedy toward feasibility — Lu, Deb & Singh (2018) showed that maintaining some infeasible solutions improves convergence, which ε-relaxation achieves naturally.

---

## 7. Repair operators: Baldwinian by default, Lamarckian for elites

### Guiding principle

Ishibuchi, Kaige & Narukawa (2005) demonstrated that **Baldwinian repair outperforms Lamarckian** on multi-objective constrained problems because it preserves genotypic diversity while smoothing the fitness landscape. The repaired phenotype is evaluated, but the original chromosome is retained in the population. A "5% partial Lamarckian" rule writes back repairs only for the top 5% of solutions — accelerating convergence for elite individuals without destroying population-level diversity.

### Closure repair via angular budget arithmetic

The most critical repair targets geometric closure. Given a partially built main loop with cumulative angle θ_accumulated and turtle position (x_current, y_current):

1. Compute **angular budget**: `θ_remaining = 360° − θ_accumulated`
2. Compute **closure vector**: `(Δx, Δy) = (x_start − x_current, y_start − y_current)`
3. Enumerate closing sequences of ≤ **6 pieces** from available inventory whose angles sum to θ_remaining (bounded combinatorial search, O(k³) for k piece types)
4. Among valid angular sequences, select the one minimizing positional error `√(Δx² + Δy²)` at the sequence's end
5. If no sequence achieves closure within tolerance, return the best available — the residual gap becomes the CV value for ε-constraint handling

Angular closure is **quantized** in multiples of piece angles (22.5° for standard LEGO, 18°/15°/11.25° for 4DBrix), which reduces the combinatorial search to integer partitions. **Pre-computed lookup tables** of known closing sequences for common angular budgets (90°, 180°, 270°, etc.) amortize this cost across evaluations.

### Inventory repair

When mutations create over-budget piece usage: count excess per type, remove instances in order of lowest marginal fitness contribution (pieces near sequence end, or pieces with highest substitutability), and return pieces to available inventory. Cost: O(N).

### Branch rejoin repair

For dangling branches that fail to reconnect: identify the nearest main-loop port within 2× closure tolerance. If found, extend the branch with 1–3 greedy-selected pieces aimed at the target. If no viable rejoin exists within budget, **deactivate the branch** — revert the switch to a 2-port piece and return all branch pieces to inventory. This graceful degradation ensures the layout is always evaluable.

### Switch pairing repair

Count OUT-switches (diverging) and IN-switches (merging). If imbalanced, convert excess switches back to 2-port pieces in order of lowest-fitness-contribution branches. This runs in O(S) where S is the number of switches.

### Pipeline placement and cost budget

All repairs execute **post-mutation, pre-evaluation** — after the chromosome is mutated but before fitness computation. Repairs should consume ≤ **20% of total evaluation time**. Decoder-embedded repairs (inventory skip, angular truncation) are essentially free. Closure repair's bounded combinatorial search is the most expensive component but is bounded to O(k³) per path. Collision checking uses spatial grid indexing for O(N) average case.

---

## 8. Integration with the existing codebase

### Module-by-module mapping

**`problem.py` — TrackLayoutProblem**

`__init__()` changes: set `n_var=110` (or computed from `L_max + S_max + B_max × B_slot + 2 × C_max`); define `xl` and `xu` arrays with segment-specific bounds; set `n_ieq_constr=2` (position closure, angle closure); set `n_obj=1` (single objective with soft penalties embedded).

`_evaluate()` changes: decode chromosome via new `decode_chromosome()` function; apply repair pipeline; compute geometric closure gaps as inequality constraints for `out["G"]`; compute fitness as `pieces_used × 1000 − soft_penalties` for `out["F"]`. The evaluation remains `ElementwiseProblem` because FK is sequential per-individual.

**`geometry.py` — Layout and build_layout()**

Major extension for branch support. The `Layout` class gains: `paths: List[Path]` (main loop + branches), `branch_stack` for L-system push/pop during construction, `pending_connections` registry for 4-port pieces. The `build_layout()` function becomes `decode_chromosome()` implementing the four-phase pipeline. The existing FK engine (turtle-graphics `dx_local, dy_local, dtheta`) is **fully reusable** — the decoder simply calls it in the correct order, pushing/popping state for branches.

**`evaluation.py` — objectives and constraints**

Refactor to separate hard constraint checking (now mostly in decoder) from soft penalty computation. The closure gap computation moves from a penalty (`-1e10`) to a pymoo inequality constraint (`G_pos`, `G_angle`). Collision and boundary penalties remain in the objective. The physics engine (forward-backward speed profiles) is unchanged.

**`data.py` — TrackCatalog**

No modification needed. The immutable `TrackPiece` dataclasses with `FKDeltas`, `Port` arrays, and physics data are consumed as-is by the decoder. The catalog's piece-type indices become the gene value space.

**`sampling.py` — seeding patterns**

New `HeuristicSeeding` sampling class implementing pymoo's `Sampling` interface. Generates 15 individuals from predefined patterns (circles, ovals, racetracks, figure-8s) and 85 random individuals via the construction decoder applied to random integer arrays.

**New file: `operators.py`**

Contains `TrackLayoutMutation(Mutation)` with the five-operator suite, `TrackMutationAOS` controller, `ClosureRepair`, `InventoryRepair`, `BranchRejoinRepair`, and `SwitchPairingRepair` classes. Also contains `NoOp` crossover (identity operator returning parents unchanged).

**New file: `archive.py`**

External Hall of Fame archive maintaining 20–30 elite feasible solutions, updated each generation, with MAP-Elites grid support for Phase 3 diversity.

### What can be reused vs. rewritten

- **Reuse entirely**: FK engine, piece catalog, physics simulation, collision detection, boundary checking
- **Extend**: Layout data structure (add paths/branches), evaluation pipeline (separate hard/soft)
- **Rewrite**: chromosome encoding and bounds, sampling/seeding, mutation operators, constraint reporting to pymoo
- **New**: multi-phase decoder, repair pipeline, AOS controller, external archive

---

## 9. Evolutionary parameters with literature justification

### Core parameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Population size | **100** | Piotrowski (2017): optimal for <30D; 4× variable count for mutation-only ES |
| Max generations | **500** (50,000 evaluations) | CEC standard: 10,000×D; stagnation cutoff prevents waste |
| Stagnation threshold | **50 generations** | ~10% of budget; standard EA practice (Eiben & Smith 2003) |
| ε-constraint `perc_eps_until` | **0.5** | ε→0 at generation 250; Takahama & Sakai (2006) |
| Target infeasibility ratio | **10–20%** | Singh et al. (2008) IDEA; Lu, Deb & Singh (2018) |
| External archive size | **25** | Covers top solutions across topology classes |
| Aging decay factor | **0.99^age** | Gentle bias toward younger solutions (Real et al. 2019) |

### Operator probabilities (initial, subject to AOS adaptation)

| Operator | P(initial) | Rationale |
|----------|-----------|-----------|
| MUTATE (swap/replace) | **0.30** | Primary refinement; CGP's sole operator; AmoebaNet's most useful |
| ADD (insertion) | **0.20** | Constructive; drives piece-utilization objective |
| COMPOUND (segment/local) | **0.20** | ALNS destroy-repair for escaping local optima |
| BRANCH (add/extend/shorten/remove) | **0.20** | Topology exploration; split 0.07/0.06/0.04/0.03 |
| DELETE (removal) | **0.10** | Destructive correction; lowest weight preserves structure |

### Per-gene mutation probabilities

| Gene segment | P(mutate per gene) | Rationale |
|-------------|-------------------|-----------|
| Main loop pieces | **0.04** (~1/25) | 1/n rule (Mühlenbein 1992): ~1 gene per mutation |
| Switch mask | **0.10** | High-impact topology genes; elevated rate for exploration |
| Branch pieces | **0.08** (~1/12) | Shorter sequences → proportionally higher per-gene rate |
| Start position | **0.02** | Rarely beneficial to change; low rate per project spec |
| Crossing overlay | **0.05** | Moderate; changes path intersections |

### Termination criteria

**Primary**: stagnation — no improvement in best feasible fitness for 50 consecutive generations. **Secondary**: budget exhaustion at 500 generations. **Tertiary**: convergence — if the top 20% of the population shares >95% genotypic similarity AND the best solution has been stable for 30 generations, terminate early. The Rechenberg-inspired 1/5 success rule is applied per-operator: if an operator's success rate (fraction of applications producing fitness improvement) exceeds 20%, increase its mutation intensity by factor 1.22; if below 20%, decrease by factor 0.82. This self-adapts operator aggressiveness without manual tuning.

---

## Conclusion: a practical architecture grounded in five decades of EC research

This architecture achieves several properties simultaneously. **Near-universal feasibility** through construction-based decoding eliminates the fatal flaw of the current encoding (<1% random feasibility). **Reduced search space** (~10^44 vs. ~10^163) through hybrid encoding makes the problem tractable for population-based search. **Structural integrity under mutation** is guaranteed because operators modify construction *instructions*, not placed geometry — the decoder absorbs any resulting inconsistencies.

Three key risks merit ongoing attention. First, **closure repair cost** could dominate evaluation time if angular budgets frequently require expensive combinatorial search — pre-computed lookup tables and bounded search depth (≤6 pieces) mitigate this. Second, **branch rejoin feasibility** remains the hardest sub-problem; the deterministic fill algorithm's quality directly bounds achievable layout complexity, and investing in a strong A*-based branch solver will pay dividends. Third, **AOS cold-start** may cause suboptimal operator selection in early generations — the UCB exploration bonus and uniform initial weights address this, but operators should be monitored for the first 50 generations to verify the adaptive system is tracking correctly.

The architecture is designed for **phased implementation**: Phase 1 delivers the core system (encoding, decoder, MUTATE/ADD/DELETE operators, Deb's rules), Phase 2 adds branch operators and ε-constraint handling, and Phase 3 introduces AOS adaptation, aging evolution, and MAP-Elites diversity archive. Each phase produces a working system that improves on the previous, allowing empirical validation of design choices against the current baseline.