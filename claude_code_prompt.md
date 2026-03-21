# Claude Code Prompt: Update GA System to Multi-Segment Architecture

## Context

You are working on a LEGO train track layout optimizer that uses pymoo's NSGA-II to evolve closed railway layouts. The project repository contains both Python source code (under `src/`) and documentation (`.md` files at the project root). Two new research documents have been added that supersede parts of the original design:

- `NewResearchDocs/Mutation-only_genetic_algorithm_for_LEGO_train_track_layout_optimization.md` — defines a new multi-segment encoding, 4-phase construction decoder, 5 mutation operators with adaptive selection, ε-constraint handling, and a phased implementation roadmap
- `NewResearchDocs/Crossover_operators_for_multi-segment_LEGO_track_genomes.md` — defines segment-selective BRKGA-style crossover that preserves the main loop from the elite parent while recombining topology segments

The older documentation files that describe the **current** implementation are:
- `docs/technical_design.md` — current module architecture, class structures, encoding scheme, pymoo integration
- `docs/implementation.md` — current optimization flow, operator setup, evaluation loop

Read all four documents before proceeding.

---

## What Has Changed: Current vs. New Design

### Encoding (MAJOR REWRITE)

**Current**: Flat `float64` array of length `total_inventory + 2`. Each gene is a piece index (-1 to 9) or a start position float. No semantic segmentation. `n_var = total_inventory + 2`.

**New**: Fixed-length `int16` multi-segment array with four semantic regions:
1. **Main loop** `[0, L_max)` — piece-type indices for construction sequence, -1 = inactive/skip
2. **Switch mask** `[L_max, L_max+S_max)` — which main-loop positions get switches and what type
3. **Branch slots** `[L_max+S_max, ...)` — B_max fixed-size slots of `[src_switch, piece×b, rejoin_target]`
4. **Crossing overlay** `[..., n_var)` — position pairs for 4-port pieces

### Decoder (NEW — replaces `build_layout()`)

**Current**: `build_layout()` does simple sequential FK. Every gene directly places a piece.

**New**: 4-phase construction decoder `decode_chromosome()`:
1. **Main loop construction** — turtle-graphics FK with inventory check, angular budget check, collision check per gene. Skips invalid genes. Closes when cumulative angle ≈ 360° and position ≈ origin.
2. **Switch placement** — scans switch mask, replaces compatible main-loop positions with switches, pushes diverging port state onto branch stack (L-system semantics).
3. **Branch construction** — builds sub-sequences from branch slot genes, checks proximity to rejoin target, invokes deterministic fill if genes exhausted without rejoin.
4. **Crossing overlay** — validates geometric intersection of path pairs, replaces with 4-port pieces if valid.

The decoder guarantees >99% feasibility by construction — any random integer array produces an evaluable Layout. This is the core architectural change.

### Operators (MAJOR REWRITE)

**Current**: Standard pymoo `SBX` crossover + `PM` mutation + simple `TrackRepair`.

**New — Mutation operators** (in a single `TrackLayoutMutation(Mutation)` class):
1. **MUTATE** (p=0.30) — swap/replace a random active gene, angle-budget-aware candidate selection
2. **ADD** (p=0.20) — insert piece, shift downstream genes right, positional bias toward violations/end
3. **DELETE** (p=0.10) — remove piece, compact array, ALNS destroy heuristics (worst/random/related)
4. **BRANCH** (p=0.20) — four sub-operators: add/extend/shorten/remove branch
5. **COMPOUND** (p=0.20) — segment replacement (ALNS destroy-repair) + local search mutation

**New — Adaptive Operator Selection (AOS)**: ALNS-UCB hybrid. Quality estimate per operator via exponential smoothing. Selection via UCB1. Phase-based scheduling overlay (early: ADD/MUTATE emphasis; mid: full AOS; late: COMPOUND emphasis).

**New — Crossover** (segment-selective BRKGA-style):
- Main loop: ρ=1.0 (always from elite parent — no crossover)
- Switch mask: parameterized uniform crossover, ρ=0.7
- Branch slots: slot-level (atomic) uniform crossover, ρ=0.6–0.7
- Crossing overlay: pair-level uniform crossover, ρ=0.6
- P_c = 0.3–0.4, kept at fixed rate separate from AOS
- BRKGA mutant injection: 10% random individuals replace worst each generation

### Constraint Handling (RESTRUCTURE)

**Current**: 5 inequality constraints reported via `out["G"]`, 2 objectives.

**New**: 4-layer hierarchical system:
- **Layer 1 — Decoder enforcement**: inventory limits, connectivity, valid piece types (never violated)
- **Layer 2 — Repair operators**: closure repair, branch validity (post-mutation, pre-evaluation)
- **Layer 3 — pymoo CV with ε-relaxation**: `G_pos = max(0, gap - 8.0)`, `G_angle = max(0, gap - 15.0)` as inequality constraints with `AdaptiveEpsilonConstraintHandling(perc_eps_until=0.5)`
- **Layer 4 — Soft penalties in objective**: boundary violation × 100, collision × 200, embedded in single fitness `F = pieces_used × 1000 − penalties`

Result: `n_obj=1`, `n_ieq_constr=2` (position closure, angle closure).

### Repair Pipeline (NEW)

**Current**: Simple `TrackRepair` that replaces excess pieces with -1.

**New**: Four repair operators running post-mutation, pre-evaluation:
1. **Closure repair** — angular budget arithmetic, enumerate ≤6-piece closing sequences, pre-computed lookup tables
2. **Inventory repair** — remove excess pieces by lowest marginal fitness contribution
3. **Branch rejoin repair** — extend dangling branches or deactivate them
4. **Switch pairing repair** — balance IN/OUT switches

Baldwinian by default (evaluate repaired phenotype, keep original genotype). 5% partial Lamarckian for top elites only.

### Population & Selection (ADJUST)

**Current**: pop_size=100, 51% heuristic seeding, max 1000 generations, SBX/PM.

**New**: pop_size=100 (unchanged), 15% heuristic seeding (reduced), max 500 generations (50K evaluations), stagnation at 50 generations with hypermutation response, aging evolution (0.99^age decay on crowding distance tiebreaker), external Hall of Fame archive of 20–30 elite feasible solutions. Equal-or-better acceptance rule for neutral drift.

### What Is REUSED Without Change

- **FK engine** (`geometry.py` core FK computation) — fully reusable, decoder calls it in correct order
- **Piece catalog** (`data.py` — `TrackCatalog`, `TrackPiece`, `FKDeltas`, `Port`) — consumed as-is
- **Physics simulation** (`evaluation.py` — `compute_speed_profile()`, forward-backward pass) — unchanged
- **Collision detection** logic — reusable
- **Boundary checking** logic — reusable
- **Visualization** — extend for branches, otherwise reusable
- **Config** (`config.py`) — extend with new parameters

---

## Implementation Plan Requirements

Create a phased implementation plan. Each phase must produce a working, runnable system.

### Phase 1: Core System (encoding, decoder, basic operators)

1. **Define segment constants and helpers** — `L_MAX`, `S_MAX`, `B_MAX`, `B_SLOT`, `C_MAX`, `N_VAR`. Functions to slice chromosome into segments: `get_main_loop(x)`, `get_switch_mask(x)`, `get_branch_slots(x)`, `get_crossing_overlay(x)`. Segment-specific `xl`/`xu` bound arrays.

2. **Implement `decode_chromosome()`** — the 4-phase construction decoder in `geometry.py` (or a new `decoder.py`). Reuse the existing FK engine. Phase 1 focus: main loop construction only (phases 2–4 can be stubs that return empty). Must handle inventory checking, angular budget, gene skipping. Returns a `Layout` object (extend existing class with `paths`, `closure_gaps`).

3. **Implement MUTATE, ADD, DELETE operators** — in `operators.py`, a `TrackLayoutMutation(Mutation)` class. Random operator selection with fixed probabilities (AOS deferred to Phase 3). Each operator works on the integer chromosome array. BRANCH and COMPOUND operators are stubs returning the parent unchanged.

4. **Implement `NoOpCrossover(Crossover)`** — identity operator returning parents unchanged. Wire into NSGA-II as `crossover=NoOpCrossover()`.

5. **Update `TrackLayoutProblem`** — segment-specific bounds, `n_obj=1` (single fitness), `n_ieq_constr=2` (position/angle closure). Call `decode_chromosome()` instead of `build_layout()`. Compute `F = pieces_used × 1000 − soft_penalties`. Report `G_pos` and `G_angle`.

6. **Update `HeuristicSampling`** — generate chromosomes in the new multi-segment format. 15 heuristic seeds (circles, ovals, racetracks encoded as main-loop genes), 85 random (random integer arrays decoded by the construction decoder).

7. **Basic closure and inventory repair** — implement as part of `TrackRepair` or a new repair class. Closure repair can start simple (swap last N pieces). Inventory repair removes excess.

8. **Wire into `main.py`** — NSGA-II with `TrackLayoutMutation`, `NoOpCrossover`, updated problem, updated sampling.

9. **Verify the system runs** — execute a short optimization (20 generations, pop=20) and confirm: no crashes, Layout objects are created, fitness values are computed, constraints are reported.

### Phase 2: Branches and ε-Constraints

1. **Implement decoder phases 2–3** — switch placement with branch stack, branch construction with proximity-based rejoin detection. Deterministic fill algorithm for branches that fail to rejoin from genes alone.

2. **Implement BRANCH operator** — four sub-operators: ADD_BRANCH, EXTEND_BRANCH, SHORTEN_BRANCH, REMOVE_BRANCH in `TrackLayoutMutation`.

3. **Implement branch rejoin repair and switch pairing repair**.

4. **Add `AdaptiveEpsilonConstraintHandling`** — wrap the algorithm with `perc_eps_until=0.5`.

5. **Implement decoder phase 4** — crossing overlay validation and placement.

6. **Implement COMPOUND operator** — segment replacement and local search mutation.

7. **Extend visualization** — `plot_layout_geometry()` should render branches and switch points.

### Phase 3: Adaptive Selection, Crossover, and Diversity

1. **Implement AOS controller** — `TrackMutationAOS` class with ALNS-UCB hybrid, operator quality tracking, phase-based scheduling overlay. Integrate into `TrackLayoutMutation._do()`.

2. **Implement segment-selective crossover** — `SegmentSelectiveCrossover(Crossover)`:
   - Main loop: ρ=1.0 (copy from elite parent)
   - Switch mask: parameterized uniform crossover, ρ=0.7
   - Branch slots: atomic slot-level uniform crossover, ρ=0.6
   - Crossing overlay: pair-level uniform crossover, ρ=0.6
   - Post-crossover repair: remap branch src_switch_idx and rejoin_target references.
   - Replace `NoOpCrossover` with this. Set `P_c=0.3`.

3. **BRKGA mutant injection** — each generation, replace bottom 10% with random chromosomes decoded via the construction decoder.

4. **Aging evolution** — add birth-generation counter per individual, apply 0.99^age decay to crowding distance in NSGA-II survival tiebreaker.

5. **External archive** — `EliteArchive` class maintaining 20–30 best feasible solutions. Updated each generation. Seed source for stagnation restarts.

6. **Stagnation detection and response** — monitor best feasible fitness; after 50 generations of no improvement, trigger hypermutation (3× operator intensity for 10 generations) and inject 10% random individuals.

---

## Critical Implementation Constraints

- **Reuse existing FK engine** — the turtle-graphics FK computation in `geometry.py` is correct and tested. The decoder calls it sequentially for each phase.
- **Reuse `TrackCatalog` and `TrackPiece` unchanged** — piece-type indices from the catalog become the gene value space.
- **Reuse physics simulation unchanged** — `compute_speed_profile()` operates on a `Layout` object regardless of how it was built.
- **pymoo compatibility** — all operators must subclass pymoo's `Mutation`, `Crossover`, `Sampling`, `Repair` properly. The problem must be `ElementwiseProblem`. Use NSGA-II.
- **Integer chromosome** — the chromosome is `int16`. Use `IntegerFromFloatMating` or handle rounding in operators. Bounds via `xl`/`xu` arrays.
- **No breaking changes to data files** — `track_pieces.yaml` and config files should be extended, not replaced.
- **Each phase must run independently** — Phase 1 must produce working results before Phase 2 begins.

## How to Proceed

1. First, read and understand the four key documents listed at the top.
2. Read the entire existing Python codebase to understand current module structure, class hierarchies, and function signatures.
3. Create a detailed implementation plan as a checklist, mapping each change to specific files and functions.
4. Implement Phase 1 completely, testing after each major step.
5. Run the system end-to-end after Phase 1 to verify it works.
6. Proceed to Phase 2, then Phase 3, verifying at each stage.

Do not rewrite modules from scratch unless the new design is fundamentally incompatible. Extend existing classes, add new methods, and preserve working code paths where possible. When a function signature changes, update all callers.
