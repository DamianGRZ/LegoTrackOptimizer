# LEGO Track Optimizer

**Code Quality**: Always use pymoo and Python best practices. Write professional, clean code - avoid excessive if/else chains and AI-generated patterns. Prefer vectorized numpy operations, early returns, and functional decomposition.

---

## Testing & Verification

- **Always run the FULL test suite after code changes.** Do not use `--quick` or `-k <subset>` for validation unless explicitly told otherwise.
- **No assertion without evidence.** Never claim a fix works without actually running the relevant command and pasting the literal output. If output contradicts your hypothesis, investigate — do not explain it away.
- **Verify feasibility, not just exit codes.** After optimizer runs, confirm closure error, orphan switches, and feasible-solution count via `/diag` before declaring success.
- **Use `/verify-fix`** for the full run-edit-test-inspect loop.

---

## MCP Servers

| Server | When to use |
|--------|-------------|
| `context7` | **Before guessing any pymoo API.** Use proactively for library/API shapes, callback signatures, operator conventions, version-specific syntax. Call `mcp__context7__resolve-library-id` then `mcp__context7__query-docs`. Do not guess pymoo internals from memory. |

---

## File & Git Operations

- **"Remove" or "untrack" never means `rm` from disk.** Use `git rm --cached <file>` or update `.gitignore`. Deleting files without asking is a hard-don't.
- **Never `git init`** without first verifying the directory is not already a git repo.
- **Auto-edits are enabled.** Do not ask for confirmation on routine file edits inside `src/`, `tests/`, `configs/`, or `data/`.
- **Still confirm for destructive ops**: `git reset --hard`, `git push --force`, deleting branches, dropping files.

---

## Project Invariants

Recurring mistakes that must not repeat:

- **Chromosome length scales with inventory dynamically.** Never hardcode `N_VAR`. The optimizer maximizes piece usage, so fixed-size slots cap the search space artificially.
- **No hardcoded dimensional or constraint limits.** Boundary, branch count, switch-pair count must all derive from inventory and config at runtime.
- **Repair must be wired into the evaluation pipeline**, not called ad-hoc. Constraint metric for switches is orphan-switch count, not just `loose_port_count`.
- **Fitness must reward branches.** If the objective does not credit multi-path topology, the GA eliminates switches as pure overhead.

---

## What, Why, How

**What**: Multi-objective genetic algorithm for optimizing closed LEGO railway layouts with fixed inventory.

**Why**: Generate feasible track layouts satisfying geometric constraints (closure, boundaries) while maximizing piece utilization and train speed.

**How**: pymoo NSGA-II/GA with heuristic sampling, template-based passing sidings, construction-based decoder, and locomotive physics model.

---

## Available Skills

| Skill | Usage | Description | When to Invoke |
|-------|-------|-------------|----------------|
| `/optimize` | Invoke with Skill tool | Run LEGO Track Optimizer with flexible config options | User asks to run optimization |
| `/test` | Invoke with Skill tool | Inline pytest runner (no agent overhead) | After code changes, before committing, when user says "run tests" |
| `/review` | Invoke with Skill tool | Compact code review against pymoo/project conventions | After writing or modifying src/ files |
| `/diag` | Invoke with Skill tool | Parse outputs/ directory and report fitness, constraints, layout | After any optimization run completes |
| `/quality` | Invoke with Skill tool | Deep Python & pymoo quality gate — rewrites code to project standards | After implementing new features or refactoring. Use instead of `/review` for deep analysis |
| `/verify-fix` | Invoke with Skill tool | End-to-end verification loop: full tests + optimizer run + diag, with literal output | After every bug fix. Enforces no-assertion-without-evidence. |
| `/verify-run` | Invoke with Skill tool | Launch a named `verify_<config>` run in the background, watch to completion, hand off to `/diag` | User asks to run/verify a config or babysit a running optimization |
| `/inspect-layout` | Invoke with Skill tool | Visually read layout/snapshot PNGs and check geometry invariants; decide viz bug vs real geometry bug | User asks to analyze snapshots or verify a layout image looks correct |
| `/code-map-audit` | Invoke with Skill tool | Parallel-agent dead-code scan + code map written into CLAUDE.md (re-grep before delete; never auto-rm) | User asks to audit for unused code or refresh the code-map section |

---

## Available Agents

Use with the Task tool for specialized work. **Prefer skills over agents when possible** — skills run inline and save ~5-8k tokens per invocation.

| Agent | Purpose | When to Use (vs Skill) |
|-------|---------|------------------------|
| `test-runner-analyzer` | Deep test analysis with coverage, flaky detection | Only for complex test debugging. For simple runs, use `/test` skill instead |
| `config-test-runner` | Run FULL optimizer with all configs, validate layouts visually, check chromosomes | When validating that all configs produce correct results (switches→branches, crossing→crossings, all closed) |
| `python-pymoo-reviewer` | Deep pymoo architecture review | Only for large refactors. For quick checks, use `/review` skill instead |
| `pymoo-error-fixer` | Fix issues identified by the review agent | After `/review` or reviewer agent finds issues |
| `ga-pymoo-implementer` | Implement GA/pymoo code after planning phase | After plan is approved, for complex implementations |
| `research-explorer` | Research technical concepts, evaluate proposals | For deep research requiring web search and documentation |

---

## Tech Stack

- **pymoo 0.6.1.6**: Multi-objective optimization (NSGA-II, GA)
- **numpy >=1.24.0, scipy >=1.10.0**: Scientific computing
- **pyyaml >=6.0, pydantic >=2.0.0**: Config and data validation
- **matplotlib >=3.7.0**: Visualization
- **pytest >=7.4.0**: Testing

---

## Project Structure

### Core Code (`/src`)

| File/Package | Purpose |
|------|---------|
| `types.py` | Shared domain types: `SwitchPair`, `TraversalPath`, `MultiPathLayout`, `PieceClass`, `FKRoute`, `PieceTopology` |
| `catalog/` | `TrackCatalog`, `TrackPiece`, `FKDeltas`, YAML loader |
| `train/` | `TrainConfig`, `SpeedProfile`, `compute_speed_profile`, lateral stability physics |
| `geometry.py` | `Layout`, `build_layout()`, `compute_fk_chain()` |
| `problem.py` | `TrackOptimizationProblem` — bi-objective NSGA-II problem |
| `decoder.py` | Construction-based decoder for partitioned chromosomes |
| `encoding.py` | Partitioned chromosome encoding, `PartitionedDimensions` |
| `config.py` | Pydantic models: `OptimizationConfig`, `BoundaryConfig` |
| `templates.py` | Template-based passing siding definitions (LEFT_SIDING, RIGHT_SIDING) |
| `sampling.py` | `IntegerSampling` — seeds with valid closed loops |
| `operators.py` | `PartitionedCrossover`, `PartitionedMutation` |
| `repair.py` | `TrackRepairPipeline` for fixing chromosomes |
| `intersection.py` | Crossing detection for self-intersecting layouts |
| `visualization.py` | `plot_layout()`, `plot_multi_path_layout()`, `plot_pareto_front()` |
| `lego_track_models.py` | Geometry constants for visualization rendering |

### Data & Configs

- **`data/track_pieces_v2.yaml`**: Track catalog with FK deltas, physics, ports, piece_index (the v1 `track_pieces.yaml` is absent — see Stale / Needs Action)
- **`configs/*.yaml`**: default, compact, with_switches, with_crossing

---

## Essential Commands

Prefer skills over raw commands — they handle argument parsing and result formatting:

```bash
# Optimization (use /optimize skill)
/optimize                           # Default config, full run
/optimize -c with_switches          # With switches config
/optimize --quick                   # Quick test (20 gen)

# Testing (use /test skill)
/test                               # Full suite, compact output
/test geometry                      # Specific module
/test decoder -v                    # Verbose

# Code review (use /review skill)
/review                             # Review current changes
/review src/problem.py              # Specific file

# Diagnostics (use /diag skill after optimization)
/diag                               # Parse outputs/ and report

# Raw commands (when skills don't fit)
python main.py --config configs/default.yaml --verbose
pytest --tb=short -q

# Architecture enforcement
lint-imports                        # Verify layer contracts (import-linter)
```

---

## Architecture (Verified 2026-05-30)

Authoritative snapshot from a read of the live NSGA-II pipeline. pymoo class names kept verbatim.

**1. Representation (chromosome).** Fixed-length-per-run `int16` vector; `n_var` is **derived from inventory at runtime — never hardcoded** (`compute_dimensions`, `src/encoding.py`; e.g. 330 for `with_switches`). Partitioned via `PartitionedDimensions` (`*_start/_end` properties):
- `[0, n_main)` main-loop piece types; alleles `xl/xu = [-1, 2]` (`-1`=INACTIVE, 0=STRAIGHT_16, 1=STRAIGHT_24, 2=R40_CURVE). Switches/crossings are **not** legal alleles here — they enter only via descriptor blocks / decoder repair.
- `[n_main, 2·n_main)` per-slot R40 flip bits `{0,1}` (handedness).
- passing-siding descriptors `J×4`, cross-junction `K×3`, double-crossover `D×5`, then 2 start-position genes.
- Sampling: custom `IntegerSampling(Sampling)` (`src/sampling.py`) — **not** `IntegerRandomSampling`.

**2. Decoder (genotype→phenotype).** `decode_chromosome(x, catalog, inventory, dims, config) -> MultiPathLayout` (`src/decoder/construction.py`). Forward kinematics segment-by-segment via `compute_fk_chain(fk_deltas)` (`src/geometry.py`) accumulating `[x,y,θ]`; R40 handedness applied in `get_fk_with_flip` (negates `dy,dθ`). Catalog inputs: `_fk_table [dx,dy,dθ°]`, `_radius_table`, `_arc_length_table`, `_speed_table` (`TrackCatalog`, `src/catalog/catalog.py`), studs/degrees. Kit is **R40-only** (no R56/R104); R40_CURVE = `[15.307, 3.045, 22.5°]`. Decoder enumerates `2^J` traversal paths and validates siding/cross/DC geometry by construction.

**3. Objectives & constraints.** `TrackOptimizationProblem(ElementwiseProblem)` (`src/problem.py`), `_evaluate(x, out)`. `n_obj=2`, `n_ieq_constr = 5 + catalog.n_pieces`, **no equality constraints (H)**.
- `F[0] = -(n_pieces / total_inventory)` — maximize inventory utilization.
- `F[1] = -(avg_speed of the slowest of the 2^J routes)` via `_slowest_route_speed(...)` at `SPEED_SAFETY_MARGIN=0.95`. Speed = 3-pass time-optimal profile `compute_speed_profile` (`src/train/scoring.py`); **Nadal's criterion** is one of the per-segment caps in `_compute_stability` (`v_eff = min(v_slide, v_tip, v_nadal, v_motor)`), not the objective itself.
- **Loop closure is an inequality (G), not H:** `G[0..2] = |dx|/tol−1, |dy|/tol−1, |dθ°|/tol−1`; `G[3]`=boundary; `G[4]`=collisions; `G[5..]`=per-type inventory excess.

**4. Operators.** Custom `PartitionedCrossover(Crossover)` (`__init__(dims, prob=0.9)`, `src/operators.py`) — one-point on the main loop (flip array cut mirrored), uniform per-slot swap on descriptors, DC blocks kept parent-intact (**not** SBX). Custom `PartitionedMutation(Mutation)` (`__init__(dims, prob=0.3)`) — weighted sub-operator portfolio (**not** PM). Selection: `NSGA2` default binary tournament (not set explicitly). Survival: `ConstrRankAndCrowding()` (NSGA2, Deb feasibility-first) or RNSGA2 niching, wrapped in `LegoAdaptiveEpsilon(AdaptiveEpsilonConstraintHandling)` (`src/algorithm/runner.py`).

**5. Repair pipeline.** `TrackRepairPipeline(Repair)` (`src/repair.py`) chains **3 stages** in `_do`: `JunctionValidityRepair` (clamp descriptor genes via `np.clip`) → `InventoryRepair` (drop excess pieces) → `MainLoopClosureRepair` (adjust curves for closure; optional via `enable_closure_repair`). **No `RoundingRepair`** (genome is already int16; clamping is manual).

**6. Heuristic sampling.** Hybrid (`IntegerSampling._do`): `heuristic_ratio` (default 0.20) of the population is inventory/boundary-aware closed loops (`_gen_simple_loop` / `_oval` / `_racetrack` / `_oval_with_siding` / `_oval_two_sidings` / `_figure_eight` / `_*_dbl_crossover`); the remainder are random partial-fill chromosomes (`_random_chromosome`).

**7. Config & scale.** `AlgorithmConfig` (`src/config.py`): `default.yaml` = RNSGA2, `pop_size=1000`, `n_gen=200`, `crossover_prob=0.2`, `mutation_prob=0.8`, `heuristic_ratio=0.20`, `seed=null`; `with_switches.yaml` = NSGA2, `n_gen=500`. Termination via `get_termination("n_gen", n_gen)`. Catalog = **7 piece types**; typical solution is tens of pieces (`n_main` up to 160). `seed=null` (non-deterministic); **one run per `main.py` invocation** (no built-in multi-seed loop).

---

## Implementation Details

### Chromosome Encoding (Multi-Segment)

Fixed-length-per-run `int16` vector; **`n_var` is derived from inventory at runtime — never hardcoded** (`PartitionedDimensions`, `src/encoding.py`). Segment offsets come from `*_start/_end` properties:

| Segment | Range | Purpose |
|---------|-------|---------|
| Main-loop types | `[0, n_main)` | piece type per slot; alleles `[-1, 2]` (-1=INACTIVE, 0=STRAIGHT_16, 1=STRAIGHT_24, 2=R40_CURVE) |
| Main-loop flips | `[n_main, 2·n_main)` | per-slot R40 handedness bit `{0, 1}` |
| Passing-siding junctions | `J × 4` | (active, position, handedness, n_straights) |
| Cross-junctions | `K × 3` | (active, position_W, handedness) |
| Double-crossovers | `D × 5` | (active, pos_1, route_1, pos_2, route_2) |
| Start position | last 2 | start_x, start_y |

Switches/crossings are NOT main-loop alleles — they enter via the descriptor blocks / decoder repair.

### Legacy Chromosome Encoding

- Integer array of length `total_inventory` (sum of all piece counts)
- Values: `-1` = empty slot, `0..N-1` = piece index from `piece_index` mapping
- Last 2 values: start_x, start_y

### Forward Kinematics
```python
# geometry.py - compute_fk_chain()
theta_rad = np.radians(states[i, 2])
x_new = x + dx * cos(theta_rad) - dy * sin(theta_rad)
y_new = y + dx * sin(theta_rad) + dy * cos(theta_rad)
theta_new = theta + dtheta
```

### Problem Class (problem.py)

Single class `TrackOptimizationProblem(ElementwiseProblem)` — bi-objective (`n_obj=2`), `n_ieq_constr = 5 + catalog.n_pieces`, no equality constraints.

### Objectives (2, both minimized)
```python
F[0] = -utilization          # Maximize piece usage (n_pieces / total_inventory)
F[1] = -slowest_route_speed  # Maximize the slowest route's avg speed
                             #   (_slowest_route_speed, SPEED_SAFETY_MARGIN=0.95)
```

### Constraints (5 + per-piece-type, g <= 0 feasible)
```python
G[0..2] = |dx|/closure_tol-1, |dy|/closure_tol-1, |dθ°|/angle_tol-1  # closure (main path)
G[3]    = (boundary_violation - boundary_tol) / diagonal             # boundary
G[4]    = collisions   # unresolved crossings/5 + dangling cross/DC ports
G[5..]  = per-type inventory excess (one per catalog piece index)
```

### Physics Model (from Locomotive_dynamics.md)
```python
# train/physics.py - v_eff_array(): per-segment derailment cap
v_eff = min(v_slide, v_tip, v_nadal, v_motor_max)
#   v_slide = sqrt(mu_design * g * R)   # mu_design=0.25, v_motor_max=1.10 m/s
# train/scoring.py - compute_speed_profile(): 3-pass time-optimal profile;
#   SPEED_SAFETY_MARGIN=0.95 derates every cap.
```

---

## Template-Based Passing Sidings

Standard LEGO passing siding pattern:
```
[IN_switch] -> [approach_curve] -> [straights×N] -> [return_curve] -> [OUT_switch]
```

A siding is an **opposite-handed pair**: 1 LEFT + 1 RIGHT switch (+ 2 R40 curves + N straights). The exit switch is installed **reversed**. Branch curves are `R40_CURVE` with a flip bit — there is no separate `R40_LEFT/RIGHT` piece.

| Template | Entry switch | Exit switch (reversed) | Branch curves |
|----------|--------------|------------------------|---------------|
| LEFT_SIDING | R40_SWITCH_LEFT (4) | R40_SWITCH_RIGHT (5) | R40_CURVE flip=1, … , R40_CURVE |
| RIGHT_SIDING | R40_SWITCH_RIGHT (5) | R40_SWITCH_LEFT (4) | R40_CURVE flip=0, … , R40_CURVE |

The decoder automatically:
1. Reads active junction descriptors, sorts by position
2. Computes branch geometry from templates and validates siding geometry by construction (drops descriptors that don't validate, releasing their inventory)
3. Injects the switch pair into the main loop (entry at `position`, exit found downstream)
4. Enumerates all 2^N traversal paths (N = number of switch pairs)

---

## Data Flow

```
data/track_pieces_v2.yaml -> TrackCatalog (FK tables, speed limits, routes)
configs/*.yaml -> OptimizationConfig (inventory, boundary, algorithm)
main.py -> GA/NSGA2 + IntegerSampling -> Problem._evaluate()
chromosome -> decode_chromosome() -> MultiPathLayout
MultiPathLayout -> compute_speed_profile() -> F[], G[]
results -> visualization -> outputs/
```

---

## Key Classes

### TrackCatalog (catalog/catalog.py)
- `_fk_table`: (n, 3) array of [dx, dy, dtheta]
- `_speed_table`, `_radius_table`, `_arc_length_table`
- `get_fk(indices)`, `get_fk_route(piece_idx, route_idx)`
- `get_radii(indices)`, `get_speed_limits(indices)`

### Layout (geometry.py)
- `indices`: piece indices used
- `states`: (n+1, 3) cumulative [x, y, theta]
- `closure_error`, `angle_error`, `bounding_box`, `is_closed()`

### MultiPathLayout (types.py)
- `main_loop_pieces`: base piece sequence
- `switch_pairs`: list of matched SwitchPair objects
- `paths`: list of TraversalPath (2^N paths for N switch pairs)
- `loose_port_count`: switches with unconnected port 2

### SpeedProfile (train/scoring.py)
- `speeds`: speed at each segment
- `avg_speed`, `lap_time`, `total_distance`

---

## IntegerSampling (Heuristic Seeding)

Seeds ~20% of population (`heuristic_ratio`, default 0.20) with inventory-valid closed loop patterns:
- Simple circles (16 R40 curves)
- Symmetric ovals (R40 curves + straights)
- Racetracks (4 corners + straights)
- Ovals with passing sidings (LEFT or RIGHT switches)

Validates patterns against inventory before use.

---

## Track Piece Index Mapping

R40 is ONE physical piece (4DBrix 2.04.069); handedness is selected per
placement via the chromosome's parallel flip array (flip=0 → LEFT +22.5°,
flip=1 → RIGHT -22.5°). Switches keep separate LEFT/RIGHT piece types
because their port/route geometry isn't a simple mirror.

| Index | Piece ID | Description |
|-------|----------|-------------|
| 0 | STRAIGHT_16 | 16-stud straight |
| 1 | STRAIGHT_24 | 24-stud straight |
| 2 | R40_CURVE | 22.5 deg curve; direction set by per-slot flip bit |
| 3 | CROSS_90 | 90 deg crossing |
| 4 | R40_SWITCH_LEFT | Left switch (diverges left) |
| 5 | R40_SWITCH_RIGHT | Right switch (diverges right) |
| 6 | DOUBLE_CROSSOVER | 4 independent switches, 48x16 studs |

Chromosome layout extends the main loop with a parallel `main_loop_flips`
segment of equal length. Decoder applies the flip at FK time
(`get_fk_with_flip`); non-R40 slots ignore it.

---

## Quick Reference

| Piece | Angle | Pieces/Circle | Speed Limit |
|-------|-------|---------------|-------------|
| R40 | 22.5 deg | 16 | 0.97 m/s |
| Straights | 0 deg | N/A | 1.57 m/s |
| CROSS_90 | 90 deg | N/A | 1.57 m/s |
| Switches (straight) | 0 deg | N/A | 1.57 m/s |
| Switches (diverge) | 22.5 deg | N/A | 0.97 m/s |

---

## Development Workflows

### Adding Track Pieces
1. Add to `data/track_pieces_v2.yaml` with `fk`, `physics`, `ports`, `routes` (for multi-port pieces)
2. Add to `piece_index` mapping with next available index
3. Update `INDEX_TO_ID` in `src/sampling.py`
4. If switch: add template to `src/templates.py`
5. Run `/test data` and `/test templates` to validate

### Adding Objectives/Constraints
1. Implement in `src/evaluation.py`
2. Update `n_obj` or `n_ieq_constr` in `src/problem.py`
3. Update `compute_objectives()` or `compute_constraints()`
4. Run `/review src/evaluation.py` then `/test evaluation`

### After Any Code Change
1. `/test` — verify nothing broke
2. `/review` — quick check for interface/convention issues
3. `/quality src/<file>.py` — deep quality gate for new or refactored code (rewrites to project standards)

### After Optimization Run
1. `/diag` — parse outputs and get diagnostic report

### Full Config Validation
Use the `config-test-runner` agent — runs ALL configs with full optimization, visually inspects layouts, validates chromosomes, checks that switches→branches, crossings→crossings, all layouts closed and connected.

---

**Status**: Implementation complete, optimization functional
**Python**: 3.x | **pymoo**: 0.6.1.6

---

## Code Map & Cleanup Findings (Audit 2026-05-18)

Snapshot from a four-agent sweep of `src/`, `tests/`, `configs/`, `data/`, root scripts, and top-level docs. Findings below were re-verified after a self-audit caught several errors in the first pass; treat as point-in-time inventory and re-verify before any actual `rm` / `git rm`. **Note**: configs in `configs/*.yaml` are CLI inputs (`main.py --config <path>`) — they are NOT imported from Python, so "no .py reference" is NOT evidence of being dead.

### Code Map (one-line purpose per module)

**`src/` core**
- `problem.py` — `TrackOptimizationProblem`, bi-objective NSGA-II (F[0]=−utilization, F[1]=−avg_speed per `problem.py:125`).
- `decoder/construction.py` + `decoder/types.py` — `decode_chromosome()`, partitioned chromosome → `MultiPathLayout`. Imported by `problem.py:16`, `algorithm/runner.py:22`, `run_info.py:25`, `run_v1_all_configs.py:12`, and tests/viz.
- `encoding.py` — partitioned-chromosome accessors (`get_main_loop_types`, `get_junction`, `INACTIVE`, `PieceIndex`, `MAIN_LOOP_PIECE_INDICES`); all symbols used.
- `geometry.py` — `compute_fk_chain`, plus the legacy Phase-1 `Layout` / `build_layout()` shim still consumed by tests, train/, and viz.
- `intersection.py` — crossing detection + dangling-port counters (`count_segment_crossings`, `count_dangling_cross_ports`, `count_dangling_double_crossover_ports`, `find_crossing_pairs`); all used by `problem.py` / `operators.py`.
- `repair.py` — `MainLoopClosureRepair`, `JunctionValidityRepair`, `InventoryRepair`, `TrackRepairPipeline` (wired into `algorithm/runner.py`).
- `templates.py` — passing-siding + cross-junction + double-crossover template tables; consumed by decoder, sampling, operators.
- `sampling.py` — `IntegerSampling` (heuristic + random seeding); only entry point for pymoo `sampling=`.
- `operators.py` — `PartitionedCrossover`, `PartitionedMutation`; private `_*` helpers all reach the two public operators.
- `types.py` — pure dataclasses (`SwitchPair`, `MultiPathLayout`, `TraversalPath`, `PieceClass`, `FKRoute`, `PieceTopology`, `DblCrossover`, `CrossJunction`); all used.
- `config.py` — Pydantic models `OptimizationConfig` / `BoundaryConfig` / `AlgorithmConfig` / `TerminationConfig`; loaded by `main.py` and `run_v1_all_configs.py`.
- `run_info.py` — provenance writer (header + summary); called only by `algorithm/runner.py`.
- `lego_track_models.py` — R40 / rail geometry constants for the renderer.

**`src/catalog/`**
- `catalog.py` — `TrackCatalog.load()` + vectorized FK/speed/radius/topology lookups; supports v1 fallback and v2 port-centric schema.
- `loader.py` — `load_catalog_spec()` (ruamel + Pydantic, with file:line error UX).
- `pieces.py` — `FKDeltas`, `Port`, `TrackPiece` dataclasses.
- `specs.py` — Pydantic v2 schema (`TrackCatalogSpec`, `TrackPieceSpec`, `PortDef`, `check_schema_version`).

**`src/train/`**
- `physics.py` — `TrainConfig`, `v_eff_array`, `available_accel`, `DEFAULT_TRAIN_CONFIG`.
- `scoring.py` — `SpeedProfile`, `compute_speed_profile()` (3-pass forward/backward friction profiler).
- `evaluation.py` — `PhysicalEvaluation`, `evaluate_layout()` (geometry / stability / kinematics / dynamics / energy in one O(n) pass).
- Not redundant: `scoring` is a building block, `evaluation` is the orchestrator.

**`src/algorithm/`**
- `runner.py` — `run_optimization()`, `save_results()`, `ProgressCallback`, `FeasibleEliteCallback`, `SnapshotCallback`, `CallbackChain`, `LegoAdaptiveEpsilon`. `FeasibleEliteCallback` + `ConvergenceMonitorCallback` always attached; `SnapshotCallback` only when `output_dir` is set (`runner.py:456`); `ProgressCallback` only when `verbose` (`runner.py:464`). Both `NSGA2` and `RNSGA2` algorithms are dispatched (`runner.py:410-413`).
- `monitoring.py` — `ConvergenceMonitorCallback` (HV/IGD/feasibility); confirmed attached at `runner.py:454`.

**`src/visualization/`**
- `track_renderer.py` — `plot_layout()`, `plot_multi_path_layout()`, `get_piece_color`, `get_piece_short_name` (both renderers called from `runner.save_results()`).
- `pareto_plot.py` — `plot_pareto_front()`.

### Candidates Needing Review

Each row is a symbol or file with no in-code importer/loader. Before deleting any of them, re-grep — string references (pymoo callback names, log messages) can hide a real consumer.

| Path / symbol | Evidence | Notes |
|---|---|---|
| Top-level `Literature-Grounded Audit ... .md` | Reference essay, no code/CLAUDE.md link. | User-authored research; ask before deleting. |
| Top-level `Structurally Similar Problems ... .md` | Same as above. | Ask before deleting. |
| `Modular9PartResearchV1/` (10 design docs) | Pre-dates current architecture; nothing in `src/` or `docs/` links in. | Archival; ask before deleting. |

**Deleted on 2026-05-18** (zero callers verified by repo-wide grep; full pytest pre/post deletion identical at `34 failed, 197 passed, 15 errors` — failures are pre-existing and unrelated):
- `src/sampling.py` end-of-file aliases `MultiSegmentSampling = IntegerSampling`, `HeuristicSampling = IntegerSampling`.
- `src/operators.py` class `NoOpCrossover` (+ its section-divider comment block).
- `src/problem.py` method `_compute_inventory_violation()` (superseded by `_compute_per_type_inventory_violation`).

**NOT candidates (correcting earlier pass):**
- `configs/with_double_crossover{,_narrow,_small}.yaml`, `configs/with_switches_and_crossing.yaml` — CLI inputs to `main.py --config`; archived milestone runs under `archive/dc/full-verify/` and `outputs/verify_with_switches_and_crossing_4/` confirm they've been used.
- `BoundaryConfig.width` / `.height` — used at `config.py:36` (inside `.diagonal`), `config.py:108`, `config.py:109` (inside `calculate_max_layout_pieces`, called by `n_var`).
- `AlgorithmConfig.name = "RNSGA2"` literal — dispatched at `runner.py:410-413` (real `RNSGA2(...)` instantiation).

### Stale / Needs Action (NOT auto-delete)

- **`data/track_pieces.yaml` is missing on disk.** Only `data/track_pieces_v2.yaml` exists. v1 filename is referenced at `tests/test_catalog.py:226`, `tests/test_catalog_parity.py:11`, and `tests/test_catalog_geometry.py:92` (comment). Either the v1-parity tests skip silently when the file is absent, or they're broken. Confirm by running `/test tests/test_catalog_parity.py -v` before changing anything.
- **Phase-1 `Layout` / `build_layout()` in `src/geometry.py`** — CLAUDE.md calls them legacy but tests (`test_geometry.py`, `test_evaluation.py`, `test_scoring.py`) and `train/` still consume them. Migration first, deletion second.
- **`src/problem.py:133-167` multi-path fallback (`hasattr(layout, 'get_main_path')`)** — defensive code path that smells retrofitted; harmless but worth re-reading once Phase-1 `Layout` is retired.

### Test Suite Notes

- 19 test files, **233** `def test_` functions (counted via `grep -c "def test_" tests/test_*.py`). 0 skip/xfail per agent sweep; no stale imports.
- The 5-way `test_catalog*.py` split (catalog / catalog_geometry / catalog_loader / catalog_parity / catalog_specs) IS justified — distinct scopes (runtime API / reference geometry doc / YAML+errors / v1↔v2 parity / Pydantic schema).
- All 6 fixtures under `tests/fixtures/` are referenced by `test_catalog_loader.py`. `tests/baselines/` contains only `.gitkeep`.

### Untouched (Run Artifacts — Do Not Delete)

All GA run output goes under `outputs/` (single gitignored tree). `main.py` writes to `outputs/` root; `run_v1_all_configs.py` writes to `outputs/<config_name>/`. Milestone runs worth keeping live under `archive/{dc,crossing,baselines}/` (also gitignored). Treat both as build output; clean only on explicit user instruction.

---
