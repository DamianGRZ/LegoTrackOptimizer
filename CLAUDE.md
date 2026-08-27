# LEGO Track Optimizer

**Code Quality**: Always use pymoo and Python best practices. Write professional, clean code - avoid excessive if/else chains and AI-generated patterns. Prefer vectorized numpy operations, early returns, and functional decomposition.

---

## Testing & Verification

- **Always run the FULL test suite after code changes.** Do not use `--quick` or `-k <subset>` for validation unless explicitly told otherwise.
- **No assertion without evidence.** Never claim a fix works without actually running the relevant command and pasting the literal output. If output contradicts your hypothesis, investigate — do not explain it away.
- **Verify feasibility, not just exit codes.** After optimizer runs, confirm closure error, orphan switches, and feasible-solution count via `/diag` before declaring success.
- **Use `/verify-fix`** for the full run-edit-test-inspect loop.
- **Style gate**: `python -m pycodestyle src tests main.py run_v1_all_configs.py run_ablation.py score_ablation.py` must exit 0 after code changes. Config in `setup.cfg`: 99-char limit; `ignore` re-lists pycodestyle defaults + E203 (slice colons like `x[start : end + 1]` are PEP 8-conformant — never "fix" them).
- **Known baseline (2026-08-26): 0 failed, 478 passed, 0 skipped** (clean).
  - The former perpetual failure (`test_two_layer_loop_closes`) was rewritten to `test_two_layer_both_through_is_infeasible`: the two-layer both-through DC pattern is geometrically infeasible (22.5° **oblique** self-crossing, unlegalizable by any catalog piece — CROSS_90 only handles 90° — plus a ~32-stud closure gap), so the test now documents that it does NOT close. The seed `_gen_two_layer_loop_dbl_crossover` stays a stub; the only single-loop DC topology that closes is the figure-8 (cross routes).

---

## MCP Servers

| Server | When to use |
|--------|-------------|
| `context7` | **ALWAYS verify with context7 when unsure about pymoo OR Python — never answer from memory.** Mandatory before asserting any pymoo API (class/operator/callback signatures, survival/crowding/constraint-handling behavior, version-specific syntax) and any non-trivial Python/stdlib/numpy/scipy semantics. Call `mcp__context7__resolve-library-id` then `mcp__context7__query-docs` (verified ids: pymoo `/anyoptimization/pymoo`; Python/stdlib `/python/cpython`; numpy `/numpy/numpy`; scipy `/websites/scipy_doc_scipy`). If a claim about library behavior gates a code change or a design decision, cite the doc — do not guess. |

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

**Why**: Generate feasible track layouts satisfying geometric constraints (closure, boundaries) while maximizing piece utilization and minimizing the expected time to traverse the whole network.

**How**: pymoo NSGA-II with heuristic sampling, template-based passing sidings, construction-based decoder, and locomotive physics model.

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
| `/inspect-genome` | Invoke with Skill tool | Decode chromosomes from a run and cross-check phenotype vs out-keys/PNG titles/reports | A layout image and a report disagree (e.g. crossing visible but counted 0); auditing piece usage |
| `/commit-slices` | Invoke with Skill tool | Slice a mixed working tree into logical, individually-green commits (no stash; worktree-verified) | User asks to split session work into commits |
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

- **pymoo 0.6.1.6**: Multi-objective optimization (NSGA2)
- **numpy >=1.24.0, scipy >=1.10.0**: Scientific computing
- **pyyaml >=6.0, ruamel.yaml, pydantic >=2.0.0**: Config and catalog validation
- **matplotlib >=3.7.0**: Visualization (forced `Agg` backend — Tk crashes under multiprocessing)
- **pytest >=7.4.0**: Testing

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
python run_v1_all_configs.py        # Batch: every config -> outputs/<config_name>/
pytest --tb=short -q
```

Note: `lint-imports` is installed in the venv but **no import-linter config exists** — running it fails with "Could not read any configuration". Either add a config or ignore the tool.

---

## Architecture (Verified 2026-06-12)

Authoritative snapshot from a full read of the live pipeline. pymoo class names kept verbatim.

**1. Representation (chromosome).** Fixed-length-per-run `int16` vector; `n_var` is **derived from inventory at runtime — never hardcoded** (`compute_dimensions`, `src/encoding.py`). Partitioned via `PartitionedDimensions` (`*_start/_end` properties):

| Segment | Range | Genes |
|---------|-------|-------|
| Main-loop types | `[0, n_main)` | piece type per slot; alleles `[-1, 2]` (-1=INACTIVE, 0=STRAIGHT_16, 1=STRAIGHT_24, 2=R40_CURVE) |
| Main-loop flips | `[n_main, 2·n_main)` | per-slot R40 handedness bit `{0, 1}` |
| Passing-siding junctions | `J × 4` | (active, position, handedness, n_straights); `J = min(LEFT, RIGHT)` switch inventory |
| Cross-junctions | `K × 3` | (active, pos_1, pos_2); `K = CROSS_90` inventory |
| Double-crossovers | `D × 5` | (active, pos_1, route_1, pos_2, route_2); `D = DOUBLE_CROSSOVER` inventory |
| Start position | last 2 | start_x, start_y (fine offset on top of decoder auto-centering) |

Switches/crossings are **not** legal main-loop alleles — they enter only via descriptor blocks / decoder repair. Sampling: custom `IntegerSampling(Sampling)` (`src/sampling.py`) — **not** `IntegerRandomSampling`.

**2. Decoder (genotype→phenotype).** `decode_chromosome(x, catalog, inventory, dims, config) -> MultiPathLayout` (`src/decoder/construction.py`). Steps: read main loop (inventory-checked) → validate junctions → inject switch pairs → inject cross-junctions → inject double-crossovers → emergent self-intersection repair (perpendicular STR-on-STR → CROSS_90, committed like a descriptor crossing: both slots rewritten, both straights released, one `CrossJunction` record with `slot=-1`) → FK + enumerate `2^J` traversal paths → auto-center in boundary. FK via `compute_fk_chain(fk_deltas)` (`src/geometry.py`, vectorized cumsum) accumulating `[x,y,θ°]`; R40 handedness applied in `get_fk_with_flip` (negates `dy,dθ`). Kit is **R40-only** (no R56/R104); R40_CURVE FK = `[15.307, 3.045, 22.5°]`. Descriptors that fail geometry/inventory validation are dropped (reasons recorded in `layout.drop_log`).

**3. Objectives & constraints.** `TrackOptimizationProblem(ElementwiseProblem)` (`src/problem.py`), `n_obj=2`, `n_ieq_constr = 5 + catalog.n_pieces`, **no equality constraints (H)**.
- `F[0] = -weighted_utilization` — `(n_physical_pieces + (special_piece_weight−1)·n_special) / total_inventory`, where `n_special` = switch pairs + cross-junctions + double-crossovers and `special_piece_weight` defaults to 3.0, so multi-path topology raises the score instead of being stripped as overhead.
- `F[1] = expected_traversal_time` (minimized directly, no negation) via `_expected_traversal_time(...)` at `SPEED_SAFETY_MARGIN=0.95`: each of the `2^J` routes is profiled (3-pass `compute_speed_profile`, `src/train/scoring.py`), every **physical** piece is charged the MEAN of its traversal times across all passages (identity via `TraversalPath.piece_uids`; a CROSS_90 — descriptor or emergent — and a DC span two slots but are one piece, unified through junction records; routes are profiled closed/open at the config closure tolerances), and the objective sums over distinct pieces. A plain (non-self-crossing) loop ⇒ exactly that loop's `lap_time`; a self-crossing one comes out **below** its lap time, because the crossing is one physical piece the lap passes twice yet is charged once. No usable route ⇒ `+inf` (time 0 would rank best). **Nadal's criterion** is one of the per-segment caps (`v_eff = min(v_slide, v_tip, v_nadal, v_motor)`), not the objective itself.
  - **Why it replaced the previous `F[1]`** (measured over the archive, not argued). The old objective was `-_slowest_route_speed(...)`: each of the `2^J` routes profiled, take that route's **`avg_speed`**, keep the MINIMUM over routes, negate — i.e. maximize the worst route's average pace. It failed because a route's `avg_speed` **rises with piece count** (0.842 on a 16-piece R40 circle → 1.009 on a 96-piece racetrack), so it moved *with* utilization instead of against it, leaving no conflict for a front to spread along; the surviving spread was a sliver, 0.974–1.007 m/s across a whole front. Median feasible-front size went from **2 points** (182 archived runs, min 1, max 12, 65 of them a single point) to **39** (1137 runs, max 130). On `all_pieces` the front is now continuous from the 16-piece R40 circle (util 7.3%, 2.39 s) to a 59.2% layout at 15.73 s, where the old objective produced 4–6 points inside a 0.03 m/s band.
- **Loop closure is an inequality (G), not H:** `G[0..2] = |dx|/tol−1, |dy|/tol−1, |dθ°|/tol−1`; `G[3]` = boundary; `G[4]` = collisions (`unresolved_crossings/5 + dangling_cross_ports + dangling_DC_ports`); `G[5..]` = per-type inventory excess (normalized by `max_occ[t]`).
- Degenerate (0-piece) layouts get `F = +inf`, `G = 1e6` — never NaN (breaks dominance comparison).
- Custom out-keys `n_sw_pairs` / `n_cross_comm` / `n_dc_comm` ride on the Population for the category-elite archive.

**4. Operators.** Custom `PartitionedCrossover(Crossover)` (`src/operators.py`) — one-point on the main loop with the cut mirrored on the flip array, uniform per-slot swap on descriptors; when either parent carries an active DC descriptor, the main loop + DC block stay parent-intact (**not** SBX). Custom `PartitionedMutation(Mutation)` — weighted sub-operator portfolio (**not** PM): piece-type change / activate / deactivate / swap / flip / crossing-aware straighten / compensated-pair grow. DC-bearing genomes receive ONLY closure-safe grows (`_compensated_pair_grow`, `_grow_dc_figure_eight`; a declined grow falls through to the other) — every other operator would break the FK-tuned figure-8; siding-bearing genomes get a junction op or the compensated-pair grow, with a declined grow falling back to a junction op. Both grows return True iff they edited the genome, and the compensated grow accepts holed loops (descriptor positions are active-order, matching the decoder). Selection: `NSGA2` default binary tournament (not set explicitly).

**5. Repair pipeline.** `TrackRepairPipeline(Repair)` (`src/repair.py`) chains **4 stages** in `_do`: `JunctionValidityRepair` (clamp descriptor genes) → `InventoryRepair` (drop excess pieces) → `MainLoopClosureRepair` (Stage 1 angular: add/remove R40s toward 360°, or nearest of {0, 360, 720} for cross/DC genomes; Stage 2 translational: drop straights to shrink the dx/dy gap) → `BoundaryAwareRepair` (re-center, else shrink via anti-parallel straight pairs). **No `RoundingRepair`** (genome is already int16; clamping is manual). Closure + boundary stages are optional via `enable_closure_repair` / `enable_boundary_repair`.

**6. Heuristic sampling.** Hybrid (`IntegerSampling._do`): `heuristic_ratio` (default 0.20) of the population is inventory/boundary-aware closed loops (`_gen_simple_loop` / `_oval` / `_racetrack` / `_oval_with_siding` / `_oval_two_sidings` / `_figure_eight` / `_figure_eight_cross` / `_figure_eight_dbl_crossover`); the remainder are random partial-fill chromosomes. All pattern dimensions derive from boundary + inventory.

**7. Survival & constraint handling.** `NSGA2` uses `ConstrRankAndCrowding()` (Deb feasibility-first), wrapped in `LegoAdaptiveEpsilon(AdaptiveEpsilonConstraintHandling)` (`src/algorithm/runner.py`): three-phase schedule (hold → linear decay → strict), epsilon_0 from the 10th percentile of infeasible CVs, capped at `SOFT_CONSTRAINT_COUNT` (4). That cap binds in every run measured so far — `cv_eps` starts at exactly 4.0 in 420/420 archived cells — so the calibration is saturated and what ships is a constant on a decay schedule, not a population-fitted epsilon. Closure + boundary (`G[0..3]`) are SOFT (relaxable); collisions + inventory are HARD — weighted ×1000 via `cv_ieq.scale` so no epsilon can relax them (never bake epsilon into CV; see memory note on the tournament crash). The crowding metric is `algorithm.crowding_func` (default `cd`, built by `_build_survival`). **`pcd` was measured and is not better**: 10 arms × 21 configs × 3 seeds on both metrics, no arm reaches `p < 0.5` for `pcd > cd`, mean HV deltas span −0.016..+0.003, and cost is equal (19.7 h vs 20.2 h) — so `cd` stays the default. Selecting `pcd` routes through `_pruning_crowding`, which restores the clamp the compiled `calc_pcd` kernel is missing: unguarded it writes past its buffer once `n_remove >= n_distinct − n_obj`, an intermittent segfault this problem hits routinely (a 1000-individual population holds ~25 distinct `F`).

**8. Callbacks.** Always attached: `FeasibleEliteCallback` (re-injects best-utilization feasible), `CategoryEliteArchive` (per switch/cross/DC category elites; must run AFTER the global elite), `ConvergenceMonitorCallback` (HV/IGD/feasibility + run-cumulative feasible front — dedupe before NDS or the run slows down quadratically). `SnapshotCallback` only when `output_dir` is set; `ProgressCallback` only when `verbose`.

**9. Config & scale.** `AlgorithmConfig` (`src/config.py`): `default.yaml` = NSGA2, `pop_size=1000`, `n_gen=200`, `crossover_prob=0.2`, `mutation_prob=0.8`, `heuristic_ratio=0.20`, `seed=null`; `with_switches.yaml` = NSGA2, `n_gen=500`, `n_workers=32`, termination `period=100`. Termination via `_build_termination`: `MaximumGenerationTermination` by default; `DefaultMultiObjectiveTermination` (improvement-aware early stop) when `termination.period > 0`, with `n_max_evals=inf` — pymoo's hidden 100000 default would cap pop-1000 runs at generation 100. The epsilon Meta-wrapper's own `evaluator.n_eval` stays 0 (the live counter is on `__super__`); the monitor reads the true one. Catalog = **7 piece types**. `seed=null` (non-deterministic); **one run per `main.py` invocation**.

---

## Track Pieces (Catalog Index, Geometry)

R40 is ONE physical piece (4DBrix 2.04.069); handedness is selected per placement via the chromosome's parallel flip array (flip=0 → LEFT +22.5°, flip=1 → RIGHT −22.5°). Switches keep separate LEFT/RIGHT piece types because their port/route geometry isn't a simple mirror.

**The catalog carries geometry only — no speed data.** Speed caps are derived at runtime from train physics (`v_eff_array`, `src/train/physics.py`): straights bind at the motor cap (`v_motor_max`, measured 1.26 m/s), R40-radius segments (curves, switch diverge, DC diagonals) at the lateral-slide cap `sqrt(mu·g·R)` ≈ 0.886 m/s at `mu_design=0.25`.

| Index | Piece ID | Geometry |
|-------|----------|----------|
| 0 | STRAIGHT_16 | 16-stud straight |
| 1 | STRAIGHT_24 | 24-stud straight |
| 2 | R40_CURVE | 22.5° curve (16/circle); direction = flip bit |
| 3 | CROSS_90 | 90° crossing; FK == STRAIGHT_16 |
| 4 | R40_SWITCH_LEFT | left switch, 32-stud body; through straight / diverge R40 arc |
| 5 | R40_SWITCH_RIGHT | right switch, 32-stud body; through straight / diverge R40 arc |
| 6 | DOUBLE_CROSSOVER | 48×16 studs, 4 routes (2 through + 2 diagonal) |

---

## Template-Based Passing Sidings

```
[IN_switch] -> [approach_curve] -> [straights×N] -> [return_curve] -> [OUT_switch]
```

A siding is an **opposite-handed pair**: 1 LEFT + 1 RIGHT switch (+ 2 R40 curves + N straights). The exit switch is installed **reversed** (its merge FK comes from `template.merge_fk`, not catalog routes).

| Template | Entry switch | Exit switch (reversed) | Branch curve flips |
|----------|--------------|------------------------|--------------------|
| LEFT_SIDING | R40_SWITCH_LEFT (4) | R40_SWITCH_RIGHT (5) | flip=1 |
| RIGHT_SIDING | R40_SWITCH_RIGHT (5) | R40_SWITCH_LEFT (4) | flip=0 |

The decoder reads active descriptors sorted by position, computes branch geometry from templates, validates by construction (dropping invalid descriptors and releasing their inventory), injects the pair into the main loop (exit found downstream by walking X-distance), and enumerates all 2^N traversal paths.

---

## Data Flow

```
data/track_pieces_v2.yaml -> TrackCatalog (FK tables, radii, routes; no speed data)
configs/*.yaml -> OptimizationConfig (inventory, boundary, algorithm)
main.py -> NSGA2 + IntegerSampling -> Problem._evaluate()
chromosome -> decode_chromosome() -> MultiPathLayout
MultiPathLayout -> compute_speed_profile() -> F[], G[]
results -> visualization + run_info.md + category_report.md -> outputs/
```

---

## Development Workflows

### Adding Track Pieces
1. Add to `data/track_pieces_v2.yaml` (V2 port-centric schema: `ports`, `routes`, `kind`)
2. Add to `_CANONICAL_PIECE_INDEX` in `src/catalog/catalog.py` and `PieceIndex` in `src/encoding.py` with the next index
3. If switch: add template to `src/templates.py`
4. Run `/test catalog` and `/test templates` to validate

### Adding Objectives/Constraints
1. Implement in `src/problem.py` (`_evaluate` and helpers)
2. Update `n_obj` or `n_ieq_constr` in `TrackOptimizationProblem.__init__`
3. Update `SOFT_CONSTRAINT_COUNT` in `src/algorithm/runner.py` if the new G is soft, and the `constraints.csv` header in `save_results`
4. Run `/review src/problem.py` then `/test problem`

### After Any Code Change
1. `/test` — verify nothing broke
2. `python -m pycodestyle src tests main.py run_v1_all_configs.py run_ablation.py score_ablation.py` — style gate, must exit 0
3. `/review` — quick check for interface/convention issues
4. `/quality src/<file>.py` — deep quality gate for new or refactored code (rewrites to project standards)

### After Optimization Run
1. `/diag` — parse outputs and get diagnostic report

### Full Config Validation
Use the `config-test-runner` agent — runs ALL configs with full optimization, visually inspects layouts, validates chromosomes, checks that switches→branches, crossings→crossings, all layouts closed and connected.

---

## Code Map (one-line purpose per module)

**Note**: configs in `configs/*.yaml` are CLI inputs (`main.py --config <path>`) — they are NOT imported from Python, so "no .py reference" is NOT evidence of being dead. Re-grep any symbol immediately before deleting it.

**Entry points**
- `main.py` — CLI: load config + catalog, run optimization, save results + run_info.
- `run_v1_all_configs.py` — batch runner: every config → `outputs/<config_name>/`.

**`src/` core**
- `problem.py` — `TrackOptimizationProblem`: weighted-utilization + expected-traversal-time objectives, 5+T inequality constraints.
- `encoding.py` — `PartitionedDimensions`, `compute_dimensions`, `generate_bounds`, gene accessors (`get_junction`, `get_cross_junction`, `get_double_crossover`, flips), chromosome construction + validation.
- `decoder/construction.py` — `decode_chromosome()`: injection pipeline + 2^J path enumeration; `decoder/types.py` — `DecoderConfig`, `InventoryTracker`, `ValidatedJunction`.
- `geometry.py` — `compute_fk_chain` (vectorized FK), `compute_closure_metrics`, plus the single-loop `Layout`/`build_layout()` still consumed by tests, train/, and viz.
- `intersection.py` — vectorized self-intersection scan (`find_crossing_pairs`, `count_segment_crossings`), `cross_pair_perpendicular` (single definition of a valid CROSS_90 crossing), dangling-port counters.
- `templates.py` — passing-siding templates (LEFT/RIGHT), reversed-OUT merge-FK derivation, DC route/port tables, siding geometry validation + inventory helpers.
- `sampling.py` — `IntegerSampling`: heuristic seed families (loops/ovals/racetracks/sidings/figure-8s) + random chromosomes; `_figure_eight_main_loop` is shared with operators.
- `operators.py` — `PartitionedCrossover`, `PartitionedMutation` + sub-operator portfolio (incl. `_compensated_pair_grow`, `_grow_dc_figure_eight`).
- `repair.py` — `JunctionValidityRepair`, `InventoryRepair`, `MainLoopClosureRepair` (angular + translational), `BoundaryAwareRepair`, chained by `TrackRepairPipeline`.
- `types.py` — pure dataclasses: `SwitchPair`, `CrossJunction`, `DblCrossover`, `TraversalPath` (incl. `piece_uids` physical-piece identity), `MultiPathLayout`, `PieceClass`, `FKRoute`, `PieceTopology`.
- `config.py` — Pydantic models `OptimizationConfig` / `BoundaryConfig` / `AlgorithmConfig` / `TerminationConfig`.
- `run_info.py` — per-run provenance writer (`run_info.md`: git state, verbatim config, run summary).
- `lego_track_models.py` — R40 / rail geometry constants for the renderer.

**`src/catalog/`** — `catalog.py` (`TrackCatalog.load()`, vectorized FK/radius/topology tables — no speed data, v2 port-centric schema only; raises on non-v2, on piece ids outside the canonical map, on missing canonical pieces, and on stud-vs-mm radius drift; per-route arc lengths derived from each route's endpoint pose); `loader.py` (ruamel + Pydantic with file:line error UX); `pieces.py` (`FKDeltas`, `Port`, `TrackPiece`); `specs.py` (Pydantic v2 schema).

**`src/train/`** — `physics.py` (`TrainConfig`, derailment caps, capped friction-circle `available_accel` = `min(cap, sqrt((mu*g)^2 - a_lat^2))`); `scoring.py` (`compute_speed_profile`, 3-pass profiler — route-aware radii AND arc lengths, triple unroll for closed loops); `evaluation.py` (`PhysicalEvaluation`, `evaluate_layout()` — full physical evaluation; scoring is the building block, evaluation the orchestrator).

**`src/algorithm/`** — `runner.py` (`run_optimization()`, `save_results()`, callbacks: `ProgressCallback`, `FeasibleEliteCallback`, `CategoryEliteArchive`, `SnapshotCallback`, `CallbackChain`, `LegoAdaptiveEpsilon`, category report writer); `monitoring.py` (`ConvergenceMonitorCallback`: HV/IGD/feasibility + cumulative feasible front).

**`src/visualization/`** — `track_renderer.py` (`plot_layout()` — the sole renderer: one full-size track view plus a metrics/legend info panel, closure tolerances required as keyword args so the drift overlay is gated by the optimizer's own thresholds; piece geometry drawn via the shared `_draw_piece`/`_draw_piece_sequence` helpers); `pareto_plot.py` (`plot_pareto_front()` built on pymoo `Scatter`).

### Stale / Needs Action (NOT auto-delete)

- **The v1 catalog `data/track_pieces.yaml` is gone; the kit is v2-only** (`data/track_pieces_v2.yaml`). The v1 deprecation test and the `test_catalog_parity.py` v1↔v2 parity suite have both been retired.
- **`Layout` / `build_layout()` in `src/geometry.py`** — legacy, but tests (`test_geometry.py`, `test_evaluation.py`, `test_scoring.py`) and `train/` still consume them (and `problem.py` builds per-route `Layout` views). Migration first, deletion second.
- **Top-level research docs** (`Literature-Grounded Audit ...md`, `Structurally Similar Problems ...md`, `Modular9PartResearchV1/`) — user-authored research with no code links; ask before deleting.

### Test Suite Notes (2026-08-18)

- 41 `tests/test_*.py` files, 442 `def test_` functions, 478 collected tests. Baseline: **0 failed, 478 passed, 0 skipped** (clean).
- The 4-way `test_catalog*.py` split (catalog / geometry / loader / specs) is justified — distinct scopes.
- All fixtures under `tests/fixtures/` are referenced by `test_catalog_loader.py`.

### Untouched (Run Artifacts — Do Not Delete)

All GA run output goes under `outputs/` (single gitignored tree). `main.py` writes to `outputs/` root; `run_v1_all_configs.py` writes to `outputs/<config_name>/`. Milestone runs worth keeping live under `archive/{dc,crossing,baselines}/` (also gitignored). Treat both as build output; clean only on explicit user instruction.
