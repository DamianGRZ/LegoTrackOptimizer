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

- **`data/track_pieces.yaml`**: Track catalog with FK deltas, physics, ports, piece_index
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

## Implementation Details

### Chromosome Encoding (Multi-Segment)

Fixed-length array: **N_VAR = 218 genes**

| Segment | Range | Length | Purpose |
|---------|-------|--------|---------|
| Main Loop | [0, 100) | 100 | Piece indices (-1 = inactive, 0-9 = piece index) |
| Switch Mask | [100, 200) | 100 | Switch type at each position (-1 = no switch) |
| Branch Slots | [200, 216) | 16 | 4 slots × 4 genes (IN_pos, handedness, n_straights, active) |
| Start Position | [216, 218) | 2 | start_x, start_y coordinates |

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

### Problem Classes (problem.py)

| Class | Objectives | Constraints | Use Case |
|-------|------------|-------------|----------|
| `TrackLayoutProblem` | 2 (utilization, speed) | 5 | Legacy bi-objective NSGA-II |
| `MultiSegmentProblem` | 1 (pieces × 1000 - penalties) | 4 | Single-objective with multi-path |
| `SingleObjectiveProblem` | 1 (pieces × 1000) | 5 | Single-objective piece maximization |

### Objectives (2, all minimized - legacy mode)
```python
F[0] = -utilization  # Maximize piece usage
F[1] = -avg_speed    # Maximize speed
```

### Constraints (5, g <= 0 feasible)
```python
G[0] = (closure_error - tolerance) / tolerance
G[1] = (angle_error - tolerance) / tolerance
G[2] = boundary_violation / diagonal
G[3] = inventory_excess
G[4] = loose_port_count  # Switches with unconnected port 2 (decoder-computed)
```

### Physics Model (from Locomotive_dynamics.md)
```python
# evaluation.py - compute_speed_limit()
v = SF * sqrt(mu * g * R)   # SF=0.8, mu=0.30
# Forward-backward pass for time-optimal speed profile
```

---

## Template-Based Passing Sidings

Standard LEGO passing siding pattern:
```
[IN_switch] -> [approach_curve] -> [straights×N] -> [return_curve] -> [OUT_switch]
```

| Template | IN Switch | OUT Switch | Approach Curve | Return Curve |
|----------|-----------|------------|----------------|--------------|
| LEFT_SIDING | R40_SWITCH_LEFT_IN (5) | R40_SWITCH_LEFT_OUT (6) | R40_RIGHT | R40_LEFT |
| RIGHT_SIDING | R40_SWITCH_RIGHT_IN (7) | R40_SWITCH_RIGHT_OUT (8) | R40_LEFT | R40_RIGHT |

The decoder automatically:
1. Detects IN/OUT switch pairs in the main loop
2. Computes branch geometry using templates
3. Enumerates all 2^N traversal paths (N = number of switch pairs)
4. Tracks loose ports (unconnected switch port 2)

---

## Data Flow

```
track_pieces.yaml -> TrackCatalog (FK tables, speed limits, routes)
configs/*.yaml -> OptimizationConfig (inventory, boundary, algorithm)
main.py -> GA/NSGA2 + MultiSegmentSampling -> Problem._evaluate()
chromosome -> decode_chromosome() -> MultiPathLayout
MultiPathLayout -> compute_speed_profile() -> F[], G[]
results -> visualization -> outputs/
```

---

## Key Classes

### TrackCatalog (data.py)
- `_fk_table`: (n, 3) array of [dx, dy, dtheta]
- `_speed_table`, `_radius_table`, `_arc_length_table`
- `get_fk(indices)`, `get_fk_route(piece_idx, route_idx)`
- `get_radii(indices)`, `get_speed_limits(indices)`

### Layout (geometry.py)
- `indices`: piece indices used
- `states`: (n+1, 3) cumulative [x, y, theta]
- `closure_error`, `angle_error`, `bounding_box`, `is_closed()`

### MultiPathLayout (topology.py)
- `main_loop_pieces`: base piece sequence
- `switch_pairs`: list of matched SwitchPair objects
- `paths`: list of TraversalPath (2^N paths for N switch pairs)
- `loose_port_count`: switches with unconnected port 2

### SpeedProfile (evaluation.py)
- `speeds`: speed at each segment
- `avg_speed`, `lap_time`, `total_distance`

---

## HeuristicSampling

Seeds 15% of population with inventory-valid closed loop patterns:
- Simple circles (16 R40 curves)
- Symmetric ovals (R40 curves + straights)
- Racetracks (4 corners + straights)
- Ovals with passing sidings (LEFT or RIGHT switches)

Validates patterns against inventory before use.

---

## Track Piece Index Mapping

| Index | Piece ID | Description |
|-------|----------|-------------|
| 0 | STRAIGHT_16 | 16-stud straight |
| 1 | STRAIGHT_24 | 24-stud straight |
| 2 | R40_LEFT | 22.5 deg left curve |
| 3 | R40_RIGHT | 22.5 deg right curve |
| 4 | CROSS_90 | 90 deg crossing |
| 5 | R40_SWITCH_LEFT_IN | Left switch IN (diverges left) |
| 6 | R40_SWITCH_LEFT_OUT | Left switch OUT (merges from left) |
| 7 | R40_SWITCH_RIGHT_IN | Right switch IN (diverges right) |
| 8 | R40_SWITCH_RIGHT_OUT | Right switch OUT (merges from right) |
| 9 | DOUBLE_CROSSOVER | 4 independent switches, 48x24 studs |

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
1. Add to `data/track_pieces.yaml` with `fk`, `physics`, `ports`, `routes` (for multi-port pieces)
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
