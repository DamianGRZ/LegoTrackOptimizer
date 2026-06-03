# Run Info — 2026-06-03T10:37:02

## Code State

- **Commit**: `681740d chore(repo): consolidate run artifacts under single outputs/ tree`
- **Branch**: `main`

**Uncommitted changes** (`git status --porcelain`):

```
 M CLAUDE.md
AM configs/all_pieces.yaml
 M configs/with_switches_and_crossing.yaml
 M src/algorithm/runner.py
 M src/config.py
 M src/decoder/construction.py
 M src/encoding.py
 M src/intersection.py
 M src/operators.py
 M src/problem.py
 M src/sampling.py
 M src/templates.py
 M src/types.py
 M tests/test_cross_junction_inject.py
 M tests/test_crossing_repair.py
 M tests/test_problem.py
 M tests/test_sampling.py
?? .claude/commands/code-map-audit.md
?? .claude/commands/inspect-layout.md
?? .claude/commands/verify-run.md
?? "docs/NSGA-II Representations and Operators for Variable-Branching Chromosomes A Specialist Review for LEGO-Style Modular Track Layout.md"
?? docs/superpowers/specs/2026-05-31-cross90-direct-placement-design.md
?? outputs_v1/
?? tests/test_cross90_descriptor.py
?? tests/test_cross90_objective.py
?? tests/test_dc_grow.py
```

**Diff stat** (`git diff HEAD --stat`):

```
 CLAUDE.md                               | 114 ++++++++-----
 configs/all_pieces.yaml                 |  46 ++++++
 configs/with_switches_and_crossing.yaml |   2 +-
 src/algorithm/runner.py                 |  19 ++-
 src/config.py                           |   6 +
 src/decoder/construction.py             | 277 +++++++++-----------------------
 src/encoding.py                         |  61 ++++---
 src/intersection.py                     |  63 +++++---
 src/operators.py                        |  76 +++++++++
 src/problem.py                          | 116 +++++++++++--
 src/sampling.py                         |  81 +++-------
 src/templates.py                        | 130 +--------------
 src/types.py                            |  35 ++--
 tests/test_cross_junction_inject.py     | 183 +++++++++++----------
 tests/test_crossing_repair.py           | 183 ++++++---------------
 tests/test_problem.py                   | 183 +++++++++++++++------
 tests/test_sampling.py                  |  22 +--
 17 files changed, 798 insertions(+), 799 deletions(-)
```

## Configuration

- **Config file**: `configs\all_pieces.yaml`
- **Total inventory**: 210 pieces

**Verbatim contents of `configs\all_pieces.yaml` at run time:**

```yaml
# LEGO Track Optimizer - Configuration with Switches
# Includes R40 switch tracks for layouts with turnout points

train_config_path: trains/default.yaml

inventory:
  # Straights
  #STRAIGHT_24: 8
  STRAIGHT_16: 120
  # R40 curves
  R40_CURVE: 80
  # R40 Switches: each passing siding consumes 1 LEFT + 1 RIGHT (opposite-handed pair).
  R40_SWITCH_LEFT: 3
  R40_SWITCH_RIGHT: 3
  # 90-degree crossing (allows track to cross itself)
  CROSS_90: 2
  DOUBLE_CROSSOVER: 2
boundary:
  min_x: -250.0
  max_x: 250.0
  min_y: -250.0
  max_y: 250.0

#boundary:
#  min_x: -100.0
#  max_x: 100.0
#  min_y: -100.0
#  max_y: 100.0

algorithm:
  name: NSGA2
  pop_size: 1000
  n_gen: 1000
  heuristic_ratio: 0.20
  crossover_prob: 0.5
  mutation_prob: 0.5
  eliminate_duplicates: true
  seed: null
  termination:
    n_max_gen: 1000
    period: 100  # Early stop if no improvement for 100 generations

closure_tolerance: 4.0
angle_tolerance: 5.0
boundary_tolerance: 2.0
n_workers: 16  # Enable parallel evaluation
```

## Run Summary

- Generations: 1000
- Population: 1000
- Feasible solutions: 1000/1000
- **Best feasible**: 122 pieces, util=59.0%, speed=1.00 m/s, switches=1
- **Best overall (feasible)**: 122 pieces, util=59.0%, speed=1.00 m/s, switches=1, CV=0.00

**Piece usage** (best feasible):

  - `CROSS_90`: 0/2
  - `DOUBLE_CROSSOVER`: 0/2
  - `R40_CURVE`: 18/80
  - `R40_SWITCH_LEFT`: 1/3
  - `R40_SWITCH_RIGHT`: 1/3
  - `STRAIGHT_16`: 102/120
