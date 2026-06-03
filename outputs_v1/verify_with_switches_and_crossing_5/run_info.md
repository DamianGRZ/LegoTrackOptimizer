# Run Info — 2026-05-30T14:56:12

## Code State

- **Commit**: `681740d chore(repo): consolidate run artifacts under single outputs/ tree`
- **Branch**: `main`

**Uncommitted changes** (`git status --porcelain`):

```
 M CLAUDE.md
 M configs/with_switches_and_crossing.yaml
 M src/problem.py
 M tests/test_problem.py
?? .claude/commands/code-map-audit.md
?? .claude/commands/inspect-layout.md
?? .claude/commands/verify-run.md
?? "docs/NSGA-II Representations and Operators for Variable-Branching Chromosomes A Specialist Review for LEGO-Style Modular Track Layout.md"
?? outputs_v1/
```

**Diff stat** (`git diff HEAD --stat`):

```
 CLAUDE.md                               |   3 +
 configs/with_switches_and_crossing.yaml |   2 +-
 src/problem.py                          |  72 ++++++++++---
 tests/test_problem.py                   | 183 ++++++++++++++++++++++++--------
 4 files changed, 202 insertions(+), 58 deletions(-)
```

## Configuration

- **Config file**: `configs\with_switches_and_crossing.yaml`
- **Total inventory**: 208 pieces

**Verbatim contents of `configs\with_switches_and_crossing.yaml` at run time:**

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
  n_gen: 500
  heuristic_ratio: 0.30
  crossover_prob: 0.8
  mutation_prob: 0.2
  eliminate_duplicates: true
  seed: null
  termination:
    n_max_gen: 1000
    period: 100  # Early stop if no improvement for 100 generations

closure_tolerance: 4.0
angle_tolerance: 5.0
boundary_tolerance: 2.0
n_workers: 32  # Enable parallel evaluation
```

## Run Summary

- Generations: 500
- Population: 1000
- Feasible solutions: 1000/1000
- **Best feasible**: 130 pieces, util=62.5%, speed=1.00 m/s, switches=1
- **Best overall (feasible)**: 130 pieces, util=62.5%, speed=1.00 m/s, switches=1, CV=0.00

**Piece usage** (best feasible):

  - `CROSS_90`: 0/2
  - `R40_CURVE`: 18/80
  - `R40_SWITCH_LEFT`: 1/3
  - `R40_SWITCH_RIGHT`: 1/3
  - `STRAIGHT_16`: 110/120
