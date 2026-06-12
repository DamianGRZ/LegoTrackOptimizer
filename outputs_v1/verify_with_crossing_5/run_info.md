# Run Info — 2026-06-12T07:42:04

## Code State

- **Commit**: `07dd539 Fixed the "each gen takes longer" problem`
- **Branch**: `main`

**Uncommitted changes** (`git status --porcelain`):

```
 M configs/all_pieces.yaml
 M configs/with_crossing.yaml
 M src/algorithm/runner.py
 M src/config.py
 M src/decoder/construction.py
 M src/problem.py
 M src/repair.py
 M src/sampling.py
 M src/types.py
 M tests/test_crossing_seed_families.py
 M tests/test_runner_wiring.py
?? docs/superpowers/specs/2026-06-11-category-elite-archive-design.md
?? outputs_v1/smoke_category/
?? outputs_v1/verify_all_pieces_categories/
?? outputs_v1/verify_with_crossing/
?? outputs_v1/verify_with_crossing_2/
?? outputs_v1/verify_with_crossing_3/
?? outputs_v1/verify_with_crossing_4/
?? tests/test_category_archive.py
?? tests/test_closure_target.py
```

**Diff stat** (`git diff HEAD --stat`):

```
 configs/all_pieces.yaml              |   2 +-
 configs/with_crossing.yaml           |   2 +-
 src/algorithm/runner.py              | 237 +++++++++++++++++++++++++++++++++--
 src/config.py                        |   7 +-
 src/decoder/construction.py          |  52 +++++++-
 src/problem.py                       |  11 ++
 src/repair.py                        |  38 +++++-
 src/sampling.py                      |  74 ++++++++---
 src/types.py                         |   3 +
 tests/test_crossing_seed_families.py |  75 +++++++++++
 tests/test_runner_wiring.py          |  17 +++
 11 files changed, 479 insertions(+), 39 deletions(-)
```

## Configuration

- **Config file**: `configs\with_crossing.yaml`
- **Total inventory**: 202 pieces

**Verbatim contents of `configs\with_crossing.yaml` at run time:**

```yaml
# LEGO Track Optimizer - Configuration with 90-Degree Crossing
# Adds CROSS_90 for figure-8 / self-intersecting layouts. No switches/DC here —
# use with_switches_and_crossing for switches+crossing, all_pieces for the full kit.

train_config_path: trains/default.yaml

inventory:
  # Straights
  #STRAIGHT_24: 8
  STRAIGHT_16: 120
  # R40 curves
  R40_CURVE: 80
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
  n_gen: 300
  heuristic_ratio: 0.20
  crossover_prob: 0.95
  mutation_prob: 0.05
  eliminate_duplicates: true
  seed: null
  termination:
    n_max_gen: 1000
    period: 0  # Early stop if no improvement for 100 generations

closure_tolerance: 4.0
angle_tolerance: 5.0
boundary_tolerance: 2.0
n_workers: 16  # Enable parallel evaluation
```

## Run Summary

- Generations: 300
- Population: 1000
- Feasible solutions: 1000/1000
- **Best feasible**: 118 pieces, util=58.9%, speed=0.99 m/s, switches=0
- **Best overall (feasible)**: 118 pieces, util=58.9%, speed=0.99 m/s, switches=0, CV=0.00

**Piece usage** (best feasible):

  - `CROSS_90`: 2/2
  - `R40_CURVE`: 24/80
  - `STRAIGHT_16`: 92/120
