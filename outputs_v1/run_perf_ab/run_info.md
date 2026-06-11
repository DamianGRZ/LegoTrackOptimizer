# Run Info — 2026-06-11T22:13:43

## Code State

- **Commit**: `0328770 Double Crossover rendering fixed`
- **Branch**: `main`

**Uncommitted changes** (`git status --porcelain`):

```
 M configs/all_pieces.yaml
 M configs/with_double_crossover.yaml
A  docs/superpowers/plans/2026-06-07-selective-epsilon-hard-collisions.md
A  docs/superpowers/specs/2026-06-07-selective-epsilon-hard-collisions-design.md
MM src/algorithm/runner.py
 M src/decoder/construction.py
 M src/geometry.py
 M src/intersection.py
 M src/repair.py
A  tests/test_selective_epsilon.py
?? docs/superpowers/plans/2026-06-07-boundary-aware-repair.md
?? outputs_v1/run13/snapshots/snapshot_03_feasible.png
?? outputs_v1/run13/snapshots/snapshot_03_infeasible.png
?? outputs_v1/run13/snapshots/snapshot_04_feasible.png
?? outputs_v1/run13/snapshots/snapshot_04_infeasible.png
?? outputs_v1/run13/snapshots/snapshot_05_feasible.png
?? outputs_v1/run13/snapshots/snapshot_05_infeasible.png
?? outputs_v1/run14/
?? outputs_v1/run15/
?? outputs_v1/run16/
?? outputs_v1/run17/
?? outputs_v1/run18/
?? outputs_v1/run19/
?? outputs_v1/run20/
?? outputs_v1/run_brepair/
?? outputs_v1/run_brepair_quick/
?? outputs_v1/run_perf_smoke/
?? tests/test_boundary_repair.py
?? tests/test_closure_translation.py
?? tests/test_runner_wiring.py
?? tests/test_vectorized_equivalence.py
```

**Diff stat** (`git diff HEAD --stat`):

```
 configs/all_pieces.yaml                            |   4 +-
 configs/with_double_crossover.yaml                 |  10 +-
 ...2026-06-07-selective-epsilon-hard-collisions.md | 322 +++++++++++++++++++++
 ...-07-selective-epsilon-hard-collisions-design.md | 141 +++++++++
 src/algorithm/runner.py                            | 126 +++++++-
 src/decoder/construction.py                        |  56 ++--
 src/geometry.py                                    |  28 +-
 src/intersection.py                                | 157 +++++-----
 src/repair.py                                      | 280 ++++++++++++++++--
 tests/test_selective_epsilon.py                    |  67 +++++
 10 files changed, 1027 insertions(+), 164 deletions(-)
```

## Configuration

- **Config file**: `configs\with_double_crossover.yaml`
- **Total inventory**: 162 pieces

**Verbatim contents of `configs\with_double_crossover.yaml` at run time:**

```yaml
# LEGO Track Optimizer - Configuration with DOUBLE_CROSSOVER
#
# Mirrors configs/with_switches.yaml, with switch inventory replaced by
# DOUBLE_CROSSOVER. The optimizer reaches feasible no-dangling layouts via
# the figure-8 and two-layer-loop heuristic seeds; the GA then refines
# around them.

train_config_path: trains/default.yaml

inventory:
  STRAIGHT_16: 80
  R40_CURVE: 80
  DOUBLE_CROSSOVER: 2

boundary:
  min_x: -250.0
  max_x: 250.0
  min_y: -250.0
  max_y: 250.0

algorithm:
  name: NSGA2
  pop_size: 500
  n_gen: 200
  heuristic_ratio: 0.20
  crossover_prob: 0.9
  mutation_prob: 0.1
  eliminate_duplicates: false
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

- Generations: 200
- Population: 500
- Feasible solutions: 494/500
- **Best feasible**: 96 pieces, util=59.3%, speed=1.00 m/s, switches=0
- **Best overall (infeasible)**: 107 pieces, util=66.0%, speed=0.89 m/s, switches=0, CV=0.54

**Piece usage** (best feasible):

  - `DOUBLE_CROSSOVER`: 0/2
  - `R40_CURVE`: 16/80
  - `STRAIGHT_16`: 80/80
