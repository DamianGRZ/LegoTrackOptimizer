# Run Info — 2026-06-07T21:37:27

## Code State

- **Commit**: `0328770 Double Crossover rendering fixed`
- **Branch**: `main`

**Uncommitted changes** (`git status --porcelain`):

```
A  docs/superpowers/plans/2026-06-07-selective-epsilon-hard-collisions.md
A  docs/superpowers/specs/2026-06-07-selective-epsilon-hard-collisions-design.md
MM src/algorithm/runner.py
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
?? outputs_v1/run_brepair/
?? outputs_v1/run_brepair_quick/
?? tests/test_boundary_repair.py
```

**Diff stat** (`git diff HEAD --stat`):

```
 ...2026-06-07-selective-epsilon-hard-collisions.md | 322 +++++++++++++++++++++
 ...-07-selective-epsilon-hard-collisions-design.md | 141 +++++++++
 src/algorithm/runner.py                            |  37 ++-
 src/repair.py                                      | 201 ++++++++++++-
 tests/test_selective_epsilon.py                    |  67 +++++
 5 files changed, 761 insertions(+), 7 deletions(-)
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
  pop_size: 1000
  n_gen: 500
  heuristic_ratio: 0.20
  crossover_prob: 0.2
  mutation_prob: 0.8
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
