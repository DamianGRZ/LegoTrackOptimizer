# Run Info — 2026-06-04T15:21:44

## Code State

- **Commit**: `2e31e69 feat(config): use_epsilon flag to toggle epsilon-constraint handling`
- **Branch**: `feat/crossing-seed-families`

**Uncommitted changes** (`git status --porcelain`):

```
 M configs/all_pieces.yaml
?? outputs_v1/run1/
?? outputs_v1/verify_crossing_seeds_1/
?? outputs_v1/verify_crossing_seeds_1_log.txt
```

**Diff stat** (`git diff HEAD --stat`):

```
 configs/all_pieces.yaml | 2 +-
 1 file changed, 1 insertion(+), 1 deletion(-)
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
  n_gen: 300
  heuristic_ratio: 0.20
  crossover_prob: 0.95
  mutation_prob: 0.05
  eliminate_duplicates: true
  use_epsilon: true  # pure feasibility-first; set true to re-enable epsilon relaxation
  seed: null
  termination:
    n_max_gen: 1000
    period: 100  # Early stop if no improvement for 100 generations

closure_tolerance: 4.0
angle_tolerance: 5.0
boundary_tolerance: 2.0
n_workers: 16  # Enable parallel evaluation
```
