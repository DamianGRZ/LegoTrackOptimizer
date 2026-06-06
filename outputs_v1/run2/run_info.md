# Run Info — 2026-06-04T15:36:39

## Code State

- **Commit**: `67d03ec Some fixed Cross_90`
- **Branch**: `main`

**Uncommitted changes** (`git status --porcelain`):

```
?? .superpowers/brainstorm/1454-1780498836/state/server-stopped
?? .superpowers/brainstorm/755-1780501870/
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

- Generations: 300
- Population: 1000
- Feasible solutions: 1000/1000
- **Best feasible**: 132 pieces, util=64.8%, speed=0.99 m/s, switches=2
- **Best overall (feasible)**: 132 pieces, util=64.8%, speed=0.99 m/s, switches=2, CV=0.00

**Piece usage** (best feasible):

  - `CROSS_90`: 0/2
  - `DOUBLE_CROSSOVER`: 0/2
  - `R40_CURVE`: 20/80
  - `R40_SWITCH_LEFT`: 2/3
  - `R40_SWITCH_RIGHT`: 2/3
  - `STRAIGHT_16`: 108/120
