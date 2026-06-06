# Run Info — 2026-06-06T13:00:53

## Code State

- **Commit**: `67d03ec Some fixed Cross_90`
- **Branch**: `main`

**Uncommitted changes** (`git status --porcelain`):

```
 M .gitignore
 D .superpowers/brainstorm/1454-1780498836/state/server-info
A  .superpowers/brainstorm/1454-1780498836/state/server-stopped
A  .superpowers/brainstorm/755-1780501870/content/refinements.html
A  .superpowers/brainstorm/755-1780501870/content/topologies-v2.html
A  .superpowers/brainstorm/755-1780501870/content/topologies.html
A  .superpowers/brainstorm/755-1780501870/content/waiting-2.html
A  .superpowers/brainstorm/755-1780501870/content/waiting.html
A  .superpowers/brainstorm/755-1780501870/state/server-info
A  .superpowers/brainstorm/755-1780501870/state/server.pid
AM configs/all_pieces_150x150.yaml
AM configs/all_pieces_350x350.yaml
 M configs/with_crossing.yaml
A  outputs_v1/run2/best_layout.png
A  outputs_v1/run2/chromosomes.csv
A  outputs_v1/run2/constraints.csv
A  outputs_v1/run2/fitness.csv
A  outputs_v1/run2/pareto_front.png
A  outputs_v1/run2/run_info.md
A  outputs_v1/run2/snapshots/snapshot_01_feasible.png
A  outputs_v1/run2/snapshots/snapshot_01_infeasible.png
A  outputs_v1/run2/snapshots/snapshot_02_feasible.png
A  outputs_v1/run2/snapshots/snapshot_02_infeasible.png
A  outputs_v1/run2/snapshots/snapshot_03_feasible.png
A  outputs_v1/run2/snapshots/snapshot_03_infeasible.png
A  outputs_v1/run2/snapshots/snapshot_04_feasible.png
A  outputs_v1/run2/snapshots/snapshot_04_infeasible.png
A  outputs_v1/run2/snapshots/snapshot_05_feasible.png
A  outputs_v1/run2/snapshots/snapshot_05_infeasible.png
A  outputs_v1/run2/snapshots/snapshot_06_feasible.png
A  outputs_v1/run2/snapshots/snapshot_06_infeasible.png
A  outputs_v1/run2/snapshots/snapshot_07_feasible.png
A  outputs_v1/run2/snapshots/snapshot_07_infeasible.png
A  outputs_v1/run2/snapshots/snapshot_08_feasible.png
A  outputs_v1/run2/snapshots/snapshot_08_infeasible.png
A  outputs_v1/run2/snapshots/snapshot_09_feasible.png
A  outputs_v1/run2/snapshots/snapshot_09_infeasible.png
A  outputs_v1/run2/snapshots/snapshot_10_feasible.png
A  outputs_v1/run3/run_info.md
A  outputs_v1/run3/snapshots/snapshot_01_feasible.png
A  outputs_v1/run3/snapshots/snapshot_01_infeasible.png
A  outputs_v1/run3/snapshots/snapshot_02_feasible.png
A  outputs_v1/run3/snapshots/snapshot_02_infeasible.png
A  outputs_v1/run3/snapshots/snapshot_03_feasible.png
A  outputs_v1/run3/snapshots/snapshot_03_infeasible.png
 M src/decoder/construction.py
 M src/sampling.py
 M src/visualization/track_renderer.py
 M tests/test_catalog.py
 M tests/test_catalog_loader.py
 M tests/test_catalog_parity.py
 M tests/test_dbl_crossover_inject.py
 M tests/test_evaluation.py
 M tests/test_sampling.py
 M tests/test_templates.py
?? docs/superpowers/plans/2026-06-03-space-filling-crossing-topologies-phase1.md
?? docs/superpowers/specs/2026-06-03-space-filling-crossing-topologies-design.md
?? outputs_v1/run1/
?? outputs_v1/run10/
?? outputs_v1/run11/
?? outputs_v1/run4/
?? outputs_v1/run5/
?? outputs_v1/run6/
?? outputs_v1/run7/
?? outputs_v1/run8/
?? outputs_v1/run9/
?? outputs_v1/verify_crossing_seeds_1/
?? outputs_v1/verify_crossing_seeds_1_log.txt
?? tests/test_crossing_seed_families.py
?? tests/test_dc_render.py
?? tests/test_seed_geometry_harness.py
```

**Diff stat** (`git diff HEAD --stat`):

```
 .gitignore                                         |    3 +
 .../brainstorm/1454-1780498836/state/server-info   |    1 -
 .../1454-1780498836/state/server-stopped           |    1 +
 .../755-1780501870/content/refinements.html        |   53 ++
 .../755-1780501870/content/topologies-v2.html      |   73 ++
 .../755-1780501870/content/topologies.html         |   62 ++
 .../755-1780501870/content/waiting-2.html          |    3 +
 .../brainstorm/755-1780501870/content/waiting.html |    3 +
 .../brainstorm/755-1780501870/state/server-info    |    1 +
 .../brainstorm/755-1780501870/state/server.pid     |    1 +
 configs/all_pieces_150x150.yaml                    |   46 +
 configs/all_pieces_350x350.yaml                    |   46 +
 configs/with_crossing.yaml                         |   32 +-
 outputs_v1/run2/best_layout.png                    |  Bin 0 -> 212499 bytes
 outputs_v1/run2/chromosomes.csv                    | 1000 +++++++++++++++++++
 outputs_v1/run2/constraints.csv                    | 1001 ++++++++++++++++++++
 outputs_v1/run2/fitness.csv                        | 1001 ++++++++++++++++++++
 outputs_v1/run2/pareto_front.png                   |  Bin 0 -> 50317 bytes
 outputs_v1/run2/run_info.md                        |   86 ++
 outputs_v1/run2/snapshots/snapshot_01_feasible.png |  Bin 0 -> 215233 bytes
 .../run2/snapshots/snapshot_01_infeasible.png      |  Bin 0 -> 185561 bytes
 outputs_v1/run2/snapshots/snapshot_02_feasible.png |  Bin 0 -> 217550 bytes
 .../run2/snapshots/snapshot_02_infeasible.png      |  Bin 0 -> 180950 bytes
 outputs_v1/run2/snapshots/snapshot_03_feasible.png |  Bin 0 -> 215509 bytes
 .../run2/snapshots/snapshot_03_infeasible.png      |  Bin 0 -> 232250 bytes
 outputs_v1/run2/snapshots/snapshot_04_feasible.png |  Bin 0 -> 217465 bytes
 .../run2/snapshots/snapshot_04_infeasible.png      |  Bin 0 -> 237386 bytes
 outputs_v1/run2/snapshots/snapshot_05_feasible.png |  Bin 0 -> 216549 bytes
 .../run2/snapshots/snapshot_05_infeasible.png      |  Bin 0 -> 214277 bytes
 outputs_v1/run2/snapshots/snapshot_06_feasible.png |  Bin 0 -> 214132 bytes
 .../run2/snapshots/snapshot_06_infeasible.png      |  Bin 0 -> 241156 bytes
 outputs_v1/run2/snapshots/snapshot_07_feasible.png |  Bin 0 -> 214083 bytes
 .../run2/snapshots/snapshot_07_infeasible.png      |  Bin 0 -> 241646 bytes
 outputs_v1/run2/snapshots/snapshot_08_feasible.png |  Bin 0 -> 215858 bytes
 .../run2/snapshots/snapshot_08_infeasible.png      |  Bin 0 -> 232188 bytes
 outputs_v1/run2/snapshots/snapshot_09_feasible.png |  Bin 0 -> 214075 bytes
 .../run2/snapshots/snapshot_09_infeasible.png      |  Bin 0 -> 152245 bytes
 outputs_v1/run2/snapshots/snapshot_10_feasible.png |  Bin 0 -> 215787 bytes
 outputs_v1/run3/run_info.md                        |   70 ++
 outputs_v1/run3/snapshots/snapshot_01_feasible.png |  Bin 0 -> 212824 bytes
 .../run3/snapshots/snapshot_01_infeasible.png      |  Bin 0 -> 195356 bytes
 outputs_v1/run3/snapshots/snapshot_02_feasible.png |  Bin 0 -> 214985 bytes
 .../run3/snapshots/snapshot_02_infeasible.png      |  Bin 0 -> 195258 bytes
 outputs_v1/run3/snapshots/snapshot_03_feasible.png |  Bin 0 -> 213110 bytes
 .../run3/snapshots/snapshot_03_infeasible.png      |  Bin 0 -> 264064 bytes
 src/decoder/construction.py                        |    2 +
 src/sampling.py                                    |   11 +-
 src/visualization/track_renderer.py                |  152 ++-
 tests/test_catalog.py                              |   24 +-
 tests/test_catalog_loader.py                       |    8 -
 tests/test_catalog_parity.py                       |    7 +
 tests/test_dbl_crossover_inject.py                 |    3 +-
 tests/test_evaluation.py                           |    7 +-
 tests/test_sampling.py                             |    4 +-
 tests/test_templates.py                            |    6 +-
 55 files changed, 3612 insertions(+), 95 deletions(-)
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
