# LEGO Track Optimizer

A multi-objective genetic algorithm that designs **closed LEGO/4DBrix railway layouts**
from a fixed box of track pieces. Given an inventory and a rectangular boundary, it
searches for layouts that maximize **piece utilization** and **train speed** while
staying geometrically valid — the loop must close, fit inside the boundary, respect
the inventory, and avoid illegal self-intersections.

Built on [pymoo](https://pymoo.org/) (NSGA-II / R-NSGA-II) with a custom
domain-specific encoding, decoder, genetic operators, repair pipeline, and a
locomotive physics model.

---

## Table of Contents

- [Why this exists](#why-this-exists)
- [Key features](#key-features)
- [How it works](#how-it-works)
- [Track pieces](#track-pieces)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Outputs](#outputs)
- [Project structure](#project-structure)
- [Testing](#testing)
- [Glossary](#glossary)
- [Known limits](#known-limits)
- [Tech stack](#tech-stack)

---

## Why this exists

Designing a closed model-railway loop by hand is a constraint puzzle: every curve
turns the track, every straight moves it, and the last piece has to meet the first
one in both **position** and **heading**. Add passing sidings, crossings, and a
finite box of parts, and the search space explodes. This project treats layout
design as a **constrained multi-objective optimization problem** and lets an
evolutionary algorithm explore it — rewarding layouts that use more of the kit and
let a train run faster, while enforcing buildability by construction and repair.

---

## Key features

- **Bi-objective optimization** — simultaneously maximizes piece utilization and the
  speed of the *slowest* traversal route (so branch geometry actually matters).
- **Inventory-driven, variable-size search** — the chromosome length is derived from
  the inventory at runtime; it is never hardcoded. Unused capacity is encoded with an
  `INACTIVE` sentinel, so topologies of different sizes compete in one population.
- **Construction-based decoder** — a deterministic genotype→geometry map using
  vectorized forward kinematics; descriptors that fail geometric/inventory validation
  are dropped, not allowed to corrupt the layout.
- **Template-based passing sidings** — opposite-handed switch pairs injected from
  validated templates, with all `2^J` take/skip-siding routes enumerated.
- **Crossings & double-crossovers** — figure-8 topologies via `CROSS_90` and
  `DOUBLE_CROSSOVER` pieces, including emergent self-intersection repair.
- **Locomotive physics model** — a 3-pass time-optimal speed profiler with
  per-segment derailment caps (slide, tip-over, **Nadal's criterion**, motor limit).
- **Custom GA operators + repair** — partition-aware crossover/mutation and a
  4-stage repair pipeline that actively drives layouts toward closure, boundary, and
  inventory feasibility.

---

## How it works

```
data/track_pieces_v2.yaml ─► TrackCatalog (FK tables, speed limits, routes)
configs/*.yaml            ─► OptimizationConfig (inventory, boundary, algorithm)
                                      │
main.py ─► NSGA-II / R-NSGA-II + IntegerSampling ─► Problem._evaluate()
                                      │
   chromosome ─► decode_chromosome() ─► MultiPathLayout
                                      │
   MultiPathLayout ─► compute_speed_profile() ─► objectives F[], constraints G[]
                                      │
   results ─► PNG layouts + Pareto plot + run_info.md + category_report.md ─► outputs/
```

### 1. Representation (chromosome)

An integer (`int16`) vector, **partitioned into segments** whose sizes all scale from
the inventory:

| Segment | Encodes |
|---|---|
| Main-loop types | piece type per slot (`-1`=inactive, `STRAIGHT_16`, `STRAIGHT_24`, `R40_CURVE`) |
| Main-loop flips | per-slot R40 handedness bit (left/right turn) |
| Passing-siding junctions | `(active, position, handedness, n_straights)` per slot |
| Cross-junctions | `(active, pos_1, pos_2)` per `CROSS_90` |
| Double-crossovers | `(active, pos_1, route_1, pos_2, route_2)` per `DOUBLE_CROSSOVER` |
| Start position | fine `(x, y)` offset on top of auto-centering |

Switches and crossings are **not** legal main-loop alleles — they enter only through
descriptor blocks and decoder repair.

### 2. Decoder (genotype → geometry)

`decode_chromosome()` is fully deterministic. It reads the main loop, injects sidings
from templates, injects cross-junctions and double-crossovers, repairs perpendicular
straight-on-straight crossings into `CROSS_90`, computes the layout via a vectorized
forward-kinematics chain accumulating `[x, y, θ]`, enumerates all `2^J` traversal
paths, and auto-centers the result in the boundary box.

### 3. Objectives & constraints

Two objectives (both minimized internally, hence the negatives):

- `F[0] = −weighted_utilization` — fraction of the inventory used, with special
  pieces (sidings/crossings/double-crossovers) weighted so multi-path topology is
  rewarded rather than stripped as overhead.
- `F[1] = −(slowest route's average speed)` at a 0.95 safety margin.

Constraints are all **inequalities** (`g ≤ 0`; there are no equality constraints):
per-axis closure (`dx`, `dy`, `dθ`), boundary violation, collisions
(unresolved crossings + dangling ports), and per-type inventory excess.

### 4. Operators, repair & sampling

- **Crossover** — one-point on the main loop (cut mirrored on the flip array),
  uniform per-slot swap on descriptors; double-crossover loops are kept parent-intact.
- **Mutation** — a weighted portfolio of sub-operators (change/activate/deactivate/
  swap/flip, crossing-aware straighten, closure-exact compensated-pair grow).
- **Repair** — a 4-stage pipeline: junction-validity clamp → inventory trim →
  main-loop closure (angular, then translational) → boundary re-center/shrink.
- **Sampling** — a hybrid: ~20% inventory/boundary-aware closed-loop seeds
  (loops, ovals, racetracks, sidings, figure-8s) plus random partial-fill chromosomes.

### 5. Algorithm

pymoo's `NSGA2` or `RNSGA2` (reference-point niching toward the utopian corner),
with the custom sampling/crossover/mutation/repair embedded, wrapped in an adaptive
epsilon-constraint handler that relaxes *soft* constraints (closure, boundary) early
in the run while keeping *hard* ones (collisions, inventory) strict.

---

## Track pieces

The kit is **R40-only** (no R56/R104). `R40_CURVE` is one physical SKU; left vs. right
turn is chosen per placement via a flip bit. Switches keep separate LEFT/RIGHT entries
because their port geometry is not a simple mirror.

| Index | Piece | Geometry | Speed limit |
|---|---|---|---|
| 0 | `STRAIGHT_16` | 16-stud straight | 1.57 m/s |
| 1 | `STRAIGHT_24` | 24-stud straight | 1.57 m/s |
| 2 | `R40_CURVE` | 22.5° curve (16 per circle); direction = flip bit | 0.97 m/s |
| 3 | `CROSS_90` | 90° crossing (FK identical to `STRAIGHT_16`) | 1.57 m/s |
| 4 | `R40_SWITCH_LEFT` | left switch, 32-stud body | through 1.57 / diverge 0.97 m/s |
| 5 | `R40_SWITCH_RIGHT` | right switch, 32-stud body | through 1.57 / diverge 0.97 m/s |
| 6 | `DOUBLE_CROSSOVER` | 48×16, 4 routes (2 through + 2 diagonal) | through 1.57 / cross 0.97 m/s |

Geometry and speed limits live in `data/track_pieces_v2.yaml` (a port-centric schema
derived from 4DBrix part dimensions).

---

## Installation

Requires **Python 3.10+**.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

Key dependencies: `pymoo==0.6.1.6`, `numpy`, `scipy`, `pydantic>=2`, `pyyaml`,
`ruamel.yaml`, `matplotlib`, `pytest`. Matplotlib is forced to the headless `Agg`
backend (the interactive Tk backend crashes under multiprocessing).

---

## Usage

### Run an optimization

```bash
# Default config (R-NSGA-II, pop 1000, 200 generations)
python main.py --config configs/default.yaml --verbose

# Layouts with passing sidings (NSGA-II, 500 generations)
python main.py --config configs/with_switches.yaml --verbose

# Fast smoke test (20 generations, pop 20)
python main.py --config configs/default.yaml --quick-test
```

| Flag | Description | Default |
|---|---|---|
| `--config` | Path to a YAML configuration file | `configs/default.yaml` |
| `--output` | Output directory for results | `outputs` |
| `--verbose` | Verbose logging (per-generation progress) | off |
| `--quick-test` | Override to 20 generations / pop 20 for a quick check | off |

Each `main.py` invocation runs **one** optimization and writes its artifacts into the
output directory.

### Batch all configs

```bash
python run_v1_all_configs.py    # every config → outputs/<config_name>/
```

---

## Configuration

Configs are plain YAML loaded via `--config`. They are **constraints, not tuning
knobs** — the boundary box and inventory describe a real, fixed kit and space; the
optimizer works *within* them.

```yaml
train_config_path: trains/default.yaml   # locomotive physics (defaults if absent)

inventory:                # available pieces (by piece_id)
  STRAIGHT_24: 8
  STRAIGHT_16: 24
  R40_CURVE: 40

boundary:                 # rectangular build area, in studs
  min_x: -150.0
  max_x: 150.0
  min_y: -150.0
  max_y: 150.0

algorithm:
  name: RNSGA2            # RNSGA2 or NSGA2
  pop_size: 1000
  n_gen: 200
  heuristic_ratio: 0.20   # fraction of seeded heuristic individuals
  crossover_prob: 0.2
  mutation_prob: 0.8
  eliminate_duplicates: true
  seed: null              # null = non-deterministic
  # termination:          # optional improvement-aware early stop
  #   n_max_gen: 1000
  #   period: 100

closure_tolerance: 4.0    # studs
angle_tolerance: 5.0      # degrees
boundary_tolerance: 2.0   # studs
n_workers: 1              # process-pool size for parallel evaluation
```

Bundled configs in `configs/` include `default`, `compact`, `with_switches`,
`with_crossing`, `with_switches_and_crossing`, `with_double_crossover` (and `_small` /
`_narrow` variants), and several `all_pieces*` boxes.

---

## Outputs

Results are written under a single `outputs/` tree (gitignored). Depending on the run:

- `best_layout.png` — the rendered champion layout.
- `pareto_front.png` — the run-level Pareto front (utilization vs. speed).
- `chromosomes.csv` / `constraints.csv` — final-population genomes and constraint values.
- `run_info.md` — provenance (git state, verbatim config, run summary).
- `category_report.md` — per-category (switch / crossing / double-crossover) elites.
- `snapshots/` — progression renders captured during the run.

---

## Project structure

```
main.py                     CLI entry point (load config + catalog, run, save)
run_v1_all_configs.py        Batch runner over every config
configs/                     Optimization configs (inventory, boundary, algorithm)
data/track_pieces_v2.yaml    Track-piece catalog (FK, speed limits, routes)
src/
  problem.py                 TrackOptimizationProblem (objectives + constraints)
  encoding.py                Partitioned chromosome dimensions, bounds, gene access
  decoder/                   decode_chromosome(): injection pipeline + path enumeration
  geometry.py                Vectorized forward kinematics + closure metrics
  intersection.py            Self-intersection / dangling-port detection
  templates.py               Passing-siding & double-crossover geometry templates
  sampling.py                IntegerSampling: heuristic seeds + random chromosomes
  operators.py               PartitionedCrossover / PartitionedMutation
  repair.py                  4-stage repair pipeline
  types.py                   Layout / path / descriptor dataclasses
  config.py                  Pydantic config models
  catalog/                   Catalog loading + FK/speed/topology tables
  train/                     Locomotive physics + speed profiler
  algorithm/                 run_optimization(), callbacks, monitoring
  visualization/             Layout & Pareto-front renderers
tests/                       pytest suite (~370 tests)
```

---

## Testing

```bash
pytest -q                    # full suite
pytest -q tests/test_decoder.py
pytest -q --tb=line          # compact failure output
```

The suite covers the catalog, geometry, decoder, operators, repair, sampling,
problem objectives/constraints, the train physics model, and the visualization paths.

---

## Glossary

- **FK (forward kinematics)** — sequentially composing each piece's `[dx, dy, dθ]`
  to compute where the track goes; the loop closes when the chain returns to its start.
- **Passing siding** — an opposite-handed switch pair (1 LEFT + 1 RIGHT) forming a
  branch the train can take or skip; the exit switch is installed reversed.
- **Figure-8** — a self-crossing closed loop whose turning sum is 0 (or ±720°) rather
  than 360°, legalized by a `CROSS_90` or a `DOUBLE_CROSSOVER` at the crossing.
- **Compensated pair** — an anti-parallel pair of equal straights whose displacements
  cancel; the one edit that lets a closed loop gain pieces without breaking closure.
- **Soft vs. hard constraints** — closure and boundary are relaxable by the adaptive
  epsilon early in a run; collisions and inventory are never relaxed.

---

## Known limits

- The kit is **R40-only** and the catalog models the existing physical pieces; new
  piece types are intentionally out of scope.
- For a simple loop, utilization is geometrically capped by the boundary box (a loop
  can only get so big inside a fixed rectangle). Higher utilization comes from
  space-filling crossings *within the same box*, never from enlarging the box.
- Runs are non-deterministic by default (`seed: null`); set a seed for reproducibility.

---

## Tech stack

`pymoo 0.6.1.6` · `numpy` · `scipy` · `pydantic 2` · `pyyaml` / `ruamel.yaml` ·
`matplotlib (Agg)` · `pytest`.
