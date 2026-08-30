# LEGO Track Optimizer

A multi-objective genetic algorithm that designs **closed LEGO/4DBrix railway layouts**
from a fixed box of track pieces. Given an inventory and a rectangular boundary, it
searches for layouts that maximize **piece utilization** and minimize the **expected
time to traverse the whole network** while staying geometrically valid — the loop
must close, fit inside the boundary, respect the inventory, and avoid illegal
self-intersections.

Built on [pymoo](https://pymoo.org/) (NSGA-II) with a custom domain-specific
encoding, decoder, genetic operators, repair pipeline, and a locomotive physics
model.

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
cover their whole network in less time, while enforcing buildability by construction
and repair.

---

## Key features

- **Bi-objective optimization** — maximizes piece utilization while minimizing the
  expected traversal time of the *whole* network (every physical piece counts, so
  branch geometry actually matters — and the two goals genuinely conflict).
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

A candidate layout makes a short journey from raw genes to a scored verdict: an
inventory-shaped **chromosome** is built into exact track by a **decoder**, two
**objectives** judge it, a handful of **constraints** decide whether it could really
be built, and an **evolutionary loop** breeds the survivors. The stages below follow
that journey.

```
data/track_pieces_v2.yaml ─► TrackCatalog (FK tables, radii, routes; no speed data)
configs/*.yaml            ─► OptimizationConfig (inventory, boundary, algorithm)
                                      │
main.py ─► NSGA-II + IntegerSampling ─► Problem._evaluate()
                                      │
   chromosome ─► decode_chromosome() ─► MultiPathLayout
                                      │
   MultiPathLayout ─► compute_speed_profile() ─► objectives F[], constraints G[]
                                      │
   results ─► PNG layouts + Pareto plot + run_info.md + category_report.md ─► outputs/
```

### 1. Representation — the chromosome

A layout is one integer (`int16`) vector, but never a flat bag of numbers: it is
**partitioned** into meaningful regions, and every region's size is computed from your
inventory at startup — never hardcoded. Unused slots are switched off with an
`INACTIVE` (−1) sentinel, so a tight 40-piece loop and a sprawling 130-piece figure-8
coexist and compete inside the same fixed-width genome:

| Segment | Encodes |
|---|---|
| Main-loop types | piece type per slot (`-1`=inactive, `STRAIGHT_16`, `STRAIGHT_24`, `R40_CURVE`) |
| Main-loop flips | per-slot R40 handedness bit (left/right turn) |
| Passing-siding junctions | `(active, position, handedness, n_straights)` per slot |
| Cross-junctions | `(active, pos_1, pos_2)` per `CROSS_90` |
| Double-crossovers | `(active, pos_1, route_1, pos_2, route_2)` per `DOUBLE_CROSSOVER` |
| Start position | fine `(x, y)` offset on top of auto-centering |

Switches and crossings are deliberately **not** legal main-loop alleles — letting
mutation scatter them at random would be hopeless. They live in dedicated descriptor
blocks instead, stitched in (and validated as a unit) by the decoder.

### 2. Decoder — from genes to geometry

The decoder is the heart of the system, and it is fully **deterministic**: the same
genes always trace the same track. Piece by piece it composes each part's
`[dx, dy, θ]` offset through forward kinematics (a vectorized cumulative sum) to find
where the rails actually go — then injects passing sidings from validated templates,
drops in cross-junctions and double-crossovers, repairs perpendicular
straight-on-straight overlaps into real `CROSS_90` pieces, enumerates every one of the
`2^J` ways a train could thread the sidings, and centers the result in the boundary
box. Anything that fails a geometry or inventory check is dropped — never allowed to
corrupt the layout — with the reason logged.

### 3. Objectives & constraints — what "good" means

Two goals pull on every layout, and they genuinely conflict:

- **Use more of the kit** — `F[0]` rewards weighted piece utilization, counting each
  siding, crossing, and double-crossover as more than one piece; otherwise the GA
  would strip multi-path topology away as dead weight.
- **Cover the network fast** — `F[1]` is the expected time to traverse every
  physical piece once: each of the `2^J` routes is speed-profiled (at a 0.95 safety
  margin), each piece is charged the mean of its traversal times across the routes
  that pass it, and the objective sums over distinct pieces. Every siding branch and
  bypassed straight costs real seconds, so density trades directly against time.

That second goal used to be *speed* rather than time: the layout was scored by the
average pace of its slowest route, and the GA was asked to push that pace up. It does not
work, for a reason worth recording. Average pace **improves as a layout grows** — a bare
16-piece circle averages 0.84 m/s because it is all curve, while a 96-piece racetrack
averages 1.01 m/s because long straights let the train reach its motor cap. So the speed
goal pulled in the *same* direction as "use more of the kit" instead of against it, and
two objectives that agree leave nothing for a trade-off curve to form on. Whole fronts
came out inside a 0.03 m/s sliver. Measured across the run archive, replacing pace with
time moved the median Pareto front from **2 points** (182 runs) to **39** (1137 runs). On
the full-inventory config the front is now continuous from a 16-piece circle (7.3% of the
kit, 2.39 s) to a 59.2% layout at 15.73 s.

Buildability is enforced separately, as **inequality constraints** (there are no
equality constraints): the loop must close in `x`, `y`, and heading; everything must
fit the boundary; segments may not illegally overlap; and no piece type may exceed its
inventory. A layout is feasible only when every one of these holds.

### 4. Operators, repair & sampling — how candidates evolve

Off-the-shelf genetic operators would shred this structured genome, so the project
ships its own:

- **Crossover** respects the partitions — one-point on the main loop (with the flip
  array cut in lockstep), uniform per-slot swaps on descriptors — and keeps delicate
  double-crossover figure-8s intact rather than splicing them apart.
- **Mutation** is a weighted portfolio of small, purposeful moves: change a piece,
  flip a curve's handedness, nudge a siding, straighten near an unresolved crossing,
  or grow the loop by a closure-exact "compensated pair" of straights that cancels its
  own displacement.
- **Repair** is a 4-stage cleanup that walks a broken layout back toward feasibility:
  clamp descriptors → trim over-budget pieces → close the loop (first by angle, then
  by position) → re-center or shrink to fit the box.
- **Sampling** seeds the first generation with a fifth of inventory- and
  boundary-aware ready-made closed loops (ovals, racetracks, sidings, figure-8s); the
  rest are random partial-fill chromosomes, so the search starts from buildable
  bridgeheads, not noise.

### 5. Algorithm — the evolutionary loop

Everything runs on pymoo's **NSGA-II** with Deb's feasibility-first ranking
(`ConstrRankAndCrowding`) and the custom sampling, crossover, mutation, and repair
plugged in. The crowding metric that breaks ties inside a front is selectable; the
default is NSGA-II's original crowding distance, and pymoo's suggested alternative for
two-objective problems (pruning crowding distance) was measured across every ablation
arm and came out indistinguishable from it, at equal cost. An adaptive epsilon handler relaxes the *soft* constraints (closure,
boundary) early in a run — so promising-but-imperfect layouts survive long enough to
be repaired — while the *hard* ones (collisions, inventory) stay strict throughout.
Category-elite callbacks protect the best switch, crossing, and double-crossover
layouts from being out-competed by simpler loops.

---

## Track pieces

The kit is **R40-only** (no R56/R104). `R40_CURVE` is one physical SKU; left vs. right
turn is chosen per placement via a flip bit. Switches keep separate LEFT/RIGHT entries
because their port geometry is not a simple mirror.

| Index | Piece | Geometry |
|---|---|---|
| 0 | `STRAIGHT_16` | 16-stud straight |
| 1 | `STRAIGHT_24` | 24-stud straight |
| 2 | `R40_CURVE` | 22.5° curve (16 per circle); direction = flip bit |
| 3 | `CROSS_90` | 90° crossing (FK identical to `STRAIGHT_16`) |
| 4 | `R40_SWITCH_LEFT` | left switch, 32-stud body; through straight / diverge R40 arc |
| 5 | `R40_SWITCH_RIGHT` | right switch, 32-stud body; through straight / diverge R40 arc |
| 6 | `DOUBLE_CROSSOVER` | 48×16, 4 routes (2 through + 2 diagonal) |

Geometry lives in `data/track_pieces_v2.yaml` (a port-centric schema derived from
4DBrix part dimensions). Speed caps are not catalog data: they are derived at runtime
from train physics (`src/train/physics.py`) — straights bind at the measured motor cap,
R40-radius segments at the lateral-slide cap.

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
# Default config (NSGA-II, pop 1000, 200 generations)
python main.py --config configs/default.yaml --verbose

# Layouts with passing sidings (NSGA-II, 500 generations)
python main.py --config configs/with_switches.yaml --verbose

# Fast smoke test (20 generations, pop 20)
python main.py --config configs/default.yaml --quick-test

# Reproducible run (override the config seed)
python main.py --config configs/default.yaml --seed 42
```

| Flag | Description | Default |
|---|---|---|
| `--config` | Path to a YAML configuration file | `configs/default.yaml` |
| `--output` | Output directory for results | auto-named `outputs/verify_<config>_<N>` (never overwrites) |
| `--verbose` | Verbose logging (per-generation progress) | off |
| `--quick-test` | Override to 20 generations / pop 20 for a quick check | off |
| `--seed` | Override `algorithm.seed` for a reproducible run | config value |

Each `main.py` invocation runs **one** optimization and writes its artifacts into the
output directory. If a run crashes mid-flight it is **salvaged**, not lost: the live
population is decoded into a partial result, the traceback is saved to `error.md`, and
the process exits non-zero.

### Batch all configs

```bash
python run_v1_all_configs.py    # every config → outputs/<config_name>/
```

Per-config failures are isolated, so one crashing config no longer aborts the batch.

### Seeded replications

For decision-grade comparisons, run one config across many seeds and get a
median ± IQR summary (one subprocess per seed, so artifacts and crash handling match a
normal single run):

```bash
python run_replications.py --config default --seeds 1..10   # → outputs/<config>_s<seed>/
python run_replications.py --config with_switches --seeds 1,2,5 --force
```

---

## Configuration

Configs are plain YAML loaded via `--config`. They are **constraints, not tuning
knobs** — the boundary box and inventory describe a real, fixed kit and space; the
optimizer works *within* them.

```yaml
train_config_path: trains/measured_consist.yaml   # locomotive physics

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
  name: NSGA2             # only NSGA2 is supported
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
- `pareto_front.png` — the run-level Pareto front (utilization vs. traversal time),
  both axes normalized to the range the run actually spanned, with 1 = best.
- `chromosomes.csv` / `constraints.csv` — final-population genomes and constraint values.
- `convergence.csv` — per-generation telemetry appended live (HV, IGD, feasibility,
  best objectives, unique-`F` counts, epsilon, and generation wall-time).
- `run_info.md` — provenance (git state, verbatim config, run summary — including
  executed-vs-planned generations and the termination reason).
- `category_report.md` — per-category (switch / crossing / double-crossover) elites.
- `snapshots/` — progression renders captured during the run.
- `error.md` / `error.log` — written only when a run crashes; the traceback that
  accompanies the salvaged partial result.

Each `save_results` block is isolated, so a single failing render can no longer cost
the rest of the artifacts.

---

## Project structure

```
main.py                     CLI entry point (load config + catalog, run, save)
run_v1_all_configs.py        Batch runner over every config
run_replications.py          Seeded-replication harness (one config × many seeds)
configs/                     Optimization configs (inventory, boundary, algorithm)
data/track_pieces_v2.yaml    Track-piece catalog (FK, radii, routes)
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
  catalog/                   Catalog loading + FK/radius/topology tables
  train/                     Locomotive physics + speed profiler
  algorithm/                 run_optimization(), callbacks, monitoring
  visualization/             Layout & Pareto-front renderers
tests/                       pytest suite (~400 tests)
```

---

## Testing

```bash
pytest -q                    # full suite (~400 tests)
pytest -q tests/test_decoder.py
pytest -q --tb=line          # compact failure output
```

The suite covers the catalog, geometry, decoder, operators, repair, sampling,
problem objectives/constraints, the train physics model, and the visualization paths.

A style gate enforces PEP 8 (99-char line limit, configured in `setup.cfg`):

```bash
python -m pycodestyle src tests main.py run_v1_all_configs.py    # must exit 0
```

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
