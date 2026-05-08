# Modular architecture for an evolutionary LEGO track optimizer

**A multi-objective evolutionary optimization system for LEGO train track layouts decomposes most cleanly into nine cohesive packages arranged in three architectural tiers: a framework-agnostic domain core (catalog, geometry, physics), an EA-integration layer (chromosome, decoder, operators, problem), and an infrastructure shell (configuration, visualization, I/O).** This decomposition draws on canonical patterns from six major EC frameworks, hexagonal architecture principles, and rail-simulation software structure. The critical design decision is isolating the `train` physics module so it depends on nothing outside the domain core — enabling independent unit testing, reuse, and caching — while the pymoo `Problem` subclass becomes a thin orchestration wrapper that calls the decoder and physics engine in sequence.

The analysis below synthesizes architectural patterns from pymoo, DEAP, jMetalPy, LEAP, BRKGA-MP-IPR, and Inspyred; clean-architecture principles applied to scientific Python codebases; rail-dynamics software structure from OpenTrack, Simpack, and RAILSIM; and domain-specific decomposition patterns for construction-based decoders, segment-aware operators, and YAML-driven catalogs.

---

## How EC frameworks define their module boundaries

Six evolutionary computation frameworks reveal a remarkably consistent set of **seven canonical package boundaries**: problem definition, algorithm driver, genetic operators, population/solution representation, performance indicators, visualization, and utilities. The pymoo framework (Blank & Deb, 2020, DOI: 10.1109/ACCESS.2020.2990567) organizes these under three first-level abstractions — Problems, Optimization (algorithms + operators), and Analytics (indicators + visualization + decision-making) — with a `core/` package housing abstract base classes (`Problem`, `Algorithm`, `Crossover`, `Mutation`, `Sampling`, `Repair`, `Survival`, `Callback`) from which all concrete implementations inherit.

The idiomatic pymoo project layout expects user code to follow a **constructor-injection composition** pattern: the user subclasses `ElementwiseProblem` (overriding `_evaluate` to set `out["F"]` and `out["G"]`), implements custom operators as subclasses of `pymoo.core.Crossover` / `pymoo.core.Mutation`, and assembles the algorithm by passing these objects to the `NSGA2()` constructor. The algorithm itself is a black box that calls operators in sequence; the user controls behavior entirely through the injected components. This plug-and-play assembly is also the pattern in jMetalPy (Benítez-Hidalgo et al., 2019, DOI: 10.1016/j.swevo.2019.100598) and Inspyred, though DEAP (Fortin et al., 2012) uses a toolbox-registration pattern with explicit user-written evolutionary loops instead.

**One critical finding concerns decoder modules.** Among the six frameworks surveyed, only two — LEAP (Coletti et al., 2020, DOI: 10.1145/3377929.3398147) and BRKGA-MP-IPR (Andrade et al., 2021, DOI: 10.1016/j.ejor.2019.11.037) — treat the genotype-to-phenotype decoder as an explicit, first-class architectural component. In pymoo, DEAP, jMetalPy, and Inspyred, decoding is silently embedded inside the evaluation function. For the LEGO track system, where a complex construction-based decoder guarantees feasibility, extracting the decoder into its own module is not merely a style preference — it is an architectural necessity that enables independent testing, caching, and reuse. The BRKGA-MP-IPR framework establishes the gold-standard pattern: the decoder is the *sole* problem-dependent component, injected into a problem-agnostic algorithm. Even though this system uses NSGA-II rather than BRKGA, the same decoder-extraction principle applies.

---

## Hexagonal architecture maps naturally onto simulation-optimization systems

The hexagonal (ports-and-adapters) architecture, originally formulated by Alistair Cockburn (2005), divides a system into a technology-agnostic domain core surrounded by abstract ports and concrete adapters. All source-code dependencies point inward. Robert C. Martin's Clean Architecture (2012) refines this into concentric layers: **entities** (pure domain data and rules), **use cases** (orchestration logic), **interface adapters**, and **frameworks/drivers**. Both models map cleanly onto a simulation-optimization pipeline.

For the LEGO track optimizer, the mapping is:

The **domain core** (innermost) contains the track catalog data model, geometric transform functions, and the physics engine — all pure Python with no framework dependencies. The **use-case layer** contains the decoder (orchestrating catalog + geometry to build a layout) and the objective evaluator (orchestrating decoder + physics to compute fitness). The **adapter layer** contains the pymoo `Problem` subclass (translating between pymoo's `_evaluate` protocol and the domain's evaluation functions), the YAML configuration loader, the matplotlib visualization module, and any checkpoint/export I/O. The **framework layer** is pymoo itself, NumPy, matplotlib, and PyYAML — external libraries that the codebase depends on but never leaks into the domain core.

This layering enforces **the fundamental dependency rule**: the physics module never imports pymoo; the catalog never imports matplotlib; the decoder never imports the configuration loader. Dependencies flow inward. The practical benefit is testability — every domain-core module can be unit-tested with simple pytest assertions and no mock framework, because it has zero external dependencies. OpenMDAO (NASA) and PyGMO (ESA, DOI: 10.21105/joss.02338) demonstrate this same principle at scale: the optimizer sees the simulation as a callable black box, and the simulation knows nothing about the optimization strategy driving it.

The Scientific Python Development Guide (learn.scientific-python.org/development/) recommends the **`src/` layout** for new packages, where the package lives under `src/package_name/` rather than at the repository root. This prevents accidental import of development code and ensures tests run against the installed package. Major packages like NumPy and scikit-learn use flat layout for historical reasons, but the `src/` convention is now the standard recommendation from pyOpenSci and PyPA.

---

## The physics module mirrors established rail-simulation decomposition

Rail-simulation software consistently decomposes into **four sub-modules** feeding a central engine. OpenTrack (Nash & Huerlimann, 2004, WIT Press) uses three input modules — rolling stock (traction curves, mass, adhesion), infrastructure (track geometry, speed limits, gradients), and timetable — feeding a simulation engine that solves the equation of motion. Simpack Rail (Dassault Systèmes) further decomposes into wheel-rail contact, drivetrain dynamics, track flexibility, and wear modules. RAILSIM's Train Performance Calculator similarly separates track geometry from rolling-stock databases.

For the LEGO train physics module, the analogous decomposition yields five internal components:

**Lateral-force model** (`lateral_force.py`): A pure mathematical function `v_max = sqrt(μ·g·R)` computing the maximum safe speed for a given curve radius. This is a stateless, deterministic function — the ideal candidate for `@functools.lru_cache` decoration. Because the set of LEGO track radii is finite (fewer than 10 distinct values), the cache hit rate approaches **100%** after warmup, making the cache effectively a startup-built lookup table.

**Motor model** (`motor_model.py`): Empirical LEGO Powered Up motor data — torque-speed curves loaded from a YAML or JSON data file. This mirrors OpenTrack's rolling-stock tractive-effort/speed diagrams. The module exposes interpolation functions: given a target speed, return available torque and power. Frozen dataclasses (`MotorSpec`) represent motor parameters, enabling hashability for downstream caching.

**Speed table** (`speed_table.py`): A per-radius safe-speed lookup constructed at startup by combining the lateral-force model with the motor model's maximum-speed constraint. This is the "derived constant" layer — computed once from catalog radii and motor specs, then frozen. The table is a `dict[float, float]` mapping radius to `min(v_lateral, v_motor)`.

**Scoring / aggregation** (`scoring.py`): The length-weighted harmonic-mean safe traversal speed (space-mean speed). This function takes arrays of segment lengths and per-segment speeds, returning a single scalar. It is naturally vectorizable: `total_length / Σ(length_i / speed_i)`. Both a single-layout function and a NumPy-vectorized batch function belong here.

**Engine** (`engine.py`): The orchestration facade — `PhysicsEngine` class with `evaluate(layout) → PhysicsResult` and `evaluate_batch(layouts) → np.ndarray`. This follows the **dual-API pattern** from scikit-learn, where the batch interface enables population-level vectorized evaluation inside pymoo's `_evaluate`, while the single-instance interface supports unit testing and interactive exploration. The engine composes the other four sub-modules but adds no physics logic of its own.

The critical architectural constraint is that **this entire `train/` package imports nothing from the EA layer**. It depends only on the catalog (for piece specs and radii) and on the geometry module (for decoded layout data). Following MuJoCo's design (Todorov et al., 2012, IEEE/RSJ IROS), the model definition (track catalog, motor specs) is separated from the evaluation computation — the model is immutable data passed in; the engine is a pure function over that data.

Caching belongs **inside** the physics module, not as external middleware. The `@lru_cache` decorator on `lateral_force.safe_speed()` and the startup-built speed table in `speed_table.py` are implementation details invisible to callers. Frozen dataclasses enable this by being hashable — two `TrackSegment` objects with identical fields hash identically, making them safe `lru_cache` keys. The **~2.4× instantiation overhead** of frozen versus mutable dataclasses (benchmarked in Python 3.12) is negligible because layout objects are created O(population_size) times per generation, not in tight inner loops.

---

## Catalog and geometry form the shared domain foundation

The track catalog is best modeled as a **three-layer data pipeline**: raw YAML at rest, derived indices frozen at load time, and runtime geometric computations on demand.

The **raw layer** is a YAML file defining each track piece: piece ID, port definitions (local offset, heading, type), arc radius, arc angle, and length. This file is the single source of truth, edited by the user, version-controlled, and never modified at runtime.

The **derived layer** is a `TrackCatalog` frozen dataclass constructed once at startup by parsing the YAML and precomputing lookup structures: piece-ID-to-index maps, per-category piece lists, chromosome dimension constants (how many genes the main-loop segment needs given the catalog size), and the set of distinct radii (passed to the physics module for speed-table construction). This follows the **Flyweight pattern** — the catalog stores intrinsic shared data (piece geometry), while placed pieces carry extrinsic state (position, orientation). The CAD software analogy is exact: SolidWorks separates the part definition (template) from part instances (placed occurrences with transforms).

The **runtime layer** lives in a separate `geometry/` package containing pure mathematical functions for 2D transforms and forward kinematics. A `Pose2D` frozen dataclass `(x, y, heading)` represents turtle state. Functions like `advance_straight(pose, length) → Pose2D` and `advance_arc(pose, radius, angle) → Pose2D` are stateless — no mutable turtle object, just `old_state → new_state` transformations. The `kinematics.py` module composes these into a layout-assembly function that walks a piece sequence and accumulates global port positions. Port geometry (local offsets and headings) is defined in the catalog; global port positions are computed at decode time. This separation prevents coupling geometric definitions to placement logic.

---

## The decoder bridges chromosome semantics and feasible layouts

The flat `int16` chromosome with four segments (main loop sequence, switch mask, branch slots, crossing overlay) requires a **SegmentMap** metadata object that encapsulates index ranges and extraction logic. This frozen dataclass is constructed once from catalog dimensions and passed to both the decoder and the genetic operators — it is the single source of truth for "which indices mean what."

```
SegmentMap
├── main_loop:      slice(0, N_main)
├── switch_mask:    slice(N_main, N_main + N_switch)
├── branch_slots:   slice(N_main + N_switch, N_main + N_switch + N_branch)
├── crossing_overlay: slice(...)
└── total_length:   int
```

The **decoder** is an independent class — not embedded in the Problem's `_evaluate`. It takes `(chromosome, catalog, segment_map)` and returns a `DecodedLayout` containing placed pieces with global positions, the computed track graph, and segment-level geometric data. The construction heuristic that guarantees >99% feasibility lives entirely within this class. The BRKGA-MP-IPR framework (Andrade et al., 2021) establishes this as canonical: the decoder is the sole problem-dependent component, injected into a problem-agnostic algorithm. Extracting it enables three benefits: **(1)** the decoder can be tested with deterministic chromosome inputs without running the EA, **(2)** the decoder can be reused with a different optimizer (e.g., simulated annealing) without modification, and **(3)** the Problem class becomes a thin three-line wrapper.

The **pymoo Problem** subclass then orchestrates the pipeline: decode the chromosome, evaluate physics on the decoded layout, evaluate constraints, and write results to `out["F"]` and `out["G"]`. Deb's feasibility-first constraint handling is automatic in pymoo — the user supplies raw constraint values via `out["G"]` (positive = violated), and pymoo internally computes the aggregate constraint violation (CV) and uses it in survival selection. Geometric feasibility constraints handled by the decoder (loop closure, port compatibility) should *not* be re-reported as CV; only residual soft constraints belong in `out["G"]`.

---

## Genetic operators live alongside the chromosome, not inside the problem

pymoo's operator architecture is strictly compositional: custom operators are independent class instances injected into the algorithm constructor. The **segment-selective BRKGA-style crossover** subclasses `pymoo.core.Crossover`, receives the `SegmentMap` in its constructor, and uses it to select per-segment parent inheritance in `_do()`. It knows nothing about track layouts — only about segment boundaries in the flat vector.

The **ALNS-UCB adaptive mutation suite** is architecturally more complex because it requires post-evaluation feedback to update bandit scores. The recommended decomposition is:

- **Five mutation operators** as standalone callables: each takes a chromosome array and `SegmentMap`, returns a mutated chromosome. These are pure functions.
- **UCB selector** (`ucb.py`): Maintains per-operator reward estimates and selection counts. Exposes `select() → operator_index` and `update(index, reward)`.
- **ALNSMutationSuite** subclasses `pymoo.core.Mutation`: in `_do()`, it calls `ucb.select()` per individual, applies the chosen operator, and records which operator was used. Reward feedback flows through a pymoo `Callback` that, after each generation's evaluation, compares pre- and post-mutation fitness and calls `ucb.update()`.

This pattern is validated by the `alns` Python library (Wouda & Lan, 2023, DOI: 10.21105/joss.05028), which similarly registers destroy/repair operators as callables and plugs in MAB-based selection schemes.

---

## Cross-cutting concerns anchor at the infrastructure boundary

**Configuration** follows the domain-driven pattern: config *schemas* (dataclasses defining what the domain needs) live in the domain layer; config *loading* (YAML parsing, CLI argument handling) lives in the infrastructure adapter layer. The domain module never imports `yaml` — it receives a populated `RunConfig` dataclass from the entry point. This is the approach used in Hydra/OmegaConf-based ML projects and recommended by Clean Architecture.

**Visualization** is a strict consumer of domain objects. The `visualization/` package imports domain types (NumPy arrays, `DecodedLayout`, `PhysicsResult`) but is never imported by domain code. Functions accept data and return matplotlib `Figure` objects. Declaring matplotlib as an optional dependency (under `[project.optional-dependencies.viz]`) ensures the core package works without it. The **observer pattern** enables live plotting: the domain emits events via pymoo's `Callback` mechanism; visualization subscribers update plots. The domain never references plotting code.

**Checkpointing and artifacts** belong in an `io/` package. pymoo provides built-in checkpointing via `Callback`, but custom serialization of the decoder state, UCB bandit scores, and run metadata requires a dedicated module. This is infrastructure — it depends on the domain, not the reverse.

**Testing boundaries** follow directly from the module split. Domain-core modules (catalog, geometry, physics) are tested with fast, deterministic unit tests — no pymoo, no YAML files, no matplotlib. The decoder gets property-based tests (using Hypothesis): verify that `decode(chromosome)` always returns a layout with valid geometry, that encoding round-trips preserve semantics, and that segment extraction is consistent. Integration tests verify that the pymoo Problem correctly wires decoder → physics → objectives. End-to-end tests run a short EA on a small catalog and verify convergence.

---

## Proposed package tree with dependency contracts

The recommended `src/` layout for the thesis codebase, with dependency direction enforced:

```
src/lego_track_optimizer/
│
├── catalog/                    # TIER 1: DOMAIN CORE
│   ├── __init__.py             #   (no EA, no pymoo, no matplotlib)
│   ├── pieces.py               # TrackPieceSpec, PortDef (frozen dataclasses)
│   ├── loader.py               # YAML → raw piece data parsing
│   └── catalog.py              # TrackCatalog: registry + derived indices
│
├── geometry/                   # TIER 1: DOMAIN CORE
│   ├── __init__.py             #   depends on: catalog
│   ├── transforms.py           # Pose2D, arc_transform, straight_transform
│   └── kinematics.py           # Forward kinematics: piece sequence → layout
│
├── train/                      # TIER 1: DOMAIN CORE — FULLY ISOLATED
│   ├── __init__.py             #   depends on: catalog (radii), geometry (layout types)
│   ├── types.py                #   NEVER imports pymoo, operators, decoder, viz
│   ├── lateral_force.py        # v_max = sqrt(μgR), @lru_cache
│   ├── motor_model.py          # Powered Up torque-speed curves
│   ├── speed_table.py          # Per-radius safe speed lookup (startup-frozen)
│   ├── scoring.py              # Space-mean speed: scalar + vectorized
│   └── engine.py               # PhysicsEngine.evaluate() / evaluate_batch()
│
├── chromosome/                 # TIER 2: EA-INTEGRATION LAYER
│   ├── __init__.py             #   depends on: catalog (dimensions)
│   └── segment_map.py          # SegmentMap: index ranges for 4 segments
│
├── decoder/                    # TIER 2: EA-INTEGRATION LAYER
│   ├── __init__.py             #   depends on: catalog, geometry, chromosome
│   ├── types.py                # DecodedLayout, PlacedPiece dataclasses
│   └── construction.py         # ConstructionDecoder.decode(chromosome) → Layout
│
├── operators/                  # TIER 2: EA-INTEGRATION LAYER
│   ├── __init__.py             #   depends on: chromosome (SegmentMap)
│   ├── crossover.py            #   depends on: pymoo.core.crossover
│   ├── mutation.py             # Five ALNS mutation callables
│   └── ucb.py                  # UCB bandit selector + reward tracking
│
├── problem/                    # TIER 2: EA-INTEGRATION LAYER
│   ├── __init__.py             #   depends on: decoder, train, chromosome
│   ├── track_problem.py        #   depends on: pymoo.core.problem
│   └── constraints.py          # Constraint evaluation → out["G"]
│
├── algorithm/                  # TIER 3: INFRASTRUCTURE SHELL
│   ├── __init__.py             #   depends on: problem, operators, pymoo
│   └── runner.py               # Algorithm assembly + minimize() wrapper
│
├── visualization/              # TIER 3: INFRASTRUCTURE SHELL
│   ├── __init__.py             #   depends on: domain types only
│   ├── track_renderer.py       #   depends on: matplotlib
│   ├── pareto_plot.py          # Objective-space scatter
│   └── convergence.py          # Convergence history
│
├── config/                     # TIER 3: INFRASTRUCTURE SHELL
│   ├── __init__.py
│   ├── schemas.py              # RunConfig, PhysicsConfig dataclasses
│   └── loader.py               # YAML → config dataclass adapter
│
└── io/                         # TIER 3: INFRASTRUCTURE SHELL
    ├── __init__.py
    ├── checkpoint.py           # EA state save/restore
    └── export.py               # Result export (CSV, JSON)
```

Supporting directories at the repository root:

```
data/
├── catalogs/
│   └── track_pieces.yaml       # Track piece definitions
├── motors/
│   └── powered_up.yaml         # Empirical motor curves
└── configs/
    └── default.yaml            # Default run configuration

tests/
├── unit/
│   ├── test_catalog/           # Piece loading, index derivation
│   ├── test_geometry/          # Transform math, kinematics
│   ├── test_train/             # Physics: lateral force, speed table, scoring
│   ├── test_chromosome/        # SegmentMap extraction
│   └── test_decoder/           # Construction heuristic, feasibility
├── integration/
│   ├── test_problem/           # Problem._evaluate wiring
│   └── test_operators/         # Crossover/mutation with pymoo
└── e2e/
    └── test_optimization.py    # Short EA run on small catalog
```

The **dependency graph** is strictly layered with no cycles:

| Module | Depends on | Never imports |
|---|---|---|
| `catalog` | PyYAML (loader only) | geometry, train, pymoo, matplotlib |
| `geometry` | catalog, NumPy | train, pymoo, decoder, matplotlib |
| `train` | catalog, geometry, NumPy | pymoo, decoder, operators, viz |
| `chromosome` | catalog, NumPy | train, geometry, pymoo |
| `decoder` | catalog, geometry, chromosome, NumPy | train, pymoo, operators |
| `operators` | chromosome, pymoo.core, NumPy | catalog, geometry, train, decoder |
| `problem` | decoder, train, chromosome, pymoo.core | operators, algorithm, viz |
| `algorithm` | problem, operators, pymoo | catalog, geometry, train directly |
| `visualization` | domain dataclasses, matplotlib, NumPy | pymoo, decoder, operators |
| `config` | PyYAML, domain schemas | everything else |
| `io` | domain types, json/csv | pymoo, matplotlib |

---

## The `train` module's scope, API, and isolation guarantees

The `train/` package is the physics module whose independence is paramount. Its **public API** consists of exactly two entry points on the `PhysicsEngine` class:

- `evaluate(layout: DecodedLayout) → PhysicsResult` — single-layout evaluation for testing, interactive use, and debugging. Returns a frozen dataclass with `mean_speed`, `segment_speeds`, `max_speed`, and `feasibility_flag`.
- `evaluate_batch(layouts: Sequence[DecodedLayout]) → np.ndarray` — vectorized population-level evaluation returning a `(pop_size,)` array of space-mean speeds, designed to be called from `_evaluate()` on an entire generation.

Internally, the engine orchestrates a **four-stage pipeline** analogous to OpenTrack's simulation: (1) extract per-segment radii from the decoded layout, (2) look up per-segment safe speeds from the prebuilt speed table, (3) clip speeds against motor constraints, (4) aggregate via length-weighted harmonic mean. Steps 1–3 are vectorized across segments; step 4 is vectorized across the population.

**Isolation contract**: `train/` imports only from `catalog.pieces` (for `TrackPieceSpec` type) and `geometry.transforms` (for `Pose2D` / layout types). It never imports `pymoo`, `decoder`, `operators`, `config`, `visualization`, or `io`. This is enforced by the dependency table above and can be verified with tools like `import-linter` or `deptry`. The physics module is a **standalone, pip-installable sub-package** in principle — it could be extracted into its own repository and used in a completely different optimization context (e.g., a brute-force enumerator or a reinforcement-learning agent) without modification.

---

## Pitfalls and anti-patterns to avoid

**Circular imports between decoder and physics.** The most common failure mode is the decoder calling the physics engine to evaluate feasibility during construction, while the physics engine imports decoder types. The solution: the decoder produces a `DecodedLayout` (a pure geometric data structure with no physics information); the physics engine consumes this layout. The two modules communicate only through shared domain types defined in `catalog` and `geometry`, never through each other.

**The God Problem class.** When `_evaluate()` contains decoding logic, physics computation, constraint evaluation, and visualization hooks in a single 200-line method, every change touches every concern. The fix is the three-way decomposition: `_evaluate` calls `self.decoder.decode(x)`, then `self.engine.evaluate(layout)`, then `self.constraint_evaluator.evaluate(layout)`, each in a separate module.

**Operator-problem coupling.** If crossover or mutation operators import the Problem class to access catalog or segment information, a circular dependency emerges. Operators should receive a `SegmentMap` at construction time and operate on raw chromosome arrays — they need no knowledge of track layouts, physics, or feasibility.

**Leaky caching.** Placing `@lru_cache` on methods of mutable objects breaks when object state changes between calls. The system avoids this by using frozen dataclasses throughout the physics module and caching only pure functions with hashable arguments. The `lru_cache` decorators live on module-level functions (e.g., `lateral_force.safe_speed`), never on methods of stateful objects.

**Visualization importing domain internals.** If `track_renderer.py` imports `ConstructionDecoder` to re-decode chromosomes for display, it creates an unnecessary dependency on the decoder module. Instead, visualization functions should accept already-decoded `DecodedLayout` objects — the caller (algorithm runner or notebook) is responsible for decoding before passing data to the renderer.

---

## Conclusion

The proposed nine-package architecture achieves three properties that a monolithic design cannot. First, the physics module is **a pure function of geometry** — it computes speeds from radii and lengths with no knowledge of chromosomes, crossover operators, or Pareto fronts, making it trivially testable and reusable. Second, the decoder is **a first-class module** in the BRKGA tradition, bridging the gap between the flat chromosome and the geometric domain without polluting either side. Third, the pymoo integration layer is **thin and replaceable** — the `TrackLayoutProblem` class is fewer than 20 lines of wiring code, meaning a future migration to a different optimizer (or the addition of a second optimizer for comparison experiments) requires changes only in `problem/` and `algorithm/`, leaving the domain core untouched.

The architectural patterns converge from multiple independent sources: the BRKGA-MP-IPR framework's decoder-centric design, hexagonal architecture's dependency-inward rule, OpenTrack's rolling-stock/infrastructure/engine decomposition, MuJoCo's immutable-model/mutable-workspace split, and scikit-learn's dual single/batch evaluation interface. For a master's thesis, this modular design not only produces cleaner code but also provides natural section boundaries for the implementation chapter — each package maps to a subsection with clear responsibilities, interfaces, and testability guarantees.