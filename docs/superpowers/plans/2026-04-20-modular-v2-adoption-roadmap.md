# Modular V2 Adoption Roadmap

> **For agentic workers:** This is a *roadmap* covering nine sequential migration phases, not a single executable plan. Each phase corresponds to one V2 research package and must be expanded into its own bite-sized plan (via `superpowers:writing-plans`) before execution. Check-boxes below mark **phase completion**, not individual tasks.

**Source:** `Modular9PartResearchV2/` — nine research documents (April 2026) specifying a principled rewrite of the optimizer around (catalog, geometry, train, chromosome, decoder, operators, problem, visualization, config/io).

**Goal:** Migrate the current codebase to the V2 architecture in nine dependency-ordered phases, preserving a working optimizer and passing test suite between every phase.

**Architecture:** V2 replaces in-place. It is not a surface-level split: the chromosome encoding (3-segment → 4-region), the catalog schema (sections → port-centric), the decoder (construction → 7-phase), the operators (legacy → 5-op UCB/FRRMAB bandit), and the config format (YAML → TOML) all change. Intermediate generations keep running by wiring the new package behind the old public surface, then retiring the old module once its last caller is migrated.

**Tech Stack:** Python 3.11+, pymoo 0.6.1.6, numpy, Pydantic v2, ruamel.yaml (catalog), tomllib + tomli-w (config), numba (optional JIT), pytest, import-linter.

---

## Guiding Principles

1. **Ship-of-Theseus, not big-bang.** Each phase builds a new module alongside the old, switches callers under a feature flag or shim, proves equivalence via tests, then retires the old module. No phase leaves `main` broken.
2. **Test suite is the regression net.** Every phase ends with full `/test` green. Phases that touch encoding or decoder semantics add a `tests/legacy_parity/` suite that decodes the same chromosome through both old and new pipelines and asserts layout equivalence.
3. **Gene-level detail before code.** Per project memory (`feedback_chromosome_plan_depth`), chromosome and decoder phase plans must detail exact byte layouts, operator gene-level reads/writes, and decoder counter rules — not just high-level comparisons.
4. **Dynamic sizing, never hardcoded.** Per `CLAUDE.md` and `feedback_no_hardcoded_constraints`, every dimension (N_VAR, max_switches, max_branches) derives from catalog + config at runtime.
5. **pymoo best practices.** Use `IntegerRandomSampling + SBX + PM + RoundingRepair` (the verified 0.6.1.x integer pipeline); `n_ieq_constr` / `n_eq_constr`, not deprecated `n_constr`; `ElementwiseProblem` when evaluation is inherently serial.
6. **Import-linter layer contracts enforced.** Every phase updates `.importlinter` to pin the new dependency direction before retiring the old module.

---

## Phase Dependency Graph

```
              ┌───────────────┐
              │ 1. catalog    │  (foundation — everyone depends on it)
              └──┬──────┬─────┘
       ┌─────────┘      └──────────┐
       ▼                           ▼
┌───────────────┐           ┌───────────────┐
│ 2. geometry   │           │ 3. train      │
└──────┬────────┘           └─────┬─────────┘
       │                          │
       │    ┌───────────────┐     │
       └───▶│ 4. chromosome │     │
            └──────┬────────┘     │
                   ▼              │
            ┌───────────────┐     │
            │ 5. decoder    │◀────┘  (reads train for speed/radius)
            └──────┬────────┘
                   ▼
            ┌───────────────┐
            │ 6. problem    │
            └──┬──────┬─────┘
               │      │
               ▼      ▼
      ┌────────────┐ ┌───────────────┐
      │7. operators│ │8. visualization│
      └────────────┘ └───────────────┘
               │             │
               └──────┬──────┘
                      ▼
               ┌───────────────┐
               │ 9. config/io  │  (cross-cutting; depends on all)
               └───────────────┘
```

**Why this order:** catalog is the type layer all domain code reads; geometry and train only talk to catalog; chromosome defines the int16 layout consumed by decoder; decoder is the joint reader of all four chromosome regions plus catalog+geometry; problem wires decoder+train into pymoo; operators and visualization are consumers of problem state; config/io is cross-cutting infrastructure that pins the whole run reproducibility story.

---

## Phase Status & Entry Criteria

| # | Phase | Entry criterion | Status |
|---|-------|-----------------|--------|
| 1 | Catalog | current `track_pieces.yaml` + tests green | ☐ |
| 2 | Geometry | Phase 1 complete, catalog exposes `PortDef(dx,dy,dθ)` | ☐ |
| 3 | Train | Phase 1 complete | ☐ |
| 4 | Chromosome | Phase 1 complete | ☐ |
| 5 | Decoder | Phases 1, 2, 3, 4 complete | ☐ |
| 6 | Problem | Phases 3, 4, 5 complete | ☐ |
| 7 | Operators | Phase 6 complete | ☐ |
| 8 | Visualization | Phase 6 complete | ☐ |
| 9 | Config/IO | Phases 1–7 complete | ☐ |

---

# Phase 1 — Catalog

**V2 reference:** *"The catalog package: a first-principles design grounded in primary geometric sources."*

**Goal:** Replace the current `SECTION_TYPES` + `piece_index` YAML schema with a port-centric, manufacturer-tagged, schema-versioned catalog whose pieces expose `PortDef(dx, dy, dθ)` relative to port A and declare named traversal routes.

## Key contracts

```python
# src/catalog/pieces.py  (expanded)
class PortDef(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    dx: float          # studs
    dy: float          # studs (y = left)
    dtheta: float      # radians, CCW positive

class TrackPieceSpec(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    piece_id: str
    kind: Literal["straight", "curve", "switch", "wye", "crossing"]
    manufacturer: Literal["lego", "4dbrix", "fxbricks", "bricktracks", "trixbrix"]
    part_numbers: tuple[str, ...] = ()
    # kind-conditional:
    length_studs: float | None = None
    radius_studs: float | None = None
    sector_angle_rad: float | None = None
    hand: Literal["left", "right"] | None = None
    body_length_studs: float | None = None
    diverging_radius_studs: float | None = None
    # topology:
    ports: Mapping[str, PortDef]
    routes: Mapping[str, tuple[str, ...]]

    @property
    def on_angle_lattice(self) -> bool: ...  # π/32 check

class CatalogMeta(BaseModel):
    schema_version: str
    unit: Literal["stud"] = "stud"
    stud_mm: float = 8.0
    atomic_angle_rad: float = math.pi / 32

class TrackCatalog(BaseModel):
    meta: CatalogMeta
    pieces: tuple[TrackPieceSpec, ...]
    # Derived helpers (no chromosome_bounds — see §5 of V2 catalog report):
    by_id, by_kind, by_manufacturer, n_types, piece_ids
```

## File changes
- **Create:** `src/catalog/specs.py` (Pydantic models), `src/catalog/loader.py` (ruamel.yaml loader with line-number error messages)
- **Modify:** `src/catalog/pieces.py` (convert `TrackPiece` into a shim that adapts `TrackPieceSpec` back to legacy shape during migration)
- **Modify:** `src/catalog/catalog.py` (expose `by_id`, `piece_ids`, keep `_fk_table` / `_speed_table` / `_radius_table` for backward compat, populate them from `TrackPieceSpec.ports`)
- **Create:** `data/track_pieces_v2.yaml` (new schema, covering every piece currently in `data/track_pieces.yaml` plus lattice-off-angle annotations; `schema_version: "1.0.0"`)
- **Create:** `tests/catalog/test_specs.py`, `tests/catalog/test_loader.py`, `tests/catalog/test_parity_v1_v2.py`

## Migration strategy
1. Build the new loader against a minimal fixture YAML (`tests/catalog/fixtures/tiny.yaml`) and prove port-A-at-origin, route-refs-real-ports, unique-piece-id validators fire with file/line context.
2. Port each section of the existing `data/track_pieces.yaml` into `data/track_pieces_v2.yaml`, deriving `PortDef(dx, dy, dθ)` from existing FK deltas. For switches, add the `diverging` route with port C computed from the 3-4-5 Pythagorean arc.
3. Switch `TrackCatalog.load` to consume the v2 YAML, rebuilding the legacy `_fk_table`/`_speed_table` from it. The public methods (`get_fk`, `get_speed_limits`, etc.) keep the same signatures.
4. Parity test: every existing test that uses the catalog passes without modification.
5. Add a deprecation warning when `data/track_pieces.yaml` (v1) is loaded and document the migration path in the module docstring.

## Validation-error UX (required by V2)
Six error classes, each with filename + line number + field path + actionable next step — per the table in V2 catalog report §"Validation error UX." Enforced via Pydantic v2 `ValidationError.errors()` + ruamel.yaml `LineCol` attachment.

## Lift-out: what this phase does *not* do
- Does **not** expose `chromosome_bounds` (V2 §"Catalog does not own chromosome bounds"; that belongs to Phase 4).
- Does **not** compute SE(2) composition (Phase 2).
- Does **not** compute `v_safe(R)` (Phase 3).

## Commit boundary & rollback
- One commit per sub-step above. Final commit retires `data/track_pieces.yaml`.
- Rollback path: revert the loader switch and keep the v2 YAML dormant.

---

# Phase 2 — Geometry

**V2 reference:** *"Designing the geometry package for LEGO track optimization."*

**Goal:** Package `geometry.py` into a submodule tree with an SE(2) pose type, canonical angle-wrap, tolerances module, and collision kernel. Switch units from degrees to radians internally.

## Key contracts

```python
# src/geometry/tolerances.py
ANGULAR_CLOSURE_TOL = 1e-9     # rad
POSITIONAL_CLOSURE_TOL = 1e-6  # studs
COLLISION_SAFETY_MARGIN = 8.0  # studs (one sleeper width)
ATOMIC_ANGLE_RAD = math.pi / 32

# src/geometry/pose.py
@dataclass(frozen=True, slots=True)
class Pose2D:
    x: float; y: float; heading: float  # radians in (-pi, +pi]

def wrap_angle(theta: float) -> float  # (-pi, +pi]
def compose(world: Pose2D, delta: Pose2D) -> Pose2D
def inverse(p: Pose2D) -> Pose2D
def transform_port(world_at_A: Pose2D, port: PortDef) -> Pose2D
def forward_kinematics(placements, start=ORIGIN) -> list[Pose2D]
def closure_residual(first: Pose2D, last: Pose2D) -> tuple[float, float, float]
def aabb(placements) -> tuple[float, float, float, float]

# src/geometry/intersect.py   (numba-eligible, stateless math kernels)
def seg_seg_distance(a0, a1, b0, b1) -> float
def seg_arc_distance(s0, s1, center, R, theta_start, theta_sweep) -> float
def arc_arc_distance(c1, R1, t1s, t1sw, c2, R2, t2s, t2sw) -> float

# src/geometry/collision.py  (broad-phase + excluded-pair filter)
def layout_has_collision(placements, adjacency) -> bool
def check_collisions(placements, excluded_pairs) -> tuple[tuple[int,int], ...]
```

## File changes
- **Create:** `src/geometry/` package with `__init__.py`, `pose.py`, `intersect.py`, `collision.py`, `tolerances.py`
- **Modify:** `src/geometry.py` → delete after `__init__.py` re-exports `Layout`, `compute_fk_chain`, `build_layout` from the package
- **Modify:** `src/intersection.py` → move content into `src/geometry/intersect.py` (this file currently handles crossing-count, not geometric distance; rename semantic)
- **Create:** `tests/geometry/test_pose.py`, `test_intersect.py`, `test_collision.py`, `test_tolerances_roundtrip.py`

## Semantic change: degrees → radians
Current `compute_fk_chain` stores heading as degrees (`states[:, 2]`). V2 mandates radians. This is a project-wide change — every consumer (decoder, visualization, tests) must be updated. Strategy:
1. Introduce `Pose2D` and `compose` in parallel; keep `compute_fk_chain` untouched.
2. Build a legacy shim `compute_fk_chain_rad(fk_deltas_rad)` that produces `Pose2D` lists.
3. Migrate callers one at a time. Decoder (Phase 5) will consume `Pose2D` natively.
4. After all callers migrated, replace `compute_fk_chain` with a rad-returning version.

## Key derivation (to include in the phase plan)
- Higham γₙ round-off analysis at N ≤ 200 shows `ε_θ ≈ 1.4e-13`, `ε_p ≈ 2.2e-12` — four+ orders below 1e-9 / 1e-6 tolerances. Tolerances are intentionally loose to absorb sin/cos library variance.
- Angle wrap: branch-free form `theta = (theta + pi) % (2pi) - pi; return pi if theta == -pi else theta` (matches POSIX atan2).

## Lift-out
- **No numba required for v1.** Pure-Python is adequate at population ≤ 200. Numba JIT is a Phase-5 optimization once the decoder hot loop is identified as the bottleneck.

---

# Phase 3 — Train

**V2 reference:** *"Designing a train-physics package for LEGO curve-speed optimization."*

**Goal:** Consolidate physics into `PhysicsParams`, `v_safe_one`, `SpeedTable`, `v_bottleneck`, `batch_v_bottleneck`, with correct μ=0.35 nominal and v_cap=1.10 m/s.

## Key contracts

```python
# src/train/physics.py
STUD_M = 0.008; G_STD = 9.80665; MU_DEFAULT = 0.35; V_CAP_DEFAULT = 1.10

@dataclass(frozen=True)
class PhysicsParams:
    mu: float = MU_DEFAULT
    g: float = G_STD
    v_cap: float = V_CAP_DEFAULT
    stud_m: float = STUD_M

@dataclass(frozen=True)
class PhysicsResult:
    v_bottleneck: float
    binding_piece_index: int
    friction_bound: bool
    transition_radius_studs: float

def v_safe_one(R_studs: float, params: PhysicsParams) -> float:
    # min(sqrt(mu * g * R_m), v_cap)

class SpeedTable:
    def __init__(self, params, catalog_radii_studs): ...
    def v_safe(self, R_studs: float) -> float

def v_bottleneck(placements, speed_table: SpeedTable) -> PhysicsResult
def batch_v_bottleneck(R_matrix: np.ndarray, params: PhysicsParams) -> np.ndarray
```

## File changes
- **Modify:** `src/train/physics.py` — replace current content with the V2 API
- **Modify:** `src/train/scoring.py` → rename to `src/train/bottleneck.py`, owns `v_bottleneck` and `batch_v_bottleneck`
- **Delete:** legacy speed-profile code path (the forward-backward pass used by `compute_speed_profile` becomes dead once Problem switches to `v_bottleneck`)
- **Create:** `tests/train/test_physics.py` with the worked-example table from V2 §"Numerical checks" — every (R, μ) cell in the table is a parametrized test case

## Derived constants to verify in tests
- R\*(μ=0.35) ≈ 44 studs (transition radius)
- v_safe(R=40, μ=0.35) ≈ 1.049 m/s (friction-bound)
- v_safe(R=56, μ=0.35) = 1.10 m/s (motor-bound)
- v_safe(R=40, μ=0.25) ≈ 0.886 m/s

## Lift-out
- Length-weighted harmonic-mean speed is documented but not implemented (V2 §"Bottleneck over harmonic mean"). Leave as a future auxiliary indicator.
- Motor dynamics, flange climb, aerodynamic drag are out of scope (V2 §"What the single-point quasi-static model deliberately does not capture").

---

# Phase 4 — Chromosome

**V2 reference:** *"Four regions of integer genes, and why they earn their place."*

**Goal:** Replace the current 3-segment encoding (main_loop + junction_descriptors + start_position) with the V2 4-region encoding (main_loop + switch_mask + branch_slots + crossing_overlay) and lift chromosome-bounds ownership out of the catalog.

## Exact byte layout (required detail per project memory)

Configure via `ChromosomeConfig(n_piece_types, max_loop_length, max_switches, max_branches, max_branch_length, max_crossings)`. The `ChromosomeLayout` is derived:

| Region | Offset | Width | Per-gene domain | Sentinel | Notes |
|---|---|---|---|---|---|
| main_loop | 0 | `max_loop_length` | `[0, n_piece_types]` | `S = n_piece_types` means "end of loop" | piece indices; left-to-right decode |
| switch_mask | `max_loop_length` | `max_switches` | `[0, 2]` | n/a | 0=through, 1=diverging, 2=branch-here |
| branch_slots | ... | `max_branches · max_branch_length` | `[0, n_piece_types]` | `S` terminates slot | rectangular; slot *k* ← *k*-th mask-2 switch |
| crossing_overlay | ... | `3 · max_crossings` | `(i, j, piece_type)` triples with `i < j ≤ max_loop_length` | `(0,0,0)` = blank | validates in decoder |

**Total width example** at `n_piece_types=22, max_loop_length=80, max_switches=15, max_branches=10, max_branch_length=20, max_crossings=6`: 313 genes × 2 bytes = 626 bytes per individual.

**Bounds builder:**
```python
def derive_bounds(catalog, cfg, layout) -> ChromosomeBounds:
    xl = zeros(layout.total_length, int16)
    xu = zeros(layout.total_length, int16)
    S = int16(cfg.n_piece_types)
    xu[layout.main_loop]    = S
    xu[layout.switch_mask]  = 2
    xu[layout.branch_slots] = S
    for k in range(cfg.max_crossings):
        base = layout.crossing_overlay.start + 3*k
        xu[base + 0] = cfg.max_loop_length   # i
        xu[base + 1] = cfg.max_loop_length   # j
        xu[base + 2] = S                     # piece type
    return ChromosomeBounds(xl=xl, xu=xu, vtype=int)
```

**`validate_invariants`:** Checks shape, per-region domain, crossing `i < j`, switch-mask length vs main-loop switch count.

## File changes
- **Delete:** `src/encoding.py` (current 3-segment scheme retired)
- **Create:** `src/chromosome/` package: `config.py` (ChromosomeConfig, ChromosomeLayout), `bounds.py` (derive_bounds, ChromosomeBounds), `invariants.py` (validate_invariants), `views.py` (segment_view)
- **Modify:** Everything that imports from `src.encoding` (main.py, problem.py, operators.py, sampling.py, repair.py, tests) — this is a big sweep. Strategy: keep `src/encoding.py` as a shim that re-exports from `src/chromosome/` under old names until all callers are migrated.

## Migration strategy
This phase **breaks wire compatibility** with existing chromosomes. Saved Pareto archives, test fixtures that hand-construct chromosomes, and every config that implies a particular N_VAR all become invalid. The plan:
1. Build `src/chromosome/` with full tests against synthetic configs. No callers yet.
2. Add a converter `src/chromosome/migrate.py::v1_to_v4(old_chromosome) -> new_chromosome` for any persisted data — but only for valid closed layouts; mutants are fresh-initialized in v2.
3. Switch `Problem.__init__` to call `derive_bounds(catalog, cfg)` and accept an int16 chromosome of the new shape. Operators and sampling update in Phase 7 (some temporary coupling is unavoidable).
4. Run every integration test. Any failure is a semantic gap in the conversion.

## Parity / regression test
`tests/chromosome/test_legacy_parity.py` decodes a known closed oval in both encodings and asserts the resulting `Layout` has the same indices and closure error.

## Lift-out
- Operator hyperparameters (SBX η, PM η, region mutation rates) are set in Phase 7.
- Crossing overlay is mutated (not frozen) — operator lives in Phase 7.

---

# Phase 5 — Decoder

**V2 reference:** *"The decoder is where feasibility stops being a hope."*

**Goal:** Replace the current construction decoder with a seven-phase turtle-kinematics pipeline that produces `DecodedLayout(placements, closure_residual, collision_list, census, status)`, with five structural invariants (P1–P5) guaranteed by construction and two residuals (C1 closure, C2 collision) reported for the problem layer.

## The seven phases (decoder internal)

1. **Pre-decode validation** — `validate_invariants(x, layout, cfg)`. O(N) integer checks. Status: `INFEASIBLE_INVARIANTS` on failure.
2. **Lattice angular prefilter** — For pure-lattice chromosomes (no 4DBrix off-angle pieces), sum integer multiples of atomic angle mod 64. Rejects ~98.4% of random closures before any sin/cos. Status: `INFEASIBLE_ANGULAR_LATTICE` on failure.
3. **Main-loop turtle walk** — Numba-eligible hot path. Walks `main_loop` left-to-right, composing SE(2) poses via catalog `PortDef`. Tracks two counters: `switch_count` (indexes `switch_mask`) and `branch_here` (indexes `branch_slots`). Validates crossings declared by `crossing_overlay` aligned geometrically.
4. **Branch extraction** — For each mask-2 switch, read `branch_slots[k·L_br : (k+1)·L_br]` up to sentinel, walk from switch's port C. Status: `INFEASIBLE_BRANCH_STRUCTURE` if a branch contains a switch or crossing.
5. **Closure measurement** — `residual = (turtle.x, turtle.y, wrap_angle(turtle.theta))`. Reported, not enforced.
6. **Collision detection** — `check_collisions(all_pieces, excluded_pairs)`. Reported as list.
7. **Census & output** — Build `DecodedLayout` frozen dataclass.

## Key contracts

```python
class DecoderStatus(Enum):
    FEASIBLE = 0
    INFEASIBLE_INVARIANTS = 1
    INFEASIBLE_ANGULAR_LATTICE = 2
    INFEASIBLE_CROSSING_GEOMETRY = 3
    INFEASIBLE_BRANCH_STRUCTURE = 4
    INFEASIBLE_INVALID_PIECE = 5

@dataclass(frozen=True, slots=True)
class PlacedPiece:
    piece_id: int
    entry_port: str
    exit_port: str
    world_pose_at_entry: Pose2D
    layer: Literal["main", "branch", "crossing"]

@dataclass(frozen=True, slots=True)
class DecodedLayout:
    placements: tuple[PlacedPiece, ...]
    closure_residual: tuple[float, float, float]
    collision_list: tuple[tuple[int, int], ...]
    census: dict[int, int]
    status: DecoderStatus
    status_reason: str = ""

def decode(x, catalog, cfg, layout, params) -> DecodedLayout
def decode_batch(X, ...) -> list[DecodedLayout]
```

## The branch-slot counter rule (critical, previously broken)

Slot *k* supplies the diverging-route content for the *k*-th switch occurrence in main-loop order whose `switch_mask[switch_count]==2` (branch-here). **Not** the *k*-th switch of any kind. Phase 3 increments `switch_count` on every switch but only appends to `switch_log` (and increments `branch_here`) when the mask is 2. Phase 4 consumes `switch_log` in order, reading branch slot *k* for the *k*-th entry.

This fixes the documented "severed genotype-to-phenotype path" from user memory. Operators (Phase 7) write to branch slots using the same counting rule.

## File changes
- **Modify:** `src/decoder/` package (already exists per April 18 plan)
- **Create:** `src/decoder/phases.py` with the seven-phase pipeline
- **Modify:** `src/decoder/construction.py` — retire legacy construction, keep shim that routes through `decode()`
- **Create:** `src/decoder/status.py` (DecoderStatus enum)
- **Create:** `tests/decoder/test_phases.py`, `test_invariants_p1_p5.py`, `test_lattice_prefilter.py`, `test_edge_cases.py` (the seven edge cases from V2 §"Seven edge cases")

## The five invariants (P1–P5) as explicit tests

| # | Invariant | Test |
|---|---|---|
| P1 | Port connectivity | assert `pose_at_exit_i == pose_at_entry_{i+1}` within tolerance |
| P2 | Branch attachment | assert every branch's first piece starts at port C of its switch |
| P3 | Crossing declaration | assert every crossing in placements has exactly two (i,j) overlay entries aligning geometrically |
| P4 | Route consistency | assert mask-gene choice ∈ {0,1,2} matches `PlacedPiece.exit_port` |
| P5 | Catalog validity | assert every piece_id in placements is in catalog |

## Lift-out
- Numba JIT is optional. v1 is pure-Python at ~150 µs/decode; numba compilation drops to ~1 µs but adds a dependency and JIT warm-up cost. Defer until profiling shows the decoder dominates runtime.

---

# Phase 6 — Problem

**V2 reference:** *"Wrapping the pipeline as a pymoo problem."*

**Goal:** Rewrite `problem.py` as a single `TrackLayoutProblem(ElementwiseProblem)` that consumes `DecodedLayout` from Phase 5 and wires it into pymoo's 2-objective NSGA-II with the corrected constraint formulation.

## Key contracts

```python
class TrackLayoutProblem(ElementwiseProblem):
    def __init__(self, catalog, cfg, physics, tolerances, elementwise_runner=None):
        self.layout = derive_layout(cfg)
        self.bounds = derive_bounds(catalog, cfg, self.layout)
        self.speed_table = SpeedTable(physics, catalog.curve_radii)
        self.piece_types = tuple(catalog.piece_ids)
        self._warm_up()
        super().__init__(
            n_var=self.bounds.n_var,
            n_obj=2,
            n_ieq_constr=4 + len(self.piece_types),
            n_eq_constr=0,                    # closure is 3 inequalities, not equalities
            xl=self.bounds.xl, xu=self.bounds.xu, vtype=int,
            elementwise_runner=elementwise_runner or LoopedElementwiseEvaluation(),
        )
```

## Objective & constraint vector

**F** (both minimized):
- `F[0] = -utilization = -(layout.n_pieces / total_inventory)`
- `F[1] = -v_bottleneck(placements, speed_table).v_bottleneck`

**G** (g ≤ 0 feasible), length `4 + |piece_types|`:
| Index | Entry | Formula | Scale |
|---|---|---|---|
| 0 | Closure x | `|dx|/S_xy − 1` | `S_xy = 0.5 studs` |
| 1 | Closure y | `|dy|/S_xy − 1` | ditto |
| 2 | Closure θ | `|dθ|/S_θ − 1` | `S_θ = π/180 rad` |
| 3 | Collisions | `len(collision_list) / COLLISION_SCALE` | 5.0 |
| 4..3+T | Per-type inventory excess | `max(0, census[t] − max_occ[t]) / max(1, max_occ[t])` | per type |

**Infeasibility sentinel** (for non-FEASIBLE decoder status):
- `F = [+inf, +inf]`
- `G = [1e6] * n_ieq_constr`

## Why closure is 3 inequalities not 1 equality
pymoo docs: *"most algorithms in pymoo will not handle equality constraints efficiently."* Deb (2000) prescribes `|h| − δ ≤ 0` reformulation. Per-axis scales differ (studs vs radians), so separate tolerances matter. Single `|·|` inequality is standard CEC practice; symmetric pair double-counts.

## ConvergenceMonitorCallback

```python
class ConvergenceMonitorCallback(Callback):
    def __init__(self, ref_point=(0.10, -0.55), pareto_ref=None):
        self.hv = HV(ref_point=np.asarray(ref_point))
        self.igd = IGD(pareto_ref) if pareto_ref is not None else None

    def notify(self, algorithm):
        F, CV = algorithm.pop.get("F"), algorithm.pop.get("CV").ravel()
        feas = CV <= 0.0; F_feas = F[feas]
        self.data.setdefault("hv", []).append(float(self.hv(F_feas)) if len(F_feas) else 0.0)
        self.data.setdefault("igd", []).append(
            float(self.igd(F_feas)) if (self.igd and len(F_feas)) else np.nan)
        self.data.setdefault("n_feas", []).append(int(feas.sum()))
        self.data.setdefault("n_gen", []).append(algorithm.n_gen)
```

## File changes
- **Rewrite:** `src/problem.py` → `src/problem/` package with `problem.py`, `callback.py`
- **Delete:** legacy `MultiSegmentProblem`, `SingleObjectiveProblem` (V2 commits to the bi-objective formulation)
- **Create:** `tests/problem/test_problem_contract.py`, `test_constraint_vector.py`, `test_infeasibility_sentinel.py`, `test_monitor_callback.py`

## Lift-out
- Parallelization (`StarmapParallelization`, `JoblibParallelization`) deferred to Phase 9 — the UCB scheduler (Phase 7) assumes single-process state.
- Batched `batch_v_bottleneck` at the problem boundary saves ~4% of runtime; skip for v1.

---

# Phase 7 — Operators

**V2 reference:** *"Five operators, one bandit, and the ρ that protects closure."*

**Goal:** Replace legacy operators with `SegmentSelectiveCrossover` (ρ_main=1.0, ρ_other=0.7), `AdaptiveMutationSuite` (5 operators behind a UCB1+FRRMAB scheduler with c=√2, window=50, warmup=5), `InitialSampling`, and `MutantInjectionCallback` at 10%.

## The five mutation operators

| Op | Region | Neighbourhood | Invariant preservation | Default |
|---|---|---|---|---|
| SWAP_MAIN | main_loop | adjacent swap at (i, i+1) | commutative sum ⇒ closure exact | n_swaps=1 |
| REPLACE_MAIN | main_loop | same-atomic-angle replacement from catalog bucket | same-angle ⇒ closure exact | n=1 |
| FLIP_SWITCH_MASK | switch_mask | 0↔2 flip (same exit port, adds/removes branch) | closure preserved; 0↔1 disabled | p_flip=0.1/switch |
| BRANCH_MUTATE | branch_slots | insert/delete/replace at slot k (k = *k*-th mask-2 switch) | sentinel S preserved | p_ins=p_del=p_rep=⅓ |
| CROSSING_TOGGLE | crossing_overlay | add/remove/retype triple | `i<j` enforced on add | p_add=0.4, p_rm=0.4, p_rt=0.2 |

## UCB scheduler

```python
class UCBScheduler:
    def __init__(self, n_arms=5, c=sqrt(2), window=50, D=1.0, warmup=5):
        self.n = np.ones(n_arms)              # cold start
        self.r = np.full(n_arms, 0.5)         # reward prior
        self.window = deque(maxlen=window)
        self.pending = []                     # (idx, op, child) this generation

    def choose(self, gen):
        if gen < self.warmup:
            return (gen * 997 + self.n.argmin()) % len(self.n)  # round-robin
        total = self.n.sum()
        ucb = self.r + self.c * np.sqrt(np.log(total) / self.n)
        return int(np.argmax(ucb))

    def update(self, op_idx, fi):
        self.window.append((op_idx, fi))
        self._recompute_means()  # FRRMAB rank-based decay
```

**Reward signal (FRRMAB from Li et al. 2014):** For each child, `FI = max(0, f_parent_rank − f_child_rank)` via non-dominated sort of parent+child. Store `(op, FI)` in a 50-entry sliding window; within window, rank FI values and assign decayed credit `D^(rank-1) * FI`; per-op reward = sum of decayed FIs / count of ops in window, min-max normalized to [0,1] before UCB.

## SegmentSelectiveCrossover

Per-mating, pick elite (fitter by rank+crowding):
- `main_loop`: copy entire slice from elite (ρ_main=1.0 — full inheritance).
- `switch_mask`, `branch_slots`, `crossing_overlay`: per-gene biased coin with ρ=0.7 (BRKGA canonical).

## MutantInjectionCallback

At end of each generation's survival, replace bottom 10% by rank-then-crowding with fresh `InitialSampling` chromosomes. Runs *after* `RankAndCrowding(crowding_func="pcd")`. Callback also owns UCB reward accounting for the generation's `pending` list and logs `op_history` for visualization.

## NSGA-II configuration

```python
algorithm = NSGA2(
    pop_size=200,
    sampling=InitialSampling(seed_library=library, seed_frac=0.0),
    crossover=SegmentSelectiveCrossover(rho_main=1.0, rho_mask=0.7,
                                         rho_branch=0.7, rho_crossing=0.7),
    mutation=AdaptiveMutationSuite(
        ops=[SwapMain(), ReplaceMain(), FlipSwitchMask(),
             BranchMutate(), CrossingToggle()],
        scheduler=UCBScheduler(n_arms=5),
    ),
    survival=RankAndCrowding(crowding_func="pcd"),  # Kukkonen & Deb 2006
    eliminate_duplicates=True,
)
```

## File changes
- **Rewrite:** `src/operators.py` → `src/operators/` package with `crossover.py`, `mutation.py`, `scheduler.py`, `sampling.py`, `callbacks.py`
- **Delete:** legacy operators after final switch
- **Create:** `tests/operators/test_crossover.py` (ρ respects region boundaries), `test_mutation.py` (each operator preserves its invariants — parameterized over operator × seed chromosome), `test_ucb.py` (cold-start, warmup, reward windowing), `test_mutant_injection.py`

## Lift-out
- Whole-loop swap crossover (alternative to ρ_main=1.0) deferred.
- Path-relinking (Andrade et al. 2021) deferred.
- Operator-hyperparameter grid search (c ∈ {0.5, 1.0, √2, 2.0}) deferred.

---

# Phase 8 — Visualization

**V2 reference:** *"The visualization package, rendered honestly."*

**Goal:** Produce five canonical thesis figures (best-layout render, Pareto front with knee, HV/IGD convergence, speed heatmap with R\*, operator-usage stacked area). Post-hoc from persisted artifacts; never imports pymoo or the optimizer.

## The five figures

| # | Figure | Claim | Data source |
|---|---|---|---|
| 1 | Best layout render | "produces closed, feasible, non-trivial layouts" | DecodedLayout of knee individual |
| 2 | Pareto front scatter + knee | "utilization-speed trade-off is real" | F_history[-1], CV_history[-1] |
| 3 | Convergence: HV + IGD + feasibility | "NSGA-II converges in HV" | monitor.pkl |
| 4 | Speed heatmap with R\* inset | "R40 is the friction bottleneck at μ=0.35" | DecodedLayout + train.v_bottleneck |
| 5 (appendix) | Operator usage stacked area | "UCB re-weights operators over phases" | op_history.npz |

## Key constraints
- Rendering uses `matplotlib.patches.Rectangle` + `LineCollection` + compound `PathPatch`. **Do NOT use `matplotlib.patches.Arc`** — `Arc.draw()` cannot be filled.
- Okabe-Ito 8-color categorical palette (piece types); viridis continuous (speed).
- `Agg` backend for thread-safety (matplotlib is not thread-safe).
- HV reference point = `(+0.10, -0.55)` — 10% of range beyond empirical nadir `(0, -0.60)`.
- Output: SVG primary + PNG 300 DPI secondary.

## File changes
- **Modify:** `src/visualization/` package (already exists)
- **Rewrite:** `track_renderer.py` using LineCollection for curves, Rectangle for straights, PathPatch for switches
- **Create:** `src/visualization/speed_heatmap.py`, `src/visualization/operator_usage.py`
- **Create:** `tests/visualization/test_renderers.py` (asserts output SVG contains expected artists), `test_color_palette.py`

## Lift-out
- Empirical attainment surface (EAS) across 20–30 seeds deferred — needs multi-run sweep infrastructure.
- Interactive plotly/bokeh deferred.

---

# Phase 9 — Config / IO

**V2 reference:** *"Wiring reproducibility into the thesis: the config and io shell."*

**Goal:** Switch config format from YAML to TOML, Pydantic v2 validation with frozen/extra-forbid, reproducibility-grade run directory (timestamp + seed + git hash), pickle+gzip checkpoints, signal handling, JSONL event log.

## Contract

```toml
# configs/default.toml
[meta]
schema_version = "1.0.0"

[catalog]
path = "data/track_pieces_v2.yaml"

[geometry]
pos_tol = 1e-6
ang_tol = 1e-9

[train]
mu = 0.35; g = 9.81; v_cap = 1.10; wheel_diameter_m = 0.017

[chromosome]
max_loop_length = 60; max_branches = 4
max_branch_length = 10; max_crossings = 2

[problem.constraints]
S_xy = 0.5; S_theta = 0.017453; collision_scale = 5.0

[problem.hv]
ref_point = [0.10, -0.55]

[operators.crossover]
rho_main = 1.0; rho_mask = 0.7; rho_branch = 0.7; rho_cross = 0.7

[operators.mutation]
c_explore = 1.4142; window = 50; decay = 1.0; warmup = 5; mutant_frac = 0.1
enabled = ["swap_main", "replace_main", "flip_switch_mask",
           "branch_mutate", "crossing_toggle"]

[algorithm]
pop_size = 200; n_gen = 1000; seed = 1

[visualization]
dpi = 300; font_family = "serif"; figure_width_mm = 88; colormap = "viridis"

[io]
run_dir = "runs/"; checkpoint_every = 50; log_level = "INFO"
```

## Run directory layout
```
runs/20260420_143012_seed01_a1b2c3/
├── config_effective.toml       # post-validation, defaults filled
├── environment.txt             # pip freeze + Python + git hash
├── callbacks/ {monitor, ucb_state, F_history, CV_history, op_history}
├── checkpoints/gen_NNNN.pkl.gz # pickle HIGHEST_PROTOCOL
├── results/ {final_pop.pkl, pareto_front.json, knee_point.json}
├── figures/ *.svg *.png
├── figure_manifest.json        # SHA-256 per artifact
└── logs/ run.log events.jsonl  # JSON Lines per generation
```

## Schema evolution policy (semver 2.0.0)
- **Same MAJOR.MINOR, any PATCH** → accept silently
- **Config MINOR < code MINOR** → warn, fill defaults
- **Config MINOR > code MINOR** → warn, `extra='ignore'`
- **Different MAJOR** → reject with migration pointer

## Precedence order
`Field(default)` < `configs/default.toml` < experiment config < `LEGOTRACK_*` env vars < `--set train.mu=0.45`

## File changes
- **Create:** `src/io/` package with `config.py` (Pydantic root model), `run_dir.py`, `checkpoint.py`, `logging_setup.py`, `signal_handling.py`
- **Migrate:** `configs/*.yaml` → `configs/*.toml` via a one-shot converter script (`scripts/yaml_to_toml.py`)
- **Modify:** `src/config.py` → retire; re-exports moved to `src/io/config.py`
- **Create:** `tests/io/test_config_load.py` (every error UX from V2 §"Config errors"), `test_checkpoint_roundtrip.py`, `test_signal_handling.py`, `test_schema_evolution.py`

## Canary modes
- `python -m legotrack.main --config configs/default.toml --validate-only` — validates config, writes `config_effective.toml`, exits. <200ms.
- `python -m legotrack.main --config configs/default.toml --dry-run` — runs 2 generations with pop_size=10, produces complete run_dir. <10s.

## Lift-out
- Multi-run sweep infrastructure (for EAS plots) deferred.
- Cross-machine cluster runs deferred.

---

# Cross-Cutting Concerns

## Import-linter contracts

After each phase, update `.importlinter` to enforce the new dependency direction:

```ini
[importlinter]
root_package = src

[importlinter:contract:layers]
name = Domain core below EA below infrastructure
type = layers
layers =
    src.io                      # Phase 9 (top)
    src.visualization           # Phase 8
    src.operators               # Phase 7
    src.problem                 # Phase 6
    src.decoder                 # Phase 5
    src.chromosome              # Phase 4
    src.geometry                # Phase 2
    src.train                   # Phase 3
    src.catalog                 # Phase 1 (bottom)
```

Verify after each phase: `lint-imports`.

## Regression guardrails

Every phase's final commit runs the full test suite, plus:

1. **Optimizer smoke test.** `python main.py --config configs/default.toml --quick` (20 generations, pop=50). Asserts `closure_error < 4 studs` and `feasible_count > 0`.
2. **Diagnostic check.** `/diag` on the run output asserts best-layout invariants. Per CLAUDE.md, no exit-code-only verification.
3. **Config parity.** Every `configs/*.yaml` that existed at phase start either (a) still loads, or (b) has been converted with a deprecation shim.

## Project invariants (from CLAUDE.md + memory) that must not regress

1. Chromosome length scales dynamically with inventory — never hardcoded.
2. No hardcoded dimensional or constraint limits — all from inventory + config.
3. Repair wired into evaluation pipeline — not called ad-hoc.
4. Fitness rewards branches — otherwise GA eliminates switches as overhead.
5. `/diag` reports closure error, orphan switches, feasible-solution count.

## Project-wide file count

Approximate after all phases complete (deletions balanced by packaging):

| Package | Files |
|---|---|
| catalog | 4 (specs, loader, pieces, catalog) |
| geometry | 4 (pose, intersect, collision, tolerances) |
| train | 3 (physics, bottleneck, scoring?) |
| chromosome | 4 (config, bounds, invariants, views) |
| decoder | 4 (phases, construction, status, types) |
| problem | 2 (problem, callback) |
| operators | 5 (crossover, mutation, scheduler, sampling, callbacks) |
| visualization | 5 (existing + speed_heatmap + operator_usage) |
| io | 5 (config, run_dir, checkpoint, logging_setup, signal_handling) |
| **total** | **~36 src files** (up from ~18 today) |

---

# Estimated Effort (Rough)

| Phase | Est. tasks | Est. commits | Risk |
|---|---|---|---|
| 1. catalog | ~15 | ~10 | Medium — touches YAML schema |
| 2. geometry | ~20 | ~12 | High — rad conversion is project-wide |
| 3. train | ~10 | ~6 | Low — clean new API |
| 4. chromosome | ~25 | ~15 | **Highest** — breaks wire compat |
| 5. decoder | ~25 | ~15 | High — 7-phase state machine |
| 6. problem | ~15 | ~10 | Medium — constraint reformulation |
| 7. operators | ~30 | ~18 | **Highest** — UCB scheduler + 5 ops + callback |
| 8. visualization | ~15 | ~10 | Low — pure consumer |
| 9. config/io | ~25 | ~15 | Medium — TOML migration + run_dir |

**Total estimate:** ~180 tasks across ~110 commits. A realistic cadence of 3–5 tasks per day puts total effort at 6–12 weeks of focused work.

---

# Execution Strategy

This roadmap is not itself executable. To begin:

1. **Pick the next phase.** Phase 1 (catalog) is the natural entry point — everyone depends on it.
2. **Invoke `superpowers:writing-plans`** with the phase's "Key contracts" and "File changes" sections as input spec.
3. **Execute the resulting bite-sized plan** via `superpowers:subagent-driven-development` or `superpowers:executing-plans`.
4. **Mark the phase check-box** in this document and commit the update before starting the next phase.

Phase boundaries are checkpoint opportunities. Abandoning mid-phase leaves a broken surface; abandoning between phases leaves the codebase in a consistent state.

---

# What This Roadmap Does Not Decide

- **Whether to start at all.** V2 is a ~6–12 week commitment on top of a working optimizer. The user may reasonably prefer incremental polish.
- **Numba adoption.** V2 recommends numba for decoder and collision hot paths. v1 can skip it; adopt only after profiling shows a real bottleneck.
- **Parallelization.** `StarmapParallelization` and `JoblibParallelization` are wired but off by default. Decide after Phase 7 based on measured single-process cost.
- **Catalog content.** V2's validation table lists ~200 real pieces across 5 manufacturers. The current catalog has ~10. Expansion scope is user-driven and independent of architecture.
- **Thesis-specific figures.** V2's §"Five figures" is the thesis narrative; other use-cases may need different figures (e.g., instruction generation).