# Batch 1 Implementation Research: Catalog + Geometry + Train

**Companion to:** `2026-04-20-modular-v2-adoption-roadmap.md`
**Status:** Research complete; ready to expand Phase 1 into a bite-sized executable plan
**Scope:** Concrete implementation patterns, library-level decisions, and corrections to V2 spec where the spec was aspirational rather than verified.

---

## Batching Decision

| Batch | Phases | Why grouped |
|---|---|---|
| **Batch 1 — Foundation** | Catalog (1), Geometry (2), Train (3) | All three are domain primitives with no dependencies on other V2 packages. Can be delivered in any order. |
| **Batch 2 — Core** | Chromosome (4), Decoder (5), Problem (6) | The encoding → evaluation pipeline. Chromosome is the gate (breaking change); decoder is the joint reader; problem wires into pymoo. |
| **Batch 3 — Polish** | Operators (7), Visualization (8), Config/IO (9) | Consumers + infrastructure. Operators lifts NSGA-II to adaptive; visualization renders thesis figures; config/io pins reproducibility. |

Implementation details below are for **Batch 1**.

---

# Catalog — Implementation Findings

## Corrections to V2 spec

| V2 says | Research finding | Action |
|---|---|---|
| ruamel.yaml `YAML(typ='safe')` | `typ='safe'` **strips LineCol** — line numbers unavailable | Use default `YAML()` (round-trip mode); strip to plain dicts before Pydantic |
| `on_angle_lattice` tolerance = 1e-9 | YAML stores dtheta to ~8 decimal digits → truncation residual ~8e-9; tolerance is too tight and false-negatives legitimate lattice pieces | Use 1e-6 OR store angle as integer `k` with `dtheta = k · π/32` derived in Python |
| Schema version error via validator | A version mismatch is a *tooling* error, not a *data* error | Raise `SchemaVersionError(RuntimeError)` in loader *before* `model_validate`, not inside a validator |

## Concrete patterns to use

### Loader with file+line error UX
```python
# src/catalog/loader.py
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq
from pydantic import ValidationError

class CatalogLoadError(ValueError): pass
class SchemaVersionError(RuntimeError): pass

def load_catalog(path):
    yaml = YAML()                                       # round-trip — preserves .lc
    raw = yaml.load(open(path, encoding="utf-8"))
    check_schema_version(raw["meta"]["schema_version"], str(path))

    piece_lines = {i: node.lc.line + 1                  # 0-based → 1-based
                   for i, node in enumerate(raw["pieces"])}
    try:
        return TrackCatalog.model_validate(_strip_comments(raw))
    except ValidationError as exc:
        _raise_with_location(exc, path, piece_lines)

def _strip_comments(obj):
    if isinstance(obj, CommentedMap):  return {k: _strip_comments(v) for k, v in obj.items()}
    if isinstance(obj, CommentedSeq):  return [_strip_comments(v) for v in obj]
    return obj
```

### Backward-compat shim (keeps current callers unchanged)
```python
# src/catalog/catalog.py  (wraps V2 TrackCatalog)
class TrackCatalog:
    def __init__(self, v2):
        self._v2 = v2
        self._id_to_index = {p.piece_id: i for i, p in enumerate(v2.pieces)}

    @cached_property
    def _fk_table(self):                                # (n, 3) np.float64
        table = np.zeros((len(self._v2.pieces), 3))
        for i, spec in enumerate(self._v2.pieces):
            main = spec.routes.get("main") or spec.routes.get("through")
            exit_name = main[1] if main else "B"
            port = spec.ports[exit_name]
            table[i] = [port.dx, port.dy, port.dtheta]
        return table

    # Existing get_fk, get_radii, get_speed_limits delegate to cached tables.
    # No call site in problem.py / sampling.py / repair.py changes.
```

### Port C verification test (3-4-5 triple)
Intermediate checkpoints as exact asserts:

| Step | Value | Role |
|---|---|---|
| `end_vec1` | `(24.0, -32.0)` | 8·(3,4,5) chord inside arc 1 |
| `p1` | `(24.0, 8.0)` | End of arc 1 (heading = 36.87°) |
| `C2` | `(48.0, -24.0)` | Center of arc 2 |
| `p_C` | `(32.71, 12.96)` | Port C final position |
| `dtheta_C` | `π/8` | Net heading change |

Use `abs(dx - 32.71) < 0.01` in tests (exact derivation uses non-rational `atan2(3,4)`).

### Kind-conditional required fields
Use `@model_validator(mode='after')` with a dict of `kind → required field list`. `mode='after'` guarantees `self.kind` is already type-checked as `Literal`. Don't attempt per-field `@field_validator` — it doesn't see other fields.

### ruamel vs PyYAML — the real reason
Not round-tripping (irrelevant since we don't re-dump). The real hazard: **PyYAML 1.1 parses bare `on`, `off`, `yes`, `no`, `true`, `false` as booleans**. Route names, part labels, manufacturer tags could silently become `True`. ruamel defaults to YAML 1.2 (`true`/`false` only). Switch to ruamel regardless of line-number requirement.

## Phase 1 task ordering

1. Add `packaging`, `pydantic>=2.0,<3.0`, `ruamel.yaml` to deps (pydantic already pinned per config.py).
2. Create `src/catalog/specs.py` — `PortDef`, `TrackPieceSpec`, `CatalogMeta`, `TrackCatalog` (V2 model).
3. Create `src/catalog/loader.py` — `load_catalog`, `check_schema_version`, error UX.
4. Write `tests/catalog/test_specs.py` covering the six validation error messages + port-C derivation.
5. Create `data/track_pieces_v2.yaml` — translate each section of `data/track_pieces.yaml` into port-centric form. Switches: compute port C via the 3-4-5 derivation.
6. Modify `src/catalog/catalog.py` to wrap V2 spec with `@cached_property` numpy tables.
7. Point `TrackCatalog.load()` at v2 YAML; add deprecation warning for v1.
8. Run full test suite — existing tests must pass unchanged.

---

# Geometry — Implementation Findings

## Corrections to V2 spec

| V2 says | Research finding | Action |
|---|---|---|
| "numba @njit over cached catalog deltas" | Numba cannot consume `@dataclass(frozen=True, slots=True)` — raises `TypingError` at compile | Keep `Pose2D` Python-only; njit hot path takes 3 scalar floats + `(P, 4)` catalog array |
| Angle wrap `(theta + pi) % (2pi) - pi` | Naive form returns `[-π, +π)`, not `(-π, +π]` — fails for right-turn chains closing at `-π` | Keep V2's `== -math.pi` special case; comparison is **bitwise deterministic** in IEEE-754 |
| Degrees → radians migration "trivial" | **Breaks 6 files** across the codebase (exact line numbers below) | Parallel-function migration; never a feature flag |

## Concrete patterns to use

### Numba hot path (no frozen dataclass)
```python
# src/geometry/pose.py
@njit(cache=True)
def fk_chain_jit(gene_array, catalog):           # catalog: (P, 3) dx/dy/dtheta
    n = len(gene_array)
    xs, ys, hs = np.empty(n+1), np.empty(n+1), np.empty(n+1)
    xs[0] = ys[0] = hs[0] = 0.0
    x, y, h = 0.0, 0.0, 0.0
    for i in range(n):
        row = catalog[gene_array[i]]
        dx, dy, dh = row[0], row[1], row[2]
        ch, sh = np.cos(h), np.sin(h)
        x += dx * ch - dy * sh
        y += dx * sh + dy * ch
        h += dh
        xs[i+1], ys[i+1], hs[i+1] = x, y, h
    return xs, ys, hs
```
Python-layer `forward_kinematics` wraps the jit output into `list[Pose2D]` when needed.

### Angle wrap — verified
```python
def wrap_angle(theta: float) -> float:
    theta = (theta + math.pi) % (2 * math.pi) - math.pi
    return math.pi if theta == -math.pi else theta
```
Test cases that must pass:
- `wrap_angle(-math.pi) == math.pi` (right-turn chain closure)
- `wrap_angle(math.pi + 1e-17) == math.pi` (collapses to 1 ULP, no special case needed)
- `wrap_angle(math.pi) == math.pi` (idempotent)

### Rad migration — 6 files identified

| File | Break | Lines |
|---|---|---|
| `src/geometry.py` | `angle_error` uses `% 360` | 70–74 |
| `src/decoder/construction.py` | `np.radians(states[i, 2])` — double conversion after migration | 329, 605 |
| `src/intersection.py` | `abs(theta[i] - theta[j]) % 180` — degree-based | 79 |
| `src/templates.py` | `np.radians(theta)` in template FK | 182, 303–304 |
| `src/visualization/track_renderer.py` | Every `_draw_*` accepts theta in degrees | 15 sites |
| `tests/test_geometry.py`, `test_decoder.py`, `test_templates.py` | Assertions on degree values | multiple |

Migration order: decoder → intersection → templates → visualization → tests → kill legacy `compute_fk_chain`. **Visualization converts at entry of each `_draw_*` function** — don't propagate radian storage into matplotlib helpers; just `np.degrees(theta)` once per call site.

### Arc AABB — cardinal direction expansion
```python
def arc_aabb(cx, cy, R, theta_start, theta_end):
    pts_x = [cx + R*math.cos(theta_start), cx + R*math.cos(theta_end)]
    pts_y = [cy + R*math.sin(theta_start), cy + R*math.sin(theta_end)]
    sweep = (theta_end - theta_start) % (2 * math.pi)
    for phi, ex, ey in [(0, cx+R, cy), (math.pi/2, cx, cy+R),
                        (math.pi, cx-R, cy), (3*math.pi/2, cx, cy-R)]:
        if (phi - theta_start) % (2*math.pi) <= sweep:
            pts_x.append(ex); pts_y.append(ey)
    return (min(pts_x), min(pts_y), max(pts_x), max(pts_y))
```
Modulo arithmetic handles the ±π wrap automatically.

### Shapely 2.x cross-check
```python
tree = STRtree(shapely_polylines)
for i, seg in enumerate(shapely_polylines):
    candidates = tree.query(seg, predicate="intersects")   # 2.x API
    for j in candidates:
        if abs(i-j) <= 1 or (min(i,j), max(i,j)) in adjacency: continue
        if seg.intersects(shapely_polylines[j]): ...
```

### Higham bound property test
```python
U = 2**-53
@pytest.mark.parametrize("n", [16, 50, 100, 200])
def test_round_off_below_tolerance(n):
    fk_deltas = np.array([[15.3073, 3.0449, math.pi/8]] * n)
    states = compute_fk_chain_rad(fk_deltas)
    gamma = n * U / (1 - n * U)
    higham_bound = gamma * n * math.pi / 8
    assert abs(states[-1, 2] - n * math.pi / 8) < 10 * higham_bound
    assert abs(states[-1, 2] - n * math.pi / 8) < ANGULAR_CLOSURE_TOL
```

## Phase 2 task ordering

1. Create `src/geometry/tolerances.py` — `ANGULAR_CLOSURE_TOL = 1e-9`, `POSITIONAL_CLOSURE_TOL = 1e-6`, `ATOMIC_ANGLE_RAD = math.pi / 32`.
2. Create `src/geometry/pose.py` — `Pose2D`, `wrap_angle`, `compose`, `inverse`, `transform_port`, `forward_kinematics`, `closure_residual`, `aabb`.
3. Create `src/geometry/intersect.py` — `seg_seg_distance`, `seg_arc_distance`, `arc_arc_distance` (pure math, numba-eligible).
4. Create `src/geometry/collision.py` — `layout_has_collision` with AABB broad phase.
5. Port degrees-based `compute_fk_chain` consumers one file at a time (order above), each commit passes full tests.
6. Delete `src/geometry.py` and `src/intersection.py` after last migration.
7. **Defer numba** until profiling shows decoder dominates — pure-Python `forward_kinematics` at N≤200 is ~2s total per run.

---

# Train — Implementation Findings

## Current code surface (verified via grep)

**`src/train/physics.py`** (172 lines):
- `TrainConfig` has 13 fields covering slide/tip/Nadal derailment modes.
- Scalar: `v_slide`, `v_tip`, `v_nadal`, `v_max`, `v_eff`.
- Vectorized: `v_eff_array(config, radii_m)` — handles `np.inf` via `np.minimum`.
- `available_accel(config, v, radius_m, is_braking)` — friction ellipse + coupler.

**`src/train/scoring.py`** (168 lines):
- `SpeedProfile(speeds, avg_speed, lap_time, total_distance, max_speed, min_speed)`.
- `compute_speed_profile(layout, catalog, train_config)` — 3-pass double-unroll algorithm.

**Call sites:**
- `src/problem.py:109` — `compute_speed_profile(...)` → reads **only `.avg_speed`**
- `tests/test_scoring.py` — reads `.speeds`, `.max_speed` in test assertions

## Corrections / decisions

| V2 says | Research finding | Action |
|---|---|---|
| `μ = 0.30` (current) | V2's corrected nominal is **0.35** (median of injection-molded ABS literature) | Default `PhysicsParams(mu=0.35)` |
| `@lru_cache` on instance method | Memory-leak footgun (cache holds `self`) — BUT single-`SpeedTable` per run lifecycle makes it moot | Accept as-is for v1; upgrade to closure pattern if sensitivity sweep creates many instances |
| `np.inf` sentinel in batch min | `sqrt(np.inf) = inf` silently (no warnings); **NaN is the hazard** | Defensive assert at decoder boundary: no `np.nan` in `R_matrix` |
| `wheel_diameter_m` configurable | It's derived/annotation, not an input — `v_cap` is the source of truth | Expose `v_cap`; document wheel_diameter in config's informational section only |

## Concrete patterns to use

### Temporary adapter (Phase 3 ships before Phase 5)
```python
# src/train/bottleneck.py
@dataclass(frozen=True)
class _PlacedPieceAdapter:
    curve_radius_studs: Optional[float]    # None = straight/through-switch

def layout_to_placements(layout, catalog):
    """TEMP: bridge until Phase 5 decoder emits PlacedPiece directly."""
    radii_mm = catalog.get_radii(layout.indices)
    return [_PlacedPieceAdapter(r / catalog.stud_mm if r > 0 else None)
            for r in radii_mm]
```
Delete this helper the day Phase 5 lands.

### Parametrized truth table (V2 spec's §"Numerical checks")
```python
@pytest.mark.parametrize("mu", [0.25, 0.35, 0.45])
@pytest.mark.parametrize("R_studs", [40, 56, 72, 88, 104, 120, 148])
def test_v_safe_table(mu, R_studs):
    params = PhysicsParams(mu=mu)
    expected = min(math.sqrt(mu * 9.80665 * R_studs * 0.008), 1.10)
    assert v_safe_one(R_studs, params) == pytest.approx(expected, rel=1e-6)

def test_v_safe_zero_or_inf_guards():
    params = PhysicsParams()
    assert v_safe_one(0.0, params) == 0.0
    assert v_safe_one(-5.0, params) == 0.0
    assert v_safe_one(np.inf, params) == 0.0           # V2 guard clause
```

### Clean retirement of `compute_speed_profile`
**Information loss analysis:** production code reads `.avg_speed` only. Per-segment `.speeds` is read in one test that validates the double-unroll closure property — that test dies with the implementation. No production regression.

### `wheel_diameter_m` handling
```python
@dataclass(frozen=True)
class PhysicsParams:
    mu: float = 0.35
    g: float = 9.80665
    v_cap: float = 1.10                 # Directly configurable scalar.
                                        # Derived from 1242 RPM × π × 0.017 m / 60 s = 1.105 m/s
                                        # (PU 88011 motor, wheel 55423c01 = 17 mm tread).
    stud_m: float = 0.008
```
`wheel_diameter_m` lives only in the TOML `[train]` section as an informational annotation (Phase 9). No recomputation path.

## Phase 3 task ordering

1. Build `PhysicsParams`, `PhysicsResult`, `v_safe_one` as pure additions to `src/train/physics.py`.
2. Build `SpeedTable`, `v_bottleneck`, `batch_v_bottleneck` in new `src/train/bottleneck.py`.
3. Build the 3×7 parametrized truth table — all green before touching problem.py.
4. Build `layout_to_placements` temp adapter.
5. Switch `problem.py:109` from `compute_speed_profile(...).avg_speed` to `v_bottleneck(layout_to_placements(...), speed_table).v_bottleneck`.
6. Run full test suite. `test_scoring.py`'s double-unroll test dies — delete it, add replacement bottleneck tests.
7. Delete `src/train/scoring.py`, legacy `v_slide`/`v_tip`/`v_nadal`/`v_max`/`v_eff` if no callers remain.
8. Update `src/train/__init__.py` exports.

---

# Cross-Batch Observations

## Shared refinements to the roadmap

1. **Phase ordering holds.** The original dependency order (catalog → geometry, train → chromosome → decoder → problem → operators, visualization → config/io) is correct. None of the research findings surface a cycle.
2. **Geometry's rad migration is the sneakiest.** Six files break, not two. The phase plan must enumerate each migration as its own bite-sized step.
3. **Train migration is cleanest of the three.** `compute_speed_profile` retires with zero production regression because only `.avg_speed` is consumed.
4. **Catalog migration can be done with zero-downtime** — the V2 spec is wrapped by a shim that exposes the legacy numpy tables via `@cached_property`. No call site in problem.py / sampling.py / repair.py changes until later phases.

## Shared library pin updates

Add to project deps:
- `pydantic>=2.0,<3.0` (already pinned per CLAUDE.md)
- `ruamel.yaml>=0.18` (Phase 1 — replace `pyyaml.safe_load` calls)
- `packaging` (stdlib-adjacent, Phase 1 for `Version`)
- `shapely>=2.0` (Phase 2 **test-only** dependency; not production)
- `numba>=0.58` (Phase 2 **optional**; defer adoption)

## Import-linter contracts after Batch 1

```ini
[importlinter:contract:batch-1-layers]
name = Batch 1 foundation layers
type = layers
layers =
    src.geometry
    src.train
    src.catalog
```
(Each must not import from the others at this level — catalog is the bottom; geometry and train sit above it independently. No cross-imports between geometry and train.)

---

## Next Steps

Batch 1 is ready to expand into an executable plan. Recommended path:

1. **Expand Phase 1 (catalog)** into a bite-sized plan via `superpowers:writing-plans`. Input: this doc's "Phase 1 task ordering" section + V2 catalog spec. Output: `docs/superpowers/plans/2026-04-21-phase-1-catalog.md`.
2. **Execute Phase 1** via `superpowers:subagent-driven-development`.
3. **Only after Phase 1 green:** expand Phase 2 (geometry) or Phase 3 (train) — they're independent and can be done in either order, but Phase 2 unlocks more downstream work (Chromosome and Decoder both depend on geometry).

Batch 2 research will be dispatched after Batch 1 is implemented, because the chromosome/decoder/problem decisions may be refined by what we learn implementing the foundation.