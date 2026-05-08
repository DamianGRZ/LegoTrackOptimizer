# Batch 2 Implementation Research: Chromosome + Decoder + Problem

**Companion to:** `2026-04-20-modular-v2-adoption-roadmap.md`
**Relates to:** `2026-04-20-batch-1-implementation-research.md` (Batch 1)
**Status:** Research complete. **Recommended strategic change to phase ordering below.**
**Scope:** Concrete implementation patterns, library decisions, current-code ground truth, and migration strategy for V2 Phases 4 (Chromosome), 5 (Decoder), 6 (Problem).

---

# Strategic Finding: Flip the migration order

The original roadmap sequences Phase 4 → Phase 5 → Phase 6. Research reveals a better order:

> **Phase 6 (Problem) can ship as a shim FIRST**, before Phases 3, 4, 5. It delivers bug fixes immediately and upgrades incrementally as upstream phases land.

Why: the current `TrackOptimizationProblem` already uses `ElementwiseProblem`. The V2 target is mostly a constraint-formulation refactor (`+inf` sentinel, expanded G vector, `ConvergenceMonitorCallback`) — none of which require the chromosome, decoder, or train changes to precede it. Each upstream phase slots into a localized call-site inside `_evaluate`/`__init__`.

**Recommended order for Batch 2:**

| Order | Phase | Ships as |
|---|---|---|
| **2.1** | Problem shim | Standalone — fixes sentinel bug, expands G, adds HV/IGD monitoring |
| **2.2** | Decoder (encoding-agnostic parts) | 80% — `DecodedLayout` types + `validate_invariants` + lattice prefilter + closure measurement |
| **2.3** | Decoder (full Phase 3–4 walk) | After Phase 4 lands — the last 20% |
| **2.4** | Chromosome | The one big breaking change |

The original Phase 4 → 5 → 6 sequence forces all three to wait on each other. The shim approach delivers progressive value and reduces the risk concentration on Phase 4.

---

# Chromosome — Phase 4 Findings

## Corrections to V2 spec

| V2 says | Research finding | Action |
|---|---|---|
| Sentinel `S = n_piece_types` (positive) | Current uses `INACTIVE = -1` (negative); current `xl[:n_main] = -1` is a **correctness hazard** — SBX can produce `-0.3` → rounds to `0` (silently activates INACTIVE slot) | Adopt V2's positive sentinel; safer with RoundingRepair |
| `max_loop_length` (magic number 60) | Must derive from inventory dynamically per project invariant `feedback_no_hardcoded_constraints` | Use `ChromosomeConfig.from_optimization_config(cfg, catalog)` computing every bound from inventory |
| Migration "lossy" but not enumerated | Specifically: **start_position dropped**, junction position decoupling, templates need expansion into gene sequences | Document the lossy points; only migrate seeded heuristics, not random mutants |

## Current state (ground truth from `src/encoding.py`)

`compute_dimensions()` derives all sizes from inventory:
- `n_main` = count of non-switch non-inactive pieces in inventory
- `max_junctions` = `min(type_A_count, type_B_count)` — physical switch PAIRS, not logical count
- `total_straights` = STRAIGHT_16 + STRAIGHT_24 counts
- `n_var = n_main + max_junctions * 4 + 2`

Three segments:

| Segment | Offset | Width | Range |
|---|---|---|---|
| main_loop | 0 | `n_main` | `[-1, 9]` (INACTIVE sentinel negative) |
| junction_descriptors | `n_main` | `max_junctions * 4` | per-gene: active∈{0,1}, pos, handedness, n_straights |
| start_position | end | 2 | boundary min/max x, y |

## V2 exact `xl`/`xu` (concrete)

For current 10-piece catalog (n_piece_types=10, S=10), example config: max_loop_length=80, max_switches=15, max_branches=10, max_branch_length=20, max_crossings=6:

```python
S = np.int16(10)
xl = np.zeros(313, dtype=np.int16)        # all 0 — no negative sentinels
xu = np.zeros(313, dtype=np.int16)
xu[0:80]   = S           # main_loop         → [0, 10]
xu[80:95]  = 2           # switch_mask       → [0, 2]
xu[95:295] = S           # branch_slots      → [0, 10]
for k in range(6):
    base = 295 + 3*k
    xu[base+0] = 80      # i ≤ max_loop_length
    xu[base+1] = 80      # j ≤ max_loop_length
    xu[base+2] = S       # crossing piece type
```

Total 313 genes × 2 bytes = 626 B per individual (v1 was similar size).

## Template reconciliation — Option A (recommended)

Templates (LEFT_SIDING / RIGHT_SIDING) stay as sampling initializer; branch_slots stores template expansions; BRANCH_MUTATE can deviate.

Rationale: templates provide an **inductive bias**, not a hard constraint. They guarantee the sampled pattern is inventory-consistent and physically connectable. Branches are dead-ends (switch port 2 is open by construction), so no "closure guarantee" is being sacrificed. Mutations that deviate get penalized by collision/boundary/inventory constraints in the problem layer — the correct level.

Translation rule:
- LEFT_SIDING = `[R40_RIGHT, STRAIGHT_16×n, R40_LEFT]` → genes `[3, 0, 0, ..., 2, S, S, ...]`
- RIGHT_SIDING = `[R40_LEFT, STRAIGHT_16×n, R40_RIGHT]` → genes `[2, 0, 0, ..., 3, S, S, ...]`

## Concrete migration converter

```python
def migrate_chromosome_v1_to_v4(
    old: np.ndarray,
    old_dims: PartitionedDimensions,
    new_layout: ChromosomeLayout,
    cfg: ChromosomeConfig,
) -> np.ndarray:
    """3-segment v1 → 4-region v4. Lossy: start_position dropped."""
    S = cfg.n_piece_types
    new = np.full(new_layout.total_length, S, dtype=np.int16)
    new[new_layout.crossing_overlay] = 0  # blank (0,0,0) triples

    # Main loop: copy active non-switch pieces
    old_main = old[:old_dims.n_main]
    active = old_main[old_main != -1]
    n_copy = min(len(active), cfg.max_loop_length)
    ml = segment_view(new, new_layout, 'main_loop')
    ml[:n_copy] = active[:n_copy]

    # Junctions → switch_mask[k]=2 + branch_slots[k] template expansion
    active_juncs = get_active_junctions(old, old_dims)
    sm = segment_view(new, new_layout, 'switch_mask')
    bs = segment_view(new, new_layout, 'branch_slots')
    sm[:] = 0
    for branch_k, (slot, _, pos, handedness, n_str) in enumerate(active_juncs):
        if branch_k >= cfg.max_branches: break
        if branch_k < cfg.max_switches: sm[branch_k] = 2
        bstart = branch_k * cfg.max_branch_length
        template = _template_for_handedness(handedness, n_str, S)
        n_tpl = min(len(template), cfg.max_branch_length)
        bs[bstart : bstart + n_tpl] = template[:n_tpl]
    return new
```

**What's lost:** `start_position` (v4 anchors at origin), junction position gene (v4 derives this implicitly from switch-occurrence order during decoding), active flag (v4 uses mask=2 presence).

## pymoo integration (verified)

Current `vtype=int` on `Problem.__init__` is the correct V2 pattern. Operators use `vtype=float`:

```python
algorithm = NSGA2(
    pop_size=200,
    sampling=IntegerRandomSampling(),
    crossover=SBX(prob=0.9, eta=15, vtype=float, repair=RoundingRepair()),
    mutation=PM(prob=1/layout.total_length, eta=20, vtype=float, repair=RoundingRepair()),
    eliminate_duplicates=True,
)
```

This is the **confirmed 0.6.1.6 integer pipeline**. Population stores float64; `RoundingRepair` casts back to int after each operator. int16 governs serialization outside the GA loop.

## `validate_invariants` hot path

```python
def validate_invariants(x, layout, cfg) -> bool:
    if x.shape != (layout.total_length,): return False
    S = cfg.n_piece_types
    ml = x[layout.main_loop]
    if np.any((ml < 0) | (ml > S)): return False
    sm = x[layout.switch_mask]
    if np.any((sm < 0) | (sm > 2)): return False
    bs = x[layout.branch_slots]
    if np.any((bs < 0) | (bs > S)): return False
    co = x[layout.crossing_overlay]
    for k in range(cfg.max_crossings):
        base = 3 * k
        i, j, p = int(co[base]), int(co[base+1]), int(co[base+2])
        if i == 0 and j == 0: continue  # blank triple = inactive
        if not (0 <= i < j <= cfg.max_loop_length): return False
        if not (0 <= p <= S): return False
    # Switch-mask length vs switches in main-loop prefix
    loop_prefix_len = int(np.argmax(ml == S)) if S in ml else cfg.max_loop_length
    switch_count = int(np.sum((ml[:loop_prefix_len] >= 5) & (ml[:loop_prefix_len] <= 8)))
    branch_here_count = int(np.sum(sm[:switch_count] == 2))
    if branch_here_count > cfg.max_branches: return False
    return True
```

## Dynamic sizing (no magic numbers)

```python
@classmethod
def from_optimization_config(cls, cfg, catalog) -> "ChromosomeConfig":
    inv = cfg.inventory
    type_a = sum(inv.get(pid, 0) for pid in ("R40_SWITCH_LEFT_IN", "R40_SWITCH_RIGHT_OUT"))
    type_b = sum(inv.get(pid, 0) for pid in ("R40_SWITCH_LEFT_OUT", "R40_SWITCH_RIGHT_IN"))
    max_switches = min(type_a, type_b)
    max_branches = max_switches
    total_straights = inv.get("STRAIGHT_16", 0) + inv.get("STRAIGHT_24", 0)
    max_branch_length = total_straights + 2
    non_switch_total = sum(c for pid, c in inv.items()
                            if catalog._id_to_index.get(pid) not in {5, 6, 7, 8})
    max_loop_length = non_switch_total + max_switches * 2
    max_crossings = inv.get("CROSS_90", 0) + inv.get("DOUBLE_CROSSOVER", 0)
    return cls(
        n_piece_types=catalog._max_index + 1,
        max_loop_length=max_loop_length,
        max_switches=max_switches,
        max_branches=max_branches,
        max_branch_length=max_branch_length,
        max_crossings=max_crossings,
    )
```

---

# Decoder — Phase 5 Findings

## Corrections to V2 spec

| V2 says | Research finding | Action |
|---|---|---|
| Decoder in radians | Current FK kernel `_compute_path_fk` is in degrees; Phase 2 geometry migration is deferred | **Operate in degrees internally, convert at DecodedLayout boundary** (option b) |
| Phase 5 blocks on Phase 4 | **Phase 5 can ship 80% against a compat shim** — encoding-agnostic parts are independent | Build `DecodedLayout` types + invariants + prefilter + closure now; branch/switch walk waits for Phase 4 |
| Numba required for performance | Current pure-Python is ~150 µs/decode; numba is a premature optimization | Structure for future JIT (pure-array signatures) but ship pure Python |
| 2^N path enumeration required | Not required — `v_bottleneck` is min over union of all pieces across paths, naturally handles multi-path | Decoder pre-collapses to flat placements list |

## Current decoder surface (ground truth)

`src/decoder/construction.py:45`: `decode_chromosome(x, catalog, inventory, dims, config) -> MultiPathLayout`. Returns mutable `MultiPathLayout` with pre-enumerated 2^N `TraversalPath` objects, `loose_port_count`, `start_position`. **No status enum, no infeasibility signal** — always returns something (empty layout is the "fail" state).

Relation to `DecodedLayout`:
- `MultiPathLayout` — mutable, path-enumerated, no census, no status
- `DecodedLayout` — frozen+slots, flat placements, explicit status, census dict
- They're conceptually orthogonal. Current optimizes for 2^N routing; V2 optimizes for problem-layer vectorization.

## 7-phase mapping to current code

| V2 Phase | Current? | Where | Reusable? |
|---|---|---|---|
| P1 validate_invariants | Partial | `validate_chromosome()` in `encoding.py:331` | Rewrite |
| P2 lattice prefilter | **Absent** | — | New |
| P3 main-loop turtle walk | Yes | `_compute_path_fk()` at `construction.py:580` | Kernel reusable |
| P4 branch extraction | Partial | `_read_junctions` + `_inject_switches` at `construction.py:131–285` | Tightly coupled to v1 encoding — rewrite |
| P5 closure measurement | Yes | `compute_closure_metrics()` in `geometry.py` | Reusable |
| P6 collision detection | Yes | `find_crossing_pairs()` in `intersection.py` | Partial — current does repair+replace, not count+report |
| P7 census/output | **Absent** | — | New |

## Lattice prefilter (numpy-vectorized)

```python
ATOMIC_ANGLE_DEG = 180.0 / 32   # = 5.625°

def lattice_prefilter(gene_array, dtheta_table) -> bool:
    """True if closure possible (lattice sum ≡ 0 mod 64) OR non-lattice (skip)."""
    active = gene_array[gene_array >= 0]
    if len(active) == 0: return True
    dthetas = dtheta_table[active]
    remainders = np.abs(dthetas % ATOMIC_ANGLE_DEG)
    on_lattice = np.all(np.minimum(remainders, ATOMIC_ANGLE_DEG - remainders) < 1e-9)
    if not on_lattice: return True  # defensive: skip on mixed lattice
    k_values = np.round(dthetas / ATOMIC_ANGLE_DEG).astype(np.int32)
    return int(k_values.sum() % 64) == 0
```

For current 10-piece catalog (all lattice): R40 contributes ±4, straights contribute 0, CROSS_90 contributes 16 (90°/5.625°), switches contribute 0 (through) or ±4 (diverging). Rejects ~98.4% of uniformly random closures.

## `DecodedLayout` (frozen with slots)

```python
class DecoderStatus(Enum):
    FEASIBLE                     = 0
    INFEASIBLE_INVARIANTS        = 1
    INFEASIBLE_ANGULAR_LATTICE   = 2
    INFEASIBLE_CROSSING_GEOMETRY = 3
    INFEASIBLE_BRANCH_STRUCTURE  = 4
    INFEASIBLE_INVALID_PIECE     = 5

@dataclass(frozen=True, slots=True)
class PlacedPiece:
    piece_id:            int
    entry_port:          str
    exit_port:           str
    world_pose_at_entry: tuple[float, float, float]  # (x, y, theta_rad)
    layer:               Literal["main", "branch", "crossing"]

@dataclass(frozen=True, slots=True)
class DecodedLayout:
    placements:       tuple[PlacedPiece, ...]
    closure_residual: tuple[float, float, float]  # (dx, dy, dtheta_rad)
    collision_list:   tuple[tuple[int, int], ...]
    census:           dict[int, int]
    status:           DecoderStatus
    status_reason:    str = ""      # non-empty only on infeasibility
```

`world_pose_at_entry` stored as plain tuple (not `Pose2D`) — decouples from Phase 2 geometry package.

## Branch-slot counter rule (fix for severed genotype-phenotype)

Current decoder counts k-th switch of any kind; V2 fix: count k-th mask=2 switch:

```python
switch_count = 0       # indexes switch_mask[]
branch_here_count = 0  # indexes branch_slots (slot k = k-th mask=2 switch)
switch_log = []

for loop_idx, g in enumerate(active_main_genes):
    spec = catalog[g]
    if spec.is_switch:
        m = switch_mask[switch_count]
        switch_count += 1
        if m == 2:
            exit_name = "B"
            switch_log.append((loop_idx, turtle, spec))
            branch_here_count += 1
        elif m == 1: exit_name = "C"
        else: exit_name = "B"
    ...
    main_layout.append(PlacedPiece(...))
    turtle = compose(turtle, spec.port_delta(exit_name))
```

Phase 4 then reads `branch_slots[k * max_branch_length : (k+1) * max_branch_length]` using `switch_log[k]`'s throat pose.

## Geometry decoupling — degrees internally

Recommended strategy: decoder's FK loop operates in degrees (matches current kernel), converts to radians ONLY at the `DecodedLayout` boundary:

```python
# world_pose_at_entry: (x_studs, y_studs, theta_rad)
theta_rad = math.radians(turtle_theta_deg)
```

When Phase 2 (geometry) lands and adds radian-native `compose`, swap the FK kernel in one localized change; `DecodedLayout` API stays unchanged.

## Numba-ready pure-array signature

```python
def _main_loop_walk_py(
    gene_array, sentinel, fk_table, is_switch, is_crossing, switch_mask
) -> tuple[turtle_xs, turtle_ys, turtle_hs, exit_ports,
           switch_log_idx, switch_log_poses, n_placed, n_branches]:
```

All inputs are numpy arrays or scalars. No `Pose2D` instances, no catalog wrapper. When numba is opted in, `@njit(cache=True)` decorator + no other changes.

## Phase 5 ship-against-shim strategy

Parts that can ship NOW (encoding-agnostic):
- `DecoderStatus` enum
- `PlacedPiece`, `DecodedLayout` dataclasses
- `validate_invariants()` — takes generic chromosome shape
- `lattice_prefilter()` — reads only gene array + catalog fk_table
- Closure measurement (reuses existing `compute_closure_metrics`)

Parts that wait for Phase 4:
- Phase 3 main-loop walk with switch_mask indexing — requires 4-region encoding
- Phase 4 branch extraction — requires branch_slots region

The shim: a thin adapter `src/decoder/compat.py` extracts `switch_mask` and `branch_slots` from current encoding by remapping `ValidatedJunction.handedness` → `switch_mask[k]=2` and `junction.branch_pieces` → `branch_slots[k·L_br : ...]`. Retired when Phase 4 lands.

---

# Problem — Phase 6 Findings

## Corrections to V2 spec

| V2 says | Research finding | Action |
|---|---|---|
| Infeasibility sentinel `F=[+inf, +inf]` | Current emits `F=[0.0, 0.0]` for empty — **a bug**: dominated by everything, corrupts Pareto front | Adopt V2 sentinel immediately (shim-mode fix) |
| `n_ieq_constr = 4 + |piece_types|` | Current is 6 (aggregated inventory); expand to 14 | Per-type inventory splits current G[3] into 10 entries |
| Closure in radians | Current angle_error is in degrees; bridge via `math.radians()` inline | Single-line bridge until Phases 2+5 ship |
| HV ref_point = `(+0.10, -0.55)` | Safe with `+inf` sentinels ONLY when filtering to feasible-only first | Implement per V2's callback pattern |

## Current problem.py surface (ground truth)

`TrackOptimizationProblem(ElementwiseProblem)` — already correct base class. No `_warm_up` method. No infeasibility sentinel (uses `F=[0.0, 0.0]` for empty — bug).

Current G (6 entries, `g ≤ 0` feasible):

| Index | Formula |
|---|---|
| G[0] | `(closure_err - closure_tol) / closure_tol` |
| G[1] | `(angle_err - angle_tol) / angle_tol`  — degrees |
| G[2] | `(boundary_violation - boundary_tol) / diagonal` |
| G[3] | `total_inventory_excess` — raw integer, unnormalized |
| G[4] | `(max_branch_closure - closure_tol) / closure_tol` or `-1` |
| G[5] | `count_segment_crossings(...)` — raw integer |

F: `[-utilization, -avg_speed]` where `avg_speed` from `SpeedProfile.avg_speed` (forward-backward 3-pass profile).

## V2 G vector (14 entries)

```python
S_XY = 0.5              # studs
S_THETA = math.pi / 180 # radians
COLLISION_SCALE = 5.0

# G[0]: closure x
abs(dx) / S_XY - 1.0
# G[1]: closure y
abs(dy) / S_XY - 1.0
# G[2]: closure theta (radians)
abs(dtheta) / S_THETA - 1.0
# G[3]: collisions
len(collision_list) / COLLISION_SCALE
# G[4..13]: per-type inventory excess (10 piece types)
max(0, census[t] - max_occ[t]) / max(1, max_occ[t])
```

## HV + `+inf` compatibility

`HV(ref_point)` is unsafe with `+inf` in F. **But** V2's `ConvergenceMonitorCallback` filters to feasible-only before calling HV:

```python
feas = CV.ravel() <= 0.0
F_feas = F[feas]
hv_val = float(self.hv.do(F_feas)) if len(F_feas) else 0.0
```

`+inf` never reaches HV. NSGA-II's feasibility-first selection (`ConstrRankAndCrowding`) uses CV alone for infeasible comparisons — never F. Non-dominated sorting on `[+inf, +inf]` vs another `[+inf, +inf]` is non-dominated (equal), which is fine; CV breaks the tie.

## `ConvergenceMonitorCallback` (full implementation)

```python
from pymoo.core.callback import Callback
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

class ConvergenceMonitorCallback(Callback):
    def __init__(self, ref_point=(0.10, -0.55), pareto_ref=None):
        super().__init__()
        self.hv = HV(ref_point=np.asarray(ref_point, dtype=float))
        self._pareto_ref = pareto_ref
        self.igd = IGD(pareto_ref) if pareto_ref is not None else None
        for k in ("n_gen", "n_eval", "hv", "igd", "n_feas", "feas_rate",
                  "mean_closure_x", "mean_closure_y", "mean_closure_theta"):
            self.data[k] = []
        self._best_F = None

    def notify(self, algorithm):
        pop = algorithm.pop
        F, G, CV = pop.get("F"), pop.get("G"), pop.get("CV")
        if F is None or CV is None: return

        feas_mask = CV.ravel() <= 0.0
        F_feas = F[feas_mask]
        n_feas = int(feas_mask.sum())

        hv_val = float(self.hv.do(F_feas)) if n_feas > 0 else 0.0
        igd_val = np.nan
        if n_feas > 0:
            self._best_F = self._update_best_front(F_feas)
            if self._pareto_ref is not None:
                igd_val = float(self.igd.do(F_feas))
            elif self._best_F is not None and len(self._best_F) > 0:
                igd_val = float(IGD(self._best_F).do(F_feas))

        mean_cx = mean_cy = mean_ct = np.nan
        if G is not None and G.shape[1] >= 3 and n_feas > 0:
            G_feas = G[feas_mask]
            mean_cx = float(np.mean((G_feas[:, 0] + 1.0) * 0.5))
            mean_cy = float(np.mean((G_feas[:, 1] + 1.0) * 0.5))
            mean_ct = float(np.mean((G_feas[:, 2] + 1.0) * (np.pi / 180)))

        self.data["n_gen"].append(algorithm.n_gen)
        self.data["n_eval"].append(algorithm.evaluator.n_eval)
        self.data["hv"].append(hv_val)
        self.data["igd"].append(igd_val)
        self.data["n_feas"].append(n_feas)
        self.data["feas_rate"].append(n_feas / max(1, len(pop)))
        self.data["mean_closure_x"].append(mean_cx)
        self.data["mean_closure_y"].append(mean_cy)
        self.data["mean_closure_theta"].append(mean_ct)

    def _update_best_front(self, F_new):
        if self._best_F is None or len(self._best_F) == 0:
            combined = F_new
        else:
            combined = np.vstack([self._best_F, F_new])
        fronts = NonDominatedSorting().do(combined, only_non_dominated_front=True)
        return combined[fronts]
```

## Shim-mode Phase 6 substitutes

Every V2 dependency has a shim before upstream phases ship:

| V2 dep | Shim substitute |
|---|---|
| `DecodedLayout.status` | `layout.n_pieces == 0` check |
| `closure_residual` (rad) | `path.closure_error` (studs) + `math.radians(path.angle_error)` inline |
| `collision_list` | `count_segment_crossings(...)` manual |
| `census` (dict) | `_compute_inventory_violation` expanded to per-type |
| `ChromosomeBounds.n_var` | `compute_dimensions(config, catalog).n_var` |
| `v_bottleneck` | `compute_speed_profile(...).min_speed` (not avg_speed!) |

## `algorithm/runner.py` integration

Existing callbacks:
- `ProgressCallback` — logs but doesn't record HV/IGD
- `FeasibleEliteCallback` — elite preservation; works with any G length
- `CallbackChain` — dispatcher
- `LegoAdaptiveEpsilon` — 3-phase epsilon schedule; CV aggregation length-agnostic

Changes for V2:
1. Swap `TrackOptimizationProblem` → `TrackLayoutProblem` in `run_optimization`
2. Add `ConvergenceMonitorCallback` to `CallbackChain` alongside existing callbacks

`save_results` is agnostic to constraint count / objective semantics — no changes.

---

# Batch 2 Summary Table

| Phase | Can ship against current code? | Key dependency | Shim value |
|---|---|---|---|
| 6 (Problem) | **Yes, shim-mode now** | none | Fix `+inf` bug, expand G, HV/IGD monitoring |
| 5 (Decoder) | **80% now** | Phase 4 for full walk | `DecodedLayout` types + prefilter + invariants |
| 4 (Chromosome) | **No — breaking change** | none (but breaks downstream) | Biggest risk; must be deliberate |

## Revised Batch 2 execution order

1. **Phase 6 shim** (~6-8 tasks): rewrite problem with V2 G vector, `+inf` sentinel, ConvergenceMonitorCallback. Bridges use `math.radians()` and `min_speed`.
2. **Phase 5 encoding-agnostic parts** (~8-10 tasks): `DecoderStatus`, `PlacedPiece`, `DecodedLayout`, `validate_invariants`, lattice prefilter. Adapter reads current `MultiPathLayout` and emits `DecodedLayout`.
3. **Phase 4 chromosome** (~15-20 tasks): the big one. Convert encoding, rebuild sampling/operators/repair to use 4-region layout. Migrate heuristics via `migrate_chromosome_v1_to_v4`.
4. **Phase 5 Phase 3–4 walk** (~5 tasks): drop the compat adapter, run turtle walk natively on 4-region chromosome.
5. **Retire shims in Phase 6** (~3 tasks): when Phase 3 (train v_bottleneck), Phase 4, Phase 5 all ship, drop the bridges.

## Cross-phase risks

1. **Switch FK discrepancy from Phase 1:** the shim preserves v1 switch values (31.0, ±6.2). V2's 3-4-5 derivation gives (32.69, ±12.96). This is **orthogonal to Phases 4/5/6** — they don't change FK values, they change encoding/decoding structure. Geometry correction remains a separate task.
2. **Phase 4's `max_switches` might differ from current `max_junctions`:** both compute `min(type_A, type_B)`, so they match — no risk.
3. **Template preservation across Phase 4:** LEFT_SIDING and RIGHT_SIDING must translate cleanly to branch_slot gene sequences (`[R40_RIGHT, straights..., R40_LEFT]`). The decoder's branch walk must produce the same placement as the template-based current decoder.

## Recommended next step

Draft Phase 6 (Problem) bite-sized plan first — it's the smallest, least risky, and delivers immediate bug fixes. Phases 5 and 4 plans follow after Phase 6 is implemented and stable.
