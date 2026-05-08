# Port-pair encoding: implementation plan

**Status**: Drafted, awaiting Phase 0 → 1 kickoff
**Branch**: `feat/catalog-v2-phase-1`
**Target tree**: `src_v2/` (parallel to existing `src/`, which stays untouched)

---

## 1. Goal

Replace the V1 partitioned chromosome (`encoding.py` main loop + junction descriptors)
with a port-graph encoding that natively expresses:

- Geometrically-derived sidings of arbitrary length (no fixed-length templates)
- Multi-path topologies: switches, CROSS_90, DOUBLE_CROSSOVER as first-class pieces
- Multi-loop layouts: figure-8s, parallel tracks joined by crossovers
- Reverse loops (where the train's direction reverses)

The V1 encoding cannot represent figure-8s, multi-loop networks, or geometrically arbitrary
sidings; the V1 decoder patches CROSS_90 in via repair injection rather than treating
crossings as topology. The port-pair encoding eliminates these limitations at the cost
of ~3-5x more chromosome genes.

---

## 2. Reuse inventory (already done)

`src_v2/` is created and contains, copied verbatim from `src/`:

| Module | Source | Status in src_v2 |
|---|---|---|
| `train/` | `src/train/` | ★ REUSED, imports clean |
| `types.py` | `src/types.py` | ★ REUSED — needs `PortGraph` extension (Phase 2) |
| `visualization/` | `src/visualization/` | ★ REUSED, absolute imports rewritten to relative |
| `sampling.py` | `src/sampling.py` | ★ COPIED — header flags V1 deps; pattern logic to be ported in Phase 5 |
| `catalog/` | `src/catalog/` | transitive dep, V1+V2 loaders intact |
| `config.py` | `src/config.py` | transitive dep |
| `geometry.py` | `src/geometry.py` | transitive dep (FK chain still used by train/visualization) |
| `lego_track_models.py` | `src/lego_track_models.py` | transitive dep (visualization constants) |

Verified: all eight modules import cleanly in `src_v2/`. No cross-package leak to old `src/`.

---

## 3. Design decisions (locked in)

| Decision | Choice |
|---|---|
| Storage dtype | **`int16` throughout** |
| Active-count gene | **None** — decoder counts sentinels |
| Sentinel | `-1` for inactive piece slots and inactive port-pair rows |
| Anchor representation | `(start_x, start_y, start_theta_deg)` as `int16` |
| Closure semantics | **Max residual over cycle-closing edges, per axis** |
| `min_cycle_count` constraint | **`≥ 1`** required (no acyclic degenerate layouts) |
| Heuristic seed ratio | **30%** of initial population from inventory/boundary-aware patterns |
| Crossover strategy (v0) | Uniform per-region (per-slot, per-pair-row, per-anchor) |
| `eliminate_duplicates` | Default array equality for v0; canonical graph hashing deferred |
| `edge_margin_studs` | 20.0 |
| `branch_overlay_factor` | 1.5 |
| Self-loops / multi-edges | Forbidden; repair drops |
| Disconnected components | Allowed; penalized via loose-port count |
| Internal pose units | Radians (matches V2 spec); convert to degrees only at `MultiPathLayout` boundary |
| Phase 8 ship gate | **≥80% V1 utilization on shared configs + feasible figure-8 layout** |

---

## 4. Chromosome layout

```
Index:   0 ────────────── N_max ──────────────── N_max + 4·E_max ── n_var
        │                │                       │                │
        │  piece_slots   │     port_pairs        │    anchor      │
        │  N_max ints    │  E_max rows × 4 ints  │    3 ints      │
```

Each piece slot: `int16` piece_id from catalog ordering, or `-1` if inactive.
Each port-pair row: `(slot_a, port_idx_a, slot_b, port_idx_b)`, all `int16`,
or `(-1, -1, -1, -1)` if inactive.
Anchor: `(x, y, theta_deg)` as `int16`.

Port indices map to position in `tuple(spec.ports)`: A=0, B=1, C=2, D=3.

### Dimension formula

```python
def compute_port_pair_dimensions(boundary, catalog, inventory=None,
                                 edge_margin_studs=20.0,
                                 branch_overlay_factor=1.5):
    avail_w = max(0, boundary.width  - 2 * edge_margin_studs)
    avail_h = max(0, boundary.height - 2 * edge_margin_studs)
    max_path = 2 * (avail_w + avail_h) * branch_overlay_factor

    min_axial = min(_axial_extent(spec) for spec in catalog.pieces)
    geometric_cap = int(max_path / min_axial)
    inventory_cap = sum(inventory.values()) if inventory else 10**9
    N_max = min(geometric_cap, inventory_cap)

    if inventory:
        total_ports = sum(len(catalog.by_id[pid].ports) * count
                          for pid, count in inventory.items()
                          if pid in catalog.by_id)
    else:
        avg_ports = sum(len(s.ports) for s in catalog.pieces) / len(catalog.pieces)
        total_ports = int(N_max * avg_ports)
    E_max = total_ports // 2

    return PortPairDimensions(
        N_max=N_max, E_max=E_max,
        n_var=N_max + 4 * E_max + 3,
        slot_start=0, slot_end=N_max,
        pair_start=N_max, pair_end=N_max + 4 * E_max,
        anchor_start=N_max + 4 * E_max,
    )
```

Worked examples:

| Box | Inventory | N_max | E_max | n_var |
|---|---|---|---|---|
| 200×200 | 60 pieces | 60 | 72 | 351 |
| 200×200 | 80 pieces | 62 (geom binds) | 74 | 361 |
| 200×200 | unlimited | 62 | 75 | 365 |
| 300×300 | 120 pieces | 101 | 121 | 590 |
| 100×100 | 60 pieces | 23 (geom binds) | 27 | 134 |

---

## 5. Phase plan

```
Phase 0 — design (DONE, decisions in §3 above)
    ↓
Phase 1 — encoding.py            (1 day)
    ↓
Phase 1.5 — se2.py               (0.5 day)
    ↓
Phase 2 — types.py PortGraph     (0.2 day)
    ↓
Phase 3 — decoder.py             (3.5 days, includes vertical slice gate)
    ↓ [GATE: vertical slice tests must pass before any further work]
Phase 4 — repair.py              (1 day)
    ↓
Phase 5 — operators.py           (2.5 days, includes sampling.py rewrite)
    ↓
Phase 6 — problem.py             (1 day)
    ↓
Phase 7 — algorithm/runner.py    (1 day)
    ↓
Phase 8 — validation             (1.5 days)
    ↓ [GATE: ship criteria from §3 must pass]

Buffer: 2 days for debugging, design pivots
Total:  ~14.5 days serial = ~3 weeks calendar
```

### Phase 1 — `encoding.py`

**Deliverables**:
- `PortPairDimensions` frozen dataclass
- `compute_port_pair_dimensions(config, catalog) -> PortPairDimensions`
- `generate_bounds(dims) -> (xl, xu)` returning `int16` arrays
- Accessors: `iter_active_slots`, `get_piece_slot`, `set_piece_slot`,
  `iter_active_pairs`, `get_port_pair`, `set_port_pair`,
  `get_anchor`, `set_anchor`
- Constants: `INACTIVE = -1`, `GENES_PER_PAIR = 4`, `ANCHOR_GENES = 3`
- `validate_chromosome(x, dims) -> List[str]`
- `create_empty_chromosome(dims)`
- `chromosome_stats(x, dims)`

**Tests** (`tests/test_encoding_v2.py`):
- Bounds shape matches `n_var`; xl/xu are int16
- Accessor round-trip: write + read returns same value for slots, pairs, anchor
- `compute_port_pair_dimensions` matches the worked examples in §4
- Geometric cap binds when inventory > geometric_cap; inventory binds otherwise
- Empty chromosome: zero active slots, zero active pairs

### Phase 1.5 — `se2.py`

**Deliverables**:
```python
def pose_compose(parent: tuple[float, float, float],
                 child_in_parent: tuple[float, float, float]) -> tuple[float, float, float]
def pose_inverse(p: tuple[float, float, float]) -> tuple[float, float, float]
def pose_diff(a: tuple[float, float, float],
              b: tuple[float, float, float]) -> tuple[float, float, float]  # residual
```

All radians internally. Pure functions, no state.

**Tests** (`tests/test_se2.py`):
- Identity: `pose_compose(A, identity) == A`
- Inverse: `pose_compose(pose_inverse(A), A) == identity` (within fp tol)
- Diff: `pose_diff(A, A) == identity`
- Hand-computed: 90° rotation + 10-stud forward → known pose

### Phase 2 — `types.py` `PortGraph` extension

**New types**:
```python
@dataclass
class PortGraph:
    slots: list[SlotInstance]
    edges: list[PortEdge]
    slot_poses: dict[int, tuple[float, float, float]]   # radians internally
    closure_residuals: list[CycleResidual]
    loose_ports: list[tuple[int, str]]                  # (slot_idx, port_name)
    connected_components: list[set[int]]

@dataclass(frozen=True)
class SlotInstance:
    slot_idx: int
    piece_id: str
    piece_index: int

@dataclass(frozen=True)
class PortEdge:
    slot_a: int
    port_a_name: str
    slot_b: int
    port_b_name: str

@dataclass(frozen=True)
class CycleResidual:
    slot_a: int
    slot_b: int
    dx: float
    dy: float
    dtheta: float  # radians
```

`MultiPathLayout`, `SwitchPair`, `TraversalPath` (existing) carry over unchanged.
A `derive_switch_pairs(graph, catalog)` adapter (in `decoder.py`, not `types.py`)
translates port-graph topology into V1-compatible `SwitchPair` instances for
visualization compatibility.

### Phase 3 — `decoder.py`

**Algorithm**: chromosome → `PortGraph` → `MultiPathLayout`.

```
decode_chromosome(x, dims, catalog, config) -> MultiPathLayout:
  1. parse_chromosome(x, dims) -> (active_slots, raw_edges, anchor)
  2. validate_edges(raw_edges, active_slots, catalog) -> sanitized_edges
     - drop edges where slot is inactive
     - drop edges where port_idx exceeds piece's port count
     - drop self-loops
     - drop double-booked ports (each port in ≤ 1 edge)
  3. union_find(sanitized_edges) -> connected_components
  4. for each component:
       anchor_slot = lowest slot_idx in component
       bfs_pose(anchor_slot, anchor pose, edges, catalog) ->
         (slot_poses, cycle_residuals)
  5. enumerate_traversal_paths(slots, edges, catalog) -> [TraversalPath]
  6. main_path = longest closed traversal path with all default routes
  7. derive_switch_pairs(graph, catalog) -> [SwitchPair]   # for vis compat
  8. auto_center(graph, boundary, anchor) -> shifted poses
  9. build MultiPathLayout
```

**Vertical slice gate** — must pass before Phase 4 begins:

| Test | Input | Expected |
|---|---|---|
| 16-piece R40 LEFT circle | hand-crafted chromosome with 16 R40_LEFT slots, 16 chain edges + 1 closing edge | closure_error < 0.5, n_paths = 1 |
| Oval (8 R40 + 4 STR) ×2 | hand-crafted oval chromosome | closure < 0.5, n_paths = 1 |
| Switch + branch + merge | IN switch, 3-piece branch, OUT switch, with both diverging ports paired | n_paths = 2 (main + branch), both closed |
| CROSS_90 figure-8 | 2 cycles sharing a CROSS_90 piece, both routes connected | n_paths = 2 closed cycles, both passing through CROSS_90 |
| Disconnected: 2 small loops | 2 separate 4-piece R40 cycles in same chromosome | 2 components, both closed |
| Invalid edges | chromosome with double-booked ports | repair drops; decoder produces valid output |

If any of these fail, **stop and re-evaluate the approach**. Don't proceed to Phase 4.

### Phase 4 — `repair.py`

**Pipeline**:
1. `EdgeValidityRepair`: drop invalid edges (sentinel-row, out-of-range port_idx, self-loop, double-booked)
2. `InventoryRepair`: count active piece types, deactivate excess slots from end
3. `ConnectednessRepair`: optional, drops slots in components below `min_component_size` (default off — disconnected components allowed)
4. **Iterate to fixed point** (drop-then-recount in case earlier steps create new orphans)

**Tests**:
- Chromosome with deliberately-broken edges → repair → `validate_chromosome` returns no errors
- Inventory-overflow → repair brings counts within budget
- Repair on already-valid chromosome is identity (no spurious changes)

### Phase 5 — `operators.py` + `sampling.py` rewrite

**`PortPairSampling`**:
- Heuristic emitters (rewritten from `sampling.py` patterns to emit port-pair tuples):
  - `_gen_simple_loop_pp`: 16 R40 cycle
  - `_gen_oval_pp`: 16 curves + 2N straights
  - `_gen_racetrack_pp`: 4 corners + N straights per long side, M per short side
  - `_gen_oval_with_siding_pp`: oval + IN/OUT switches with branch
  - `_gen_oval_two_sidings_pp`: oval + two sidings
  - `_gen_figure_8_pp`: NEW — two cycles sharing a CROSS_90
  - `_gen_multi_loop_pp`: NEW — two disconnected cycles
- 30% heuristic / 70% random per `OptimizationConfig.algorithm.heuristic_ratio`
- Random sampling: random piece type per slot, random port-pair generation, anchor offset

**`PortPairCrossover`** (uniform per-region):
- Per-slot uniform from parent A or B
- Per-pair-row uniform from parent A or B
- Anchor uniform from one parent

**`PortPairMutation`** (sub-operators):
| Op | Weight | Effect |
|---|---|---|
| `mutate_piece_type` | 0.20 | change one slot's piece_id |
| `activate_slot` | 0.15 | set inactive slot to random valid piece_id |
| `deactivate_slot` | 0.10 | drop one slot + edges referencing it (bounded by min-active) |
| `add_edge` | 0.20 | pair two currently-loose ports |
| `remove_edge` | 0.10 | drop one edge |
| `rewire_edge` | 0.15 | change one endpoint of an edge |
| `perturb_anchor` | 0.10 | shift anchor by ±delta |

**Tests**:
- Each emitter produces inventory-valid chromosome that decodes to expected n_paths
- Crossover output decodes successfully after repair
- Each mutation sub-op preserves validity after repair

### Phase 6 — `problem.py`

**Constraint vector**:
- `G[0..2]`: max-residual closure per axis (over all cycle-closing edges)
- `G[3]`: boundary violation (max excess across all paths)
- `G[4]`: collisions (true geometric crossings without CROSS_90/DOUBLE_CROSSOVER piece)
- `G[5..4+T]`: per-type inventory excess (T = catalog.n_pieces)
- `G[5+T]`: loose-port count (must be 0 unless terminator pieces in catalog)
- `G[6+T]`: `1 - cycle_count` (ensures ≥ 1 closed cycle)

Total: `7 + T` inequality constraints.

**Objectives** (unchanged from V1):
- `F[0] = -utilization`
- `F[1] = -min_speed` (V2 bottleneck semantics)

### Phase 7 — `algorithm/runner.py`

**Reuse verbatim from existing `src/algorithm/runner.py`**:
- `ProgressCallback`, `FeasibleEliteCallback`, `LegoAdaptiveEpsilon`,
  `SnapshotCallback`, `CallbackChain`, `_compute_snapshot_targets`
- `save_results`
- All callbacks consume `pop.get("X")` opaquely → encoding-agnostic

**Wire new operators**:
- `IntegerSampling` → `PortPairSampling`
- `PartitionedCrossover` → `PortPairCrossover`
- `PartitionedMutation` → `PortPairMutation`
- `TrackRepairPipeline` → port-pair `RepairPipeline`

### Phase 8 — validation gate

| Test | V1 baseline | V2 must achieve |
|---|---|---|
| `default.yaml` best feasible utilization | run V1, record | ≥ 80% of V1 |
| `with_switches.yaml` feasible-solution count | run V1, record | ≥ 50% of V1 |
| `with_switches.yaml` switches in best layout | run V1, record | ≥ matched-switch-pair count of V1 |
| `with_crossing.yaml` CROSS_90 used in best | possibly absent in V1 | present and active |
| `figure_8.yaml` (NEW config) | V1 cannot represent | V2 produces feasible figure-8 |

**V2 ships if all five pass.**

If V2 fails: archive `src_v2/` as `experiments/port_pair_attempt/`, write postmortem, return to V1.
If V2 partially succeeds: keep both, document per-config selection.

---

## 6. Risk register

| Risk | Severity | Mitigation |
|---|---|---|
| Decoder cycle-closure math wrong | **High** | Vertical slice gate at Phase 3; hand-verified 4-/6-/8-piece tests before scaling |
| Random sampling produces 0% feasible chromosomes | **Medium-high** | 30% heuristic ratio (chosen); adaptive ε keeps infeasible-but-promising alive; aggressive repair |
| Operators break invariants faster than repair fixes | **Medium** | Each mutation sub-op asserts validity after repair; in-place rejection if repair can't recover |
| Inflated diversity from graph-isomorphism duplicates | **Medium** | Accepted for v0; pop_size 1000 mitigates; canonical hashing deferred to v1 |
| Convergence noticeably worse than V1 | **Low-medium** | Expected 1.5–2× longer runs; Phase 8 gate accepts up to 80% V1 quality |
| pymoo API drift | **Low** | Verify via context7 before Phases 4 + 5 |
| Decoder performance regression | **Low** | Profile after Phase 3; optimize only if eval rate < 100/sec |
| V2 underperforms V1 with no figure-8 win | **Project-level** | Phase 8 explicit ship gate; archive path defined |

---

## 7. Existing src/ during this work

Existing `src/` stays untouched and functional throughout. `main.py`, `configs/`, all
existing tests keep using `src/`. New `main_v2.py` (or `--encoding port-pair` flag in
`main.py`) selects `src_v2/` for runs.

After Phase 8 validation:
- **Pass**: promote `src_v2/` to `src/`, archive old `src/` as `experiments/v1_partitioned/`,
  update `main.py` to use the new modules.
- **Fail**: archive `src_v2/` as `experiments/port_pair_attempt/`, write postmortem.

---

## 8. Open questions deferred to follow-ups

- Canonical graph hashing for `eliminate_duplicates` (v1)
- Subgraph-swap crossover as additional sub-operator (v1, if uniform-per-region produces too little structural mixing)
- Radian-native `geometry.compute_fk_chain` (separate concern from port-pair encoding)
- Automatic per-route physics block in V2 YAML (separate concern; tracked in Phase 4 of V2 schema migration)

---

## 9. Immediate next actions

1. Review this plan; redirect or approve
2. Phase 1: implement `src_v2/encoding.py` + tests
3. Phase 1.5: implement `src_v2/se2.py` + tests
4. Phase 3: implement `src_v2/decoder.py` with vertical slice tests as the first deliverable
5. **Stop at vertical slice gate**, re-evaluate if any test fails

The vertical slice gate is the single most important risk-management mechanism in this plan. It validates the entire approach with ~3 days of work; failure caps the loss at ~3 days rather than the full ~3 weeks.
