# Design: Simplify CROSS_90 to a DC-style 2-traversal descriptor

**Date:** 2026-05-31
**Status:** Implemented & unit/integration-verified (full GA run pending)

> **Implementation notes (2026-05-31):**
> - **§5.8 deviation:** no naive cross activate/move mutation was added. Per the
>   existing DC precedent (random descriptor mutation reliably breaks the geometric
>   pair and the piece is dropped), it would be net-harmful. Cross descriptors enter
>   via the figure-8 seed + crossover, and emergent crossings via the kept
>   `_straighten_near_unresolved_crossing` + FK-neutral repair (the by-chance path).
> - **Follow-up:** `runner.py` progress logging derives piece count from the now-
>   weighted utilization objective, so logged "pcs" reads effective (not physical)
>   once specials appear. Cosmetic; not yet fixed.
> - Zero test regressions vs HEAD (34→26 pre-existing failures, +25 passes).
**Author:** Damian Grzesło (with Claude)

---

## 1. Problem & motivation

In the `all_pieces` config, `CROSS_90` is **never placed** — it is structurally
impossible to express, not merely out-competed.

### Evidence (measured, not assumed)

Run `outputs_v1/verify_all_pieces_2` (NSGA2, pop 1000, 300 gen, 1000/1000 feasible)
ended with `CROSS_90: 0/2` and `DOUBLE_CROSSOVER: 0/2`. Tracing the cause:

- **CROSS_90 is not a legal main-loop allele** (`MAX_MAIN_LOOP_PIECE = 2 = R40_CURVE`,
  `encoding.py:116`). The only way it can enter a layout is the cross-junction
  descriptor block.
- That block is dimensioned to **zero** slots in this config:
  ```
  compute_dimensions (encoding.py:228-229):
    cross_junc_switches = (left//4) + (right//4) = (3//4) + (3//4) = 0
    max_cross_junctions = min(cross_count=2, 0, r40//4=20) = 0
  ```
  Confirmed against the live chromosome: the cross block spans `420 → 420` (empty),
  and the run log's `junctions=3` counts only passing sidings.

The root cause is the **cross-junction construction model**: it bundles a CROSS_90
with **4 same-handed switches + 4 spur curves** into one elaborate junction. With
only 3 LEFT + 3 RIGHT switches, `floor(3/4) = 0`, so not even one junction can be
built and the `CROSS_90: 2` inventory is dead weight.

(For context, DOUBLE_CROSSOVER fails for a *different* reason — it is expressible
and feasible, but its small figure-8 seeds reach only 19–42% utilization vs 51–61%
for plain racetracks, and the degenerate speed objective leaves utilization as the
sole selective pressure, so DC is bred out. The incentive change in this spec
addresses that too.)

## 2. Goal

Replace the 4-switch cross-junction model with a **direct, geometrically honest**
CROSS_90 placement mechanism, and give the GA a reason to keep topological pieces
so they are not stripped as overhead.

## 3. Key insight

A self-crossing is geometrically **one physical CROSS_90 traversed twice** by the
loop — both passes straight-through, the two passes intersecting at ~90° — covering
all four ports. This is structurally identical to `DOUBLE_CROSSOVER` (one piece,
two traversals, four ports), whose descriptor model already works (its seeds decode
feasibly).

Because `CROSS_90` FK is `[16, 0, 0]` — identical to `STRAIGHT_16` — rewriting two
straight slots into a crossing is **exactly FK-preserving**. Closure cannot break.
This removes the fatal flaw of the existing `_apply_crossing_repair`, which rewrites
arbitrary pieces (often curves) into a straight CROSS_90 and thereby drops a 22.5°
heading change, breaking closure and rendering self-crossing candidates infeasible.

## 4. Design decisions (all confirmed)

1. **Approach A — DC-style descriptor.** CROSS_90 enters via an explicit
   `(active, pos_1, pos_2)` descriptor, dimensioned directly from `inv["CROSS_90"]`,
   mirroring `_inject_double_crossovers`.
2. **Descriptor (deliberate) + repair (by-chance).** *Revised 2026-05-31 after the
   user clarified the cross-junction concept.* The bare-crossing descriptor gives
   deliberate CROSS_90 placement (figure-8). **Keep** the self-intersection repair as
   the by-chance / emergent path, but make it **FK-neutral**: only fire on
   `STRAIGHT_16`-on-`STRAIGHT_16` crossings (where `CROSS_90 FK == STRAIGHT_16 FK`, so
   closure cannot break). The 4-switch routing cross-junction is *not* deliberately
   modelled — accepted as not-simulated (it never built in practice: dimensioned to 0,
   seeder/injector admit the geometry rarely validates). See
   [[project-cross-junction-definition]].
3. **Weighted utilization incentive.** Special pieces count for more toward the
   utilization objective so they are never pure overhead.
4. **Inventory census fix.** Count physical CROSS_90/DC pieces (via decoder records),
   not traversal slots — also fixes a latent DC double-count.

## 5. Detailed design

### 5.1 Encoding (`src/encoding.py`)

- Redefine the cross-junction descriptor semantics. Block width stays **3 genes**,
  so `n_var` formula and all `*_start/_end` offsets are unchanged:
  | gene | old | new |
  |------|-----|-----|
  | 0 | `CJ_ACTIVE` | `CJ_ACTIVE` (unchanged) |
  | 1 | `CJ_POSITION_W` | `CJ_POSITION_1` ∈ `[0, n_main-1]` |
  | 2 | `CJ_HANDEDNESS` | `CJ_POSITION_2` ∈ `[0, n_main-1]` |
- **Dimensioning** (`compute_dimensions`): replace the switch-derived formula with
  ```python
  max_cross_junctions = inv.get("CROSS_90", 0)
  ```
  Delete `cross_count`, `cross_junc_switches`, and the `r40//4` terms.
- Update `generate_bounds` cross-junction loop: both position genes bounded to
  `[0, n_main-1]`; remove the handedness bound.
- Update accessors `get_cross_junction` / `set_cross_junction` /
  `get_active_cross_junctions` to `(active, pos_1, pos_2)` and sort active
  descriptors by `pos_1`.
- Update `create_empty_chromosome` and `create_chromosome_from_pieces`
  (`cross_junctions` tuple shape → `(active, pos_1, pos_2)`).
- Update module docstrings and the `[junc_end, cross_junc_end)` comment.

### 5.2 Decoder (`src/decoder/construction.py`)

Rewrite `_inject_cross_junctions` as a near-copy of `_inject_double_crossovers`.
For each active `(slot, pos_1, pos_2)`:

1. Order `pos_1 < pos_2`; skip if equal or out of range.
2. Skip if either slot is occupied (switch / siding / DC) or not `STRAIGHT_16`.
3. Skip if `CROSS_90` inventory is unavailable (`tracker.can_use`).
4. **Geometric validation:** compute FK states (honoring the existing DC `route_map`)
   at `pos_1` and `pos_2`; require their world midpoints coincide within
   `config.siding_position_tolerance` **and** their headings differ by ~90° within
   `config.siding_angle_tolerance`. Reuse the midpoint / perpendicular logic that
   `count_dangling_cross_ports` already encodes.
5. **Commit:** release the two `STRAIGHT_16`s, consume **one** `CROSS_90`
   (`tracker.use`), set both slots to `CROSS_90_INDEX`, append a `CrossJunction`
   record, mark both positions occupied.

Delete the now-unused helpers `_cross_center_from_w_switch`, `_target_switch_state`,
and `_find_main_position_matching` (callers: only `_inject_cross_junctions`).

**Keep** decoder Step 4 (`_apply_crossing_repair`) as the by-chance path, but make it
**FK-neutral**: only convert a crossing slot when **both** crossing segments are
`STRAIGHT_16` (so the rewrite to `CROSS_90`, FK `[16,0,0]`, preserves geometry).
Crossings involving a curve are left unconverted (and remain penalized via
`g_collisions`) rather than rewritten — eliminating the closure-breaking behaviour.

### 5.3 Types (`src/types.py`)

Redefine `CrossJunction` to mirror `DblCrossover`:
```python
@dataclass
class CrossJunction:
    slot: int
    positions: Tuple[int, int]            # two main-loop slots that coincide
    origin: Tuple[float, float, float]    # world pose of the crossing center

    def is_valid(self) -> bool:
        return (self.positions[0] >= 0 and self.positions[1] >= 0
                and self.positions[0] != self.positions[1])
```
`MultiPathLayout.cross_junctions` / `n_cross_junctions` are unchanged in shape.
No external readers of the old `switch_positions` / `cross_center` fields exist
(verified: only `is_valid()` and the decoder constructor).

### 5.4 Templates (`src/templates.py`)

Delete the 4-switch cross model: `CrossJunctionTemplate`, `CROSS_JUNCTION_LEFT`,
`CROSS_JUNCTION_RIGHT`, `CROSS_JUNCTION_TEMPLATES`, `switch_position_for_cross_port`,
`get_cross_junction_inventory_requirements`. (Callers: only the old inject + its test.)

### 5.5 Objective — weighted utilization (`src/problem.py`, `src/config.py`)

**Physical-piece count (shared definition).** `MultiPathLayout.n_pieces`
(types.py:358) counts the main path's `piece_sequence`, which traverses **both**
slots of a crossing/DC — so it counts one physical CROSS_90/DC as **2**. That is
inconsistent with `total_inventory` (denominator) and the constraint census, which
count physical pieces. Define a single physical count, used by **both** the objective
and §5.6:
```python
n_paired = len(layout.cross_junctions) + len(layout.dbl_crossovers)
physical_main = len(layout.main_loop_pieces) - n_paired   # each paired piece spans 2 slots
physical_pieces = physical_main + sum(len(sp.branch_pieces) for sp in layout.switch_pairs)
```
Leave `MultiPathLayout.n_pieces` untouched (other consumers / path-level uses depend
on it); compute `physical_pieces` in `problem.py`. The run-summary log should report
`physical_pieces` so logged utilization matches the objective.

- Add `special_piece_weight: float = Field(default=3.0, ge=1.0)` to
  **`OptimizationConfig`** (alongside `closure_tolerance` / `boundary_tolerance`) and
  thread it into `TrackOptimizationProblem.__init__`. Grep `src/` to confirm a real
  consumer reads the field — orphan yaml fields silently lie about being controls.
- Change `F[0]`:
  ```python
  n_special = len(layout.switch_pairs) + len(layout.cross_junctions) + len(layout.dbl_crossovers)
  effective = physical_pieces + (self.special_piece_weight - 1.0) * n_special
  utilization = effective / self.total_inventory
  ```
  Each special piece is worth `W` physical pieces toward the score, so folding a
  crossing into an otherwise-large loop *raises* it. Raw `physical_pieces` and raw
  utilization% remain in logging for honest reporting.

### 5.6 Inventory constraint fix (`src/problem.py`)

`_compute_per_type_inventory_violation` currently counts **every** occurrence in
`main_loop_pieces`, so a twice-traversed piece counts as 2 against inventory — a
latent double-count that already affects DC and would cap CROSS_90 at one physical
crossing despite `inv=2`. Fix: in the census loop, skip `CROSS_90_INDEX` and
`DOUBLE_CROSSOVER_INDEX`, then add `len(layout.cross_junctions)` to `census[CROSS_90]`
and `len(layout.dbl_crossovers)` to `census[DOUBLE_CROSSOVER]`. This counts physical
pieces, consistent with the decoder's `InventoryTracker` (which uses 1 per physical
piece).

### 5.7 Sampling (`src/sampling.py`)

- Update the `CrossJunctionDescriptor` type alias to `(active, pos_1, pos_2)`.
- Replace `_gen_oval_with_cross_junction` (4-switch seed) with a **CROSS_90 figure-8
  seed**, analogous to `_gen_figure_eight_dbl_crossover`: a self-crossing loop whose
  two STRAIGHT_16 slots coincide perpendicular, emitting one `(1, pos_1, pos_2)`
  descriptor. Gate on `dims.max_cross_junctions >= 1` and `inv[CROSS_90] >= 1`.
- Update the heuristic pattern tuple plumbing accordingly.

### 5.8 Operators (`src/operators.py`)

- **Keep** `_straighten_near_unresolved_crossing` — it nudges curve-on-curve crossings
  toward the STR-on-STR crossings the FK-neutral repair converts (supports the
  by-chance path).
- Add a CROSS_90 descriptor mutation portfolio mirroring the DC operators:
  an activate-toggle and a move-position operator over `(pos_1, pos_2)`, guarded by
  `dims.max_cross_junctions == 0`.
- Confirm `PartitionedCrossover` already handles the cross block generically (it
  operates on descriptor ranges); adjust if it referenced handedness semantics.

## 6. Test plan

- **Rewrite `tests/test_cross_junction_inject.py`** for the `(active, pos_1, pos_2)`
  signature. The 4 failure-path tests (non-straight slot, insufficient inventory,
  occupied slot, geometry-unmatched) still apply — only the descriptor setup and the
  "4-port" framing change to "perpendicular coincidence." Add a **success path**: a
  hand-built self-crossing loop where two STRAIGHT_16 slots coincide perpendicular →
  decoder places one CROSS_90 at both slots, consumes exactly 1 from inventory,
  closure preserved, `count_dangling_cross_ports == 0`.
- **Update `tests/test_sampling.py::TestCrossJunctionSeeder`** for the new figure-8
  seed.
- **Update `tests/test_crossing_repair.py`** for the FK-neutral policy: STR-on-STR
  crossings still convert to CROSS_90; a curve-involved crossing is now left
  unconverted (was previously rewritten). Keep the file.
- Update any encoding/`test_problem.py` assertions touching cross-junction semantics,
  the weighted-utilization objective, and the inventory-census fix.
- Run the **full** suite (no `-k` subset) before and after.

## 7. Out of scope (flagged)

- **Rendering:** `visualization/track_renderer.py` does not draw crossings distinctly;
  a placed CROSS_90 will not be visually obvious. Worth a follow-up, not part of this
  change.
- **Speed objective degeneracy:** the ~1.0 m/s flat second objective provides no
  selective pressure. The weighted-utilization incentive is a targeted mitigation;
  reworking the speed objective is a separate effort.

## 8. Risks

- **GA discovery:** direct placement + incentive *enable* crossings but do not
  guarantee the GA finds high-utilization self-crossing layouts. The figure-8 seed
  plus the descriptor mutation are the discovery aids; validate empirically with a
  full `all_pieces` run and `/diag` afterward (target: `CROSS_90` usage > 0 in a
  feasible best, layout closed and connected).
- **`special_piece_weight` tuning:** 3.0 is a starting default; may need adjustment
  if topology pieces dominate or remain ignored.

## 9. Verification

After implementation: full pytest suite green, then a full `all_pieces` optimizer run
followed by `/diag` confirming closure error, zero dangling/orphan ports, feasible
best uses ≥1 CROSS_90, and the layout is closed and connected. No assertion of success
without literal command output.