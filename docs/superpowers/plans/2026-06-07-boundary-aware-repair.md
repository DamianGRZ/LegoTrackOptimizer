# Boundary-Aware Repair Operator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> ⚠️ **COMMIT POLICY (hard rule for this repo):** Do **NOT** run `git add` / `git commit` autonomously. The user is the *only* one who decides when to commit. Every "Checkpoint" step below is a **STOP point**: show the diff + test output and ask for explicit approval before committing. Never bundle commits.

**Goal:** Add a `BoundaryAwareRepair` stage that rescues out-of-bounds layouts by (a) re-centering via the start-position genes when the loop already fits, or (b) symmetrically removing anti-parallel straight pairs to shrink a genuinely-too-big loop while preserving closure and the 360° angular budget.

**Architecture:** A new `Repair` subclass appended as the **4th stage** of `TrackRepairPipeline` (after `MainLoopClosureRepair`, so it acts on geometrically-closed chromosomes). It is Lamarckian (pymoo writes the repaired genotype back). It works on the raw partitioned chromosome using only `dims` + the catalog FK table — no full decode. Two branches: **translate** (zero `start_x/start_y` so the decoder's existing `_auto_center` fully centers the loop) and **shrink** (deactivate anti-parallel same-type straight pairs to reduce the over-the-box axis, then translate). After a shrink it re-runs junction clamping because deactivating straights shifts active-piece indices.

**Tech Stack:** Python 3.x, numpy, pymoo 0.6.1.6 (`pymoo.core.repair.Repair`), pytest.

---

## Critical Grounding Facts (verified in source)

- **`pymoo.core.repair.Repair._do(self, problem, X, **kwargs)`** — `X` is `(n_individuals, n_var)` int16; loop rows and mutate in place; return `X`. Confirmed by the three existing repairs in `src/repair.py` (e.g. `MainLoopClosureRepair._do`, `repair.py:71-74`). It is **Lamarckian** (returned `X` becomes the population).
- **Decoder already centers:** `src/decoder/construction.py:927-960` `_auto_center` sets `shift = boundary_center − layout_center + start_offset` over **all paths**, then `path.states += shift`. ⇒ **start_x = start_y = 0 ⇒ loop sits at box center (max boundary margin).** A non-zero offset is the main reason a *fitting* loop goes out of bounds.
- **Boundary constraint** `G[3]` (`src/problem.py:199-201, 234-253`) = `(max over all paths of overshoot − boundary_tolerance)/diagonal`, overshoot measured against `config.boundary.{min,max}_{x,y}`.
- **Encoding API** (`src/encoding.py`): `INACTIVE = -1`; `PieceIndex.STRAIGHT_16=0, STRAIGHT_24=1, R40_CURVE=2`; main types in `x[:dims.n_main]`; flips in `x[dims.main_flips_start:dims.main_flips_end]`; start genes at `x[dims.start_pos_start]`, `x[dims.start_pos_start+1]`; box on `dims.boundary_{min,max}_{x,y}`. Helpers: `get_active_main_pieces_with_flips`, `set_main_loop_type`, `set_flip`, `get_active_junctions`.
- **FK:** `src/geometry.py:14-38` `compute_fk_chain(fk_deltas)` returns `(n+1,3)` states from origin `[0,0,0]`. R40 flip negates `dy,dθ` (`construction.py:62-74` `get_fk_with_flip`). The catalog FK rows: `STRAIGHT_16=[16,0,0]`, `STRAIGHT_24=[24,0,0]`, `R40_CURVE=[15.307,3.045,22.5]`.
- **Pipeline & wiring:** `TrackRepairPipeline` chains JunctionValidity → Inventory → MainLoopClosure (`repair.py:397-401`); instantiated in `src/algorithm/runner.py:435-440` with `catalog_fk_table=catalog._fk_table`.
- **Corrected geometry:** removing one anti-parallel **pair** (one straight heading θ + one heading θ+180°, **same length L**) preserves closure (Σ vectors = 0) and angular budget (straights are 0°), and reduces that axis's span by **L** (not 2L). ⇒ `n_pairs = ceil(deficit / L)`, 2 straights removed per pair.

## Design Decisions (explicit, with rationale)

1. **v1 fit-test uses the MAIN-LOOP FK span only** (cheap, no decode). The siding branch adds a bounded perpendicular reach, so when ≥1 junction is active we shrink the effective box by a conservative `SIDING_MARGIN` on each side. This is conservative (never falsely declares "fits"); residual siding poke-out is left to the `G[3]` penalty. Full all-paths extent is a documented future refinement.
2. **`SIDING_MARGIN = 16.0` studs** — conservative: parallel-siding lateral spacing is 8 studs (project geometry) plus branch-curve reach; 16 is a safe over-estimate. Documented constant, refine empirically.
3. **Shrink is axis-aligned only in v1**: X via straights at heading ≈ 0°/180°, Y via ≈ 90°/270°. If the over-box axis has no same-type anti-parallel pair, **decline gracefully** (leave individual to the penalty) — never break closure.
4. **Probabilistic guards default OFF (prob=1.0)** so unit tests are deterministic and the empirical run shows the full effect. `translate_prob` / `shrink_prob` are exposed for tuning if the known feasibility/diversity collapse worsens. (Generation-gating via `kwargs['algorithm']` is **not** used — its availability in `Repair._do` is unverified; an internal call-counter is the documented fallback if gating is ever needed.)

## File Structure

- **Modify** `src/repair.py` — add module-level helpers `_main_loop_states`, `_active_straight_headings`, `_find_antiparallel_pairs`; add class `BoundaryAwareRepair(Repair)`; extend `TrackRepairPipeline.__init__`/`_do` with `enable_boundary_repair`.
- **Modify** `src/algorithm/runner.py:435-440` — pass `enable_boundary_repair=True` explicitly.
- **Create** `tests/test_boundary_repair.py` — unit tests for helpers + both branches + pipeline integration.

---

### Task 1: Main-loop FK span helper

**Files:**
- Modify: `src/repair.py` (add near top, after imports)
- Test: `tests/test_boundary_repair.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_boundary_repair.py
import numpy as np
import pytest

from src.encoding import (
    PartitionedDimensions, INACTIVE, PieceIndex,
    create_empty_chromosome, set_main_loop_type, set_flip,
)

# Catalog FK rows (index = piece index): [dx, dy, dtheta_deg]
FK = np.array([
    [16.0, 0.0, 0.0],      # 0 STRAIGHT_16
    [24.0, 0.0, 0.0],      # 1 STRAIGHT_24
    [15.307, 3.045, 22.5], # 2 R40_CURVE
    [0.0, 0.0, 90.0],      # 3 CROSS_90 (unused here)
    [0.0, 0.0, 0.0],       # 4 SWITCH_LEFT
    [0.0, 0.0, 0.0],       # 5 SWITCH_RIGHT
    [0.0, 0.0, 0.0],       # 6 DOUBLE_CROSSOVER
], dtype=np.float64)

S16 = int(PieceIndex.STRAIGHT_16)
R40 = int(PieceIndex.R40_CURVE)


def make_dims(n_main=24, max_junctions=0, box=250.0):
    return PartitionedDimensions(
        n_main=n_main, max_junctions=max_junctions,
        max_cross_junctions=0, max_double_crossovers=0,
        total_straights=n_main,
        boundary_min_x=-box, boundary_max_x=box,
        boundary_min_y=-box, boundary_max_y=box,
    )


def test_main_loop_states_single_straight():
    from src.repair import _main_loop_states
    dims = make_dims(n_main=4)
    x = create_empty_chromosome(dims)
    set_main_loop_type(x, dims, 0, S16)  # one 16-stud straight at heading 0
    states = _main_loop_states(x, dims, FK)
    # origin + one straight along +x
    assert states.shape == (2, 3)
    np.testing.assert_allclose(states[1], [16.0, 0.0, 0.0], atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_boundary_repair.py::test_main_loop_states_single_straight -v`
Expected: FAIL — `ImportError: cannot import name '_main_loop_states'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/repair.py — add after the existing imports / constants block
SIDING_MARGIN = 16.0  # conservative per-side reach of an active passing siding (studs)


def _main_loop_states(x: NDArray, dims: PartitionedDimensions,
                      fk_table: NDArray) -> NDArray:
    """FK states (origin-based) of the active main-loop pieces, flips applied."""
    from .geometry import compute_fk_chain
    types = x[:dims.n_main]
    flips = x[dims.main_flips_start:dims.main_flips_end]
    mask = types != INACTIVE
    active_types = types[mask].astype(int)
    active_flips = flips[mask].astype(int)
    if active_types.size == 0:
        return np.zeros((1, 3), dtype=np.float64)
    deltas = fk_table[active_types].copy()
    negate = (active_types == R40_CURVE) & (active_flips == 1)
    deltas[negate, 1] *= -1.0
    deltas[negate, 2] *= -1.0
    return compute_fk_chain(deltas)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_boundary_repair.py::test_main_loop_states_single_straight -v`
Expected: PASS.

- [ ] **Step 5: Checkpoint (STOP — do not commit without approval)**

Show the diff + test output to the user. Only `git add src/repair.py tests/test_boundary_repair.py && git commit` **if and when** they explicitly approve.

---

### Task 2: Anti-parallel straight-pair detection

**Files:**
- Modify: `src/repair.py`
- Test: `tests/test_boundary_repair.py`

- [ ] **Step 1: Write the failing test**

```python
def test_antiparallel_pair_detected_for_S16():
    from src.repair import _active_straight_headings, _find_antiparallel_pairs
    dims = make_dims(n_main=24)
    x = create_empty_chromosome(dims)
    # straight @0, 8 R40 left turns (=> heading 180), straight @180
    set_main_loop_type(x, dims, 0, S16)
    for i in range(1, 9):
        set_main_loop_type(x, dims, i, R40)
        set_flip(x, dims, i, 0)  # +22.5 each => 180 total
    set_main_loop_type(x, dims, 9, S16)

    headings = _active_straight_headings(x, dims, FK)
    # two straights: slot 0 @ ~0deg, slot 9 @ ~180deg
    by_slot = {slot: round(h % 360, 1) for slot, ptype, h in headings}
    assert by_slot[0] == pytest.approx(0.0, abs=0.5)
    assert by_slot[9] == pytest.approx(180.0, abs=0.5)

    pairs = _find_antiparallel_pairs(headings, axis="x")  # 0/180 axis
    assert (0, 9) in [tuple(sorted(p)) for p in pairs]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_boundary_repair.py::test_antiparallel_pair_detected_for_S16 -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/repair.py
_STRAIGHT_TYPES = (STRAIGHT_16, STRAIGHT_24)
# Axis -> the two anti-parallel headings (degrees) we shrink along.
_AXIS_HEADINGS = {"x": (0.0, 180.0), "y": (90.0, 270.0)}
_HEADING_TOL = 1.0  # deg; headings are exact multiples of 22.5


def _active_straight_headings(x, dims, fk_table):
    """List of (slot, piece_type, world_heading_deg) for each active straight."""
    types = x[:dims.n_main]
    flips = x[dims.main_flips_start:dims.main_flips_end]
    out = []
    heading = 0.0
    for slot in range(dims.n_main):
        pt = int(types[slot])
        if pt == INACTIVE:
            continue
        if pt in (int(STRAIGHT_16), int(STRAIGHT_24)):
            out.append((slot, pt, heading))
        dtheta = float(fk_table[pt, 2])
        if pt == R40_CURVE and int(flips[slot]) == 1:
            dtheta = -dtheta
        heading += dtheta
    return out


def _find_antiparallel_pairs(headings, axis):
    """Pairs (slot_a, slot_b) of SAME-type straights on opposite headings of `axis`."""
    h_lo, h_hi = _AXIS_HEADINGS[axis]
    side_a, side_b = {}, {}  # piece_type -> [slots]
    for slot, pt, h in headings:
        hm = h % 360.0
        if abs((hm - h_lo + 180) % 360 - 180) <= _HEADING_TOL:
            side_a.setdefault(pt, []).append(slot)
        elif abs((hm - h_hi + 180) % 360 - 180) <= _HEADING_TOL:
            side_b.setdefault(pt, []).append(slot)
    pairs = []
    for pt in set(side_a) & set(side_b):
        for a, b in zip(side_a[pt], side_b[pt]):  # same-type, equal length
            pairs.append((a, b))
    return pairs
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_boundary_repair.py::test_antiparallel_pair_detected_for_S16 -v`
Expected: PASS.

- [ ] **Step 5: Checkpoint (STOP — do not commit without approval).**

---

### Task 3: `BoundaryAwareRepair` — translate branch (zero offset)

**Files:**
- Modify: `src/repair.py`
- Test: `tests/test_boundary_repair.py`

- [ ] **Step 1: Write the failing test**

```python
def test_translate_zeros_offset_when_loop_fits():
    from src.repair import BoundaryAwareRepair
    dims = make_dims(n_main=24, box=250.0)
    x = create_empty_chromosome(dims)
    # Small loop that easily fits the 500x500 box: a few straights + curves.
    for i in range(8):
        set_main_loop_type(x, dims, i, R40)  # 8 curves, ~half circle
    set_main_loop_type(x, dims, 8, S16)
    # Push it out of bounds via a large fine-tuning offset.
    x[dims.start_pos_start] = 240
    x[dims.start_pos_start + 1] = 0

    rep = BoundaryAwareRepair(dims, FK)
    X = np.array([x])
    rep._do(None, X)
    # Fits => translate => offset zeroed.
    assert int(X[0, dims.start_pos_start]) == 0
    assert int(X[0, dims.start_pos_start + 1]) == 0
    # No straights removed (still active).
    assert int(X[0, 8]) == S16
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_boundary_repair.py::test_translate_zeros_offset_when_loop_fits -v`
Expected: FAIL — `ImportError: cannot import name 'BoundaryAwareRepair'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/repair.py  (place before TrackRepairPipeline)
import math


class BoundaryAwareRepair(Repair):
    """Rescue out-of-bounds layouts: re-center if they fit, else symmetric shrink.

    Branch 1 (translate): if the main-loop span fits the box, zero start_x/start_y
        so the decoder's _auto_center places the loop at box center (max margin).
    Branch 2 (shrink): if an axis is genuinely too big, deactivate same-type
        anti-parallel straight pairs (closure- and angle-preserving), then translate.
    """

    def __init__(self, dims, catalog_fk_table, *, siding_margin=SIDING_MARGIN,
                 boundary_tolerance=0.0, translate_prob=1.0, shrink_prob=1.0,
                 junction_repair=None, **kwargs):
        super().__init__(**kwargs)
        self.dims = dims
        self.fk_table = catalog_fk_table
        self.siding_margin = siding_margin
        self.boundary_tolerance = boundary_tolerance
        self.translate_prob = translate_prob
        self.shrink_prob = shrink_prob
        self.junction_repair = junction_repair  # re-clamp after shrink (set by pipeline)

    def _do(self, problem, X, **kwargs):
        for i in range(len(X)):
            self._repair_chromosome(X[i])
        return X

    def _box_dims(self):
        return (self.dims.boundary_max_x - self.dims.boundary_min_x,
                self.dims.boundary_max_y - self.dims.boundary_min_y)

    def _effective_box(self, x):
        box_w, box_h = self._box_dims()
        # If any siding junction is active, reserve a conservative margin.
        margin = 0.0
        for k in range(self.dims.max_junctions):
            if int(get_junction(x, self.dims, k)[0]) == 1:
                margin = self.siding_margin
                break
        return box_w - 2 * margin, box_h - 2 * margin

    def _repair_chromosome(self, x):
        states = _main_loop_states(x, self.dims, self.fk_table)
        if states.shape[0] <= 1:
            return
        w = float(states[:, 0].max() - states[:, 0].min())
        h = float(states[:, 1].max() - states[:, 1].min())
        box_w_eff, box_h_eff = self._effective_box(x)

        start_x = float(x[self.dims.start_pos_start])
        start_y = float(x[self.dims.start_pos_start + 1])
        slack_x = (box_w_eff - w) / 2.0
        slack_y = (box_h_eff - h) / 2.0
        tol = self.boundary_tolerance

        x_too_big = w > box_w_eff
        y_too_big = h > box_h_eff
        x_offset_out = abs(start_x) > slack_x + tol
        y_offset_out = abs(start_y) > slack_y + tol

        if not (x_too_big or y_too_big or x_offset_out or y_offset_out):
            return  # already in bounds

        shrank = False
        if (x_too_big or y_too_big) and np.random.random() < self.shrink_prob:
            shrank = self._shrink(x, w, h, box_w_eff, box_h_eff)

        # Translate: zero the fine-tuning offset so _auto_center fully centers.
        if np.random.random() < self.translate_prob:
            x[self.dims.start_pos_start] = 0
            x[self.dims.start_pos_start + 1] = 0

        if shrank and self.junction_repair is not None:
            # Deactivating straights shifted active-piece indices: re-clamp junctions.
            self.junction_repair._repair_chromosome(x)

    def _shrink(self, x, w, h, box_w_eff, box_h_eff):
        headings = _active_straight_headings(x, self.dims, self.fk_table)
        removed_any = False
        for axis, span, box in (("x", w, box_w_eff), ("y", h, box_h_eff)):
            deficit = span - box
            if deficit <= 0:
                continue
            pairs = _find_antiparallel_pairs(headings, axis)
            if not pairs:
                continue  # decline gracefully — never break closure
            # Length L = dx of the straight type in the first pair.
            a, b = pairs[0]
            ptype = int(x[a])
            length = float(self.fk_table[ptype, 0])
            n_pairs = min(len(pairs), int(math.ceil(deficit / max(length, 1e-6))))
            for a, b in pairs[:n_pairs]:
                set_main_loop_type(x, self.dims, a, INACTIVE)
                set_flip(x, self.dims, a, 0)
                set_main_loop_type(x, self.dims, b, INACTIVE)
                set_flip(x, self.dims, b, 0)
                removed_any = True
        return removed_any
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_boundary_repair.py::test_translate_zeros_offset_when_loop_fits -v`
Expected: PASS.

- [ ] **Step 5: Checkpoint (STOP — do not commit without approval).**

---

### Task 4: `BoundaryAwareRepair` — shrink branch removes a pair

**Files:**
- Modify: `src/repair.py` (no new code expected; this validates Task 3's `_shrink`)
- Test: `tests/test_boundary_repair.py`

- [ ] **Step 1: Write the failing test**

```python
def test_shrink_removes_antiparallel_pair_when_too_wide():
    from src.repair import BoundaryAwareRepair
    # Tiny box so the loop is "too wide" on x and must shrink.
    dims = make_dims(n_main=24, box=20.0)  # box width = 40 studs
    x = create_empty_chromosome(dims)
    # Degenerate loop: S16 @0, 8 R40 (=>180), S16 @180, 8 R40 (=>360).
    set_main_loop_type(x, dims, 0, S16)
    for i in range(1, 9):
        set_main_loop_type(x, dims, i, R40)
    set_main_loop_type(x, dims, 9, S16)
    for i in range(10, 18):
        set_main_loop_type(x, dims, i, R40)

    rep = BoundaryAwareRepair(dims, FK, siding_margin=0.0)
    X = np.array([x])
    rep._do(None, X)
    # The anti-parallel S16 pair (slots 0 and 9) should be deactivated.
    assert int(X[0, 0]) == INACTIVE
    assert int(X[0, 9]) == INACTIVE
    # Offset zeroed too.
    assert int(X[0, dims.start_pos_start]) == 0
```

- [ ] **Step 2: Run test to verify it fails (or passes — confirm behavior)**

Run: `pytest tests/test_boundary_repair.py::test_shrink_removes_antiparallel_pair_when_too_wide -v`
Expected: PASS if Task 3's `_shrink` is correct. If it FAILS, debug `_shrink`/`_find_antiparallel_pairs` until green — do not move on.

- [ ] **Step 3: Add the graceful-decline test**

```python
def test_shrink_declines_when_no_antiparallel_pair():
    from src.repair import BoundaryAwareRepair
    dims = make_dims(n_main=8, box=5.0)  # box width 10 -> too small
    x = create_empty_chromosome(dims)
    # Two straights both heading 0 (no anti-parallel partner).
    set_main_loop_type(x, dims, 0, S16)
    set_main_loop_type(x, dims, 1, S16)
    rep = BoundaryAwareRepair(dims, FK, siding_margin=0.0)
    X = np.array([x])
    rep._do(None, X)  # must not raise, must not remove (no valid pair)
    assert int(X[0, 0]) == S16
    assert int(X[0, 1]) == S16
```

- [ ] **Step 4: Run both shrink tests**

Run: `pytest tests/test_boundary_repair.py -k shrink -v`
Expected: PASS (both).

- [ ] **Step 5: Checkpoint (STOP — do not commit without approval).**

---

### Task 5: Wire into `TrackRepairPipeline` + runner

**Files:**
- Modify: `src/repair.py` (`TrackRepairPipeline.__init__` and `_do`)
- Modify: `src/algorithm/runner.py:435-440`
- Test: `tests/test_boundary_repair.py`

- [ ] **Step 1: Write the failing test**

```python
def test_pipeline_includes_boundary_repair_and_runs():
    from src.repair import TrackRepairPipeline, BoundaryAwareRepair
    dims = make_dims(n_main=24, max_junctions=0, box=250.0)
    inv = {S16: 24, R40: 24}
    pipe = TrackRepairPipeline(
        dims=dims, inventory_by_index=inv, catalog_fk_table=FK,
        enable_boundary_repair=True,
    )
    assert isinstance(pipe.boundary_repair, BoundaryAwareRepair)
    # Out-of-bounds-by-offset individual gets re-centered by the full pipeline.
    x = create_empty_chromosome(dims)
    for i in range(8):
        set_main_loop_type(x, dims, i, R40)
    set_main_loop_type(x, dims, 8, S16)
    x[dims.start_pos_start] = 240
    X = pipe._do(None, np.array([x]))
    assert int(X[0, dims.start_pos_start]) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_boundary_repair.py::test_pipeline_includes_boundary_repair_and_runs -v`
Expected: FAIL — `AttributeError: 'TrackRepairPipeline' object has no attribute 'boundary_repair'`.

- [ ] **Step 3: Extend the pipeline**

```python
# src/repair.py — TrackRepairPipeline.__init__ : add param + instantiation
    def __init__(
        self,
        dims,
        inventory_by_index,
        catalog_fk_table,
        enable_closure_repair: bool = True,
        enable_boundary_repair: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.junction_repair = JunctionValidityRepair(dims, inventory_by_index)
        self.inventory_repair = InventoryRepair(dims, inventory_by_index)
        self.closure_repair = (
            MainLoopClosureRepair(dims, catalog_fk_table, inventory_by_index)
            if enable_closure_repair else None
        )
        self.boundary_repair = (
            BoundaryAwareRepair(dims, catalog_fk_table,
                                junction_repair=self.junction_repair)
            if enable_boundary_repair else None
        )

    def _do(self, problem, X, **kwargs):
        X = self.junction_repair._do(problem, X, **kwargs)
        X = self.inventory_repair._do(problem, X, **kwargs)
        if self.closure_repair is not None:
            X = self.closure_repair._do(problem, X, **kwargs)
        if self.boundary_repair is not None:
            X = self.boundary_repair._do(problem, X, **kwargs)
        return X
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_boundary_repair.py::test_pipeline_includes_boundary_repair_and_runs -v`
Expected: PASS.

- [ ] **Step 5: Update the runner wiring**

```python
# src/algorithm/runner.py:435-440
    repair = TrackRepairPipeline(
        dims=dims,
        inventory_by_index=inventory_by_index,
        catalog_fk_table=catalog._fk_table,
        enable_closure_repair=True,
        enable_boundary_repair=True,
    )
```

- [ ] **Step 6: Run the FULL test suite (project rule — no `-k` subset for validation)**

Run: `pytest --tb=short -q`
Expected: No NEW failures vs. the pre-change baseline. Record the baseline counts first (run once before Task 1) and compare. Investigate any new failure — do not explain it away.

- [ ] **Step 7: Checkpoint (STOP — do not commit without approval).**

---

### Task 6: Empirical validation on the real config

**Files:** none (run + observe)

- [ ] **Step 1: Quick optimizer run (smoke, must not crash)**

Run: `.venv/Scripts/python.exe main.py --config configs/all_pieces.yaml --output outputs_v1/run_brepair_quick --quick --verbose`
Expected: completes; no exceptions from `BoundaryAwareRepair`.

- [ ] **Step 2: Full run and compare against run14 baseline**

Run: `.venv/Scripts/python.exe main.py --config configs/all_pieces.yaml --output outputs_v1/run_brepair --verbose`
Then `/diag outputs_v1/run_brepair`.
Expected/observe: **feasible count should rise materially** above run14's 54/1000 (translate rescues offset-pushed fitting loops). Best feasible util should be **≥ 58.1%** (translate is lossless; shrink only lowers genuinely-too-big ones). Confirm best layout is closed + no orphan switch via `/diag`.

- [ ] **Step 3: Record results in the design doc**

Append the before/after (feasible count, best util, best speed, switches) to `docs/superpowers/specs/` as a short result note. **Do not commit** — leave staged/working for the user.

- [ ] **Step 4: Checkpoint — report literal `/diag` output to the user and ask for next steps (tune `shrink_prob`/`SIDING_MARGIN`, or proceed to the separate switch-ring density lever).**

---

## Honest Scope Note (read before executing)

This operator **heals feasibility** — it converts the large mass of "infeasible only by the boundary" layouts (946/1000 in run14) into usable feasibles, and the **translate** branch can recover a new champion *for free* if a fitting-but-mis-offset layout has util > 58.1%. It does **not** manufacture denser topology: **shrink lowers util** of too-big loops, and the ~60% simple-loop ceiling is unchanged. Breaking past ~60% remains a **separate lever** (switch-ring / nested-loop generation in `sampling.py`/`operators.py`), out of scope here.

## Self-Review

- **Spec coverage:** translate (zero offset) ✓ Task 3; shrink anti-parallel pairs ✓ Tasks 3–4; intelligent translate-vs-shrink choice ✓ Task 3 `_repair_chromosome`; closure/angle preservation ✓ (straights-only, same-type pairs); junction re-clamp after shrink ✓ Task 3 + Task 5 wiring; pipeline placement ✓ Task 5; empirical check ✓ Task 6.
- **Placeholder scan:** none — every code step has complete code.
- **Type consistency:** `_main_loop_states`, `_active_straight_headings`, `_find_antiparallel_pairs`, `BoundaryAwareRepair(dims, catalog_fk_table, ...)`, `pipe.boundary_repair`, `junction_repair._repair_chromosome` used consistently across tasks. `_find_antiparallel_pairs(headings, axis)` signature matches both call sites (Task 2 test, Task 3 `_shrink`).
