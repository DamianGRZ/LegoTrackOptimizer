# Space-Filling Crossing Topologies — Phase 1 (C & D) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two dense, crossing-using closed-loop seed families to `src/sampling.py` — D (multi-CROSS_90) and C (multi/bigger DOUBLE_CROSSOVER weave) — so the GA has higher-utilization, crossing-bearing options than the existing tiny figure-8 seeds.

**Architecture:** Seed generators only — **no decoder/encoding changes**. Both reuse the existing multi-descriptor injection (`_inject_cross_junctions`, `_inject_double_crossovers`) which already loop over multiple descriptors. Geometry constants are **derived by a verification harness** (decode → must close + commit), never hand-asserted. Each generator self-validates and scales with inventory + boundary (no hardcoded counts — project invariant).

**Tech Stack:** Python, numpy, pymoo 0.6.1.6, pytest. Spec: `docs/superpowers/specs/2026-06-03-space-filling-crossing-topologies-design.md`.

> **PL:** Faza 1 = tylko nowe generatory nasion (C i D), bez zmian dekodera. Geometrię (dokładne sekwencje) wyprowadzamy harnessem weryfikującym (dekoduj → musi się zamknąć i „zatwierdzić" krzyżowania), nie zgaduję jej. Nie ruszamy pudełka, zestawu ani promienia R40.

---

## Pre-flight

- [ ] **Create the feature branch** (off the spec branch or `main`).

Run:
```bash
git checkout -b feat/crossing-seed-families
```

## File structure

- `tests/test_seed_geometry_harness.py` — **new**: reusable decode-and-assert helper + the
  derivation harness used to find/verify geometry constants.
- `src/sampling.py` — **modify**: add `_gen_multi_cross` (D) and `_gen_dc_weave` (C);
  register both in `_get_heuristic_patterns` (`sampling.py:614-629`).
- `tests/test_crossing_seed_families.py` — **new**: per-family acceptance tests.

The existing templates to copy/extend (read them first):
- `_gen_figure_eight_cross` (`sampling.py:462-490`) — emits 1 cross descriptor `(1, 2, 19)`, fixed 34-piece loop.
- `_gen_figure_eight_dbl_crossover` (`sampling.py:397-438`) + `_figure_eight_main_loop` (`sampling.py:372-394`) — parametric DC figure-8.

---

## Task 1: Shared decode-and-assert helper

**Files:**
- Create: `tests/test_seed_geometry_harness.py`

- [ ] **Step 1: Write the helper + a self-test that the EXISTING figure-8 seeds validate**

```python
"""Reusable seed-geometry oracle: decode a Pattern and assert it is a feasible,
committed, boundary-fitting closed loop. Also used as the derivation harness for
new crossing seeds (run candidate geometries through `decode_seed` until it passes).
"""
import numpy as np
import pytest

from src.catalog import TrackCatalog
from src.config import OptimizationConfig
from src.decoder import DecoderConfig, decode_chromosome
from src.encoding import (
    CROSS_90, DOUBLE_CROSSOVER, PartitionedDimensions,
    compute_dimensions, create_chromosome_from_pieces,
)


@pytest.fixture
def cat() -> TrackCatalog:
    return TrackCatalog.load("data/track_pieces_v2.yaml")


@pytest.fixture
def cfg() -> OptimizationConfig:
    return OptimizationConfig.load("configs/all_pieces.yaml")


@pytest.fixture
def dims(cfg, cat) -> PartitionedDimensions:
    return compute_dimensions(cfg, cat)


def decode_seed(pattern, cfg, cat, dims):
    """Decode one Pattern tuple -> MultiPathLayout (using config inventory + boundary)."""
    main_pieces, main_flips, junctions, cross_junctions, dbl_crossovers = pattern
    x = create_chromosome_from_pieces(
        dims, main_pieces, main_loop_flips=main_flips,
        junctions=junctions, cross_junctions=cross_junctions,
        double_crossovers=dbl_crossovers,
    )
    return decode_chromosome(x, cat, cfg.inventory, dims=dims)


def assert_valid_closed(layout, cfg, *, n_cross=0, n_dc=0):
    """Oracle: closed, committed crossings, fits boundary."""
    assert layout.max_closure_error < 4.0, f"not closed: {layout.max_closure_error}"
    assert len(layout.cross_junctions) == n_cross, \
        f"CROSS_90 committed {len(layout.cross_junctions)} != {n_cross}"
    assert layout.n_dbl_crossovers == n_dc, \
        f"DC committed {layout.n_dbl_crossovers} != {n_dc}"
    xs, ys = [], []
    for p in layout.paths:
        if len(p.states):
            xs.append(p.states[:, 0]); ys.append(p.states[:, 1])
    xs = np.concatenate(xs); ys = np.concatenate(ys)
    b = cfg.boundary
    assert (xs.max() - xs.min()) <= (b.max_x - b.min_x), "too wide for boundary"
    assert (ys.max() - ys.min()) <= (b.max_y - b.min_y), "too tall for boundary"


def test_existing_figure_eight_cross_validates(cfg, cat, dims):
    """Sanity: the existing 1-cross seed passes the oracle (proves the oracle is correct)."""
    from src.sampling import _gen_figure_eight_cross
    inv = {cat._id_to_index[k]: v for k, v in cfg.inventory.items() if k in cat._id_to_index}
    variants = _gen_figure_eight_cross(inv, dims)
    assert variants, "expected the existing 1-cross seed to be emitted for all_pieces"
    layout = decode_seed(variants[0], cfg, cat, dims)
    assert_valid_closed(layout, cfg, n_cross=1, n_dc=0)
```

- [ ] **Step 2: Run it — verifies the oracle against known-good geometry**

Run: `pytest tests/test_seed_geometry_harness.py -v`
Expected: PASS. If `assert_valid_closed` is wrong, this catches it now (before we trust it for new geometry).

- [ ] **Step 3: Commit**

```bash
git add tests/test_seed_geometry_harness.py
git commit -m "test(sampling): seed-geometry oracle + harness, verified vs existing figure-8-cross"
```

---

## Task 2: Family D step 1 — scale the single-CROSS_90 figure-8 with `k`

The existing `_gen_figure_eight_cross` is a fixed 34-piece loop (16% util). First make it
**scale** (longer lobes via `k`) so 1-cross seeds reach competitive size; this is the
low-geometry-risk concrete win and the base the 2-cross variant builds on.

**Files:**
- Modify: `src/sampling.py` (add `_gen_multi_cross`)
- Test: `tests/test_crossing_seed_families.py`

- [ ] **Step 1: Write the failing acceptance test**

```python
import pytest
from src.catalog import TrackCatalog
from src.config import OptimizationConfig
from src.encoding import compute_dimensions
from src.sampling import _gen_multi_cross
from tests.test_seed_geometry_harness import decode_seed, assert_valid_closed


@pytest.fixture
def cat(): return TrackCatalog.load("data/track_pieces_v2.yaml")
@pytest.fixture
def cfg(): return OptimizationConfig.load("configs/all_pieces.yaml")
@pytest.fixture
def dims(cfg, cat): return compute_dimensions(cfg, cat)
@pytest.fixture
def inv(cfg, cat):
    return {cat._id_to_index[k]: v for k, v in cfg.inventory.items() if k in cat._id_to_index}


def test_multi_cross_emits_scaled_closed_one_cross_seeds(inv, cfg, cat, dims):
    variants = _gen_multi_cross(inv, dims)
    assert variants, "expected at least one scaled 1-cross variant"
    # every emitted variant must decode closed with exactly its declared crossings
    biggest = 0
    for pat in variants:
        n_cross = len(pat[3] or [])
        lay = decode_seed(pat, cfg, cat, dims)
        assert_valid_closed(lay, cfg, n_cross=n_cross, n_dc=0)
        biggest = max(biggest, lay.n_pieces)
    # must beat the fixed 34-piece base meaningfully
    assert biggest >= 60, f"largest scaled cross seed only {biggest} pieces"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/test_crossing_seed_families.py::test_multi_cross_emits_scaled_closed_one_cross_seeds -v`
Expected: FAIL with `ImportError: cannot import name '_gen_multi_cross'`.

- [ ] **Step 3: Derive the scaled geometry with the harness, then implement**

Derivation: the existing seed is two 12-R40 lobes joined by two 5-STR runs, cross at slots
`(2, 19)`. Generalize the straight runs to length `5 + s` and re-find the crossing slot pair
for each `s` by running the oracle. Use this scratch derivation loop (run in a REPL/`python -`,
NOT committed) to print working `(s, cross_slots, k_pieces)` tuples:

```python
# scratch: find closing scaled 1-cross variants
import numpy as np
from src.catalog import TrackCatalog
from src.config import OptimizationConfig
from src.encoding import compute_dimensions, STRAIGHT_16, R40_CURVE
from tests.test_seed_geometry_harness import decode_seed, assert_valid_closed
cat = TrackCatalog.load("data/track_pieces_v2.yaml"); cfg = OptimizationConfig.load("configs/all_pieces.yaml")
dims = compute_dimensions(cfg, cat)
for s in range(0, 18):
    run = 5 + s
    pieces = [int(STRAIGHT_16)]*run + [int(R40_CURVE)]*12 + [int(STRAIGHT_16)]*run + [int(R40_CURVE)]*12
    flips  = [0]*run + [0]*12 + [0]*run + [1]*12
    # the two STR slots that physically coincide are the 3rd of each run: indices 2 and run+12+2
    cs = (2, run + 12 + 2)
    pat = (pieces, flips, None, [(1, cs[0], cs[1])], None)
    try:
        lay = decode_seed(pat, cfg, cat, dims); assert_valid_closed(lay, cfg, n_cross=1, n_dc=0)
        print("OK", s, cs, lay.n_pieces)
    except AssertionError as e:
        print("no", s, str(e)[:40])
```

Then implement `_gen_multi_cross` emitting the `s` values that printed `OK`, scaled to
inventory/boundary (mirror the inventory/boundary guards in `_gen_figure_eight_cross`,
`sampling.py:476-482`). Skeleton:

```python
def _gen_multi_cross(inv, dims):
    """Scaled single-CROSS_90 figure-8 variants (and, later, the 2-cross triple lobe).

    Generalizes _gen_figure_eight_cross: two 12-R40 lobes joined by straight runs of
    length 5+s; the crossing slot pair is the 3rd straight of each run. Emits the s
    values that decode closed with the crossing committed and fit the boundary.
    """
    if dims.max_cross_junctions < 1 or inv.get(CROSS_90, 0) < 1:
        return []
    if inv.get(R40_CURVE, 0) < 24:
        return []
    w, h = _boundary_wh(dims)
    if w < 170 or h < 170:
        return []
    variants: List[Pattern] = []
    s_inv = (inv.get(STRAIGHT_16, 0) - 0) // 2          # two runs
    s_fit = max(0, int((min(w, h) - 170) // 16))         # boundary headroom per derivation
    for s in sorted({min(s_inv, s_fit), min(s_inv, s_fit) // 2, 0}, reverse=True):
        run = 5 + s
        pieces = [int(STRAIGHT_16)]*run + [int(R40_CURVE)]*12 + [int(STRAIGHT_16)]*run + [int(R40_CURVE)]*12
        if not _pieces_fit_inventory(_count_pieces(pieces), inv):
            continue
        flips = [0]*run + [0]*12 + [0]*run + [1]*12
        descriptors = [(1, 2, run + 12 + 2)]
        variants.append((pieces, flips, None, descriptors, None))
    return variants
```

> If the harness shows the naive crossing-slot formula does not close for some `s`
> (R40 lobe is 12 pieces = 270°, so the runs must align the ports), keep only the `s`
> values it confirms. **Do not emit a variant the oracle rejects.**

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_crossing_seed_families.py::test_multi_cross_emits_scaled_closed_one_cross_seeds -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sampling.py tests/test_crossing_seed_families.py
git commit -m "feat(sampling): scalable single-CROSS_90 figure-8 seed (_gen_multi_cross)"
```

---

## Task 3: Family D step 2 — the 2-CROSS_90 triple-lobe variant

**Files:**
- Modify: `src/sampling.py` (extend `_gen_multi_cross` to also emit a 2-cross variant)
- Test: `tests/test_crossing_seed_families.py`

- [ ] **Step 1: Write the failing test**

```python
def test_multi_cross_emits_a_two_cross_variant(inv, cfg, cat, dims):
    variants = _gen_multi_cross(inv, dims)
    two = [p for p in variants if len(p[3] or []) == 2]
    assert two, "expected at least one 2-CROSS_90 variant"
    lay = decode_seed(two[0], cfg, cat, dims)
    assert_valid_closed(lay, cfg, n_cross=2, n_dc=0)
    assert lay.n_pieces >= 40
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/test_crossing_seed_families.py::test_multi_cross_emits_a_two_cross_variant -v`
Expected: FAIL (`expected at least one 2-CROSS_90 variant`).

- [ ] **Step 3: Derive the triple-lobe geometry, then implement**

This is the genuinely new geometry. Build the candidate with the harness BEFORE coding it
into the generator. Derivation procedure (scratch REPL): construct a closed main loop of
**three lobes** (left, middle, right), the middle lobe’s two straight crossings coinciding
perpendicular with the left and right lobes respectively; search lobe/run sizes and the two
crossing slot-pairs until `assert_valid_closed(..., n_cross=2)` passes. Start from two stacked
copies of the verified 1-cross loop sharing a middle run. Record the exact
`(pieces, flips, [(1,a,b),(1,c,d)])` the harness confirms, then add it to `_gen_multi_cross`
guarded by `inv.get(CROSS_90,0) >= 2`.

> The committed code contains the harness-confirmed constants only. If the triple-lobe
> cannot be closed within the boundary with this kit, STOP and report — do not ship a
> 2-cross variant the oracle rejects. (Per the spec, A′ — not D — is the real density play;
> D failing to reach 2 crosses is an acceptable negative result to surface, not to fake.)

- [ ] **Step 4: Run the test to verify it passes** (or report the negative result)

Run: `pytest tests/test_crossing_seed_families.py::test_multi_cross_emits_a_two_cross_variant -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sampling.py tests/test_crossing_seed_families.py
git commit -m "feat(sampling): 2-CROSS_90 triple-lobe seed variant"
```

---

## Task 4: Family C — bigger / multi-DC weave seeds

The existing `_gen_figure_eight_dbl_crossover` already scales one DC. Family C adds a
**denser DC weave** that can use **both** DCs (inventory owns 2) for more woven loop.

**Files:**
- Modify: `src/sampling.py` (add `_gen_dc_weave`)
- Test: `tests/test_crossing_seed_families.py`

- [ ] **Step 1: Write the failing test**

```python
from src.sampling import _gen_dc_weave

def test_dc_weave_emits_closed_committed_seeds(inv, cfg, cat, dims):
    variants = _gen_dc_weave(inv, dims)
    assert variants, "expected at least one DC-weave variant"
    best = 0
    for pat in variants:
        n_dc = len(pat[4] or [])
        lay = decode_seed(pat, cfg, cat, dims)
        assert_valid_closed(lay, cfg, n_cross=0, n_dc=n_dc)
        best = max(best, lay.n_pieces)
    assert best >= 60, f"largest DC-weave only {best} pieces"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/test_crossing_seed_families.py::test_dc_weave_emits_closed_committed_seeds -v`
Expected: FAIL (`ImportError: cannot import name '_gen_dc_weave'`).

- [ ] **Step 3: Derive with the harness, then implement**

Start from `_figure_eight_main_loop(k)` + the DC descriptor from
`_gen_figure_eight_dbl_crossover` (`sampling.py:434-437`). For a single bigger DC, reuse it
at higher `k`. For the 2-DC variant, derive a main loop whose FK presents two valid DC
crossing sites (each a `both-cross` 2-route cover), confirmed by the oracle with `n_dc=2`,
guarded by `dims.max_double_crossovers >= 2` and `inv DOUBLE_CROSSOVER >= 2`. Emit only
oracle-confirmed geometries; scale with inventory/boundary like the existing DC seed
(`sampling.py:416-424`).

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_crossing_seed_families.py::test_dc_weave_emits_closed_committed_seeds -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sampling.py tests/test_crossing_seed_families.py
git commit -m "feat(sampling): denser DOUBLE_CROSSOVER weave seed (_gen_dc_weave)"
```

---

## Task 5: Register the new families in the seed pipeline

**Files:**
- Modify: `src/sampling.py:614-629` (`_get_heuristic_patterns`)
- Test: `tests/test_crossing_seed_families.py`

- [ ] **Step 1: Write the failing test**

```python
from src.sampling import IntegerSampling

def test_new_families_registered(cfg, cat):
    import numpy as np
    s = IntegerSampling(cat, cfg, heuristic_ratio=1.0)
    pats = s._get_heuristic_patterns(np.random.default_rng(0))
    # at least one pattern uses 2 crossings and at least one uses a DC weave
    n_two_cross = sum(1 for p in pats if len(p[3] or []) == 2)
    n_dc = sum(1 for p in pats if len(p[4] or []) >= 1)
    assert n_two_cross >= 1
    assert n_dc >= 1
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/test_crossing_seed_families.py::test_new_families_registered -v`
Expected: FAIL (new generators not yet wired in).

- [ ] **Step 3: Register both generators**

In `_get_heuristic_patterns` (`sampling.py:619-627`), add after the existing
`_gen_figure_eight_*` lines:

```python
        patterns += _gen_multi_cross(inv, dims)
        patterns += _gen_dc_weave(inv, dims)
```

- [ ] **Step 4: Run it to verify it passes**

Run: `pytest tests/test_crossing_seed_families.py::test_new_families_registered -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sampling.py tests/test_crossing_seed_families.py
git commit -m "feat(sampling): register multi-cross + DC-weave seed families"
```

---

## Task 6: Full suite + GA smoke verification

- [ ] **Step 1: Run the FULL test suite** (project policy — no `-k` subset)

Run: `pytest --tb=short -q`
Expected: no NEW failures vs the pre-change baseline (record both counts).

- [ ] **Step 2: GA smoke run** to confirm crossings now get used

Run:
```bash
.venv/Scripts/python.exe main.py --config configs/all_pieces.yaml --output outputs_v1/verify_crossing_seeds_1 --verbose
```
Expected: in the final "Piece usage", `CROSS_90` and/or `DOUBLE_CROSSOVER` usage > 0 in the
best feasible solution (vs `0/2` baseline). Inspect with `/diag`.

- [ ] **Step 3: Record result, commit any test-baseline notes**

```bash
git add -A
git commit -m "test: full-suite + GA smoke baseline for crossing seed families"
```

> **Honest expectation (from spec §9):** C/D likely land ~60–75% utilization — the value is
> proving the pipeline and getting the crossing pieces *used*, not beating the racetrack by
> much. The big jump is A′ (Phase 2), which is gated on a buildable single-radius threading.

---

## Self-review checklist (run before handoff)

1. **Spec coverage:** Phase 1 families C (§5) and D (§6) → Tasks 2–5. Validation/testing
   (§8) → Tasks 1 & 6. A′ (§7) is explicitly Phase 2, out of this plan. ✓
2. **No fabricated geometry:** every emitted seed is gated by the oracle; constants come from
   the harness, not assertion. Negative results are reported, not faked. ✓
3. **Type/name consistency:** `_gen_multi_cross`, `_gen_dc_weave`, `decode_seed`,
   `assert_valid_closed` used identically across tasks. Layout attrs (`max_closure_error`,
   `cross_junctions`, `n_dbl_crossovers`, `n_pieces`, `paths`) match `test_dc_grow.py` /
   `test_cross90_objective.py`. ✓
4. **Project invariants:** no hardcoded counts (scale with inventory/boundary); no config
   changes; full-suite run in Task 6. ✓
